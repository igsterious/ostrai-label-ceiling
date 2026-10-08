"""Cross-fitted self-training: let the classifier correct its own labels.

Benchmark A's pseudo-labels are ~85% accurate, and that caps what any model
trained on them can score -- the leaderboard put A at 81.25%. This tries to
raise the ceiling using nothing but the provided data (rule 2 holds: no new
labels come from outside).

THE IDEA. A classifier trained on 9,000 noisy labels learns the consistent
signal and partly averages out the scatter. Where it confidently disagrees
with a pseudo-label, the pseudo-label is more often the wrong one. Flipping
those cleans the training set.

WHY CROSS-FITTING IS NOT OPTIONAL. A model asked about its own training rows
mostly parrots the labels it was fitted to, so naive self-training finds few
disagreements and the ones it finds are biased toward what it already
memorised. Instead the pool is split into k folds; each fold is predicted by a
model trained only on the OTHER folds, so every prediction is out-of-sample.
That costs k trainings and is the whole reason this can work at all.

THE RISK, STATED PLAINLY. Self-training amplifies confident mistakes: a model
sure of a wrong answer will rewrite a correct label. Two guards. First, only
flip above a confidence threshold. Second, every revision is scored against
the 96 human-judged messages before you train on it -- that gate uses both the
labels a human APPROVED and the ones a human REJECTED, so unlike the
classifier's validation set it is not conditioned on the labeller being right.

    python src/self_train.py benchmark_a                    # 3 folds, 0.9
    python src/self_train.py benchmark_a --folds 5 --threshold 0.95
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from config import artifact_dir, labels_dir
from data import load_pool_unlabeled
from human_gate import score as gate_score
from labeller import read_labels
from train import (
    DEFAULT_BATCH,
    DEFAULT_ENCODER,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SEED,
    MAX_LENGTH,
    make_training_args,
)
from training_data import find_pseudo_label_file, label_maps, load_fewshot


def tokenize_frame(df, tokenizer):
    keep = ["input_ids", "attention_mask", "token_type_ids", "label"]
    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False).map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True,
    )
    return ds.remove_columns([c for c in ds.column_names if c not in keep])


def fit_and_predict(train_df, predict_df, intents, intent2id, args, tag):
    """Train on train_df, return (predicted intent, confidence) for predict_df.

    Imports its hyperparameters from train.py rather than restating them, so
    the fold models are trained with exactly the recipe the final model uses.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.encoder,
        num_labels=len(intents),
        id2label={i: n for i, n in enumerate(intents)},
        label2id=intent2id,
    )
    # DeBERTa-v3 ships fp16 weights and newer transformers loads a checkpoint in
    # the dtype it was saved in, which kills mixed-precision training in the
    # gradient unscaler. train.py has carried this line for weeks; this file was
    # written before DeBERTa was the encoder and never got it.
    model = model.float()

    out = artifact_dir(args.benchmark) / "selftrain" / tag
    targs = make_training_args(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        seed=args.seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tokenize_frame(train_df, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    logits = trainer.predict(tokenize_frame(predict_df, tokenizer)).predictions
    # softmax over the class axis; the max is the model's confidence
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    idx = probs.argmax(axis=-1)
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [intents[i] for i in idx], probs.max(axis=-1)


def main():
    ap = argparse.ArgumentParser(description="Cross-fitted self-training label refinement")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="minimum confidence to overwrite a pseudo-label")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # Without this the script picks whichever labelling find_pseudo_label_file
    # settles on, which is no longer the one we train on. Refining the wrong
    # label set would produce a plausible-looking result about nothing.
    ap.add_argument("--labels-file",
                    help="the pseudo-label file to refine (default: the "
                         "canonical one for this benchmark)")
    ap.add_argument("--tag", help="goes in the output filename, so two runs "
                                  "over different label sets do not collide")
    args = ap.parse_args()

    intent2id, intents = label_maps(args.benchmark)
    fewshot = load_fewshot(args.benchmark)[["text", "intent"]].copy()
    fewshot["label"] = fewshot["intent"].map(intent2id)

    source = (Path(args.labels_file) if args.labels_file
              else find_pseudo_label_file(args.benchmark))
    pool = load_pool_unlabeled(args.benchmark).merge(read_labels(source), on="id")
    pool["label"] = pool["intent"].map(intent2id)

    print(f"{args.benchmark}: {len(pool):,} pool rows, {len(fewshot)} gold, "
          f"{args.folds} folds, threshold {args.threshold}")
    print(f"  labels from {source.name}")
    print(f"  every prediction is out-of-sample: a fold is never scored by a "
          f"model that trained on it\n")

    rng = np.random.default_rng(args.seed)
    fold_of = rng.integers(0, args.folds, len(pool))

    preds = np.empty(len(pool), dtype=object)
    confs = np.zeros(len(pool))
    for k in range(args.folds):
        held = fold_of == k
        # gold few-shot goes into every fold's training set: it is the only
        # trustworthy signal and withholding it would weaken each fold model
        train_df = pd.concat([fewshot[["text", "label"]], pool.loc[~held, ["text", "label"]]])
        print(f"=== fold {k+1}/{args.folds}: train {len(train_df):,}, "
              f"predict {int(held.sum()):,} ===", flush=True)
        p, c = fit_and_predict(train_df, pool.loc[held], intents, intent2id, args, f"fold{k}")
        preds[held] = p
        confs[held] = c

    pool["pred"] = preds
    pool["conf"] = confs
    agree = pool["pred"] == pool["intent"]
    flip = (~agree) & (pool["conf"] >= args.threshold)

    print(f"\n{'='*66}")
    print("OUT-OF-SAMPLE AGREEMENT WITH THE PSEUDO-LABELS")
    print(f"{'='*66}")
    print(f"  agree                     {agree.sum():,} / {len(pool):,} = {agree.mean():.1%}")
    print(f"  disagree                  {(~agree).sum():,}")
    print(f"  disagree AND conf >= {args.threshold}  {flip.sum():,}  <- would be rewritten")
    print(f"  mean confidence: agree {pool.loc[agree,'conf'].mean():.3f}, "
          f"disagree {pool.loc[~agree,'conf'].mean():.3f}")

    # Save the out-of-sample predictions BEFORE thresholding. They cost k
    # trainings to produce; without this, trying another threshold means
    # re-running the whole thing.
    tag = f"_{args.tag}" if args.tag else ""
    oos = artifact_dir(args.benchmark) / f"selftrain_oos{tag}.tsv"
    pool[["id", "text", "intent", "pred", "conf"]].to_csv(
        oos, sep="\t", index=False, quoting=3
    )
    print(f"\n  out-of-sample predictions -> {oos.name} "
          f"(re-threshold from these, no retraining needed)")

    revised = pool[["id", "intent"]].copy()
    revised.loc[flip.values, "intent"] = pool.loc[flip, "pred"].values
    out = labels_dir(args.benchmark) / f"pseudo_labels.selftrain{tag}_t{args.threshold}.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for r in revised.itertuples():
            fh.write(json.dumps({"id": r.id, "intent": r.intent,
                                 "model": f"selftrain(t={args.threshold})"}) + "\n")
    print(f"\n  wrote {out.name}  ({flip.sum():,} labels changed)")

    # The gate. Uses approved AND rejected human verdicts, so it measures label
    # quality without conditioning on the original labeller having been right.
    print(f"\n{'='*66}")
    print("SCORED AGAINST THE HUMAN-JUDGED MESSAGES")
    print(f"{'='*66}")
    for name, labels in (("before", read_labels(source)), ("after ", read_labels(out))):
        try:
            r = gate_score(labels, args.benchmark)
        except SystemExit as exc:
            print(f"  {exc}")
            return
        print(f"  {name}: kept {r['kept']}/{r['n_approved']} approved, "
              f"changed {r['changed']}/{r['n_rejected']} rejected  ->  {r['score']:.1%}")
    print("\n  Only train on the revision if that score went UP. Self-training\n"
          "  amplifies confident mistakes, and this gate is what catches it.")

    # Free threshold sweep from the saved predictions, so the trade-off between
    # "few safe fixes" and "many risky ones" is visible without more training.
    print(f"\n{'='*66}")
    print("OTHER THRESHOLDS (from the same predictions, no retraining)")
    print(f"{'='*66}")
    print(f"  {'thresh':>7s} {'changed':>8s} {'gate':>7s}")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        f = (~agree) & (pool["conf"] >= t)
        rev = pool[["id", "intent"]].copy()
        rev.loc[f.values, "intent"] = pool.loc[f, "pred"].values
        try:
            s = gate_score(rev.astype(str), args.benchmark)["score"]
            print(f"  {t:>7.2f} {int(f.sum()):>8,} {s:>6.1%}")
        except SystemExit:
            break


if __name__ == "__main__":
    main()
