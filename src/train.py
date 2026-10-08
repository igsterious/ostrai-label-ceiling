"""Fine-tune a small encoder for intent classification. Colab, not local.

Needs torch + transformers, which have no wheels for the local Python 3.14 --
run this in Colab (see README). Everything it reads is produced by scripts that
DO run locally, so the data is inspectable before it ever reaches a GPU.

Two deliberate departures from the lecture's 02_spam_finetuning notebook:

1. AN ENCODER, NOT A DECODER. SmolLM2-135M is a generative LM pressed into
   classification. Encoders are purpose-built for it and, at these sizes,
   faster. Rule 2 is safe for any generic encoder: it carries no intent
   labels, unlike a classifier pretrained on this label space.
   Which encoder is a measurable question -- see sweep_encoders.py.

2. DYNAMIC PADDING, NOT padding="max_length". The lecture pads every message
   to a fixed 64 tokens. Our median message is ~12 tokens, so that is ~5x
   wasted compute on padding -- paid on the axis we are scored on.
   DataCollatorWithPadding pads each batch only to its own longest member.

Validation reports TWO accuracies, because the test set's class prior is
unknowable (rule 1 forbids inspecting it):
  balanced_acc       -- every class counted equally. Holdout is 5/class, so
                        this is what plain accuracy on holdout measures.
  prior_weighted_acc -- classes weighted by their frequency in the pool. What
                        accuracy would look like if test mirrors the pool.
The truth is bracketed by the two. If they diverge sharply, the rare classes
are weak and the gap tells you how much that costs under each assumption.

Usage (in Colab):
    python src/train.py benchmark_a
    python src/train.py benchmark_b --epochs 3
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from config import artifact_dir
from metrics import make_compute_metrics
from training_data import (
    build_training_set,
    class_prior,
    human_judged_ids,
    label_maps,
    load_holdout,
    load_human_validation,
    save_label_map,
)

# DistilBERT by default. All the candidates in sweep_encoders.py are generic
# pretrained encoders carrying no intent labels, so rule 2 is satisfied for any
# of them -- a model fine-tuned on this label space would not be. Which one
# actually belongs in the submission is a speed/accuracy trade-off, so run
# sweep_encoders.py and change this to whatever wins on both axes.
DEFAULT_ENCODER = "distilbert-base-uncased"
# Truncation cap only. Measured: p99 is ~70 tokens for A, ~28 for B, so this
# discards essentially nothing. Dynamic padding means it costs nothing either.
MAX_LENGTH = 128

# Module constants, not argparse literals, so other scripts import the same
# recipe instead of keeping their own copy -- sweep_encoders.py silently ran a
# stale 3-epoch recipe for exactly that reason.
#
# 3 epochs at 3e-5 measurably underfits: it left 11 of 77 classes never
# predicted at all, on ~446 optimizer steps for a freshly-initialised 77-way
# head. More steps and a hotter rate are the fix.
DEFAULT_EPOCHS = 8
DEFAULT_LR = 5e-5
DEFAULT_BATCH = 64
DEFAULT_SEED = 42


def make_training_args(**kwargs):
    """Build TrainingArguments against whatever transformers version is present.

    Colab's transformers moves independently of this repo and has renamed or
    dropped arguments across majors (`evaluation_strategy` -> `eval_strategy`,
    and `warmup_ratio` gone in the version Colab served in Sept 2026). Passing
    an argument the installed version does not know is a hard TypeError, which
    would strand the whole training run on a detail that does not change the
    result.

    So: rename what has a known new spelling, drop what is genuinely absent,
    and SAY which were dropped -- a silent drop would quietly change training
    behaviour.
    """
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)

    # known renames, newest spelling first
    renames = [("eval_strategy", "evaluation_strategy")]
    for new, old in renames:
        if new in kwargs and new not in accepted and old in accepted:
            kwargs[old] = kwargs.pop(new)
        elif old in kwargs and old not in accepted and new in accepted:
            kwargs[new] = kwargs.pop(old)

    dropped = sorted(k for k in kwargs if k not in accepted)
    for k in dropped:
        kwargs.pop(k)
    if dropped:
        print(f"  note: transformers {__import__('transformers').__version__} does not "
              f"accept {dropped} -- omitted")
    return TrainingArguments(**kwargs)


class SoftCollator:
    """Pad the tokenised fields as usual, then re-attach the soft targets.

    DataCollatorWithPadding pads what the tokenizer knows about and does not
    know what to do with a 77-float vector per row, so `soft` is lifted out
    before padding and stacked back on afterwards.
    """

    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        soft = None
        if features and "soft" in features[0]:
            soft = [f.pop("soft") for f in features]
        batch = self.base(features)
        if soft is not None:
            batch["soft"] = torch.tensor(soft, dtype=torch.float)
        return batch


class DistilTrainer(Trainer):
    """Cross-entropy against the teacher's distribution instead of its argmax.

    The teacher says 0.5 / 0.4 / 0.1 across three neighbouring intents on the
    rows it finds hard. Training on the argmax discards that and asks the
    student to be certain about a coin flip; training on the distribution
    tells it which categories are genuinely adjacent, which is the thing every
    audit of this project has found it getting wrong.

    KL(teacher || student) differs from cross-entropy only by the teacher's own
    entropy, which is constant in the student's parameters -- so on a one-hot
    row (every gold example) this reduces exactly to the loss those rows would
    have had anyway, and gold and pseudo rows need no special-casing.

    alpha mixes in ordinary cross-entropy against the hard label. Strictly it
    adds no information the teacher's mode did not already carry, but it keeps
    the student anchored if the teacher is diffuse across many classes.
    """

    def __init__(self, *args, alpha=0.9, temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        soft = inputs.pop("soft", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if soft is None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        else:
            # The teacher was softened by the same T when the vectors were
            # loaded. The T**2 restores the gradient magnitude, which softmax
            # otherwise scales down by 1/T**2 -- without it, raising T quietly
            # lowers the learning rate on this term instead of softening it.
            t = self.temperature
            logp = torch.nn.functional.log_softmax(logits / t, dim=-1)
            kl = torch.nn.functional.kl_div(
                logp, soft.to(logits.device), reduction="batchmean") * (t * t)
            if self.alpha >= 1.0:
                loss = kl
            else:
                # The hard-label term stays at T=1: it is the anchor, and
                # softening it too would leave nothing holding the student to
                # the actual answer.
                ce = torch.nn.functional.cross_entropy(logits, labels)
                loss = self.alpha * kl + (1.0 - self.alpha) * ce
        return (loss, outputs) if return_outputs else loss


class WeightedTrainer(Trainer):
    """Trainer with per-class loss weights. Only used with --class-weights.

    Off by default on purpose. Benchmark A's imbalance was verified genuine,
    and the leaderboard scores plain accuracy, so upweighting rare classes
    shifts the decision boundary away from the true prior and trades many
    correct common-class predictions for a few rare-class ones. The flag
    exists so that claim can be tested against holdout rather than assumed.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # label_smoothing is read off TrainingArguments and applied HERE too.
        # This method replaces the Trainer's own loss, so anything the Trainer
        # would normally do with label_smoothing_factor simply would not happen
        # -- --class-weights --label-smoothing 0.1 would run with no smoothing
        # at all and report a perfectly plausible number for a setting that was
        # never in effect.
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device),
            label_smoothing=getattr(self.args, "label_smoothing_factor", 0.0) or 0.0,
        )
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser(description="Fine-tune an encoder for intent classification")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--labeller-model", help="which pseudo-label file to train on")
    ap.add_argument("--cap-per-class", type=int, help="trim over-represented classes")
    ap.add_argument("--class-weights", action="store_true", help="weight the loss by inverse frequency")
    ap.add_argument("--drop-disagreed", action="store_true",
                    help="drop pool rows where two labellers disagreed (~10%% of A)")
    ap.add_argument("--labels-file",
                    help="explicit pseudo-label JSONL, e.g. a self-training revision")
    ap.add_argument("--out", help="output dir (default artifacts/<benchmark>/model)")
    ap.add_argument("--use-holdout", action="store_true",
                    help="train on the held-out gold rows too, doubling gold data "
                         "(holdout then measures nothing -- judge on the audit)")
    ap.add_argument("--gold-weight", type=int, default=1,
                    help="repeat gold rows N times so the clean signal is not "
                         "drowned by ~10x more pseudo-labels")
    # We measured the training labels at ~90% accuracy, so roughly 900 of A's
    # 9,398 rows are wrong. Hard targets tell the model to become CONFIDENT on
    # every one of them. Smoothing spreads a little mass off the labelled class,
    # which is the standard treatment for known label noise and costs nothing at
    # inference. Untried until now, and the most directly indicated thing we
    # have left: every other lever assumed the labels could be improved, and
    # tonight showed they cannot be.
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="label smoothing factor, e.g. 0.1 (labels are ~90%% correct)")
    # The 220 hand-reviewed pool messages are held out of training to keep the
    # audit honest. The audit has since saturated -- 100/120 for eleven straight
    # variants -- so it no longer discriminates, and ~183 certain labels on
    # confusable, test-length messages are worth more in the training set than
    # in a gate that always returns the same number. Spending them is one-way.
    ap.add_argument("--augment-file",
                    help="TSV of generated (text, intent) rows for rare classes")
    ap.add_argument("--use-judged", action="store_true",
                    help="train on the human-confirmed pool rows too "
                         "(the audit then measures memorisation, not accuracy)")
    ap.add_argument("--soft-labels",
                    help="a .soft.jsonl from distil_labels.py --soft: train "
                         "against the teacher's distribution, not its argmax")
    ap.add_argument("--soft-alpha", type=float, default=0.9,
                    help="weight on the KL term; the rest is cross-entropy "
                         "against the hard label")
    ap.add_argument("--soft-temp", type=float, default=3.0,
                    help="softens teacher and student alike. The teacher is "
                         "overconfident (median 0.995), so T=1 carries almost "
                         "nothing its argmax did not")
    args = ap.parse_args()

    intent2id, intents = label_maps(args.benchmark)
    prior = class_prior(args.benchmark, args.labeller_model)

    train_df = build_training_set(
        args.benchmark, args.labeller_model, args.cap_per_class,
        drop_disagreed=args.drop_disagreed, labels_file=args.labels_file,
        use_holdout=args.use_holdout, gold_weight=args.gold_weight,
        use_judged=args.use_judged, soft_labels_file=args.soft_labels,
        soft_temperature=args.soft_temp, augment_file=args.augment_file,
    )
    if args.use_judged:
        print("  --use-judged: the 220 hand-reviewed rows are IN training; "
              "the audit now measures memorisation, not accuracy")
    holdout_df = load_holdout(args.benchmark).copy()
    holdout_df["label"] = holdout_df["intent"].map(intent2id)

    if args.use_holdout:
        # Holdout is training data now, so it can no longer validate anything.
        # Say so loudly and stop selecting on it, rather than reporting a number
        # that would be pure memorisation -- the exact mistake the contaminated
        # audit made, where the model scored 85.8% on 120 rows it had trained on.
        print("  --use-holdout: holdout rows are IN training; "
              "holdout accuracy below is memorisation, not validation")
        print("  model selection falls back to the final epoch; "
              "judge this run on the audit or the leaderboard")
    else:
        # The guarantee every number below depends on: nothing we fit on appears
        # in what we validate on.
        overlap = set(train_df["text"]) & set(holdout_df["text"])
        if overlap:
            raise SystemExit(
                f"LEAK: {len(overlap)} texts appear in both training and holdout. "
                "Refusing to train -- validation numbers would be fiction."
            )

    # The 220 human-judged messages stay out unless releasing them was the
    # explicit point of the run. They are the audit, and --use-holdout must not
    # be a back door into re-contaminating it -- which is why this guard fires
    # on TEXT rather than id, and fires regardless of how a row got in.
    #
    # --use-judged is the one sanctioned way through. It is not an override of
    # the safety property; it retires the audit on purpose, having decided the
    # rows are worth more as training labels than as a gate that has returned
    # 100/120 for eleven consecutive variants.
    if not args.use_judged:
        judged_texts = set()
        if args.benchmark == "benchmark_a":
            from data import load_pool_unlabeled
            pool = load_pool_unlabeled(args.benchmark).set_index("id")["text"].to_dict()
            judged_texts = {pool[i] for i in human_judged_ids(args.benchmark) if i in pool}
        leaked = judged_texts & set(train_df["text"])
        if leaked:
            raise SystemExit(
                f"LEAK: {len(leaked)} human-judged messages are in training. "
                "Refusing to train -- the audit would stop measuring anything."
            )

    import transformers

    print(f"{args.benchmark}: {len(train_df):,} train / {len(holdout_df):,} holdout, "
          f"{len(intents)} classes")
    print(f"  encoder {args.encoder}")
    print(f"  torch {torch.__version__}, transformers {transformers.__version__}, "
          f"cuda {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    def tokenize(batch):
        # No padding here -- the collator pads per batch. This is the speed win.
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    # token_type_ids kept for BERT-family encoders that expect it
    keep = ["input_ids", "attention_mask", "token_type_ids", "label"]
    train_cols = ["text", "label"]
    if args.soft_labels:
        keep = keep + ["soft"]
        train_cols = train_cols + ["soft"]
    train_ds = (
        Dataset.from_pandas(train_df[train_cols], preserve_index=False)
        .map(tokenize, batched=True)
    )
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])
    eval_ds = (
        Dataset.from_pandas(holdout_df[["text", "label"]], preserve_index=False)
        .map(tokenize, batched=True)
    )
    eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in keep])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.encoder,
        num_labels=len(intents),
        id2label={i: n for i, n in enumerate(intents)},
        label2id=intent2id,
    )

    # Force fp32 weights before training. Newer transformers loads a checkpoint
    # in the dtype it was SAVED in, and some encoders ship fp16 -- deberta-v3
    # among them. Mixed-precision training then dies in the gradient unscaler
    # ("Attempting to unscale FP16 gradients"), because AMP keeps fp32 master
    # weights and scales fp16 gradients back onto them; if the masters are
    # already fp16 there is nothing to unscale onto.
    #
    # A no-op for distilbert and roberta, which ship fp32. Inference fp16 is a
    # separate decision, applied in classify.py where it is also measured.
    model = model.float()

    out_dir = Path(args.out) if args.out else artifact_dir(args.benchmark) / "model"

    # Warmup under whichever spelling this transformers version has. It matters
    # here: the classification head starts random, and large early updates from
    # a cold head are what strand rare classes at zero.
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    steps_per_epoch = max(1, -(-len(train_ds) // args.batch_size))
    total_steps = int(steps_per_epoch * args.epochs)
    if "warmup_ratio" in accepted:
        warm = {"warmup_ratio": 0.1}
    elif "warmup_steps" in accepted:
        warm = {"warmup_steps": max(1, int(0.1 * total_steps))}
    else:
        warm = {}
        print("  note: no warmup argument available in this transformers version")
    print(f"  {total_steps} optimizer steps ({steps_per_epoch}/epoch x {args.epochs:g})"
          + (f", warmup {list(warm.values())[0]}" if warm else ""))

    targs = make_training_args(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        **warm,
        # With holdout in training, "best epoch by holdout" would select the
        # most memorised checkpoint. Take the last epoch instead: 8 is already
        # the recipe we settled, so there is nothing to select.
        eval_strategy="epoch" if not args.use_holdout else "no",
        save_strategy="epoch" if not args.use_holdout else "no",
        save_total_limit=1,
        load_best_model_at_end=not args.use_holdout,
        **({} if args.use_holdout else {
            "metric_for_best_model": "balanced_acc",
            "greater_is_better": True,
        }),
        logging_steps=50,
        seed=args.seed,
        report_to="none",
        fp16=torch.cuda.is_available(),  # T4 handles fp16 well; big training speedup
        label_smoothing_factor=args.label_smoothing,
    )

    trainer_kwargs = dict(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=(SoftCollator(DataCollatorWithPadding(tokenizer))
                       if args.soft_labels
                       else DataCollatorWithPadding(tokenizer)),  # dynamic padding
        compute_metrics=make_compute_metrics(intents, prior),
    )

    # Both of these replace the loss, so a combination would silently run with
    # only one of them in effect and report a plausible number for a setting
    # that was never applied -- the same trap WeightedTrainer documents above.
    if args.soft_labels and args.class_weights:
        raise SystemExit("--soft-labels and --class-weights use different losses; pick one")
    if args.soft_labels and args.label_smoothing:
        raise SystemExit(
            "--soft-labels ignores --label-smoothing: smoothing is a crude "
            "stand-in for exactly what the teacher's distribution provides. "
            "Lower --soft-alpha instead if you want the hard label to count."
        )

    if args.soft_labels:
        print(f"  soft distillation on, T={args.soft_temp}, alpha {args.soft_alpha} "
              f"(KL to the teacher) / {1 - args.soft_alpha:.2f} (hard cross-entropy)")
        trainer = DistilTrainer(alpha=args.soft_alpha, temperature=args.soft_temp,
                                **trainer_kwargs)
    elif args.class_weights:
        counts = train_df["label"].value_counts().reindex(range(len(intents))).fillna(1.0)
        weights = torch.tensor(
            (len(train_df) / (len(intents) * counts)).values, dtype=torch.float
        )
        print(f"  class weights on: {weights.min():.2f} .. {weights.max():.2f}")
        trainer = WeightedTrainer(class_weights=weights, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    trainer.train()
    metrics = trainer.evaluate()

    print("\n" + "=" * 60)
    print(f"HOLDOUT -- {args.benchmark}")
    print("=" * 60)
    print(f"  balanced accuracy        {metrics['eval_balanced_acc']:.4f}   (every class equal)")
    print(f"  prior-weighted accuracy  {metrics['eval_prior_weighted_acc']:.4f}   (weighted by pool frequency)")
    print(f"  macro F1                 {metrics['eval_macro_f1']:.4f}")
    print(f"  worst class recall       {metrics['eval_worst_class_recall']:.4f}")
    print(f"  classes at zero          {metrics['eval_classes_at_zero']}")
    gap = metrics["eval_prior_weighted_acc"] - metrics["eval_balanced_acc"]
    print(f"\n  gap {gap:+.4f} -- prior-weighted above balanced means the rare classes")
    print("  are the weak ones; the true test score sits between the two.")

    # The honest number. Holdout is curated and capped at 60 characters while
    # test.tsv is 24.4% longer than that, so holdout FLATTERS -- the labelling
    # gate read 98.4% on holdout against 85% measured by hand on pool messages.
    # These rows are pool-drawn, human-confirmed, and excluded from training.
    human = load_human_validation(args.benchmark)
    if len(human):
        hv = Dataset.from_pandas(
            human.assign(label=human["intent"].map(intent2id))[["text", "label"]],
            preserve_index=False,
        ).map(tokenize, batched=True)
        hv = hv.remove_columns([c for c in hv.column_names if c not in keep])
        hm = trainer.evaluate(eval_dataset=hv, metric_key_prefix="human")
        metrics.update(hm)
        print(f"\n  HUMAN-VERIFIED SET ({len(human)} pool messages, same "
              f"distribution as test)")
        print(f"    accuracy {hm['human_accuracy']:.4f}")
        print(f"    vs holdout balanced {metrics['eval_balanced_acc']:.4f} "
              f"-- difference {hm['human_accuracy'] - metrics['eval_balanced_acc']:+.4f}")
        print("    Trust this one for absolute performance; holdout is curated\n"
              "    and reads high. Use holdout only to compare runs.")
    else:
        print("\n  no human-verified set -- run: python src/human_gate.py")

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    save_label_map(args.benchmark, out_dir, train_prior=prior)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "encoder": args.encoder,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "cap_per_class": args.cap_per_class,
                "class_weights": args.class_weights,
                "n_train": len(train_df),
                "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  saved -> {out_dir}")


if __name__ == "__main__":
    main()
