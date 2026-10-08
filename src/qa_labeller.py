"""QA gate: score the labelling pipeline against known-good labels.

Runs the exact same labeller over holdout.tsv, whose true labels we have but
the labeller has never seen, and reports how often it agrees. This is the step
the brief requires before any pseudo-label is allowed into training.

Why the number it produces is honest: build_system_prompt() reads fewshot.tsv
only, and make_splits.py guarantees fewshot and holdout share no ids and no
text. assert_disjoint() below re-checks that at runtime rather than trusting
the comment -- if someone later points the labeller at labeled_examples.tsv,
this fails loudly instead of silently reporting a great score.

Overall accuracy is reported but is not the number to trust. Across 77 and 150
classes a good average hides classes at zero, and a class the labeller always
gets wrong becomes a block of confidently mislabelled training data. So the
report leads with per-class accuracy, the worst classes, and the confusion
pairs driving the errors.

Usage:
    python src/qa_labeller.py benchmark_a --limit 100   # cheap smoke test
    python src/qa_labeller.py benchmark_a               # the real gate
"""

from __future__ import annotations

import argparse
import csv

import pandas as pd

from config import artifact_dir
from data import load_intents, write_tsv
from labeller import DEFAULT_MODEL, label_messages, load_fewshot, model_slug, read_labels


def load_holdout(benchmark):
    path = artifact_dir(benchmark) / "holdout.tsv"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run: python src/make_splits.py")
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def assert_disjoint(benchmark):
    """Fail loudly if the labeller could have seen the holdout messages."""
    fewshot, holdout = load_fewshot(benchmark), load_holdout(benchmark)
    shared_ids = set(fewshot["id"]) & set(holdout["id"])
    shared_text = set(fewshot["text"]) & set(holdout["text"])
    if shared_ids or shared_text:
        raise SystemExit(
            f"LEAK: fewshot and holdout overlap "
            f"({len(shared_ids)} ids, {len(shared_text)} texts). "
            "QA numbers would be meaningless. Re-run make_splits.py."
        )
    return fewshot, holdout


def per_class_report(merged, intents):
    """Per-class precision, recall and F1, computed by hand.

    Done manually rather than with sklearn to keep the local dependency list at
    pandas + requests -- see CLAUDE.md on why torch and friends stay in Colab.

    recall    = of the messages truly in this class, how many did we get?
                (low recall = this class gets lost)
    precision = of the messages we put in this class, how many belong?
                (low precision = this class is a magnet for other classes)
    """
    rows = []
    for intent in intents:
        truth = merged["intent_true"] == intent
        pred = merged["intent_pred"] == intent
        tp = int((truth & pred).sum())
        n_true = int(truth.sum())
        n_pred = int(pred.sum())
        recall = tp / n_true if n_true else float("nan")
        precision = tp / n_pred if n_pred else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if n_true and n_pred and (precision + recall) > 0
            else 0.0
        )
        rows.append(
            {
                "intent": intent,
                "n_true": n_true,
                "n_pred": n_pred,
                "correct": tp,
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def confusion_pairs(merged, top=15):
    """The most frequent (true -> predicted) mistakes."""
    wrong = merged[merged["intent_true"] != merged["intent_pred"]]
    if wrong.empty:
        return pd.DataFrame(columns=["intent_true", "intent_pred", "n"])
    pairs = (
        wrong.groupby(["intent_true", "intent_pred"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    return pairs.head(top)


def main():
    ap = argparse.ArgumentParser(description="Score the labeller against holdout labels")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--limit", type=int, help="only score the first N holdout messages")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fewshot, holdout = assert_disjoint(args.benchmark)
    print(
        f"leak check passed: {len(fewshot)} fewshot / {len(holdout)} holdout, "
        "no shared ids or text\n"
    )

    intents = load_intents(args.benchmark)
    suffix = f"_limit{args.limit}" if args.limit else ""
    if args.limit:
        holdout = holdout.head(args.limit)

    # Batch size goes in the name for the same reason the model does: the
    # resume logic skips ids already present, so a second run at a different
    # batch size would find the first run's answers and score them again.
    batch_tag = "" if args.batch_size == 50 else f".b{args.batch_size}"
    out = (artifact_dir(args.benchmark)
           / f"qa_predictions.{model_slug(args.model)}{batch_tag}{suffix}.jsonl")
    label_messages(
        args.benchmark,
        holdout[["id", "text"]],
        out,
        model=args.model,
        batch_size=args.batch_size,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return

    preds = read_labels(out)
    merged = holdout.merge(
        preds.rename(columns={"intent": "intent_pred"}), on="id", how="left"
    ).rename(columns={"intent": "intent_true"})
    merged["intent_pred"] = merged["intent_pred"].fillna("<UNLABELED>")

    n = len(merged)
    correct = int((merged["intent_true"] == merged["intent_pred"]).sum())
    unlabeled = int((merged["intent_pred"] == "<UNLABELED>").sum())

    stats = per_class_report(merged, intents)
    present = stats[stats["n_true"] > 0]

    print("\n" + "=" * 66)
    print(f"QA REPORT -- {args.benchmark}  ({args.model})")
    print("=" * 66)
    print(f"  overall accuracy   {correct}/{n} = {correct/n:.1%}")
    print(f"  macro F1           {present['f1'].mean():.3f}")
    print(f"  macro recall       {present['recall'].mean():.3f}")
    print(f"  unlabeled          {unlabeled}")
    print(f"  classes at 100%    {int((present['recall'] == 1.0).sum())}/{len(present)}")
    print(f"  classes at 0%      {int((present['recall'] == 0.0).sum())}/{len(present)}")

    worst = present.sort_values(["recall", "f1"]).head(12)
    print("\n  WORST CLASSES (low recall = the labeller loses this category)")
    print(f"    {'intent':<42s} {'recall':>7s} {'prec':>7s} {'n':>4s}")
    for r in worst.itertuples():
        print(f"    {r.intent:<42s} {r.recall:>7.0%} {r.precision:>7.0%} {r.n_true:>4d}")

    pairs = confusion_pairs(merged)
    if not pairs.empty:
        print("\n  TOP CONFUSIONS (true -> predicted)")
        for r in pairs.itertuples():
            print(f"    {r.n:>3d}x  {r.intent_true}  ->  {r.intent_pred}")

    stats_path = artifact_dir(args.benchmark) / f"qa_per_class{suffix}.tsv"
    write_tsv(stats.round(4).astype(str), stats_path)
    errors = merged[merged["intent_true"] != merged["intent_pred"]]
    errors_path = artifact_dir(args.benchmark) / f"qa_errors{suffix}.tsv"
    write_tsv(errors[["id", "text", "intent_true", "intent_pred"]], errors_path)

    print(f"\n  per-class metrics -> {stats_path}")
    print(f"  every mistake     -> {errors_path}")
    print(
        "\n  Read qa_errors.tsv before training. Systematic confusions between "
        "two\n  categories are usually fixable in the prompt; scattered errors "
        "usually\n  are not, and are the labelling ceiling."
    )


if __name__ == "__main__":
    main()
