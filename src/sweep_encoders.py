"""Train several encoders and compare them on BOTH axes at once.

The leaderboard scores accuracy and speed, so an encoder cannot be chosen on
accuracy alone. RoBERTa-base has ~5.5x the parameters of MiniLM-L6 and will
almost certainly win on accuracy while losing on ms/example. Which one is the
better submission depends on how those trade off, and that is measurable rather
than arguable -- so measure it.

Every candidate is a GENERIC pretrained encoder. None carries intent labels, so
rule 2 is satisfied; a model fine-tuned on Banking77 or CLINC150 would not be.

Accuracy comes from holdout (never fitted on). Speed comes from the same
protocol used for submission, on the same test-set texts -- reading test.tsv's
TEXT for timing is fine and does not violate rule 1, which forbids training,
tuning, or selecting on it. Selection here is on holdout accuracy and measured
latency, never on test labels, which we do not have.

Run in Colab on a T4 -- speed numbers from any other device are meaningless:

    python src/sweep_encoders.py benchmark_a
    python src/sweep_encoders.py benchmark_a --encoders roberta-base distilbert-base-uncased
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from config import artifact_dir
from data import load_test

# The candidates, smallest first. Params are approximate and include the
# embedding table, which dominates for the smaller models.
CANDIDATES = [
    "sentence-transformers/all-MiniLM-L6-v2",  # ~22M   6 layers, 384 hidden
    "distilbert-base-uncased",                 # ~66M   6 layers, 768 hidden
    "distilroberta-base",                      # ~82M   6 layers, 768 hidden
    "roberta-base",                            # ~125M 12 layers, 768 hidden
]


def slug(encoder):
    return encoder.replace("/", "_")


def run_one(benchmark, encoder, epochs, batch_size, lr, fp16):
    """Train one encoder, then time it. Returns a result dict or None.

    epochs/batch_size/lr are None unless the caller overrode them, and are
    then simply not passed -- train.py owns the defaults. Keeping a second
    copy here is how this script silently ran the pre-fix 3-epoch recipe after
    train.py moved to 8, producing a comparison that measured undertraining
    rather than the encoders.
    """
    out_dir = artifact_dir(benchmark) / "sweep" / slug(encoder)
    cmd = [
        sys.executable, str(Path(__file__).parent / "train.py"), benchmark,
        "--encoder", encoder,
        "--out", str(out_dir),
    ]
    for flag, value in (("--epochs", epochs), ("--batch-size", batch_size), ("--lr", lr)):
        if value is not None:
            cmd += [flag, str(value)]
    print(f"\n{'='*70}\nTRAINING {encoder}\n{'='*70}", flush=True)
    if subprocess.run(cmd).returncode != 0:
        print(f"  {encoder} FAILED to train -- skipping")
        return None

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]

    # Time it on the real test texts, through the real pipeline.
    from classify import Classifier, time_classifier

    texts = load_test(benchmark)["text"].tolist()
    clf = Classifier(out_dir, fp16=fp16)
    ms, _ = time_classifier(clf, texts)
    del clf
    try:
        import torch

        torch.cuda.empty_cache()  # so the next candidate starts with a clean GPU
    except ImportError:
        pass

    return {
        "encoder": encoder,
        "balanced_acc": metrics["eval_balanced_acc"],
        "prior_weighted_acc": metrics["eval_prior_weighted_acc"],
        # the honest one -- holdout is curated and reads high
        "human_acc": metrics.get("human_accuracy"),
        "worst_class_recall": metrics["eval_worst_class_recall"],
        "classes_at_zero": int(metrics["eval_classes_at_zero"]),
        "ms_per_example": ms,
    }


def main():
    ap = argparse.ArgumentParser(description="Compare encoders on accuracy and speed")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--encoders", nargs="+", default=CANDIDATES)
    # No defaults here on purpose -- train.py is the single source of truth.
    ap.add_argument("--epochs", type=float, help="override train.py's default")
    ap.add_argument("--batch-size", type=int, help="override train.py's default")
    ap.add_argument("--lr", type=float, help="override train.py's default")
    ap.add_argument("--fp16", action="store_true", help="half precision at inference")
    args = ap.parse_args()

    results = [
        r for enc in args.encoders
        if (r := run_one(args.benchmark, enc, args.epochs, args.batch_size, args.lr, args.fp16))
    ]
    if not results:
        raise SystemExit("every candidate failed")

    results.sort(key=lambda r: -r["balanced_acc"])
    # The fastest model is the baseline for BOTH columns. Measuring accuracy
    # against the best model while measuring latency against the fastest made
    # the winner read "+0.00 pp for 1.90x the latency", which states the
    # trade-off backwards.
    baseline = min(results, key=lambda r: r["ms_per_example"])
    fastest = baseline["ms_per_example"]

    print("\n" + "=" * 88)
    print(f"ENCODER COMPARISON -- {args.benchmark}" + (" (fp16)" if args.fp16 else ""))
    print("=" * 88)
    n_eval = None
    print(f"{'encoder':<42s} {'bal_acc':>8s} {'human':>7s} {'worst':>6s} {'ms/ex':>8s} {'vs fastest':>11s}")
    for r in results:
        human = f"{r['human_acc']:.4f}" if r.get("human_acc") is not None else "  --  "
        print(f"{r['encoder']:<42s} {r['balanced_acc']:>8.4f} {human:>7s} "
              f"{r['worst_class_recall']:>6.2f} {r['ms_per_example']:>8.4f} "
              f"{r['ms_per_example']/fastest:>10.2f}x")

    print(f"\nTrade-off vs the fastest model ({baseline['encoder']}):")
    for r in results:
        d_acc = (r["balanced_acc"] - baseline["balanced_acc"]) * 100
        slower = r["ms_per_example"] / fastest
        if r is baseline:
            print(f"  {r['encoder']:<42s} baseline: {r['balanced_acc']:.4f} at "
                  f"{fastest:.4f} ms/example")
        else:
            print(f"  {r['encoder']:<42s} {d_acc:+6.2f} pp accuracy for "
                  f"{slower:5.2f}x the latency")

    out = artifact_dir(args.benchmark) / "encoder_sweep.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    # A holdout of n rows cannot resolve an accuracy difference smaller than
    # 1/n -- on benchmark A's 385 rows that is 0.26pp, which is exactly the
    # gap DistilBERT and RoBERTa showed. Say so, rather than letting a
    # one-message difference look like a reason to pick the slower model.
    from training_data import load_holdout

    n_holdout = len(load_holdout(args.benchmark))
    resolution = 100.0 / n_holdout
    spread = (max(r["balanced_acc"] for r in results)
              - min(r["balanced_acc"] for r in results)) * 100
    print(f"\nHoldout is {n_holdout} rows, so the smallest difference it can resolve is "
          f"{resolution:.2f} pp (one message).")
    if spread <= resolution * 1.5:
        print(f"The spread here is {spread:.2f} pp -- within noise. Treat these encoders as\n"
              f"equally accurate and pick on latency alone.")
    else:
        print(f"The spread here is {spread:.2f} pp, or about {spread/resolution:.1f} messages.")
    print(
        "\nThe standing averages the two benchmark accuracies and scores speed\n"
        "separately, so a model that buys +0.2pp for 2x the latency is the wrong\n"
        "trade, and one that buys +3pp is not."
    )


if __name__ == "__main__":
    main()
