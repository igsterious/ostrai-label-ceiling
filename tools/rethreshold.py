"""Rebuild a self-training label set at a different threshold, without retraining.

self_train.py saves every out-of-sample prediction and its confidence before
thresholding, precisely so the threshold can be reconsidered -- but it has no
path to read that file back, so changing the threshold means re-running all the
folds. Three DeBERTa trainings to move one number is absurd, and it also
introduces run-to-run variation that makes two thresholds incomparable: the same
procedure run twice gave 1,007 and 971 disagreements, and gate scores a full
point apart.

Reading from the saved predictions keeps every threshold on identical
underlying predictions, so a comparison between them is a comparison of the
threshold and nothing else.

    python tools/rethreshold.py benchmark_a --tag gpt3way --threshold 0.9
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import artifact_dir, labels_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Re-threshold saved self-training predictions")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--tag", help="the --tag the self_train run used")
    ap.add_argument("--threshold", type=float, required=True)
    args = ap.parse_args()

    tag = f"_{args.tag}" if args.tag else ""
    oos = artifact_dir(args.benchmark) / f"selftrain_oos{tag}.tsv"
    if not oos.exists():
        raise SystemExit(f"{oos} not found -- run src/self_train.py first")

    df = pd.read_csv(oos, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    df["conf"] = df["conf"].astype(float)

    flip = (df["pred"] != df["intent"]) & (df["conf"] >= args.threshold)
    out_intent = df["intent"].where(~flip, df["pred"])

    out = labels_dir(args.benchmark) / f"pseudo_labels.selftrain{tag}_t{args.threshold}.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for i, v in zip(df["id"], out_intent):
            fh.write(json.dumps({"id": i, "intent": v,
                                 "model": f"selftrain(t={args.threshold})"}) + "\n")

    n = len(df)
    disagree = int((df["pred"] != df["intent"]).sum())
    print(f"{args.benchmark}: {n:,} rows from {oos.name}")
    print(f"  model disagrees with the label on {disagree:,} ({disagree/n:.1%})")
    print(f"  of those, {int(flip.sum()):,} are at confidence >= {args.threshold} "
          f"and get rewritten")
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
