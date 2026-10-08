"""Fold the hand-decided rows back into a label file.

Corrections REPLACE labels in place rather than adding rows: the training set
keeps its size and its distribution, and only the noise goes down. That is the
distinction from the 220 judged messages, which were added as extra gold rows
and cost 1.13 points because they skewed the set toward confusable cases.

Starts from the 3-way consensus labelling (the best we have) and overwrites
only the ids a human decided.

    python tools/apply_disputed.py benchmark_a
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import labels_dir  # noqa: E402
from data import load_intents  # noqa: E402
from labeller import read_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Apply hand decisions to the labels")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--base", default="pseudo_labels.gpt3way.jsonl",
                    help="the labelling to correct")
    ap.add_argument("--tag", default="handfixed")
    args = ap.parse_args()

    d = labels_dir(args.benchmark)
    decisions = d / "human_disputed.tsv"
    if not decisions.exists():
        raise SystemExit(f"{decisions} not found -- run tools/review_disputed.py first")

    fixed = pd.read_csv(decisions, sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                        keep_default_na=False)
    fixed = dict(zip(fixed["id"], fixed["intent"]))

    valid = set(load_intents(args.benchmark))
    bad = {v for v in fixed.values() if v not in valid}
    if bad:
        raise SystemExit(f"decisions contain unknown intents: {sorted(bad)[:5]}")

    base = read_labels(d / args.base).set_index("id")["intent"].to_dict()
    changed = sum(1 for i, v in fixed.items() if base.get(i) not in (None, v))

    out = d / f"pseudo_labels.{args.tag}.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for i, v in base.items():
            fh.write(json.dumps({"id": i, "intent": fixed.get(i, v),
                                 "model": "gpt3way+human",
                                 "rules": "handfixed"}) + "\n")

    n = len(base)
    print(f"{args.benchmark}: {n:,} rows written to {out.name}")
    print(f"  human decisions available : {len(fixed):,}")
    print(f"  of those, CHANGED a label : {changed:,} "
          f"({changed/max(1,len(fixed)):.0%} of decisions, {changed/n:.1%} of the set)")
    print(f"  agreed with the consensus : {len(fixed) - changed:,}")
    print(f"\n  train with: --labels-file labels/{args.benchmark}/{out.name}")


if __name__ == "__main__":
    main()
