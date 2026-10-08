"""Paired comparison of two labeller QA runs on the same holdout messages.

Overall accuracy on 385 rows carries a Wilson interval of roughly +/-3 points,
so two runs a point apart are not distinguishable by their headline numbers.
But the runs are PAIRED -- same messages, same truth -- so the rows where both
agree carry no information and can be dropped. What is left is the discordant
pairs, and McNemar's exact test asks whether their split is worse than a coin
flip. That is a far sharper instrument on the same data.

    python tools/compare_qa.py benchmark_a \
        qa_predictions.openai_gpt-5.1.jsonl qa_predictions.openai_gpt-5.1.b10.jsonl
"""

from __future__ import annotations

import argparse
import csv
import sys
from math import comb
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import artifact_dir  # noqa: E402
from labeller import read_labels  # noqa: E402


def mcnemar_exact(b, c):
    """Two-sided exact p for a binomial(b+c, 0.5) split -- no chi-square
    approximation, which is unreliable when b+c is small."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser(description="Compare two labeller QA runs")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("first")
    ap.add_argument("second")
    args = ap.parse_args()

    d = artifact_dir(args.benchmark)
    holdout = pd.read_csv(d / "holdout.tsv", sep="\t", quoting=csv.QUOTE_NONE,
                          dtype=str, keep_default_na=False)
    truth = holdout.set_index("id")["intent"]

    a = read_labels(d / args.first).set_index("id")["intent"]
    b = read_labels(d / args.second).set_index("id")["intent"]
    ids = [i for i in truth.index if i in a.index and i in b.index]
    if not ids:
        raise SystemExit("no shared ids -- did both runs complete?")

    ok_a = {i: a[i] == truth[i] for i in ids}
    ok_b = {i: b[i] == truth[i] for i in ids}

    both = sum(1 for i in ids if ok_a[i] and ok_b[i])
    only_a = sum(1 for i in ids if ok_a[i] and not ok_b[i])
    only_b = sum(1 for i in ids if ok_b[i] and not ok_a[i])
    neither = sum(1 for i in ids if not ok_a[i] and not ok_b[i])

    na, nb = both + only_a, both + only_b
    n = len(ids)
    print(f"{args.benchmark}: {n} holdout messages scored by both runs\n")
    print(f"  {args.first:50s} {na:4d}/{n}  {na/n:6.1%}")
    print(f"  {args.second:50s} {nb:4d}/{n}  {nb/n:6.1%}")
    print(f"\n  both right {both:4d}    both wrong {neither:4d}")
    print(f"  only first {only_a:4d}    only second {only_b:4d}   <- the informative rows")
    p = mcnemar_exact(only_a, only_b)
    print(f"\n  McNemar exact p = {p:.4f}")
    if p < 0.05:
        better = args.second if only_b > only_a else args.first
        print(f"  A real difference; {better} is better.")
    else:
        print("  Not distinguishable on this holdout. A difference this small is\n"
              "  not worth $6.50 of relabelling -- the pool run would be a coin flip.")


if __name__ == "__main__":
    main()
