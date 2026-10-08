"""Majority vote across several labellings of the same pool, plus the subset
they all agree on.

WHY THE SAME MODEL, SEVERAL TIMES. Earlier consensus attempts here mixed
labellers of different quality -- sonnet three ways, or gpt-5.1 with gemini and
deepseek -- and a consensus containing a bad voter is worse than the best voter
alone, which is what those experiments found. Three runs of the SAME top-grade
labeller is a different operation: it separates the errors that are stochastic
(a different sample would have got it right) from the ones that are systematic
(the labeller genuinely believes the wrong thing). Only the first kind is
fixable by voting.

WHAT AGREEMENT BUYS. Measured on benchmark A with three gpt-5.1 runs: 91.6% of
rows are unanimous and a human judged those right 92% of the time; the 7.8%
that split 2/1 were right 57% of the time. So agreement is a usable proxy for
correctness, and the non-unanimous rows are where the training noise lives.

    python tools/consensus.py benchmark_a --tag gpt3way \
        pseudo_labels.openai_gpt-5.1.jsonl \
        pseudo_labels.openai_gpt-5.1.rules-v2.jsonl \
        pseudo_labels.openai_gpt-5.1.b10.jsonl

Writes two files: the majority label for every row, and the unanimous subset.
Which is better is an empirical question -- dropping rows removes wrong labels
and correct ones together -- so both are produced and both get tested.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import labels_dir  # noqa: E402
from labeller import read_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Majority vote over several labellings")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("files", nargs="+", help="label filenames inside labels/<benchmark>/")
    ap.add_argument("--tag", required=True, help="goes in the output filenames")
    args = ap.parse_args()

    if len(args.files) < 3:
        raise SystemExit("majority voting needs at least three labellings")

    d = labels_dir(args.benchmark)
    series = []
    for f in args.files:
        p = d / f
        if not p.exists():
            raise SystemExit(f"no such labelling: {p}")
        series.append(read_labels(p).set_index("id")["intent"])

    ids = series[0].index
    for s in series[1:]:
        ids = ids.intersection(s.index)
    print(f"{args.benchmark}: {len(ids):,} rows labelled by all {len(series)} runs")

    majority, unanimous, levels = {}, {}, Counter()
    for i in ids:
        votes = [s[i] for s in series]
        c = Counter(votes)
        top, n = c.most_common(1)[0]
        levels[n] += 1
        majority[i] = top
        if n == len(series):
            unanimous[i] = top

    for n in sorted(levels, reverse=True):
        share = levels[n] / len(ids)
        mark = "  <- unanimous" if n == len(series) else ""
        print(f"  {n}/{len(series)} agree  {levels[n]:6,}  ({share:5.1%}){mark}")

    # A tie among three distinct answers has no majority; most_common picks one
    # arbitrarily, so those rows are majority-labelled but never unanimous, and
    # they are excluded from the unanimous file where it matters.
    def write(path, mapping, note):
        with open(path, "w", encoding="utf-8") as fh:
            for i, v in mapping.items():
                fh.write(json.dumps({"id": i, "intent": v,
                                     "model": f"consensus/{args.tag}",
                                     "rules": "consensus"}) + "\n")
        print(f"  wrote {path.name}: {len(mapping):,} rows -- {note}")

    write(d / f"pseudo_labels.{args.tag}.jsonl", majority,
          "majority label for every row")
    write(d / f"pseudo_labels.{args.tag}_unanimous.jsonl", unanimous,
          f"only where all {len(series)} agree ({len(ids) - len(unanimous):,} dropped)")

    first = series[0]
    k = sum(1 for i in ids if majority[i] != first[i])
    print(f"\n  majority differs from {args.files[0]} on {k:,} rows ({k/len(ids):.1%})")


if __name__ == "__main__":
    main()
