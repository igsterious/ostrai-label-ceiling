"""Build a stratified sample of pseudo-labels for manual human review.

Why this is needed: the QA gate scored the labeller on holdout, but holdout
maxes out at 60 characters while 30.4% of benchmark A's pool is longer than
that. The gate therefore says nothing about a third of the data it was used to
approve. Only human eyes can close that gap -- there is no other ground truth.

Sampling is stratified into two strata, and the split is the whole point:

  LONG     messages longer than any holdout example. The blind spot. Reviewing
           these is the actual purpose of the exercise.
  TYPICAL  messages inside holdout's length range. A control: QA already
           predicts ~98% here, so if manual review disagrees wildly, the
           problem is the review process rather than the labels.

Each item carries the few-shot examples for its assigned category, so a
reviewer can judge "does this belong with these?" without memorising 150
category definitions.

    python src/make_review_set.py            # both benchmarks
    python src/make_review_set.py --n-long 80
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from config import BENCHMARKS, artifact_dir
from data import load_pool_unlabeled
from labeller import read_labels
from training_data import find_pseudo_label_file, load_fewshot, load_holdout

SEED = 20260905


def build(benchmark, n_long, n_typical):
    pool = load_pool_unlabeled(benchmark)
    labels = read_labels(find_pseudo_label_file(benchmark))
    fewshot = load_fewshot(benchmark)
    holdout_max = int(load_holdout(benchmark)["text"].str.len().max())

    df = pool.merge(labels, on="id")
    df["length"] = df["text"].str.len()

    long_pool = df[df["length"] > holdout_max]
    typical_pool = df[df["length"] <= holdout_max]

    rng_long = long_pool.sample(min(n_long, len(long_pool)), random_state=SEED)
    rng_typ = typical_pool.sample(min(n_typical, len(typical_pool)), random_state=SEED)
    rng_long = rng_long.assign(stratum="long")
    rng_typ = rng_typ.assign(stratum="typical")

    sample = pd.concat([rng_long, rng_typ])
    # Shuffle so the reviewer cannot tell which stratum they are in -- knowing
    # "this is the suspect group" would bias how harshly they judge it.
    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    examples = {
        intent: list(rows["text"])
        for intent, rows in fewshot.groupby("intent")
    }

    items = [
        {
            "id": r.id,
            "text": r.text,
            "assigned": r.intent,
            "stratum": r.stratum,
            "length": int(r.length),
            "examples": examples.get(r.intent, []),
        }
        for r in sample.itertuples()
    ]
    return {
        "benchmark": benchmark,
        "holdout_max_len": holdout_max,
        "pool_size": len(df),
        "n_long_available": len(long_pool),
        "items": items,
    }


def main():
    ap = argparse.ArgumentParser(description="Sample pseudo-labels for manual review")
    ap.add_argument("--n-long", type=int, default=50, help="samples beyond holdout's length range")
    ap.add_argument("--n-typical", type=int, default=30, help="control samples inside it")
    args = ap.parse_args()

    for benchmark in BENCHMARKS:
        payload = build(benchmark, args.n_long, args.n_typical)
        out = artifact_dir(benchmark) / "review_set.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        n_long = sum(1 for i in payload["items"] if i["stratum"] == "long")
        print(
            f"{benchmark}: {len(payload['items'])} items "
            f"({n_long} long, {len(payload['items'])-n_long} typical) -> {out}"
        )
        print(
            f"  holdout tested up to {payload['holdout_max_len']} chars; "
            f"{payload['n_long_available']:,} pool messages exceed that"
        )


if __name__ == "__main__":
    main()
