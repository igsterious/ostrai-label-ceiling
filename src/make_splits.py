"""Split the 10-per-class labeled examples into few-shot and holdout halves.

Run this once, before anything else. Everything downstream reads the files
it writes, so the boundary between "the labeller has seen this" and "this is
clean evaluation data" is decided in exactly one place and never drifts.

    python src/make_splits.py

Writes, per benchmark, into artifacts/<benchmark>/:
    fewshot.tsv   5 per class -- may go into the LLM labelling prompt
    holdout.tsv   5 per class -- must never reach the labeller

The holdout half does double duty:
  1. QA gate for the labeller. Run the labelling pipeline over these
     messages, compare to the known intent, get per-class accuracy. This
     is honest because the labeller's prompt only ever contained fewshot.
  2. Validation for the trained classifier, which is fitted on fewshot +
     pseudo-labelled pool -- again disjoint from holdout.

The split is stratified per class rather than by percentage. A 50% random
split over 77 or 150 classes leaves some classes with zero holdout rows by
chance, and a class you cannot measure is a class you cannot fix.
"""

import numpy as np
import pandas as pd

from config import BENCHMARKS, N_FEWSHOT_PER_CLASS, N_HOLDOUT_PER_CLASS, SPLIT_SEED, artifact_dir
from data import load_labeled, validate, write_tsv


def split_one(benchmark):
    labeled = load_labeled(benchmark)

    # One generator seeded once per benchmark, so re-running gives identical
    # files. Reproducibility matters here: if the split moved between runs,
    # a validation score from Tuesday would not be comparable to one today.
    rng = np.random.default_rng(SPLIT_SEED)

    fewshot_parts, holdout_parts = [], []
    # sort_values first so group order does not depend on row order in the file
    for intent, group in labeled.sort_values("id").groupby("intent", sort=True):
        need = N_FEWSHOT_PER_CLASS + N_HOLDOUT_PER_CLASS
        if len(group) < need:
            raise ValueError(
                f"{benchmark}: class {intent!r} has only {len(group)} examples, "
                f"need {need} for a {N_FEWSHOT_PER_CLASS}/{N_HOLDOUT_PER_CLASS} split"
            )
        # permutation over positions, then take the first N for few-shot
        order = rng.permutation(len(group))
        shuffled = group.iloc[order]
        fewshot_parts.append(shuffled.iloc[:N_FEWSHOT_PER_CLASS])
        holdout_parts.append(shuffled.iloc[N_FEWSHOT_PER_CLASS:need])

    fewshot = pd.concat(fewshot_parts).sort_values("id").reset_index(drop=True)
    holdout = pd.concat(holdout_parts).sort_values("id").reset_index(drop=True)

    # The guarantee the rest of the project relies on: no message is in both.
    overlap = set(fewshot["id"]) & set(holdout["id"])
    if overlap:
        raise AssertionError(f"{benchmark}: {len(overlap)} ids in both splits")

    out = artifact_dir(benchmark)
    write_tsv(fewshot, out / "fewshot.tsv")
    write_tsv(holdout, out / "holdout.tsv")
    return fewshot, holdout


def main():
    for benchmark in BENCHMARKS:
        summary = validate(benchmark)
        fewshot, holdout = split_one(benchmark)
        print(
            f"{benchmark}: {summary['n_intents']} intents, "
            f"{summary['n_labeled']} labeled ({summary['per_class_min']}-"
            f"{summary['per_class_max']} per class)\n"
            f"  pool {summary['n_pool']} ({summary['n_pool_unlabeled']} to label), "
            f"test {summary['n_test']}\n"
            f"  -> fewshot {len(fewshot)} ({fewshot['intent'].value_counts().min()}/class), "
            f"holdout {len(holdout)} ({holdout['intent'].value_counts().min()}/class)"
        )


if __name__ == "__main__":
    main()
