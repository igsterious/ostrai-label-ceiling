"""Investigate whether the QA scores are real or inflated.

98.4% / 99.7% is high enough to deserve suspicion. Four mechanisms could
produce a number that good without the labeller being that good:

1. NEAR-DUPLICATE LEAK. assert_disjoint() catches exact text matches between
   fewshot and holdout. It would not catch "my card is broken" against "my
   card's broken" -- effectively showing the labeller the answer.

2. UNREPRESENTATIVE HOLDOUT. holdout comes from labeled_examples.tsv. If those
   10-per-class examples were hand-picked as clean, prototypical demonstrations
   rather than sampled at random from the pool, then holdout is EASIER than the
   pool, and QA overstates pseudo-label quality on the thing we actually
   labelled. This is the most dangerous of the four because nothing in the
   pipeline would reveal it.

3. MEMORISATION. These label spaces are public benchmarks. A frontier model may
   recall the answers from pretraining rather than reasoning from the prompt.
   Note this does not make the pseudo-labels wrong -- recalled labels are still
   correct labels -- but it does mean QA measures recall, not generalisation,
   and the score would not transfer to genuinely novel data.

4. TRIVIAL SEPARABILITY. The categories may simply be easy to tell apart, in
   which case the score is honest and there is nothing to fix.

This script tests 1 and 2. Test 3 is addressed by cross-model agreement
(cross_check.py) and test 4 is what manual review settles.

    python src/qa_diagnostics.py
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from config import BENCHMARKS
from data import load_labeled, load_pool, load_pool_unlabeled
from training_data import load_fewshot, load_holdout


def normalise(text):
    """Aggressive normalisation, so only real wording differences survive."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)   # drop punctuation and apostrophes
    return re.sub(r"\s+", " ", text)


def token_set(text):
    return set(normalise(text).split())


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def near_duplicate_leak(benchmark, threshold=0.8):
    """Find holdout messages suspiciously similar to a fewshot message.

    Exact matches are already ruled out; this looks for paraphrases. A holdout
    item that shares >=80% of its words with a prompt example is, in practice,
    an answer the labeller was shown.
    """
    fewshot, holdout = load_fewshot(benchmark), load_holdout(benchmark)
    fs_norm = {normalise(t) for t in fewshot["text"]}
    fs_tokens = [(t, token_set(t)) for t in fewshot["text"]]

    exact_after_norm = sum(1 for t in holdout["text"] if normalise(t) in fs_norm)

    hits = []
    for row in holdout.itertuples():
        ht = token_set(row.text)
        best_score, best_text = 0.0, None
        for ft, ftok in fs_tokens:
            s = jaccard(ht, ftok)
            if s > best_score:
                best_score, best_text = s, ft
        if best_score >= threshold:
            hits.append((round(best_score, 3), row.text, best_text))
    return exact_after_norm, sorted(hits, reverse=True)


def representativeness(benchmark):
    """Are the gold examples a random sample of the pool, or curated?

    Two tells. (a) Length: curated demonstrations are usually shorter and
    cleaner than real traffic. (b) Id spread: the gold examples are a subset of
    the pool by id, so if they were sampled at random their ids should be
    spread uniformly across the pool's id range. Clustering would suggest they
    were taken from a particular slice.
    """
    labeled = load_labeled(benchmark)
    pool = load_pool(benchmark)
    unlabeled = load_pool_unlabeled(benchmark)

    lab_len = labeled["text"].str.len()
    pool_len = unlabeled["text"].str.len()

    lab_ids = labeled["id"].astype(int).values
    pool_ids = pool["id"].astype(int).values
    lo, hi = pool_ids.min(), pool_ids.max()
    # Kolmogorov-Smirnov style: max gap between the empirical CDF of the gold
    # ids and the uniform CDF over the pool's id range.
    u = np.sort((lab_ids - lo) / (hi - lo))
    n = len(u)
    ks = float(np.max(np.abs(u - np.arange(1, n + 1) / n)))
    ks_crit = 1.36 / np.sqrt(n)  # ~5% significance

    return {
        "gold_len_median": float(lab_len.median()),
        "pool_len_median": float(pool_len.median()),
        "gold_len_mean": float(lab_len.mean()),
        "pool_len_mean": float(pool_len.mean()),
        "gold_len_p90": float(lab_len.quantile(0.9)),
        "pool_len_p90": float(pool_len.quantile(0.9)),
        "ks_stat": ks,
        "ks_critical": float(ks_crit),
        "ids_look_random": ks < ks_crit,
    }


def duplicate_texts_in_pool(benchmark):
    """Exact duplicate messages within the pool, which inflate easy classes."""
    pool = load_pool(benchmark)
    counts = Counter(normalise(t) for t in pool["text"])
    dupes = {t: c for t, c in counts.items() if c > 1}
    return len(dupes), sum(dupes.values()) - len(dupes)


def main():
    for benchmark in BENCHMARKS:
        print("=" * 72)
        print(f"{benchmark}")
        print("=" * 72)

        exact_norm, hits = near_duplicate_leak(benchmark)
        print(f"\n1. NEAR-DUPLICATE LEAK (fewshot -> holdout)")
        print(f"   identical after normalising punctuation/case: {exact_norm}")
        print(f"   >=80% word overlap with a prompt example:     {len(hits)}")
        for score, ht, ft in hits[:5]:
            print(f"     {score:.2f}  holdout: {ht[:58]}")
            print(f"           fewshot: {ft[:58]}")

        r = representativeness(benchmark)
        print(f"\n2. IS HOLDOUT REPRESENTATIVE OF THE POOL?")
        print(f"   message length  gold vs pool: "
              f"median {r['gold_len_median']:.0f} vs {r['pool_len_median']:.0f}, "
              f"mean {r['gold_len_mean']:.0f} vs {r['pool_len_mean']:.0f}, "
              f"p90 {r['gold_len_p90']:.0f} vs {r['pool_len_p90']:.0f}")
        verdict = "consistent with random sampling" if r["ids_look_random"] else "NOT random -- clustered"
        print(f"   gold id spread: KS {r['ks_stat']:.4f} vs critical {r['ks_critical']:.4f}  -> {verdict}")

        n_dup, extra = duplicate_texts_in_pool(benchmark)
        print(f"\n3. DUPLICATE MESSAGES IN POOL")
        print(f"   {n_dup} texts appear more than once ({extra} redundant copies)")
        print()


if __name__ == "__main__":
    main()
