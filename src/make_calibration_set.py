"""Build a blind calibration round: gold labels mixed with pseudo-labels.

The first review round put benchmark A's pseudo-labels at 88.8%, against the
automated gate's 98.4%. Two explanations fit that gap equally well:

  (a) the pseudo-labels really are ~11% wrong, or
  (b) the reviewer applies a stricter standard than the dataset's own
      labelling convention, and would reject the dataset's labels at a
      similar rate.

They call for opposite responses -- (a) means re-label with a stronger model,
(b) means do nothing -- so guessing between them wastes either money or the
labels. This distinguishes them.

Half the items carry the DATASET's gold label (from holdout, known-correct by
definition). Half carry a pseudo-label on a fresh pool message. Both are
presented identically and shuffled, so the reviewer cannot tell them apart.

  reject gold at ~the same rate as pseudo  -> (b). The reviewer's standard
                                              differs from the dataset's; the
                                              labeller is fine.
  accept gold, reject pseudo               -> (a). The error is real.

Note the gold items are drawn from holdout, whose labels the reviewer has
never seen and the labeller was never shown -- so nothing here contaminates
the validation set. It is used as a source of TRUTH, not of training signal.

    python src/make_calibration_set.py
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from config import artifact_dir
from data import load_pool_unlabeled
from labeller import read_labels
from training_data import find_pseudo_label_file, load_fewshot, load_holdout

SEED = 20260906


def build(benchmark, n_gold, n_pseudo):
    fewshot = load_fewshot(benchmark)
    examples = {i: list(r["text"]) for i, r in fewshot.groupby("intent")}

    # GOLD: holdout messages shown with the dataset's own label.
    gold = load_holdout(benchmark).sample(n_gold, random_state=SEED)
    gold_items = [
        {"id": f"g{r.id}", "text": r.text, "assigned": r.intent,
         "source": "gold", "examples": examples.get(r.intent, [])}
        for r in gold.itertuples()
    ]

    # PSEUDO: fresh pool messages, excluding anything reviewed in round 1.
    seen = set()
    prev = artifact_dir(benchmark) / "review_set.json"
    if prev.exists():
        seen = {str(i["id"]) for i in json.loads(prev.read_text(encoding="utf-8"))["items"]}

    labels = read_labels(find_pseudo_label_file(benchmark))
    pool = load_pool_unlabeled(benchmark).merge(labels, on="id")
    pool = pool[~pool["id"].astype(str).isin(seen)]
    pick = pool.sample(n_pseudo, random_state=SEED)
    pseudo_items = [
        {"id": f"p{r.id}", "text": r.text, "assigned": r.intent,
         "source": "pseudo", "examples": examples.get(r.intent, [])}
        for r in pick.itertuples()
    ]

    items = pd.DataFrame(gold_items + pseudo_items).sample(
        frac=1.0, random_state=SEED
    ).to_dict("records")
    return {"benchmark": benchmark, "items": items,
            "n_gold": len(gold_items), "n_pseudo": len(pseudo_items)}


def main():
    ap = argparse.ArgumentParser(description="Blind gold-vs-pseudo calibration round")
    ap.add_argument("--benchmark", default="benchmark_a")
    ap.add_argument("--n-gold", type=int, default=20)
    ap.add_argument("--n-pseudo", type=int, default=20)
    args = ap.parse_args()

    payload = build(args.benchmark, args.n_gold, args.n_pseudo)
    out = artifact_dir(args.benchmark) / "calibration_set.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{args.benchmark}: {len(payload['items'])} items "
          f"({payload['n_gold']} gold, {payload['n_pseudo']} pseudo) -> {out}")
    print("  the two are shuffled and visually identical in the review UI")


if __name__ == "__main__":
    main()
