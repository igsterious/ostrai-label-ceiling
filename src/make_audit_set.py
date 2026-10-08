"""Sample the MODEL's own predictions on fresh pool messages, for human audit.

Why this exists. Every accuracy number in this project so far is biased:

  holdout          curated examples capped at 60 chars; test.tsv is 24% longer
  human-verified   built by keeping rows where the human judged the LABELLER
                   correct -- conditioned on labeller success, which correlates
                   with classifier success. It read 92.9%; the leaderboard
                   said 81.25%.

Both flatter. Neither can tell you whether a change helped, and the brief warns
that tuning against the visible leaderboard overfits, because the final ranking
uses a hidden split.

This builds the missing instrument: a uniform random sample of pool messages,
each shown with the MODEL's prediction, for a human to accept or reject.
Nothing conditions the sample on anything, so accepted/total is an unbiased
estimate of model accuracy on pool-like data -- and test.tsv is pool-like.

Ask for the correct intent on rejections and the result becomes a reusable
labelled validation set: accepted rows are labelled by the prediction, rejected
rows by the correction. Roughly 15% need typing, so ~20 of 120.

Runs in Colab (needs the trained model). Writes artifacts/<benchmark>/audit_set.json.

    python src/make_audit_set.py benchmark_a --n 120
"""

from __future__ import annotations

import argparse
import json

from classify import load_classifier
from config import artifact_dir, labels_dir
from data import load_intents, load_pool_unlabeled
from training_data import human_judged_ids, load_fewshot

SEED = 20260907


def main():
    ap = argparse.ArgumentParser(description="Sample model predictions for human audit")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--model-dir", help="default artifacts/<benchmark>/model")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    model_dir = args.model_dir or (artifact_dir(args.benchmark) / "model")

    # Uniform random over the pool, minus anything a human has already judged.
    # No stratification, no filtering on confidence or length -- any of those
    # would reintroduce exactly the conditioning that broke the last estimate.
    pool = load_pool_unlabeled(args.benchmark)
    already = human_judged_ids(args.benchmark)
    pool = pool[~pool["id"].isin(already)]
    sample = pool.sample(min(args.n, len(pool)), random_state=SEED).reset_index(drop=True)

    clf = load_classifier(model_dir, fp16=args.fp16)
    texts = sample["text"].tolist()
    preds = []
    for i in range(0, len(texts), args.batch_size):
        preds.extend(clf.classify(texts[i : i + args.batch_size]))

    fewshot = load_fewshot(args.benchmark)
    examples = {i: list(r["text"]) for i, r in fewshot.groupby("intent")}

    payload = {
        "benchmark": args.benchmark,
        "model_dir": str(model_dir),
        "intents": load_intents(args.benchmark),
        "items": [
            {"id": r.id, "text": r.text, "assigned": p,
             "stratum": "audit", "examples": examples.get(p, [])}
            for r, p in zip(sample.itertuples(), preds)
        ],
    }
    out = artifact_dir(args.benchmark) / "audit_set.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{args.benchmark}: {len(payload['items'])} predictions sampled uniformly "
          f"from {len(pool):,} unjudged pool messages")
    print(f"  model: {model_dir}  ({clf.device}{', fp16' if args.fp16 else ''})")
    print(f"  excluded {len(already)} messages already judged by a human")
    print(f"  wrote {out}")
    print("\n  Download this file and share it, so the audit round can be published.")
    print("  Accepted/total will be the first unbiased read on model accuracy.")


if __name__ == "__main__":
    main()
