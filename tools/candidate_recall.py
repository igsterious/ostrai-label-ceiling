"""Is the right answer inside the model's top-k? The ceiling on any reranker.

WHY THIS DECIDES SOMETHING. Reranking narrows 77 categories to a shortlist and
asks an LLM to choose among them. However good the chooser, it can never beat
the fraction of messages whose true category is ON the shortlist -- and when it
is absent, the reranker returns the best wrong answer with full confidence,
which is worse than a plain labelling that at least had the right option
available.

Our first rerank attempt built shortlists from TF-IDF plus whatever the
labellers proposed, and scored 82.55 against the plain labelling's 84.33. That
is consistent with a recall problem, but we never measured recall, so it is
equally consistent with the chooser simply being bad. This measures it.

The measuring stick is the 188 pool messages whose category a human confirmed
-- the only labels in the project that are both certain and drawn from the pool
rather than the curated examples. They are deliberately the CONFUSABLE ones, so
recall here is a pessimistic estimate, which is the right direction for a
go/no-go.

    python tools/candidate_recall.py benchmark_a --model <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classify import load_classifier  # noqa: E402
from training_data import human_confirmed, load_holdout  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Top-k recall of the true category")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--model", required=True, help="model dir, or comma-separated ensemble")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--fp16", action="store_true")
    # The judged rows are selected on verdict == "ok", i.e. conditioned on the
    # LABELLER having been right, and the model largely agrees with the
    # labeller -- so recall measured there reads high for a reason that has
    # nothing to do with the shortlist. Holdout is not conditioned on any
    # model's output, which is the property that matters here; it is merely
    # easier (40 chars against test's 54). Neither is unbiased AND hard, so
    # read both and treat the truth as bracketed between them.
    ap.add_argument("--source", choices=["judged", "holdout"], default="holdout")
    args = ap.parse_args()

    if args.source == "judged":
        rows = human_confirmed(args.benchmark)
    else:
        rows = load_holdout(args.benchmark)[["text", "intent"]]
    if rows.empty:
        raise SystemExit(f"no {args.source} rows -- nothing to measure against")

    clf = load_classifier(args.model, fp16=args.fp16)
    intents = clf.intents
    texts = rows["text"].tolist()
    truth = rows["intent"].tolist()

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            probs.append(clf.predict_proba(texts[i : i + args.batch_size]).float().cpu())
    probs = torch.cat(probs)

    order = probs.argsort(dim=-1, descending=True)
    ranks = []
    for row, t in zip(order.tolist(), truth):
        j = intents.index(t)
        ranks.append(row.index(j) + 1)  # 1-based: 1 means the model already had it

    n = len(ranks)
    print(f"{args.benchmark}: {n} {args.source} messages, {args.model}")
    if args.source == "judged":
        print("  WARNING: selected on verdict=='ok', i.e. rows where the labeller")
        print("  was right. The model agrees with the labeller, so top-1 here is")
        print("  inflated and top-k recall with it. Compare against --source holdout.")
    print()
    print(f"  {'shortlist':>10s}  {'recall':>8s}   {'ceiling this puts on any reranker'}")
    for k in (1, 3, 5, 10, 15, 20):
        hit = sum(1 for r in ranks if r <= k)
        bar = "#" * round(40 * hit / n)
        print(f"  top-{k:<6d}  {hit/n:>7.1%}   {bar}")

    print(f"\n  median rank of the true category: {sorted(ranks)[n // 2]}")
    missed = [(t, r) for t, r in zip(truth, ranks) if r > 10]
    if missed:
        print(f"  outside top-10 ({len(missed)}): "
              + ", ".join(f"{t}@{r}" for t, r in sorted(missed, key=lambda x: -x[1])[:8]))

    k5 = sum(1 for r in ranks if r <= 5) / n
    print("\n  READ THIS AS: a reranker choosing from the model's top-5 cannot")
    print(f"  exceed {k5:.1%} on messages like these, however good the chooser.")
    print(f"  The model alone is already right {sum(1 for r in ranks if r == 1)/n:.1%} of the time here,")
    print("  so the headroom is the gap between those two numbers.")


if __name__ == "__main__":
    main()
