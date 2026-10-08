"""Score a model variant against the human audit, reviewing only what changed.

The audit set is 120 uniformly sampled pool messages a human judged against
one model's predictions. It read 85.8% where the leaderboard read 81.25% --
consistent, and the only measurement in this project that is.

Its weakness is resolution: at n=120 the interval is about +/-6 points, so
comparing two models as independent percentages cannot see a 2-point gain.

Paired comparison can. Where a variant predicts what the audited model
predicted, the human verdict already applies -- no review needed. Only the
disagreements are unjudged, and there are usually few. That turns "15 minutes
per variant" into "judge the handful that changed", and a paired test is far
more sensitive than two separate proportions.

Output is a bound, not a point estimate:

    lower  every unjudged disagreement is wrong
    upper  every unjudged disagreement is right

If the lower bound already beats the audited model, the variant is better
without any further review at all.

    python src/audit_compare.py benchmark_a --model-dir artifacts/benchmark_a/model_selftrain
"""

from __future__ import annotations

import argparse
import csv

import pandas as pd

from classify import load_classifier
from config import LABELS_ROOT, artifact_dir
from data import load_pool_unlabeled


def load_audit_verdicts(benchmark):
    """The human verdicts, archived under labels/ as paid-for, unrepeatable data."""
    path = LABELS_ROOT / f"audit_{benchmark}.tsv"
    if not path.exists():
        raise SystemExit(
            f"{path} missing. Save the audit verdicts there first "
            "(id, assigned, verdict, note)."
        )
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def main():
    ap = argparse.ArgumentParser(description="Score a variant against the human audit")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prior-tau", type=float, default=0.0,
                    help="label-shift correction; test.tsv is class-uniform")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    gate = load_audit_verdicts(args.benchmark)
    # Texts come from the pool, not from a saved copy of the audit sample.
    # artifacts/ is gitignored, so any file there is gone after a fresh clone --
    # pool.tsv is committed and is the same source the sample was drawn from.
    texts = load_pool_unlabeled(args.benchmark).set_index("id")["text"].to_dict()
    gate["text"] = gate["id"].map(texts)
    missing = gate[gate["text"].isna()]
    if len(missing):
        raise SystemExit(
            f"{len(missing)} audited ids are not in the pool: {list(missing['id'][:5])}"
        )

    clf = load_classifier(args.model_dir, fp16=args.fp16, prior_tau=args.prior_tau)
    order = gate["text"].tolist()
    preds = []
    for i in range(0, len(order), args.batch_size):
        preds.extend(clf.classify(order[i : i + args.batch_size]))
    gate["new"] = preds

    same = gate["new"] == gate["assigned"]
    # Verdicts carry over only where the prediction is unchanged.
    known_ok = int((same & (gate["verdict"] == "ok")).sum())
    known_bad = int((same & (gate["verdict"] == "bad")).sum())
    known_amb = int((same & (gate["verdict"] == "maybe")).sum())
    changed = gate[~same]
    n = len(gate)

    base_ok = int((gate["verdict"] == "ok").sum())

    print(f"AUDIT COMPARISON -- {args.model_dir}")
    print(f"  audited model accuracy: {base_ok}/{n} = {base_ok/n:.1%}\n")
    print(f"  predictions unchanged: {int(same.sum())}/{n}")
    print(f"    of those, human said correct {known_ok}, wrong {known_bad}, ambiguous {known_amb}")
    print(f"  predictions changed:   {len(changed)}  <- these need judging\n")

    lo = known_ok / n
    hi = (known_ok + len(changed)) / n
    print(f"  lower bound (all changes wrong):  {lo:.1%}")
    print(f"  upper bound (all changes right):  {hi:.1%}")
    if lo > base_ok / n:
        print("  -> already beats the audited model with no further review")
    elif hi < base_ok / n:
        print("  -> already worse than the audited model with no further review")
    else:
        print(f"  -> inconclusive until those {len(changed)} are judged")

    if len(changed):
        out = artifact_dir(args.benchmark) / "audit_changed.tsv"
        changed[["id", "text", "assigned", "verdict", "new"]].to_csv(
            out, sep="\t", index=False, quoting=csv.QUOTE_NONE
        )
        print(f"\n  changed predictions -> {out.name}")
        print(f"  {'id':>6s}  {'was':<34s} {'human':<6s} now")
        for r in changed.itertuples():
            print(f"  {r.id:>6s}  {r.assigned:<34s} {r.verdict:<6s} {r.new}")


if __name__ == "__main__":
    main()
