"""The human-verified gate: 100 hand-judged pool messages for benchmark A.

Why this exists. The automated gate scored the labeller against holdout and
said 98.4%; hand review of pool messages said 85%. The difference is that
holdout is CURATED -- prototypical examples capped at 60 characters -- while
the pool is raw traffic, 30% of which is longer. test.tsv looks like the pool
(24.4% over 60 chars), not like holdout, so holdout systematically flatters
anything measured on it.

These 100 messages are the only representative ground truth in the project:
sampled from the pool, judged by a human, spanning the real length range.
Small, but honest. Any change to the labelling has to be judged here.

Scoring a candidate labelling against human verdicts, where for rejected items
we know the assigned label was WRONG but not what the right one is:

    approved item (85)  -> candidate should KEEP the label the human approved
    rejected item (15)  -> candidate should CHANGE it to something else

Both are checkable without knowing the true label for rejected items. A
labelling that keeps every approved label and changes every rejected one
scores 100/100; the labelling we have today scores 85/100 by construction,
since it IS what was judged.

    python src/human_gate.py                    # build the file, show the split
    python src/human_gate.py --score <jsonl>    # score a candidate labelling
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import pandas as pd

from config import LABELS_ROOT, labels_dir
from data import load_pool_unlabeled
from labeller import read_labels

ROUND_DIRS = ("review_round1", "review_round3_calibration")


def build(benchmark="benchmark_a"):
    """Collect every human verdict on this benchmark's pool messages."""
    rows = []
    for sub in ROUND_DIRS:
        d = LABELS_ROOT / sub
        if not d.exists():
            continue
        for path in d.rglob("*.json"):
            try:
                r = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not r.get("verdict"):
                continue
            # round 1 rows are keyed by benchmark; round 3 rows carry
            # benchmark "calibration" and prefix pool ids with p / gold with g
            item_id = str(r.get("item_id", ""))
            if r.get("benchmark") == benchmark:
                pool_id = item_id
            elif r.get("benchmark") == "calibration" and r.get("stratum") == "pseudo":
                pool_id = item_id.lstrip("p")
            else:
                continue  # gold calibration items are not pool messages
            rows.append({
                "id": pool_id,
                "assigned": r.get("assigned", ""),
                "verdict": r["verdict"],
                "note": (r.get("note") or "").replace("\t", " ").replace("\n", " "),
            })

    df = pd.DataFrame(rows).drop_duplicates(subset="id", keep="last")
    pool = load_pool_unlabeled(benchmark)[["id", "text"]]
    df = df.merge(pool, on="id", how="left")
    df["text"] = df["text"].fillna("").str.replace(r"[\t\r\n]", " ", regex=True)
    return df.sort_values("id").reset_index(drop=True)


def gate_path(benchmark="benchmark_a"):
    return labels_dir(benchmark) / "human_verdicts.tsv"


def load_gate(benchmark="benchmark_a"):
    path = gate_path(benchmark)
    if not path.exists():
        raise SystemExit(f"{path} missing -- run: python src/human_gate.py")
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def load_audit_gate(benchmark="benchmark_a"):
    """The 120-message uniform audit, reused as ground truth about messages.

    These verdicts were collected on the trained MODEL's predictions, not the
    labeller's, but a verdict is a statement about the message and the category
    -- "for this message, that category is right/wrong" -- and it holds no
    matter who proposed the category. So they score a candidate labelling too.

    Disjoint from load_gate() by construction: make_audit_set.py excludes
    human_judged_ids(), so no message is counted twice.
    """
    path = LABELS_ROOT / f"audit_{benchmark}.tsv"
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def combined_gate(benchmark="benchmark_a"):
    """Both judged sets, 220 messages for benchmark A against 100."""
    parts = [load_gate(benchmark)[["id", "assigned", "verdict"]]]
    audit = load_audit_gate(benchmark)
    if audit is not None:
        parts.append(audit[["id", "assigned", "verdict"]])
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)


def score(candidate_labels, benchmark="benchmark_a", gate=None):
    """Score a candidate labelling (DataFrame of id, intent) against the gate."""
    if gate is None:
        gate = load_gate(benchmark)
    m = gate.merge(candidate_labels.rename(columns={"intent": "candidate"}), on="id", how="left")
    m["candidate"] = m["candidate"].fillna("<MISSING>")

    approved = m[m["verdict"] == "ok"]
    rejected = m[m["verdict"] == "bad"]
    ambiguous = m[m["verdict"] == "maybe"]

    kept = int((approved["candidate"] == approved["assigned"]).sum())
    changed = int((rejected["candidate"] != rejected["assigned"]).sum())
    scored = len(approved) + len(rejected)
    return {
        "n_approved": len(approved), "kept": kept,
        "n_rejected": len(rejected), "changed": changed,
        "n_ambiguous": len(ambiguous),
        "score": (kept + changed) / scored if scored else None,
        "broke": approved[approved["candidate"] != approved["assigned"]],
        "fixed": rejected[rejected["candidate"] != rejected["assigned"]],
        "still_wrong": rejected[rejected["candidate"] == rejected["assigned"]],
    }


def main():
    ap = argparse.ArgumentParser(description="Human-verified gate for benchmark A")
    ap.add_argument("--benchmark", default="benchmark_a")
    ap.add_argument("--score", help="a pseudo-label JSONL to score against the gate")
    args = ap.parse_args()

    if not args.score:
        df = build(args.benchmark)
        out = gate_path(args.benchmark)
        df.to_csv(out, sep="\t", index=False, quoting=csv.QUOTE_NONE)
        counts = df["verdict"].value_counts()
        print(f"{args.benchmark}: {len(df)} human-judged pool messages -> {out}")
        for v in ("ok", "bad", "maybe"):
            print(f"  {v:6s} {int(counts.get(v, 0)):3d}")
        missing = int((df['text'] == '').sum())
        if missing:
            print(f"  WARNING: {missing} ids not found in the pool")
        return

    cand = read_labels(pathlib.Path(args.score))
    gates = [("labeller-judged", load_gate(args.benchmark))]
    audit = load_audit_gate(args.benchmark)
    if audit is not None:
        gates.append(("model-audit", audit))
        gates.append(("COMBINED", combined_gate(args.benchmark)))

    for name, g in gates:
        r = score(cand, args.benchmark, gate=g)
        n = r["n_approved"] + r["n_rejected"]
        print(f"{name:16s} {n:4d} judged   kept {r['kept']:3d}/{r['n_approved']}"
              f"   changed {r['changed']:2d}/{r['n_rejected']}"
              f"   SCORE {r['score']:.1%}")

    result = score(cand, args.benchmark, gate=gates[-1][1])
    if len(result["broke"]):
        print(f"\n  regressions -- changed a label the human had approved ({len(result['broke'])}):")
        for r in result["broke"].itertuples():
            print(f"    id={r.id}  {r.assigned}  ->  {r.candidate}")
    if len(result["fixed"]):
        print(f"\n  fixes -- changed a label the human rejected ({len(result['fixed'])}):")
        for r in result["fixed"].itertuples():
            print(f"    id={r.id}  {r.assigned}  ->  {r.candidate}")
    if len(result["still_wrong"]):
        print(f"\n  unfixed -- kept a label the human rejected ({len(result['still_wrong'])}):")
        for r in result["still_wrong"].itertuples():
            print(f"    id={r.id}  {r.assigned}")


if __name__ == "__main__":
    main()
