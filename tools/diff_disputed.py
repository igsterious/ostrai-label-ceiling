"""What did the hand annotation actually change, and can we undo it?

Three files, deliberately separate, so a bad annotation session is always
recoverable:

  disputed_original.tsv   the 771 rows as the 3-way vote left them, with all
                          three votes -- the BEFORE record, committed to git
  human_disputed.tsv      one line per hand decision -- the AFTER record,
                          appended as you work
  pseudo_labels.gpt3way.jsonl   never modified; apply_disputed writes a NEW
                          file, so reverting is "train on gpt3way again"

This prints the decisions side by side with what the vote said, so a session
can be inspected before a training run is spent on it -- and any single row can
be undone by deleting its line from human_disputed.tsv.

    python tools/diff_disputed.py benchmark_a
    python tools/diff_disputed.py benchmark_a --show-changed 40
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import labels_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Compare hand decisions to the vote")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--show-changed", type=int, default=15)
    args = ap.parse_args()

    d = labels_dir(args.benchmark)
    orig_path, hand_path = d / "disputed_original.tsv", d / "human_disputed.tsv"
    if not orig_path.exists():
        raise SystemExit(f"{orig_path.name} missing -- the BEFORE record is gone")
    if not hand_path.exists():
        raise SystemExit(f"{hand_path.name} missing -- nothing annotated yet")

    orig = pd.read_csv(orig_path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                       keep_default_na=False).set_index("id")
    hand = pd.read_csv(hand_path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                       keep_default_na=False)

    kept, changed, unknown = [], [], []
    for r in hand.itertuples():
        if r.id not in orig.index:
            unknown.append(r.id)
            continue
        (changed if r.intent != orig.loc[r.id, "consensus"] else kept).append(r.id)

    n = len(orig)
    print(f"\n{args.benchmark}: {n:,} disputed rows, {len(hand):,} annotated "
          f"({len(hand)/n:.0%})")
    print(f"  agreed with the vote  {len(kept):,}")
    print(f"  OVERRODE the vote     {len(changed):,}  "
          f"({len(changed)/max(1,len(hand)):.0%} of your decisions)")
    if unknown:
        print(f"  ids not in the disputed set: {len(unknown)} -- check these")

    # A rate far from ~43% is worth noticing: the vote is right ~57% of the
    # time on these rows, so overriding far less may mean deferring to it, and
    # far more may mean drifting from the test key's conventions.
    if hand.shape[0] >= 30:
        rate = len(changed) / len(hand)
        note = ("in line with the vote being ~57% right"
                if 0.30 <= rate <= 0.58 else
                "unusually low -- are you defaulting to option 1?"
                if rate < 0.30 else
                "unusually high -- worth spot-checking a few")
        print(f"  override rate {rate:.0%} -- {note}")

    if changed:
        print(f"\n  first {min(args.show_changed, len(changed))} overrides:")
        chosen = dict(zip(hand["id"], hand["intent"]))
        for i in changed[: args.show_changed]:
            row = orig.loc[i]
            print(f"\n    {row['text'][:70]}")
            print(f"      vote said : {row['consensus']}")
            print(f"      you said  : {chosen[i]}")

    print("\n  To undo one row: delete its line from human_disputed.tsv.")
    print("  To undo everything: train on pseudo_labels.gpt3way.jsonl as before.")


if __name__ == "__main__":
    main()
