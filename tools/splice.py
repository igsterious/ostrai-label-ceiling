"""Take benchmark A from one submission file and benchmark B from another.

Each entry in a submission is self-contained -- name, benchmark letter, its own
average_time_per_example, its own predictions -- so the two halves can come from
different runs without either one being recomputed or retimed. That is what lets
us test a new A model while keeping the B predictions that scored 97.17, which is
currently the best B on the board.

    python tools/splice.py --a new_sub.json --b submissions/2026-09-08_priortau_89.65.json \
                           --out submissions/candidate.json

Validation here is the same as submit.py's, because this path bypasses submit.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def entry(path, letter):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return next(e for e in doc if e["benchmark"] == letter)


def main():
    ap = argparse.ArgumentParser(description="Splice A from one file, B from another")
    ap.add_argument("--a", required=True, help="submission JSON to take benchmark A from")
    ap.add_argument("--b", required=True, help="submission JSON to take benchmark B from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", help="override the leaderboard name on both entries")
    args = ap.parse_args()

    a, b = entry(args.a, "A"), entry(args.b, "B")

    # One name across both entries, as the site requires. The two halves came
    # from separate runs, so pick one rather than trusting them to match.
    name = args.name or a["name"]
    a["name"] = b["name"] = name
    spliced = [a, b]

    assert {e["benchmark"] for e in spliced} == {"A", "B"}
    assert len({e["name"] for e in spliced}) == 1
    for e in spliced:
        ids = [p["id"] for p in e["predictions"]]
        assert all(isinstance(i, int) for i in ids), f"{e['benchmark']}: non-int id"
        assert sorted(ids) == list(range(1, len(ids) + 1)), f"{e['benchmark']}: ids not 1..N"
        assert all(isinstance(p["intent"], str) and p["intent"] for p in e["predictions"])
        assert isinstance(e["average_time_per_example"], float)

    out = Path(args.out)
    out.write_text(json.dumps(spliced), encoding="utf-8")

    # Re-read, so what is reported describes the FILE, not what we meant to write.
    check = json.loads(out.read_text(encoding="utf-8"))
    assert next(e for e in check if e["benchmark"] == "A")["predictions"] == a["predictions"]
    assert next(e for e in check if e["benchmark"] == "B")["predictions"] == b["predictions"]

    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    print(f"  name  {name!r}")
    print(f"  A  {len(a['predictions']):,} predictions from {Path(args.a).name}, "
          f"{a['average_time_per_example']} ms/example")
    print(f"  B  {len(b['predictions']):,} predictions from {Path(args.b).name}, "
          f"{b['average_time_per_example']} ms/example")


if __name__ == "__main__":
    main()
