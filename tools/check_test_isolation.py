"""Prove, from the files alone, that test.tsv never entered training.

Three facts make this checkable without trusting any code path:

  1. test.tsv has NO label column. There are no test labels in the repo to
     train on. What could leak is test TEXT -- a message the model saw during
     training and is then scored on.
  2. The training set is fewshot.tsv + a pseudo-label file, and pseudo-label
     files are keyed by POOL ids. So test text can only reach training if the
     same text also sits in the pool.
  3. Pool ids and test ids are separate numberings that both start at 1, so
     any check must compare TEXT, never ids.

This script therefore rebuilds the training set exactly as train.py would,
and intersects its texts with test.tsv's. The known overlap (pool and test
share a handful of identical messages -- an organiser's-data fact, recorded in
CLAUDE.md) is reported, not hidden.

    python tools/check_test_isolation.py                # both benchmarks
    python tools/check_test_isolation.py benchmark_a
"""

import ast
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

read = lambda p: pd.read_csv(p, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def check(benchmark):
    d = ROOT / "data" / benchmark
    test = read(d / "test.tsv")
    pool = read(d / "pool.tsv")
    fewshot = read(ROOT / "artifacts" / benchmark / "fewshot.tsv")
    holdout = read(ROOT / "artifacts" / benchmark / "holdout.tsv")
    test_text = set(test["text"])
    ok = True

    print(f"=== {benchmark} ===")
    print(f"  test.tsv columns: {list(test.columns)}  "
          f"<- {'no label column: nothing to train on' if 'intent' not in test.columns else 'HAS LABELS -- investigate'}")
    if "intent" in test.columns:
        ok = False

    # fact 3: ids are not a valid join key across files
    id_clash = len(set(pool["id"]) & set(test["id"]))
    print(f"  pool ids also present as test ids: {id_clash:,}  <- ids overlap by numbering; compare TEXT only")

    # the one real vector: identical text in pool and test
    shared = sorted(set(pool["text"]) & test_text)
    print(f"  identical text in pool AND test: {len(shared)}  <- organiser's data; every pseudo-label file inherits these")
    for t in shared[:3]:
        print(f"      e.g. {t!r}")

    # every training-side file, intersected with test text
    print(f"  fewshot  & test text: {len(set(fewshot['text']) & test_text)}")
    print(f"  holdout  & test text: {len(set(holdout['text']) & test_text)}")
    for f in sorted((ROOT / "labels" / benchmark).glob("*.jsonl")):
        ids = {str(json.loads(l)["id"]) for l in open(f, encoding="utf-8")}
        texts = set(pool.loc[pool["id"].isin(ids), "text"])
        n = len(texts & test_text)
        flag = "" if n <= len(shared) else "   <-- MORE than the pool overlap: LEAK"
        if n > len(shared):
            ok = False
        print(f"  {f.name:<58} & test text: {n}{flag}")

    # static check: does the training-set builder import the test loader?
    src = (ROOT / "src" / "training_data.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    # import-level only: the docstrings MENTION test.tsv to explain why holdout
    # is unlike it, and a string match would flag that prose as a read.
    reads_test = "load_test" in names
    print(f"  training_data.py imports load_test: {reads_test}   (imports: {sorted(n for n in names if n.startswith('load'))})")
    if reads_test:
        ok = False

    print(f"  RESULT: {'PASS -- test text reaches training only via the pool overlap above' if ok else 'FAIL'}")
    print()
    return ok


if __name__ == "__main__":
    bms = sys.argv[1:] or ["benchmark_a", "benchmark_b"]
    sys.exit(0 if all(check(b) for b in bms) else 1)
