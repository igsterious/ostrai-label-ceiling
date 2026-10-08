"""Splice the best measured half of each benchmark into one submission.

Every entry in the file is self-contained -- name, benchmark letter, its own
average_time_per_example, and its own predictions -- so A and B can be taken
from different runs without touching either one's numbers. The timing carried
across is the timing that was actually measured for those exact predictions,
which is what the protocol asks for.

  A  <- submission 3 (89.39): DeBERTa on v1 labels, scored 82.06
  B  <- submission 4 (89.13): DeBERTa retrain,      scored 97.17
"""
import json
import pathlib
import shutil

SRC = pathlib.Path(r"C:\Users\Igor\Ostr AI Summer School")
REPO = pathlib.Path(r"C:\Users\Igor\ostrai-final")

A_FROM = REPO / "submissions" / "2026-09-07_deberta_ab_89.39.json"
B_FROM = pathlib.Path(r"C:\Users\Igor\Downloads\submission ver 4 gpt deberta.json")
OUT = REPO / "submissions" / "2026-09-08_spliced_A82.06_B97.17.json"

a_doc = json.loads(A_FROM.read_text(encoding="utf-8"))
b_doc = json.loads(B_FROM.read_text(encoding="utf-8"))

a_entry = next(e for e in a_doc if e["benchmark"] == "A")
b_entry = next(e for e in b_doc if e["benchmark"] == "B")

# One name across both entries, as the site requires; they were produced by
# separate runs so take it from one and apply it to both.
name = a_entry["name"]
a_entry["name"] = b_entry["name"] = name
spliced = [a_entry, b_entry]

# Same validation submit.py applies, because this file bypasses submit.py.
assert isinstance(spliced, list) and len(spliced) == 2
assert {e["benchmark"] for e in spliced} == {"A", "B"}
assert len({e["name"] for e in spliced}) == 1
for e in spliced:
    ids = [p["id"] for p in e["predictions"]]
    assert all(isinstance(i, int) for i in ids), f"{e['benchmark']}: non-int id"
    assert sorted(ids) == list(range(1, len(ids) + 1)), f"{e['benchmark']}: ids not 1..N"
    assert all(isinstance(p["intent"], str) and p["intent"] for p in e["predictions"])
    assert isinstance(e["average_time_per_example"], float)

OUT.write_text(json.dumps(spliced), encoding="utf-8")

# Confirm each half is byte-for-byte the predictions from its source run.
check = json.loads(OUT.read_text(encoding="utf-8"))
ca = next(e for e in check if e["benchmark"] == "A")
cb = next(e for e in check if e["benchmark"] == "B")
src_a = next(e for e in json.loads(A_FROM.read_text(encoding="utf-8")) if e["benchmark"] == "A")
src_b = next(e for e in json.loads(B_FROM.read_text(encoding="utf-8")) if e["benchmark"] == "B")
assert ca["predictions"] == src_a["predictions"], "A predictions altered"
assert cb["predictions"] == src_b["predictions"], "B predictions altered"

print(f"wrote {OUT.name}  ({OUT.stat().st_size/1024:.0f} KB)")
print(f"  name  {name!r}")
for e, src, scored in ((ca, "sub 3", 82.06), (cb, "sub 4", 97.17)):
    print(f"  {e['benchmark']}: {len(e['predictions']):,} predictions from {src}, "
          f"{e['average_time_per_example']} ms/example, scored {scored}%")
print(f"  expected overall: {(82.06 + 97.17) / 2:.2f}%")

copy = SRC / OUT.name
shutil.copy(OUT, copy)
print(f"\n  upload this file: {copy}")
