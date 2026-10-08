"""Compare every labelling of benchmark A on one gate, with a paired test.

Earlier scores for v1 and v2 were computed when audit_benchmark_a.tsv still
described the contaminated model, so they are not comparable to anything
scored today. Everything here is re-scored against the current gate.
"""
import sys
from math import comb

sys.path.insert(0, r"C:\Users\Igor\ostrai-final\src")
from config import LABELS_ROOT
from human_gate import combined_gate
from labeller import read_labels

d = LABELS_ROOT / "benchmark_a"
CANDIDATES = {
    "v1 (5-shot)":      d / "pseudo_labels.anthropic_claude-sonnet-5.jsonl",
    "v2 (rules)":       d / "pseudo_labels.anthropic_claude-sonnet-5.rules-v2.jsonl",
    "10-shot":          d / "pseudo_labels.anthropic_claude-sonnet-5.shot10.jsonl",
    "gemini (5-shot)":  d / "pseudo_labels.google_gemini-3.1-flash-lite.jsonl",
    "gpt-5.1":          d / "pseudo_labels.openai_gpt-5.1.jsonl",
    "consensus(3)":     d / "pseudo_labels.consensus_sonnet3.jsonl",
}

gate = combined_gate("benchmark_a")
gate = gate[gate["verdict"].isin(["ok", "bad"])].reset_index(drop=True)

labels = {}
for name, path in CANDIDATES.items():
    if not path.exists():
        print(f"  (missing: {path.name})")
        continue
    labels[name] = dict(zip(*read_labels(path).values.T.tolist()[:2])) if False else \
        dict(read_labels(path).set_index("id")["intent"])


def correct(name, row):
    """A labelling is right if it keeps an approved label or changes a rejected one."""
    got = labels[name].get(row.id)
    return (got == row.assigned) if row.verdict == "ok" else (got != row.assigned)


print(f"gate: {len(gate)} judged messages "
      f"({(gate.verdict == 'ok').sum()} approved, {(gate.verdict == 'bad').sum()} rejected)\n")
scores = {}
for name in labels:
    hits = [correct(name, r) for r in gate.itertuples()]
    scores[name] = hits
    kept = sum(h for h, r in zip(hits, gate.itertuples()) if r.verdict == "ok")
    chg = sum(h for h, r in zip(hits, gate.itertuples()) if r.verdict == "bad")
    print(f"  {name:18s} {sum(hits):3d}/{len(gate)} = {sum(hits)/len(gate):6.1%}"
          f"   (kept {kept}, fixed {chg})")

print()
base = "v1 (5-shot)"
for name in labels:
    if name == base:
        continue
    w = sum(1 for a, b in zip(scores[name], scores[base]) if a and not b)
    l = sum(1 for a, b in zip(scores[name], scores[base]) if b and not a)
    n = w + l
    p = (sum(comb(n, k) for k in range(n + 1) if abs(k - n / 2) >= abs(w - n / 2)) / 2 ** n
         if n else 1.0)
    verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"  {name:18s} vs {base}: wins {w}, loses {l}, exact McNemar p = {p:.4f}  {verdict}")
