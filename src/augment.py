"""Generate training examples for the classes the pool barely covers.

WHY. Benchmark A's pool is 13.7:1 imbalanced -- contactless_not_working has 18
rows against transfer_timing's 249, median 115. The rare classes are exactly
the ones the model under-predicts (12 against a true 40 on test). Class weights
or a prior correction re-weight the evidence that exists; they cannot create
evidence that does not.

WHY IT BEATS EVERYTHING ELSE WE TRIED. Three days of experiments all failed
against the same ceiling: the pseudo-labels are ~90% correct and the errors are
systematic, so every reshuffling of them inherits the same bias. Generated
examples are CLEAN BY CONSTRUCTION -- we choose the category and ask for text
that belongs to it, so there is no labelling step to be wrong. It is the only
route that adds information rather than redistributing it.

WHY THE NEIGHBOURS ARE IN THE PROMPT. Every audited error is a confusion
between adjacent categories, never a wild miss. Generic examples of a rare
class would not touch that. So each prompt names the categories this one is
actually confused with -- computed from where our three labelling runs
disagree -- and asks for messages that clearly belong to the target and clearly
do NOT belong to those. The examples are generated where the boundary is.

RULES. Sanctioned explicitly by the organiser: "mohls taky treba pouzit silne
LLM, kteremu bys dal priklady a nechal jej vygenerovat data v kategoriich, kde
data chybi." Offline, ahead of time, like all our LLM use; the classifier still
runs locally with no network.

    python src/augment.py benchmark_a --dry-run
    python src/augment.py benchmark_a --floor 100
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from config import labels_dir
from data import load_labeled, write_tsv
from labeller import call_openrouter, get_api_key, read_labels

RUNS = (
    "pseudo_labels.openai_gpt-5.1.jsonl",
    "pseudo_labels.openai_gpt-5.1.rules-v2.jsonl",
    "pseudo_labels.openai_gpt-5.1.b10.jsonl",
)


def confusable(benchmark, top=3):
    """For each class, the classes our labellings actually confuse it with."""
    d = labels_dir(benchmark)
    runs = [read_labels(d / f).set_index("id")["intent"] for f in RUNS
            if (d / f).exists()]
    pairs = defaultdict(Counter)
    if len(runs) < 2:
        return {}
    ids = runs[0].index
    for r in runs[1:]:
        ids = ids.intersection(r.index)
    for i in ids:
        v = sorted({r[i] for r in runs})
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                pairs[v[a]][v[b]] += 1
                pairs[v[b]][v[a]] += 1
    return {k: [c for c, _ in v.most_common(top)] for k, v in pairs.items()}


def build_prompt(intent, examples, neighbours, n):
    blocks = "\n".join(f"  - {t}" for t in examples)
    near = ""
    if neighbours:
        near = "\n\nThese categories are easily confused with it. Your messages\n" \
               "must NOT fit any of them:\n"
        for name, ex in neighbours.items():
            near += f"\n### {name}\n" + "\n".join(f"  - {t}" for t in ex)
    return f"""You write realistic customer messages for a banking app.

Write {n} NEW messages that a customer would send, all belonging to the
category "{intent}".

Real examples of this category:
{blocks}{near}

RULES:
- Return exactly {n} lines, one message per line, nothing else. No numbering,
  no quotes, no commentary.
- Vary the wording, length, politeness and phrasing. Some terse, some polite,
  some with typos or missing punctuation, as real customers write.
- Every message must be unmistakably "{intent}" and not the neighbouring
  categories above.
- Do not repeat the examples."""


CHUNK = 20      # asking for more than this in one call, the model stops early


def _one_call(intent, examples, neighbours, n, model, api_key):
    messages = [{"role": "system",
                 "content": build_prompt(intent, examples, neighbours, n)},
                {"role": "user", "content": f"Write the {n} messages now."}]
    resp = call_openrouter(messages, model, api_key)
    if resp is None:
        return [], 0.0
    lines = []
    for line in (resp.get("content") or "").splitlines():
        line = line.strip().strip('"').strip()
        line = re.sub(r"^\s*[-*\d]+[.)]?\s*", "", line)   # strip any numbering
        # tabs and newlines would break the TSV; the data files contain neither
        if 8 <= len(line) <= 200 and "\t" not in line:
            lines.append(line)
    return lines, resp.get("cost", 0.0) or 0.0


def generate(intent, examples, neighbours, n, model, api_key):
    """Ask in chunks. Requesting 60 in one call returned as few as 2 -- the
    model stops early on long enumerations -- so several short calls yield far
    more, and the duplicate filter downstream handles the overlap between them.
    """
    out, cost = [], 0.0
    while len(out) < n:
        want = min(CHUNK, n - len(out))
        lines, c = _one_call(intent, examples, neighbours, want, model, api_key)
        cost += c
        if not lines:                 # a failed or empty call: stop, do not spin
            break
        out.extend(lines)
        if len(lines) < want // 2:    # the model is running dry on this class
            break
    return out[:n], cost


def main():
    ap = argparse.ArgumentParser(description="Generate examples for rare classes")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--floor", type=int, default=100,
                    help="bring every class up to at least this many pool rows")
    ap.add_argument("--cap", type=int, default=60,
                    help="most examples to generate for any one class")
    ap.add_argument("--labels-file", default="pseudo_labels.gpt3way.jsonl")
    ap.add_argument("--model", default="openai/gpt-5.1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = labels_dir(args.benchmark)
    counts = Counter(read_labels(d / args.labels_file)["intent"])
    lab = load_labeled(args.benchmark)
    examples = {i: list(g["text"]) for i, g in lab.groupby("intent")}
    near = confusable(args.benchmark)

    need = {k: min(args.cap, args.floor - counts.get(k, 0))
            for k in examples if counts.get(k, 0) < args.floor}
    total = sum(need.values())
    print(f"{args.benchmark}: {len(need)} classes below {args.floor}, "
          f"{total:,} examples to generate")
    for k, v in sorted(need.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {counts.get(k,0):4d} -> +{v:<4d} {k}   "
              f"(vs {', '.join(near.get(k, [])[:2]) or 'no confusions recorded'})")
    if args.dry_run:
        print("\n  DRY RUN -- no API calls made")
        return

    api_key = get_api_key()
    out = d / "augmented.tsv"
    lock = threading.Lock()
    rows, totals = [], {"cost": 0.0, "done": 0}
    started = time.time()

    def work(item):
        intent, n = item
        nb = {c: examples.get(c, [])[:4] for c in near.get(intent, [])}
        lines, cost = generate(intent, examples[intent], nb, n, args.model, api_key)
        with lock:
            totals["cost"] += cost
            totals["done"] += 1
            rows.extend((t, intent) for t in lines[:n])
            print(f"  [{totals['done']}/{len(need)}] {intent}: {len(lines)} "
                  f"generated  ${totals['cost']:.3f}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, need.items()))

    # A generated line identical to a real one adds nothing and risks
    # duplicating a message that is also in test.
    seen = set(lab["text"])
    uniq, dropped = [], 0
    for t, i in rows:
        if t in seen:
            dropped += 1
            continue
        seen.add(t)
        uniq.append({"text": t, "intent": i})

    import pandas as pd
    write_tsv(pd.DataFrame(uniq), out)
    print(f"\ndone in {(time.time()-started)/60:.1f} min, cost ${totals['cost']:.3f}")
    print(f"  wrote {out.name}: {len(uniq):,} rows ({dropped} duplicates dropped)")
    print(f"  train with: --augment-file labels/{args.benchmark}/{out.name}")


if __name__ == "__main__":
    main()
