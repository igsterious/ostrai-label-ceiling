"""Retrieve-then-rerank labelling: narrow to a few candidates, then decide.

WHY. The main labeller does 77-way classification in one prompt, with 5 examples
per class and 50 messages per call. Every error a human has audited is a
confusion between two NEIGHBOURING categories -- pending_top_up vs top_up_failed,
why_verify_identity vs verify_my_identity -- never a wild miss. That is the
signature of a decision made with too little room per candidate, not of a model
that misunderstands the message.

So: shrink the question. Retrieve a handful of plausible categories, show ALL
ten examples of each, and ask for one message at a time. A 10-way choice with
7x the context per candidate is a far easier task than a 77-way one, and the
prompt is SMALLER because 67 irrelevant categories are gone.

Applied only where it can pay. Across our four labellings of benchmark A, 85.2%
of pool rows get the same label from all four; those are left alone. The 14.8%
that are disputed are where the labellers are visibly unsure, and where hand
review has repeatedly found the errors.

CANDIDATE RECALL is the thing that can silently ruin this: if the true category
is not among the candidates, reranking cannot find it and confidently returns
the best wrong answer. So candidates are the union of what every labeller
proposed (they disagree, but one of them is usually right) topped up by lexical
retrieval over the labelled examples -- never retrieval alone.

Retrieval is TF-IDF in numpy rather than a sentence encoder: labelling runs on
the local machine, which has no torch, and this only has to get the right
category into a shortlist of ten.

    python src/rerank_labeller.py benchmark_a --dry-run
    python src/rerank_labeller.py benchmark_a --model openai/gpt-5.1
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from collections import Counter

import numpy as np
import pandas as pd

from config import labels_dir
from data import load_intents, load_labeled, load_pool_unlabeled
from labeller import call_openrouter, get_api_key, model_slug, read_labels

TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(s):
    return TOKEN.findall(s.lower())


class IntentRetriever:
    """TF-IDF over intents, each represented by its labelled examples.

    Only used to top up a candidate list that already contains every label a
    labeller proposed, so it needs recall inside ten, not precision at one.
    """

    def __init__(self, benchmark):
        self.intents = load_intents(benchmark)
        lab = load_labeled(benchmark)
        docs = {i: [] for i in self.intents}
        for r in lab.itertuples():
            docs[r.intent].extend(tokenize(r.text))
        self.examples = {i: list(g["text"]) for i, g in lab.groupby("intent")}

        vocab = sorted({t for toks in docs.values() for t in toks})
        self.vocab = {t: j for j, t in enumerate(vocab)}
        M = np.zeros((len(self.intents), len(vocab)), dtype=np.float32)
        for r, i in enumerate(self.intents):
            for t, c in Counter(docs[i]).items():
                M[r, self.vocab[t]] = 1.0 + np.log(c)
        df = (M > 0).sum(axis=0)
        self.idf = np.log((1 + len(self.intents)) / (1 + df)) + 1.0
        M *= self.idf
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        self.M = M / np.maximum(norms, 1e-9)

    def top(self, text, k):
        v = np.zeros(len(self.vocab), dtype=np.float32)
        for t, c in Counter(tokenize(text)).items():
            j = self.vocab.get(t)
            if j is not None:
                v[j] = (1.0 + np.log(c)) * self.idf[j]
        n = np.linalg.norm(v)
        if n < 1e-9:
            return []
        sims = self.M @ (v / n)
        return [self.intents[i] for i in np.argsort(-sims)[:k]]


def build_prompt(text, candidates, examples):
    blocks = []
    for c in candidates:
        ex = "\n".join(f"  - {t}" for t in examples[c])
        blocks.append(f"### {c}\n{ex}")
    listing = "\n".join(blocks)
    return f"""You assign ONE intent category to a customer message.

Exactly one of the {len(candidates)} categories below is correct. Examples of each:

{listing}

RULES:
- Answer with one category name from the list above, character for character.
- No prose, no explanation, no punctuation. The category name and nothing else.
- These categories are deliberately similar. Decide on the customer's SITUATION
  -- what actually happened to them, and what they are asking -- not on shared
  vocabulary. Whether something already happened, failed, or has not happened
  yet distinguishes several of them."""


def label_one(text, candidates, examples, model, api_key):
    """Returns (intent or None, cost). None means the answer was unusable."""
    messages = [
        {"role": "system", "content": build_prompt(text, candidates, examples)},
        {"role": "user", "content": f"MESSAGE: {text}"},
    ]
    resp = call_openrouter(messages, model, api_key)
    if resp is None:
        return None, 0.0
    cost = resp.get("cost", 0.0) or 0.0
    answer = (resp.get("content") or "").strip().strip('"').strip()
    if answer in candidates:
        return answer, cost
    # Models sometimes wrap the name in a sentence. Accept only an unambiguous
    # mention: if two candidates appear, we cannot tell which was meant, and a
    # guess here would be a silent wrong label rather than a visible failure.
    hits = [c for c in candidates if c in answer]
    return (hits[0] if len(hits) == 1 else None), cost


def main():
    ap = argparse.ArgumentParser(description="Retrieve-then-rerank the disputed labels")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--candidates", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = labels_dir(args.benchmark)
    sources = sorted(p for p in d.glob("pseudo_labels.*.jsonl")
                     if "_limit" not in p.name and "consensus" not in p.name
                     and "rerank" not in p.name and "selftrain" not in p.name)
    if len(sources) < 2:
        raise SystemExit("need at least two labellings to find disputed rows")

    frame = pd.DataFrame({p.name: read_labels(p).set_index("id")["intent"]
                          for p in sources}).dropna()
    proposals = frame.apply(lambda r: sorted(set(r)), axis=1)
    disputed = proposals[proposals.map(len) > 1]

    print(f"{args.benchmark}: {len(frame):,} rows across {len(sources)} labellings")
    for p in sources:
        print(f"    {p.name}")
    print(f"  agreed    {len(frame) - len(disputed):,} ({1 - len(disputed)/len(frame):.1%}) -- kept as is")
    print(f"  DISPUTED  {len(disputed):,} ({len(disputed)/len(frame):.1%}) -- reranked")

    retr = IntentRetriever(args.benchmark)
    texts = load_pool_unlabeled(args.benchmark).set_index("id")["text"].to_dict()

    todo = list(disputed.items())
    if args.limit:
        todo = todo[: args.limit]

    # candidates = every proposal, topped up by retrieval, order stable
    jobs = []
    recall_top = 0
    for i, props in todo:
        cands = list(props)
        for c in retr.top(texts[i], args.candidates * 2):
            if len(cands) >= args.candidates:
                break
            if c not in cands:
                cands.append(c)
        if props[0] in retr.top(texts[i], args.candidates):
            recall_top += 1
        jobs.append((i, cands))

    sizes = [len(c) for _, c in jobs]
    print(f"  candidates per message: min {min(sizes)}, mean {np.mean(sizes):.1f}, max {max(sizes)}")
    print(f"  retrieval alone would have found a proposed label for "
          f"{recall_top}/{len(jobs)} ({recall_top/len(jobs):.0%})")

    if args.dry_run:
        est = sum(len(build_prompt(texts[i], c, retr.examples)) + len(texts[i])
                  for i, c in jobs) // 4
        print(f"\n  DRY RUN -- ~{est/1e6:.2f}M input tokens over {len(jobs):,} calls")
        print("  no API calls made")
        return

    out = d / f"pseudo_labels.rerank.{model_slug(args.model)}.jsonl"
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(str(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    jobs = [(i, c) for i, c in jobs if i not in done]
    print(f"  {len(done):,} already done, {len(jobs):,} to go\n")

    api_key = get_api_key()
    lock = threading.Lock()
    totals = {"cost": 0.0, "failed": 0, "n": 0, "changed": 0}
    started = time.time()

    def run(job):
        i, cands = job
        answer, cost = label_one(texts[i], cands, retr.examples, args.model, api_key)
        with lock:
            totals["cost"] += cost
            totals["n"] += 1
            if answer is None:
                totals["failed"] += 1
            else:
                if answer != proposals[i][0]:
                    totals["changed"] += 1
                with open(out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "id": i, "intent": answer, "model": args.model,
                        "rules": "rerank", "candidates": len(cands),
                    }) + "\n")
            if totals["n"] % 100 == 0 or totals["n"] == len(jobs):
                print(f"  [{totals['n']}/{len(jobs)}] ${totals['cost']:.3f}  "
                      f"{totals['failed']} failed")

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, jobs))

    print(f"\ndone in {(time.time()-started)/60:.1f} min, cost ${totals['cost']:.3f}, "
          f"{totals['failed']} failed")

    # merge: agreed rows keep their label, disputed rows take the rerank
    rerank = read_labels(out).set_index("id")["intent"].to_dict()
    primary = frame[sources[0].name]
    merged = {i: rerank.get(i, primary[i]) for i in frame.index}
    full = d / f"pseudo_labels.reranked_full.{model_slug(args.model)}.jsonl"
    with open(full, "w", encoding="utf-8") as fh:
        for i, v in merged.items():
            fh.write(json.dumps({"id": i, "intent": v,
                                 "model": f"rerank/{args.model}",
                                 "rules": "rerank"}) + "\n")
    diff = sum(1 for i in frame.index if merged[i] != primary[i])
    print(f"wrote {full.name}: {len(merged):,} rows, {diff:,} differ from {sources[0].name}")


if __name__ == "__main__":
    main()
