"""Hand-decide the rows the labellers could not agree on.

WHY THESE ROWS. Three gpt-5.1 runs labelled the pool with different prompts.
Where all three agree (8,462 rows, 91.6%) a human judged the label right 92% of
the time. Where they split (771 rows, 8.4%) the same judging said 57%. So
roughly 330 wrong labels sit inside a list we can enumerate -- no searching, no
sampling, and they are concentrated in exactly the neighbouring-category pairs
the model keeps getting wrong.

WHY IT SHOULD PAY. Swapping sonnet for gpt-5.1 moved label accuracy 86.4% ->
89.5% and bought +2.19 on the board. Correcting these rows moves it ~90% ->
~93.5%. It is the only remaining action that changes the labeller's BIAS rather
than resampling its noise, which is why every other idea has flattened out.

WHY IT IS NOT THE 220 THAT HURT. Those were ADDED as extra gold rows, skewing
the training set toward deliberately confusable messages. This REPLACES wrong
labels with right ones in place: same rows, same distribution, less noise.

    python tools/review_disputed.py benchmark_a
    python tools/review_disputed.py benchmark_a --limit 100

Answers are appended as you go, so you can stop at any point with Ctrl+C and
keep everything decided so far. Re-running skips what is already done.
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
from data import load_intents, load_labeled, load_pool_unlabeled  # noqa: E402
from labeller import read_labels  # noqa: E402

RUNS = (
    "pseudo_labels.openai_gpt-5.1.jsonl",
    "pseudo_labels.openai_gpt-5.1.rules-v2.jsonl",
    "pseudo_labels.openai_gpt-5.1.b10.jsonl",
)


def load_done(path):
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                     keep_default_na=False)
    return dict(zip(df["id"], df["intent"]))


def main():
    ap = argparse.ArgumentParser(description="Review the disputed pool labels")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--limit", type=int, help="stop after this many decisions")
    ap.add_argument("--examples", type=int, default=3,
                    help="labelled examples to show per candidate")
    args = ap.parse_args()

    d = labels_dir(args.benchmark)
    missing = [f for f in RUNS if not (d / f).exists()]
    if missing:
        raise SystemExit(f"missing labelling(s): {missing}")

    runs = [read_labels(d / f).set_index("id")["intent"] for f in RUNS]
    ids = runs[0].index
    for r in runs[1:]:
        ids = ids.intersection(r.index)

    texts = load_pool_unlabeled(args.benchmark).set_index("id")["text"].to_dict()
    examples = {i: list(g["text"])[: args.examples]
                for i, g in load_labeled(args.benchmark).groupby("intent")}

    disputed = []
    for i in ids:
        votes = [r[i] for r in runs]
        if len(set(votes)) > 1:
            # commonest first: the majority label is usually the right default
            order = [v for v, _ in Counter(votes).most_common()]
            disputed.append((i, order))

    out = d / "human_disputed.tsv"
    done = load_done(out)
    todo = [(i, c) for i, c in disputed if i not in done]

    intents = set(load_intents(args.benchmark))

    print(f"\n{args.benchmark}: {len(disputed):,} disputed rows, "
          f"{len(done):,} already decided, {len(todo):,} to go\n")
    print("  1 / 2 / 3       pick that candidate")
    print("  2 why it is 2   pick, and record the reason")
    print("  <intent_name>   a category neither run proposed (name must match "
          "intents.txt)")
    print("  ?word           list intents containing 'word'")
    print("  s               skip        q  save and quit\n")

    if not out.exists():
        with open(out, "w", encoding="utf-8", newline="") as fh:
            fh.write("id\tintent\tnote\n")

    decided = 0
    n = 0
    while n < len(todo):
        if args.limit and decided >= args.limit:
            break
        i, cands = todo[n]
        n += 1
        print("=" * 72)
        print(f"[{n}/{len(todo)}]  {texts[i]}")
        print()
        for k, c in enumerate(cands, 1):
            print(f"  {k}. {c}")
            for ex in examples.get(c, []):
                print(f"       {ex}")
        print()
        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  stopped")
            break
        key = raw.lower()
        if key == "q":
            break
        if key in ("s", "u", ""):
            continue

        # "?word" searches intents.txt -- 77 names is too many to remember, and
        # a typo would otherwise be silently rejected below.
        if key.startswith("?"):
            hits = sorted(x for x in intents if key[1:] in x)
            print("   " + ("\n   ".join(hits) if hits else "no match"))
            n -= 1   # ask this one again
            continue

        # choice is matched lowercase (intent names are), the note keeps its case
        choice = key.partition(" ")[0]
        note = raw.partition(" ")[2].strip().replace("\t", " ")

        if choice.isdigit() and 1 <= int(choice) <= len(cands):
            intent = cands[int(choice) - 1]
        elif choice in intents:
            # a third category: exactly the case the audit kept finding, where
            # both runs were wrong in the same direction
            intent = choice
            print(f"  -> {intent} (neither candidate)")
        else:
            print("  ? not a candidate and not an intent name -- try ?word to search")
            n -= 1
            continue

        with open(out, "a", encoding="utf-8", newline="") as fh:
            fh.write(f"{i}\t{intent}\t{note}\n")
        decided += 1

    print(f"\n  {decided} decided this session, "
          f"{len(done) + decided} of {len(disputed):,} total -> {out.name}")
    print("  next: python tools/apply_disputed.py " + args.benchmark)


if __name__ == "__main__":
    main()
