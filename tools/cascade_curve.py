"""Choose the cascade threshold, using pool and holdout -- never test.tsv.

THE QUESTION. A cascade answers cheaply whenever the fast model's confidence
clears a threshold, and escalates the rest. Two numbers decide whether that is
a good trade, and both move with the threshold:

  coverage   what fraction never reaches the slow model  -> the latency saving
  agreement  of those, how often fast and slow would have said the same
             -> an upper bound on what the shortcut can cost

Agreement needs no labels at all, which is what makes this measurable honestly:
we compare two models' opinions on messages whose answers we do not know. Where
labels DO exist (holdout, and the human-confirmed pool rows) accuracy is
reported per side, so the disagreements can be attributed rather than assumed
to favour the slow model.

HARD RULE 1. test.tsv is never opened here. The threshold is fixed before the
submission runs, from data we are allowed to fit on.

    python tools/cascade_curve.py benchmark_a --fast <dir> --slow <dir>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classify import BATCH_SIZE, Classifier  # noqa: E402
from config import artifact_dir  # noqa: E402
from data import load_pool_unlabeled  # noqa: E402
from training_data import human_confirmed  # noqa: E402

THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999)


def run(clf, texts, batch_size):
    """Predictions and top-1 confidence for every text, in input order."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    preds = [None] * len(texts)
    conf = [0.0] * len(texts)
    with torch.inference_mode():
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            p = clf.predict_proba([texts[i] for i in idx]).float()
            c, t = p.max(dim=-1)
            for i, ci, ti in zip(idx, c.tolist(), t.tolist()):
                preds[i], conf[i] = clf.intents[ti], ci
    return preds, conf


def report(name, texts, truth, fast, slow, batch_size):
    fp, fc = run(fast, texts, batch_size)
    sp, _ = run(slow, texts, batch_size)
    n = len(texts)

    print(f"\n{'='*74}\n{name}: {n:,} messages\n{'='*74}")
    if truth is not None:
        fa = sum(1 for p, t in zip(fp, truth) if p == t) / n
        sa = sum(1 for p, t in zip(sp, truth) if p == t) / n
        print(f"  fast alone {fa:.1%}    slow alone {sa:.1%}    "
              f"(gap {sa-fa:+.1%})")
    print(f"  the two disagree on {sum(1 for a, b in zip(fp, sp) if a != b):,} "
          f"of {n:,} ({sum(1 for a, b in zip(fp, sp) if a != b)/n:.1%})\n")

    head = f"  {'thresh':>7s} {'covered':>9s} {'agree|cov':>10s}"
    if truth is not None:
        head += f" {'cascade acc':>12s} {'vs slow':>9s}"
    print(head)

    for th in THRESHOLDS:
        cov = [i for i in range(n) if fc[i] >= th]
        if not cov:
            continue
        agree = sum(1 for i in cov if fp[i] == sp[i]) / len(cov)
        line = f"  {th:>7.3f} {len(cov)/n:>8.1%} {agree:>10.2%}"
        if truth is not None:
            # what the cascade would actually predict: fast where covered,
            # slow elsewhere
            hits = sum(1 for i in range(n)
                       if (fp[i] if fc[i] >= th else sp[i]) == truth[i])
            sa = sum(1 for p, t in zip(sp, truth) if p == t)
            line += f" {hits/n:>11.1%} {(hits-sa)/n:>+8.1%}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description="Pick a cascade threshold on pool/holdout")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--fast", required=True)
    ap.add_argument("--slow", required=True)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--pool-sample", type=int, default=4000,
                    help="pool rows to measure agreement on (no labels needed)")
    args = ap.parse_args()

    fast = Classifier(args.fast, fp16=args.fp16)
    slow = Classifier(args.slow, fp16=args.fp16)
    if fast.intents != slow.intents:
        raise SystemExit("label orders differ between the two models")
    print(f"fast {args.fast}\nslow {args.slow}\ndevice {fast.device}")

    # 1. Holdout: true labels, never trained on by either model. Small and
    #    easier than test, so read the SHAPE of the curve here, not the level.
    hp = artifact_dir(args.benchmark) / "holdout.tsv"
    if hp.exists():
        h = pd.read_csv(hp, sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                        keep_default_na=False)
        report("HOLDOUT (true labels, unseen by both)", h["text"].tolist(),
               h["intent"].tolist(), fast, slow, args.batch_size)

    # 2. Human-confirmed pool rows: true labels AND pool-distributed, but
    #    selected for being confusable, so this is the pessimistic end.
    conf_rows = human_confirmed(args.benchmark)
    if not conf_rows.empty:
        report("HUMAN-CONFIRMED POOL ROWS (pessimistic: chosen as confusable)",
               conf_rows["text"].tolist(), conf_rows["intent"].tolist(),
               fast, slow, args.batch_size)

    # 3. Pool at scale: no labels, so only agreement -- but thousands of rows
    #    drawn from the same distribution as test, which is what we need the
    #    coverage number from.
    pool = load_pool_unlabeled(args.benchmark)["text"].tolist()[: args.pool_sample]
    report("POOL SAMPLE (no labels: coverage and agreement only)",
           pool, None, fast, slow, args.batch_size)

    print("\nHow to read this: pick the smallest threshold whose agreement is")
    print("high enough that (1 - agreement) x coverage is a cost you accept.")
    print("Latency is then roughly  fast_ms + (1 - coverage) x slow_ms.")


if __name__ == "__main__":
    main()
