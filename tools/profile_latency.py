"""Where does the per-message time actually go?

We have optimised the forward pass three times (fp16, length-sorted batching,
torch.compile) on the assumption that it dominates. It may not any more: after
a 2.5x speedup the fixed costs -- tokenisation on the CPU, the host/device
transfer, the Python that turns logits back into intent names -- are a larger
share of a smaller total. Optimising the wrong one wastes the time we have.

So this splits the timed region into its parts, under the same conditions the
protocol prescribes: batches of 64, one untimed warm-up, length-sorted, on the
GPU. It reads test.tsv's TEXT for timing only, exactly as sweep_encoders and
submit already do -- no labels, no tuning, no model selection (hard rule 1).

    python tools/profile_latency.py --model DIR --fp16 --compile
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classify import (BATCH_SIZE, MAX_LENGTH, PAD_MULTIPLE, _sync,  # noqa: E402
                      compile_model, load_classifier)
from data import load_test  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Split the timed region into its parts")
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", default="benchmark_a")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--compile", action="store_true", dest="compile_graph")
    args = ap.parse_args()

    clf = load_classifier(args.model, fp16=args.fp16)
    if args.compile_graph:
        compile_model(clf)
    texts = load_test(args.benchmark)["text"].tolist()
    n = len(texts)
    order = sorted(range(n), key=lambda i: len(texts[i]))
    batches = [[texts[i] for i in order[s : s + args.batch_size]]
               for s in range(0, n, args.batch_size)]

    for b in batches[:2]:                      # warm up, untimed
        clf.classify(b)
    _sync(clf.device)

    tok = fwd = post = 0.0
    widths = []
    with torch.inference_mode():
        for b in batches:
            t0 = time.perf_counter()
            enc = clf._encode(b)
            _sync(clf.device)
            t1 = time.perf_counter()
            logits = clf.model(**enc).logits.float()
            _sync(clf.device)
            t2 = time.perf_counter()
            _ = [clf.intents[i] for i in clf._adjust(logits).argmax(dim=-1).tolist()]
            _sync(clf.device)
            t3 = time.perf_counter()
            tok += t1 - t0
            fwd += t2 - t1
            post += t3 - t2
            widths.append(enc["input_ids"].shape[1])

    total = tok + fwd + post
    print(f"\n{args.model}   {n:,} messages, {len(batches)} batches of "
          f"{args.batch_size}, PAD_MULTIPLE={PAD_MULTIPLE}")
    print(f"  padded width: min {min(widths)}, mean {sum(widths)/len(widths):.1f}, "
          f"max {max(widths)}  (cap {MAX_LENGTH})\n")
    for name, t in (("tokenise (CPU)", tok), ("forward (GPU)", fwd),
                    ("argmax + names", post)):
        print(f"  {name:<16s} {t*1000/n:>7.4f} ms/msg   {t/total:>6.1%}")
    print(f"  {'TOTAL':<16s} {total*1000/n:>7.4f} ms/msg")
    print("\n  Note: summing the parts costs more than the real pipeline, because")
    print("  each part is synchronised separately. Read the SHARES, not the total.")


if __name__ == "__main__":
    main()
