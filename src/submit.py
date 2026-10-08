"""Produce the submission JSON: predictions for both benchmarks, plus timing.

This is the only place test.tsv is ever read for predictions, and it is read
exactly once, at the end -- hard rule 1. Nothing here fits, tunes or selects
anything; the models are already trained and frozen.

OUTPUT SHAPE is dictated by the leaderboard, not by us. It is a JSON ARRAY of
two entries, one per benchmark:

    [{"name": ..., "benchmark": "A", "average_time_per_example": 0.34,
      "predictions": [{"id": 1, "intent": "card_arrival"}, ...]}, {...B...}]

Details the site rejects on, each enforced below:
  - ids are INTEGERS, not strings ("numpy types will not serialise" -- and
    neither do the strings our TSV reader produces, which is why int() is
    applied explicitly rather than trusted)
  - every id from 1..N appears exactly once, taken from test.tsv
  - intent copied verbatim from intents.txt, case included
  - both benchmarks in one file, under one identical name
  - average_time_per_example is PER BENCHMARK, not a combined figure

Timing reuses classify.py rather than reimplementing it, so the number
reported describes the same code that produced the predictions:
  - both models load BEFORE timing starts; load time is not counted
  - one untimed warm-up batch of 64 per benchmark
  - batches of 64 through the full pipeline, tokenisation included

    python src/submit.py --name "Your Name"
    python src/submit.py --name "Your Name" --fp16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from classify import (BATCH_SIZE, _sync, balanced_assign, classify_all,
                      compile_model, load_classifier, proba_all, warm_buckets)
from config import BENCHMARKS, REPO_ROOT, artifact_dir
from data import load_intents, load_test


def run_benchmark(benchmark, model_dir, fp16, batch_size, prior_tau=0.0,
                  sort_by_length=True, compile_graph=False, compile_mode=None,
                  balance=0.0):
    """Load, warm up, then time predictions over the whole test set.

    Returns (predictions, elapsed_seconds, n_messages). Loading happens before
    the clock starts, exactly as the protocol requires.
    """
    test = load_test(benchmark)
    texts = test["text"].tolist()

    clf = load_classifier(model_dir, fp16=fp16, prior_tau=prior_tau)  # untimed by design

    if compile_graph:
        compile_model(clf, mode=compile_mode)

    # one untimed warm-up batch: the first call pays for CUDA context setup,
    # kernel selection and allocation, none of which is per-message cost --
    # and, when compiling, the compilation itself.
    clf.classify(texts[:batch_size])
    _sync(clf.device)

    if compile_graph and compile_mode == "reduce-overhead":
        # CUDA graphs capture per shape, and a capture is expensive. The
        # prescribed warm-up covers one shape; this covers every shape the
        # timed run will meet, so no capture happens inside the measurement.
        buckets = warm_buckets(clf, texts, batch_size)
        print(f"  warmed {len(buckets)} shape buckets: {buckets}")
    elif compile_graph:
        # dynamic compile: warm the extreme length too, so the timed region
        # does not pay for specialising on it
        clf.classify(sorted(texts, key=len, reverse=True)[:batch_size])
        _sync(clf.device)

    # classify_all groups similar-length messages into a batch, which cuts the
    # padding waste (74% of A's forward pass, 55% of B's) without changing a
    # single prediction -- padding is already masked out of attention. The
    # grouping happens inside the timed region because it is real per-request
    # work; the predictions come back in the original order.
    _sync(clf.device)
    started = time.perf_counter()
    if balance:
        # Inside the timed region: it is real work the pipeline does. Sinkhorn
        # on 3,080 x 77 is a few milliseconds, so it barely registers.
        probs = proba_all(clf, texts, batch_size, sort_by_length)
        per_class = len(texts) / len(clf.intents)
        raw = probs.argmax(dim=1)
        idx = balanced_assign(probs, per_class, strength=balance)
        preds = [clf.intents[i] for i in idx.tolist()]
        _before, _after = raw.bincount(minlength=len(clf.intents)), \
            idx.bincount(minlength=len(clf.intents))
        _miss = lambda c: int((c - per_class).clamp(min=0).sum())
        _balance_report = (
            f"  balance {balance}: per-class {int(_before.min())}-{int(_before.max())}"
            f" -> {int(_after.min())}-{int(_after.max())} (target {per_class:.0f}), "
            f"misallocated {_miss(_before)} -> {_miss(_after)}, "
            f"{int((raw != idx).sum())} predictions changed")
    else:
        _balance_report = None
        preds = classify_all(clf, texts, batch_size, sort_by_length)
    _sync(clf.device)
    elapsed = time.perf_counter() - started
    if _balance_report:
        print(_balance_report)

    valid = set(load_intents(benchmark))
    bad = sorted({p for p in preds if p not in valid})
    if bad:
        raise SystemExit(f"{benchmark}: predictions not in intents.txt: {bad[:5]}")
    if len(preds) != len(test):
        raise SystemExit(f"{benchmark}: {len(preds)} predictions for {len(test)} messages")

    # int(), not the string the TSV reader gives us -- the site rejects string ids
    rows = [{"id": int(i), "intent": str(p)} for i, p in zip(test["id"], preds)]

    ids = [r["id"] for r in rows]
    expected = set(range(1, len(test) + 1))
    if set(ids) != expected:
        missing = sorted(expected - set(ids))[:5]
        extra = sorted(set(ids) - expected)[:5]
        raise SystemExit(
            f"{benchmark}: ids must be exactly 1..{len(test)}; "
            f"missing {missing}, unexpected {extra}"
        )
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{benchmark}: duplicate ids")

    return rows, elapsed, len(texts), clf.device


def main():
    ap = argparse.ArgumentParser(description="Build the submission JSON")
    ap.add_argument("--name", required=True, help="your name for the leaderboard")
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--prior-tau", type=float, default=0.0,
                    help="label-shift correction; test.tsv is class-uniform")
    ap.add_argument("--fp16", action="store_true", help="half precision at inference")
    ap.add_argument("--compile", action="store_true", dest="compile_graph",
                    help="torch.compile the forward pass; compilation lands in "
                         "the untimed warm-up")
    ap.add_argument("--compile-mode", choices=["default", "reduce-overhead"],
                    default="default",
                    help="reduce-overhead captures CUDA graphs, which removes "
                         "per-kernel launch overhead but needs static shapes "
                         "(bucketed padding, one warm-up per bucket)")
    ap.add_argument("--no-sort", action="store_true",
                    help="batch in file order, as before length-grouping existed "
                         "(same predictions, roughly 2-3x slower)")
    ap.add_argument("--model-dir", help="template with {benchmark}; default artifacts/<b>/model")
    # A and B are tuned separately on purpose -- different class counts, different
    # confusion structure -- so the winner on one is not the winner on the other.
    # These override the template per benchmark; each accepts a comma-separated
    # list for an ensemble, exactly as --model-dir does everywhere else.
    ap.add_argument("--model-a", help="model dir(s) for benchmark A, overriding --model-dir")
    ap.add_argument("--model-b", help="model dir(s) for benchmark B, overriding --model-dir")
    # Run one benchmark alone. The file it writes is not a valid submission
    # on its own -- the site rejects a file with only one entry -- but it is
    # exactly what tools/splice.py needs when the other half is being carried
    # over from a banked run, which is the normal case for B.
    ap.add_argument("--only", choices=["benchmark_a", "benchmark_b"],
                    help="predict just one benchmark, for splicing")
    ap.add_argument("--balance", type=float, default=0.0, metavar="STRENGTH",
                    help="force each class to its known count (test.tsv is "
                         "77x40 and 150x30). 1.0 = full, 0.5 = half force")
    args = ap.parse_args()

    override = {"benchmark_a": args.model_a, "benchmark_b": args.model_b}

    # the site keys benchmarks as "A"/"B", not by our directory names
    letters = {"benchmark_a": "A", "benchmark_b": "B"}

    entries = []
    total_elapsed = 0.0
    total_messages = 0

    wanted = [args.only] if args.only else list(BENCHMARKS)
    for benchmark in wanted:
        if override[benchmark]:
            spec = override[benchmark]
        elif args.model_dir:
            spec = args.model_dir.format(benchmark=benchmark)
        else:
            spec = str(artifact_dir(benchmark) / "model")

        # A spec may name several models -- comma for an ensemble, "|" for a
        # cascade, with an optional "@threshold" -- so every part is checked
        # rather than the string as a whole; otherwise a typo in the second
        # member would surface as a confusing load error much later.
        bare = spec.partition("@")[0]
        parts = [p.strip() for p in bare.replace("|", ",").split(",") if p.strip()]
        for part in parts:
            if not Path(part).exists():
                raise SystemExit(
                    f"no model at {part}. Run: python src/train.py {benchmark}"
                )
        if "|" in spec:
            print(f"{letters[benchmark]}: cascade, fast -> slow "
                  f"(threshold {spec.partition('@')[2] or '0.95'})")
        elif len(parts) > 1:
            print(f"{letters[benchmark]}: ensemble of {len(parts)} models")

        rows, elapsed, n, device = run_benchmark(
            benchmark, spec, args.fp16, args.batch_size, args.prior_tau,
            sort_by_length=not args.no_sort, compile_graph=args.compile_graph,
            compile_mode=args.compile_mode, balance=args.balance)
        ms = elapsed * 1000.0 / n
        entries.append({
            "name": args.name,                      # identical across both entries
            "benchmark": letters[benchmark],
            "average_time_per_example": float(round(ms, 4)),
            "predictions": rows,
        })
        total_elapsed += elapsed
        total_messages += n
        print(f"{letters[benchmark]}: {n:,} predictions, {ms:.4f} ms/example  ({device})")
        if device != "cuda":
            print("  WARNING: not on a GPU -- the official number must come from a Colab T4")

    if len({e["name"] for e in entries}) != 1:
        raise SystemExit("both entries must carry the same name")
    if not args.only and {e["benchmark"] for e in entries} != {"A", "B"}:
        raise SystemExit("the file must contain exactly one entry for A and one for B")
    if args.only:
        print(f"\n  --only {args.only}: this file has ONE entry, which the site "
              "rejects on its own.\n  Splice it against a banked run before uploading.")

    out = Path(args.out)
    out.write_text(json.dumps(entries), encoding="utf-8")

    print(f"\n  name                      {args.name!r} (same on both entries)")
    for e in entries:
        print(f"  benchmark {e['benchmark']}               "
              f"{len(e['predictions']):,} predictions, "
              f"{e['average_time_per_example']} ms/example")
    print(f"  combined (not submitted)  {total_elapsed*1000/total_messages:.4f} ms "
          f"over {total_messages:,} messages")
    print(f"  wrote {out.resolve()}  ({out.stat().st_size/1024:.0f} KB)")

    # re-read what was actually written, so the checks describe the FILE
    check = json.loads(out.read_text(encoding="utf-8"))
    id_types = {type(p["id"]).__name__ for e in check for p in e["predictions"]}
    print(f"\n  top level is a list:      {isinstance(check, list)} ({len(check)} entries)")
    print(f"  id types in the file:     {sorted(id_types)}")
    print(f"  ids are ints:             {id_types == {'int'}}")


if __name__ == "__main__":
    main()
