"""Relabel the pool with an ensemble, so one model can be trained on its answers.

WHY. The ensemble is now our best labeller of benchmark A -- better than any
single LLM labelling we have bought. But it is four models, and every one of
them has to run at inference. Distillation moves what the ensemble knows into
a single student: relabel the pool with the ensemble, train one model on those
labels, ship that.

WHY THIS IS NOT CIRCULAR, quite. Each member was fitted on the pool, so asked
about a pool row it largely parrots the label it was trained on -- that is the
memorisation problem that makes naive self-training useless. What rescues it
here is that the four members memorised FOUR DIFFERENT label sets. On the ~85%
of rows where every labelling already agreed, the ensemble returns that
agreed label and nothing is gained or lost. On the disputed rows the members
pull against each other, and the averaged probability is an actual decision
informed by four independently-fitted views rather than one labeller's guess.

So the output is a soft consensus over the disputed rows, which is exactly the
part of the training set that was costing us accuracy.

RULES. The pool only -- test.tsv is never read here (hard rule 1), and no
external model or label is involved (hard rule 2): these are our own weights.

    python src/distil_labels.py benchmark_a --models dirA,dirB,dirC,dirD
    python src/distil_labels.py benchmark_a --models ... --min-conf 0.5
"""

from __future__ import annotations

import argparse
import json

import torch

from classify import BATCH_SIZE, load_classifier
from config import labels_dir
from data import load_pool_unlabeled
from labeller import read_labels


def main():
    ap = argparse.ArgumentParser(description="Relabel the pool with an ensemble")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--models", required=True,
                    help="comma-separated model dirs, exactly as submit.py takes them")
    ap.add_argument("--tag", default="ens", help="goes in the output filename")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--prior-tau", type=float, default=0.0,
                    help="label-shift correction; the POOL is not uniform, so "
                         "the default 0 is usually right here even though "
                         "submit.py uses 1.0 against a uniform test set")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="drop rows the ensemble is less sure of than this")
    # The argmax throws away the part of the teacher that is worth having. On a
    # disputed row the ensemble says something like 0.5 / 0.4 / 0.1 across three
    # neighbouring intents, and that spread IS the information -- it says these
    # two are nearly the same category and the third is close. Training on the
    # argmax instead tells the student to be fully confident about a coin flip,
    # which is exactly the near-miss confusion every audit has found.
    ap.add_argument("--soft", action="store_true",
                    help="also write the teacher's probability distribution")
    ap.add_argument("--topk", type=int, default=15,
                    help="how many intents to keep per row when --soft (the tail "
                         "is numerically irrelevant and dominates the file size)")
    args = ap.parse_args()

    pool = load_pool_unlabeled(args.benchmark)
    texts = pool["text"].tolist()
    ids = pool["id"].tolist()

    clf = load_classifier(args.models, fp16=args.fp16, prior_tau=args.prior_tau)
    n_members = len(getattr(clf, "members", [clf]))
    print(f"{args.benchmark}: relabelling {len(texts):,} pool rows "
          f"with {n_members} model(s) on {clf.device}")

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i : i + args.batch_size]
            p = (clf.predict_proba(batch) if n_members == 1
                 else _ensemble_proba(clf, batch))
            probs.append(p.float().cpu())
            if (i // args.batch_size) % 25 == 0:
                print(f"  {min(i + args.batch_size, len(texts)):,}/{len(texts):,}")
    probs = torch.cat(probs)
    conf, idx = probs.max(dim=-1)
    labels = [clf.intents[j] for j in idx.tolist()]

    out = labels_dir(args.benchmark) / f"pseudo_labels.distil_{args.tag}.jsonl"
    kept = 0
    with open(out, "w", encoding="utf-8") as fh:
        for i, lab, c in zip(ids, labels, conf.tolist()):
            if c < args.min_conf:
                continue
            fh.write(json.dumps({"id": i, "intent": lab, "model": f"distil/{args.tag}",
                                 "rules": "distil", "conf": round(c, 4)}) + "\n")
            kept += 1
    print(f"\nwrote {out.name}: {kept:,} rows"
          + (f" ({len(ids) - kept:,} dropped below {args.min_conf})" if args.min_conf else ""))
    print(f"  mean confidence {conf.mean():.3f}, median {conf.median():.3f}")

    if args.soft:
        k = min(args.topk, len(clf.intents))
        top_p, top_i = probs.topk(k, dim=-1)
        soft = labels_dir(args.benchmark) / f"pseudo_labels.distil_{args.tag}.soft.jsonl"
        with open(soft, "w", encoding="utf-8") as fh:
            for i, lab, c, ps, js in zip(ids, labels, conf.tolist(),
                                         top_p.tolist(), top_i.tolist()):
                if c < args.min_conf:
                    continue
                # Renormalised over the kept k, so the student's target is a
                # proper distribution rather than one summing to 0.98.
                s = sum(ps)
                fh.write(json.dumps({
                    "id": i, "intent": lab,
                    "top": [[clf.intents[j], round(p / s, 6)] for p, j in zip(ps, js)],
                }) + "\n")
        mass = top_p.sum(dim=-1)
        print(f"wrote {soft.name}: top-{k} per row, "
              f"capturing {mass.mean():.4f} of the mass on average "
              f"(worst row {mass.min():.4f})")
        # How much of the teacher is actually soft? If nearly every row is a
        # confident one-hot there is nothing here the argmax did not already say.
        spread = (conf < 0.9).float().mean()
        print(f"  {spread:.1%} of rows are below 0.9 confidence -- "
              "these are the rows soft targets can teach something about")

    # How far did it move? A distillation that changes nothing is a distillation
    # that will train an identical model -- worth knowing before spending a run.
    d = labels_dir(args.benchmark)
    written = dict(zip(ids, labels))
    for p in sorted(d.glob("pseudo_labels.*.jsonl")):
        if "distil" in p.name or "_limit" in p.name or "rerank." in p.name:
            continue
        other = read_labels(p).set_index("id")["intent"].to_dict()
        shared = [i for i in written if i in other]
        if not shared:
            continue
        k = sum(1 for i in shared if written[i] != other[i])
        print(f"  differs from {p.name[14:-6]:38s} {k:5,}/{len(shared):,} ({k/len(shared):.1%})")


def _ensemble_proba(clf, batch):
    total = None
    for w, m in zip(clf.weights, clf.members):
        p = m.predict_proba(batch) * w
        total = p if total is None else total + p
    return total / sum(clf.weights)


if __name__ == "__main__":
    main()
