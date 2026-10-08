"""Backfill train_prior into an already-trained model's labels.json.

Models trained before the label-shift correction existed have no prior stored,
and inference cannot recompute it: classify.py must not import training_data,
which reaches labeller and therefore requests, breaking hard rule 3. So the
prior has to be written next to the model as plain data, once, offline.

    python src/add_prior.py /content/drive/MyDrive/ostrai/model_deberta benchmark_a

Why the prior is needed at all: test.tsv is exactly 3,080 = 77 x 40 for A and
4,500 = 150 x 30 for B, i.e. class-uniform, while the pseudo-labelled pool A
trains on runs 13.7:1 from rarest class to commonest. A model fitted under one
prior and scored under another is biased toward whatever was frequent in
training; subtracting log P_train(y) from the logits corrects it exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_data import class_prior, label_maps


def main():
    ap = argparse.ArgumentParser(description="Add train_prior to a model's labels.json")
    ap.add_argument("model_dir")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--labeller-model", help="the pseudo-label file the model trained on")
    ap.add_argument("--force", action="store_true", help="overwrite an existing prior")
    args = ap.parse_args()

    path = Path(args.model_dir) / "labels.json"
    if not path.exists():
        raise SystemExit(f"no labels.json at {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))

    if meta.get("train_prior") and not args.force:
        raise SystemExit(f"{path} already has train_prior ({len(meta['train_prior'])} classes)")

    _, intents = label_maps(args.benchmark)
    if meta.get("intents") != intents:
        raise SystemExit(
            "intent list in labels.json does not match this benchmark -- "
            "writing a prior in the wrong order would silently mislabel everything"
        )

    prior = class_prior(args.benchmark, args.labeller_model)
    meta["train_prior"] = [float(prior[i]) for i in intents]
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    vals = meta["train_prior"]
    n = len(vals)
    print(f"wrote train_prior for {n} classes -> {path}")
    print(f"  uniform would be {1/n:.4f}")
    print(f"  max {max(vals):.4f} ({max(vals)*n:.2f}x uniform)  "
          f"min {min(vals):.4f} ({min(vals)*n:.2f}x)  ratio {max(vals)/min(vals):.1f}:1")
    print(f"  sums to {sum(vals):.4f}")


if __name__ == "__main__":
    main()
