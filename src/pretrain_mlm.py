"""Adapt the encoder to this domain before fine-tuning, using no labels at all.

WHY. Every model in this project went straight from a generic checkpoint to
fine-tuning on 9,400 pseudo-labelled rows. The encoder therefore meets banking
and assistant vocabulary for the first time while it is also learning a 77-way
decision -- and the labels it learns that decision from are ~10% wrong, which
is where three days of experiments have all bottomed out.

Masked-language-model training on the pool text separates those two jobs. The
encoder learns the domain's vocabulary, phrasings and distinctions FIRST, from
27,000 messages, with no labels involved and therefore no label noise to
inherit. Fine-tuning then starts from a checkpoint that already understands
"top-up", "direct debit", "PTO" and the difference between "hasn't gone through
yet" and "was declined".

This is standard practice (Gururangan et al., "Don't Stop Pretraining"), and it
is the only lever left that acts BEFORE labels enter the pipeline.

RULES. Reads pool.tsv for both benchmarks -- provided files, no labels, no
external data. Strictly less information than the pseudo-labelled training we
already do, so if that is permitted this is.

BOTH POOLS TOGETHER. A's banking messages and B's assistant messages are
different domains, but the encoder is adapting to short customer utterances in
general, and 27,000 rows adapt better than 10,000. The fine-tune afterwards is
per-benchmark as always.

    python src/pretrain_mlm.py --encoder microsoft/deberta-v3-large --out DIR
    python src/train.py benchmark_a --encoder DIR --labels-file ...
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
)

from config import BENCHMARKS
from data import load_pool
from train import MAX_LENGTH, make_training_args


def main():
    ap = argparse.ArgumentParser(description="Domain-adaptive MLM on the pool text")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-large")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS),
                    help="whose pool text to adapt on; both by default")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    texts = []
    for b in args.benchmarks:
        t = load_pool(b)["text"].tolist()
        texts.extend(t)
        print(f"  {b}: {len(t):,} pool messages")
    print(f"total {len(texts):,} messages, "
          f"mean {sum(map(len, texts))/len(texts):.1f} chars")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    ds = Dataset.from_dict({"text": texts}).map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True, remove_columns=["text"],
    )

    # The MLM head is usually absent from a fine-tuning checkpoint and gets
    # initialised fresh. That is fine and expected -- we are not keeping the
    # head, only the encoder body it trains.
    model = AutoModelForMaskedLM.from_pretrained(args.encoder)
    model = model.float()   # deberta-v3 ships fp16; AMP dies on fp16 masters

    out = Path(args.out)
    targs = make_training_args(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        seed=args.seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=args.mask_prob),
    )
    result = trainer.train()

    loss = result.training_loss
    print(f"\n  final training loss {loss:.4f}  (perplexity {math.exp(loss):.2f})")

    # Save the ENCODER, not the MLM head: train.py will attach a fresh
    # classification head, and it loads a directory the same way it loads a hub
    # name, so nothing downstream needs to know this step happened.
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"  adapted encoder -> {out}")
    print(f"\n  now: python src/train.py benchmark_a --encoder {out} "
          f"--labels-file labels/benchmark_a/pseudo_labels.gpt3way.jsonl ...")


if __name__ == "__main__":
    main()
