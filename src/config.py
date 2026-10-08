"""Paths and constants shared by every script in this project.

Everything resolves relative to the repo root, which is found from this
file's own location rather than the current working directory. That means
scripts behave the same whether you run them from the repo root, from
src/, or from a Colab notebook cell in some other directory.

Override the data and artifact roots with the OSTRAI_DATA / OSTRAI_ARTIFACTS
environment variables if the layout differs (e.g. data on Google Drive).
"""

import os
from pathlib import Path

# __file__ is src/config.py, so one .parent gets src/, two gets the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("OSTRAI_DATA", REPO_ROOT / "data"))
ARTIFACTS_ROOT = Path(os.environ.get("OSTRAI_ARTIFACTS", REPO_ROOT / "artifacts"))

# Two output roots, split by whether the contents can be recreated for free.
#
#   artifacts/  reproducible. Splits come from a fixed seed, models from a
#               training run, QA reports from re-scoring. Gitignored.
#   labels/     NOT reproducible. Pseudo-labels cost real money at the API and
#               would differ on a re-run. Committed, so Colab gets them by
#               cloning rather than by a manual upload.
LABELS_ROOT = Path(os.environ.get("OSTRAI_LABELS", REPO_ROOT / "labels"))

BENCHMARKS = ("benchmark_a", "benchmark_b")

# Which labelling is canonical for each benchmark. A second labelling exists
# for A purely to derive an agreement signal, not to replace the first.
#
#   A: sonnet-5 scores 88.5% on the 100 human-judged messages, flash-lite
#      87.5%, opus-5 87.5%. Opus fixed four labels and broke five, so model
#      strength does not help here -- the residual errors sit in genuinely
#      ambiguous category boundaries. Sonnet stays primary on the measurement.
#   B: flash-lite, which scored 80/80 on human review and 99.7% on the gate,
#      at a tenth of sonnet's cost.
PRIMARY_LABELLER = {
    "benchmark_a": "anthropic/claude-sonnet-5",
    "benchmark_b": "google/gemini-3.1-flash-lite",
}

# The label budget. Each class has exactly 10 labeled examples and that is the
# ONLY ground truth in the whole project, so it gets split once, here, and
# every downstream script reads that split rather than re-deriving one.
#
#   few-shot  -> examples allowed inside the LLM labelling prompt
#   holdout   -> never shown to the labeller; scores the labeller (QA gate)
#                and validates the trained classifier
#
# Splitting per class, not by percentage: with 77 and 150 classes a 50%
# random split would leave some classes with zero holdout examples.
N_FEWSHOT_PER_CLASS = 5
N_HOLDOUT_PER_CLASS = 5

# Fixed so the split is identical on every machine and every re-run.
SPLIT_SEED = 20260904


def benchmark_dir(benchmark: str) -> Path:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}, expected one of {BENCHMARKS}")
    return DATA_ROOT / benchmark


def artifact_dir(benchmark: str) -> Path:
    """Where reproducible generated files go. Created on demand."""
    d = ARTIFACTS_ROOT / benchmark
    d.mkdir(parents=True, exist_ok=True)
    return d


def labels_dir(benchmark: str) -> Path:
    """Where paid-for pseudo-labels go. Committed; see LABELS_ROOT."""
    d = LABELS_ROOT / benchmark
    d.mkdir(parents=True, exist_ok=True)
    return d
