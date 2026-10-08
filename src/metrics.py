"""Validation metrics. numpy only, no torch -- so this is testable locally.

Split out of train.py deliberately: train.py can only run in Colab, and the
dual-accuracy logic here is the part most worth being able to check by hand.
See metrics_selftest() at the bottom.

The two accuracies exist because the test set's class prior is unknowable --
hard rule 1 forbids putting test.tsv through anything that would reveal it.

  balanced_acc        every class weighted equally. Holdout is exactly 5 per
                      class, so plain accuracy on holdout IS this number.
  prior_weighted_acc  classes weighted by their frequency in the pool. What
                      accuracy would look like if the test set mirrors the
                      pool's distribution.

Neither is "the" answer. Benchmark A's pool is genuinely imbalanced (13.7:1),
so the two diverge there and the real test score sits between them. For
benchmark B the pool is uniform to within sampling noise, so they nearly
coincide and the question is moot.
"""

from __future__ import annotations

import numpy as np


def classification_report(preds, labels, n_classes, prior_vec=None):
    """Per-class recall/F1 plus the summary numbers, as a plain dict.

    preds, labels -- integer class ids, same length
    prior_vec     -- optional per-class weights (need not be normalised);
                     omitted means uniform, making prior_weighted == balanced
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    recalls = np.full(n_classes, np.nan)
    f1s = np.zeros(n_classes)
    for c in range(n_classes):
        truth = labels == c
        n_true = int(truth.sum())
        if n_true == 0:
            continue  # class absent from this eval set: no opinion, not a zero
        pred_c = preds == c
        tp = int((truth & pred_c).sum())
        recall = tp / n_true
        precision = tp / int(pred_c.sum()) if pred_c.any() else 0.0
        recalls[c] = recall
        if precision + recall > 0:
            f1s[c] = 2 * precision * recall / (precision + recall)

    present = ~np.isnan(recalls)
    if not present.any():
        raise ValueError("no classes present in labels")

    if prior_vec is None:
        weights = np.ones(int(present.sum()), dtype=np.float64)
    else:
        weights = np.asarray(prior_vec, dtype=np.float64)[present]
    weights = weights / weights.sum()

    return {
        "accuracy": float((preds == labels).mean()),
        "balanced_acc": float(recalls[present].mean()),
        "prior_weighted_acc": float((recalls[present] * weights).sum()),
        "macro_f1": float(f1s[present].mean()),
        "worst_class_recall": float(recalls[present].min()),
        "classes_at_zero": int((recalls[present] == 0).sum()),
    }


def make_compute_metrics(intents, prior):
    """Adapt classification_report to the shape HF Trainer expects."""
    prior_vec = np.array([prior[name] for name in intents], dtype=np.float64)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return classification_report(
            logits.argmax(axis=-1), labels, len(intents), prior_vec
        )

    return compute_metrics


def metrics_selftest():
    """Hand-checkable cases. Run: python src/metrics.py"""
    # Two classes. Class 0 perfect (2/2), class 1 all wrong (0/2).
    # balanced = (1.0 + 0.0)/2 = 0.5 regardless of how common each class is.
    r = classification_report(preds=[0, 0, 0, 0], labels=[0, 0, 1, 1], n_classes=2)
    assert r["balanced_acc"] == 0.5, r
    assert r["accuracy"] == 0.5, r
    assert r["worst_class_recall"] == 0.0, r
    assert r["classes_at_zero"] == 1, r

    # Same predictions, but class 0 is 9x commoner in the wild. Weighting by
    # that prior should pull accuracy up towards class 0's perfect recall.
    r = classification_report([0, 0, 0, 0], [0, 0, 1, 1], 2, prior_vec=[0.9, 0.1])
    assert r["balanced_acc"] == 0.5, r
    assert abs(r["prior_weighted_acc"] - 0.9) < 1e-9, r

    # ...and the reverse prior pulls it down. This asymmetry is the whole point.
    r = classification_report([0, 0, 0, 0], [0, 0, 1, 1], 2, prior_vec=[0.1, 0.9])
    assert abs(r["prior_weighted_acc"] - 0.1) < 1e-9, r

    # An unnormalised prior must behave identically to its normalised form.
    a = classification_report([0, 0, 0, 0], [0, 0, 1, 1], 2, prior_vec=[9.0, 1.0])
    assert abs(a["prior_weighted_acc"] - 0.9) < 1e-9, a

    # A class with no examples in the eval set must be ignored, not scored 0.
    r = classification_report([0, 0], [0, 0], n_classes=3, prior_vec=[0.5, 0.3, 0.2])
    assert r["balanced_acc"] == 1.0, r
    assert r["classes_at_zero"] == 0, r

    # Perfect predictions score 1.0 under any prior. Compared with a tolerance:
    # normalising the weights makes the sum land a float ulp off exactly 1.0.
    r = classification_report([0, 1, 2], [0, 1, 2], 3, prior_vec=[0.7, 0.2, 0.1])
    assert abs(r["balanced_acc"] - 1.0) < 1e-9, r
    assert abs(r["prior_weighted_acc"] - 1.0) < 1e-9, r
    assert abs(r["macro_f1"] - 1.0) < 1e-9, r

    # Uniform prior must reproduce balanced accuracy exactly.
    preds = [0, 1, 1, 2, 2, 0]
    labels = [0, 1, 2, 2, 1, 0]
    u = classification_report(preds, labels, 3)
    w = classification_report(preds, labels, 3, prior_vec=[1 / 3, 1 / 3, 1 / 3])
    assert abs(u["balanced_acc"] - w["prior_weighted_acc"]) < 1e-9, (u, w)

    print("metrics self-test passed")


if __name__ == "__main__":
    metrics_selftest()
