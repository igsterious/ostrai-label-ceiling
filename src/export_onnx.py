"""Export a trained model to ONNX, then prove it predicts the same thing.

WHY. After length-sorted batching removed the padding waste, what is left in
our 1.76 ms is mostly fixed overhead: Python dispatch, kernel launches, and
unfused elementwise ops. That is exactly what an ONNX Runtime graph removes --
it fuses operations and drops the training-time machinery PyTorch carries.

WHERE IT RUNS. Exporting happens HERE, offline, ahead of time, and writes a
directory. classify.py only ever LOADS that directory, so the classify pipeline
gains no export step and no new failure mode at submission time (hard rule 3
is about what the pipeline can reach, and it reaches a local folder).

THE CHECK IS THE POINT. A faster model that predicts differently is not a
faster model, it is a different submission. So this exports, then runs both the
PyTorch and ONNX versions over pool messages and reports how many predictions
disagree. Anything above a handful and the export is wrong -- do not ship it.

    python src/export_onnx.py /content/drive/MyDrive/ostrai/m_maj --out .../m_maj_onnx
    python src/export_onnx.py <src> --out <dst> --fp16 --check 1024
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Export a model to ONNX and verify it")
    ap.add_argument("model_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark", choices=["benchmark_a", "benchmark_b"],
                    default="benchmark_a", help="which pool to verify against")
    ap.add_argument("--check", type=int, default=1024,
                    help="how many pool messages to compare predictions on")
    ap.add_argument("--fp16", action="store_true",
                    help="also convert the ONNX graph to half precision")
    args = ap.parse_args()

    src, dst = Path(args.model_dir), Path(args.out)
    if not (src / "labels.json").exists():
        raise SystemExit(f"no labels.json in {src} -- is that a trained model?")

    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    print(f"exporting {src} -> {dst}")
    model = ORTModelForSequenceClassification.from_pretrained(str(src), export=True)
    model.save_pretrained(str(dst))
    AutoTokenizer.from_pretrained(str(src)).save_pretrained(str(dst))

    # labels.json carries the intent order AND the training prior. classify.py
    # reads both from the model directory, so the export is useless without it.
    shutil.copy(src / "labels.json", dst / "labels.json")
    print(f"  copied labels.json ({len(json.loads((dst / 'labels.json').read_text())['intents'])} intents)")

    if args.fp16:
        # Optimum's API for this has moved across versions; a failure here
        # leaves a perfectly good fp32 export rather than nothing.
        try:
            from optimum.onnxruntime import ORTOptimizer
            from optimum.onnxruntime.configuration import OptimizationConfig
            opt = ORTOptimizer.from_pretrained(str(dst))
            opt.optimize(save_dir=str(dst),
                         optimization_config=OptimizationConfig(
                             optimization_level=99, fp16=True))
            print("  optimised graph to fp16")
        except Exception as exc:  # noqa: BLE001 -- any failure is non-fatal here
            print(f"  fp16 optimisation skipped: {type(exc).__name__}: {exc}")
            print("  (the fp32 export is still valid; run inference with --fp16)")

    # --- the verification -------------------------------------------------
    from classify import classify_all, load_classifier
    from data import load_pool_unlabeled

    texts = load_pool_unlabeled(args.benchmark)["text"].tolist()[: args.check]
    print(f"\nverifying on {len(texts):,} pool messages")

    torch_clf = load_classifier(str(src), fp16=True)
    onnx_clf = load_classifier(str(dst), fp16=args.fp16)

    a = classify_all(torch_clf, texts)
    b = classify_all(onnx_clf, texts)

    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print(f"  predictions differing: {len(diff)} / {len(texts)} "
          f"({len(diff)/len(texts):.2%})")
    if diff[:5]:
        for i in diff[:5]:
            print(f"    {texts[i][:56]:58s} torch={a[i]:28s} onnx={b[i]}")

    if len(diff) == 0:
        print("\n  IDENTICAL -- safe to submit with this export")
    elif len(diff) <= 0.002 * len(texts):
        print("\n  a handful of boundary flips, consistent with fp16 rounding")
    else:
        print("\n  TOO MANY DIFFERENCES -- do not ship this export")


if __name__ == "__main__":
    main()
