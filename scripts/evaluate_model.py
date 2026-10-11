#!/usr/bin/env python
"""Repeatable evaluation for EcoLens AI.

What this measures (honesty statement)
--------------------------------------
The default model is an ImageNet *object* classifier. It is **not** trained on
waste-material classes, so a naive "folder name == predicted class" accuracy is
meaningless. This script therefore evaluates the model against a labelled set
using an explicit **mapping file** that says, for each dataset class folder,
which recognised object labels are considered acceptable.

Without a dataset + mapping, this script cannot produce real metrics and will
say so instead of inventing numbers.

Dataset layout
--------------
    <data-dir>/<class_folder>/image1.jpg
    <data-dir>/<class_folder>/image2.png
    ...

Mapping file (JSON):
    {
      "plastic_bottle": ["water bottle", "pop bottle"],
      "glass_bottle": ["wine bottle", "beer bottle"]
    }

Usage:
    python scripts/evaluate_model.py --data-dir path/to/test --mapping mapping.json
    python scripts/evaluate_model.py --help

Reports: accuracy, macro precision/recall/F1, per-class metrics, confusion
matrix, and inference latency. Only real, measured values are printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Ensure `src` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.inference import analyze_image_bytes

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def _compute_metrics(labels: list[str], preds: list[str], classes: list[str]):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    for true, pred in zip(labels, preds):
        confusion[true][pred] += 1
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    per_class = {}
    for c in classes:
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = (precision, recall, f1)

    accuracy = sum(1 for t, p in zip(labels, preds) if t == p) / len(labels) if labels else 0.0
    macro_p = sum(v[0] for v in per_class.values()) / len(classes) if classes else 0.0
    macro_r = sum(v[1] for v in per_class.values()) / len(classes) if classes else 0.0
    macro_f1 = sum(v[2] for v in per_class.values()) / len(classes) if classes else 0.0
    return accuracy, macro_p, macro_r, macro_f1, per_class, confusion


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate EcoLens AI on a labelled set.")
    parser.add_argument("--data-dir", type=Path, help="Directory of per-class subfolders.")
    parser.add_argument("--mapping", type=Path, help="JSON mapping class -> acceptable labels.")
    parser.add_argument(
        "--limit-per-class", type=int, default=0, help="Cap images per class (0 = all)."
    )
    parser.add_argument("--out", type=Path, help="Optional path to write the JSON report.")
    args = parser.parse_args()

    if not args.data_dir or not args.mapping:
        print("No dataset/mapping provided.\n")
        print("This project does not ship a waste dataset. To measure real metrics:")
        print("  1. Obtain a licensed waste-image dataset (see DATA_SOURCES.md).")
        print("  2. Arrange it as <data-dir>/<class>/<image>.")
        print("  3. Provide a mapping JSON (dataset class -> acceptable object labels).")
        print("  4. Re-run: python scripts/evaluate_model.py --data-dir ... --mapping ...")
        print("\nUntil then, no metrics are reported — metrics are never fabricated.")
        return 1

    if not args.data_dir.is_dir():
        print(f"Data directory not found: {args.data_dir}")
        return 1
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))

    labels: list[str] = []
    preds: list[str] = []
    latencies: list[float] = []
    skipped = 0

    for class_dir in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
        class_name = class_dir.name
        acceptable = {a.lower() for a in mapping.get(class_name, [])}
        if not acceptable:
            print(f"  ! skipping '{class_name}' (not in mapping)")
            continue
        images = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        if args.limit_per_class:
            images = images[: args.limit_per_class]

        for img_path in images:
            result = analyze_image_bytes(
                img_path.read_bytes(), filename=img_path.name, with_explanation=False
            )
            if not result.ok or result.top1 is None:
                skipped += 1
                continue
            latencies.append(result.inference_duration_ms)
            predicted = result.top1.class_name.lower()
            labels.append(class_name)
            # Credit the correct class if the top-1 label is acceptable, otherwise
            # record the raw top-1 label for the confusion matrix.
            preds.append(class_name if predicted in acceptable else result.top1.class_name)

    if not labels:
        print("No evaluable images found. Nothing measured.")
        return 1

    classes = sorted(set(labels))
    accuracy, macro_p, macro_r, macro_f1, per_class, confusion = _compute_metrics(
        labels, preds, classes
    )

    report = {
        "dataset_dir": str(args.data_dir),
        "mapping_file": str(args.mapping),
        "num_images_evaluated": len(labels),
        "num_skipped": skipped,
        "classes": classes,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            c: {
                "precision": round(v[0], 4),
                "recall": round(v[1], 4),
                "f1": round(v[2], 4),
            }
            for c, v in per_class.items()
        },
        "confusion_matrix": {t: dict(row) for t, row in confusion.items()},
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 2),
            "p50": round(_percentile(latencies, 0.50), 2),
            "p95": round(_percentile(latencies, 0.95), 2),
        },
        "note": (
            "Metrics reflect agreement of an ImageNet object classifier with the "
            "provided mapping on this test set. They are not validation of a "
            "dedicated waste-material classifier."
        ),
    }

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
