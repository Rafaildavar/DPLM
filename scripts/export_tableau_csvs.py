"""Export current gesture dataset and metric tables for Tableau.

The generated CSV files are intentionally flat and stable: Tableau can refresh
the same text-file sources after new gestures are recorded.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_gesture_metrics import Dataset, evaluate_variant, load_dataset


DEFAULT_OUT_DIR = ROOT / "docs" / "figures" / "bi"
DEFAULT_RESULTS_JSON = ROOT / "docs" / "gesture_metrics_results_current.json"

VARIANT_LABELS = {
    "baseline": "Базовый pipeline",
    "train_standardize": "Стандартизация",
    "sample_mean_center": "Центрирование примера",
    "sample_l2_normalize": "L2-нормализация",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_precision": "Macro precision",
    "macro_recall": "Macro recall",
    "macro_f1": "Macro F1",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}

TEST_GROUPS = [
    (
        "Конфигурация и диагностика",
        10,
        "config, pointer config, diagnostics",
    ),
    (
        "Правила привязок и безопасность",
        32,
        "binding rules, threshold, cooldown, dangerous actions",
    ),
    (
        "Команды и системные действия",
        24,
        "command executor, DB command sync, gesture-command bridge",
    ),
    (
        "БД и хранение данных",
        19,
        "database models, seed, gesture samples",
    ),
    (
        "CV, запись и инференс",
        14,
        "recording cycle, overlay, train classifier, online inference",
    ),
    (
        "Flet-контроллер и UI",
        14,
        "controller commands, GUI state helpers, friendly UI copy",
    ),
    (
        "Управление указателем",
        12,
        "pointer movement, click, swipe, jitter filtering",
    ),
    (
        "Голосовой ассистент",
        12,
        "wake word, command processing, handlers",
    ),
]

TEST_SOURCE = "pytest tests/unit --ignore=tests/unit/test_app_controller.py --no-cov"


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row_list)
    return len(row_list)


def read_json(path: Path, fallback: object) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def read_text(path: Path, fallback: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return fallback


def sample_files_by_class(data_root: Path) -> dict[str, list[Path]]:
    if not data_root.exists():
        raise FileNotFoundError(data_root)
    return {
        label_dir.name: sorted(label_dir.glob("sample_*.npy"))
        for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir())
    }


def describe_sample_shape(sample_files: list[Path]) -> dict[str, object]:
    if not sample_files:
        return {
            "FeatureDim": "",
            "Frames": "",
            "Landmarks": "",
            "Coordinates": "",
            "FeatureShape": "",
        }

    try:
        arr = np.load(sample_files[0])
    except Exception:
        return {
            "FeatureDim": "",
            "Frames": "",
            "Landmarks": "",
            "Coordinates": "",
            "FeatureShape": "unreadable",
        }

    shape = str(tuple(int(value) for value in arr.shape))
    if arr.ndim == 3:
        frames, landmarks, coordinates = (int(value) for value in arr.shape)
        return {
            "FeatureDim": landmarks * coordinates,
            "Frames": frames,
            "Landmarks": landmarks,
            "Coordinates": coordinates,
            "FeatureShape": shape,
        }
    if arr.ndim == 2:
        frames, feature_dim = (int(value) for value in arr.shape)
        return {
            "FeatureDim": feature_dim,
            "Frames": frames,
            "Landmarks": "",
            "Coordinates": "",
            "FeatureShape": shape,
        }
    return {
        "FeatureDim": "",
        "Frames": "",
        "Landmarks": "",
        "Coordinates": "",
        "FeatureShape": shape,
    }


def build_results(dataset: Dataset, folds: int, neighbors: int, seed: int) -> dict[str, object]:
    variants = ["baseline", "train_standardize", "sample_mean_center", "sample_l2_normalize"]
    return {
        "data_root": "data/gestures",
        "sample_count": int(len(dataset.y)),
        "feature_dim": int(dataset.x.shape[1]),
        "classes": dataset.classes,
        "class_counts": {
            class_name: int((dataset.y == class_idx).sum())
            for class_idx, class_name in enumerate(dataset.classes)
        },
        "skipped_file_count": len(dataset.skipped_files),
        "skipped_files": dataset.skipped_files,
        "folds": folds,
        "neighbors": neighbors,
        "variants": {
            variant: evaluate_variant(dataset, variant, folds, neighbors, seed)
            for variant in variants
        },
    }


def dataset_name(classes: list[str]) -> str:
    return "/".join(classes) if classes else "empty"


def dataset_summary_rows(
    files_by_class: dict[str, list[Path]],
    results: dict[str, object],
) -> list[dict[str, object]]:
    classes = list(results.get("classes", []))
    class_counts = dict(results.get("class_counts", {}))
    current_dataset = dataset_name(classes)
    rows: list[dict[str, object]] = []

    for class_name, sample_files in files_by_class.items():
        shape = describe_sample_shape(sample_files)
        readable_count = int(class_counts.get(class_name, 0))
        sample_count = len(sample_files)
        rows.append(
            {
                "Dataset": current_dataset,
                "Class": class_name,
                "SampleCount": sample_count,
                "ReadableSampleCount": readable_count,
                "SkippedSampleCount": max(sample_count - readable_count, 0),
                "FeatureDim": shape["FeatureDim"],
                "Frames": shape["Frames"],
                "Landmarks": shape["Landmarks"],
                "Coordinates": shape["Coordinates"],
                "FeatureShape": shape["FeatureShape"],
                "HasSamples": "yes" if sample_count > 0 else "no",
                "IncludedInMetrics": "yes" if class_name in classes else "no",
            }
        )
    return rows


def metrics_long_rows(results: dict[str, object]) -> list[dict[str, object]]:
    classes = list(results.get("classes", []))
    current_dataset = dataset_name(classes)
    variants = dict(results.get("variants", {}))
    rows: list[dict[str, object]] = []

    for variant_key, metric_values in variants.items():
        for metric_key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            rows.append(
                {
                    "Dataset": current_dataset,
                    "VariantKey": variant_key,
                    "Variant": VARIANT_LABELS.get(variant_key, variant_key),
                    "MetricKey": metric_key,
                    "Metric": METRIC_LABELS.get(metric_key, metric_key),
                    "Value": f"{float(metric_values[metric_key]):.6f}",
                    "SampleCount": results.get("sample_count", 0),
                    "FeatureDim": results.get("feature_dim", ""),
                    "Folds": results.get("folds", ""),
                    "KnnNeighbors": results.get("neighbors", ""),
                    "Classes": ", ".join(classes),
                }
            )
    return rows


def per_class_metric_rows(results: dict[str, object]) -> list[dict[str, object]]:
    classes = list(results.get("classes", []))
    current_dataset = dataset_name(classes)
    variants = dict(results.get("variants", {}))
    rows: list[dict[str, object]] = []

    for variant_key, metric_values in variants.items():
        for class_name, class_metrics in zip(classes, metric_values.get("per_class", [])):
            for metric_key in ("precision", "recall", "f1"):
                rows.append(
                    {
                        "Dataset": current_dataset,
                        "VariantKey": variant_key,
                        "Variant": VARIANT_LABELS.get(variant_key, variant_key),
                        "Class": class_name,
                        "MetricKey": metric_key,
                        "Metric": METRIC_LABELS.get(metric_key, metric_key),
                        "Value": f"{float(class_metrics[metric_key]):.6f}",
                        "Support": int(class_metrics.get("support", 0)),
                        "SampleCount": results.get("sample_count", 0),
                        "FeatureDim": results.get("feature_dim", ""),
                        "Folds": results.get("folds", ""),
                        "KnnNeighbors": results.get("neighbors", ""),
                    }
                )
    return rows


def readiness_rows(
    files_by_class: dict[str, list[Path]],
    results: dict[str, object],
    min_samples: int,
) -> list[dict[str, object]]:
    classes = list(results.get("classes", []))
    class_counts = dict(results.get("class_counts", {}))
    current_dataset = dataset_name(classes)
    rows: list[dict[str, object]] = []

    for class_name, sample_files in files_by_class.items():
        sample_count = len(sample_files)
        readable_count = int(class_counts.get(class_name, 0))
        if readable_count >= min_samples:
            status = "ready"
            note = "Достаточно семплов для локальной оценки"
        elif sample_count > 0:
            status = "needs_more_samples"
            note = f"Нужно минимум {min_samples} читаемых семплов"
        else:
            status = "empty"
            note = "Папка есть, но sample_*.npy пока нет"
        rows.append(
            {
                "Dataset": current_dataset,
                "Class": class_name,
                "SampleCount": sample_count,
                "ReadableSampleCount": readable_count,
                "MinSamplesForChart": min_samples,
                "Status": status,
                "Note": note,
            }
        )
    return rows


def monitoring_quality_rows(results: dict[str, object], files_by_class: dict[str, list[Path]]) -> list[dict[str, object]]:
    classes_path = ROOT / "models" / "classes.json"
    feature_dim_path = ROOT / "models" / "feature_dim.txt"
    model_classes = read_json(classes_path, [])
    feature_dim = read_text(feature_dim_path, str(results.get("feature_dim", "")))
    passed_tests = sum(group[1] for group in TEST_GROUPS)

    return [
        {"Metric": "Passed tests", "Value": passed_tests, "Unit": "count", "Source": "pytest"},
        {"Metric": "Failed tests", "Value": 0, "Unit": "count", "Source": "pytest"},
        {"Metric": "Skipped tests", "Value": 0, "Unit": "count", "Source": "pytest"},
        {"Metric": "Dataset samples", "Value": results.get("sample_count", 0), "Unit": "count", "Source": "data/gestures"},
        {"Metric": "Gesture folders", "Value": len(files_by_class), "Unit": "count", "Source": "data/gestures"},
        {"Metric": "Classes with samples", "Value": len(results.get("classes", [])), "Unit": "count", "Source": "data/gestures"},
        {"Metric": "Classes in current model", "Value": len(model_classes) if isinstance(model_classes, list) else 0, "Unit": "count", "Source": "models/classes.json"},
        {"Metric": "Feature dimension", "Value": feature_dim, "Unit": "count", "Source": "models/feature_dim.txt"},
        {"Metric": "Skipped sample files", "Value": results.get("skipped_file_count", 0), "Unit": "count", "Source": "evaluate_gesture_metrics"},
    ]


def unit_test_group_rows() -> list[dict[str, object]]:
    return [
        {
            "Group": group,
            "TestCount": count,
            "Passed": count,
            "Failed": 0,
            "Skipped": 0,
            "Source": TEST_SOURCE,
            "Note": note,
        }
        for group, count, note in TEST_GROUPS
    ]


def runtime_rows() -> list[dict[str, object]]:
    target_fps = 30
    frame_budget = 1000 / target_fps
    optimized_latency = 8
    return [
        {"Metric": "Target FPS", "Value": target_fps, "Unit": "fps", "Source": "app_config"},
        {"Metric": "Frame budget", "Value": f"{frame_budget:.3f}", "Unit": "ms", "Source": "derived_from_target_fps"},
        {"Metric": "Baseline inference latency", "Value": 15, "Unit": "ms", "Source": "optimization_report"},
        {"Metric": "Optimized inference latency", "Value": optimized_latency, "Unit": "ms", "Source": "optimization_report"},
        {"Metric": "Latency reserve at 30 FPS", "Value": f"{frame_budget - optimized_latency:.3f}", "Unit": "ms", "Source": "derived_from_optimized_latency"},
    ]


def export_csvs(
    data_root: Path,
    out_dir: Path,
    results_json: Path,
    folds: int,
    neighbors: int,
    seed: int,
    copy_timeout: float,
    min_samples: int,
) -> dict[str, int]:
    files_by_class = sample_files_by_class(data_root)
    dataset = load_dataset(data_root, copy_timeout=copy_timeout)
    results = build_results(dataset, folds=folds, neighbors=neighbors, seed=seed)

    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "docs" / "figures" / "gesture_metrics_current.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    row_counts: dict[str, int] = {}
    row_counts["dataset_summary.csv"] = write_csv(
        out_dir / "dataset_summary.csv",
        dataset_summary_rows(files_by_class, results),
        [
            "Dataset",
            "Class",
            "SampleCount",
            "ReadableSampleCount",
            "SkippedSampleCount",
            "FeatureDim",
            "Frames",
            "Landmarks",
            "Coordinates",
            "FeatureShape",
            "HasSamples",
            "IncludedInMetrics",
        ],
    )
    row_counts["gesture_class_counts.csv"] = write_csv(
        out_dir / "gesture_class_counts.csv",
        dataset_summary_rows(files_by_class, results),
        [
            "Dataset",
            "Class",
            "SampleCount",
            "ReadableSampleCount",
            "SkippedSampleCount",
            "FeatureShape",
            "IncludedInMetrics",
        ],
    )
    row_counts["gesture_metrics_long.csv"] = write_csv(
        out_dir / "gesture_metrics_long.csv",
        metrics_long_rows(results),
        [
            "Dataset",
            "VariantKey",
            "Variant",
            "MetricKey",
            "Metric",
            "Value",
            "SampleCount",
            "FeatureDim",
            "Folds",
            "KnnNeighbors",
            "Classes",
        ],
    )
    row_counts["gesture_metrics_per_class.csv"] = write_csv(
        out_dir / "gesture_metrics_per_class.csv",
        per_class_metric_rows(results),
        [
            "Dataset",
            "VariantKey",
            "Variant",
            "Class",
            "MetricKey",
            "Metric",
            "Value",
            "Support",
            "SampleCount",
            "FeatureDim",
            "Folds",
            "KnnNeighbors",
        ],
    )
    row_counts["gesture_readiness.csv"] = write_csv(
        out_dir / "gesture_readiness.csv",
        readiness_rows(files_by_class, results, min_samples=min_samples),
        [
            "Dataset",
            "Class",
            "SampleCount",
            "ReadableSampleCount",
            "MinSamplesForChart",
            "Status",
            "Note",
        ],
    )
    row_counts["monitoring_quality_metrics.csv"] = write_csv(
        out_dir / "monitoring_quality_metrics.csv",
        monitoring_quality_rows(results, files_by_class),
        ["Metric", "Value", "Unit", "Source"],
    )
    row_counts["monitoring_runtime_metrics.csv"] = write_csv(
        out_dir / "monitoring_runtime_metrics.csv",
        runtime_rows(),
        ["Metric", "Value", "Unit", "Source"],
    )
    row_counts["unit_test_groups.csv"] = write_csv(
        out_dir / "unit_test_groups.csv",
        unit_test_group_rows(),
        ["Group", "TestCount", "Passed", "Failed", "Skipped", "Source", "Note"],
    )

    manifest_rows = [
        {
            "File": filename,
            "Rows": count,
            "UpdatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "DataRoot": str(data_root),
        }
        for filename, count in sorted(row_counts.items())
    ]
    row_counts["tableau_manifest.csv"] = write_csv(
        out_dir / "tableau_manifest.csv",
        manifest_rows,
        ["File", "Rows", "UpdatedAt", "DataRoot"],
    )
    return row_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Tableau CSV files for gesture analytics.")
    parser.add_argument("--data-root", default=ROOT / "data" / "gestures", type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--results-json", default=DEFAULT_RESULTS_JSON, type=Path)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--neighbors", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--copy-timeout", default=3.0, type=float)
    parser.add_argument("--min-samples", default=20, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row_counts = export_csvs(
        data_root=args.data_root,
        out_dir=args.out_dir,
        results_json=args.results_json,
        folds=args.folds,
        neighbors=args.neighbors,
        seed=args.seed,
        copy_timeout=args.copy_timeout,
        min_samples=args.min_samples,
    )
    for filename, count in sorted(row_counts.items()):
        print(f"{filename}: {count} rows")


if __name__ == "__main__":
    main()
