"""Evaluate gesture classifier metrics on recorded .npy samples.

The project trains a KNN classifier on hand-landmark recordings. This script
reuses the same feature preprocessing as cv/train_classifier.py and reports
cross-validation metrics for the baseline and a few preprocessing variants.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    y: np.ndarray
    classes: list[str]
    skipped_files: list[str]


def _copy_with_timeout(source: Path, target: Path, timeout: float) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["cp", str(source), str(target)], check=True, timeout=timeout)
    except Exception:
        target.unlink(missing_ok=True)
        return False
    return True


def _iter_sample_files(data_root: Path) -> Iterable[tuple[str, Path]]:
    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for sample_path in sorted(label_dir.glob("sample_*.npy")):
            yield label_dir.name, sample_path


def stage_dataset(data_root: Path, copy_timeout: float) -> tuple[Path, list[str]]:
    """Copy samples to a temp dir so stalled local files can be skipped."""
    temp_root = Path(tempfile.mkdtemp(prefix="dplm_metrics_"))
    staged_root = temp_root / "gestures"
    skipped: list[str] = []

    for label, source in _iter_sample_files(data_root):
        target = staged_root / label / source.name
        if not _copy_with_timeout(source, target, copy_timeout):
            skipped.append(str(source))

    return staged_root, skipped


def load_dataset(data_root: Path, copy_timeout: float = 3.0) -> Dataset:
    staged_root, skipped_files = stage_dataset(data_root, copy_timeout=copy_timeout)
    features: list[np.ndarray] = []
    labels: list[int] = []
    classes: list[str] = []

    for label_dir in sorted(path for path in staged_root.iterdir() if path.is_dir()):
        class_idx = len(classes)
        classes.append(label_dir.name)
        for sample_path in sorted(label_dir.glob("sample_*.npy")):
            try:
                arr = np.load(sample_path)
                if arr.ndim == 3:
                    arr = arr.reshape(arr.shape[0], -1)
                elif arr.ndim != 2:
                    raise ValueError(f"unexpected shape {arr.shape}")
                features.append(arr.mean(axis=0).astype(np.float32, copy=False))
                labels.append(class_idx)
            except Exception:
                skipped_files.append(str(sample_path))

    shutil.rmtree(staged_root.parent, ignore_errors=True)
    if not features:
        raise RuntimeError(f"No readable samples found in {data_root}")

    target_dim = max(feature.shape[0] for feature in features)
    aligned = []
    for feature in features:
        if feature.shape[0] > target_dim:
            aligned.append(feature[:target_dim])
        elif feature.shape[0] < target_dim:
            aligned.append(np.pad(feature, (0, target_dim - feature.shape[0])))
        else:
            aligned.append(feature)

    return Dataset(
        x=np.stack(aligned, axis=0),
        y=np.asarray(labels, dtype=np.int64),
        classes=classes,
        skipped_files=skipped_files,
    )


def stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for class_idx in sorted(set(y.tolist())):
        indices = np.where(y == class_idx)[0].tolist()
        rng.shuffle(indices)
        for offset, sample_idx in enumerate(indices):
            folds[offset % n_splits].append(sample_idx)
    return [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]


def transform_features(x: np.ndarray, variant: str, train_idx: np.ndarray | None = None) -> np.ndarray:
    if variant == "baseline":
        return x.copy()
    if variant == "sample_mean_center":
        return x - x.mean(axis=1, keepdims=True)
    if variant == "sample_l2_normalize":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms
    if variant == "train_standardize":
        if train_idx is None:
            raise ValueError("train_idx is required for train_standardize")
        mean = x[train_idx].mean(axis=0)
        std = x[train_idx].std(axis=0)
        std[std == 0] = 1.0
        return (x - mean) / std
    raise ValueError(f"Unknown variant: {variant}")


def predict_knn(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, neighbors: int) -> np.ndarray:
    predictions: list[int] = []
    k = min(neighbors, len(y_train))
    for sample in x_test:
        distances = ((x_train - sample) ** 2).sum(axis=1)
        nearest = np.argsort(distances)[:k]
        values, counts = np.unique(y_train[nearest], return_counts=True)
        predictions.append(int(values[np.argmax(counts)]))
    return np.asarray(predictions, dtype=np.int64)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> dict[str, object]:
    accuracy = float((y_true == y_pred).mean())
    per_class = []
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for class_idx in range(class_count):
        tp = int(((y_true == class_idx) & (y_pred == class_idx)).sum())
        fp = int(((y_true != class_idx) & (y_pred == class_idx)).sum())
        fn = int(((y_true == class_idx) & (y_pred != class_idx)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int((y_true == class_idx).sum())
        per_class.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision / class_count,
        "macro_recall": macro_recall / class_count,
        "macro_f1": macro_f1 / class_count,
        "per_class": per_class,
    }


def evaluate_variant(dataset: Dataset, variant: str, folds: int, neighbors: int, seed: int) -> dict[str, object]:
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    all_indices = np.arange(len(dataset.y))

    for test_idx in stratified_folds(dataset.y, n_splits=folds, seed=seed):
        train_idx = np.setdiff1d(all_indices, test_idx)
        x_transformed = transform_features(dataset.x, variant, train_idx=train_idx)
        predictions = predict_knn(
            x_transformed[train_idx],
            dataset.y[train_idx],
            x_transformed[test_idx],
            neighbors=neighbors,
        )
        y_true_all.extend(dataset.y[test_idx].tolist())
        y_pred_all.extend(predictions.tolist())

    return classification_metrics(
        np.asarray(y_true_all, dtype=np.int64),
        np.asarray(y_pred_all, dtype=np.int64),
        class_count=len(dataset.classes),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate gesture KNN metrics.")
    parser.add_argument("--data-root", default="data/gestures", type=Path)
    parser.add_argument("--out", default="docs/gesture_metrics_results.json", type=Path)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--neighbors", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--copy-timeout", default=3.0, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.data_root, copy_timeout=args.copy_timeout)
    variants = ["baseline", "train_standardize", "sample_mean_center", "sample_l2_normalize"]

    results = {
        "data_root": str(args.data_root),
        "sample_count": int(len(dataset.y)),
        "feature_dim": int(dataset.x.shape[1]),
        "classes": dataset.classes,
        "class_counts": {
            class_name: int((dataset.y == class_idx).sum())
            for class_idx, class_name in enumerate(dataset.classes)
        },
        "skipped_file_count": len(dataset.skipped_files),
        "skipped_files": dataset.skipped_files,
        "folds": args.folds,
        "neighbors": args.neighbors,
        "variants": {
            variant: evaluate_variant(dataset, variant, args.folds, args.neighbors, args.seed)
            for variant in variants
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
