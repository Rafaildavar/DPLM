import json
from pathlib import Path

import numpy as np

from scripts.compare_models import build_markdown_report, compare_models


def _write_sequence(root: Path, label: str, index: int, sequence: np.ndarray) -> None:
    label_dir = root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    np.save(label_dir / f"sample_{index:04d}.npy", sequence.astype(np.float32))


def _write_augmented_sequence(
    root: Path,
    label: str,
    source_index: int,
    variant_index: int,
    sequence: np.ndarray,
) -> None:
    label_dir = root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    sample_path = label_dir / f"aug_sample_{source_index:04d}_{variant_index:02d}.npy"
    np.save(sample_path, sequence.astype(np.float32))
    sample_path.with_suffix(".meta.json").write_text(
        json.dumps({"transform": "gislr_landmark_v1"}),
        encoding="utf-8",
    )


def test_compare_models_builds_report_for_small_dataset(tmp_path):
    data_root = tmp_path / "gestures"
    for idx in range(6):
        _write_sequence(
            data_root,
            "open",
            idx,
            np.full((4, 4), float(idx) * 0.01, dtype=np.float32),
        )
        _write_sequence(
            data_root,
            "close",
            idx,
            np.full((4, 4), 1.0 + float(idx) * 0.01, dtype=np.float32),
        )

    report = compare_models(
        data_root=data_root,
        feature_modes=["static_mean", "dynamic_stats"],
        model_names=["knn"],
        min_samples_per_class=2,
        max_folds=3,
        motion_threshold=0.01,
    )

    assert report.dataset.sample_count == 12
    assert report.dataset.class_count == 2
    assert report.best_overall.model_name == "knn"
    assert report.results

    markdown = build_markdown_report(report)
    assert "# GestureBind Cross-Validation Model Comparison" in markdown
    assert "Рекомендация" in markdown


def test_compare_models_can_include_gislr_augmented_and_lowercase_labels(tmp_path):
    data_root = tmp_path / "gestures"
    for idx in range(3):
        _write_sequence(
            data_root,
            "Open",
            idx,
            np.full((4, 4), float(idx) * 0.01, dtype=np.float32),
        )
        _write_sequence(
            data_root,
            "Close",
            idx,
            np.full((4, 4), 1.0 + float(idx) * 0.01, dtype=np.float32),
        )
    _write_augmented_sequence(
        data_root,
        "Open",
        0,
        0,
        np.full((4, 4), 0.02, dtype=np.float32),
    )
    _write_augmented_sequence(
        data_root,
        "Close",
        0,
        0,
        np.full((4, 4), 1.02, dtype=np.float32),
    )

    report = compare_models(
        data_root=data_root,
        feature_modes=["static_mean"],
        model_names=["knn"],
        min_samples_per_class=2,
        max_folds=2,
        include_augmented=True,
        include_labels=["open", "close"],
        lowercase_labels=True,
    )

    assert report.dataset.sample_count == 8
    assert report.dataset.include_augmented is True
    assert report.dataset.class_counts == {"close": 4, "open": 4}
    assert report.dataset.group_count == 6
    assert report.dataset.grouped_cv is True

    markdown = build_markdown_report(report)
    assert "| Augmented samples | included |" in markdown
