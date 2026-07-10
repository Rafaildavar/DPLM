import json
from pathlib import Path

import numpy as np

from scripts.rejection_method_benchmark import (
    benchmark_rejection_methods,
    build_markdown_report,
)


def _write_sequence(root: Path, label: str, index: int, value: float) -> None:
    label_dir = root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(index)
    sequence = np.full((5, 4), value, dtype=np.float32)
    sequence += rng.normal(0.0, 0.005, size=sequence.shape).astype(np.float32)
    np.save(label_dir / f"sample_{index:04d}.npy", sequence)


def _write_augmented_sequence(root: Path, label: str, index: int, value: float) -> None:
    label_dir = root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    sequence = np.full((5, 4), value, dtype=np.float32)
    sample_path = label_dir / f"aug_sample_{index:04d}_00.npy"
    np.save(sample_path, sequence)
    sample_path.with_suffix(".meta.json").write_text(
        json.dumps({"transform": "gislr_landmark_v1"}),
        encoding="utf-8",
    )


def _write_taxonomy(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "default_type": "static",
                "types": {
                    "static": ["gun", "three"],
                    "dynamic": [],
                    "negative": ["no_gesture_static"],
                },
                "patterns": {
                    "negative": ["no_gesture*"],
                },
            }
        ),
        encoding="utf-8",
    )


def test_rejection_method_benchmark_compares_methods(tmp_path):
    data_root = tmp_path / "gestures"
    taxonomy_path = tmp_path / "taxonomy.json"
    _write_taxonomy(taxonomy_path)
    for index in range(8):
        _write_sequence(data_root, "gun", index, 0.0)
        _write_sequence(data_root, "three", index, 1.0)
        _write_sequence(data_root, "no_gesture_static", index, 2.0)

    report = benchmark_rejection_methods(
        data_root=data_root,
        taxonomy_path=taxonomy_path,
        scope="static",
        feature_mode="static_mean",
        target_dim=4,
        methods=[
            "negative_classes",
            "open_set_policy",
            "one_vs_rest_logreg",
            "metric_nca_centroid",
        ],
        max_folds=2,
        random_state=7,
    )

    assert report.dataset.sample_count == 24
    assert report.dataset.positive_labels == ["gun", "three"]
    assert report.dataset.negative_labels == ["no_gesture_static"]
    assert report.best_method
    assert {result.method for result in report.methods} == {
        "negative_classes",
        "open_set_policy",
        "one_vs_rest_logreg",
        "metric_nca_centroid",
    }
    assert all(result.total == 24 for result in report.methods)

    markdown = build_markdown_report(report)
    assert "# Rejection Method Benchmark" in markdown
    assert "negative_classes" in markdown
    assert "Recommendation" in markdown


def test_rejection_method_benchmark_supports_extra_trees_augmented_lowercase(tmp_path):
    data_root = tmp_path / "gestures"
    taxonomy_path = tmp_path / "taxonomy.json"
    _write_taxonomy(taxonomy_path)
    for index in range(3):
        _write_sequence(data_root, "Gun", index, 0.0)
        _write_sequence(data_root, "Three", index, 1.0)
        _write_sequence(data_root, "No_Gesture_Static", index, 2.0)
    _write_augmented_sequence(data_root, "Gun", 0, 0.01)
    _write_augmented_sequence(data_root, "Three", 0, 1.01)
    _write_augmented_sequence(data_root, "No_Gesture_Static", 0, 2.01)

    report = benchmark_rejection_methods(
        data_root=data_root,
        taxonomy_path=taxonomy_path,
        scope="static",
        feature_mode="static_mean",
        candidate_model="extra_trees",
        target_dim=4,
        methods=["negative_classes"],
        include_labels=["gun", "three", "no_gesture_static"],
        include_augmented=True,
        lowercase_labels=True,
        min_samples_per_class=2,
        max_folds=2,
        random_state=7,
    )

    assert report.dataset.sample_count == 12
    assert report.dataset.candidate_model == "extra_trees"
    assert report.dataset.include_augmented is True
    assert report.dataset.group_count == 9
    assert report.dataset.grouped_cv is True
    assert report.methods[0].negative_label_metrics

    markdown = build_markdown_report(report)
    assert "candidate model: `extra_trees`" in markdown
    assert "Best Method Negative Breakdown" in markdown
