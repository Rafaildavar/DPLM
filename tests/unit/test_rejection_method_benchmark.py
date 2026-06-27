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
