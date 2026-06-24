from pathlib import Path

import numpy as np

from scripts.compare_models import build_markdown_report, compare_models


def _write_sequence(root: Path, label: str, index: int, sequence: np.ndarray) -> None:
    label_dir = root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    np.save(label_dir / f"sample_{index:04d}.npy", sequence.astype(np.float32))


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
    assert "# JMLC Model Comparison" in markdown
    assert "Рекомендация" in markdown

