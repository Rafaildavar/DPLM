from pathlib import Path

import numpy as np

from scripts.threshold_report import (
    build_markdown,
    build_threshold_report,
    parse_candidates,
)


def _write_sequence(root: Path, label: str, index: int, sequence: np.ndarray) -> None:
    label_dir = root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    np.save(label_dir / f"sample_{index:04d}.npy", sequence.astype(np.float32))


def test_parse_candidates():
    assert parse_candidates("static_mean:knn, hybrid_stats:svm") == [
        ("static_mean", "knn"),
        ("hybrid_stats", "svm"),
    ]


def test_build_threshold_report_for_small_dataset(tmp_path):
    data_root = tmp_path / "gestures"
    for idx in range(6):
        _write_sequence(
            data_root,
            "left",
            idx,
            np.full((4, 4), float(idx) * 0.01, dtype=np.float32),
        )
        _write_sequence(
            data_root,
            "right",
            idx,
            np.full((4, 4), 1.0 + float(idx) * 0.01, dtype=np.float32),
        )

    report = build_threshold_report(
        data_root,
        candidates=[("static_mean", "knn")],
        min_samples_per_class=2,
        max_folds=3,
        min_accuracy=0.90,
        min_coverage=0.50,
    )

    assert report.dataset.sample_count == 12
    assert report.recommended_candidate.model_name == "knn"
    assert report.recommended_candidate.curve

    markdown = build_markdown(report)
    assert "# GestureBind Threshold Report" in markdown
    assert "Recommended Curve" in markdown
