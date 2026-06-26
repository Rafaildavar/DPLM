from pathlib import Path

import numpy as np

from scripts.dynamic_data_analysis import analyze_dynamic_data, build_markdown_report


def _write_motion_sample(
    root: Path,
    label: str,
    index: int,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    label_dir = root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    frames = 8
    arr = np.zeros((frames, 44), dtype=np.float32)
    arr[:, 42] = np.linspace(start[0], end[0], frames)
    arr[:, 43] = np.linspace(start[1], end[1], frames)
    np.save(label_dir / f"sample_{index:04d}.npy", arr)


def test_dynamic_data_analysis_reports_motion_quality(tmp_path):
    data_root = tmp_path / "gestures"
    for idx in range(3):
        offset = idx * 0.01
        _write_motion_sample(
            data_root,
            "swipe_down",
            idx,
            start=(0.5 + offset, 0.2),
            end=(0.5 + offset, 0.8),
        )
        _write_motion_sample(
            data_root,
            "swipe_up",
            idx,
            start=(0.5 + offset, 0.8),
            end=(0.5 + offset, 0.2),
        )

    report = analyze_dynamic_data(
        data_root,
        include_prefix="swipe_",
        live_log=None,
        min_samples_per_class=2,
        max_folds=2,
    )

    assert report.sample_count == 6
    assert report.class_counts == {"swipe_down": 3, "swipe_up": 3}
    assert report.prediction is not None
    assert report.prediction.accuracy is not None
    assert report.feature_relevance
    assert report.classes[0].direction_ok_rate == 1.0

    markdown = build_markdown_report(report)
    assert "# Dynamic Gesture Data Analysis" in markdown
    assert "Решения по предобработке" in markdown
    assert "`swipe_down`" in markdown
