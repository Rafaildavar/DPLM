from pathlib import Path

import numpy as np

from scripts.jmlc_dataset_profile import build_markdown, build_profile


def _write_sample(root: Path, label: str, index: int, arr: np.ndarray) -> None:
    label_dir = root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    np.save(label_dir / f"sample_{index:04d}.npy", arr)


def test_build_profile_summarizes_classes_and_risks(tmp_path):
    data_root = tmp_path / "gestures"
    _write_sample(
        data_root,
        "open",
        0,
        np.zeros((30, 21, 2), dtype=np.float32),
    )
    _write_sample(
        data_root,
        "open",
        1,
        np.ones((40, 21, 2), dtype=np.float32),
    )
    _write_sample(
        data_root,
        "zoom",
        0,
        np.ones((20, 84), dtype=np.float32),
    )
    (data_root / "empty").mkdir()

    profile = build_profile(data_root)

    assert profile.class_count == 3
    assert profile.active_class_count == 2
    assert profile.empty_class_count == 1
    assert profile.sample_count == 3
    assert profile.valid_sample_count == 3
    assert profile.frame_min == 20
    assert profile.frame_max == 40
    assert profile.feature_dims == [42, 84]
    assert "active_class_count_below_target" in profile.risks
    assert "empty_classes:empty" in profile.risks
    assert "mixed_feature_dimensions:42,84" in profile.risks
    assert any(risk.startswith("low_samples_per_class:") for risk in profile.risks)


def test_build_markdown_contains_jmlc_interpretation(tmp_path):
    data_root = tmp_path / "gestures"
    _write_sample(
        data_root,
        "open",
        0,
        np.zeros((30, 21, 2), dtype=np.float32),
    )

    markdown = build_markdown(build_profile(data_root))

    assert "# JMLC Dataset Profile" in markdown
    assert "`open`" in markdown
    assert "macro_f1" in markdown

