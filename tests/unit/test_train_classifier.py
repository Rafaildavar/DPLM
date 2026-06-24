from pathlib import Path

import numpy as np

from cv.train_classifier import build_classifier, load_dataset


def _write_sample(root: Path, label: str, index: int, value: float = 0.0) -> None:
    label_dir = root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        label_dir / f"sample_{index:04d}.npy",
        np.full((3, 21, 2), value, dtype=np.float32),
    )


def test_load_dataset_can_filter_and_normalize_labels(tmp_path):
    data_root = tmp_path / "gestures"
    _write_sample(data_root, "New", 0, value=1.0)
    _write_sample(data_root, "new2", 0, value=2.0)
    _write_sample(data_root, "zoom", 0, value=3.0)

    x, y, classes = load_dataset(
        data_root,
        expect_dim=42,
        include_labels=["new", "new2"],
        lowercase_labels=True,
    )

    assert x.shape == (2, 42)
    assert y.tolist() == [0, 1]
    assert classes == ["new", "new2"]


def test_load_dataset_supports_dynamic_feature_mode(tmp_path):
    data_root = tmp_path / "gestures"
    _write_sample(data_root, "swipe", 0, value=0.0)
    _write_sample(data_root, "circle", 0, value=1.0)

    x, y, classes = load_dataset(
        data_root,
        include_labels=["swipe", "circle"],
        feature_mode="dynamic_stats",
    )

    assert x.shape == (2, 42 * 6)
    assert y.tolist() == [0, 1]
    assert classes == ["circle", "swipe"]


def test_build_classifier_supports_non_knn_models():
    clf = build_classifier("extra_trees", random_state=7)

    assert clf.__class__.__name__ == "ExtraTreesClassifier"
