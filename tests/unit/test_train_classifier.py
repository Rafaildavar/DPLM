from pathlib import Path

import numpy as np

from cv.train_classifier import load_dataset


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
