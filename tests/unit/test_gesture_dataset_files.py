import numpy as np

from cv.gesture_dataset_files import (
    augmented_sample_path,
    gesture_sample_paths,
    stable_sample_index,
)
from cv.train_classifier import load_dataset


def test_gesture_sample_paths_ignore_augmented_samples_by_default(tmp_path):
    label_dir = tmp_path / "Wave"
    label_dir.mkdir()
    real = label_dir / "sample_0000.npy"
    aug = label_dir / "aug_sample_0000_00.npy"
    later = label_dir / "sample_0001.npy"
    for path in (aug, later, real):
        np.save(path, np.zeros((3, 21, 2), dtype=np.float32))

    assert gesture_sample_paths(label_dir) == [real, later]
    assert gesture_sample_paths(label_dir, include_augmented=True) == [real, later, aug]
    assert augmented_sample_path(real, 2).name == "aug_sample_0000_02.npy"
    assert stable_sample_index(real) == 0
    assert stable_sample_index(aug) == 1_000_000


def test_train_classifier_ignores_augmented_samples_by_default(tmp_path):
    label_dir = tmp_path / "gestures" / "Wave"
    label_dir.mkdir(parents=True)
    np.save(label_dir / "sample_0000.npy", np.zeros((30, 21, 2), dtype=np.float32))
    np.save(label_dir / "aug_sample_0000_00.npy", np.ones((30, 21, 2), dtype=np.float32))

    X, y, classes = load_dataset(tmp_path / "gestures", expect_dim=42)

    assert X.shape == (1, 42)
    assert y.tolist() == [0]
    assert classes == ["Wave"]
