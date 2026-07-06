import numpy as np

from cv.gesture_augmentation import augment_gislr_landmark_sequence


def test_gislr_augmentation_preserves_static_sample_shape_and_seed():
    sample = np.zeros((12, 21, 2), dtype=np.float32)
    sample[:, :, 0] = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    sample[:, :, 1] = np.linspace(0.0, 0.5, 21, dtype=np.float32)

    aug1, meta1 = augment_gislr_landmark_sequence(sample, seed=123)
    aug2, meta2 = augment_gislr_landmark_sequence(sample, seed=123)

    assert aug1.shape == sample.shape
    assert aug1.dtype == np.float32
    assert np.allclose(aug1, aug2)
    assert meta1 == meta2
    assert meta1["policy"] == "gislr_landmark_v1"
    assert not np.allclose(aug1, sample)


def test_gislr_augmentation_keeps_dynamic_global_direction_sign():
    sample = np.zeros((20, 44), dtype=np.float32)
    sample[:, :42] = 0.1
    sample[:, 42] = np.linspace(0.8, 0.2, sample.shape[0], dtype=np.float32)
    sample[:, 43] = 0.4

    augmented, metadata = augment_gislr_landmark_sequence(
        sample,
        seed=7,
        include_global_motion=True,
    )

    assert augmented.shape == sample.shape
    assert augmented[-1, 42] - augmented[0, 42] < 0.0
    assert metadata["include_global_motion"] is True
    assert any("global_motion" in item for item in metadata["applied"])
