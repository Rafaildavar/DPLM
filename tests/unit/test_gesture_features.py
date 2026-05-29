import numpy as np

from cv.gesture_features import (
    FEATURE_DYNAMIC_STATS,
    FEATURE_STATIC_MEAN,
    align_sequence,
    build_feature_matrix,
    build_feature_vector,
    infer_target_dim,
    sequence_to_matrix,
)


def test_sequence_to_matrix_flattens_landmarks():
    raw = np.zeros((5, 21, 2), dtype=np.float32)

    matrix = sequence_to_matrix(raw)

    assert matrix.shape == (5, 42)
    assert matrix.dtype == np.float32


def test_align_sequence_pads_and_truncates_features():
    sequence = np.ones((3, 42), dtype=np.float32)

    padded = align_sequence(sequence, 84)
    truncated = align_sequence(padded, 42)

    assert padded.shape == (3, 84)
    assert np.allclose(padded[:, :42], 1.0)
    assert np.allclose(padded[:, 42:], 0.0)
    assert truncated.shape == (3, 42)


def test_dynamic_stats_has_expected_size_and_motion_signal():
    sequence = np.stack(
        [
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.full(4, 3.0, dtype=np.float32),
        ],
        axis=0,
    )

    static_features = build_feature_vector(sequence, FEATURE_STATIC_MEAN, target_dim=4)
    dynamic_features = build_feature_vector(sequence, FEATURE_DYNAMIC_STATS, target_dim=4)

    assert static_features.shape == (4,)
    assert dynamic_features.shape == (32,)
    assert np.allclose(dynamic_features[16:20], 3.0)


def test_build_feature_matrix_returns_sorted_labels():
    class Record:
        def __init__(self, label, sequence):
            self.label = label
            self.sequence = sequence

    records = [
        Record("b", np.ones((2, 2), dtype=np.float32)),
        Record("a", np.zeros((2, 2), dtype=np.float32)),
    ]

    X, y, labels = build_feature_matrix(records, FEATURE_STATIC_MEAN, target_dim=2)

    assert X.shape == (2, 2)
    assert y.tolist() == [1, 0]
    assert labels == ["a", "b"]
    assert infer_target_dim(record.sequence for record in records) == 2
