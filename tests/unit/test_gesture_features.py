from pathlib import Path

import numpy as np

from cv.gesture_features import (
    FEATURE_DYNAMIC_STATS,
    FEATURE_HYBRID_STATS,
    FEATURE_STATIC_MEAN,
    FEATURE_STATIC_STATS,
    DYNAMIC_TRAJECTORY_FEATURE_DIM,
    MOTION_DYNAMIC_LIKE,
    MOTION_STATIC_LIKE,
    align_sequence,
    build_feature_matrix,
    build_feature_vector,
    class_motion_profiles,
    infer_target_dim,
    sequence_motion_energy,
    sequence_to_matrix,
    trajectory_features,
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


def test_feature_modes_have_expected_sizes_and_motion_signal():
    sequence = np.stack(
        [
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.full(4, 3.0, dtype=np.float32),
        ],
        axis=0,
    )

    static_mean = build_feature_vector(sequence, FEATURE_STATIC_MEAN, target_dim=4)
    static_stats = build_feature_vector(sequence, FEATURE_STATIC_STATS, target_dim=4)
    dynamic_stats = build_feature_vector(sequence, FEATURE_DYNAMIC_STATS, target_dim=4)
    hybrid_stats = build_feature_vector(sequence, FEATURE_HYBRID_STATS, target_dim=4)

    assert static_mean.shape == (4,)
    assert static_stats.shape == (16,)
    assert dynamic_stats.shape == (24 + DYNAMIC_TRAJECTORY_FEATURE_DIM,)
    assert hybrid_stats.shape == (40 + DYNAMIC_TRAJECTORY_FEATURE_DIM,)
    assert np.allclose(dynamic_stats[:4], 3.0)
    assert sequence_motion_energy(sequence) > 0


def test_trajectory_features_capture_vertical_direction_from_global_wrist():
    up = np.zeros((3, 44), dtype=np.float32)
    down = np.zeros((3, 44), dtype=np.float32)
    up[:, -2] = 0.5
    down[:, -2] = 0.5
    up[:, -1] = [0.8, 0.6, 0.3]
    down[:, -1] = [0.3, 0.6, 0.8]

    up_features = trajectory_features(up, target_dim=44)
    down_features = trajectory_features(down, target_dim=44)

    assert up_features.shape == (DYNAMIC_TRAJECTORY_FEATURE_DIM,)
    assert up_features[1] < 0.0
    assert down_features[1] > 0.0
    assert up_features[6] < 0.0
    assert down_features[6] > 0.0


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


def test_class_motion_profiles_separate_static_and_dynamic():
    class Record:
        def __init__(self, label, sequence):
            self.label = label
            self.path = Path(f"{label}.npy")
            self.sequence = sequence

    static = np.zeros((3, 2), dtype=np.float32)
    dynamic = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.float32)

    profiles = class_motion_profiles(
        [
            Record("static", static),
            Record("dynamic", dynamic),
        ],
        motion_threshold=0.1,
    )
    by_label = {profile.label: profile for profile in profiles}

    assert by_label["static"].suggested_type == MOTION_STATIC_LIKE
    assert by_label["dynamic"].suggested_type == MOTION_DYNAMIC_LIKE
