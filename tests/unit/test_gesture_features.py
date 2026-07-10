from pathlib import Path

import numpy as np

from cv.gesture_features import (
    FEATURE_DYNAMIC_CRAFT_FULL_STATS,
    FEATURE_DYNAMIC_CRAFT_STATS,
    FEATURE_DYNAMIC_LANDMARK_IMAGE,
    FEATURE_DYNAMIC_SEQUENCE,
    FEATURE_DYNAMIC_SEQUENCE_72,
    FEATURE_DYNAMIC_STATS,
    FEATURE_HYBRID_STATS,
    FEATURE_STATIC_CRAFT_FULL_STATS,
    FEATURE_STATIC_LANDMARK_IMAGE,
    FEATURE_STATIC_MEAN,
    FEATURE_STATIC_STATS,
    HAND_XYZ_DIM,
    HAND_XYZ_GLOBAL_DIM,
    DYNAMIC_CRAFT_AGGREGATION_COUNT,
    DYNAMIC_CRAFT_FULL_SIGNALS_PER_HAND,
    DYNAMIC_CRAFT_SIGNALS_PER_HAND,
    DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES,
    STATIC_LANDMARK_IMAGE_TARGET_FRAMES,
    DYNAMIC_TRAJECTORY_FEATURE_DIM,
    DYNAMIC_TRAJECTORY_WEIGHT,
    DYNAMIC_SEQUENCE_TARGET_FRAMES,
    DYNAMIC_SEQUENCE_LONG_TARGET_FRAMES,
    STATIC_CRAFT_AGGREGATION_COUNT,
    STATIC_CRAFT_FULL_SIGNALS_PER_HAND,
    MOTION_DYNAMIC_LIKE,
    MOTION_STATIC_LIKE,
    align_sequence,
    build_feature_matrix,
    build_feature_vector,
    class_motion_profiles,
    feature_vector_size,
    infer_target_dim,
    infer_raw_dim_from_feature_size,
    normalize_landmark_z_with_xy,
    normalize_sequence_landmark_z,
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


def test_align_sequence_converts_42_dim_static_sample_to_63_dim_layout():
    sequence = np.zeros((3, 42), dtype=np.float32)
    sequence[:, :42] = np.arange(42, dtype=np.float32)

    converted = align_sequence(sequence, HAND_XYZ_DIM)

    assert converted.shape == (3, 63)
    assert np.allclose(converted[:, 0], sequence[:, 0])
    assert np.allclose(converted[:, 1], sequence[:, 1])
    assert np.allclose(converted[:, 2::3], 0.0)


def test_align_sequence_converts_44_dim_dynamic_sample_to_65_dim_layout():
    sequence = np.zeros((3, 44), dtype=np.float32)
    sequence[:, :42] = np.arange(42, dtype=np.float32)
    sequence[:, 42] = [0.1, 0.2, 0.3]
    sequence[:, 43] = [0.4, 0.5, 0.6]

    converted = align_sequence(sequence, HAND_XYZ_GLOBAL_DIM)

    assert converted.shape == (3, 65)
    assert np.allclose(converted[:, 0], sequence[:, 0])
    assert np.allclose(converted[:, 1], sequence[:, 1])
    assert np.allclose(converted[:, 2::3], 0.0)
    assert np.allclose(converted[:, -2:], sequence[:, -2:])


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
    dynamic_sequence = build_feature_vector(
        sequence,
        FEATURE_DYNAMIC_SEQUENCE,
        target_dim=4,
    )
    dynamic_sequence_72 = build_feature_vector(
        sequence,
        FEATURE_DYNAMIC_SEQUENCE_72,
        target_dim=4,
    )

    assert static_mean.shape == (4,)
    assert static_stats.shape == (16,)
    assert dynamic_stats.shape == (24 + DYNAMIC_TRAJECTORY_FEATURE_DIM,)
    assert hybrid_stats.shape == (40 + DYNAMIC_TRAJECTORY_FEATURE_DIM,)
    assert dynamic_sequence.shape == (4 * DYNAMIC_SEQUENCE_TARGET_FRAMES,)
    assert dynamic_sequence_72.shape == (4 * DYNAMIC_SEQUENCE_LONG_TARGET_FRAMES,)
    assert infer_raw_dim_from_feature_size(
        FEATURE_DYNAMIC_SEQUENCE,
        dynamic_sequence.shape[0],
    ) == 4
    assert infer_raw_dim_from_feature_size(
        FEATURE_DYNAMIC_SEQUENCE_72,
        dynamic_sequence_72.shape[0],
    ) == 4
    assert np.allclose(dynamic_stats[:4], 3.0)
    assert sequence_motion_energy(sequence) > 0


def test_dynamic_craft_stats_add_hand_geometry_and_motion_aggregates():
    sequence = np.zeros((8, 65), dtype=np.float32)
    for frame in range(sequence.shape[0]):
        for point in range(21):
            base = point * 3
            sequence[frame, base] = point / 20.0
            sequence[frame, base + 1] = frame / 10.0
            sequence[frame, base + 2] = point / 100.0
        sequence[frame, -2:] = [frame / 10.0, 0.25]

    features = build_feature_vector(
        sequence,
        FEATURE_DYNAMIC_CRAFT_STATS,
        target_dim=65,
    )

    expected_size = feature_vector_size(FEATURE_DYNAMIC_CRAFT_STATS, 65)
    craft_size = DYNAMIC_CRAFT_SIGNALS_PER_HAND * DYNAMIC_CRAFT_AGGREGATION_COUNT
    assert features.shape == (expected_size,)
    assert expected_size == 65 * 6 + DYNAMIC_TRAJECTORY_FEATURE_DIM + craft_size
    assert np.isfinite(features).all()
    assert infer_raw_dim_from_feature_size(FEATURE_DYNAMIC_CRAFT_STATS, expected_size) == 65


def test_static_craft_full_stats_add_all_pairwise_hand_distances():
    sequence = np.zeros((6, 63), dtype=np.float32)
    for frame in range(sequence.shape[0]):
        for point in range(21):
            base = point * 3
            sequence[frame, base] = point / 20.0
            sequence[frame, base + 1] = frame / 20.0
            sequence[frame, base + 2] = (point + frame) / 100.0

    features = build_feature_vector(
        sequence,
        FEATURE_STATIC_CRAFT_FULL_STATS,
        target_dim=63,
    )

    expected_size = feature_vector_size(FEATURE_STATIC_CRAFT_FULL_STATS, 63)
    craft_size = (
        STATIC_CRAFT_FULL_SIGNALS_PER_HAND * STATIC_CRAFT_AGGREGATION_COUNT
    )
    assert features.shape == (expected_size,)
    assert expected_size == 63 * 4 + craft_size
    assert np.isfinite(features).all()
    assert (
        infer_raw_dim_from_feature_size(FEATURE_STATIC_CRAFT_FULL_STATS, expected_size)
        == 63
    )


def test_static_landmark_image_flattens_time_points_xyz_tensor():
    sequence = np.zeros((6, 42), dtype=np.float32)
    sequence[:, 0::2] = 0.25
    sequence[:, 1::2] = np.linspace(0.1, 0.9, 21, dtype=np.float32)

    features = build_feature_vector(
        sequence,
        FEATURE_STATIC_LANDMARK_IMAGE,
        target_dim=63,
    )

    expected_size = feature_vector_size(FEATURE_STATIC_LANDMARK_IMAGE, 63)
    image = features.reshape(STATIC_LANDMARK_IMAGE_TARGET_FRAMES, 21, 3)
    assert features.shape == (expected_size,)
    assert expected_size == STATIC_LANDMARK_IMAGE_TARGET_FRAMES * 21 * 3
    assert np.allclose(image[:, :, 2], 0.0)
    assert infer_raw_dim_from_feature_size(FEATURE_STATIC_LANDMARK_IMAGE, expected_size) == 63


def test_dynamic_craft_full_stats_add_all_pairwise_hand_distances():
    sequence = np.zeros((8, 65), dtype=np.float32)
    for frame in range(sequence.shape[0]):
        for point in range(21):
            base = point * 3
            sequence[frame, base] = point / 20.0
            sequence[frame, base + 1] = frame / 10.0
            sequence[frame, base + 2] = (point + frame) / 100.0
        sequence[frame, -2:] = [frame / 10.0, 0.25]

    features = build_feature_vector(
        sequence,
        FEATURE_DYNAMIC_CRAFT_FULL_STATS,
        target_dim=65,
    )

    expected_size = feature_vector_size(FEATURE_DYNAMIC_CRAFT_FULL_STATS, 65)
    craft_size = (
        DYNAMIC_CRAFT_FULL_SIGNALS_PER_HAND * DYNAMIC_CRAFT_AGGREGATION_COUNT
    )
    light_size = feature_vector_size(FEATURE_DYNAMIC_CRAFT_STATS, 65)
    assert features.shape == (expected_size,)
    assert expected_size == 65 * 6 + DYNAMIC_TRAJECTORY_FEATURE_DIM + craft_size
    assert expected_size > light_size
    assert np.isfinite(features).all()
    assert (
        infer_raw_dim_from_feature_size(FEATURE_DYNAMIC_CRAFT_FULL_STATS, expected_size)
        == 65
    )


def test_dynamic_landmark_image_flattens_time_points_xyz_tensor():
    sequence = np.zeros((6, 65), dtype=np.float32)
    sequence[:, -2] = np.linspace(0.1, 0.9, sequence.shape[0])
    sequence[:, -1] = 0.5

    features = build_feature_vector(
        sequence,
        FEATURE_DYNAMIC_LANDMARK_IMAGE,
        target_dim=65,
    )

    expected_size = feature_vector_size(FEATURE_DYNAMIC_LANDMARK_IMAGE, 65)
    assert features.shape == (expected_size,)
    assert expected_size == DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES * 22 * 3
    assert infer_raw_dim_from_feature_size(FEATURE_DYNAMIC_LANDMARK_IMAGE, expected_size) == 65


def test_normalize_landmark_z_with_xy_uses_wrist_and_projected_hand_scale():
    xy = np.zeros((21, 2), dtype=np.float32)
    xy[8, 1] = 0.5
    z = np.zeros((21, 1), dtype=np.float32)
    z[0, 0] = 0.2
    z[8, 0] = 0.7

    normalized = normalize_landmark_z_with_xy(xy, z)

    assert normalized.shape == (21, 1)
    assert normalized[0, 0] == 0.0
    assert np.isclose(normalized[8, 0], 1.0)


def test_dynamic_landmark_image_centers_and_scales_z_channel():
    sequence = np.zeros((4, 65), dtype=np.float32)
    sequence[:, 8 * 3 + 1] = 0.5
    sequence[:, 2] = 0.2
    sequence[:, 8 * 3 + 2] = 0.7

    features = build_feature_vector(
        sequence,
        FEATURE_DYNAMIC_LANDMARK_IMAGE,
        target_dim=65,
    )
    image = features.reshape(DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES, 22, 3)

    assert np.allclose(image[:, 0, 2], 0.0)
    assert np.allclose(image[:, 8, 2], 1.0)


def test_normalize_sequence_landmark_z_preserves_global_wrist_xy():
    sequence = np.zeros((3, 65), dtype=np.float32)
    sequence[:, 8 * 3 + 1] = 0.5
    sequence[:, 2] = 0.2
    sequence[:, 8 * 3 + 2] = 0.7
    sequence[:, -2:] = [0.4, 0.6]

    normalized = normalize_sequence_landmark_z(sequence)

    assert np.allclose(normalized[:, 0 * 3 + 2], 0.0)
    assert np.allclose(normalized[:, 8 * 3 + 2], 1.0)
    assert np.allclose(normalized[:, -2:], [0.4, 0.6])


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


def test_trajectory_features_use_65_dim_global_wrist_layout():
    sequence = np.zeros((3, 65), dtype=np.float32)
    sequence[:, -2] = [0.2, 0.5, 0.8]
    sequence[:, -1] = 0.5

    features = trajectory_features(sequence, target_dim=65)

    assert features.shape == (DYNAMIC_TRAJECTORY_FEATURE_DIM,)
    assert features[0] > 0.0
    assert features[5] > 0.0


def test_dynamic_stats_weight_global_trajectory_for_distance_models():
    sequence = np.zeros((3, 44), dtype=np.float32)
    sequence[:, -2] = [0.8, 0.6, 0.3]
    sequence[:, -1] = 0.5

    raw_trajectory = trajectory_features(sequence, target_dim=44)
    dynamic = build_feature_vector(
        sequence,
        FEATURE_DYNAMIC_STATS,
        target_dim=44,
    )

    assert np.allclose(
        dynamic[-DYNAMIC_TRAJECTORY_FEATURE_DIM:],
        raw_trajectory * DYNAMIC_TRAJECTORY_WEIGHT,
    )


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
