"""Feature extraction for recorded gesture sequences.

The production static pipeline uses hand-crafted geometry, while dynamic and
hybrid modes share this module so experiments can compare representations on
the same dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

import numpy as np

from cv.gesture_dataset_files import gesture_sample_paths

FEATURE_STATIC_MEAN = "static_mean"
FEATURE_STATIC_STATS = "static_stats"
FEATURE_STATIC_CRAFT_FULL_STATS = "static_craft_full_stats"
FEATURE_STATIC_LANDMARK_IMAGE = "static_landmark_image"
FEATURE_DYNAMIC_STATS = "dynamic_stats"
FEATURE_DYNAMIC_CRAFT_STATS = "dynamic_craft_stats"
FEATURE_DYNAMIC_CRAFT_FULL_STATS = "dynamic_craft_full_stats"
FEATURE_HYBRID_STATS = "hybrid_stats"
FEATURE_DYNAMIC_SEQUENCE = "dynamic_sequence"
FEATURE_DYNAMIC_SEQUENCE_72 = "dynamic_sequence_72"
FEATURE_DYNAMIC_LANDMARK_IMAGE = "dynamic_landmark_image"

DYNAMIC_STATS_BASE_MULTIPLIER = 6
DYNAMIC_TRAJECTORY_FEATURE_DIM = 7
DYNAMIC_TRAJECTORY_WEIGHT = 8.0
DYNAMIC_SEQUENCE_TARGET_FRAMES = 36
DYNAMIC_SEQUENCE_LONG_TARGET_FRAMES = 72
DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES = 72
STATIC_LANDMARK_IMAGE_TARGET_FRAMES = 30
DYNAMIC_CRAFT_AGGREGATION_COUNT = 5
STATIC_CRAFT_AGGREGATION_COUNT = 5

HAND_XY_DIM = 42
HAND_XY_GLOBAL_DIM = 44
HAND_XYZ_DIM = 63
HAND_XYZ_GLOBAL_DIM = 65

HAND_ROUTES: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
HAND_DISTANCE_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 4),
    (0, 8),
    (0, 12),
    (0, 16),
    (0, 20),
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 5),
    (0, 9),
    (0, 13),
    (0, 17),
    (4, 20),
)
HAND_FULL_DISTANCE_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (left, right) for left in range(21) for right in range(left + 1, 21)
)
HAND_ANGLE_TRIPLES: tuple[tuple[int, int, int], ...] = tuple(
    (route[index], route[index + 1], route[index + 2])
    for route in HAND_ROUTES
    for index in range(len(route) - 2)
)
DYNAMIC_CRAFT_MOTION_SIGNAL_DIM = 4
DYNAMIC_CRAFT_SIGNALS_PER_HAND = (
    len(HAND_DISTANCE_PAIRS)
    + len(HAND_ANGLE_TRIPLES)
    + DYNAMIC_CRAFT_MOTION_SIGNAL_DIM
)
DYNAMIC_CRAFT_FULL_SIGNALS_PER_HAND = (
    len(HAND_FULL_DISTANCE_PAIRS)
    + len(HAND_ANGLE_TRIPLES)
    + DYNAMIC_CRAFT_MOTION_SIGNAL_DIM
)
STATIC_CRAFT_FULL_SIGNALS_PER_HAND = (
    len(HAND_FULL_DISTANCE_PAIRS)
    + len(HAND_ANGLE_TRIPLES)
)

SUPPORTED_FEATURE_MODES = (
    FEATURE_STATIC_MEAN,
    FEATURE_STATIC_STATS,
    FEATURE_STATIC_CRAFT_FULL_STATS,
    FEATURE_STATIC_LANDMARK_IMAGE,
    FEATURE_DYNAMIC_STATS,
    FEATURE_DYNAMIC_CRAFT_STATS,
    FEATURE_DYNAMIC_CRAFT_FULL_STATS,
    FEATURE_HYBRID_STATS,
    FEATURE_DYNAMIC_SEQUENCE,
    FEATURE_DYNAMIC_SEQUENCE_72,
    FEATURE_DYNAMIC_LANDMARK_IMAGE,
)

MOTION_STATIC_LIKE = "static_like"
MOTION_DYNAMIC_LIKE = "dynamic_like"


@dataclass(frozen=True)
class GestureSequence:
    label: str
    path: Path
    sequence: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.sequence.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.sequence.shape[1])


@dataclass(frozen=True)
class ClassMotionProfile:
    label: str
    sample_count: int
    mean_motion_energy: float
    median_motion_energy: float
    mean_displacement: float
    median_displacement: float
    suggested_type: str


@dataclass(frozen=True)
class _HandFeatureBlock:
    start: int
    per_hand_dim: int
    coords_per_point: int
    has_global_wrist: bool = False

    @property
    def pose_dim(self) -> int:
        return 21 * self.coords_per_point

    @property
    def pose_end(self) -> int:
        return self.start + self.pose_dim

    @property
    def global_start(self) -> int | None:
        return self.pose_end if self.has_global_wrist else None


def sequence_to_matrix(raw: np.ndarray) -> np.ndarray:
    """Convert a saved gesture sample to a finite ``(frames, features)`` matrix."""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 3:
        frames, d1, d2 = arr.shape
        arr = arr.reshape(frames, d1 * d2)
    elif arr.ndim != 2:
        raise ValueError(f"expected 2D or 3D sample, got shape={arr.shape}")

    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"empty gesture sample shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("gesture sample contains non-finite values")
    return arr.astype(np.float32, copy=False)


def _layout_for_feature_dim(feature_dim: int) -> tuple[int, int, int, bool] | None:
    dim = int(feature_dim)
    if dim <= 0:
        return None
    for per_hand_dim, coords_per_point, has_global_wrist in (
        (HAND_XYZ_GLOBAL_DIM, 3, True),
        (HAND_XYZ_DIM, 3, False),
        (HAND_XY_GLOBAL_DIM, 2, True),
        (HAND_XY_DIM, 2, False),
    ):
        if dim % per_hand_dim == 0:
            return per_hand_dim, dim // per_hand_dim, coords_per_point, has_global_wrist
    return None


def hand_feature_blocks(feature_dim: int) -> list[_HandFeatureBlock]:
    layout = _layout_for_feature_dim(feature_dim)
    if layout is None:
        return []
    per_hand_dim, hand_count, coords_per_point, has_global_wrist = layout
    return [
        _HandFeatureBlock(
            start=hand_index * per_hand_dim,
            per_hand_dim=per_hand_dim,
            coords_per_point=coords_per_point,
            has_global_wrist=has_global_wrist,
        )
        for hand_index in range(hand_count)
    ]


def global_wrist_slices(feature_dim: int) -> list[tuple[int, int]]:
    return [
        (block.global_start, block.global_start + 2)
        for block in hand_feature_blocks(feature_dim)
        if block.global_start is not None
    ]


def _points_from_block(sequence: np.ndarray, block: _HandFeatureBlock) -> np.ndarray:
    raw = sequence[:, block.start : block.pose_end]
    return raw.reshape(sequence.shape[0], 21, block.coords_per_point).astype(
        np.float32,
        copy=False,
    )


def _points_xyz_from_block(sequence: np.ndarray, block: _HandFeatureBlock) -> np.ndarray:
    points = _points_from_block(sequence, block)
    if block.coords_per_point == 3:
        return points.astype(np.float32, copy=False)
    zeros = np.zeros((points.shape[0], points.shape[1], 1), dtype=np.float32)
    return np.concatenate([points, zeros], axis=2).astype(np.float32, copy=False)


def _global_wrist_from_block(sequence: np.ndarray, block: _HandFeatureBlock) -> np.ndarray:
    if block.global_start is not None:
        wrist = sequence[:, block.global_start : block.global_start + 2]
        if np.any(np.abs(wrist) > 1e-6):
            return wrist.astype(np.float32, copy=False)
    points = _points_from_block(sequence, block)
    return points[:, 0, :2].astype(np.float32, copy=False)


def _convert_sequence_layout(sequence: np.ndarray, target_dim: int) -> np.ndarray | None:
    source = sequence_to_matrix(sequence)
    target = int(target_dim)
    source_blocks = hand_feature_blocks(source.shape[1])
    target_blocks = hand_feature_blocks(target)
    if not source_blocks or not target_blocks:
        return None

    converted = np.zeros((source.shape[0], target), dtype=np.float32)
    for source_block, target_block in zip(source_blocks, target_blocks):
        points_xyz = _points_xyz_from_block(source, source_block)
        wrist = _global_wrist_from_block(source, source_block)
        if target_block.coords_per_point == 3:
            pose = points_xyz.reshape(source.shape[0], HAND_XYZ_DIM)
        else:
            pose = points_xyz[:, :, :2].reshape(source.shape[0], HAND_XY_DIM)
        converted[:, target_block.start : target_block.pose_end] = pose
        if target_block.global_start is not None:
            converted[:, target_block.global_start : target_block.global_start + 2] = wrist
    return converted


def align_sequence(sequence: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or truncate every frame to ``target_dim`` features."""
    seq = sequence_to_matrix(sequence)
    target = int(target_dim)
    if target <= 0:
        raise ValueError("target_dim must be positive")
    if seq.shape[1] == target:
        return seq
    converted = _convert_sequence_layout(seq, target)
    if converted is not None and converted.shape[1] == target:
        return converted
    if seq.shape[1] > target:
        return seq[:, :target]
    pad = np.zeros((seq.shape[0], target - seq.shape[1]), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=1)


def infer_target_dim(sequences: Iterable[np.ndarray], default: int = 42) -> int:
    dims: list[int] = []
    for sequence in sequences:
        dims.append(int(sequence_to_matrix(sequence).shape[1]))
    return max(dims) if dims else int(default)


def _as_aligned(sequence: np.ndarray, target_dim: int | None) -> np.ndarray:
    seq = sequence_to_matrix(sequence)
    if target_dim is not None:
        seq = align_sequence(seq, target_dim)
    return seq


def static_mean_features(sequence: np.ndarray, target_dim: int | None = None) -> np.ndarray:
    seq = _as_aligned(sequence, target_dim)
    return seq.mean(axis=0).astype(np.float32, copy=False)


def static_stats_features(sequence: np.ndarray, target_dim: int | None = None) -> np.ndarray:
    seq = _as_aligned(sequence, target_dim)
    return np.concatenate(
        [
            seq.mean(axis=0),
            seq.std(axis=0),
            seq.min(axis=0),
            seq.max(axis=0),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _static_craft_signal_series(sequence: np.ndarray) -> np.ndarray:
    seq = sequence_to_matrix(sequence)
    blocks = hand_feature_blocks(seq.shape[1])
    if not blocks:
        return np.zeros((seq.shape[0], 0), dtype=np.float32)

    series: list[np.ndarray] = []
    for block in blocks:
        points = _points_from_block(seq, block)
        if not np.any(np.abs(points) > 1e-6):
            series.append(
                np.zeros(
                    (seq.shape[0], STATIC_CRAFT_FULL_SIGNALS_PER_HAND),
                    dtype=np.float32,
                )
            )
            continue
        series.append(
            np.concatenate(
                [
                    _hand_distance_signals(points, pairs=HAND_FULL_DISTANCE_PAIRS),
                    _hand_angle_signals(points),
                ],
                axis=1,
            )
        )
    return np.concatenate(series, axis=1).astype(np.float32, copy=False)


def static_craft_full_signal_dim(raw_dim: int) -> int:
    return len(hand_feature_blocks(raw_dim)) * STATIC_CRAFT_FULL_SIGNALS_PER_HAND


def static_craft_full_stats_features(
    sequence: np.ndarray,
    target_dim: int | None = None,
) -> np.ndarray:
    seq = _as_aligned(sequence, target_dim)
    base = static_stats_features(seq)
    craft = _static_craft_signal_series(seq)
    if craft.shape[1] <= 0:
        return base

    aggregates = np.concatenate(
        [
            craft.mean(axis=0),
            craft.std(axis=0),
            craft.min(axis=0),
            craft.max(axis=0),
            np.percentile(craft, 95, axis=0),
        ],
        axis=0,
    )
    return np.concatenate([base, aggregates], axis=0).astype(np.float32, copy=False)


def _trajectory_xy(sequence: np.ndarray) -> np.ndarray:
    """Return a compact per-frame hand trajectory as ``(frames, 2)``.

    Dynamic samples recorded from the Flet UI can append global wrist ``x/y`` to
    each hand block: either 42 normalized pose features + 2 global coordinates
    or 63 xyz pose features + 2 global coordinates. Those coordinates carry the
    actual screen-space motion that separates ``swipe_up`` from ``swipe_down``.
    Older samples without this global signal fall back to the landmark center.
    """
    seq = sequence_to_matrix(sequence)
    dims = int(seq.shape[1])
    wrists = [
        seq[:, start:end]
        for start, end in global_wrist_slices(dims)
        if np.any(np.abs(seq[:, start:end]) > 1e-6)
    ]
    if wrists:
        return np.mean(np.stack(wrists, axis=0), axis=0).astype(np.float32, copy=False)

    usable = (dims // 2) * 2
    if usable >= 2:
        pts = seq[:, :usable].reshape(seq.shape[0], usable // 2, 2)
        return pts.mean(axis=1).astype(np.float32, copy=False)
    return np.zeros((seq.shape[0], 2), dtype=np.float32)


def global_trajectory_xy(sequence: np.ndarray) -> np.ndarray:
    return _trajectory_xy(sequence)


def trajectory_features(sequence: np.ndarray, target_dim: int | None = None) -> np.ndarray:
    """Global motion summary for dynamic gestures.

    The vector layout is:
    ``delta_x, delta_y, abs_delta_x, abs_delta_y, path_length,
    direction_cos, direction_sin``.
    """
    seq = _as_aligned(sequence, target_dim)
    xy = _trajectory_xy(seq)
    if xy.shape[0] <= 1:
        return np.zeros(DYNAMIC_TRAJECTORY_FEATURE_DIM, dtype=np.float32)

    delta = xy[-1] - xy[0]
    steps = np.diff(xy, axis=0)
    path_length = float(np.linalg.norm(steps, axis=1).sum())
    displacement = float(np.linalg.norm(delta))
    if displacement > 1e-6:
        direction = delta / displacement
    else:
        direction = np.zeros(2, dtype=np.float32)

    return np.asarray(
        [
            float(delta[0]),
            float(delta[1]),
            abs(float(delta[0])),
            abs(float(delta[1])),
            path_length,
            float(direction[0]),
            float(direction[1]),
        ],
        dtype=np.float32,
    )


def _motion_progress_signals(wrist_xy: np.ndarray) -> np.ndarray:
    wrist = np.asarray(wrist_xy, dtype=np.float32)
    out = np.zeros((wrist.shape[0], DYNAMIC_CRAFT_MOTION_SIGNAL_DIM), dtype=np.float32)
    if wrist.shape[0] <= 1:
        return out

    steps = np.diff(wrist, axis=0, prepend=wrist[0:1])
    delta = wrist[-1] - wrist[0]
    displacement = float(np.linalg.norm(delta))
    if displacement <= 1e-6:
        return out

    direction = delta / displacement
    signed = steps @ direction
    lateral = steps - signed[:, None] * direction[None, :]
    out[:, 0] = signed
    out[:, 1] = np.maximum(signed, 0.0)
    out[:, 2] = np.maximum(-signed, 0.0)
    out[:, 3] = np.linalg.norm(lateral, axis=1)
    return out.astype(np.float32, copy=False)


def _hand_distance_signals(
    points: np.ndarray,
    pairs: Sequence[tuple[int, int]] = HAND_DISTANCE_PAIRS,
) -> np.ndarray:
    values = [
        np.linalg.norm(points[:, left, :] - points[:, right, :], axis=1)
        for left, right in pairs
    ]
    return np.stack(values, axis=1).astype(np.float32, copy=False)


def _hand_angle_signals(points: np.ndarray) -> np.ndarray:
    values: list[np.ndarray] = []
    eps = 1e-6
    for left, center, right in HAND_ANGLE_TRIPLES:
        a = points[:, left, :] - points[:, center, :]
        b = points[:, right, :] - points[:, center, :]
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cosine = np.sum(a * b, axis=1) / np.maximum(denom, eps)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi
        values.append(angle.astype(np.float32, copy=False))
    return np.stack(values, axis=1).astype(np.float32, copy=False)


def _craft_signal_series(
    sequence: np.ndarray,
    *,
    full_distances: bool = False,
) -> np.ndarray:
    seq = sequence_to_matrix(sequence)
    blocks = hand_feature_blocks(seq.shape[1])
    if not blocks:
        return np.zeros((seq.shape[0], 0), dtype=np.float32)

    series: list[np.ndarray] = []
    distance_pairs = HAND_FULL_DISTANCE_PAIRS if full_distances else HAND_DISTANCE_PAIRS
    signal_dim = (
        DYNAMIC_CRAFT_FULL_SIGNALS_PER_HAND
        if full_distances
        else DYNAMIC_CRAFT_SIGNALS_PER_HAND
    )
    for block in blocks:
        points = _points_from_block(seq, block)
        if not np.any(np.abs(points) > 1e-6):
            series.append(
                np.zeros(
                    (seq.shape[0], signal_dim),
                    dtype=np.float32,
                )
            )
            continue
        wrist = _global_wrist_from_block(seq, block)
        series.append(
            np.concatenate(
                [
                    _hand_distance_signals(points, pairs=distance_pairs),
                    _hand_angle_signals(points),
                    _motion_progress_signals(wrist),
                ],
                axis=1,
            )
        )
    return np.concatenate(series, axis=1).astype(np.float32, copy=False)


def dynamic_craft_signal_dim(raw_dim: int) -> int:
    return len(hand_feature_blocks(raw_dim)) * DYNAMIC_CRAFT_SIGNALS_PER_HAND


def dynamic_craft_full_signal_dim(raw_dim: int) -> int:
    return len(hand_feature_blocks(raw_dim)) * DYNAMIC_CRAFT_FULL_SIGNALS_PER_HAND


def dynamic_craft_stats_features(
    sequence: np.ndarray,
    target_dim: int | None = None,
) -> np.ndarray:
    return _dynamic_craft_stats_features(
        sequence,
        target_dim=target_dim,
        full_distances=False,
    )


def dynamic_craft_full_stats_features(
    sequence: np.ndarray,
    target_dim: int | None = None,
) -> np.ndarray:
    return _dynamic_craft_stats_features(
        sequence,
        target_dim=target_dim,
        full_distances=True,
    )


def _dynamic_craft_stats_features(
    sequence: np.ndarray,
    target_dim: int | None = None,
    *,
    full_distances: bool,
) -> np.ndarray:
    seq = _as_aligned(sequence, target_dim)
    from cv.dynamic_motion import canonical_dynamic_sequence

    seq = canonical_dynamic_sequence(seq, target_frames=36)
    base = dynamic_stats_features(seq)
    craft = _craft_signal_series(seq, full_distances=full_distances)
    if craft.shape[1] <= 0:
        return base

    aggregates = np.concatenate(
        [
            craft.mean(axis=0),
            craft.std(axis=0),
            craft.min(axis=0),
            craft.max(axis=0),
            np.percentile(craft, 95, axis=0),
        ],
        axis=0,
    )
    return np.concatenate([base, aggregates], axis=0).astype(np.float32, copy=False)


def dynamic_stats_features(sequence: np.ndarray, target_dim: int | None = None) -> np.ndarray:
    seq = _as_aligned(sequence, target_dim)
    # Import locally to avoid a module cycle: the segmenter reuses the public
    # sequence parsing and trajectory helpers from this module.
    from cv.dynamic_motion import canonical_dynamic_sequence

    seq = canonical_dynamic_sequence(seq, target_frames=36)
    if seq.shape[0] > 1:
        velocity = np.diff(seq, axis=0)
        velocity_mean = velocity.mean(axis=0)
        velocity_std = velocity.std(axis=0)
        velocity_abs_mean = np.abs(velocity).mean(axis=0)
        path_abs_sum = np.abs(velocity).sum(axis=0)
        max_step = np.abs(velocity).max(axis=0)
    else:
        velocity_mean = np.zeros(seq.shape[1], dtype=np.float32)
        velocity_std = np.zeros(seq.shape[1], dtype=np.float32)
        velocity_abs_mean = np.zeros(seq.shape[1], dtype=np.float32)
        path_abs_sum = np.zeros(seq.shape[1], dtype=np.float32)
        max_step = np.zeros(seq.shape[1], dtype=np.float32)

    # Pose velocity has hundreds of components while the screen-space
    # trajectory has only seven. Without weighting, Euclidean KNN distance is
    # dominated by hand pose and can confuse a horizontal swipe with up/down
    # when the live pose differs from the recording session.
    weighted_trajectory = trajectory_features(seq) * DYNAMIC_TRAJECTORY_WEIGHT
    return np.concatenate(
        [
            seq[-1] - seq[0],
            velocity_mean,
            velocity_std,
            velocity_abs_mean,
            path_abs_sum,
            max_step,
            weighted_trajectory,
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def hybrid_stats_features(sequence: np.ndarray, target_dim: int | None = None) -> np.ndarray:
    return np.concatenate(
        [
            static_stats_features(sequence, target_dim=target_dim),
            dynamic_stats_features(sequence, target_dim=target_dim),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def dynamic_sequence_features(
    sequence: np.ndarray,
    target_dim: int | None = None,
    *,
    target_frames: int = DYNAMIC_SEQUENCE_TARGET_FRAMES,
) -> np.ndarray:
    seq = _as_aligned(sequence, target_dim)
    from cv.dynamic_motion import canonical_dynamic_sequence

    seq = canonical_dynamic_sequence(seq, target_frames=target_frames)
    return seq.reshape(-1).astype(np.float32, copy=False)


def _resize_sequence_nearest(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    seq = sequence_to_matrix(sequence)
    frames = int(seq.shape[0])
    target = max(1, int(target_frames))
    if frames == target:
        return seq.astype(np.float32, copy=False)
    if frames <= 1:
        return np.repeat(seq, target, axis=0).astype(np.float32, copy=False)
    indices = np.rint(np.linspace(0, frames - 1, target)).astype(np.int64)
    return seq[indices].astype(np.float32, copy=False)


def _landmark_image_point_count(raw_dim: int) -> int:
    blocks = hand_feature_blocks(raw_dim)
    if blocks:
        return sum(22 if block.has_global_wrist else 21 for block in blocks)
    usable = (int(raw_dim) // 2) * 2
    return usable // 2


def _landmark_image_from_sequence(sequence: np.ndarray) -> np.ndarray:
    seq = sequence_to_matrix(sequence)
    blocks = hand_feature_blocks(seq.shape[1])
    point_groups: list[np.ndarray] = []
    for block in blocks:
        points = _points_xyz_from_block(seq, block)
        point_groups.append(points)
        if block.has_global_wrist:
            wrist = _global_wrist_from_block(seq, block)
            wrist_xyz = np.concatenate(
                [wrist, np.zeros((wrist.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            point_groups.append(wrist_xyz[:, None, :])

    if point_groups:
        return np.concatenate(point_groups, axis=1).astype(np.float32, copy=False)

    usable = (seq.shape[1] // 2) * 2
    if usable <= 0:
        return np.zeros((seq.shape[0], 0, 3), dtype=np.float32)
    xy = seq[:, :usable].reshape(seq.shape[0], usable // 2, 2)
    zeros = np.zeros((xy.shape[0], xy.shape[1], 1), dtype=np.float32)
    return np.concatenate([xy, zeros], axis=2).astype(np.float32, copy=False)


def static_landmark_image_features(
    sequence: np.ndarray,
    target_dim: int | None = None,
    *,
    target_frames: int = STATIC_LANDMARK_IMAGE_TARGET_FRAMES,
) -> np.ndarray:
    seq = _as_aligned(sequence, target_dim)
    seq = _resize_sequence_nearest(seq, target_frames=target_frames)
    image = _landmark_image_from_sequence(seq)
    return image.reshape(-1).astype(np.float32, copy=False)


def dynamic_landmark_image_features(
    sequence: np.ndarray,
    target_dim: int | None = None,
    *,
    target_frames: int = DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES,
) -> np.ndarray:
    seq = _as_aligned(sequence, target_dim)
    from cv.dynamic_motion import canonical_dynamic_sequence

    seq = canonical_dynamic_sequence(seq, target_frames=target_frames)
    image = _landmark_image_from_sequence(seq)
    return image.reshape(-1).astype(np.float32, copy=False)


def feature_vector_size(mode: str, target_dim: int) -> int:
    raw_dim = int(target_dim)
    if raw_dim <= 0:
        raise ValueError("target_dim must be positive")
    if mode == FEATURE_STATIC_MEAN:
        return raw_dim
    if mode == FEATURE_STATIC_STATS:
        return raw_dim * 4
    if mode == FEATURE_STATIC_CRAFT_FULL_STATS:
        return (
            raw_dim * 4
            + static_craft_full_signal_dim(raw_dim) * STATIC_CRAFT_AGGREGATION_COUNT
        )
    if mode == FEATURE_STATIC_LANDMARK_IMAGE:
        return (
            STATIC_LANDMARK_IMAGE_TARGET_FRAMES
            * _landmark_image_point_count(raw_dim)
            * 3
        )
    if mode == FEATURE_DYNAMIC_STATS:
        return raw_dim * DYNAMIC_STATS_BASE_MULTIPLIER + DYNAMIC_TRAJECTORY_FEATURE_DIM
    if mode == FEATURE_DYNAMIC_CRAFT_STATS:
        return (
            raw_dim * DYNAMIC_STATS_BASE_MULTIPLIER
            + DYNAMIC_TRAJECTORY_FEATURE_DIM
            + dynamic_craft_signal_dim(raw_dim) * DYNAMIC_CRAFT_AGGREGATION_COUNT
        )
    if mode == FEATURE_DYNAMIC_CRAFT_FULL_STATS:
        return (
            raw_dim * DYNAMIC_STATS_BASE_MULTIPLIER
            + DYNAMIC_TRAJECTORY_FEATURE_DIM
            + dynamic_craft_full_signal_dim(raw_dim) * DYNAMIC_CRAFT_AGGREGATION_COUNT
        )
    if mode == FEATURE_HYBRID_STATS:
        return raw_dim * 10 + DYNAMIC_TRAJECTORY_FEATURE_DIM
    if mode == FEATURE_DYNAMIC_SEQUENCE:
        return raw_dim * DYNAMIC_SEQUENCE_TARGET_FRAMES
    if mode == FEATURE_DYNAMIC_SEQUENCE_72:
        return raw_dim * DYNAMIC_SEQUENCE_LONG_TARGET_FRAMES
    if mode == FEATURE_DYNAMIC_LANDMARK_IMAGE:
        return (
            DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES
            * _landmark_image_point_count(raw_dim)
            * 3
        )
    raise ValueError(f"unsupported feature mode: {mode}")


def infer_raw_dim_from_feature_size(mode: str, feature_dim: int) -> int:
    """Infer the per-frame raw dimension from a trained model feature size.

    ``dynamic_stats`` used to be exactly ``raw_dim * 6``. New models append
    trajectory features, so their size is ``raw_dim * 6 + 7``. The old form is
    still accepted to keep already-trained local models usable.
    """
    size = int(feature_dim)
    if size <= 0:
        return size
    if mode == FEATURE_STATIC_MEAN:
        return size
    if mode == FEATURE_STATIC_STATS:
        return size // 4 if size % 4 == 0 else size
    if mode == FEATURE_STATIC_CRAFT_FULL_STATS:
        for raw_dim in (42, 44, 63, 65, 84, 88, 126, 130):
            if feature_vector_size(mode, raw_dim) == size:
                return raw_dim
        if size % 4 == 0:
            return size // 4
        return size
    if mode == FEATURE_STATIC_LANDMARK_IMAGE:
        frame_width = size // STATIC_LANDMARK_IMAGE_TARGET_FRAMES
        if size % STATIC_LANDMARK_IMAGE_TARGET_FRAMES == 0 and frame_width % 3 == 0:
            points = frame_width // 3
            if points % 22 == 0:
                return (points // 22) * HAND_XYZ_GLOBAL_DIM
            if points % 21 == 0:
                return (points // 21) * HAND_XYZ_DIM
        return size
    if mode == FEATURE_DYNAMIC_STATS:
        shifted = size - DYNAMIC_TRAJECTORY_FEATURE_DIM
        if shifted > 0 and shifted % DYNAMIC_STATS_BASE_MULTIPLIER == 0:
            return shifted // DYNAMIC_STATS_BASE_MULTIPLIER
        if size % DYNAMIC_STATS_BASE_MULTIPLIER == 0:
            return size // DYNAMIC_STATS_BASE_MULTIPLIER
        return size
    if mode in {FEATURE_DYNAMIC_CRAFT_STATS, FEATURE_DYNAMIC_CRAFT_FULL_STATS}:
        for raw_dim in (42, 44, 63, 65, 84, 88, 126, 130):
            if feature_vector_size(mode, raw_dim) == size:
                return raw_dim
        shifted = size - DYNAMIC_TRAJECTORY_FEATURE_DIM
        if shifted > 0 and shifted % DYNAMIC_STATS_BASE_MULTIPLIER == 0:
            return shifted // DYNAMIC_STATS_BASE_MULTIPLIER
        return size
    if mode == FEATURE_HYBRID_STATS:
        shifted = size - DYNAMIC_TRAJECTORY_FEATURE_DIM
        if shifted > 0 and shifted % 10 == 0:
            return shifted // 10
        if size % 10 == 0:
            return size // 10
        return size
    if mode == FEATURE_DYNAMIC_SEQUENCE:
        if size % DYNAMIC_SEQUENCE_TARGET_FRAMES == 0:
            return size // DYNAMIC_SEQUENCE_TARGET_FRAMES
        return size
    if mode == FEATURE_DYNAMIC_SEQUENCE_72:
        if size % DYNAMIC_SEQUENCE_LONG_TARGET_FRAMES == 0:
            return size // DYNAMIC_SEQUENCE_LONG_TARGET_FRAMES
        return size
    if mode == FEATURE_DYNAMIC_LANDMARK_IMAGE:
        frame_width = size // DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES
        if size % DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES == 0 and frame_width % 3 == 0:
            points = frame_width // 3
            if points % 22 == 0:
                return (points // 22) * HAND_XYZ_GLOBAL_DIM
            if points % 21 == 0:
                return (points // 21) * HAND_XY_DIM
        return size
    return size


def build_feature_vector(
    sequence: np.ndarray,
    mode: str = FEATURE_STATIC_MEAN,
    target_dim: int | None = None,
) -> np.ndarray:
    if mode == FEATURE_STATIC_MEAN:
        return static_mean_features(sequence, target_dim=target_dim)
    if mode == FEATURE_STATIC_STATS:
        return static_stats_features(sequence, target_dim=target_dim)
    if mode == FEATURE_STATIC_CRAFT_FULL_STATS:
        return static_craft_full_stats_features(sequence, target_dim=target_dim)
    if mode == FEATURE_STATIC_LANDMARK_IMAGE:
        return static_landmark_image_features(sequence, target_dim=target_dim)
    if mode == FEATURE_DYNAMIC_STATS:
        return dynamic_stats_features(sequence, target_dim=target_dim)
    if mode == FEATURE_DYNAMIC_CRAFT_STATS:
        return dynamic_craft_stats_features(sequence, target_dim=target_dim)
    if mode == FEATURE_DYNAMIC_CRAFT_FULL_STATS:
        return dynamic_craft_full_stats_features(sequence, target_dim=target_dim)
    if mode == FEATURE_HYBRID_STATS:
        return hybrid_stats_features(sequence, target_dim=target_dim)
    if mode == FEATURE_DYNAMIC_SEQUENCE:
        return dynamic_sequence_features(sequence, target_dim=target_dim)
    if mode == FEATURE_DYNAMIC_SEQUENCE_72:
        return dynamic_sequence_features(
            sequence,
            target_dim=target_dim,
            target_frames=DYNAMIC_SEQUENCE_LONG_TARGET_FRAMES,
        )
    if mode == FEATURE_DYNAMIC_LANDMARK_IMAGE:
        return dynamic_landmark_image_features(sequence, target_dim=target_dim)
    raise ValueError(f"unsupported feature mode: {mode}")


def sequence_motion_energy(sequence: np.ndarray) -> float:
    """Mean frame-to-frame movement, normalized by feature count."""
    seq = sequence_to_matrix(sequence)
    if seq.shape[0] <= 1:
        return 0.0
    velocity = np.diff(seq, axis=0)
    per_frame = np.linalg.norm(velocity, axis=1) / np.sqrt(seq.shape[1])
    return float(np.mean(per_frame))


def sequence_displacement(sequence: np.ndarray) -> float:
    seq = sequence_to_matrix(sequence)
    return float(np.linalg.norm(seq[-1] - seq[0]) / np.sqrt(seq.shape[1]))


def load_gesture_sequences(data_root: Path) -> list[GestureSequence]:
    records: list[GestureSequence] = []
    if not data_root.exists():
        return records

    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        label = label_dir.name
        for sample_path in gesture_sample_paths(label_dir):
            try:
                sequence = sequence_to_matrix(np.load(sample_path))
            except Exception:
                continue
            records.append(GestureSequence(label=label, path=sample_path, sequence=sequence))
    return records


def class_counts(records: Sequence[GestureSequence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1
    return dict(sorted(counts.items()))


def class_motion_profiles(
    records: Sequence[GestureSequence],
    *,
    motion_threshold: float,
) -> list[ClassMotionProfile]:
    grouped: dict[str, list[GestureSequence]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)

    profiles: list[ClassMotionProfile] = []
    for label in sorted(grouped):
        label_records = grouped[label]
        energies = [sequence_motion_energy(record.sequence) for record in label_records]
        displacements = [sequence_displacement(record.sequence) for record in label_records]
        median_energy = float(median(energies)) if energies else 0.0
        suggested = (
            MOTION_DYNAMIC_LIKE
            if median_energy >= float(motion_threshold)
            else MOTION_STATIC_LIKE
        )
        profiles.append(
            ClassMotionProfile(
                label=label,
                sample_count=len(label_records),
                mean_motion_energy=round(float(mean(energies)), 6) if energies else 0.0,
                median_motion_energy=round(median_energy, 6),
                mean_displacement=round(float(mean(displacements)), 6)
                if displacements
                else 0.0,
                median_displacement=round(float(median(displacements)), 6)
                if displacements
                else 0.0,
                suggested_type=suggested,
            )
        )
    return profiles


def build_feature_matrix(
    records: Sequence[GestureSequence],
    mode: str,
    target_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not records:
        raise ValueError("no gesture records provided")

    labels = sorted({record.label for record in records})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    actual_target_dim = target_dim
    if actual_target_dim is None:
        actual_target_dim = infer_target_dim(record.sequence for record in records)

    features = [
        build_feature_vector(record.sequence, mode=mode, target_dim=actual_target_dim)
        for record in records
    ]
    X = np.stack(features, axis=0)
    y = np.asarray([label_to_idx[record.label] for record in records], dtype=np.int64)
    return X, y, labels
