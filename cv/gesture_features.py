"""Feature extraction for recorded gesture sequences.

The production baseline in ``cv.train_classifier`` uses a static mean over
frames. JMLC experiments need a slightly wider API so we can compare static,
dynamic and hybrid representations on the same dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

import numpy as np

FEATURE_STATIC_MEAN = "static_mean"
FEATURE_STATIC_STATS = "static_stats"
FEATURE_DYNAMIC_STATS = "dynamic_stats"
FEATURE_HYBRID_STATS = "hybrid_stats"

DYNAMIC_STATS_BASE_MULTIPLIER = 6
DYNAMIC_TRAJECTORY_FEATURE_DIM = 7

SUPPORTED_FEATURE_MODES = (
    FEATURE_STATIC_MEAN,
    FEATURE_STATIC_STATS,
    FEATURE_DYNAMIC_STATS,
    FEATURE_HYBRID_STATS,
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


def align_sequence(sequence: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or truncate every frame to ``target_dim`` features."""
    seq = sequence_to_matrix(sequence)
    target = int(target_dim)
    if target <= 0:
        raise ValueError("target_dim must be positive")
    if seq.shape[1] == target:
        return seq
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


def _trajectory_xy(sequence: np.ndarray) -> np.ndarray:
    """Return a compact per-frame hand trajectory as ``(frames, 2)``.

    Dynamic samples recorded from the Flet UI can append global wrist ``x/y`` to
    each hand block: 42 normalized pose features + 2 global coordinates. Those
    coordinates carry the actual screen-space motion that separates
    ``swipe_up`` from ``swipe_down``. Older 42-feature samples do not have this
    global signal, so we fall back to the normalized landmark center.
    """
    seq = sequence_to_matrix(sequence)
    dims = int(seq.shape[1])
    if dims >= 44 and dims % 44 == 0:
        wrists: list[np.ndarray] = []
        for offset in range(0, dims, 44):
            wrist = seq[:, offset + 42 : offset + 44]
            if np.any(np.abs(wrist) > 1e-6):
                wrists.append(wrist)
        if wrists:
            return np.mean(np.stack(wrists, axis=0), axis=0).astype(np.float32, copy=False)

    usable = (dims // 2) * 2
    if usable >= 2:
        pts = seq[:, :usable].reshape(seq.shape[0], usable // 2, 2)
        return pts.mean(axis=1).astype(np.float32, copy=False)
    return np.zeros((seq.shape[0], 2), dtype=np.float32)


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


def dynamic_stats_features(sequence: np.ndarray, target_dim: int | None = None) -> np.ndarray:
    seq = _as_aligned(sequence, target_dim)
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

    return np.concatenate(
        [
            seq[-1] - seq[0],
            velocity_mean,
            velocity_std,
            velocity_abs_mean,
            path_abs_sum,
            max_step,
            trajectory_features(seq),
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


def feature_vector_size(mode: str, target_dim: int) -> int:
    raw_dim = int(target_dim)
    if raw_dim <= 0:
        raise ValueError("target_dim must be positive")
    if mode == FEATURE_STATIC_MEAN:
        return raw_dim
    if mode == FEATURE_STATIC_STATS:
        return raw_dim * 4
    if mode == FEATURE_DYNAMIC_STATS:
        return raw_dim * DYNAMIC_STATS_BASE_MULTIPLIER + DYNAMIC_TRAJECTORY_FEATURE_DIM
    if mode == FEATURE_HYBRID_STATS:
        return raw_dim * 10 + DYNAMIC_TRAJECTORY_FEATURE_DIM
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
    if mode == FEATURE_DYNAMIC_STATS:
        shifted = size - DYNAMIC_TRAJECTORY_FEATURE_DIM
        if shifted > 0 and shifted % DYNAMIC_STATS_BASE_MULTIPLIER == 0:
            return shifted // DYNAMIC_STATS_BASE_MULTIPLIER
        if size % DYNAMIC_STATS_BASE_MULTIPLIER == 0:
            return size // DYNAMIC_STATS_BASE_MULTIPLIER
        return size
    if mode == FEATURE_HYBRID_STATS:
        shifted = size - DYNAMIC_TRAJECTORY_FEATURE_DIM
        if shifted > 0 and shifted % 10 == 0:
            return shifted // 10
        if size % 10 == 0:
            return size // 10
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
    if mode == FEATURE_DYNAMIC_STATS:
        return dynamic_stats_features(sequence, target_dim=target_dim)
    if mode == FEATURE_HYBRID_STATS:
        return hybrid_stats_features(sequence, target_dim=target_dim)
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
        for sample_path in sorted(label_dir.glob("sample_*.npy")):
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
