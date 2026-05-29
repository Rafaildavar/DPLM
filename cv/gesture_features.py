from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


FEATURE_STATIC_MEAN = "static_mean"
FEATURE_DYNAMIC_STATS = "dynamic_stats"
SUPPORTED_FEATURE_MODES = (FEATURE_STATIC_MEAN, FEATURE_DYNAMIC_STATS)


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


def sequence_to_matrix(raw: np.ndarray) -> np.ndarray:
    """Convert a saved gesture sample to a finite (frames, features) matrix."""
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
    """Pad or truncate every frame to target_dim features."""
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


def static_mean_features(sequence: np.ndarray, target_dim: int | None = None) -> np.ndarray:
    seq = sequence_to_matrix(sequence)
    if target_dim is not None:
        seq = align_sequence(seq, target_dim)
    return seq.mean(axis=0).astype(np.float32, copy=False)


def dynamic_stats_features(sequence: np.ndarray, target_dim: int | None = None) -> np.ndarray:
    seq = sequence_to_matrix(sequence)
    if target_dim is not None:
        seq = align_sequence(seq, target_dim)

    if seq.shape[0] > 1:
        velocity = np.diff(seq, axis=0)
        velocity_mean = velocity.mean(axis=0)
        velocity_std = velocity.std(axis=0)
        velocity_abs_mean = np.abs(velocity).mean(axis=0)
    else:
        velocity_mean = np.zeros(seq.shape[1], dtype=np.float32)
        velocity_std = np.zeros(seq.shape[1], dtype=np.float32)
        velocity_abs_mean = np.zeros(seq.shape[1], dtype=np.float32)

    feature_blocks = [
        seq.mean(axis=0),
        seq.std(axis=0),
        seq.min(axis=0),
        seq.max(axis=0),
        seq[-1] - seq[0],
        velocity_mean,
        velocity_std,
        velocity_abs_mean,
    ]
    return np.concatenate(feature_blocks, axis=0).astype(np.float32, copy=False)


def build_feature_vector(
    sequence: np.ndarray,
    mode: str = FEATURE_STATIC_MEAN,
    target_dim: int | None = None,
) -> np.ndarray:
    if mode == FEATURE_STATIC_MEAN:
        return static_mean_features(sequence, target_dim=target_dim)
    if mode == FEATURE_DYNAMIC_STATS:
        return dynamic_stats_features(sequence, target_dim=target_dim)
    raise ValueError(f"unsupported feature mode: {mode}")


def load_gesture_sequences(data_root: Path) -> list[GestureSequence]:
    records: list[GestureSequence] = []
    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        label = label_dir.name
        for sample_path in sorted(label_dir.glob("sample_*.npy")):
            try:
                sequence = sequence_to_matrix(np.load(sample_path))
            except ValueError:
                continue
            records.append(GestureSequence(label=label, path=sample_path, sequence=sequence))
    return records


def class_counts(records: Sequence[GestureSequence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1
    return dict(sorted(counts.items()))


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
