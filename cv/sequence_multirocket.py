"""MultiRocket-style features for multivariate gesture sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from cv.gesture_features import DYNAMIC_SEQUENCE_TARGET_FRAMES, global_wrist_slices


@dataclass(frozen=True)
class _MultiKernel:
    length: int
    dilation: int
    channels: np.ndarray
    weights: np.ndarray
    bias: float
    use_difference: bool


class RandomMultiRocketSequenceTransformer(BaseEstimator, TransformerMixin):
    """Random convolution features with multiple pooling operators.

    The input is a flattened ``dynamic_sequence`` matrix: ``frames * channels``.
    Compared with the project's first ROCKET baseline, this transformer also
    applies kernels to first-order temporal differences and emits richer pooling
    statistics. That gives the classifier direct evidence about motion changes,
    not only about the absolute sequence shape.
    """

    def __init__(
        self,
        *,
        n_kernels: int = 256,
        target_frames: int = DYNAMIC_SEQUENCE_TARGET_FRAMES,
        kernel_lengths: Iterable[int] = (3, 5, 7, 9),
        max_dilation: int = 6,
        max_channels_per_kernel: int = 8,
        include_differences: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_kernels = n_kernels
        self.target_frames = target_frames
        self.kernel_lengths = tuple(kernel_lengths)
        self.max_dilation = max_dilation
        self.max_channels_per_kernel = max_channels_per_kernel
        self.include_differences = include_differences
        self.random_state = random_state

    def fit(self, X, y=None):  # noqa: D401 - sklearn API
        matrix = self._validate_X(X)
        self.n_features_in_ = int(matrix.shape[1])
        self.n_channels_ = int(self.n_features_in_ // int(self.target_frames))
        rng = np.random.default_rng(int(self.random_state))
        self.kernels_ = self._build_kernels(rng)
        return self

    def transform(self, X):  # noqa: D401 - sklearn API
        if not hasattr(self, "kernels_"):
            raise RuntimeError("RandomMultiRocketSequenceTransformer is not fitted")
        matrix = self._validate_X(X)
        if int(matrix.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                "feature dimension mismatch: "
                f"{matrix.shape[1]} != {self.n_features_in_}"
            )

        sequences = matrix.reshape(
            matrix.shape[0],
            int(self.target_frames),
            int(self.n_channels_),
        )
        feature_count = len(self.kernels_) * 5
        out = np.empty((matrix.shape[0], feature_count), dtype=np.float32)
        for sample_index, sequence in enumerate(sequences):
            diff = np.diff(sequence, axis=0)
            features: list[float] = []
            for kernel in self.kernels_:
                source = diff if kernel.use_difference else sequence
                activations = self._apply_kernel(source, kernel)
                features.extend(self._pool_activations(activations))
            out[sample_index] = np.asarray(features, dtype=np.float32)
        return out

    def _validate_X(self, X) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"expected 2D feature matrix, got shape={matrix.shape}")
        target_frames = int(self.target_frames)
        if target_frames <= 1:
            raise ValueError("target_frames must be greater than 1")
        if matrix.shape[1] <= 0 or matrix.shape[1] % target_frames != 0:
            raise ValueError(
                "sequence_multirocket expects flattened dynamic_sequence features "
                f"with dimension divisible by {target_frames}; got {matrix.shape[1]}"
            )
        if not np.isfinite(matrix).all():
            matrix = np.nan_to_num(matrix, copy=False)
        return matrix

    def _build_kernels(self, rng: np.random.Generator) -> list[_MultiKernel]:
        kernels: list[_MultiKernel] = []
        n_kernels = max(1, int(self.n_kernels))
        n_channels = max(1, int(self.n_channels_))
        max_channels = max(1, min(int(self.max_channels_per_kernel), n_channels))

        for index in range(n_kernels):
            use_difference = bool(self.include_differences and index % 2 == 1)
            source_frames = max(2, int(self.target_frames) - (1 if use_difference else 0))
            valid_lengths = [
                int(length)
                for length in self.kernel_lengths
                if 1 < int(length) <= source_frames
            ] or [min(3, source_frames)]
            length = int(rng.choice(valid_lengths))
            valid_dilations = [
                dilation
                for dilation in range(1, max(1, int(self.max_dilation)) + 1)
                if (length - 1) * dilation + 1 <= source_frames
            ] or [1]
            dilation = int(rng.choice(valid_dilations))
            channel_count = int(rng.integers(1, max_channels + 1))
            channels = np.sort(
                rng.choice(n_channels, size=channel_count, replace=False)
            ).astype(np.int64)
            channels = self._maybe_include_global_motion_channels(
                channels,
                rng=rng,
                index=index,
                n_channels=n_channels,
                max_channels=max_channels,
            )
            weights = rng.normal(size=(length, len(channels))).astype(np.float32)
            weights -= np.mean(weights)
            bias = float(rng.uniform(-1.0, 1.0))
            kernels.append(
                _MultiKernel(
                    length=length,
                    dilation=dilation,
                    channels=channels,
                    weights=weights,
                    bias=bias,
                    use_difference=use_difference,
                )
            )
        return kernels

    def _maybe_include_global_motion_channels(
        self,
        channels: np.ndarray,
        *,
        rng: np.random.Generator,
        index: int,
        n_channels: int,
        max_channels: int,
    ) -> np.ndarray:
        if index % 3 != 0:
            return channels

        global_channels: list[int] = []
        for start, end in global_wrist_slices(n_channels):
            global_channels.extend(range(start, end))
        if not global_channels:
            return channels
        selected = list(channels.astype(int))
        for channel in global_channels:
            if channel not in selected:
                selected.append(channel)
        if len(selected) > max_channels:
            keep = selected[: len(global_channels)]
            remaining = [item for item in selected if item not in keep]
            rng.shuffle(remaining)
            selected = (keep + remaining)[:max_channels]
        return np.asarray(sorted(set(selected)), dtype=np.int64)

    def _apply_kernel(self, sequence: np.ndarray, kernel: _MultiKernel) -> np.ndarray:
        effective_length = (kernel.length - 1) * kernel.dilation + 1
        positions = int(sequence.shape[0]) - effective_length + 1
        if positions <= 0:
            return np.asarray([kernel.bias], dtype=np.float32)

        activations = np.empty(positions, dtype=np.float32)
        selected = sequence[:, kernel.channels]
        for pos in range(positions):
            window = selected[
                pos : pos + effective_length : kernel.dilation,
                :,
            ]
            activations[pos] = float(np.sum(window * kernel.weights) + kernel.bias)
        return activations

    def _pool_activations(self, activations: np.ndarray) -> tuple[float, ...]:
        values = np.asarray(activations, dtype=np.float32)
        if values.size == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        positive = values > 0.0
        if np.any(positive):
            positive_values = values[positive]
            mean_positive = float(np.mean(positive_values))
            longest_positive = float(_longest_true_run(positive) / max(1, values.size))
        else:
            mean_positive = 0.0
            longest_positive = 0.0
        return (
            float(np.max(values)),
            float(np.min(values)),
            float(np.mean(positive)),
            mean_positive,
            longest_positive,
        )


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for item in mask.astype(bool):
        if item:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return int(longest)
