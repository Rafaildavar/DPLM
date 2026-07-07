"""Lightweight ROCKET-style features for dynamic gesture sequences.

The transformer accepts flattened canonical sequences produced by the
``dynamic_sequence`` feature mode: ``(samples, frames * channels)``. It applies
random temporal convolution kernels and returns two statistics per kernel:
maximum activation and proportion of positive values. This is a practical
time-series baseline that needs only NumPy and scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from cv.gesture_features import DYNAMIC_SEQUENCE_TARGET_FRAMES, global_wrist_slices


@dataclass(frozen=True)
class _Kernel:
    length: int
    dilation: int
    channels: np.ndarray
    weights: np.ndarray
    bias: float


class RandomConvolutionSequenceTransformer(BaseEstimator, TransformerMixin):
    """Random temporal convolution features for multivariate sequences."""

    def __init__(
        self,
        *,
        n_kernels: int = 256,
        target_frames: int = DYNAMIC_SEQUENCE_TARGET_FRAMES,
        kernel_lengths: Iterable[int] = (3, 5, 7, 9),
        max_dilation: int = 4,
        max_channels_per_kernel: int = 8,
        random_state: int = 42,
    ) -> None:
        self.n_kernels = n_kernels
        self.target_frames = target_frames
        self.kernel_lengths = tuple(kernel_lengths)
        self.max_dilation = max_dilation
        self.max_channels_per_kernel = max_channels_per_kernel
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
            raise RuntimeError("RandomConvolutionSequenceTransformer is not fitted")
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
        out = np.empty((matrix.shape[0], len(self.kernels_) * 2), dtype=np.float32)
        for sample_index, sequence in enumerate(sequences):
            features: list[float] = []
            for kernel in self.kernels_:
                activations = self._apply_kernel(sequence, kernel)
                features.append(float(np.max(activations)))
                features.append(float(np.mean(activations > 0.0)))
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
                "sequence_rocket expects flattened dynamic_sequence features "
                f"with dimension divisible by {target_frames}; got {matrix.shape[1]}"
            )
        if not np.isfinite(matrix).all():
            matrix = np.nan_to_num(matrix, copy=False)
        return matrix

    def _build_kernels(self, rng: np.random.Generator) -> list[_Kernel]:
        kernels: list[_Kernel] = []
        n_kernels = max(1, int(self.n_kernels))
        n_channels = max(1, int(self.n_channels_))
        target_frames = max(2, int(self.target_frames))
        valid_lengths = [
            int(length)
            for length in self.kernel_lengths
            if 1 < int(length) <= target_frames
        ] or [min(3, target_frames)]
        max_channels = max(1, min(int(self.max_channels_per_kernel), n_channels))

        for index in range(n_kernels):
            length = int(rng.choice(valid_lengths))
            valid_dilations = [
                dilation
                for dilation in range(1, max(1, int(self.max_dilation)) + 1)
                if (length - 1) * dilation + 1 <= target_frames
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
                _Kernel(
                    length=length,
                    dilation=dilation,
                    channels=channels,
                    weights=weights,
                    bias=bias,
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
        # Dynamic samples recorded by the Flet UI often use 44 features per
        # hand: 42 pose values plus global wrist x/y. Give these global motion
        # channels regular coverage because they are important for gestures.
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

    def _apply_kernel(self, sequence: np.ndarray, kernel: _Kernel) -> np.ndarray:
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
