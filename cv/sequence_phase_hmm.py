"""Phase-aware generative classifier for dynamic gesture sequences."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from cv.gesture_features import DYNAMIC_SEQUENCE_TARGET_FRAMES, global_wrist_slices


class PhaseHMMSequenceClassifier(ClassifierMixin, BaseEstimator):
    """Small fixed-phase HMM-style classifier.

    The model splits each completed gesture into ordered temporal phases and
    fits a diagonal Gaussian emission profile per class. It is deliberately
    lightweight: the hidden states are fixed by time order, which keeps live
    inference fast while still checking that a gesture has the right start,
    middle and end dynamics.
    """

    def __init__(
        self,
        *,
        n_states: int = 5,
        target_frames: int = DYNAMIC_SEQUENCE_TARGET_FRAMES,
        max_channels: int = 16,
        variance_floor: float = 1e-3,
        variance_regularization: float = 0.20,
        use_uniform_prior: bool = True,
    ) -> None:
        self.n_states = n_states
        self.target_frames = target_frames
        self.max_channels = max_channels
        self.variance_floor = variance_floor
        self.variance_regularization = variance_regularization
        self.use_uniform_prior = use_uniform_prior

    def fit(self, X, y):  # noqa: D401 - sklearn API
        matrix = self._validate_X(X)
        labels = np.asarray(y)
        if labels.shape[0] != matrix.shape[0]:
            raise ValueError("X and y have different sample counts")
        if labels.size == 0:
            raise ValueError("empty training labels")

        self.n_features_in_ = int(matrix.shape[1])
        self.n_channels_ = int(self.n_features_in_ // int(self.target_frames))
        sequences = self._reshape(matrix)
        self.selected_channels_ = self._select_channels(sequences)
        features = self._phase_features_from_sequences(sequences)
        self.feature_mean_ = features.mean(axis=0)
        self.feature_scale_ = features.std(axis=0)
        self.feature_scale_ = np.maximum(self.feature_scale_, 1e-6)
        normalized = (features - self.feature_mean_) / self.feature_scale_

        self.classes_, inverse = np.unique(labels, return_inverse=True)
        global_variance = np.var(normalized, axis=0) + float(self.variance_floor)
        means: list[np.ndarray] = []
        variances: list[np.ndarray] = []
        priors: list[float] = []
        reg = max(0.0, min(1.0, float(self.variance_regularization)))
        for class_index, _label in enumerate(self.classes_):
            rows = normalized[inverse == class_index]
            if rows.size == 0:
                continue
            class_variance = np.var(rows, axis=0) + float(self.variance_floor)
            variance = (1.0 - reg) * class_variance + reg * global_variance
            means.append(rows.mean(axis=0))
            variances.append(np.maximum(variance, float(self.variance_floor)))
            priors.append(float(rows.shape[0] / max(1, normalized.shape[0])))

        self.class_means_ = np.stack(means, axis=0).astype(np.float32, copy=False)
        self.class_variances_ = np.stack(variances, axis=0).astype(
            np.float32,
            copy=False,
        )
        if bool(self.use_uniform_prior):
            self.class_log_priors_ = np.full(
                len(self.classes_),
                -np.log(max(1, len(self.classes_))),
                dtype=np.float32,
            )
        else:
            self.class_log_priors_ = np.log(np.maximum(priors, 1e-8)).astype(
                np.float32,
                copy=False,
            )
        return self

    def predict(self, X):  # noqa: D401 - sklearn API
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def predict_proba(self, X):  # noqa: D401 - sklearn API
        check_is_fitted(
            self,
            [
                "classes_",
                "class_means_",
                "class_variances_",
                "feature_mean_",
                "feature_scale_",
            ],
        )
        matrix = self._validate_X(X)
        if int(matrix.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                "feature dimension mismatch: "
                f"{matrix.shape[1]} != {self.n_features_in_}"
            )
        features = self._phase_features_from_sequences(self._reshape(matrix))
        normalized = (features - self.feature_mean_) / self.feature_scale_
        log_likelihoods = self._log_likelihood(normalized)
        return _softmax(log_likelihoods)

    def score(self, X, y):  # noqa: D401 - sklearn API
        predictions = self.predict(X)
        return float(np.mean(predictions == np.asarray(y)))

    def _validate_X(self, X) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"expected 2D feature matrix, got shape={matrix.shape}")
        target_frames = int(self.target_frames)
        if target_frames <= 1:
            raise ValueError("target_frames must be greater than 1")
        if matrix.shape[1] <= 0 or matrix.shape[1] % target_frames != 0:
            raise ValueError(
                "sequence_phase_hmm expects flattened dynamic_sequence features "
                f"with dimension divisible by {target_frames}; got {matrix.shape[1]}"
            )
        if not np.isfinite(matrix).all():
            matrix = np.nan_to_num(matrix, copy=False)
        return matrix

    def _reshape(self, matrix: np.ndarray) -> np.ndarray:
        return matrix.reshape(
            matrix.shape[0],
            int(self.target_frames),
            int(self.n_channels_),
        )

    def _select_channels(self, sequences: np.ndarray) -> np.ndarray:
        n_channels = int(sequences.shape[2])
        max_channels = max(1, min(int(self.max_channels), n_channels))
        if sequences.shape[1] > 1:
            temporal_energy = np.mean(np.abs(np.diff(sequences, axis=1)), axis=(0, 1))
        else:
            temporal_energy = np.std(sequences, axis=(0, 1))

        global_channels: list[int] = []
        for start, end in global_wrist_slices(n_channels):
            global_channels.extend(range(start, end))

        order = list(np.argsort(-temporal_energy).astype(int))
        selected: list[int] = []
        for channel in global_channels:
            if channel not in selected:
                selected.append(channel)
        for channel in order:
            if channel not in selected:
                selected.append(channel)
            if len(selected) >= max_channels:
                break
        return np.asarray(sorted(selected[:max_channels]), dtype=np.int64)

    def _phase_features_from_sequences(self, sequences: np.ndarray) -> np.ndarray:
        selected = sequences[:, :, self.selected_channels_]
        boundaries = np.linspace(
            0,
            int(self.target_frames),
            num=max(2, int(self.n_states) + 1),
            dtype=int,
        )
        phase_features: list[np.ndarray] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                end = min(int(self.target_frames), start + 1)
            segment = selected[:, start:end, :]
            phase_features.append(segment.mean(axis=1))
            phase_features.append(segment[:, -1, :] - segment[:, 0, :])
            if segment.shape[1] > 1:
                velocity = np.diff(segment, axis=1)
                phase_features.append(velocity.mean(axis=1))
                phase_features.append(np.abs(velocity).sum(axis=1))
            else:
                zeros = np.zeros((sequences.shape[0], selected.shape[2]), dtype=np.float32)
                phase_features.append(zeros)
                phase_features.append(zeros)

        global_features = _global_motion_features(sequences)
        return np.concatenate([*phase_features, global_features], axis=1).astype(
            np.float32,
            copy=False,
        )

    def _log_likelihood(self, normalized: np.ndarray) -> np.ndarray:
        diff = normalized[:, None, :] - self.class_means_[None, :, :]
        variance = self.class_variances_[None, :, :]
        log_prob = -0.5 * (
            np.mean((diff * diff) / variance, axis=2)
            + np.mean(np.log(variance), axis=2)
        )
        return (log_prob + self.class_log_priors_[None, :]).astype(
            np.float32,
            copy=False,
        )


def _global_motion_features(sequences: np.ndarray) -> np.ndarray:
    n_samples, frames, channels = sequences.shape
    wrist_slices = global_wrist_slices(channels)
    if wrist_slices:
        paths: list[np.ndarray] = []
        for start, end in wrist_slices:
            wrist = sequences[:, :, start:end]
            if np.any(np.abs(wrist) > 1e-6):
                paths.append(wrist)
        if paths:
            xy = np.mean(np.stack(paths, axis=0), axis=0)
        else:
            xy = np.zeros((n_samples, frames, 2), dtype=np.float32)
    else:
        usable = (channels // 2) * 2
        if usable >= 2:
            xy = sequences[:, :, :usable].reshape(n_samples, frames, usable // 2, 2)
            xy = xy.mean(axis=2)
        else:
            xy = np.zeros((n_samples, frames, 2), dtype=np.float32)

    delta = xy[:, -1, :] - xy[:, 0, :]
    steps = np.diff(xy, axis=1)
    step_lengths = np.linalg.norm(steps, axis=2)
    path_length = step_lengths.sum(axis=1)
    displacement = np.linalg.norm(delta, axis=1)
    straightness = displacement / np.maximum(path_length, 1e-6)
    axis_ratio = np.maximum(np.abs(delta[:, 0]), np.abs(delta[:, 1])) / np.maximum(
        np.minimum(np.abs(delta[:, 0]), np.abs(delta[:, 1])),
        1e-6,
    )
    return np.stack(
        [
            delta[:, 0],
            delta[:, 1],
            np.abs(delta[:, 0]),
            np.abs(delta[:, 1]),
            path_length,
            displacement,
            straightness,
            axis_ratio,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(values)
    denom = np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    return (exp / denom).astype(np.float32, copy=False)
