"""SPROCKET-style prototype features for dynamic gesture sequences."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from cv.sequence_multirocket import RandomMultiRocketSequenceTransformer


class SprocketSequenceTransformer(BaseEstimator, TransformerMixin):
    """Combine random convolution features with class-prototype distances.

    This is a lightweight SPROCKET-style representation tailored for local
    gesture data. It keeps the ROCKET idea for temporal shape, then adds
    distances to representative training sequences so the classifier can learn
    "looks like this user's gesture" signals.
    """

    def __init__(
        self,
        *,
        n_kernels: int = 192,
        prototypes_per_class: int = 3,
        target_frames: int = 36,
        max_dilation: int = 6,
        max_channels_per_kernel: int = 8,
        random_state: int = 42,
    ) -> None:
        self.n_kernels = n_kernels
        self.prototypes_per_class = prototypes_per_class
        self.target_frames = target_frames
        self.max_dilation = max_dilation
        self.max_channels_per_kernel = max_channels_per_kernel
        self.random_state = random_state

    def fit(self, X, y=None):  # noqa: D401 - sklearn API
        matrix = self._validate_X(X)
        if y is None:
            raise ValueError("SprocketSequenceTransformer requires y during fit")
        labels = np.asarray(y)
        if labels.shape[0] != matrix.shape[0]:
            raise ValueError("X and y have different sample counts")

        self.n_features_in_ = int(matrix.shape[1])
        self.n_channels_ = int(self.n_features_in_ // int(self.target_frames))
        self.class_labels_ = np.unique(labels)
        self.prototypes_, self.prototype_labels_ = self._build_prototypes(matrix, labels)
        self.rocket_ = RandomMultiRocketSequenceTransformer(
            n_kernels=max(1, int(self.n_kernels)),
            target_frames=int(self.target_frames),
            max_dilation=max(1, int(self.max_dilation)),
            max_channels_per_kernel=max(1, int(self.max_channels_per_kernel)),
            include_differences=True,
            random_state=int(self.random_state),
        )
        self.rocket_.fit(matrix, labels)
        return self

    def transform(self, X):  # noqa: D401 - sklearn API
        if not hasattr(self, "rocket_") or not hasattr(self, "prototypes_"):
            raise RuntimeError("SprocketSequenceTransformer is not fitted")
        matrix = self._validate_X(X)
        if int(matrix.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                "feature dimension mismatch: "
                f"{matrix.shape[1]} != {self.n_features_in_}"
            )

        rocket_features = self.rocket_.transform(matrix)
        prototype_features = self._prototype_features(matrix)
        return np.concatenate([rocket_features, prototype_features], axis=1).astype(
            np.float32,
            copy=False,
        )

    def _validate_X(self, X) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"expected 2D feature matrix, got shape={matrix.shape}")
        target_frames = int(self.target_frames)
        if target_frames <= 1:
            raise ValueError("target_frames must be greater than 1")
        if matrix.shape[1] <= 0 or matrix.shape[1] % target_frames != 0:
            raise ValueError(
                "sequence_sprocket expects flattened dynamic_sequence features "
                f"with dimension divisible by {target_frames}; got {matrix.shape[1]}"
            )
        if not np.isfinite(matrix).all():
            matrix = np.nan_to_num(matrix, copy=False)
        return matrix

    def _build_prototypes(
        self,
        matrix: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        prototypes: list[np.ndarray] = []
        prototype_labels: list[object] = []
        per_class = max(1, int(self.prototypes_per_class))

        for label in np.unique(labels):
            rows = matrix[labels == label]
            if rows.size == 0:
                continue
            centroid = rows.mean(axis=0).astype(np.float32, copy=False)
            prototypes.append(centroid)
            prototype_labels.append(label)

            distances = np.linalg.norm(rows - centroid, axis=1)
            order = np.argsort(distances)
            for row_index in order[: max(0, per_class - 1)]:
                prototypes.append(rows[int(row_index)].astype(np.float32, copy=False))
                prototype_labels.append(label)

        if not prototypes:
            raise ValueError("no prototypes built")
        return (
            np.stack(prototypes, axis=0).astype(np.float32, copy=False),
            np.asarray(prototype_labels),
        )

    def _prototype_features(self, matrix: np.ndarray) -> np.ndarray:
        distances = _normalized_l2_distances(matrix, self.prototypes_)
        similarities = _cosine_similarities(matrix, self.prototypes_)
        class_min_distances = []
        class_max_similarities = []
        for label in self.class_labels_:
            mask = self.prototype_labels_ == label
            if not np.any(mask):
                class_min_distances.append(np.zeros(matrix.shape[0], dtype=np.float32))
                class_max_similarities.append(np.zeros(matrix.shape[0], dtype=np.float32))
                continue
            class_min_distances.append(np.min(distances[:, mask], axis=1))
            class_max_similarities.append(np.max(similarities[:, mask], axis=1))
        return np.concatenate(
            [
                distances,
                similarities,
                np.stack(class_min_distances, axis=1),
                np.stack(class_max_similarities, axis=1),
            ],
            axis=1,
        ).astype(np.float32, copy=False)


def _normalized_l2_distances(matrix: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    diff = matrix[:, None, :] - prototypes[None, :, :]
    denom = np.sqrt(max(1, matrix.shape[1]))
    return (np.linalg.norm(diff, axis=2) / denom).astype(np.float32, copy=False)


def _cosine_similarities(matrix: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    sample_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    proto_norm = np.linalg.norm(prototypes, axis=1, keepdims=True).T
    denom = np.maximum(sample_norm * proto_norm, 1e-8)
    return ((matrix @ prototypes.T) / denom).astype(np.float32, copy=False)
