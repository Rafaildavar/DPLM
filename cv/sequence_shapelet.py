"""Shapelet-style features for multivariate dynamic gesture sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from cv.gesture_features import DYNAMIC_SEQUENCE_TARGET_FRAMES


@dataclass(frozen=True)
class _Shapelet:
    label: object
    length: int
    start: int
    channels: np.ndarray
    values: np.ndarray
    use_difference: bool


class ShapeletSequenceTransformer(BaseEstimator, TransformerMixin):
    """Transform a sequence into distances to discriminative subsequences.

    The transformer uses training samples to build short temporal templates
    (shapelets). At inference time it slides each shapelet over the completed
    gesture and emits the best normalized distance plus match position. This is
    useful for complex gestures where a key sub-motion matters more than a
    single global direction.
    """

    def __init__(
        self,
        *,
        shapelets_per_class: int = 18,
        target_frames: int = DYNAMIC_SEQUENCE_TARGET_FRAMES,
        shapelet_lengths: Iterable[int] = (5, 9, 13),
        max_channels_per_shapelet: int = 12,
        include_differences: bool = True,
        random_state: int = 42,
    ) -> None:
        self.shapelets_per_class = shapelets_per_class
        self.target_frames = target_frames
        self.shapelet_lengths = tuple(shapelet_lengths)
        self.max_channels_per_shapelet = max_channels_per_shapelet
        self.include_differences = include_differences
        self.random_state = random_state

    def fit(self, X, y=None):  # noqa: D401 - sklearn API
        matrix = self._validate_X(X)
        if y is None:
            raise ValueError("ShapeletSequenceTransformer requires y during fit")
        labels = np.asarray(y)
        if labels.shape[0] != matrix.shape[0]:
            raise ValueError("X and y have different sample counts")

        self.n_features_in_ = int(matrix.shape[1])
        self.n_channels_ = int(self.n_features_in_ // int(self.target_frames))
        sequences = matrix.reshape(
            matrix.shape[0],
            int(self.target_frames),
            int(self.n_channels_),
        )
        self.selected_channels_ = self._select_channels(sequences)
        self.shapelets_ = self._build_shapelets(sequences, labels)
        if not self.shapelets_:
            raise ValueError("no shapelets built")
        return self

    def transform(self, X):  # noqa: D401 - sklearn API
        if not hasattr(self, "shapelets_"):
            raise RuntimeError("ShapeletSequenceTransformer is not fitted")
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
        out = np.empty((matrix.shape[0], len(self.shapelets_) * 2), dtype=np.float32)
        for sample_index, sequence in enumerate(sequences):
            features: list[float] = []
            for shapelet in self.shapelets_:
                distance, position = self._best_match(sequence, shapelet)
                features.append(distance)
                features.append(position)
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
                "sequence_shapelet expects flattened dynamic_sequence features "
                f"with dimension divisible by {target_frames}; got {matrix.shape[1]}"
            )
        if not np.isfinite(matrix).all():
            matrix = np.nan_to_num(matrix, copy=False)
        return matrix

    def _select_channels(self, sequences: np.ndarray) -> np.ndarray:
        n_channels = int(sequences.shape[2])
        max_channels = max(1, min(int(self.max_channels_per_shapelet), n_channels))
        if sequences.shape[1] > 1:
            temporal_energy = np.mean(np.abs(np.diff(sequences, axis=1)), axis=(0, 1))
        else:
            temporal_energy = np.std(sequences, axis=(0, 1))

        global_channels: list[int] = []
        if n_channels >= 44 and n_channels % 44 == 0:
            for offset in range(0, n_channels, 44):
                global_channels.extend([offset + 42, offset + 43])

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

    def _build_shapelets(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
    ) -> list[_Shapelet]:
        rng = np.random.default_rng(int(self.random_state))
        shapelets: list[_Shapelet] = []
        per_class = max(1, int(self.shapelets_per_class))
        lengths = self._valid_lengths()
        for label in np.unique(labels):
            rows = sequences[labels == label]
            if rows.size == 0:
                continue
            centroid = rows.mean(axis=0).astype(np.float32, copy=False)
            distances = np.linalg.norm(
                rows.reshape(rows.shape[0], -1) - centroid.reshape(1, -1),
                axis=1,
            )
            ordered = rows[np.argsort(distances)]
            sources = [centroid]
            sources.extend(ordered[: min(3, ordered.shape[0])])

            built_for_label = 0
            source_index = 0
            use_difference_options = [False, True] if self.include_differences else [False]
            while built_for_label < per_class:
                source = sources[source_index % len(sources)]
                use_difference = use_difference_options[
                    (source_index // max(1, len(sources))) % len(use_difference_options)
                ]
                length = lengths[built_for_label % len(lengths)]
                view = self._source_view(source, use_difference=use_difference)
                positions = self._candidate_positions(view.shape[0], length)
                if not positions:
                    break
                position = positions[
                    (built_for_label + int(rng.integers(0, len(positions)))) % len(positions)
                ]
                channels = self.selected_channels_
                values = view[position : position + length, :][:, channels]
                shapelets.append(
                    _Shapelet(
                        label=label,
                        length=int(length),
                        start=int(position),
                        channels=channels.astype(np.int64, copy=True),
                        values=values.astype(np.float32, copy=True),
                        use_difference=bool(use_difference),
                    )
                )
                built_for_label += 1
                source_index += 1
        return shapelets

    def _valid_lengths(self) -> list[int]:
        target_frames = int(self.target_frames)
        lengths = [
            int(length)
            for length in self.shapelet_lengths
            if 1 < int(length) <= target_frames
        ]
        return lengths or [min(5, target_frames)]

    @staticmethod
    def _candidate_positions(frames: int, length: int) -> list[int]:
        max_start = int(frames) - int(length)
        if max_start < 0:
            return []
        if max_start == 0:
            return [0]
        return sorted(
            {
                int(round(value))
                for value in np.linspace(0, max_start, num=min(5, max_start + 1))
            }
        )

    @staticmethod
    def _source_view(sequence: np.ndarray, *, use_difference: bool) -> np.ndarray:
        source = np.asarray(sequence, dtype=np.float32)
        if not use_difference:
            return source
        diff = np.diff(source, axis=0)
        if diff.shape[0] <= 0:
            return np.zeros_like(source)
        first = np.zeros((1, source.shape[1]), dtype=np.float32)
        return np.concatenate([first, diff], axis=0).astype(np.float32, copy=False)

    def _best_match(
        self,
        sequence: np.ndarray,
        shapelet: _Shapelet,
    ) -> tuple[float, float]:
        source = self._source_view(sequence, use_difference=shapelet.use_difference)
        selected = source[:, shapelet.channels]
        length = int(shapelet.length)
        max_start = selected.shape[0] - length
        if max_start < 0:
            return 1e6, 0.0

        best_distance = float("inf")
        best_position = 0
        denom = float(np.sqrt(max(1, shapelet.values.size)))
        for position in range(max_start + 1):
            window = selected[position : position + length, :]
            distance = float(np.linalg.norm(window - shapelet.values) / denom)
            if distance < best_distance:
                best_distance = distance
                best_position = position
        position_feature = (
            float(best_position / max(1, max_start))
            if max_start > 0
            else 0.0
        )
        return float(best_distance), position_feature
