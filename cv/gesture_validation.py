"""Leakage-safe validation helpers for gesture samples and augmentations."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def normalized_groups(groups: Iterable[object], sample_count: int) -> np.ndarray:
    """Return validated group labels aligned with a feature matrix."""
    values = np.asarray(list(groups), dtype=object).reshape(-1)
    if values.shape[0] != int(sample_count):
        raise ValueError(
            f"groups and samples have different lengths: {values.shape[0]} != {sample_count}"
        )
    return values


def grouped_fold_count(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    max_folds: int,
) -> int:
    """Choose folds from distinct source samples per class, not file count."""
    labels = np.asarray(y).reshape(-1)
    group_values = normalized_groups(groups, labels.shape[0])
    per_class = [
        len(set(group_values[labels == label].tolist()))
        for label in np.unique(labels)
    ]
    min_groups = min(per_class) if per_class else 0
    folds = min(max(2, int(max_folds)), int(min_groups))
    if folds < 2:
        raise ValueError("at least two distinct source groups per class are required")
    return folds


def make_grouped_splitter(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    max_folds: int,
    random_state: int,
) -> tuple[StratifiedGroupKFold, int]:
    """Build a shuffled stratified splitter that keeps source families intact."""
    folds = grouped_fold_count(y, groups, max_folds=max_folds)
    return (
        StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=int(random_state),
        ),
        folds,
    )


def grouped_holdout_indices(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    validation_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return one stratified group-safe train/validation split when possible."""
    fraction = max(0.0, min(0.50, float(validation_fraction)))
    if fraction <= 0.0:
        return None
    requested_folds = max(2, int(round(1.0 / fraction)))
    try:
        splitter, _folds = make_grouped_splitter(
            y,
            groups,
            max_folds=requested_folds,
            random_state=random_state,
        )
    except ValueError:
        return None
    matrix_stub = np.zeros((len(y), 1), dtype=np.float32)
    train_idx, validation_idx = next(
        splitter.split(matrix_stub, np.asarray(y), groups=np.asarray(groups, dtype=object))
    )
    return train_idx.astype(np.int64), validation_idx.astype(np.int64)
