# -*- coding: utf-8 -*-
"""Persistence helpers for gesture samples and trained model metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from sqlalchemy.orm import Session

from app.models.database import Gesture, GestureSample, RecognitionModel
from cv.gesture_dataset_files import (
    gesture_sample_paths,
    sample_source_from_path,
    stable_sample_index,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_relative(path: Path | str) -> str:
    """Return a stable project-relative path when possible."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except (OSError, ValueError):
        return str(p)


def ensure_gesture(
    session: Session,
    *,
    label: str,
    samples_path: Optional[Path | str] = None,
    model_class_id: Optional[int] = None,
    accuracy: Optional[float] = None,
    is_two_hands: Optional[bool] = None,
    commit: bool = False,
) -> Gesture:
    clean_label = (label or "").strip()
    if not clean_label:
        raise ValueError("label is required")

    gesture = session.query(Gesture).filter(Gesture.label == clean_label).first()
    if gesture is None:
        gesture = Gesture(label=clean_label)
        session.add(gesture)

    if samples_path is not None:
        gesture.samples_path = project_relative(samples_path)
    if model_class_id is not None:
        gesture.model_class_id = int(model_class_id)
    if accuracy is not None:
        gesture.accuracy = float(accuracy)
    if is_two_hands is not None:
        gesture.is_two_hands = bool(is_two_hands)
    gesture.is_active = True

    if commit:
        session.commit()
    else:
        session.flush()
    return gesture


def record_gesture_sample(
    session: Session,
    *,
    label: str,
    sample_index: int,
    features_path: Path | str,
    frames: Optional[int] = None,
    hand_count: Optional[int] = None,
    source: str = "camera",
    samples_path: Optional[Path | str] = None,
    is_two_hands: Optional[bool] = None,
    commit: bool = True,
) -> GestureSample:
    gesture = ensure_gesture(
        session,
        label=label,
        samples_path=samples_path or Path(features_path).parent,
        is_two_hands=is_two_hands,
        commit=False,
    )
    idx = int(sample_index)
    row = (
        session.query(GestureSample)
        .filter(GestureSample.gesture_id == gesture.id)
        .filter(GestureSample.sample_index == idx)
        .first()
    )
    if row is None:
        row = GestureSample(gesture_id=gesture.id, sample_index=idx)
        session.add(row)

    row.features_path = project_relative(features_path)
    row.frames = int(frames) if frames is not None else None
    row.hand_count = int(hand_count) if hand_count is not None else None
    row.source = (source or "camera").strip()[:32] or "camera"

    if commit:
        session.commit()
    else:
        session.flush()
    return row


def resolve_project_path(path: Path | str) -> Path:
    """Resolve a project-relative or absolute features path."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def list_gestures_with_sample_counts(session: Session) -> list[dict[str, int | str | None]]:
    """Return gestures from DB with the number of stored samples."""
    rows: list[dict[str, int | str | None]] = []
    gestures = session.query(Gesture).order_by(Gesture.label).all()
    for gesture in gestures:
        sample_count = (
            session.query(GestureSample)
            .filter(GestureSample.gesture_id == gesture.id)
            .count()
        )
        if sample_count <= 0:
            continue
        rows.append(
            {
                "label": str(gesture.label or ""),
                "samples": int(sample_count),
                "modelClassId": gesture.model_class_id,
            }
        )
    return rows


def assign_gesture_model_class_ids(
    session: Session,
    classes: Iterable[str],
    *,
    commit: bool = True,
) -> int:
    """Assign ``model_class_id`` for trained gesture labels."""
    updated = 0
    for class_idx, label in enumerate(classes):
        clean = (label or "").strip()
        if not clean:
            continue
        gesture = session.query(Gesture).filter(Gesture.label == clean).first()
        if gesture is None:
            continue
        gesture.model_class_id = int(class_idx)
        gesture.is_active = True
        updated += 1
    if commit:
        session.commit()
    else:
        session.flush()
    return updated


def load_dataset_from_db(
    session: Session,
    expect_dim: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load gesture samples from ``gestures`` / ``gesture_samples`` tables.

    Returns ``(X, y, classes)`` in the same format as ``cv.train_classifier.load_dataset``.
    """
    feats_raw: list[np.ndarray] = []
    y_list: list[int] = []
    classes: list[str] = []
    max_dim = 0

    gestures = (
        session.query(Gesture)
        .filter(Gesture.is_active.is_(True))
        .order_by(Gesture.label)
        .all()
    )
    for gesture in gestures:
        samples = (
            session.query(GestureSample)
            .filter(GestureSample.gesture_id == gesture.id)
            .order_by(GestureSample.sample_index)
            .all()
        )
        if not samples:
            continue

        class_idx = len(classes)
        classes.append(str(gesture.label or ""))

        for sample in samples:
            if not sample.features_path:
                print(f"[i] Пропуск sample #{sample.sample_index} для {gesture.label}: нет features_path")
                continue
            sample_path = resolve_project_path(sample.features_path)
            if not sample_path.exists():
                print(f"[!] Файл не найден: {sample_path}")
                continue

            arr = np.load(sample_path)
            if arr.ndim == 3:
                t_len, d1, d2 = arr.shape
                arr = arr.reshape(t_len, d1 * d2)
            elif arr.ndim != 2:
                print(f"[!] Неожиданная форма {sample_path}: {arr.shape}, пропуск")
                continue

            feat = arr.mean(axis=0)
            feats_raw.append(feat.astype(np.float32, copy=False))
            y_list.append(class_idx)
            if feat.shape[0] > max_dim:
                max_dim = int(feat.shape[0])

    if not feats_raw:
        raise RuntimeError("В БД нет ни одного семпла жеста для обучения")

    target_dim = expect_dim if expect_dim is not None else max_dim
    if expect_dim is not None and max_dim > expect_dim:
        print(
            f"[w] Найдены признаки длиной {max_dim} > ожидаемой {expect_dim}. "
            "Лишние компоненты будут обрезаны."
        )
    if expect_dim is None and len({f.shape[0] for f in feats_raw}) > 1:
        print(f"[i] Выравниваем разные длины признаков до {target_dim} (дополнение нулями/обрезка)")

    x_aligned: list[np.ndarray] = []
    for feat in feats_raw:
        if feat.shape[0] == target_dim:
            x_aligned.append(feat)
        elif feat.shape[0] > target_dim:
            x_aligned.append(feat[:target_dim])
        else:
            pad = np.zeros(target_dim - feat.shape[0], dtype=feat.dtype)
            x_aligned.append(np.concatenate([feat, pad], axis=0))

    x = np.stack(x_aligned, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    return x, y, classes


def sample_index_from_path(path: Path) -> int:
    return stable_sample_index(path)


def sample_shape_metadata(path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return None, None

    frames: Optional[int] = int(arr.shape[0]) if arr.ndim >= 1 else None
    hand_count: Optional[int] = None
    if arr.ndim == 3 and arr.shape[1] >= 42:
        hand_count = 2
    elif arr.ndim == 3 and arr.shape[1] >= 21:
        hand_count = 1
    elif arr.ndim == 2 and arr.shape[1] >= 84:
        hand_count = 2
    elif arr.ndim == 2 and arr.shape[1] >= 42:
        hand_count = 1
    return frames, hand_count


def sync_gesture_dataset_to_db(
    session: Session,
    *,
    data_root: Path,
    classes: Iterable[str],
    source: str = "dataset",
    commit: bool = True,
) -> int:
    total = 0
    for class_idx, label in enumerate(classes):
        label_dir = data_root / label
        sample_paths = gesture_sample_paths(label_dir)
        is_two_hands: Optional[bool] = None
        if sample_paths:
            _, first_hand_count = sample_shape_metadata(sample_paths[0])
            if first_hand_count is not None:
                is_two_hands = first_hand_count >= 2
        ensure_gesture(
            session,
            label=label,
            samples_path=label_dir,
            model_class_id=class_idx,
            is_two_hands=is_two_hands,
            commit=False,
        )
        for path in sample_paths:
            frames, hand_count = sample_shape_metadata(path)
            record_gesture_sample(
                session,
                label=label,
                sample_index=sample_index_from_path(path),
                features_path=path,
                frames=frames,
                hand_count=hand_count,
                source=sample_source_from_path(path) if source == "dataset" else source,
                samples_path=label_dir,
                is_two_hands=is_two_hands,
                commit=False,
            )
            total += 1
    if commit:
        session.commit()
    else:
        session.flush()
    return total


def register_recognition_model(
    session: Session,
    *,
    name: str,
    algorithm: str,
    model_path: Path | str,
    classes_path: Path | str,
    feature_dim: int,
    n_classes: int,
    n_samples: int,
    accuracy: Optional[float] = None,
    hyperparams: Optional[dict] = None,
    is_active: bool = True,
    commit: bool = True,
) -> RecognitionModel:
    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("model name is required")

    if is_active:
        session.query(RecognitionModel).filter(RecognitionModel.is_active.is_(True)).update(
            {RecognitionModel.is_active: False},
            synchronize_session=False,
        )

    row = session.query(RecognitionModel).filter(RecognitionModel.name == clean_name).first()
    if row is None:
        row = RecognitionModel(name=clean_name)
        session.add(row)

    row.algorithm = algorithm
    row.model_path = project_relative(model_path)
    row.classes_path = project_relative(classes_path)
    row.feature_dim = int(feature_dim)
    row.n_classes = int(n_classes)
    row.n_samples = int(n_samples)
    row.accuracy = float(accuracy) if accuracy is not None else None
    row.hyperparams_json = json.dumps(hyperparams or {}, ensure_ascii=False, sort_keys=True)
    row.is_active = bool(is_active)

    if commit:
        session.commit()
    else:
        session.flush()
    return row
