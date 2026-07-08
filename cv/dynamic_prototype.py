"""Prototype-based dynamic gesture recognition.

This module keeps the model deliberately small: it compares a completed
dynamic segment against user-recorded landmark sequences and rejects the event
when the nearest positive prototype is too far away or a negative prototype is
closer. It is meant as an open-set verifier on top of the existing
MediaPipe/segmenter pipeline, not as a replacement for data collection.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from cv.dynamic_motion import canonical_dynamic_sequence
from cv.gesture_features import (
    align_sequence,
    normalize_sequence_landmark_z,
    sequence_to_matrix,
)

METHOD_PROTOTYPE_DISTANCE = "prototype_distance"
METHOD_PROTOTYPE_DTW = "prototype_dtw"
SUPPORTED_DYNAMIC_PROTOTYPE_METHODS = (
    METHOD_PROTOTYPE_DISTANCE,
    METHOD_PROTOTYPE_DTW,
)

NEGATIVE_PREFIXES = (
    "background_",
    "negative_",
    "no_gesture",
    "partial_",
    "random_",
    "return_",
    "wrong_axis_",
)


@dataclass(frozen=True)
class DynamicSequenceRecord:
    label: str
    sequence: np.ndarray
    path: str = ""
    is_negative: bool = False


def is_negative_label(label: str) -> bool:
    clean = str(label or "").strip().lower()
    return any(clean.startswith(prefix) for prefix in NEGATIVE_PREFIXES)


def normalize_dynamic_sequence(
    sequence: np.ndarray,
    *,
    target_dim: int = 44,
    target_frames: int = 36,
) -> np.ndarray:
    seq = align_sequence(sequence_to_matrix(sequence), int(target_dim))
    seq = normalize_sequence_landmark_z(seq)
    return canonical_dynamic_sequence(seq, target_frames=int(target_frames)).astype(
        np.float32,
        copy=False,
    )


def load_dynamic_sequence_file(
    path: Path,
    *,
    target_dim: int = 44,
    target_frames: int = 36,
) -> np.ndarray:
    return normalize_dynamic_sequence(
        np.load(path),
        target_dim=target_dim,
        target_frames=target_frames,
    )


def distance_between_sequences(
    left: np.ndarray,
    right: np.ndarray,
    *,
    method: str,
) -> float:
    method = _normalize_method(method)
    a = sequence_to_matrix(left)
    b = sequence_to_matrix(right)
    if method == METHOD_PROTOTYPE_DISTANCE:
        usable_frames = min(a.shape[0], b.shape[0])
        usable_dim = min(a.shape[1], b.shape[1])
        if usable_frames <= 0 or usable_dim <= 0:
            return float("inf")
        delta = a[:usable_frames, :usable_dim] - b[:usable_frames, :usable_dim]
        return float(np.linalg.norm(delta) / math.sqrt(delta.size))
    return _dtw_distance(a, b)


def fit_dynamic_prototype_model(
    records: Sequence[DynamicSequenceRecord],
    *,
    method: str,
    target_dim: int = 44,
    target_frames: int = 36,
    max_prototypes_per_label: int = 8,
    threshold_multiplier: float = 1.25,
    threshold_floor: float = 0.015,
    negative_guard_quantile: float = 10.0,
    negative_guard_multiplier: float = 0.85,
) -> dict[str, Any]:
    method = _normalize_method(method)
    grouped: dict[str, list[DynamicSequenceRecord]] = {}
    for record in records:
        label = str(record.label or "").strip()
        if not label:
            continue
        normalized = DynamicSequenceRecord(
            label=label,
            sequence=normalize_dynamic_sequence(
                record.sequence,
                target_dim=target_dim,
                target_frames=target_frames,
            ),
            path=record.path,
            is_negative=record.is_negative,
        )
        grouped.setdefault(label, []).append(normalized)

    positive_labels = sorted(
        label
        for label, items in grouped.items()
        if items and not any(item.is_negative for item in items)
    )
    negative_labels = sorted(
        label
        for label, items in grouped.items()
        if items and any(item.is_negative for item in items)
    )

    prototypes: list[dict[str, Any]] = []
    for label in [*positive_labels, *negative_labels]:
        selected = _select_label_prototypes(
            grouped[label],
            method=method,
            max_prototypes=max_prototypes_per_label,
        )
        for record in selected:
            prototypes.append(
                {
                    "label": record.label,
                    "type": "negative" if record.is_negative else "positive",
                    "path": record.path,
                    "sequence": np.asarray(record.sequence, dtype=np.float32).tolist(),
                }
            )

    thresholds: dict[str, dict[str, float]] = {}
    for label in positive_labels:
        label_records = grouped[label]
        own_distances = _leave_one_out_positive_distances(
            label_records,
            method=method,
        )
        impostor_distances = _nearest_impostor_distances(
            label_records,
            [
                item
                for other_label, items in grouped.items()
                if other_label != label
                for item in items
            ],
            method=method,
        )
        positive_radius = _positive_radius(
            own_distances,
            floor=threshold_floor,
            multiplier=threshold_multiplier,
        )
        threshold = positive_radius
        if impostor_distances:
            guard = (
                float(np.percentile(impostor_distances, negative_guard_quantile))
                * float(negative_guard_multiplier)
            )
            if guard > 0:
                threshold = min(threshold, max(float(threshold_floor), guard))
        thresholds[label] = {
            "threshold": float(threshold),
            "positive_radius": float(positive_radius),
            "positive_distance_mean": float(np.mean(own_distances))
            if own_distances
            else 0.0,
            "positive_distance_p95": float(np.percentile(own_distances, 95))
            if own_distances
            else 0.0,
            "impostor_distance_p10": float(np.percentile(impostor_distances, 10))
            if impostor_distances
            else 0.0,
        }

    return {
        "schema_version": 1,
        "method": method,
        "target_dim": int(target_dim),
        "target_frames": int(target_frames),
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
        "thresholds": thresholds,
        "prototypes": prototypes,
        "metadata": {
            "record_count": int(len(records)),
            "prototype_count": int(len(prototypes)),
            "max_prototypes_per_label": int(max_prototypes_per_label),
            "threshold_multiplier": float(threshold_multiplier),
            "threshold_floor": float(threshold_floor),
            "negative_guard_quantile": float(negative_guard_quantile),
            "negative_guard_multiplier": float(negative_guard_multiplier),
        },
    }


def predict_dynamic_prototype(
    payload: dict[str, Any],
    sequence: np.ndarray,
) -> dict[str, Any]:
    if not payload:
        return {"accepted": False, "label": "", "reason": "missing_model"}
    method = _normalize_method(str(payload.get("method") or METHOD_PROTOTYPE_DISTANCE))
    target_dim = int(payload.get("target_dim") or 44)
    target_frames = int(payload.get("target_frames") or 36)
    seq = normalize_dynamic_sequence(
        sequence,
        target_dim=target_dim,
        target_frames=target_frames,
    )
    prototypes = payload.get("prototypes")
    if not isinstance(prototypes, list) or not prototypes:
        return {"accepted": False, "label": "", "reason": "missing_prototypes"}

    ranked: list[tuple[float, dict[str, Any]]] = []
    for prototype in prototypes:
        if not isinstance(prototype, dict):
            continue
        try:
            proto_seq = np.asarray(prototype.get("sequence"), dtype=np.float32)
            distance = distance_between_sequences(seq, proto_seq, method=method)
        except Exception:
            continue
        ranked.append((float(distance), prototype))
    if not ranked:
        return {"accepted": False, "label": "", "reason": "no_valid_prototypes"}

    ranked.sort(key=lambda item: item[0])
    best_distance, best = ranked[0]
    second_distance = ranked[1][0] if len(ranked) > 1 else float("inf")
    raw_label = str(best.get("label") or "").strip()
    label = raw_label
    prototype_type = str(best.get("type") or "")
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    threshold_payload = (
        thresholds.get(raw_label) or thresholds.get(label)
        if isinstance(thresholds, dict)
        else None
    )
    threshold = (
        float(threshold_payload.get("threshold") or 0.0)
        if isinstance(threshold_payload, dict)
        else 0.0
    )
    margin = float(second_distance - best_distance) if math.isfinite(second_distance) else 0.0
    base = {
        "method": method,
        "nearest_label": label,
        "nearest_type": prototype_type,
        "distance": float(best_distance),
        "threshold": float(threshold),
        "margin": margin,
        "second_distance": float(second_distance) if math.isfinite(second_distance) else None,
    }
    if prototype_type == "negative" or is_negative_label(label):
        return {
            **base,
            "accepted": False,
            "label": "",
            "confidence": 0.0,
            "reason": "nearest_negative",
        }
    if threshold <= 0.0:
        return {
            **base,
            "accepted": False,
            "label": "",
            "confidence": 0.0,
            "reason": "missing_threshold",
        }
    if best_distance > threshold:
        return {
            **base,
            "accepted": False,
            "label": "",
            "confidence": 0.0,
            "reason": "far_from_prototype",
        }

    ratio = max(0.0, min(1.0, best_distance / max(threshold, 1e-6)))
    confidence = max(0.01, min(1.0, 1.0 - 0.5 * ratio))
    return {
        **base,
        "accepted": True,
        "label": label,
        "confidence": float(confidence),
        "reason": "accepted",
    }


def save_dynamic_prototype_model(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_dynamic_prototype_model(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _normalize_method(method: str) -> str:
    clean = str(method or "").strip().lower()
    if clean not in SUPPORTED_DYNAMIC_PROTOTYPE_METHODS:
        raise ValueError(f"unsupported dynamic prototype method: {method}")
    return clean


def _dtw_distance(left: np.ndarray, right: np.ndarray) -> float:
    a = sequence_to_matrix(left)
    b = sequence_to_matrix(right)
    usable_dim = min(a.shape[1], b.shape[1])
    if usable_dim <= 0:
        return float("inf")
    a = a[:, :usable_dim]
    b = b[:, :usable_dim]
    rows, cols = a.shape[0], b.shape[0]
    dp = np.full((rows + 1, cols + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    norm = math.sqrt(max(1, usable_dim))
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]) / norm)
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[rows, cols] / max(rows, cols, 1))


def _select_label_prototypes(
    records: Sequence[DynamicSequenceRecord],
    *,
    method: str,
    max_prototypes: int,
) -> list[DynamicSequenceRecord]:
    items = list(records)
    limit = max(1, int(max_prototypes))
    if len(items) <= limit:
        return items
    if len(items) > max(32, limit * 4):
        step = max(1, len(items) // limit)
        return items[::step][:limit]

    distances = np.zeros((len(items), len(items)), dtype=np.float32)
    for i, left in enumerate(items):
        for j in range(i + 1, len(items)):
            distance = distance_between_sequences(left.sequence, items[j].sequence, method=method)
            distances[i, j] = distance
            distances[j, i] = distance
    medoid_order = np.argsort(distances.mean(axis=1))
    return [items[int(index)] for index in medoid_order[:limit]]


def _leave_one_out_positive_distances(
    records: Sequence[DynamicSequenceRecord],
    *,
    method: str,
) -> list[float]:
    items = list(records)
    if len(items) <= 1:
        return [0.0]
    out: list[float] = []
    for index, record in enumerate(items):
        distances = [
            distance_between_sequences(record.sequence, other.sequence, method=method)
            for other_index, other in enumerate(items)
            if other_index != index
        ]
        if distances:
            out.append(float(min(distances)))
    return out


def _nearest_impostor_distances(
    positives: Sequence[DynamicSequenceRecord],
    impostors: Sequence[DynamicSequenceRecord],
    *,
    method: str,
) -> list[float]:
    if not positives or not impostors:
        return []
    out: list[float] = []
    for impostor in impostors:
        distances = [
            distance_between_sequences(impostor.sequence, positive.sequence, method=method)
            for positive in positives
        ]
        if distances:
            out.append(float(min(distances)))
    return out


def _positive_radius(
    distances: Sequence[float],
    *,
    floor: float,
    multiplier: float,
) -> float:
    clean = [float(value) for value in distances if np.isfinite(value)]
    if not clean:
        return float(floor)
    mean = float(np.mean(clean))
    std = float(np.std(clean))
    p95 = float(np.percentile(clean, 95))
    return max(float(floor), max(p95, mean + 2.0 * std) * float(multiplier))
