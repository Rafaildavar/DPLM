"""Intent gate for first-stage static/dynamic/none routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from cv.gesture_features import (
    DYNAMIC_SEQUENCE_TARGET_FRAMES,
    align_sequence,
    sequence_to_matrix,
    trajectory_features,
)

INTENT_STATIC = "static"
INTENT_DYNAMIC = "dynamic"
INTENT_NONE = "none"
INTENT_LABELS = (INTENT_STATIC, INTENT_DYNAMIC, INTENT_NONE)
DEFAULT_INTENT_TARGET_DIM = 44
DEFAULT_INTENT_CONFIDENCE_THRESHOLD = 0.75


def build_intent_feature_vector(
    sequence: np.ndarray,
    *,
    target_dim: int = DEFAULT_INTENT_TARGET_DIM,
) -> np.ndarray:
    """Build compact sequence-level features for intent spotting.

    The gate must decide whether the user is holding a static gesture, doing a
    dynamic gesture, or just moving a hand in a way that should be ignored. It
    should not memorize gesture labels, so the representation focuses on motion
    and stability rather than class identity.
    """
    seq = align_sequence(sequence_to_matrix(sequence), int(target_dim))
    frames = int(seq.shape[0])
    feature_dim = int(seq.shape[1])
    if frames <= 1:
        velocity = np.zeros((1, feature_dim), dtype=np.float32)
    else:
        velocity = np.diff(seq, axis=0).astype(np.float32, copy=False)

    velocity_norms = (
        np.linalg.norm(velocity, axis=1) / np.sqrt(max(1, feature_dim))
        if velocity.size
        else np.zeros(1, dtype=np.float32)
    )
    pose_std = seq.std(axis=0)
    pose_range = seq.max(axis=0) - seq.min(axis=0)
    displacement = float(np.linalg.norm(seq[-1] - seq[0]) / np.sqrt(max(1, feature_dim)))

    traj = trajectory_features(seq, target_dim=int(target_dim))
    dx, dy, abs_dx, abs_dy, path_length, direction_cos, direction_sin = [
        float(value) for value in traj
    ]
    trajectory_displacement = float(np.hypot(dx, dy))
    straightness = (
        trajectory_displacement / path_length if path_length > 1e-6 else 0.0
    )
    axis_ratio = max(abs_dx, abs_dy) / max(min(abs_dx, abs_dy), 1e-6)
    dominant_axis = 1.0 if abs_dx >= abs_dy else -1.0

    values = [
        min(frames / float(DYNAMIC_SEQUENCE_TARGET_FRAMES), 3.0),
        displacement,
        float(np.mean(velocity_norms)),
        float(np.std(velocity_norms)),
        float(np.max(velocity_norms)),
        float(np.percentile(velocity_norms, 90)),
        float(np.sum(velocity_norms)),
        float(np.mean(np.abs(velocity))),
        float(np.std(velocity)),
        float(np.max(np.abs(velocity))),
        float(np.mean(pose_std)),
        float(np.max(pose_std)),
        float(np.mean(pose_range)),
        float(np.max(pose_range)),
        dx,
        dy,
        abs_dx,
        abs_dy,
        path_length,
        trajectory_displacement,
        straightness,
        axis_ratio,
        dominant_axis,
        direction_cos,
        direction_sin,
    ]
    return np.asarray(values, dtype=np.float32)


def load_intent_gate_model(path: Path | str) -> dict[str, Any]:
    payload = joblib.load(str(path))
    if not isinstance(payload, dict):
        raise ValueError("intent gate payload must be a dict")
    if payload.get("model") is None:
        raise ValueError("intent gate payload missing model")
    return payload


def save_intent_gate_model(payload: dict[str, Any], path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, str(target))


def predict_intent_gate(
    payload: dict[str, Any],
    feature: np.ndarray,
) -> dict[str, Any]:
    model = payload.get("model")
    if model is None:
        return _empty_decision("model_missing")

    vector = np.asarray(feature, dtype=np.float32).reshape(1, -1)
    payload_labels = [str(label) for label in payload.get("classes") or INTENT_LABELS]
    try:
        probabilities = np.asarray(model.predict_proba(vector)[0], dtype=float)
        classes = list(getattr(model, "classes_", range(len(probabilities))))
    except Exception:
        try:
            predicted = model.predict(vector)
            label = (
                _decode_model_class(predicted[0], payload_labels)
                if len(predicted)
                else ""
            )
        except Exception as exc:
            return _empty_decision(f"predict_failed:{exc}")
        return {
            "label": label,
            "confidence": 1.0 if label else 0.0,
            "margin": 1.0 if label else 0.0,
            "probabilities": {label: 1.0} if label else {},
            "reason": "predict_only",
        }

    ranked = sorted(
        (
            (
                float(probability),
                _decode_model_class(classes[index], payload_labels),
            )
            for index, probability in enumerate(probabilities)
        ),
        reverse=True,
    )
    if not ranked:
        return _empty_decision("no_probabilities")

    top_confidence, top_label = ranked[0]
    second = float(ranked[1][0]) if len(ranked) > 1 else 0.0
    return {
        "label": top_label,
        "confidence": float(top_confidence),
        "margin": float(top_confidence - second),
        "probabilities": {
            str(label): float(probability) for probability, label in ranked
        },
        "reason": "ok",
    }


def metadata_for_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = {
        "schema_version": 1,
        "labels": list(payload.get("classes") or INTENT_LABELS),
        "target_dim": int(payload.get("target_dim") or DEFAULT_INTENT_TARGET_DIM),
        "feature_dim": int(payload.get("feature_dim") or 0),
        "confidence_threshold": float(
            payload.get("confidence_threshold") or DEFAULT_INTENT_CONFIDENCE_THRESHOLD
        ),
        "model_type": str(payload.get("model_type") or "intent_gate_mlp"),
    }
    return out


def write_intent_gate_metadata(payload: dict[str, Any], path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(metadata_for_payload(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _empty_decision(reason: str) -> dict[str, Any]:
    return {
        "label": "",
        "confidence": 0.0,
        "margin": 0.0,
        "probabilities": {},
        "reason": str(reason),
    }


def _decode_model_class(raw_class: Any, labels: list[str]) -> str:
    try:
        index = int(raw_class)
    except (TypeError, ValueError):
        return str(raw_class)
    if 0 <= index < len(labels):
        return labels[index]
    return str(raw_class)
