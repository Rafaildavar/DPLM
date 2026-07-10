"""Motion-first dynamic gesture classification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DynamicDirectionDecision:
    label: str
    confidence: float
    reason: str
    axis: str
    direction: str
    dx: float
    dy: float
    path_length: float
    displacement: float
    axis_ratio: float
    straightness: float

    @property
    def accepted(self) -> bool:
        return bool(self.label)

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "accepted": self.accepted,
            "label": self.label,
            "confidence": self.confidence,
            "reason": self.reason,
            "axis": self.axis,
            "direction": self.direction,
            "dx": self.dx,
            "dy": self.dy,
            "path_length": self.path_length,
            "displacement": self.displacement,
            "axis_ratio": self.axis_ratio,
            "straightness": self.straightness,
        }


def classify_swipe_direction(
    motion: dict[str, float],
    labels: Iterable[str],
    *,
    min_path_length: float = 0.12,
    min_displacement: float = 0.06,
    min_axis_ratio: float = 1.20,
) -> DynamicDirectionDecision:
    """Classify swipe direction from a completed global wrist trajectory.

    This deliberately ignores hand pose. For simple swipe classes, the robust
    signal is the completed motion segment direction; pose-heavy KNN should be
    a diagnostic/fallback channel, not the primary router.
    """
    dx = _finite_float(motion.get("dx"))
    dy = _finite_float(motion.get("dy"))
    path_length = _finite_float(motion.get("path_length"))
    displacement = _finite_float(motion.get("displacement"))
    if displacement <= 0.0:
        displacement = float(np.hypot(dx, dy))

    base = {
        "dx": dx,
        "dy": dy,
        "path_length": path_length,
        "displacement": displacement,
    }
    if path_length < float(min_path_length) or displacement < float(min_displacement):
        return _decision(
            reason="insufficient_motion",
            axis_ratio=0.0,
            straightness=0.0,
            **base,
        )

    horizontal = abs(dx)
    vertical = abs(dy)
    if horizontal <= 1e-8 and vertical <= 1e-8:
        return _decision(
            reason="zero_direction",
            axis_ratio=0.0,
            straightness=0.0,
            **base,
        )

    if horizontal >= vertical * float(min_axis_ratio):
        axis = "horizontal"
        direction = "left" if dx < 0.0 else "right"
        axis_ratio = horizontal / max(vertical, 1e-6)
    elif vertical >= horizontal * float(min_axis_ratio):
        axis = "vertical"
        direction = "up" if dy < 0.0 else "down"
        axis_ratio = vertical / max(horizontal, 1e-6)
    else:
        return _decision(
            reason="ambiguous_axis",
            axis_ratio=max(horizontal, vertical) / max(min(horizontal, vertical), 1e-6),
            straightness=_straightness(displacement, path_length),
            **base,
        )

    label = _resolve_label(f"swipe_{direction}", labels)
    if not label:
        return _decision(
            reason="missing_direction_label",
            axis=axis,
            direction=direction,
            axis_ratio=axis_ratio,
            straightness=_straightness(displacement, path_length),
            **base,
        )

    straightness = _straightness(displacement, path_length)
    confidence = _direction_confidence(
        displacement=displacement,
        path_length=path_length,
        axis_ratio=axis_ratio,
        straightness=straightness,
        min_axis_ratio=float(min_axis_ratio),
    )
    return _decision(
        label=label,
        confidence=confidence,
        reason="motion_direction",
        axis=axis,
        direction=direction,
        axis_ratio=axis_ratio,
        straightness=straightness,
        **base,
    )


def _direction_confidence(
    *,
    displacement: float,
    path_length: float,
    axis_ratio: float,
    straightness: float,
    min_axis_ratio: float,
) -> float:
    amplitude_score = min(1.0, max(0.0, displacement / 0.50))
    path_score = min(1.0, max(0.0, path_length / 0.50))
    dominance_score = min(
        1.0,
        max(0.0, (axis_ratio - min_axis_ratio) / max(min_axis_ratio * 2.0, 1e-6)),
    )
    confidence = (
        0.60
        + 0.12 * amplitude_score
        + 0.08 * path_score
        + 0.14 * min(1.0, straightness)
        + 0.06 * dominance_score
    )
    return round(float(min(0.98, max(0.0, confidence))), 4)


def _resolve_label(target: str, labels: Iterable[str]) -> str:
    clean_target = str(target or "").strip().lower()
    compact_target = _compact_label(clean_target)
    for label in labels:
        raw = str(label or "").strip()
        if raw.lower() == clean_target:
            return raw
    compact_matches = [
        raw
        for label in labels
        if (raw := str(label or "").strip())
        and _compact_label(raw) == compact_target
    ]
    if len(compact_matches) == 1:
        return compact_matches[0]
    return ""


def _compact_label(value: str) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _straightness(displacement: float, path_length: float) -> float:
    if path_length <= 1e-8:
        return 0.0
    return float(min(1.0, max(0.0, displacement / path_length)))


def _finite_float(value: object) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


def _decision(
    *,
    label: str = "",
    confidence: float = 0.0,
    reason: str,
    axis: str = "",
    direction: str = "",
    dx: float,
    dy: float,
    path_length: float,
    displacement: float,
    axis_ratio: float,
    straightness: float,
) -> DynamicDirectionDecision:
    return DynamicDirectionDecision(
        label=str(label or ""),
        confidence=float(confidence or 0.0),
        reason=str(reason or ""),
        axis=str(axis or ""),
        direction=str(direction or ""),
        dx=float(dx),
        dy=float(dy),
        path_length=float(path_length),
        displacement=float(displacement),
        axis_ratio=float(axis_ratio),
        straightness=float(straightness),
    )
