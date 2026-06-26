"""Runtime routing between static and dynamic gesture recognizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.gesture_taxonomy import (
    GESTURE_TYPE_DYNAMIC,
    GestureTaxonomy,
    load_gesture_taxonomy,
)

ROUTE_NONE = "none"
ROUTE_STATIC = "static"
ROUTE_DYNAMIC = "dynamic"
DEFAULT_DYNAMIC_CONFIDENCE_THRESHOLD = 0.60


@dataclass(frozen=True)
class RecognitionCandidate:
    route: str
    label: str
    confidence: float
    output: dict[str, Any]


class GestureRecognitionRouter:
    """Small agent that chooses the safest runtime recognizer output.

    Static inference is still the baseline channel. Dynamic inference can win
    only when it predicts a taxonomy-confirmed dynamic label with enough
    confidence. This keeps old static/quasi-static classes from leaking through
    the dynamic model while we are moving toward one unified live interface.
    """

    def __init__(
        self,
        *,
        static_infer: Any | None,
        dynamic_infer: Any | None,
        taxonomy: GestureTaxonomy | None = None,
        dynamic_confidence_threshold: float = DEFAULT_DYNAMIC_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._static_infer = static_infer
        self._dynamic_infer = dynamic_infer
        self._taxonomy = taxonomy or load_gesture_taxonomy()
        self._dynamic_confidence_threshold = max(
            0.0,
            min(1.0, float(dynamic_confidence_threshold)),
        )

    @property
    def init_error(self) -> str:
        errors = [
            str(getattr(infer, "init_error", "") or "")
            for infer in (self._static_infer, self._dynamic_infer)
            if infer is not None and str(getattr(infer, "init_error", "") or "")
        ]
        return "; ".join(errors)

    @property
    def model_error(self) -> str:
        errors = [
            str(getattr(infer, "model_error", "") or "")
            for infer in (self._static_infer, self._dynamic_infer)
            if infer is not None and str(getattr(infer, "model_error", "") or "")
        ]
        return "; ".join(errors)

    @property
    def ready(self) -> bool:
        return any(
            bool(getattr(infer, "ready", False))
            for infer in (self._static_infer, self._dynamic_infer)
            if infer is not None
        )

    @property
    def has_classifier(self) -> bool:
        return any(
            bool(getattr(infer, "has_classifier", False))
            for infer in (self._static_infer, self._dynamic_infer)
            if infer is not None
        )

    @property
    def two_hands(self) -> bool:
        return bool(getattr(self._static_infer, "two_hands", False))

    @property
    def classifier_requires_two_hands(self) -> bool:
        return any(
            bool(getattr(infer, "classifier_requires_two_hands", False))
            for infer in (self._static_infer, self._dynamic_infer)
            if infer is not None
        )

    def set_two_hands(self, enabled: bool) -> None:
        for infer in (self._static_infer, self._dynamic_infer):
            setter = getattr(infer, "set_two_hands", None)
            if callable(setter):
                setter(bool(enabled))

    def close(self) -> None:
        for infer in (self._static_infer, self._dynamic_infer):
            closer = getattr(infer, "close", None)
            if callable(closer):
                closer()

    def process_frame_rgb(self, frame_rgb: Any) -> dict[str, Any]:
        static_out = self._process(self._static_infer, frame_rgb)
        dynamic_out = self._process(self._dynamic_infer, frame_rgb)
        return self._choose(static_out, dynamic_out)

    def _process(self, infer: Any | None, frame_rgb: Any) -> dict[str, Any]:
        if infer is None:
            return self._empty()
        try:
            out = infer.process_frame_rgb(frame_rgb)
        except Exception as exc:
            print(f"[w] recognition route failed: {exc}", flush=True)
            return self._empty()
        if not isinstance(out, dict):
            return self._empty()
        return out

    def _choose(
        self,
        static_out: dict[str, Any],
        dynamic_out: dict[str, Any],
    ) -> dict[str, Any]:
        static_candidate = self._candidate(ROUTE_STATIC, static_out)
        dynamic_candidate = self._candidate(ROUTE_DYNAMIC, dynamic_out)

        if self._dynamic_candidate_is_valid(dynamic_candidate):
            return self._with_route(dynamic_candidate, static_out, dynamic_out)
        if static_candidate.label:
            return self._with_route(static_candidate, static_out, dynamic_out)

        base = static_out if static_out.get("landmarks_json") else dynamic_out
        out = dict(base or self._empty())
        out["label"] = ""
        out["confidence"] = 0.0
        out["route"] = ROUTE_NONE
        out["router"] = self._router_payload(ROUTE_NONE, static_out, dynamic_out)
        return out

    def _candidate(self, route: str, out: dict[str, Any]) -> RecognitionCandidate:
        return RecognitionCandidate(
            route=route,
            label=str(out.get("label") or "").strip(),
            confidence=float(out.get("confidence") or 0.0),
            output=out,
        )

    def _dynamic_candidate_is_valid(self, candidate: RecognitionCandidate) -> bool:
        if not candidate.label:
            return False
        if candidate.confidence < self._dynamic_confidence_threshold:
            return False
        return (
            self._taxonomy.gesture_type_for_label(candidate.label)
            == GESTURE_TYPE_DYNAMIC
        )

    def _with_route(
        self,
        candidate: RecognitionCandidate,
        static_out: dict[str, Any],
        dynamic_out: dict[str, Any],
    ) -> dict[str, Any]:
        out = dict(candidate.output)
        out["label"] = candidate.label
        out["confidence"] = candidate.confidence
        out["route"] = candidate.route
        out["router"] = self._router_payload(candidate.route, static_out, dynamic_out)
        return out

    def _router_payload(
        self,
        route: str,
        static_out: dict[str, Any],
        dynamic_out: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "route": route,
            "static_label": str(static_out.get("label") or ""),
            "static_confidence": float(static_out.get("confidence") or 0.0),
            "dynamic_label": str(dynamic_out.get("label") or ""),
            "dynamic_confidence": float(dynamic_out.get("confidence") or 0.0),
            "dynamic_threshold": self._dynamic_confidence_threshold,
        }

    def _empty(self) -> dict[str, Any]:
        return {"label": "", "confidence": 0.0, "landmarks_json": "[]"}
