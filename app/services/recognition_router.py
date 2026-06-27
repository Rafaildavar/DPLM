"""Runtime routing between static and dynamic gesture recognizers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.services.gesture_taxonomy import (
    GESTURE_TYPE_DYNAMIC,
    GestureTaxonomy,
    load_gesture_taxonomy,
)

ROUTE_NONE = "none"
ROUTE_STATIC = "static"
ROUTE_DYNAMIC = "dynamic"
REASON_DYNAMIC_ACCEPTED = "dynamic_accepted"
REASON_STATIC_FALLBACK = "static_fallback"
REASON_NO_VALID_CANDIDATE = "no_valid_candidate"
REASON_NO_LABEL = "no_label"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_NOT_DYNAMIC_TYPE = "not_dynamic_type"
REASON_DYNAMIC_LABEL_REQUIRES_DYNAMIC_ROUTE = "dynamic_label_requires_dynamic_route"
REASON_DYNAMIC_OBSERVATION_PENDING = "dynamic_observation_pending"
DEFAULT_DYNAMIC_CONFIDENCE_THRESHOLD = 0.60
DEFAULT_STATIC_CONFIDENCE_THRESHOLD = 0.50


@dataclass(frozen=True)
class RecognitionCandidate:
    route: str
    label: str
    confidence: float
    output: dict[str, Any]


@dataclass(frozen=True)
class CandidateAssessment:
    candidate: RecognitionCandidate
    gesture_type: str
    accepted: bool
    reason: str


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
        static_confidence_threshold: float = DEFAULT_STATIC_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._static_infer = static_infer
        self._dynamic_infer = dynamic_infer
        self._taxonomy = taxonomy or load_gesture_taxonomy()
        self._dynamic_confidence_threshold = max(
            0.0,
            min(1.0, float(dynamic_confidence_threshold)),
        )
        self._static_confidence_threshold = max(
            0.0,
            min(1.0, float(static_confidence_threshold)),
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

    def reset_temporal_state(self) -> None:
        resetter = getattr(self._dynamic_infer, "reset_temporal_state", None)
        if callable(resetter):
            resetter()

    def acknowledge_dynamic_event(self) -> None:
        acknowledger = getattr(
            self._dynamic_infer,
            "acknowledge_dynamic_event",
            None,
        )
        if callable(acknowledger):
            acknowledger()

    def process_frame_rgb(self, frame_rgb: Any) -> dict[str, Any]:
        started = perf_counter()
        shared_detection = self._supports_shared_detection()
        detection_ms = 0.0

        if shared_detection:
            detection_started = perf_counter()
            try:
                hands = self._static_infer.detect_hands(frame_rgb)
            except Exception as exc:
                print(f"[w] shared hand detection failed: {exc}", flush=True)
                hands = []
            detection_ms = (perf_counter() - detection_started) * 1000.0
            static_out = self._process_detected(self._static_infer, hands)
            dynamic_out = self._process_detected(self._dynamic_infer, hands)
        else:
            static_out = self._process(self._static_infer, frame_rgb)
            dynamic_out = self._process(self._dynamic_infer, frame_rgb)

        out = self._choose(static_out, dynamic_out)
        performance = {
            "shared_detection": shared_detection,
            "detection_ms": round(detection_ms, 3),
            "total_inference_ms": round((perf_counter() - started) * 1000.0, 3),
        }
        out["performance"] = performance
        router_payload = out.get("router")
        if isinstance(router_payload, dict):
            router_payload["shared_detection"] = shared_detection
        return out

    def _supports_shared_detection(self) -> bool:
        return all(
            callable(method)
            for method in (
                getattr(self._static_infer, "detect_hands", None),
                getattr(self._static_infer, "process_detected_hands", None),
                getattr(self._dynamic_infer, "process_detected_hands", None),
            )
        )

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

    def _process_detected(
        self,
        infer: Any | None,
        hands: Any,
    ) -> dict[str, Any]:
        if infer is None:
            return self._empty()
        try:
            out = infer.process_detected_hands(hands)
        except Exception as exc:
            print(f"[w] shared recognition route failed: {exc}", flush=True)
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
        static_assessment = self._assess_static_candidate(static_candidate)
        dynamic_assessment = self._assess_dynamic_candidate(dynamic_candidate)

        if dynamic_assessment.accepted:
            return self._with_route(
                dynamic_candidate,
                static_out,
                dynamic_out,
                static_assessment=static_assessment,
                dynamic_assessment=dynamic_assessment,
                selected_reason=REASON_DYNAMIC_ACCEPTED,
            )
        if static_assessment.accepted and self._should_hold_static(dynamic_out):
            return self._without_candidate(
                static_out,
                dynamic_out,
                static_assessment=static_assessment,
                dynamic_assessment=dynamic_assessment,
                selected_reason=REASON_DYNAMIC_OBSERVATION_PENDING,
            )
        if static_assessment.accepted:
            return self._with_route(
                static_candidate,
                static_out,
                dynamic_out,
                static_assessment=static_assessment,
                dynamic_assessment=dynamic_assessment,
                selected_reason=REASON_STATIC_FALLBACK,
            )

        return self._without_candidate(
            static_out,
            dynamic_out,
            static_assessment=static_assessment,
            dynamic_assessment=dynamic_assessment,
            selected_reason=REASON_NO_VALID_CANDIDATE,
        )

    def _should_hold_static(self, dynamic_out: dict[str, Any]) -> bool:
        temporal = dynamic_out.get("temporal")
        if not isinstance(temporal, dict) or not bool(temporal.get("enabled")):
            return False
        return str(temporal.get("phase") or "") in {
            "warming_up",
            "active",
            "cooldown",
        }

    def _without_candidate(
        self,
        static_out: dict[str, Any],
        dynamic_out: dict[str, Any],
        *,
        static_assessment: CandidateAssessment,
        dynamic_assessment: CandidateAssessment,
        selected_reason: str,
    ) -> dict[str, Any]:
        base = static_out if static_out.get("landmarks_json") else dynamic_out
        out = dict(base or self._empty())
        out["label"] = ""
        out["confidence"] = 0.0
        out["route"] = ROUTE_NONE
        out["route_reason"] = selected_reason
        out["router"] = self._router_payload(
            ROUTE_NONE,
            static_out,
            dynamic_out,
            static_assessment=static_assessment,
            dynamic_assessment=dynamic_assessment,
            selected_reason=selected_reason,
        )
        return out

    def _candidate(self, route: str, out: dict[str, Any]) -> RecognitionCandidate:
        return RecognitionCandidate(
            route=route,
            label=str(out.get("label") or "").strip(),
            confidence=float(out.get("confidence") or 0.0),
            output=out,
        )

    def _assess_dynamic_candidate(
        self,
        candidate: RecognitionCandidate,
    ) -> CandidateAssessment:
        gesture_type = self._gesture_type_for_label(candidate.label)
        if not candidate.label:
            return CandidateAssessment(
                candidate=candidate,
                gesture_type=gesture_type,
                accepted=False,
                reason=REASON_NO_LABEL,
            )
        if candidate.confidence < self._dynamic_confidence_threshold:
            return CandidateAssessment(
                candidate=candidate,
                gesture_type=gesture_type,
                accepted=False,
                reason=REASON_LOW_CONFIDENCE,
            )
        if gesture_type != GESTURE_TYPE_DYNAMIC:
            return CandidateAssessment(
                candidate=candidate,
                gesture_type=gesture_type,
                accepted=False,
                reason=REASON_NOT_DYNAMIC_TYPE,
            )
        return CandidateAssessment(
            candidate=candidate,
            gesture_type=gesture_type,
            accepted=True,
            reason=REASON_DYNAMIC_ACCEPTED,
        )

    def _assess_static_candidate(
        self,
        candidate: RecognitionCandidate,
    ) -> CandidateAssessment:
        gesture_type = self._gesture_type_for_label(candidate.label)
        if not candidate.label:
            return CandidateAssessment(
                candidate=candidate,
                gesture_type=gesture_type,
                accepted=False,
                reason=REASON_NO_LABEL,
            )
        if candidate.confidence < self._static_confidence_threshold:
            return CandidateAssessment(
                candidate=candidate,
                gesture_type=gesture_type,
                accepted=False,
                reason=REASON_LOW_CONFIDENCE,
            )
        if gesture_type == GESTURE_TYPE_DYNAMIC:
            return CandidateAssessment(
                candidate=candidate,
                gesture_type=gesture_type,
                accepted=False,
                reason=REASON_DYNAMIC_LABEL_REQUIRES_DYNAMIC_ROUTE,
            )
        return CandidateAssessment(
            candidate=candidate,
            gesture_type=gesture_type,
            accepted=True,
            reason=REASON_STATIC_FALLBACK,
        )

    def _gesture_type_for_label(self, label: str) -> str:
        clean = str(label or "").strip()
        if not clean:
            return ""
        return self._taxonomy.gesture_type_for_label(clean)

    def _with_route(
        self,
        candidate: RecognitionCandidate,
        static_out: dict[str, Any],
        dynamic_out: dict[str, Any],
        *,
        static_assessment: CandidateAssessment,
        dynamic_assessment: CandidateAssessment,
        selected_reason: str,
    ) -> dict[str, Any]:
        out = dict(candidate.output)
        out["label"] = candidate.label
        out["confidence"] = candidate.confidence
        out["route"] = candidate.route
        out["route_reason"] = selected_reason
        out["router"] = self._router_payload(
            candidate.route,
            static_out,
            dynamic_out,
            static_assessment=static_assessment,
            dynamic_assessment=dynamic_assessment,
            selected_reason=selected_reason,
        )
        return out

    def _router_payload(
        self,
        route: str,
        static_out: dict[str, Any],
        dynamic_out: dict[str, Any],
        *,
        static_assessment: CandidateAssessment,
        dynamic_assessment: CandidateAssessment,
        selected_reason: str,
    ) -> dict[str, Any]:
        dynamic_temporal = dynamic_out.get("temporal")
        if not isinstance(dynamic_temporal, dict):
            dynamic_temporal = {}
        dynamic_decision = dynamic_out.get("dynamic_decision")
        if not isinstance(dynamic_decision, dict):
            dynamic_decision = {}
        return {
            "route": route,
            "selected_reason": selected_reason,
            "static_label": str(static_out.get("label") or ""),
            "static_confidence": float(static_out.get("confidence") or 0.0),
            "static_type": static_assessment.gesture_type,
            "static_reject_reason": (
                "" if static_assessment.accepted else static_assessment.reason
            ),
            "static_threshold": self._static_confidence_threshold,
            "dynamic_label": str(dynamic_out.get("label") or ""),
            "dynamic_confidence": float(dynamic_out.get("confidence") or 0.0),
            "dynamic_type": dynamic_assessment.gesture_type,
            "dynamic_reject_reason": (
                "" if dynamic_assessment.accepted else dynamic_assessment.reason
            ),
            "dynamic_threshold": self._dynamic_confidence_threshold,
            "dynamic_phase": str(dynamic_temporal.get("phase") or ""),
            "dynamic_segment_frames": int(dynamic_temporal.get("frames") or 0),
            "dynamic_motion_scale": float(
                dynamic_temporal.get("motion_scale") or 0.0
            ),
            "dynamic_decision_source": str(dynamic_decision.get("source") or ""),
            "dynamic_motion_label": str(dynamic_decision.get("motion_label") or ""),
            "dynamic_motion_confidence": float(
                dynamic_decision.get("motion_confidence") or 0.0
            ),
            "dynamic_model_label": str(dynamic_decision.get("model_label") or ""),
            "dynamic_model_confidence": float(
                dynamic_decision.get("model_confidence") or 0.0
            ),
            "dynamic_model_confidence_for_motion": float(
                dynamic_decision.get("model_confidence_for_motion") or 0.0
            ),
            "dynamic_axis": str(dynamic_decision.get("axis") or ""),
            "dynamic_direction": str(dynamic_decision.get("direction") or ""),
            "dynamic_axis_ratio": float(dynamic_decision.get("axis_ratio") or 0.0),
            "dynamic_straightness": float(
                dynamic_decision.get("straightness") or 0.0
            ),
        }

    def _empty(self) -> dict[str, Any]:
        return {"label": "", "confidence": 0.0, "landmarks_json": "[]"}
