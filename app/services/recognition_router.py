"""Runtime routing between static and dynamic gesture recognizers."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.services.gesture_taxonomy import (
    GESTURE_TYPE_DYNAMIC,
    GESTURE_TYPE_NEGATIVE,
    GESTURE_TYPE_QUASI_STATIC,
    GESTURE_TYPE_STATIC,
    GestureTaxonomy,
    load_gesture_taxonomy,
)
from cv.intent_gate import (
    DEFAULT_INTENT_CONFIDENCE_THRESHOLD,
    INTENT_DYNAMIC,
    INTENT_NONE,
    INTENT_STATIC,
    load_intent_gate_model,
    predict_intent_gate,
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
REASON_INTENT_NONE = "intent_none"
REASON_INTENT_DYNAMIC_PENDING = "intent_dynamic_pending"
REASON_INTENT_STATIC_FALLBACK = "intent_static_fallback"
REASON_INTENT_STATIC_NO_CANDIDATE = "intent_static_no_candidate"
REASON_DYNAMIC_OVERRIDES_INTENT_GATE = "dynamic_overrides_intent_gate"
DEFAULT_DYNAMIC_CONFIDENCE_THRESHOLD = 0.60
DEFAULT_STATIC_CONFIDENCE_THRESHOLD = 0.80
DYNAMIC_INTENT_OVERRIDE_CONFIDENCE_THRESHOLD = 0.85


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
        intent_gate: Any | None = None,
        intent_gate_path: str | None = None,
        intent_confidence_threshold: float = DEFAULT_INTENT_CONFIDENCE_THRESHOLD,
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
        self._intent_gate = intent_gate or self._load_intent_gate(intent_gate_path)
        self._intent_confidence_threshold = max(
            0.0,
            min(1.0, float(intent_confidence_threshold)),
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

    @property
    def static_rejection_method(self) -> str:
        return str(getattr(self._static_infer, "static_rejection_method", "") or "")

    def set_static_rejection_method(self, method: str) -> None:
        setter = getattr(self._static_infer, "set_static_rejection_method", None)
        if callable(setter):
            setter(method)

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

    def process_frame_rgb(
        self,
        frame_rgb: Any,
        timestamp_ms: int | float | None = None,
    ) -> dict[str, Any]:
        started = perf_counter()
        shared_detection = self._supports_shared_detection()
        detection_ms = 0.0

        if shared_detection:
            detection_started = perf_counter()
            try:
                hands = self._call_detect_hands(
                    self._static_infer,
                    frame_rgb,
                    timestamp_ms=timestamp_ms,
                )
            except Exception as exc:
                print(f"[w] shared hand detection failed: {exc}", flush=True)
                hands = []
            detection_ms = (perf_counter() - detection_started) * 1000.0
            static_out = self._process_detected(self._static_infer, hands)
            dynamic_out = self._process_detected(self._dynamic_infer, hands)
        else:
            static_out = self._process(
                self._static_infer,
                frame_rgb,
                timestamp_ms=timestamp_ms,
            )
            dynamic_out = self._process(
                self._dynamic_infer,
                frame_rgb,
                timestamp_ms=timestamp_ms,
            )

        out = self._choose(static_out, dynamic_out)
        hand_tracking = self._hand_tracking_from_outputs(
            static_out,
            dynamic_out,
            out,
        )
        performance = {
            "shared_detection": shared_detection,
            "detection_ms": round(detection_ms, 3),
            "total_inference_ms": round((perf_counter() - started) * 1000.0, 3),
        }
        performance.update(hand_tracking)
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

    def _process(
        self,
        infer: Any | None,
        frame_rgb: Any,
        timestamp_ms: int | float | None = None,
    ) -> dict[str, Any]:
        if infer is None:
            return self._empty()
        try:
            if timestamp_ms is None:
                out = infer.process_frame_rgb(frame_rgb)
            else:
                try:
                    out = infer.process_frame_rgb(frame_rgb, timestamp_ms=timestamp_ms)
                except TypeError as exc:
                    if "timestamp" not in str(exc):
                        raise
                    out = infer.process_frame_rgb(frame_rgb)
        except Exception as exc:
            print(f"[w] recognition route failed: {exc}", flush=True)
            return self._empty()
        if not isinstance(out, dict):
            return self._empty()
        return out

    def _call_detect_hands(
        self,
        infer: Any,
        frame_rgb: Any,
        *,
        timestamp_ms: int | float | None,
    ) -> Any:
        if timestamp_ms is None:
            return infer.detect_hands(frame_rgb)
        try:
            return infer.detect_hands(frame_rgb, timestamp_ms=timestamp_ms)
        except TypeError as exc:
            if "timestamp" not in str(exc):
                raise
            return infer.detect_hands(frame_rgb)

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
        intent_decision = self._intent_decision(static_out, dynamic_out)

        if intent_decision.get("accepted"):
            if (
                dynamic_assessment.accepted
                and self._dynamic_can_override_intent_gate(
                    dynamic_candidate,
                    dynamic_out,
                    intent_decision,
                )
            ):
                return self._with_route(
                    dynamic_candidate,
                    static_out,
                    dynamic_out,
                    static_assessment=static_assessment,
                    dynamic_assessment=dynamic_assessment,
                    selected_reason=REASON_DYNAMIC_OVERRIDES_INTENT_GATE,
                    intent_decision=intent_decision,
                )

            intent_label = str(intent_decision.get("label") or "")
            if intent_label == INTENT_NONE:
                return self._without_candidate(
                    static_out,
                    dynamic_out,
                    static_assessment=static_assessment,
                    dynamic_assessment=dynamic_assessment,
                    selected_reason=REASON_INTENT_NONE,
                    intent_decision=intent_decision,
                )
            if intent_label == INTENT_DYNAMIC:
                if dynamic_assessment.accepted:
                    return self._with_route(
                        dynamic_candidate,
                        static_out,
                        dynamic_out,
                        static_assessment=static_assessment,
                        dynamic_assessment=dynamic_assessment,
                        selected_reason=REASON_DYNAMIC_ACCEPTED,
                        intent_decision=intent_decision,
                    )
                return self._without_candidate(
                    static_out,
                    dynamic_out,
                    static_assessment=static_assessment,
                    dynamic_assessment=dynamic_assessment,
                    selected_reason=REASON_INTENT_DYNAMIC_PENDING,
                    intent_decision=intent_decision,
                )
            if intent_label == INTENT_STATIC:
                if static_assessment.accepted:
                    return self._with_route(
                        static_candidate,
                        static_out,
                        dynamic_out,
                        static_assessment=static_assessment,
                        dynamic_assessment=dynamic_assessment,
                        selected_reason=REASON_INTENT_STATIC_FALLBACK,
                        intent_decision=intent_decision,
                    )
                return self._without_candidate(
                    static_out,
                    dynamic_out,
                    static_assessment=static_assessment,
                    dynamic_assessment=dynamic_assessment,
                    selected_reason=REASON_INTENT_STATIC_NO_CANDIDATE,
                    intent_decision=intent_decision,
                )

        if dynamic_assessment.accepted:
            return self._with_route(
                dynamic_candidate,
                static_out,
                dynamic_out,
                static_assessment=static_assessment,
                dynamic_assessment=dynamic_assessment,
                selected_reason=REASON_DYNAMIC_ACCEPTED,
                intent_decision=intent_decision,
            )
        if static_assessment.accepted and self._should_hold_static(dynamic_out):
            return self._without_candidate(
                static_out,
                dynamic_out,
                static_assessment=static_assessment,
                dynamic_assessment=dynamic_assessment,
                selected_reason=REASON_DYNAMIC_OBSERVATION_PENDING,
                intent_decision=intent_decision,
            )
        if static_assessment.accepted:
            return self._with_route(
                static_candidate,
                static_out,
                dynamic_out,
                static_assessment=static_assessment,
                dynamic_assessment=dynamic_assessment,
                selected_reason=REASON_STATIC_FALLBACK,
                intent_decision=intent_decision,
            )

        return self._without_candidate(
            static_out,
            dynamic_out,
            static_assessment=static_assessment,
            dynamic_assessment=dynamic_assessment,
            selected_reason=REASON_NO_VALID_CANDIDATE,
            intent_decision=intent_decision,
        )

    def _should_hold_static(self, dynamic_out: dict[str, Any]) -> bool:
        temporal = dynamic_out.get("temporal")
        if not isinstance(temporal, dict) or not bool(temporal.get("enabled")):
            return False
        return str(temporal.get("phase") or "") in {
            "active",
            "cooldown",
            "hand_lost_grace",
        }

    def _without_candidate(
        self,
        static_out: dict[str, Any],
        dynamic_out: dict[str, Any],
        *,
        static_assessment: CandidateAssessment,
        dynamic_assessment: CandidateAssessment,
        selected_reason: str,
        intent_decision: dict[str, Any] | None = None,
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
            intent_decision=intent_decision,
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
        gesture_type = self._gesture_type_for_dynamic_candidate(candidate.label)
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

    def _gesture_type_for_dynamic_candidate(self, label: str) -> str:
        clean = str(label or "").strip().lower()
        if not clean:
            return ""

        base_type = self._gesture_type_for_label(clean)
        if base_type == GESTURE_TYPE_DYNAMIC:
            return base_type

        declared_type = self._declared_taxonomy_type(clean)
        if declared_type:
            return declared_type

        if self._looks_like_negative_label(clean):
            return GESTURE_TYPE_NEGATIVE

        if self._label_belongs_to_dynamic_model(clean):
            return GESTURE_TYPE_DYNAMIC

        if base_type == GESTURE_TYPE_STATIC:
            return GESTURE_TYPE_DYNAMIC
        return base_type

    def _label_belongs_to_dynamic_model(self, label: str) -> bool:
        clean = str(label or "").strip().lower()
        if not clean:
            return False
        compact = self._compact_label(clean)
        raw_classes = getattr(self._dynamic_infer, "_classes", None)
        if raw_classes is None:
            raw_classes = getattr(self._dynamic_infer, "classes", ())
        for raw_label in raw_classes or ():
            candidate = str(raw_label or "").strip().lower()
            if not candidate:
                continue
            if candidate == clean or self._compact_label(candidate) == compact:
                return True
        return False

    @staticmethod
    def _compact_label(label: str) -> str:
        return "".join(ch for ch in str(label or "").lower() if ch.isalnum())

    def _dynamic_can_override_intent_gate(
        self,
        candidate: RecognitionCandidate,
        dynamic_out: dict[str, Any],
        intent_decision: dict[str, Any],
    ) -> bool:
        intent_label = str(intent_decision.get("label") or "")
        if intent_label not in {INTENT_NONE, INTENT_STATIC}:
            return False
        if candidate.confidence < DYNAMIC_INTENT_OVERRIDE_CONFIDENCE_THRESHOLD:
            return False

        temporal = dynamic_out.get("temporal")
        if not isinstance(temporal, dict) or not bool(temporal.get("enabled")):
            return False
        phase = str(temporal.get("phase") or "")
        if phase not in {"completed", "cooldown"}:
            return False

        decision = dynamic_out.get("dynamic_decision")
        if not isinstance(decision, dict):
            return False
        source = str(decision.get("source") or "")
        if not source:
            return False
        return source not in {
            "negative_rejected",
            "prototype_rejected",
            "prototype_motion_conflict",
            "rejected",
            "motion_fallback_suppressed_for_custom_labels",
        }

    def _declared_taxonomy_type(self, label: str) -> str:
        clean = str(label or "").strip().lower()
        if not clean:
            return ""

        explicit = getattr(self._taxonomy, "label_to_type", {}).get(clean)
        if explicit:
            return str(explicit)

        patterns_by_type = getattr(self._taxonomy, "patterns_by_type", {})
        if not isinstance(patterns_by_type, dict):
            return ""
        for gesture_type in (
            GESTURE_TYPE_NEGATIVE,
            GESTURE_TYPE_QUASI_STATIC,
            GESTURE_TYPE_STATIC,
            GESTURE_TYPE_DYNAMIC,
        ):
            for pattern in patterns_by_type.get(gesture_type, ()):
                if fnmatch.fnmatchcase(clean, str(pattern)):
                    return gesture_type
        return ""

    @staticmethod
    def _looks_like_negative_label(label: str) -> bool:
        clean = str(label or "").strip().lower()
        return clean.startswith(
            (
                "background_",
                "negative_",
                "no_gesture",
                "partial_",
                "random_",
                "return_",
                "wrong_axis_",
            )
        )

    def _with_route(
        self,
        candidate: RecognitionCandidate,
        static_out: dict[str, Any],
        dynamic_out: dict[str, Any],
        *,
        static_assessment: CandidateAssessment,
        dynamic_assessment: CandidateAssessment,
        selected_reason: str,
        intent_decision: dict[str, Any] | None = None,
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
            intent_decision=intent_decision,
        )
        return out

    def _load_intent_gate(self, path: str | None) -> Any | None:
        if not path:
            return None
        try:
            from pathlib import Path

            source = Path(path)
            if not source.exists():
                return None
            return load_intent_gate_model(source)
        except Exception as exc:
            print(f"[w] intent gate ignored: {exc}", flush=True)
            return None

    def _intent_decision(
        self,
        static_out: dict[str, Any],
        dynamic_out: dict[str, Any],
    ) -> dict[str, Any]:
        gate = getattr(self, "_intent_gate", None)
        if gate is None:
            return {"enabled": False, "accepted": False, "reason": "missing"}
        raw_features = dynamic_out.get("intent_features") or static_out.get(
            "intent_features"
        )
        if not raw_features:
            return {"enabled": True, "accepted": False, "reason": "no_features"}
        try:
            if hasattr(gate, "predict_intent"):
                decision = gate.predict_intent(raw_features)
            else:
                decision = predict_intent_gate(gate, raw_features)
        except Exception as exc:
            return {
                "enabled": True,
                "accepted": False,
                "reason": f"predict_failed:{exc}",
            }
        if not isinstance(decision, dict):
            return {"enabled": True, "accepted": False, "reason": "bad_decision"}
        confidence = float(decision.get("confidence") or 0.0)
        label = str(decision.get("label") or "")
        out = dict(decision)
        out.update(
            {
                "enabled": True,
                "label": label,
                "confidence": confidence,
                "threshold": self._intent_confidence_threshold,
                "accepted": bool(
                    label in {INTENT_STATIC, INTENT_DYNAMIC, INTENT_NONE}
                    and confidence >= self._intent_confidence_threshold
                ),
            }
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
        intent_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        dynamic_temporal = dynamic_out.get("temporal")
        if not isinstance(dynamic_temporal, dict):
            dynamic_temporal = {}
        dynamic_decision = dynamic_out.get("dynamic_decision")
        if not isinstance(dynamic_decision, dict):
            dynamic_decision = {}
        static_decision = static_out.get("static_decision")
        if not isinstance(static_decision, dict):
            static_decision = {}
        static_runtime_reject_reason = str(
            static_decision.get("rejection_reason") or ""
        )
        hand_tracking = self._hand_tracking_from_outputs(static_out, dynamic_out)

        def _decision_float(decision: dict[str, Any], key: str) -> float:
            try:
                return float(decision.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        intent = intent_decision if isinstance(intent_decision, dict) else {}
        return {
            "route": route,
            "selected_reason": selected_reason,
            "intent_gate_enabled": bool(intent.get("enabled")),
            "intent_gate_label": str(intent.get("label") or ""),
            "intent_gate_confidence": float(intent.get("confidence") or 0.0),
            "intent_gate_margin": float(intent.get("margin") or 0.0),
            "intent_gate_threshold": float(
                intent.get("threshold") or self._intent_confidence_threshold
            ),
            "intent_gate_accepted": bool(intent.get("accepted")),
            "intent_gate_reason": str(intent.get("reason") or ""),
            "static_label": str(static_out.get("label") or ""),
            "static_confidence": float(static_out.get("confidence") or 0.0),
            "static_type": static_assessment.gesture_type,
            "static_reject_reason": (
                ""
                if static_assessment.accepted
                else static_runtime_reject_reason or static_assessment.reason
            ),
            "static_threshold": self._static_confidence_threshold,
            "static_decision_source": str(static_decision.get("source") or ""),
            "static_rejection_method": str(
                static_decision.get("rejection_method") or ""
            ),
            "static_model_label": str(static_decision.get("model_label") or ""),
            "static_model_confidence": _decision_float(
                static_decision,
                "model_confidence",
            ),
            "static_top2_label": str(static_decision.get("top2_label") or ""),
            "static_top2_confidence": _decision_float(
                static_decision,
                "top2_confidence",
            ),
            "static_margin": _decision_float(static_decision, "margin"),
            "static_min_margin": _decision_float(static_decision, "min_margin"),
            "static_negative_label": str(static_decision.get("negative_label") or ""),
            "static_negative_confidence": _decision_float(
                static_decision,
                "negative_confidence",
            ),
            "static_negative_threshold": _decision_float(
                static_decision,
                "negative_threshold",
            ),
            "static_prototype_distance": _decision_float(
                static_decision,
                "prototype_distance",
            ),
            "static_prototype_radius": _decision_float(
                static_decision,
                "prototype_radius",
            ),
            "static_prototype_threshold": _decision_float(
                static_decision,
                "prototype_threshold",
            ),
            "static_verifier_method": str(
                static_decision.get("verifier_method") or ""
            ),
            "static_verifier_label": str(
                static_decision.get("verifier_label") or ""
            ),
            "static_verifier_probability": _decision_float(
                static_decision,
                "verifier_probability",
            ),
            "static_verifier_confidence": _decision_float(
                static_decision,
                "verifier_confidence",
            ),
            "static_verifier_score": _decision_float(
                static_decision,
                "verifier_score",
            ),
            "static_verifier_distance": _decision_float(
                static_decision,
                "verifier_distance",
            ),
            "static_verifier_threshold": _decision_float(
                static_decision,
                "verifier_threshold",
            ),
            "dynamic_label": str(dynamic_out.get("label") or ""),
            "dynamic_confidence": float(dynamic_out.get("confidence") or 0.0),
            "dynamic_type": dynamic_assessment.gesture_type,
            "dynamic_reject_reason": (
                "" if dynamic_assessment.accepted else dynamic_assessment.reason
            ),
            "dynamic_threshold": self._dynamic_confidence_threshold,
            "dynamic_phase": str(dynamic_temporal.get("phase") or ""),
            "dynamic_end_reason": str(dynamic_temporal.get("end_reason") or ""),
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
            "dynamic_prototype_method": str(
                dynamic_decision.get("prototype_method") or ""
            ),
            "dynamic_prototype_label": str(
                dynamic_decision.get("prototype_label") or ""
            ),
            "dynamic_prototype_nearest_type": str(
                dynamic_decision.get("prototype_nearest_type") or ""
            ),
            "dynamic_prototype_confidence": float(
                dynamic_decision.get("prototype_confidence") or 0.0
            ),
            "dynamic_prototype_distance": float(
                dynamic_decision.get("prototype_distance") or 0.0
            ),
            "dynamic_prototype_threshold": float(
                dynamic_decision.get("prototype_threshold") or 0.0
            ),
            "dynamic_prototype_margin": float(
                dynamic_decision.get("prototype_margin") or 0.0
            ),
            "dynamic_prototype_reason": str(
                dynamic_decision.get("prototype_reason") or ""
            ),
            "dynamic_negative_label": str(
                dynamic_decision.get("negative_label") or ""
            ),
            "dynamic_negative_confidence": float(
                dynamic_decision.get("negative_confidence") or 0.0
            ),
            "dynamic_negative_threshold": float(
                dynamic_decision.get("negative_threshold") or 0.0
            ),
            "dynamic_axis": str(dynamic_decision.get("axis") or ""),
            "dynamic_direction": str(dynamic_decision.get("direction") or ""),
            "dynamic_axis_ratio": float(dynamic_decision.get("axis_ratio") or 0.0),
            "dynamic_straightness": float(
                dynamic_decision.get("straightness") or 0.0
            ),
            **hand_tracking,
        }

    def _empty(self) -> dict[str, Any]:
        return {"label": "", "confidence": 0.0, "landmarks_json": "[]"}

    def _hand_tracking_from_outputs(
        self,
        *outputs: dict[str, Any],
    ) -> dict[str, Any]:
        for output in outputs:
            if not isinstance(output, dict):
                continue
            payload = output.get("hand_tracking")
            if isinstance(payload, dict) and payload:
                return self._flatten_hand_tracking(payload)
        infer = getattr(self, "_static_infer", None)
        payload = getattr(infer, "last_hand_tracking_metrics", {})
        if isinstance(payload, dict) and payload:
            return self._flatten_hand_tracking(payload)
        return {}

    @staticmethod
    def _flatten_hand_tracking(payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "mediapipe_profile",
            "mediapipe_min_detection_confidence",
            "mediapipe_min_presence_confidence",
            "mediapipe_min_tracking_confidence",
            "mediapipe_smoothing_alpha",
            "mediapipe_timestamp_source",
            "mediapipe_detection_ms",
            "hand_detected",
            "hand_count",
            "handedness_score_max",
            "landmark_z_available",
            "world_landmarks_available",
            "landmark_z_range",
            "world_z_range",
            "hand_bbox_area",
            "hand_bbox_diag",
            "primary_wrist_step",
            "hand_lost_streak",
            "hand_lost_grace_frames",
        }
        return {key: payload.get(key) for key in allowed if key in payload}
