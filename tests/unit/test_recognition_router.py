import numpy as np

from app.services.gesture_taxonomy import GestureTaxonomy
from app.services.recognition_router import (
    ROUTE_DYNAMIC,
    ROUTE_NONE,
    ROUTE_STATIC,
    REASON_DYNAMIC_LABEL_REQUIRES_DYNAMIC_ROUTE,
    REASON_DYNAMIC_OBSERVATION_PENDING,
    REASON_INTENT_DYNAMIC_PENDING,
    REASON_INTENT_NONE,
    REASON_INTENT_STATIC_FALLBACK,
    REASON_LOW_CONFIDENCE,
    REASON_NOT_DYNAMIC_TYPE,
    REASON_DYNAMIC_OVERRIDES_INTENT_GATE,
    REASON_STATIC_FALLBACK,
    GestureRecognitionRouter,
)


class _FakeInfer:
    def __init__(self, outputs, classes=None):
        self.outputs = list(outputs)
        self.calls = 0
        self.closed = False
        self.two_hands_values = []
        self.init_error = ""
        self.model_error = ""
        self.ready = True
        self.has_classifier = True
        self.two_hands = False
        self.classifier_requires_two_hands = False
        self.reset_calls = 0
        self.acknowledge_calls = 0
        self._classes = list(classes or [])

    def process_frame_rgb(self, _frame_rgb):
        self.calls += 1
        if self.outputs:
            return self.outputs.pop(0)
        return {"label": "", "confidence": 0.0, "landmarks_json": "[]"}

    def set_two_hands(self, enabled):
        self.two_hands_values.append(bool(enabled))

    def close(self):
        self.closed = True

    def reset_temporal_state(self):
        self.reset_calls += 1

    def acknowledge_dynamic_event(self):
        self.acknowledge_calls += 1


class _SharedFakeInfer(_FakeInfer):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.detect_calls = 0
        self.shared_process_calls = 0
        self.detected_hands = None

    def detect_hands(self, _frame_rgb):
        self.detect_calls += 1
        return ["shared-hand"]

    def process_detected_hands(self, hands):
        self.shared_process_calls += 1
        self.detected_hands = hands
        if self.outputs:
            return self.outputs.pop(0)
        return {"label": "", "confidence": 0.0, "landmarks_json": "[]"}


class _TimestampSharedFakeInfer(_SharedFakeInfer):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.timestamp_ms = None

    def detect_hands(self, _frame_rgb, timestamp_ms=None):
        self.detect_calls += 1
        self.timestamp_ms = timestamp_ms
        return ["shared-hand"]


class _FakeIntentGate:
    def __init__(self, label: str, confidence: float = 0.90):
        self.label = label
        self.confidence = confidence

    def predict_intent(self, _features):
        return {
            "label": self.label,
            "confidence": self.confidence,
            "margin": 0.40,
            "reason": "test",
        }


def _taxonomy() -> GestureTaxonomy:
    return GestureTaxonomy(
        source_path=__file__,
        label_to_type={
            "palm": "static",
            "hand_left": "quasi_static",
            "swipe_up": "dynamic",
        },
        patterns_by_type={},
    )


def test_router_prefers_confident_dynamic_label() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.90, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [{"label": "swipe_up", "confidence": 0.92, "landmarks_json": "[dynamic]"}]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == "swipe_up"
    assert out["confidence"] == 0.92
    assert out["route"] == ROUTE_DYNAMIC
    assert out["route_reason"] == "dynamic_accepted"
    assert out["router"]["static_label"] == "palm"
    assert out["router"]["dynamic_label"] == "swipe_up"
    assert out["router"]["static_type"] == "static"
    assert out["router"]["dynamic_type"] == "dynamic"
    assert static.calls == 1
    assert dynamic.calls == 1


def test_intent_gate_none_blocks_otherwise_valid_dynamic_label() -> None:
    static = _FakeInfer(
        [{"label": "", "confidence": 0.0, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [
            {
                "label": "swipe_up",
                "confidence": 0.95,
                "landmarks_json": "[dynamic]",
                "intent_features": [0.0, 1.0],
            }
        ]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
        intent_gate=_FakeIntentGate("none"),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["route"] == ROUTE_NONE
    assert out["route_reason"] == REASON_INTENT_NONE
    assert out["router"]["intent_gate_label"] == "none"
    assert out["router"]["intent_gate_accepted"] is True


def test_intent_gate_static_selects_static_over_dynamic_candidate() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.90, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [
            {
                "label": "swipe_up",
                "confidence": 0.95,
                "landmarks_json": "[dynamic]",
                "intent_features": [0.0, 1.0],
            }
        ]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
        intent_gate=_FakeIntentGate("static"),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == "palm"
    assert out["route"] == ROUTE_STATIC
    assert out["route_reason"] == REASON_INTENT_STATIC_FALLBACK


def test_intent_gate_none_does_not_block_completed_user_dynamic_model_label() -> None:
    static = _FakeInfer(
        [{"label": "", "confidence": 0.0, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [
            {
                "label": "SwipeLeft",
                "confidence": 0.98,
                "landmarks_json": "[dynamic]",
                "intent_features": [0.0, 1.0],
                "dynamic_decision": {
                    "source": "motion_over_prototype_reject",
                    "motion_label": "SwipeLeft",
                    "model_label": "SwipeLeft",
                },
                "temporal": {
                    "enabled": True,
                    "phase": "completed",
                    "end_reason": "velocity_drop",
                },
            }
        ],
        classes=["SwipeLeft", "no_gesture_static"],
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
        intent_gate=_FakeIntentGate("none"),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == "SwipeLeft"
    assert out["route"] == ROUTE_DYNAMIC
    assert out["route_reason"] == REASON_DYNAMIC_OVERRIDES_INTENT_GATE
    assert out["router"]["dynamic_type"] == "dynamic"
    assert out["router"]["intent_gate_label"] == "none"


def test_intent_gate_static_does_not_hide_completed_dynamic_result() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.90, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [
            {
                "label": "SwipeLeft",
                "confidence": 0.98,
                "landmarks_json": "[dynamic]",
                "intent_features": [0.0, 1.0],
                "dynamic_decision": {
                    "source": "motion_and_model_agree",
                    "motion_label": "SwipeLeft",
                    "model_label": "SwipeLeft",
                },
                "temporal": {
                    "enabled": True,
                    "phase": "completed",
                    "end_reason": "pending_repeat",
                },
            }
        ],
        classes=["SwipeLeft", "no_gesture_static"],
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
        intent_gate=_FakeIntentGate("static"),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == "SwipeLeft"
    assert out["route"] == ROUTE_DYNAMIC
    assert out["route_reason"] == REASON_DYNAMIC_OVERRIDES_INTENT_GATE
    assert out["router"]["static_label"] == "palm"
    assert out["router"]["intent_gate_label"] == "static"


def test_intent_gate_dynamic_holds_static_until_dynamic_candidate_arrives() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.90, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [
            {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": "[dynamic]",
                "intent_features": [0.0, 1.0],
            }
        ]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
        intent_gate=_FakeIntentGate("dynamic"),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["route"] == ROUTE_NONE
    assert out["route_reason"] == REASON_INTENT_DYNAMIC_PENDING


def test_router_accepts_unlisted_dynamic_channel_label() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.90, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [
            {
                "label": "circle_clockwise",
                "confidence": 0.92,
                "landmarks_json": "[dynamic]",
            }
        ]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == "circle_clockwise"
    assert out["route"] == ROUTE_DYNAMIC
    assert out["router"]["dynamic_type"] == "dynamic"
    assert out["route_reason"] == "dynamic_accepted"


def test_router_runs_one_shared_detection_for_both_models() -> None:
    static = _SharedFakeInfer(
        [{"label": "palm", "confidence": 0.90, "landmarks_json": "[shared]"}]
    )
    dynamic = _SharedFakeInfer(
        [{"label": "swipe_up", "confidence": 0.92, "landmarks_json": "[shared]"}]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == "swipe_up"
    assert static.detect_calls == 1
    assert dynamic.detect_calls == 0
    assert static.shared_process_calls == 1
    assert dynamic.shared_process_calls == 1
    assert static.detected_hands is dynamic.detected_hands
    assert static.calls == 0
    assert dynamic.calls == 0
    assert out["performance"]["shared_detection"] is True


def test_router_passes_timestamp_and_logs_shared_hand_metrics() -> None:
    static = _TimestampSharedFakeInfer(
        [
            {
                "label": "palm",
                "confidence": 0.90,
                "landmarks_json": "[shared]",
                "hand_tracking": {
                    "mediapipe_profile": "recall_05",
                    "mediapipe_timestamp_source": "real_monotonic",
                    "hand_detected": True,
                    "hand_count": 1,
                },
            }
        ]
    )
    dynamic = _SharedFakeInfer(
        [{"label": "", "confidence": 0.0, "landmarks_json": "[shared]"}]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(
        np.zeros((8, 8, 3), dtype=np.uint8),
        timestamp_ms=1234,
    )

    assert static.timestamp_ms == 1234
    assert out["performance"]["mediapipe_profile"] == "recall_05"
    assert out["performance"]["mediapipe_timestamp_source"] == "real_monotonic"
    assert out["router"]["mediapipe_profile"] == "recall_05"
    assert out["router"]["shared_detection"] is True
    assert out["router"]["dynamic_phase"] == ""
    assert out["router"]["dynamic_motion_scale"] == 0.0


def test_router_rejects_dynamic_model_quasi_static_label() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.90, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [{"label": "hand_left", "confidence": 0.99, "landmarks_json": "[dynamic]"}]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == "palm"
    assert out["route"] == ROUTE_STATIC
    assert out["route_reason"] == REASON_STATIC_FALLBACK
    assert out["router"]["dynamic_label"] == "hand_left"
    assert out["router"]["dynamic_type"] == "quasi_static"
    assert out["router"]["dynamic_reject_reason"] == REASON_NOT_DYNAMIC_TYPE


def test_router_rejects_low_confidence_dynamic_label() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.90, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [{"label": "swipe_up", "confidence": 0.40, "landmarks_json": "[dynamic]"}]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == "palm"
    assert out["route"] == ROUTE_STATIC
    assert out["router"]["dynamic_reject_reason"] == REASON_LOW_CONFIDENCE


def test_router_release_dynamic_threshold_rejects_below_90_percent() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.90, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [{"label": "swipe_up", "confidence": 0.89, "landmarks_json": "[dynamic]"}]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["route"] == ROUTE_STATIC
    assert out["router"]["dynamic_reject_reason"] == REASON_LOW_CONFIDENCE
    assert out["router"]["dynamic_threshold"] == 0.90


def test_router_release_dynamic_threshold_accepts_90_percent() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.90, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [{"label": "swipe_up", "confidence": 0.90, "landmarks_json": "[dynamic]"}]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["route"] == ROUTE_DYNAMIC
    assert out["label"] == "swipe_up"
    assert out["confidence"] == 0.90


def test_dynamic_intent_override_cannot_bypass_release_threshold() -> None:
    static = _FakeInfer(
        [{"label": "", "confidence": 0.0, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [
            {
                "label": "SwipeLeft",
                "confidence": 0.89,
                "landmarks_json": "[dynamic]",
                "intent_features": [0.0, 1.0],
                "dynamic_decision": {
                    "source": "motion_and_model_agree",
                    "motion_label": "SwipeLeft",
                    "model_label": "SwipeLeft",
                },
                "temporal": {
                    "enabled": True,
                    "phase": "completed",
                    "end_reason": "velocity_drop",
                },
            }
        ],
        classes=["SwipeLeft", "no_gesture_static"],
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
        dynamic_confidence_threshold=0.80,
        intent_gate=_FakeIntentGate("none"),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["route"] == ROUTE_NONE
    assert out["route_reason"] == REASON_INTENT_NONE


def test_router_release_static_threshold_rejects_below_80_percent() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.79, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [{"label": "", "confidence": 0.0, "landmarks_json": "[dynamic]"}]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["route"] == ROUTE_NONE
    assert out["router"]["static_reject_reason"] == REASON_LOW_CONFIDENCE
    assert out["router"]["static_threshold"] == 0.80


def test_router_release_static_threshold_accepts_80_percent() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.80, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [{"label": "", "confidence": 0.0, "landmarks_json": "[dynamic]"}]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["route"] == ROUTE_STATIC
    assert out["label"] == "palm"
    assert out["confidence"] == 0.80


def test_router_rejects_low_confidence_static_label() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.40, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [{"label": "", "confidence": 0.0, "landmarks_json": "[dynamic]"}]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == ""
    assert out["confidence"] == 0.0
    assert out["route"] == ROUTE_NONE
    assert out["router"]["static_reject_reason"] == REASON_LOW_CONFIDENCE


def test_router_exposes_static_rejection_policy_metadata() -> None:
    static = _FakeInfer(
        [
            {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": "[static]",
                "static_decision": {
                    "source": "margin_rejected",
                    "rejection_reason": "low_margin",
                    "model_label": "palm",
                    "model_confidence": 0.53,
                    "top2_label": "gun",
                    "top2_confidence": 0.47,
                    "margin": 0.06,
                    "min_margin": 0.10,
                },
            }
        ]
    )
    dynamic = _FakeInfer([{"label": "", "confidence": 0.0, "landmarks_json": "[]"}])
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["route"] == ROUTE_NONE
    assert out["router"]["static_reject_reason"] == "low_margin"
    assert out["router"]["static_decision_source"] == "margin_rejected"
    assert out["router"]["static_model_label"] == "palm"
    assert out["router"]["static_margin"] == 0.06


def test_router_rejects_static_dynamic_label_without_dynamic_confirmation() -> None:
    static = _FakeInfer(
        [{"label": "swipe_up", "confidence": 0.99, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [{"label": "", "confidence": 0.0, "landmarks_json": "[dynamic]"}]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == ""
    assert out["confidence"] == 0.0
    assert out["route"] == ROUTE_NONE
    assert out["router"]["static_type"] == "dynamic"
    assert (
        out["router"]["static_reject_reason"]
        == REASON_DYNAMIC_LABEL_REQUIRES_DYNAMIC_ROUTE
    )


def test_router_returns_none_with_landmarks_when_no_candidate() -> None:
    static = _FakeInfer(
        [{"label": "", "confidence": 0.0, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [{"label": "", "confidence": 0.0, "landmarks_json": "[dynamic]"}]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == ""
    assert out["confidence"] == 0.0
    assert out["landmarks_json"] == "[static]"
    assert out["route"] == ROUTE_NONE
    assert out["route_reason"] == "no_valid_candidate"


def test_router_allows_static_while_dynamic_channel_is_warming_up() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.99, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [
            {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": "[dynamic]",
                "temporal": {
                    "enabled": True,
                    "phase": "warming_up",
                    "frames": 6,
                    "required_frames": 36,
                },
            }
        ]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == "palm"
    assert out["route"] == ROUTE_STATIC
    assert out["route_reason"] == REASON_STATIC_FALLBACK
    assert out["router"]["static_label"] == "palm"


def test_router_holds_static_while_dynamic_channel_is_active() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.99, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [
            {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": "[dynamic]",
                "temporal": {
                    "enabled": True,
                    "phase": "active",
                    "frames": 8,
                    "required_frames": 36,
                },
            }
        ]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == ""
    assert out["route"] == ROUTE_NONE
    assert out["route_reason"] == REASON_DYNAMIC_OBSERVATION_PENDING
    assert out["router"]["static_label"] == "palm"


def test_router_allows_static_after_dynamic_channel_becomes_idle() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.99, "landmarks_json": "[static]"}]
    )
    dynamic = _FakeInfer(
        [
            {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": "[dynamic]",
                "temporal": {
                    "enabled": True,
                    "phase": "idle",
                    "frames": 36,
                    "required_frames": 36,
                },
            }
        ]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == "palm"
    assert out["route"] == ROUTE_STATIC


def test_router_forwards_lifecycle_calls() -> None:
    static = _FakeInfer([])
    dynamic = _FakeInfer([])
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    router.set_two_hands(True)
    router.reset_temporal_state()
    router.acknowledge_dynamic_event()
    router.close()

    assert static.two_hands_values == [True]
    assert dynamic.two_hands_values == [True]
    assert dynamic.reset_calls == 1
    assert dynamic.acknowledge_calls == 1
    assert static.closed is True
    assert dynamic.closed is True
