import numpy as np

from app.services.gesture_taxonomy import GestureTaxonomy
from app.services.recognition_router import (
    ROUTE_DYNAMIC,
    ROUTE_NONE,
    ROUTE_STATIC,
    GestureRecognitionRouter,
)


class _FakeInfer:
    def __init__(self, outputs):
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

    def process_frame_rgb(self, _frame_rgb):
        self.calls += 1
        if self.outputs:
            return self.outputs.pop(0)
        return {"label": "", "confidence": 0.0, "landmarks_json": "[]"}

    def set_two_hands(self, enabled):
        self.two_hands_values.append(bool(enabled))

    def close(self):
        self.closed = True


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
        [{"label": "swipe_up", "confidence": 0.88, "landmarks_json": "[dynamic]"}]
    )
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    out = router.process_frame_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert out["label"] == "swipe_up"
    assert out["confidence"] == 0.88
    assert out["route"] == ROUTE_DYNAMIC
    assert out["router"]["static_label"] == "palm"
    assert out["router"]["dynamic_label"] == "swipe_up"
    assert static.calls == 1
    assert dynamic.calls == 1


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
    assert out["router"]["dynamic_label"] == "hand_left"


def test_router_rejects_low_confidence_dynamic_label() -> None:
    static = _FakeInfer(
        [{"label": "palm", "confidence": 0.70, "landmarks_json": "[static]"}]
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


def test_router_forwards_lifecycle_calls() -> None:
    static = _FakeInfer([])
    dynamic = _FakeInfer([])
    router = GestureRecognitionRouter(
        static_infer=static,
        dynamic_infer=dynamic,
        taxonomy=_taxonomy(),
    )

    router.set_two_hands(True)
    router.close()

    assert static.two_hands_values == [True]
    assert dynamic.two_hands_values == [True]
    assert static.closed is True
    assert dynamic.closed is True
