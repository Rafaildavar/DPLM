from collections import deque

import numpy as np

from app.gesture_online_infer import GestureOnlineInfer


class _NoHandsDetector:
    def detect_for_video_rgb(self, _frame_rgb):
        return []


class _CountingClassifier:
    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, _features):
        self.predict_calls += 1
        return np.asarray([0])

    def predict_proba(self, _features):
        return np.asarray([[1.0]])


def test_no_hand_frame_clears_window_and_does_not_predict() -> None:
    infer = object.__new__(GestureOnlineInfer)
    clf = _CountingClassifier()
    infer._detector = _NoHandsDetector()
    infer._clf = clf
    infer._classes = ["new"]
    infer._feature_dim = 42
    infer._classifier_two_hands = False
    infer._window = deque([np.ones(42, dtype=np.float32)], maxlen=30)

    out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert out == {"label": "", "confidence": 0.0, "landmarks_json": "[]"}
    assert len(infer._window) == 0
    assert clf.predict_calls == 0
