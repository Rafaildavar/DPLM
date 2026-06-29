from collections import deque

import numpy as np

from app.gesture_online_infer import GestureOnlineInfer
from cv.gesture_pose_signature import frame_non_thumb_count
from cv.hand_landmarker import DetectedHand


class _StaticDetector:
    def __init__(self, landmarks):
        self._landmarks = landmarks

    def detect_for_video_rgb(self, _frame_rgb):
        return [DetectedHand(landmarks=self._landmarks, handedness="Right", score=1.0)]


class _AlwaysThreeClassifier:
    def predict(self, _features):
        return np.asarray([0])

    def predict_proba(self, _features):
        return np.asarray([[0.96]])


def test_non_thumb_count_distinguishes_two_and_three_fingers() -> None:
    assert frame_non_thumb_count(_hand_with_extended_fingers(2)) == 2
    assert frame_non_thumb_count(_hand_with_extended_fingers(3)) == 3


def test_infer_rejects_prediction_when_finger_count_mismatches() -> None:
    infer = _infer_for_pose_guard(
        _hand_with_extended_fingers(2),
        signature={"non_thumb_count": 3, "stability": 1.0},
    )

    out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert out["label"] == ""
    assert out["confidence"] == 0.0
    assert len(infer._window) == 0


def test_infer_allows_prediction_when_signature_is_not_stable_enough() -> None:
    infer = _infer_for_pose_guard(
        _hand_with_extended_fingers(2),
        signature={"non_thumb_count": 3, "stability": 0.70},
    )

    out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert out["label"] == "three_fingers"
    assert out["confidence"] == 0.96


def test_infer_allows_prediction_when_live_finger_count_is_zero() -> None:
    infer = _infer_for_pose_guard(
        _hand_with_extended_fingers(0),
        signature={"non_thumb_count": 3, "stability": 1.0},
    )

    out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert out["label"] == "three_fingers"
    assert out["confidence"] == 0.96


def _infer_for_pose_guard(landmarks, *, signature):
    infer = object.__new__(GestureOnlineInfer)
    infer._detector = _StaticDetector(landmarks)
    infer._clf = _AlwaysThreeClassifier()
    infer._classes = ["three_fingers"]
    infer._feature_dim = 42
    infer._raw_feature_dim = 42
    infer._feature_mode = "static_mean"
    infer._classifier_two_hands = False
    infer._window = deque(maxlen=30)
    infer._finger_count_window = deque(maxlen=5)
    infer._gesture_signatures = {"three_fingers": signature}
    infer._gesture_rejection = {}
    infer._last_static_decision = {}
    return infer


def _hand_with_extended_fingers(count: int):
    landmarks = [[0.5, 0.7] for _ in range(21)]
    landmarks[0] = [0.5, 0.82]
    landmarks[1] = [0.42, 0.72]
    landmarks[2] = [0.38, 0.68]
    landmarks[3] = [0.35, 0.66]
    landmarks[4] = [0.33, 0.65]

    _set_finger(landmarks, 5, 0.44, extended=count >= 1)
    _set_finger(landmarks, 9, 0.50, extended=count >= 2)
    _set_finger(landmarks, 13, 0.56, extended=count >= 3)
    _set_finger(landmarks, 17, 0.62, extended=count >= 4)
    return landmarks


def _set_finger(landmarks, mcp_index: int, x: float, *, extended: bool) -> None:
    landmarks[mcp_index] = [x, 0.58]
    if extended:
        landmarks[mcp_index + 1] = [x, 0.43]
        landmarks[mcp_index + 2] = [x, 0.31]
        landmarks[mcp_index + 3] = [x, 0.20]
    else:
        landmarks[mcp_index + 1] = [x + 0.01, 0.62]
        landmarks[mcp_index + 2] = [x + 0.02, 0.66]
        landmarks[mcp_index + 3] = [x + 0.03, 0.69]
