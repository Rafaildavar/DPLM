from collections import deque
from pathlib import Path

import numpy as np

from app.gesture_online_infer import GestureOnlineInfer
from cv.gesture_features import (
    FEATURE_DYNAMIC_CRAFT_FULL_STATS,
    FEATURE_DYNAMIC_CRAFT_STATS,
    FEATURE_DYNAMIC_LANDMARK_IMAGE,
    FEATURE_DYNAMIC_SEQUENCE,
    FEATURE_DYNAMIC_SEQUENCE_72,
    FEATURE_STATIC_CRAFT_FULL_STATS,
    FEATURE_STATIC_LANDMARK_IMAGE,
    feature_vector_size,
)
from cv.hand_landmarker import DetectedHand


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


def test_dynamic_sequence_artifact_forces_sequence_feature_mode():
    infer = GestureOnlineInfer.__new__(GestureOnlineInfer)
    infer._feature_dim = 1584
    infer._feature_mode = "static_mean"

    infer._ensure_dynamic_sequence_feature_mode(Path("dynamic_sequence_rocket.pkl"))

    assert infer._feature_mode == FEATURE_DYNAMIC_SEQUENCE


def test_long_dynamic_sequence_artifact_forces_long_sequence_feature_mode():
    infer = GestureOnlineInfer.__new__(GestureOnlineInfer)
    infer._feature_dim = 3168
    infer._feature_mode = "static_mean"

    infer._ensure_dynamic_sequence_feature_mode(
        Path("dynamic_sequence_shapelet_72.pkl")
    )

    assert infer._feature_mode == FEATURE_DYNAMIC_SEQUENCE_72


def _open_hand_landmarks(x: float = 0.5, wrist_y: float = 0.82):
    offset_y = wrist_y - 0.82

    def y(value: float) -> float:
        return value + offset_y

    landmarks = [[x, y(0.5)] for _ in range(21)]
    landmarks[0] = [x, y(0.82)]
    landmarks[5] = [x, y(0.56)]
    landmarks[6] = [x, y(0.42)]
    landmarks[7] = [x, y(0.30)]
    landmarks[8] = [x, y(0.18)]
    return landmarks


class _TwoHandsDetector:
    def detect_for_video_rgb(self, _frame_rgb):
        landmarks = _open_hand_landmarks()
        from cv.hand_landmarker import DetectedHand

        return [
            DetectedHand(landmarks=landmarks, handedness="Left", score=0.9),
            DetectedHand(landmarks=landmarks, handedness="Right", score=1.0),
        ]


class _OneHandDetector:
    def detect_for_video_rgb(self, _frame_rgb):
        from cv.hand_landmarker import DetectedHand

        return [DetectedHand(landmarks=_open_hand_landmarks(), handedness="Right", score=1.0)]


class _MovingOneHandDetector:
    def __init__(self, *, start_y: float = 0.82, step_y: float = -0.01) -> None:
        self.index = 0
        self.start_y = start_y
        self.step_y = step_y

    def detect_for_video_rgb(self, _frame_rgb):
        from cv.hand_landmarker import DetectedHand

        wrist_y = self.start_y + self.step_y * self.index
        self.index += 1
        return [
            DetectedHand(
                landmarks=_open_hand_landmarks(wrist_y=wrist_y),
                handedness="Right",
                score=1.0,
            )
        ]


class _MovingHorizontalHandDetector:
    def __init__(self, *, start_x: float = 0.80, step_x: float = -0.01) -> None:
        self.index = 0
        self.start_x = start_x
        self.step_x = step_x

    def detect_for_video_rgb(self, _frame_rgb):
        from cv.hand_landmarker import DetectedHand

        x = self.start_x + self.step_x * self.index
        self.index += 1
        return [
            DetectedHand(
                landmarks=_open_hand_landmarks(x=x),
                handedness="Right",
                score=1.0,
            )
        ]


class _MovingThenLostHorizontalHandDetector(_MovingHorizontalHandDetector):
    def __init__(
        self,
        *,
        start_x: float = 0.80,
        step_x: float = -0.04,
        visible_frames: int = 10,
    ) -> None:
        super().__init__(start_x=start_x, step_x=step_x)
        self.visible_frames = visible_frames

    def detect_for_video_rgb(self, frame_rgb):
        if self.index >= self.visible_frames:
            self.index += 1
            return []
        return super().detect_for_video_rgb(frame_rgb)


class _DirectionConfusedClassifier:
    classes_ = np.asarray([0, 1, 2])

    def predict(self, _features):
        return np.asarray([2])

    def predict_proba(self, _features):
        # The pose-heavy model prefers up, but left remains the best candidate
        # compatible with the measured horizontal trajectory.
        return np.asarray([[0.1, 0.2, 0.7]])


class _NegativeRejectingClassifier:
    classes_ = np.asarray([0, 1, 2])

    def predict(self, _features):
        return np.asarray([2])

    def predict_proba(self, _features):
        return np.asarray([[0.05, 0.10, 0.85]])


class _StaticProbabilityClassifier:
    def __init__(self, probabilities):
        self._probabilities = np.asarray([probabilities], dtype=float)
        self.classes_ = np.arange(self._probabilities.shape[1])

    def predict(self, _features):
        return np.asarray([int(np.argmax(self._probabilities[0]))])

    def predict_proba(self, _features):
        return self._probabilities


class _BinaryVerifier:
    classes_ = np.asarray([0, 1])

    def __init__(self, positive_probability: float) -> None:
        self._positive_probability = float(positive_probability)

    def predict_proba(self, _features):
        p = self._positive_probability
        return np.asarray([[1.0 - p, p]], dtype=float)


class _ShapeCheckingClassifier:
    def __init__(self, expected_dim: int = 42) -> None:
        self.expected_shape = (1, expected_dim)
        self.seen_shape = None

    def predict(self, features):
        self.seen_shape = features.shape
        assert features.shape == self.expected_shape
        return np.asarray([0])

    def predict_proba(self, features):
        assert features.shape == self.expected_shape
        return np.asarray([[0.9]])


class _FeatureCaptureClassifier:
    def __init__(self, expected_dim: int) -> None:
        self.expected_shape = (1, expected_dim)
        self.features = None

    def predict(self, features):
        self.features = features
        assert features.shape == self.expected_shape
        return np.asarray([0])

    def predict_proba(self, features):
        assert features.shape == self.expected_shape
        return np.asarray([[0.9]])


class _BadShapeClassifier:
    def predict(self, _features):
        raise ValueError("bad shape")

    def predict_proba(self, _features):
        raise AssertionError("predict_proba must not be called after predict failure")


def test_no_hand_frame_clears_window_and_does_not_predict() -> None:
    infer = object.__new__(GestureOnlineInfer)
    clf = _CountingClassifier()
    infer._detector = _NoHandsDetector()
    infer._clf = clf
    infer._classes = ["new"]
    infer._feature_dim = 42
    infer._classifier_two_hands = False
    infer._window = deque([np.ones(42, dtype=np.float32)], maxlen=30)
    infer._finger_count_window = deque([2], maxlen=5)

    out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert out["label"] == ""
    assert out["confidence"] == 0.0
    assert out["landmarks_json"] == "[]"
    assert out["hand_tracking"]["hand_detected"] is False
    assert out["hand_tracking"]["hand_lost_streak"] == 1
    assert len(infer._window) == 0
    assert len(infer._finger_count_window) == 0
    assert clf.predict_calls == 0


def test_single_hand_classifier_gets_42_features_when_two_hands_are_detected() -> None:
    infer = object.__new__(GestureOnlineInfer)
    clf = _ShapeCheckingClassifier(42)
    infer._detector = _TwoHandsDetector()
    infer._clf = clf
    infer._classes = ["new"]
    infer._feature_dim = 42
    infer._classifier_two_hands = False
    infer._window = deque(maxlen=30)
    infer._finger_count_window = deque(maxlen=5)
    infer._gesture_signatures = {}

    out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert clf.seen_shape == (1, 42)
    assert out["landmarks_json"] != "[]"


def test_classifier_can_process_hands_from_shared_detector() -> None:
    infer = object.__new__(GestureOnlineInfer)
    clf = _ShapeCheckingClassifier(42)
    infer._detector = None
    infer._clf = clf
    infer._classes = ["new"]
    infer._feature_dim = 42
    infer._classifier_two_hands = False
    infer._window = deque(maxlen=30)
    infer._finger_count_window = deque(maxlen=5)
    infer._gesture_signatures = {}
    hands = _OneHandDetector().detect_for_video_rgb(None)

    out = infer.process_detected_hands(hands)

    assert out["label"] == "new"
    assert clf.seen_shape == (1, 42)
    assert out["landmarks_json"] != "[]"


def test_two_hand_classifier_gets_84_features_with_one_hand_padded() -> None:
    infer = object.__new__(GestureOnlineInfer)
    clf = _FeatureCaptureClassifier(84)
    infer._detector = _OneHandDetector()
    infer._clf = clf
    infer._classes = ["two_hand_ready"]
    infer._feature_dim = 84
    infer._classifier_two_hands = True
    infer._window = deque(maxlen=30)
    infer._finger_count_window = deque(maxlen=5)
    infer._gesture_signatures = {}

    out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert out["label"] == "two_hand_ready"
    assert clf.features is not None
    assert clf.features.shape == (1, 84)
    assert np.allclose(clf.features[0, 42:], 0.0)


def test_dynamic_classifier_gets_expanded_sequence_features() -> None:
    infer = object.__new__(GestureOnlineInfer)
    clf = _FeatureCaptureClassifier(259)
    infer._detector = _OneHandDetector()
    infer._clf = clf
    infer._classes = ["hand_left"]
    infer._feature_dim = 259
    infer._raw_feature_dim = 42
    infer._feature_mode = "dynamic_stats"
    infer._classifier_two_hands = False
    infer._window = deque(maxlen=60)
    infer._finger_count_window = deque(maxlen=5)
    infer._gesture_signatures = {"hand_left": {"non_thumb_count": 0, "stability": 1.0}}

    for _ in range(59):
        out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))
        assert out["label"] == ""
        assert clf.features is None

    out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert out["label"] == "hand_left"
    assert clf.features is not None
    assert clf.features.shape == (1, 259)


def test_dynamic_global_classifier_gets_wrist_motion_features() -> None:
    infer = object.__new__(GestureOnlineInfer)
    clf = _FeatureCaptureClassifier(271)
    infer._detector = _MovingOneHandDetector(start_y=0.82, step_y=-0.01)
    infer._clf = clf
    infer._classes = ["swipe_up"]
    infer._feature_dim = 271
    infer._raw_feature_dim = 44
    infer._feature_mode = "dynamic_stats"
    infer._classifier_two_hands = False
    infer._window = deque(maxlen=60)
    infer._finger_count_window = deque(maxlen=5)
    infer._gesture_signatures = {}

    for _ in range(60):
        infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert clf.features is not None
    assert clf.features.shape == (1, 271)
    assert infer._window[-1].shape == (44,)
    assert infer._window[-1][-1] < infer._window[0][-1]


def test_dynamic_motion_gate_rejects_static_global_window() -> None:
    infer = object.__new__(GestureOnlineInfer)
    clf = _CountingClassifier()
    infer._detector = _OneHandDetector()
    infer._clf = clf
    infer._classes = ["swipe_up"]
    infer._feature_dim = 271
    infer._raw_feature_dim = 44
    infer._feature_mode = "dynamic_stats"
    infer._classifier_two_hands = False
    infer._window = deque(maxlen=36)
    infer._finger_count_window = deque(maxlen=5)
    infer._gesture_signatures = {}

    out = {}
    for _ in range(70):
        out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert out["label"] == ""
    assert out["confidence"] == 0.0
    assert clf.predict_calls == 0


def test_dynamic_motion_gate_rejects_wrong_direction_label() -> None:
    infer = object.__new__(GestureOnlineInfer)
    clf = _CountingClassifier()
    infer._detector = _MovingOneHandDetector(start_y=0.30, step_y=0.01)
    infer._clf = clf
    infer._classes = ["swipe_up"]
    infer._feature_dim = 271
    infer._raw_feature_dim = 44
    infer._feature_mode = "dynamic_stats"
    infer._classifier_two_hands = False
    infer._window = deque(maxlen=36)
    infer._finger_count_window = deque(maxlen=5)
    infer._gesture_signatures = {}

    out = {}
    for _ in range(70):
        out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert out["label"] == ""
    assert out["confidence"] == 0.0
    assert clf.predict_calls == 1


def test_dynamic_prediction_reranks_classes_by_dominant_motion_axis() -> None:
    infer = object.__new__(GestureOnlineInfer)
    infer._detector = _MovingHorizontalHandDetector()
    infer._clf = _DirectionConfusedClassifier()
    infer._classes = ["swipe_down", "swipe_left", "swipe_up"]
    infer._feature_dim = 271
    infer._raw_feature_dim = 44
    infer._feature_mode = "dynamic_stats"
    infer._classifier_two_hands = False
    infer._window = deque(maxlen=36)
    infer._finger_count_window = deque(maxlen=5)
    infer._gesture_signatures = {}

    out = {}
    for _ in range(60):
        out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert out["label"] == "swipe_left"
    assert out["confidence"] >= 0.80
    assert out["temporal"]["phase"] == "completed"
    assert out["dynamic_decision"]["motion_label"] == "swipe_left"
    assert out["dynamic_decision"]["model_label"] == "swipe_up"
    assert out["dynamic_decision"]["source"] == "motion_and_model_agree"


def test_motion_first_dynamic_prediction_works_without_estimator() -> None:
    infer = object.__new__(GestureOnlineInfer)
    infer._detector = _MovingHorizontalHandDetector()
    infer._clf = None
    infer._classes = ["swipe_down", "swipe_left", "swipe_up"]
    infer._feature_dim = 271
    infer._raw_feature_dim = 44
    infer._feature_mode = "dynamic_stats"
    infer._classifier_two_hands = False
    infer._window = deque(maxlen=36)
    infer._finger_count_window = deque(maxlen=5)
    infer._gesture_signatures = {}

    out = {}
    for _ in range(60):
        out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert out["label"] == "swipe_left"
    assert out["confidence"] >= 0.80
    assert out["dynamic_decision"]["source"] == "motion_first"
    assert out["dynamic_decision"]["model_label"] == ""


def test_dynamic_swipe_completes_when_hand_leaves_frame() -> None:
    infer = object.__new__(GestureOnlineInfer)
    infer._detector = _MovingThenLostHorizontalHandDetector()
    infer._clf = None
    infer._classes = ["swipe_down", "swipe_left", "swipe_up"]
    infer._feature_dim = 271
    infer._raw_feature_dim = 44
    infer._feature_mode = "dynamic_stats"
    infer._classifier_two_hands = False
    infer._window = deque(maxlen=36)
    infer._finger_count_window = deque(maxlen=5)
    infer._gesture_signatures = {}

    out = {}
    for _ in range(14):
        out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))
        if out.get("label"):
            break

    assert out["label"] == "swipe_left"
    assert out["confidence"] >= 0.80
    assert out["temporal"]["end_reason"] == "hand_lost"
    assert out["dynamic_decision"]["source"] == "motion_first"


def test_dynamic_negative_prediction_rejects_motion_first_swipe() -> None:
    infer = object.__new__(GestureOnlineInfer)
    infer._detector = _MovingHorizontalHandDetector()
    infer._clf = _NegativeRejectingClassifier()
    infer._classes = ["swipe_down", "swipe_left", "no_gesture_static"]
    infer._feature_dim = 271
    infer._raw_feature_dim = 44
    infer._feature_mode = "dynamic_stats"
    infer._classifier_two_hands = False
    infer._window = deque(maxlen=36)
    infer._finger_count_window = deque(maxlen=5)
    infer._gesture_signatures = {}
    infer._gesture_taxonomy = None

    out = {}
    for _ in range(60):
        out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert out["label"] == ""
    assert out["confidence"] == 0.0
    assert out["dynamic_decision"]["source"] == "negative_rejected"
    assert out["dynamic_decision"]["motion_label"] == "swipe_left"
    assert out["dynamic_decision"]["negative_label"] == "no_gesture_static"
    assert out["dynamic_decision"]["negative_confidence"] >= 0.72


def test_static_prediction_rejects_negative_class_probability() -> None:
    infer = object.__new__(GestureOnlineInfer)
    infer._clf = _StaticProbabilityClassifier([0.20, 0.80])
    infer._classes = ["palm", "no_gesture_static"]
    infer._gesture_taxonomy = None
    infer._gesture_rejection = {
        "thresholds": {
            "negative_confidence": 0.65,
            "min_top1_top2_margin": 0.10,
            "distance_multiplier": 2.5,
        },
        "classes": {},
    }

    label, confidence = infer._static_prediction(
        np.asarray([[0.0, 0.0]], dtype=np.float32)
    )

    assert label == ""
    assert confidence == 0.0
    assert infer._last_static_decision["source"] == "negative_rejected"
    assert infer._last_static_decision["rejection_reason"] == "negative_class"
    assert infer._last_static_decision["negative_label"] == "no_gesture_static"


def test_static_prediction_rejects_low_top1_top2_margin() -> None:
    infer = object.__new__(GestureOnlineInfer)
    infer._clf = _StaticProbabilityClassifier([0.53, 0.47])
    infer._classes = ["palm", "gun"]
    infer._gesture_taxonomy = None
    infer._gesture_rejection = {
        "thresholds": {
            "negative_confidence": 0.65,
            "min_top1_top2_margin": 0.10,
            "distance_multiplier": 2.5,
        },
        "classes": {},
    }

    label, confidence = infer._static_prediction(
        np.asarray([[0.0, 0.0]], dtype=np.float32)
    )

    assert label == ""
    assert confidence == 0.0
    assert infer._last_static_decision["source"] == "margin_rejected"
    assert infer._last_static_decision["rejection_reason"] == "low_margin"


def test_static_prediction_rejects_far_from_class_prototype() -> None:
    infer = object.__new__(GestureOnlineInfer)
    infer._clf = _StaticProbabilityClassifier([0.99, 0.01])
    infer._classes = ["palm", "gun"]
    infer._gesture_taxonomy = None
    infer._gesture_rejection = {
        "thresholds": {
            "negative_confidence": 0.65,
            "min_top1_top2_margin": 0.10,
            "distance_multiplier": 2.0,
        },
        "classes": {
            "palm": {
                "centroid": [0.0, 0.0],
                "prototype_radius": 0.10,
            }
        },
    }

    label, confidence = infer._static_prediction(
        np.asarray([[1.0, 1.0]], dtype=np.float32)
    )

    assert label == ""
    assert confidence == 0.0
    assert infer._last_static_decision["source"] == "prototype_rejected"
    assert infer._last_static_decision["rejection_reason"] == "far_from_prototype"
    assert infer._last_static_decision["prototype_distance"] > 0.20


def test_static_prediction_rejects_with_one_vs_rest_verifier() -> None:
    infer = object.__new__(GestureOnlineInfer)
    infer._clf = _StaticProbabilityClassifier([0.92, 0.08])
    infer._classes = ["palm", "gun"]
    infer._gesture_taxonomy = None
    infer._static_rejection_method = "one_vs_rest_logreg"
    infer._static_rejection_verifiers = {
        "methods": {
            "one_vs_rest_logreg": {
                "status": "ok",
                "threshold": 0.50,
                "verifiers": {"palm": _BinaryVerifier(0.24)},
            }
        }
    }
    infer._gesture_rejection = {
        "thresholds": {
            "negative_confidence": 0.65,
            "min_top1_top2_margin": 0.10,
            "distance_multiplier": 2.5,
        },
        "classes": {},
    }

    label, confidence = infer._static_prediction(
        np.asarray([[0.0, 0.0]], dtype=np.float32)
    )

    assert label == ""
    assert confidence == 0.0
    assert infer._last_static_decision["rejection_method"] == "one_vs_rest_logreg"
    assert infer._last_static_decision["source"] == "verifier_rejected"
    assert (
        infer._last_static_decision["rejection_reason"]
        == "one_vs_rest_low_probability"
    )
    assert infer._last_static_decision["verifier_probability"] == 0.24


def test_reset_temporal_state_clears_dynamic_windows() -> None:
    infer = object.__new__(GestureOnlineInfer)
    infer._window = deque([np.ones(44, dtype=np.float32)], maxlen=36)
    infer._finger_count_window = deque([4], maxlen=5)

    infer.reset_temporal_state()

    assert not infer._window
    assert not infer._finger_count_window


def test_acknowledge_dynamic_event_preserves_segmenter_cooldown() -> None:
    from cv.dynamic_motion import DynamicMotionSegmenter

    infer = object.__new__(GestureOnlineInfer)
    infer._window = deque([np.ones(44, dtype=np.float32)], maxlen=36)
    infer._finger_count_window = deque([4], maxlen=5)
    infer._dynamic_segmenter = DynamicMotionSegmenter()
    infer._dynamic_segmenter._cooldown_remaining = 5
    infer._pending_dynamic_prediction = ("swipe_left", 0.9)
    infer._pending_dynamic_repeats = 1

    infer.acknowledge_dynamic_event()

    assert not infer._window
    assert not infer._finger_count_window
    assert infer._pending_dynamic_prediction is None
    assert infer._dynamic_segmenter.phase == "cooldown"


def test_dynamic_raw_dim_inference_supports_new_and_legacy_sizes() -> None:
    infer = object.__new__(GestureOnlineInfer)
    infer._feature_mode = FEATURE_STATIC_CRAFT_FULL_STATS
    infer._feature_dim = feature_vector_size(FEATURE_STATIC_CRAFT_FULL_STATS, 63)
    assert infer._infer_raw_feature_dim() == 63

    infer._feature_mode = "dynamic_stats"

    infer._feature_dim = 271
    assert infer._infer_raw_feature_dim() == 44

    infer._feature_dim = 264
    assert infer._infer_raw_feature_dim() == 44

    infer._feature_mode = FEATURE_DYNAMIC_CRAFT_STATS
    infer._feature_dim = feature_vector_size(FEATURE_DYNAMIC_CRAFT_STATS, 65)
    assert infer._infer_raw_feature_dim() == 65

    infer._feature_mode = FEATURE_DYNAMIC_CRAFT_FULL_STATS
    infer._feature_dim = feature_vector_size(FEATURE_DYNAMIC_CRAFT_FULL_STATS, 65)
    assert infer._infer_raw_feature_dim() == 65

    infer._feature_mode = FEATURE_DYNAMIC_LANDMARK_IMAGE
    infer._feature_dim = feature_vector_size(FEATURE_DYNAMIC_LANDMARK_IMAGE, 65)
    assert infer._infer_raw_feature_dim() == 65

    infer._feature_mode = FEATURE_STATIC_LANDMARK_IMAGE
    infer._feature_dim = feature_vector_size(FEATURE_STATIC_LANDMARK_IMAGE, 63)
    assert infer._infer_raw_feature_dim() == 63


def test_hand_frame_feature_builds_65_dim_xyz_wrist_layout() -> None:
    infer = object.__new__(GestureOnlineInfer)
    landmarks = _open_hand_landmarks()
    normalized = np.zeros((21, 2), dtype=np.float32)
    xyz = [
        (float(point[0]), float(point[1]), float(index) / 100.0)
        for index, point in enumerate(landmarks)
    ]
    hand = DetectedHand(
        landmarks=landmarks,
        handedness="Right",
        score=1.0,
        landmarks_xyz=xyz,
    )

    feature = infer._hand_frame_feature(hand, normalized, target_dim=65)

    hand_scale = np.max(
        np.linalg.norm(np.asarray(landmarks) - np.asarray(landmarks)[0], axis=1)
    )
    expected_z = [float(index) / 100.0 / hand_scale for index in range(21)]
    assert feature.shape == (65,)
    assert np.allclose(feature[2:63:3], expected_z)
    assert np.allclose(feature[-2:], landmarks[0])


def test_classifier_failure_keeps_landmarks_for_overlay_and_pointer() -> None:
    infer = object.__new__(GestureOnlineInfer)
    infer._detector = _OneHandDetector()
    infer._clf = _BadShapeClassifier()
    infer._classes = ["new"]
    infer._feature_dim = 42
    infer._classifier_two_hands = False
    infer._window = deque(maxlen=30)
    infer._finger_count_window = deque(maxlen=5)
    infer._gesture_signatures = {}

    out = infer.process_frame_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert out["label"] == ""
    assert out["confidence"] == 0.0
    assert out["landmarks_json"] != "[]"
    assert len(infer._window) == 0
