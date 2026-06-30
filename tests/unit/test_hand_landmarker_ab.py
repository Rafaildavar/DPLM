import numpy as np

from cv.hand_landmarker import (
    DetectedHand,
    HandLandmarkerVideo,
    mediapipe_profile_settings,
)


def _hand(x: float, *, score: float = 0.9) -> DetectedHand:
    landmarks = [(x, 0.5) for _ in range(21)]
    landmarks_xyz = [(x, 0.5, 0.1 + index * 0.01) for index in range(21)]
    world = [(x, 0.5, -0.2 + index * 0.02) for index in range(21)]
    return DetectedHand(
        landmarks=landmarks,
        handedness="Right",
        score=score,
        landmarks_xyz=landmarks_xyz,
        world_landmarks=world,
    )


def test_mediapipe_profiles_expose_ab_thresholds(monkeypatch):
    monkeypatch.setenv("DPLM_MEDIAPIPE_PROFILE", "strict_tracking")

    settings = mediapipe_profile_settings()

    assert settings["profile"] == "strict_tracking"
    assert settings["min_detection_confidence"] == 0.7
    assert settings["min_presence_confidence"] == 0.5
    assert settings["min_tracking_confidence"] == 0.7


def test_real_timestamp_is_monotonic_when_camera_time_repeats():
    detector = object.__new__(HandLandmarkerVideo)
    detector._ts_ms = 1000
    detector._last_timestamp_source = "synthetic_33ms"

    assert detector._next_timestamp_ms(1010) == 1010
    assert detector._last_timestamp_source == "real_monotonic"
    assert detector._next_timestamp_ms(1005) == 1011
    assert detector._last_timestamp_source == "real_monotonic_adjusted"


def test_landmark_ema_smoothing_keeps_depth_channels():
    detector = object.__new__(HandLandmarkerVideo)
    detector._smoothing_alpha = 0.5
    detector._previous_smoothed_hands = {}

    first = detector._smooth_hands([_hand(0.0)])[0]
    second = detector._smooth_hands([_hand(1.0)])[0]

    assert first.landmarks[0] == (0.0, 0.5)
    assert np.isclose(second.landmarks[0][0], 0.5)
    assert second.landmarks_xyz is not None
    assert second.world_landmarks is not None
    assert np.isclose(second.landmarks_xyz[-1][2], 0.3)
    assert np.isclose(second.world_landmarks[-1][2], 0.2)


def test_quality_metrics_report_z_and_world_availability():
    detector = object.__new__(HandLandmarkerVideo)
    detector._profile = "baseline_06"
    detector._thresholds = {
        "min_detection_confidence": 0.6,
        "min_presence_confidence": 0.6,
        "min_tracking_confidence": 0.6,
    }
    detector._smoothing_alpha = 0.0

    metrics = detector._quality_metrics(
        [_hand(0.2)],
        timestamp_ms=123,
        timestamp_source="real_monotonic",
        detection_ms=4.2,
    )

    assert metrics["hand_detected"] is True
    assert metrics["landmark_z_available"] is True
    assert metrics["world_landmarks_available"] is True
    assert metrics["mediapipe_timestamp_source"] == "real_monotonic"
