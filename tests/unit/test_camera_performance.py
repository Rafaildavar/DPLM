import numpy as np

from app.flet_app.controller import (
    CAMERA_INFERENCE_MAX_FPS,
    CAMERA_PREVIEW_MAX_FPS,
    DYNAMIC_RECOGNITION_WINDOW,
    _camera_preview_enabled,
    _resize_frame_to_max_width,
)


def test_camera_profile_uses_30fps_preview_and_36_inference_frames():
    assert CAMERA_PREVIEW_MAX_FPS == 30.0
    assert CAMERA_INFERENCE_MAX_FPS == 30.0
    assert DYNAMIC_RECOGNITION_WINDOW == 36


def test_camera_preview_can_be_disabled_for_detection_only(monkeypatch):
    monkeypatch.setenv("DPLM_DISABLE_CAMERA_PREVIEW", "1")

    assert _camera_preview_enabled() is False


def test_resize_frame_to_max_width_downscales_wide_frame():
    frame = np.zeros((540, 960, 3), dtype=np.uint8)

    resized = _resize_frame_to_max_width(frame, 480)

    assert resized.shape == (270, 480, 3)


def test_resize_frame_to_max_width_keeps_small_frame_object():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    resized = _resize_frame_to_max_width(frame, 480)

    assert resized is frame
