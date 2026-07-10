import numpy as np

from app.flet_app.controller import (
    AppController,
    CAMERA_INFERENCE_MAX_FPS,
    CAMERA_PREVIEW_MAX_FPS,
    DYNAMIC_RECOGNITION_WINDOW,
    _CameraFrame,
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


def test_ml_path_uses_fresh_capture_loop_without_preview_throttle():
    class FakeInfer:
        def __init__(self):
            self.frames = []

        def process_frame_rgb(self, rgb):
            self.frames.append(rgb)
            return {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": "[]",
                "performance": {
                    "total_inference_ms": 1.0,
                    "detection_ms": 0.5,
                },
            }

    controller = object.__new__(AppController)
    fake_infer = FakeInfer()
    dispatched = []
    controller._sample_recording = None
    controller._embedded_active = True
    controller._embedded_infer = fake_infer
    controller._target_fps = 30
    controller._dispatch_infer_result = dispatched.append

    frame1 = _CameraFrame(1, np.zeros((24, 32, 3), dtype=np.uint8), 1.0)
    frame2 = _CameraFrame(2, np.zeros((24, 32, 3), dtype=np.uint8), 2.0)

    controller._process_camera_frame_for_ml(frame1)
    controller._process_camera_frame_for_ml(frame2)

    assert len(fake_infer.frames) == 2
    assert len(dispatched) == 2
    assert dispatched[0]["performance"]["camera_frame_sequence"] == 1
    assert dispatched[1]["performance"]["camera_frame_sequence"] == 2
    assert dispatched[0]["performance"]["inference_frame_policy"] == "fresh_capture_loop"
    assert dispatched[0]["performance"]["inference_queue_depth"] == 0
