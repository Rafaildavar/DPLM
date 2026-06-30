import numpy as np

from app.flet_app.controller import AppController, _CameraFrame


def test_camera_ml_path_passes_capture_timestamp_to_infer():
    class FakeInfer:
        def __init__(self):
            self.timestamps = []

        def process_frame_rgb(self, _rgb, timestamp_ms=None):
            self.timestamps.append(timestamp_ms)
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

    frame = _CameraFrame(1, np.zeros((24, 32, 3), dtype=np.uint8), 12.345)

    controller._process_camera_frame_for_ml(frame)

    assert fake_infer.timestamps == [12345]
    assert dispatched[0]["performance"]["camera_frame_sequence"] == 1
