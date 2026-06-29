import numpy as np

from app.flet_app.controller import _resize_frame_to_max_width


def test_resize_frame_to_max_width_downscales_wide_frame():
    frame = np.zeros((540, 960, 3), dtype=np.uint8)

    resized = _resize_frame_to_max_width(frame, 480)

    assert resized.shape == (270, 480, 3)


def test_resize_frame_to_max_width_keeps_small_frame_object():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    resized = _resize_frame_to_max_width(frame, 480)

    assert resized is frame
