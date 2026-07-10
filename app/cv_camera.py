"""
Общие параметры захвата камеры OpenCV для desktop и headless runtime.
"""
from __future__ import annotations

import sys
from typing import Any


def open_default_capture(
    camera_index: int = 0,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> Any:
    """
    Открыть камеру с тем же бэкендом и разрешением, что в realtime_infer.py.
    """
    import cv2

    if sys.platform == "darwin":
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(int(camera_index))
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, int(fps))
        # Backends may ignore this, but when supported it prevents processing
        # stale buffered frames after a slow inference iteration.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
