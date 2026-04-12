"""
Параметры захвата камеры OpenCV — согласованы с cv/realtime_infer.py и app/gui_main.py.
"""
from __future__ import annotations

import sys
from typing import Any


def open_default_capture() -> Any:
    """
    Открыть камеру с тем же бэкендом и разрешением, что в realtime_infer.py.
    """
    import cv2

    if sys.platform == "darwin":
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap
