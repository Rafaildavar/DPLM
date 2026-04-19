"""
Базовый демонстрационный модуль: видеозахват + MediaPipe Hands (Tasks API)
+ нормализация координат. Работает на mediapipe>=0.10.33.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cv.hand_landmarker import (  # noqa: E402
    HandLandmarkerVideo,
    draw_hand_overlay_bgr,
    normalize_landmarks,
)


MACOS_BACKEND = cv2.CAP_AVFOUNDATION


def main() -> None:
    cap = cv2.VideoCapture(0, MACOS_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(
            "Ошибка: не удалось открыть камеру. Проверьте разрешения в macOS → Privacy & Security → Camera."
        )
        sys.exit(1)

    detector = HandLandmarkerVideo(
        num_hands=1,
        min_detection_confidence=0.6,
        min_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    print("Старт. Нажмите 'q' в окне видео для выхода.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Предупреждение: не удалось прочитать кадр с камеры.")
                continue

            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            hands = detector.detect_for_video_rgb(frame_rgb)

            for h in hands:
                draw_hand_overlay_bgr(frame_bgr, h, label=h.handedness)
                norm_points = normalize_landmarks(h.landmarks, method="wrist_scale")
                np.set_printoptions(precision=3, suppress=True)
                print(norm_points.tolist())

            cv2.imshow("MediaPipe Hands (press 'q' to quit)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
