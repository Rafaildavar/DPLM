"""
Реал-тайм детекция двух рук (MediaPipe Tasks API) с подписями Left/Right
и нормализацией координат каждой руки. Совместим с mediapipe>=0.10.33.
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
            "Ошибка: не удалось открыть камеру. Разрешите доступ в macOS → Privacy & Security → Camera."
        )
        sys.exit(1)

    detector = HandLandmarkerVideo(
        num_hands=2,
        min_detection_confidence=0.6,
        min_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    print("Старт. Нажмите 'q' для выхода.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Предупреждение: не удалось прочитать кадр.")
                continue

            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            hands = detector.detect_for_video_rgb(frame_rgb)

            for idx, h in enumerate(hands):
                draw_hand_overlay_bgr(frame_bgr, h, label=h.handedness or f"hand_{idx}")
                norm_pts = normalize_landmarks(h.landmarks, method="wrist_scale")
                np.set_printoptions(precision=3, suppress=True)
                print({"hand": h.handedness or f"hand_{idx}", "norm": norm_pts.tolist()})

            cv2.imshow("Hands (dual) — press 'q' to quit", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
