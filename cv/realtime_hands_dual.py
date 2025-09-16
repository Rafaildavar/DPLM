import sys
from typing import List, Tuple

import cv2
import numpy as np
import mediapipe as mp


# ------------------------------------------------------------
# Реал-тайм детекция двух рук с подписями Left/Right и нормализацией
# Комментарии на русском для понятности
# ------------------------------------------------------------

MACOS_BACKEND = cv2.CAP_AVFOUNDATION  # стабильный бэкенд для macOS (M1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def normalize_landmarks(
    landmarks_xy: List[Tuple[float, float]],
    method: str = "wrist_scale",
) -> np.ndarray:
    """
    Нормализует 2D-координаты ландмарков (21 точка) для инвариантности к положению и масштабу.

    method="wrist_scale": вычитание запястья (0‑я точка) и деление на максимум L2‑дистанций.
    """
    points = np.asarray(landmarks_xy, dtype=np.float32)
    if points.shape != (21, 2):
        raise ValueError("Ожидалось 21 точка (x, y)")

    if method == "wrist_scale":
        wrist = points[0].copy()
        points -= wrist
        dists = np.linalg.norm(points, axis=1)
        scale = float(np.max(dists))
        if scale < 1e-6:
            scale = 1.0
        points /= scale
    else:
        raise ValueError(f"Неизвестный метод нормализации: {method}")
    return points


def main() -> None:
    """
    - Захват видео (отзеркаленный кадр)
    - Детекция до 2 рук, отрисовка ландмарков и подписей Left/Right
    - Нормализация координат каждой руки и вывод в консоль
    - Выход по клавише 'q'
    """

    cap = cv2.VideoCapture(0, MACOS_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру. Разрешите доступ в macOS → Privacy & Security → Camera.")
        sys.exit(1)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,            # ключевое изменение: до двух рук
        model_complexity=1,
        min_detection_confidence=0.6,
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

            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                # multi_handedness содержит метаданные (Left/Right) в той же индексации
                handedness = results.multi_handedness or []
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                    )

                    # Подпись руки (Left/Right) возле запястья
                    label = ""
                    if idx < len(handedness):
                        try:
                            label = handedness[idx].classification[0].label
                        except Exception:
                            label = ""

                    h, w = frame_bgr.shape[:2]
                    x = int(hand_landmarks.landmark[0].x * w)
                    y = int(hand_landmarks.landmark[0].y * h)
                    if label:
                        cv2.putText(
                            frame_bgr,
                            label,
                            (x + 10, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                    # Сбор и нормализация координат этой руки
                    landmarks_xy = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    norm_pts = normalize_landmarks(landmarks_xy, method="wrist_scale")
                    np.set_printoptions(precision=3, suppress=True)
                    print({"hand": label or f"hand_{idx}", "norm": norm_pts.tolist()})

            cv2.imshow("Hands (dual) — press 'q' to quit", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


