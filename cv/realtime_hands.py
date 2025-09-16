import sys
import time
from typing import List, Tuple

import cv2
import numpy as np
import mediapipe as mp


# ------------------------------------------------------------
# Базовый модуль: захват видео + MediaPipe Hands + нормализация
# Комментарии на русском для понятности каждой важной строки
# ------------------------------------------------------------

# Для macOS на Apple Silicon наиболее стабильный бэкенд захвата — AVFoundation
# Если камера не открывается: попробуйте убрать второй аргумент или сменить индекс камеры на 1
MACOS_BACKEND = cv2.CAP_AVFOUNDATION

# Простые псевдонимы пространств имён MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def normalize_landmarks(
    landmarks_xy: List[Tuple[float, float]],
    method: str = "wrist_scale",
) -> np.ndarray:
    """
    Нормализует 2D-координаты ландмарков руки, чтобы уменьшить влияние расстояния до камеры
    и позиции в кадре.

    Параметры:
    - landmarks_xy: 21 точка [(x, y), ...] в относительных координатах кадра [0..1].
    - method:
        "wrist_scale" — вычитание запястья (index 0) + деление на максимальную L2-дистанцию.
        "bbox"        — центровка по центру bbox + деление на диагональ bbox.

    Возвращает:
    - np.ndarray формы (21, 2) — нормализованные координаты.
    """
    points = np.asarray(landmarks_xy, dtype=np.float32)
    if points.shape != (21, 2):
        raise ValueError("Ожидалось 21 точка (x, y)")

    if method == "wrist_scale":
        # Центрируем по запястью — landmark 0 по спецификации MediaPipe
        wrist = points[0].copy()
        points -= wrist
        # Масштаб — максимальная дистанция до запястья среди всех точек
        dists = np.linalg.norm(points, axis=1)
        scale = float(np.max(dists))
        if scale < 1e-6:
            scale = 1.0  # защита от деления на ноль
        points /= scale
    elif method == "bbox":
        # Центрируем по центру ограничивающей рамки и масштабируем на её диагональ
        xmin, ymin = np.min(points, axis=0)
        xmax, ymax = np.max(points, axis=0)
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        diag = float(np.linalg.norm([xmax - xmin, ymax - ymin]))
        if diag < 1e-6:
            diag = 1.0
        points -= np.array([cx, cy], dtype=np.float32)
        points /= diag
    else:
        raise ValueError(f"Неизвестный метод нормализации: {method}")

    return points


def main() -> None:
    """
    Главный цикл:
    - Захват видео с веб-камеры
    - Детекция руки и извлечение 21 ландмарка
    - Отрисовка скелета руки на кадре
    - Нормализация координат и вывод в консоль
    - Выход по клавише 'q'
    """

    # ---------------------------
    # Инициализация видеозахвата
    # ---------------------------
    cap = cv2.VideoCapture(0, MACOS_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(
            "Ошибка: не удалось открыть камеру. Проверьте разрешения в macOS → Privacy & Security → Camera."
        )
        sys.exit(1)

    # ---------------------------
    # Инициализация MediaPipe Hands
    # ---------------------------
    hands = mp_hands.Hands(
        static_image_mode=False,       # оптимизация под видео
        max_num_hands=1,               # можем увеличить до 2 позже
        model_complexity=1,            # баланс скорость/точность: 0/1/2
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    print("Старт. Нажмите 'q' в окне видео для выхода.")

    try:
        while True:
            t0 = time.time()

            ok, frame_bgr = cap.read()
            if not ok:
                print("Предупреждение: не удалось прочитать кадр с камеры.")
                continue

            # Отзеркаливаем кадр для удобства пользователя
            frame_bgr = cv2.flip(frame_bgr, 1)

            # MediaPipe ожидает RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Запуск детектора/трекера рук
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Отрисовка ландмарков и соединений на исходном BGR-кадре
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                    )

                    # Сбор (x, y) в относительных координатах [0..1]
                    landmarks_xy = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                    # Нормализация координат (инвариантность к положению/масштабу)
                    norm_points = normalize_landmarks(landmarks_xy, method="wrist_scale")

                    # Печать нормализованных координат одной строки на кадр (для первой руки)
                    np.set_printoptions(precision=3, suppress=True)
                    print(norm_points.tolist())

            # Показ окна с видео
            cv2.imshow("MediaPipe Hands (press 'q' to quit)", frame_bgr)

            # Выход по клавише 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Можно вывести FPS при отладке
            # fps = 1.0 / max(1e-6, (time.time() - t0))
            # print(f"FPS: {fps:.1f}")

    finally:
        # Корректно освобождаем ресурсы
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


