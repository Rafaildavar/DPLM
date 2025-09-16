import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3


# ----------------------------------------------
# Онлайн‑классификация жестов + опциональный TTS
# Комментарии на русском
# ----------------------------------------------

MACOS_BACKEND = cv2.CAP_AVFOUNDATION
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def normalize_landmarks(landmarks_xy: List[Tuple[float, float]]) -> np.ndarray:
    """Нормализация относительно запястья и масштаба (21×2)."""
    pts = np.asarray(landmarks_xy, dtype=np.float32)
    if pts.shape != (21, 2):
        raise ValueError("Ожидалось 21 точка (x, y)")
    wrist = pts[0].copy()
    pts -= wrist
    d = np.linalg.norm(pts, axis=1)
    scale = float(np.max(d))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Онлайн‑классификация жестов (KNN) + TTS")
    p.add_argument("--model", default="models/knn.pkl", help="Путь к модели KNN (joblib)")
    p.add_argument("--classes", default="models/classes.json", help="JSON со списком классов")
    p.add_argument(
        "--feature-dim-file",
        default="models/feature_dim.txt",
        help="Файл с размерностью признака (из тренировки)",
    )
    p.add_argument("--window", type=int, default=10, help="Длина окна (кадров) для усреднения")
    p.add_argument("--two-hands", action="store_true", help="Учитывать вторую руку (42×2)")
    p.add_argument("--tts", action="store_true", help="Озвучивать распознанный жест")
    p.add_argument("--min-say-interval", type=float, default=1.5, help="Интервал между озвучиваниями, сек")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Загрузка модели и метаданных
    clf = joblib.load(args.model)
    classes = json.loads(Path(args.classes).read_text())
    feature_dim = int(Path(args.feature_dim_file).read_text().strip())

    # Автонастройка режима рук по размерности признака
    # 42 = одна рука (21×2), 84 = две руки (42×2)
    if feature_dim == 84 and not args.two_hands:
        print("[i] Обнаружена размерность 84 → переключаюсь в режим двух рук (--two-hands)")
        args.two_hands = True
    elif feature_dim == 42 and args.two_hands:
        print("[i] Обнаружена размерность 42 → отключаю режим двух рук (ожидается одна рука)")
        args.two_hands = False

    # Подготовка TTS (при необходимости)
    tts_engine = None
    last_spoken_label = None
    last_spoken_ts = 0.0
    if args.tts:
        try:
            tts_engine = pyttsx3.init()
            # Можно подобрать голос/скорость при желании
        except Exception as e:
            print(f"[!] Не удалось инициализировать TTS: {e}")
            args.tts = False

    # Очередь кадров для окна
    window: Deque[np.ndarray] = deque(maxlen=max(1, args.window))

    cap = cv2.VideoCapture(0, MACOS_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Ошибка открытия камеры (проверьте доступ в Privacy & Security → Camera)")
        sys.exit(1)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2 if args.two_hands else 1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    print("Старт онлайн‑инференса. Нажмите 'q' для выхода.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[w] Кадр не прочитан")
                continue

            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            landmarks_frames: List[np.ndarray] = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                    )
                    pts = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    pts_norm = normalize_landmarks(pts)
                    landmarks_frames.append(pts_norm)

            # Формирование вектора признака кадра согласно feature_dim
            if args.two_hands:
                if len(landmarks_frames) == 2:
                    frame_vec = np.concatenate(landmarks_frames, axis=0)  # (42,2)
                elif len(landmarks_frames) == 1:
                    frame_vec = np.concatenate(
                        [landmarks_frames[0], np.zeros((21, 2), dtype=np.float32)], axis=0
                    )
                else:
                    frame_vec = np.zeros((42, 2), dtype=np.float32)
            else:
                frame_vec = landmarks_frames[0] if landmarks_frames else np.zeros((21, 2), dtype=np.float32)

            # Разворачиваем до (D,)
            feat = frame_vec.reshape(-1)

            # Если ожидалась другая размерность (например, модель обучена на одну руку)
            if feat.shape[0] != feature_dim:
                # подгоняем: либо обрезаем/дополняем нулями
                if feat.shape[0] > feature_dim:
                    feat = feat[:feature_dim]
                else:
                    pad = np.zeros(feature_dim - feat.shape[0], dtype=feat.dtype)
                    feat = np.concatenate([feat, pad], axis=0)

            window.append(feat)

            # Предсказание по усреднению окна
            avg_feat = np.mean(np.stack(window, axis=0), axis=0).reshape(1, -1)
            pred_idx = int(clf.predict(avg_feat)[0])
            label = classes[pred_idx] if 0 <= pred_idx < len(classes) else str(pred_idx)

            # Рисуем предсказание
            cv2.putText(frame_bgr, f"Pred: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # Озвучка при смене класса с дебаунсом
            if args.tts:
                import time
                now = time.time()
                if label != last_spoken_label and (now - last_spoken_ts) >= args.min_say_interval:
                    try:
                        tts_engine.say(label)
                        tts_engine.runAndWait()
                        last_spoken_label = label
                        last_spoken_ts = now
                    except Exception as e:
                        print(f"[w] TTS error: {e}")

            cv2.imshow("Realtime Inference — press 'q' to quit", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
