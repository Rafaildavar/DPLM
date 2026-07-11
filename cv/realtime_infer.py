"""
Онлайн-классификация жестов (KNN) + опциональный TTS.

С mediapipe>=0.10.33 используется Tasks API через общий хелпер
``cv.hand_landmarker.HandLandmarkerVideo``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Deque, List

import cv2
import joblib
import numpy as np
import pyttsx3

# При запуске как ``python cv/realtime_infer.py`` корень проекта не в sys.path.
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


def _ordered_hands(hands):
    def sort_key(hand):
        handedness = (getattr(hand, "handedness", "") or "").strip().lower()
        if handedness == "right":
            side_rank = 0
        elif handedness == "left":
            side_rank = 1
        else:
            side_rank = 2
        return side_rank, -float(getattr(hand, "score", 0.0) or 0.0)

    return sorted(hands, key=sort_key)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Онлайн-классификация жестов (KNN) + TTS")
    p.add_argument("--model", default="models/knn.pkl", help="Путь к модели KNN (joblib)")
    p.add_argument("--classes", default="models/classes.json", help="JSON со списком классов")
    p.add_argument(
        "--feature-dim-file",
        default="models/feature_dim.txt",
        help="Файл с размерностью признака (из тренировки)",
    )
    p.add_argument("--window", type=int, default=30, help="Длина окна (кадров) для усреднения")
    p.add_argument(
        "--two-hands",
        action="store_true",
        help="Совместимость: live-инференс сам выбирает размерность модели",
    )
    p.add_argument("--tts", action="store_true", help="Озвучивать распознанный жест")
    p.add_argument("--camera-index", type=int, default=0, help="Индекс камеры OpenCV")
    p.add_argument("--fps", type=int, default=30, help="Целевой FPS камеры")
    p.add_argument(
        "--min-say-interval", type=float, default=1.5, help="Интервал между озвучиваниями, сек"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    clf = joblib.load(args.model)
    classes = json.loads(Path(args.classes).read_text())
    feature_dim = int(Path(args.feature_dim_file).read_text().strip())
    model_feature_dim = int(getattr(clf, "n_features_in_", 0) or 0)
    if model_feature_dim > 0 and model_feature_dim != feature_dim:
        print(
            "[w] feature_dim.txt не совпадает с knn.pkl: "
            f"{feature_dim} -> {model_feature_dim}"
        )
        feature_dim = model_feature_dim

    classifier_two_hands = feature_dim == 84
    if not args.two_hands:
        print("[i] Auto-hand: модель сама задаёт live-формат признаков")

    tts_engine = None
    last_spoken_label = None
    last_spoken_ts = 0.0
    if args.tts:
        try:
            tts_engine = pyttsx3.init()
        except Exception as e:
            print(f"[!] Не удалось инициализировать TTS: {e}")
            args.tts = False

    window: Deque[np.ndarray] = deque(maxlen=max(1, args.window))

    cap = cv2.VideoCapture(int(args.camera_index), MACOS_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, int(args.fps))

    if not cap.isOpened():
        print("Ошибка открытия камеры (проверьте доступ в Privacy & Security → Camera)")
        sys.exit(1)

    detector = HandLandmarkerVideo(
        num_hands=2,
        min_detection_confidence=0.6,
        min_presence_confidence=0.6,
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
            hands = _ordered_hands(detector.detect_for_video_rgb(frame_rgb))

            normalized: List[np.ndarray] = []
            for h in hands:
                draw_hand_overlay_bgr(frame_bgr, h, label=h.handedness)
                try:
                    normalized.append(normalize_landmarks(h.landmarks))
                except Exception:
                    continue

            normalized = normalized[:2]
            if classifier_two_hands:
                if len(normalized) >= 2:
                    frame_vec = np.concatenate(normalized[:2], axis=0)
                elif len(normalized) == 1:
                    frame_vec = np.concatenate(
                        [normalized[0], np.zeros((21, 2), dtype=np.float32)], axis=0
                    )
                else:
                    frame_vec = np.zeros((42, 2), dtype=np.float32)
            else:
                frame_vec = (
                    normalized[0] if normalized else np.zeros((21, 2), dtype=np.float32)
                )

            feat = frame_vec.reshape(-1)
            if feat.shape[0] != feature_dim:
                if feat.shape[0] > feature_dim:
                    feat = feat[:feature_dim]
                else:
                    pad = np.zeros(feature_dim - feat.shape[0], dtype=feat.dtype)
                    feat = np.concatenate([feat, pad], axis=0)

            window.append(feat)

            avg_feat = np.mean(np.stack(window, axis=0), axis=0).reshape(1, -1)
            pred_idx = int(clf.predict(avg_feat)[0])
            label = classes[pred_idx] if 0 <= pred_idx < len(classes) else str(pred_idx)

            cv2.putText(
                frame_bgr,
                f"Pred: {label}  hands={len(hands)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

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
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
