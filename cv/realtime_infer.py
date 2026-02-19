import argparse
import csv
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Deque, List

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3

from cv.gesture_features import build_frame_feature


MACOS_BACKEND = cv2.CAP_AVFOUNDATION
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Онлайн‑классификация жестов (KNN) + TTS")
    p.add_argument("--model", default="models/knn.pkl", help="Путь к модели KNN (joblib)")
    p.add_argument("--classes", default="models/classes.json", help="JSON со списком классов")
    p.add_argument("--feature-dim-file", default="models/feature_dim.txt", help="Файл с размерностью признака")
    p.add_argument("--window", type=int, default=12, help="Размер окна для стабильного класса")
    p.add_argument("--stable-threshold", type=int, default=8, help="Минимум повторений класса в окне")
    p.add_argument("--ema-alpha", type=float, default=0.3, help="Коэффициент EMA по признакам")
    p.add_argument("--two-hands", action="store_true", help="Учитывать вторую руку")
    p.add_argument("--tts", action="store_true", help="Озвучивать распознанный жест")
    p.add_argument("--min-say-interval", type=float, default=1.5, help="Интервал между озвучиваниями, сек")
    p.add_argument("--log-csv", default="", help="CSV для логирования fps и latency")
    p.add_argument("--no-draw", action="store_true", help="Не рисовать ландмарки и подписи")
    return p.parse_args()


def _fit_feature_dim(feat: np.ndarray, feature_dim: int) -> np.ndarray:
    if feat.shape[0] > feature_dim:
        return feat[:feature_dim]
    if feat.shape[0] < feature_dim:
        return np.concatenate([feat, np.zeros(feature_dim - feat.shape[0], dtype=feat.dtype)], axis=0)
    return feat


def main() -> None:
    args = parse_args()

    clf = joblib.load(args.model)
    classes = json.loads(Path(args.classes).read_text())
    feature_dim = int(Path(args.feature_dim_file).read_text().strip())

    if feature_dim > 60 and not args.two_hands:
        print("[i] Обнаружена размерность для двух рук → включаю режим --two-hands")
        args.two_hands = True

    tts_engine = None
    last_spoken_label = None
    last_spoken_ts = 0.0
    if args.tts:
        try:
            tts_engine = pyttsx3.init()
        except Exception as exc:
            print(f"[!] Не удалось инициализировать TTS: {exc}")
            args.tts = False

    pred_window: Deque[str] = deque(maxlen=max(1, args.window))
    ema_feat: np.ndarray | None = None

    csv_writer = None
    csv_file = None
    if args.log_csv:
        log_path = Path(args.log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = log_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp",
            "fps",
            "t_capture_ms",
            "t_mediapipe_ms",
            "t_features_ms",
            "t_predict_ms",
            "t_draw_ms",
            "t_total_ms",
            "predicted_class",
        ])

    cap = cv2.VideoCapture(0, MACOS_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("Ошибка открытия камеры")
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
            loop_t0 = time.perf_counter()

            cap_t0 = time.perf_counter()
            ok, frame_bgr = cap.read()
            t_capture = (time.perf_counter() - cap_t0) * 1000.0
            if not ok:
                continue

            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            mp_t0 = time.perf_counter()
            results = hands.process(frame_rgb)
            t_mediapipe = (time.perf_counter() - mp_t0) * 1000.0

            feat_t0 = time.perf_counter()
            hand_points: List[np.ndarray] = []
            handedness_labels: List[str] = []
            if results.multi_hand_landmarks:
                handedness = results.multi_handedness or []
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if not args.no_draw:
                        mp_drawing.draw_landmarks(
                            image=frame_bgr,
                            landmark_list=hand_landmarks,
                            connections=mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                            connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                        )
                    hand_points.append(np.asarray([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32))
                    label = ""
                    if idx < len(handedness):
                        try:
                            label = handedness[idx].classification[0].label
                        except Exception:
                            label = ""
                    handedness_labels.append(label)

            feat = build_frame_feature(
                hand_landmarks=hand_points,
                handedness_labels=handedness_labels,
                two_hands=args.two_hands,
                include_presence_mask=True,
            )
            feat = _fit_feature_dim(feat, feature_dim)
            t_features = (time.perf_counter() - feat_t0) * 1000.0

            if ema_feat is None:
                ema_feat = feat.copy()
            else:
                ema_feat = args.ema_alpha * feat + (1.0 - args.ema_alpha) * ema_feat

            pred_t0 = time.perf_counter()
            pred_idx = int(clf.predict(ema_feat.reshape(1, -1))[0])
            raw_label = classes[pred_idx] if 0 <= pred_idx < len(classes) else str(pred_idx)
            pred_window.append(raw_label)
            counts = Counter(pred_window)
            stable_label = max(counts, key=counts.get)
            if counts[stable_label] < args.stable_threshold:
                stable_label = "..."
            t_predict = (time.perf_counter() - pred_t0) * 1000.0

            draw_t0 = time.perf_counter()
            if not args.no_draw:
                cv2.putText(frame_bgr, f"Pred: {stable_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("Realtime Inference — press 'q' to quit", frame_bgr)
            t_draw = (time.perf_counter() - draw_t0) * 1000.0

            if args.tts and stable_label != "...":
                now = time.time()
                if stable_label != last_spoken_label and (now - last_spoken_ts) >= args.min_say_interval:
                    try:
                        tts_engine.say(stable_label)
                        tts_engine.runAndWait()
                        last_spoken_label = stable_label
                        last_spoken_ts = now
                    except Exception as exc:
                        print(f"[w] TTS error: {exc}")

            t_total = (time.perf_counter() - loop_t0) * 1000.0
            fps = 1000.0 / max(t_total, 1e-6)

            if csv_writer:
                csv_writer.writerow([
                    time.time(),
                    round(fps, 3),
                    round(t_capture, 3),
                    round(t_mediapipe, 3),
                    round(t_features, 3),
                    round(t_predict, 3),
                    round(t_draw, 3),
                    round(t_total, 3),
                    stable_label,
                ])

            if not args.no_draw and (cv2.waitKey(1) & 0xFF == ord("q")):
                break

    finally:
        if csv_file:
            csv_file.close()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
