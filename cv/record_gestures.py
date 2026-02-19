import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import mediapipe as mp

from cv.gesture_features import build_frame_feature


# -----------------------------------------------
# Запись жестов: сохраняем последовательности ландмарков в .npy
# Комментарии на русском
# -----------------------------------------------

MACOS_BACKEND = cv2.CAP_AVFOUNDATION
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Запись жестов в NPY")
    p.add_argument("--label", required=True, help="Имя жеста (папка в data/gestures/<label>)")
    p.add_argument("--num-samples", type=int, default=20, help="Сколько семплов записать")
    p.add_argument("--two-hands", action="store_true", help="Учитывать вторую руку (если есть)")
    p.add_argument("--fps", type=int, default=30, help="Целевой FPS для записи")
    p.add_argument("--frames", type=int, default=30, help="Длина одного семпла (число кадров)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    label = args.label.strip()
    target_samples = max(1, args.num_samples)
    seq_len = max(1, args.frames)

    out_dir = Path("data/gestures") / label
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0, MACOS_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
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

    print("Управление: s — старт/стоп записи; n — сохранить семпл; q — выход")
    print(f"Метка жеста: {label}; нужно семплов: {target_samples}; длина семпла: {seq_len} кадров")

    recording = False
    buffer: List[np.ndarray] = []  # список кадров (D,)
    saved = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Предупреждение: кадр не прочитан")
                continue

            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            landmarks_this_frame: List[np.ndarray] = []
            handedness_labels: List[str] = []

            if results.multi_hand_landmarks:
                handedness = results.multi_handedness or []
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                    )
                    pts = np.asarray([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
                    landmarks_this_frame.append(pts)

                    hand_label = ""
                    if idx < len(handedness):
                        try:
                            hand_label = handedness[idx].classification[0].label
                        except Exception:
                            hand_label = ""
                    handedness_labels.append(hand_label)

            frame_vec = build_frame_feature(
                hand_landmarks=landmarks_this_frame,
                handedness_labels=handedness_labels,
                two_hands=args.two_hands,
                include_presence_mask=True,
            )

            # Режим записи: собираем кадры в буфер до длины seq_len
            if recording and frame_vec is not None:
                buffer.append(frame_vec)
                if len(buffer) >= seq_len:
                    print("[i] Достигнута длина семпла, нажмите n для сохранения или s для перезапуска")

            # Оверлей статуса
            status = f"label={label} saved={saved}/{target_samples} rec={'ON' if recording else 'OFF'} len={len(buffer)}"
            cv2.putText(frame_bgr, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            cv2.imshow("Record Gestures — s:rec n:save q:quit", frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # старт/стоп записи — очищаем буфер при старте
                recording = not recording
                if recording:
                    buffer = []
                    print("[+] Запись начата")
                else:
                    print("[i] Запись остановлена (буфер сохранён в памяти, нажмите n)")
            elif key == ord('n'):
                # сохранить текущий буфер как семпл (если есть данные)
                if len(buffer) == 0:
                    print("[!] Буфер пуст — нечего сохранять")
                    continue
                if len(buffer) < seq_len:
                    print(f"[!] Слишком короткий семпл: {len(buffer)}<{seq_len}")
                    continue

                # Сохраняем как (T, D), где D = 42×2 или 21×2 развёрнутые в вектор
                arr = np.asarray(buffer, dtype=np.float32)  # (T, 21×2) или (T, 42×2)
                # опционально можно разворачивать в (T, 84) или (T, 42)

                # Имя файла
                existing = sorted(out_dir.glob("sample_*.npy"))
                idx = len(existing)
                out_path = out_dir / f"sample_{idx:04d}.npy"
                np.save(out_path, arr)
                saved += 1
                print(f"[✓] Сохранено: {out_path}")

                # Останавливаем запись и чистим буфер
                recording = False
                buffer = []

                if saved >= target_samples:
                    print("[✓] Достигнуто целевое число семплов — выходим")
                    break

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


