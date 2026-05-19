"""
Запись жестов: сохраняем последовательности ландмарков в .npy.

С mediapipe>=0.10.33 используется Tasks API через ``cv.hand_landmarker``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Запись жестов в NPY")
    p.add_argument("--label", required=True, help="Имя жеста (папка в data/gestures/<label>)")
    p.add_argument("--num-samples", type=int, default=20, help="Сколько семплов записать")
    p.add_argument("--two-hands", action="store_true", help="Учитывать вторую руку (если есть)")
    p.add_argument("--fps", type=int, default=30, help="Целевой FPS для записи")
    p.add_argument("--frames", type=int, default=30, help="Длина одного семпла (число кадров)")
    p.add_argument("--data-root", default="data/gestures", help="Корень датасета")
    p.add_argument("--camera-index", type=int, default=0, help="Индекс камеры OpenCV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    label = args.label.strip()
    target_samples = max(1, args.num_samples)
    seq_len = max(1, args.frames)

    out_dir = Path(args.data_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(int(args.camera_index), MACOS_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        print("Ошибка открытия камеры (проверьте доступ в Privacy & Security → Camera)")
        sys.exit(1)

    detector = HandLandmarkerVideo(
        num_hands=2 if args.two_hands else 1,
        min_detection_confidence=0.6,
        min_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    print("Управление: s — старт/стоп записи; n — сохранить семпл; q — выход")
    print(f"Метка жеста: {label}; нужно семплов: {target_samples}; длина семпла: {seq_len} кадров")

    recording = False
    buffer: List[np.ndarray] = []
    saved = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Предупреждение: кадр не прочитан")
                continue

            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            hands = detector.detect_for_video_rgb(frame_rgb)

            landmarks_this_frame: List[np.ndarray] = []
            for h in hands:
                draw_hand_overlay_bgr(frame_bgr, h, label=h.handedness)
                try:
                    landmarks_this_frame.append(normalize_landmarks(h.landmarks))
                except Exception:
                    continue

            if args.two_hands:
                if len(landmarks_this_frame) >= 2:
                    frame_vec = np.concatenate(landmarks_this_frame[:2], axis=0)
                elif len(landmarks_this_frame) == 1:
                    frame_vec = np.concatenate(
                        [landmarks_this_frame[0], np.zeros((21, 2), dtype=np.float32)],
                        axis=0,
                    )
                else:
                    frame_vec = np.zeros((42, 2), dtype=np.float32)
            else:
                if len(landmarks_this_frame) >= 1:
                    frame_vec = landmarks_this_frame[0]
                else:
                    frame_vec = np.zeros((21, 2), dtype=np.float32)

            if recording and frame_vec is not None:
                buffer.append(frame_vec)
                if len(buffer) >= seq_len:
                    print("[i] Достигнута длина семпла, нажмите n для сохранения или s для перезапуска")

            status = (
                f"label={label} saved={saved}/{target_samples} "
                f"rec={'ON' if recording else 'OFF'} len={len(buffer)} hands={len(hands)}"
            )
            cv2.putText(frame_bgr, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Record Gestures — s:rec n:save q:quit", frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                recording = not recording
                if recording:
                    buffer = []
                    print("[+] Запись начата")
                else:
                    print("[i] Запись остановлена (буфер сохранён в памяти, нажмите n)")
            elif key == ord('n'):
                if len(buffer) == 0:
                    print("[!] Буфер пуст — нечего сохранять")
                    continue
                if len(buffer) < seq_len:
                    print(f"[!] Слишком короткий семпл: {len(buffer)}<{seq_len}")
                    continue

                arr = np.asarray(buffer, dtype=np.float32)
                existing = sorted(out_dir.glob("sample_*.npy"))
                idx = len(existing)
                out_path = out_dir / f"sample_{idx:04d}.npy"
                np.save(out_path, arr)
                saved += 1
                print(f"[✓] Сохранено: {out_path}")

                recording = False
                buffer = []

                if saved >= target_samples:
                    print("[✓] Достигнуто целевое число семплов — выходим")
                    break

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
