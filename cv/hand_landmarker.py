"""
Единый адаптер MediaPipe Hand Landmarker (Tasks API).

Зачем: начиная с mediapipe 0.10.33, модуль ``mp.solutions`` удалён, и весь
проект (CLI-скрипты в ``cv/`` + встроенный пайплайн в ``app/``) должен
переходить на Tasks API. Здесь собрана общая логика:

* Поиск/скачивание ``hand_landmarker.task`` (env-переменная, ``models/``,
  кэш в ``~/.dplm/``).
* Создание детектора с поддержкой одной/двух рук, с режимом VIDEO.
* Удобный ``detect_for_video_rgb()`` + готовая отрисовка скелета на BGR.

Старый API ``mp.solutions.hands`` имитируется минимально — только то, что
нужно нашим скриптам.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# 21 ребро скелета руки MediaPipe (как HAND_CONNECTIONS).
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def _hand_task_cache_path() -> Path:
    return Path.home() / ".dplm" / "hand_landmarker.task"


def resolve_hand_landmarker_task_path() -> Path:
    """Найти/скачать файл модели hand_landmarker.task. Возвращает абсолютный путь."""
    env = os.environ.get("MEDIAPIPE_HAND_TASK", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"MEDIAPIPE_HAND_TASK не найден: {p}")

    bundled = PROJECT_ROOT / "models" / "hand_landmarker.task"
    if bundled.is_file():
        return bundled.resolve()

    cache = _hand_task_cache_path()
    if cache.is_file():
        return cache.resolve()

    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".task.part")
    try:
        print(f"[i] Загрузка Hand Landmarker → {cache}")
        req = urllib.request.Request(
            _HAND_TASK_URL,
            headers={"User-Agent": "DPLM/1.0 (cv.hand_landmarker)"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        tmp.write_bytes(data)
        tmp.replace(cache)
    except (urllib.error.URLError, OSError) as e:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"Не удалось скачать hand_landmarker.task ({e}). "
            "Задайте MEDIAPIPE_HAND_TASK=путь к .task."
        ) from e
    return cache.resolve()


@dataclass
class DetectedHand:
    """Один обнаруженный объект руки в кадре."""
    landmarks: List[Tuple[float, float]]  # 21 пары (x, y) в долях кадра 0..1
    handedness: str = ""                  # "Left" / "Right" / ""
    score: float = 0.0


class HandLandmarkerVideo:
    """
    Тонкая обёртка над ``mp.tasks.vision.HandLandmarker`` (RunningMode.VIDEO).

    Использование (один поток!):

        det = HandLandmarkerVideo(num_hands=2)
        for frame_rgb in stream:
            hands = det.detect_for_video_rgb(frame_rgb)
            ...
        det.close()
    """

    def __init__(
        self,
        num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        task_path: Optional[str] = None,
    ) -> None:
        import mediapipe as mp  # noqa: WPS433 - локальный импорт намеренно

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        path = task_path or str(resolve_hand_landmarker_task_path())
        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=path,
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max(1, int(num_hands)),
            min_hand_detection_confidence=float(min_detection_confidence),
            min_hand_presence_confidence=float(min_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._mp = mp
        self._landmarker = HandLandmarker.create_from_options(options)
        self._ts_ms = 0

    @property
    def closed(self) -> bool:
        return self._landmarker is None

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    def __enter__(self) -> "HandLandmarkerVideo":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def detect_for_video_rgb(self, frame_rgb: np.ndarray) -> List[DetectedHand]:
        """Получить список рук для очередного кадра RGB (uint8)."""
        if self._landmarker is None or frame_rgb is None or frame_rgb.size == 0:
            return []

        rgb = np.ascontiguousarray(frame_rgb)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # MediaPipe требует строго возрастающий timestamp; шаг ~33 мс (≈30 FPS) подходит.
        self._ts_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._ts_ms)

        hands_out: List[DetectedHand] = []
        if not result.hand_landmarks:
            return hands_out

        handedness_lists = getattr(result, "handedness", None) or []
        for i, lm_list in enumerate(result.hand_landmarks):
            pts = [(float(lm.x), float(lm.y)) for lm in lm_list]
            if len(pts) != 21:
                continue
            label = ""
            score = 0.0
            if i < len(handedness_lists) and handedness_lists[i]:
                top = handedness_lists[i][0]
                label = getattr(top, "category_name", "") or getattr(top, "display_name", "")
                score = float(getattr(top, "score", 0.0) or 0.0)
            hands_out.append(DetectedHand(landmarks=pts, handedness=label, score=score))

        return hands_out


def normalize_landmarks(
    landmarks_xy: List[Tuple[float, float]],
    method: str = "wrist_scale",
) -> np.ndarray:
    """Нормализация (21,2): запястье в (0,0), масштаб = max L2 до запястья."""
    pts = np.asarray(landmarks_xy, dtype=np.float32)
    if pts.shape != (21, 2):
        raise ValueError("Ожидалось 21 точка (x, y)")

    if method == "wrist_scale":
        wrist = pts[0].copy()
        pts -= wrist
        d = np.linalg.norm(pts, axis=1)
        scale = float(np.max(d))
        if scale < 1e-6:
            scale = 1.0
        pts /= scale
    elif method == "bbox":
        xmin, ymin = np.min(pts, axis=0)
        xmax, ymax = np.max(pts, axis=0)
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        diag = float(np.linalg.norm([xmax - xmin, ymax - ymin]))
        if diag < 1e-6:
            diag = 1.0
        pts -= np.array([cx, cy], dtype=np.float32)
        pts /= diag
    else:
        raise ValueError(f"Неизвестный метод нормализации: {method}")
    return pts


def draw_hand_overlay_bgr(
    frame_bgr: np.ndarray,
    hand: DetectedHand,
    *,
    color_lines: Tuple[int, int, int] = (0, 215, 255),
    color_points: Tuple[int, int, int] = (0, 255, 80),
    label: str = "",
) -> None:
    """Нарисовать скелет одной руки прямо на BGR-кадре (in-place)."""
    h, w = frame_bgr.shape[:2]
    pts_px = [(int(round(x * w)), int(round(y * h))) for (x, y) in hand.landmarks]

    for a, b in HAND_CONNECTIONS:
        if a < len(pts_px) and b < len(pts_px):
            cv2.line(frame_bgr, pts_px[a], pts_px[b], color_lines, 2, cv2.LINE_AA)

    for x, y in pts_px:
        cv2.circle(frame_bgr, (x, y), 4, color_points, -1, cv2.LINE_AA)

    text = label or hand.handedness
    if text and pts_px:
        x0, y0 = pts_px[0]
        cv2.putText(
            frame_bgr,
            text,
            (x0 + 10, max(0, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
