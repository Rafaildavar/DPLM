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
from time import perf_counter
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDIAPIPE_PROFILE_ENV = "DPLM_MEDIAPIPE_PROFILE"
MEDIAPIPE_SMOOTHING_ENV = "DPLM_MEDIAPIPE_SMOOTHING_ALPHA"

MEDIAPIPE_THRESHOLD_PROFILES: dict[str, dict[str, float]] = {
    "baseline_06": {
        "min_detection_confidence": 0.6,
        "min_presence_confidence": 0.6,
        "min_tracking_confidence": 0.6,
    },
    "recall_05": {
        "min_detection_confidence": 0.5,
        "min_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    },
    "strict_tracking": {
        "min_detection_confidence": 0.7,
        "min_presence_confidence": 0.5,
        "min_tracking_confidence": 0.7,
    },
    "redetect_presence": {
        "min_detection_confidence": 0.5,
        "min_presence_confidence": 0.7,
        "min_tracking_confidence": 0.5,
    },
}

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
            headers={"User-Agent": "GestureBind/1.0 (cv.hand_landmarker)"},
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
    landmarks_xyz: List[Tuple[float, float, float]] | None = None
    world_landmarks: List[Tuple[float, float, float]] | None = None


def mediapipe_profile_settings(profile: str | None = None) -> dict[str, Any]:
    """Return named MediaPipe threshold profile used by live A/B tests."""
    requested = (
        profile
        if profile is not None
        else os.environ.get(MEDIAPIPE_PROFILE_ENV, "baseline_06")
    )
    clean = str(requested or "baseline_06").strip().lower()
    if clean not in MEDIAPIPE_THRESHOLD_PROFILES:
        clean = "baseline_06"
    settings: dict[str, Any] = {"profile": clean}
    settings.update(MEDIAPIPE_THRESHOLD_PROFILES[clean])
    return settings


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return float(default)


def _confidence(value: float | None, default: float) -> float:
    if value is None:
        return float(default)
    return max(0.0, min(1.0, float(value)))


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
        min_detection_confidence: float | None = None,
        min_presence_confidence: float | None = None,
        min_tracking_confidence: float | None = None,
        task_path: Optional[str] = None,
        profile: str | None = None,
        landmark_smoothing_alpha: float | None = None,
    ) -> None:
        import mediapipe as mp  # noqa: WPS433 - локальный импорт намеренно

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        path = task_path or str(resolve_hand_landmarker_task_path())
        profile_settings = mediapipe_profile_settings(profile)
        detection_confidence = _confidence(
            min_detection_confidence,
            float(profile_settings["min_detection_confidence"]),
        )
        presence_confidence = _confidence(
            min_presence_confidence,
            float(profile_settings["min_presence_confidence"]),
        )
        tracking_confidence = _confidence(
            min_tracking_confidence,
            float(profile_settings["min_tracking_confidence"]),
        )
        smoothing_alpha = (
            _float_env(MEDIAPIPE_SMOOTHING_ENV, 0.0)
            if landmark_smoothing_alpha is None
            else float(landmark_smoothing_alpha)
        )
        smoothing_alpha = max(0.0, min(1.0, smoothing_alpha))
        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=path,
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max(1, int(num_hands)),
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._mp = mp
        self._landmarker = HandLandmarker.create_from_options(options)
        self._ts_ms = 0
        self._profile = str(profile_settings["profile"])
        self._thresholds = {
            "min_detection_confidence": detection_confidence,
            "min_presence_confidence": presence_confidence,
            "min_tracking_confidence": tracking_confidence,
        }
        self._smoothing_alpha = smoothing_alpha
        self._previous_smoothed_hands: dict[str, DetectedHand] = {}
        self._last_timestamp_source = "synthetic_33ms"
        self._last_metrics: dict[str, Any] = self._empty_metrics(
            timestamp_ms=0,
            timestamp_source=self._last_timestamp_source,
            detection_ms=0.0,
        )

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def thresholds(self) -> dict[str, float]:
        return dict(self._thresholds)

    @property
    def smoothing_alpha(self) -> float:
        return float(self._smoothing_alpha)

    @property
    def last_metrics(self) -> dict[str, Any]:
        return dict(self._last_metrics)

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

    def _next_timestamp_ms(self, timestamp_ms: int | float | None = None) -> int:
        if timestamp_ms is None:
            self._ts_ms += 33
            self._last_timestamp_source = "synthetic_33ms"
            return self._ts_ms

        try:
            candidate = int(round(float(timestamp_ms)))
        except (TypeError, ValueError):
            candidate = self._ts_ms + 33
            self._last_timestamp_source = "synthetic_33ms_invalid_input"
        else:
            self._last_timestamp_source = "real_monotonic"

        if candidate <= self._ts_ms:
            candidate = self._ts_ms + 1
            if self._last_timestamp_source == "real_monotonic":
                self._last_timestamp_source = "real_monotonic_adjusted"
        self._ts_ms = candidate
        return self._ts_ms

    def detect_for_video_rgb(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int | float | None = None,
    ) -> List[DetectedHand]:
        """Получить список рук для очередного кадра RGB (uint8)."""
        if self._landmarker is None or frame_rgb is None or frame_rgb.size == 0:
            self._last_metrics = self._empty_metrics(
                timestamp_ms=int(self._ts_ms),
                timestamp_source=self._last_timestamp_source,
                detection_ms=0.0,
            )
            return []

        rgb = np.ascontiguousarray(frame_rgb)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # MediaPipe требует строго возрастающий timestamp. В live-пайплайне
        # передаем timestamp кадра; для CLI/старых вызовов оставлен fallback.
        ts_ms = self._next_timestamp_ms(timestamp_ms)
        started = perf_counter()
        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        detection_ms = (perf_counter() - started) * 1000.0

        hands_out: List[DetectedHand] = []
        if not result.hand_landmarks:
            self._previous_smoothed_hands.clear()
            self._last_metrics = self._empty_metrics(
                timestamp_ms=ts_ms,
                timestamp_source=self._last_timestamp_source,
                detection_ms=detection_ms,
            )
            return hands_out

        handedness_lists = getattr(result, "handedness", None) or []
        world_lists = getattr(result, "hand_world_landmarks", None) or []
        for i, lm_list in enumerate(result.hand_landmarks):
            pts = [(float(lm.x), float(lm.y)) for lm in lm_list]
            pts_xyz = [
                (float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0)))
                for lm in lm_list
            ]
            if len(pts) != 21:
                continue
            world_pts: List[Tuple[float, float, float]] | None = None
            if i < len(world_lists) and world_lists[i]:
                world_pts = [
                    (
                        float(getattr(lm, "x", 0.0)),
                        float(getattr(lm, "y", 0.0)),
                        float(getattr(lm, "z", 0.0)),
                    )
                    for lm in world_lists[i]
                ]
                if len(world_pts) != 21:
                    world_pts = None
            label = ""
            score = 0.0
            if i < len(handedness_lists) and handedness_lists[i]:
                top = handedness_lists[i][0]
                label = getattr(top, "category_name", "") or getattr(top, "display_name", "")
                score = float(getattr(top, "score", 0.0) or 0.0)
            hands_out.append(
                DetectedHand(
                    landmarks=pts,
                    handedness=label,
                    score=score,
                    landmarks_xyz=pts_xyz,
                    world_landmarks=world_pts,
                )
            )

        hands_out = self._smooth_hands(hands_out)
        self._last_metrics = self._quality_metrics(
            hands_out,
            timestamp_ms=ts_ms,
            timestamp_source=self._last_timestamp_source,
            detection_ms=detection_ms,
        )
        return hands_out

    def _empty_metrics(
        self,
        *,
        timestamp_ms: int,
        timestamp_source: str,
        detection_ms: float,
    ) -> dict[str, Any]:
        return {
            "mediapipe_profile": self._profile,
            "mediapipe_min_detection_confidence": self._thresholds[
                "min_detection_confidence"
            ],
            "mediapipe_min_presence_confidence": self._thresholds[
                "min_presence_confidence"
            ],
            "mediapipe_min_tracking_confidence": self._thresholds[
                "min_tracking_confidence"
            ],
            "mediapipe_smoothing_alpha": self._smoothing_alpha,
            "mediapipe_timestamp_ms": int(timestamp_ms),
            "mediapipe_timestamp_source": str(timestamp_source),
            "mediapipe_detection_ms": round(float(detection_ms), 3),
            "hand_detected": False,
            "hand_count": 0,
            "handedness_score_max": 0.0,
            "landmark_z_available": False,
            "world_landmarks_available": False,
            "landmark_z_range": 0.0,
            "world_z_range": 0.0,
            "hand_bbox_area": 0.0,
            "hand_bbox_diag": 0.0,
        }

    def _quality_metrics(
        self,
        hands: List[DetectedHand],
        *,
        timestamp_ms: int,
        timestamp_source: str,
        detection_ms: float,
    ) -> dict[str, Any]:
        metrics = self._empty_metrics(
            timestamp_ms=timestamp_ms,
            timestamp_source=timestamp_source,
            detection_ms=detection_ms,
        )
        if not hands:
            return metrics

        metrics["hand_detected"] = True
        metrics["hand_count"] = len(hands)
        metrics["handedness_score_max"] = round(
            max(float(hand.score or 0.0) for hand in hands),
            4,
        )
        primary = hands[0]
        pts = np.asarray(primary.landmarks, dtype=np.float32)
        if pts.shape == (21, 2):
            size = pts.max(axis=0) - pts.min(axis=0)
            metrics["hand_bbox_area"] = round(float(size[0] * size[1]), 6)
            metrics["hand_bbox_diag"] = round(float(np.linalg.norm(size)), 6)

        xyz = np.asarray(primary.landmarks_xyz or [], dtype=np.float32)
        if xyz.shape == (21, 3):
            metrics["landmark_z_available"] = True
            metrics["landmark_z_range"] = round(
                float(np.max(xyz[:, 2]) - np.min(xyz[:, 2])),
                6,
            )
        world = np.asarray(primary.world_landmarks or [], dtype=np.float32)
        if world.shape == (21, 3):
            metrics["world_landmarks_available"] = True
            metrics["world_z_range"] = round(
                float(np.max(world[:, 2]) - np.min(world[:, 2])),
                6,
            )
        return metrics

    def _smooth_hands(self, hands: List[DetectedHand]) -> List[DetectedHand]:
        alpha = float(self._smoothing_alpha)
        if alpha <= 0.0 or alpha >= 1.0:
            self._previous_smoothed_hands = {
                self._smooth_key(hand, index): hand
                for index, hand in enumerate(hands)
            }
            return hands
        if not hands:
            self._previous_smoothed_hands.clear()
            return hands

        smoothed: List[DetectedHand] = []
        next_previous: dict[str, DetectedHand] = {}
        for index, hand in enumerate(hands):
            key = self._smooth_key(hand, index)
            previous = self._previous_smoothed_hands.get(key)
            current_xy = np.asarray(hand.landmarks, dtype=np.float32)
            if previous is None or current_xy.shape != (21, 2):
                smooth_hand = hand
            else:
                previous_xy = np.asarray(previous.landmarks, dtype=np.float32)
                if previous_xy.shape == (21, 2):
                    xy = alpha * current_xy + (1.0 - alpha) * previous_xy
                    landmarks = [
                        (float(x), float(y))
                        for x, y in xy.astype(float).tolist()
                    ]
                else:
                    landmarks = hand.landmarks
                landmarks_xyz = self._smooth_xyz(
                    hand.landmarks_xyz,
                    previous.landmarks_xyz,
                    alpha,
                )
                world_landmarks = self._smooth_xyz(
                    hand.world_landmarks,
                    previous.world_landmarks,
                    alpha,
                )
                smooth_hand = DetectedHand(
                    landmarks=landmarks,
                    handedness=hand.handedness,
                    score=hand.score,
                    landmarks_xyz=landmarks_xyz,
                    world_landmarks=world_landmarks,
                )
            smoothed.append(smooth_hand)
            next_previous[key] = smooth_hand
        self._previous_smoothed_hands = next_previous
        return smoothed

    @staticmethod
    def _smooth_key(hand: DetectedHand, index: int) -> str:
        handedness = str(hand.handedness or "").strip().lower()
        return f"{handedness or 'unknown'}:{index}"

    @staticmethod
    def _smooth_xyz(
        current: List[Tuple[float, float, float]] | None,
        previous: List[Tuple[float, float, float]] | None,
        alpha: float,
    ) -> List[Tuple[float, float, float]] | None:
        if current is None:
            return None
        current_arr = np.asarray(current, dtype=np.float32)
        if current_arr.shape != (21, 3):
            return current
        previous_arr = np.asarray(previous or [], dtype=np.float32)
        if previous_arr.shape != (21, 3):
            return current
        smoothed = alpha * current_arr + (1.0 - alpha) * previous_arr
        return [
            (float(x), float(y), float(z))
            for x, y, z in smoothed.astype(float).tolist()
        ]


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
