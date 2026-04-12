"""
Онлайн-распознавание жестов в процессе приложения (MediaPipe + KNN).
Совместимость: legacy mp.solutions (mediapipe < 0.10.31) и Tasks API (новые версии).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Официальная модель Hand Landmarker (Tasks), ~10 MB
_HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


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


def _hand_task_cache_path() -> Path:
    return Path.home() / ".dplm" / "hand_landmarker.task"


def _resolve_hand_landmarker_task_path() -> Path:
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
            headers={"User-Agent": "DPLM/1.0 (gesture_online_infer)"},
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
            "Задайте MEDIAPIPE_HAND_TASK=путь к .task или установите mediapipe<0.10.31."
        ) from e
    return cache.resolve()


class _HandDetector(ABC):
    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def detect(
        self, frame_rgb: np.ndarray
    ) -> Tuple[List[np.ndarray], List[List[float]]]:
        """
        Returns:
            landmarks_frames: список нормализованных (21,2) на каждую руку
            overlay_pts: точки [[x,y],...] первой руки для QML, норм. 0..1
        """
        ...


class _LegacySolutionsHands(_HandDetector):
    def __init__(self, two_hands: bool) -> None:
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2 if two_hands else 1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def close(self) -> None:
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass
            self._hands = None

    def detect(self, frame_rgb: np.ndarray) -> Tuple[List[np.ndarray], List[List[float]]]:
        landmarks_frames: List[np.ndarray] = []
        overlay_pts: List[List[float]] = []
        results = self._hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return landmarks_frames, overlay_pts
        for hand_landmarks in results.multi_hand_landmarks:
            pts = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            try:
                landmarks_frames.append(normalize_landmarks(pts))
            except Exception:
                continue
            if not overlay_pts:
                overlay_pts = [[float(lm.x), float(lm.y)] for lm in hand_landmarks.landmark]
        return landmarks_frames, overlay_pts


class _TasksApiHands(_HandDetector):
    def __init__(self, task_path: str, two_hands: bool) -> None:
        import mediapipe as mp

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2 if two_hands else 1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._mp = mp
        self._landmarker = HandLandmarker.create_from_options(options)
        self._ts_ms = 0

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    def detect(self, frame_rgb: np.ndarray) -> Tuple[List[np.ndarray], List[List[float]]]:
        landmarks_frames: List[np.ndarray] = []
        overlay_pts: List[List[float]] = []
        if self._landmarker is None:
            return landmarks_frames, overlay_pts

        rgb = np.ascontiguousarray(frame_rgb)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._ts_ms)

        if not result.hand_landmarks:
            return landmarks_frames, overlay_pts

        for lm_list in result.hand_landmarks:
            pts = [(float(lm.x), float(lm.y)) for lm in lm_list]
            if len(pts) != 21:
                continue
            try:
                landmarks_frames.append(normalize_landmarks(pts))
            except Exception:
                continue
            if not overlay_pts:
                overlay_pts = [[float(lm.x), float(lm.y)] for lm in lm_list]
        return landmarks_frames, overlay_pts


class GestureOnlineInfer:
    """
    Один кадр RGB → ключевые точки + (опционально) класс жеста и уверенность.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        classes_path: Optional[Path] = None,
        feature_dim_path: Optional[Path] = None,
        window: int = 30,
    ) -> None:
        self._init_error = ""
        self._model_error = ""
        self._detector: Optional[_HandDetector] = None
        self._clf: Any = None
        self._classes: List[str] = []
        self._feature_dim = 42
        self._two_hands = False
        self._window: Deque[np.ndarray] = deque(maxlen=max(1, window))

        model_path = model_path or (PROJECT_ROOT / "models" / "knn.pkl")
        classes_path = classes_path or (PROJECT_ROOT / "models" / "classes.json")
        feature_dim_path = feature_dim_path or (PROJECT_ROOT / "models" / "feature_dim.txt")

        try:
            import mediapipe as mp
        except ImportError as e:
            self._init_error = f"mediapipe: {e}"
            return

        if feature_dim_path.exists():
            try:
                self._feature_dim = int(feature_dim_path.read_text(encoding="utf-8").strip())
            except Exception:
                self._feature_dim = 42
        self._two_hands = self._feature_dim == 84

        if classes_path.exists():
            try:
                self._classes = json.loads(classes_path.read_text(encoding="utf-8"))
            except Exception as e:
                self._model_error = f"classes.json: {e}"
        else:
            self._model_error = "Нет models/classes.json"

        if model_path.exists():
            try:
                import joblib

                self._clf = joblib.load(str(model_path))
            except Exception as e:
                self._clf = None
                self._model_error = f"knn.pkl: {e}"
        else:
            self._clf = None
            if not self._model_error:
                self._model_error = "Нет models/knn.pkl"

        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self._detector = _LegacySolutionsHands(self._two_hands)
            elif hasattr(mp, "tasks"):
                task_p = str(_resolve_hand_landmarker_task_path())
                self._detector = _TasksApiHands(task_p, self._two_hands)
            else:
                self._init_error = "mediapipe: нет ни solutions, ни tasks"
        except Exception as e:
            self._init_error = str(e)
            self._detector = None

    @property
    def init_error(self) -> str:
        return self._init_error

    @property
    def model_error(self) -> str:
        return self._model_error

    @property
    def ready(self) -> bool:
        return self._detector is not None

    @property
    def has_classifier(self) -> bool:
        return self._clf is not None and bool(self._classes)

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
        self._window.clear()

    def process_frame_rgb(self, frame_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Args:
            frame_rgb: uint8 RGB, произвольный размер (как после cv2.flip + cvtColor).
        Returns:
            label, confidence [0..1], landmarks_json (список точек первой руки [[x,y],...] в 0..1).
        """
        empty: Dict[str, Any] = {
            "label": "",
            "confidence": 0.0,
            "landmarks_json": "[]",
        }
        if self._detector is None or frame_rgb is None or frame_rgb.size == 0:
            return empty

        landmarks_frames, overlay_pts = self._detector.detect(frame_rgb)
        landmarks_json = json.dumps(overlay_pts, separators=(",", ":"))

        if self._two_hands:
            if len(landmarks_frames) == 2:
                frame_vec = np.concatenate(landmarks_frames, axis=0)
            elif len(landmarks_frames) == 1:
                frame_vec = np.concatenate(
                    [landmarks_frames[0], np.zeros((21, 2), dtype=np.float32)], axis=0
                )
            else:
                frame_vec = np.zeros((42, 2), dtype=np.float32)
        else:
            frame_vec = (
                landmarks_frames[0]
                if landmarks_frames
                else np.zeros((21, 2), dtype=np.float32)
            )

        feat = frame_vec.reshape(-1)
        if feat.shape[0] != self._feature_dim:
            if feat.shape[0] > self._feature_dim:
                feat = feat[: self._feature_dim]
            else:
                pad = np.zeros(self._feature_dim - feat.shape[0], dtype=feat.dtype)
                feat = np.concatenate([feat, pad], axis=0)

        self._window.append(feat)

        label = ""
        confidence = 0.0

        if self._clf is not None and self._classes and len(self._window) > 0:
            avg_feat = np.mean(np.stack(tuple(self._window), axis=0), axis=0).reshape(1, -1)
            pred_idx = int(self._clf.predict(avg_feat)[0])
            label = (
                self._classes[pred_idx] if 0 <= pred_idx < len(self._classes) else str(pred_idx)
            )
            try:
                proba = self._clf.predict_proba(avg_feat)[0]
                confidence = float(np.max(proba))
            except Exception:
                confidence = 0.75 if landmarks_frames else 0.0
        elif landmarks_frames:
            label = ""
            confidence = min(0.35 + 0.02 * len(self._window), 0.55)

        return {
            "label": label,
            "confidence": confidence,
            "landmarks_json": landmarks_json,
        }