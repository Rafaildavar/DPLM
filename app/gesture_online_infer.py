"""
Онлайн-распознавание жестов в процессе приложения (MediaPipe + KNN).

С mediapipe>=0.10.33 модуль ``mp.solutions`` удалён, поэтому используется
исключительно Tasks API через общий хелпер ``cv.hand_landmarker``.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from cv.hand_landmarker import (
    DetectedHand,
    HandLandmarkerVideo,
    normalize_landmarks,
    resolve_hand_landmarker_task_path,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class GestureOnlineInfer:
    """
    Один кадр RGB → ключевые точки + (опционально) класс жеста и уверенность.

    Поддерживает runtime-переключение режима «одна рука / две руки» через
    :py:meth:`set_two_hands` (детектор пересоздаётся в этом же потоке).
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        classes_path: Optional[Path] = None,
        feature_dim_path: Optional[Path] = None,
        window: int = 30,
        two_hands: bool = False,
    ) -> None:
        self._init_error = ""
        self._model_error = ""
        self._detector: Optional[HandLandmarkerVideo] = None
        self._clf: Any = None
        self._classes: List[str] = []
        self._feature_dim = 42
        # Режим классификатора (фиксируется обученной моделью):
        self._classifier_two_hands = False
        # Режим детектора рук (что мы реально ловим/рисуем):
        self._detector_two_hands = bool(two_hands)
        self._window: Deque[np.ndarray] = deque(maxlen=max(1, window))

        model_path = model_path or (PROJECT_ROOT / "models" / "knn.pkl")
        classes_path = classes_path or (PROJECT_ROOT / "models" / "classes.json")
        feature_dim_path = feature_dim_path or (PROJECT_ROOT / "models" / "feature_dim.txt")

        try:
            import mediapipe  # noqa: F401  - проверим наличие пакета
        except ImportError as e:
            self._init_error = f"mediapipe: {e}"
            return

        if feature_dim_path.exists():
            try:
                self._feature_dim = int(feature_dim_path.read_text(encoding="utf-8").strip())
            except Exception:
                self._feature_dim = 42
        self._classifier_two_hands = self._feature_dim == 84
        # Если классификатор обучен на 2 руки — детектор тоже должен ловить 2.
        if self._classifier_two_hands:
            self._detector_two_hands = True

        if classes_path.exists():
            try:
                self._classes = json.loads(classes_path.read_text(encoding="utf-8"))
            except Exception as e:
                self._model_error = f"classes.json: {e}"
        else:
            self._model_error = "Нет models/classes.json"

        if model_path.exists():
            try:
                import warnings

                import joblib

                try:
                    from sklearn.exceptions import InconsistentVersionWarning
                except ImportError:
                    InconsistentVersionWarning = UserWarning  # type: ignore[misc,assignment]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    self._clf = joblib.load(str(model_path))
            except Exception as e:
                self._clf = None
                self._model_error = f"knn.pkl: {e}"
        else:
            self._clf = None
            if not self._model_error:
                self._model_error = "Нет models/knn.pkl"

        try:
            task_path = str(resolve_hand_landmarker_task_path())
            self._detector = HandLandmarkerVideo(
                num_hands=2 if self._detector_two_hands else 1,
                task_path=task_path,
            )
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

    @property
    def two_hands(self) -> bool:
        return self._detector_two_hands

    @property
    def classifier_requires_two_hands(self) -> bool:
        return self._classifier_two_hands

    def set_two_hands(self, enabled: bool) -> None:
        """
        Переключить режим детектора (1 ↔ 2 руки) во время работы.

        Если классификатор обучен на 2 руки (feature_dim=84), отключение
        игнорируется — иначе предсказание не будет иметь смысла.
        Должно вызываться из того же потока, где вызывается ``process_frame_rgb``.
        """
        target = bool(enabled) or self._classifier_two_hands
        if target == self._detector_two_hands:
            return

        self._detector_two_hands = target
        if self._detector is None:
            return
        try:
            self._detector.close()
        except Exception:
            pass
        self._detector = None
        try:
            task_path = str(resolve_hand_landmarker_task_path())
            self._detector = HandLandmarkerVideo(
                num_hands=2 if target else 1,
                task_path=task_path,
            )
            self._window.clear()
        except Exception as e:
            self._init_error = str(e)
            self._detector = None

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
        self._window.clear()

    def _build_overlay_payload(self, hands: List[DetectedHand]) -> str:
        """
        Сериализовать ключевые точки рук для QML.

        Формат: список рук, каждая рука — список [x, y] в нормализованных
        кадровых координатах 0..1. QML сторона способна нарисовать произвольное
        количество рук (см. CameraPreview.qml).
        """
        payload: List[List[List[float]]] = []
        for h in hands:
            payload.append([[float(x), float(y)] for (x, y) in h.landmarks])
        return json.dumps(payload, separators=(",", ":"))

    def process_frame_rgb(self, frame_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Args:
            frame_rgb: uint8 RGB, произвольный размер (как после cv2.flip + cvtColor).
        Returns:
            ``label``, ``confidence`` [0..1], ``landmarks_json`` (список рук
            ``[[[x,y],...], ...]`` в 0..1).
        """
        empty: Dict[str, Any] = {
            "label": "",
            "confidence": 0.0,
            "landmarks_json": "[]",
        }
        if self._detector is None or frame_rgb is None or frame_rgb.size == 0:
            return empty

        hands = self._detector.detect_for_video_rgb(frame_rgb)
        landmarks_json = self._build_overlay_payload(hands)

        # Нормализуем точки рук для классификатора.
        normalized: List[np.ndarray] = []
        for h in hands:
            try:
                normalized.append(normalize_landmarks(h.landmarks))
            except Exception:
                continue

        if self._classifier_two_hands:
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
                confidence = 0.75 if normalized else 0.0
        elif normalized:
            confidence = min(0.35 + 0.02 * len(self._window), 0.55)

        return {
            "label": label,
            "confidence": confidence,
            "landmarks_json": landmarks_json,
        }
