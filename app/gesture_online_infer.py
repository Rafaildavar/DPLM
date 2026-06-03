"""
Онлайн-распознавание жестов в процессе приложения (MediaPipe + KNN).

С mediapipe>=0.10.33 модуль ``mp.solutions`` удалён, поэтому используется
исключительно Tasks API через общий хелпер ``cv.hand_landmarker``.
"""
from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from cv.hand_landmarker import (
    DetectedHand,
    HandLandmarkerVideo,
    normalize_landmarks,
    resolve_hand_landmarker_task_path,
)
from cv.gesture_pose_signature import (
    build_signature_metadata_from_data_root,
    hands_non_thumb_count,
    load_signature_metadata,
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
        gesture_signatures_path: Optional[Path] = None,
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
        # Режим детектора рук (что мы реально ловим/рисуем). В auto-hand
        # режиме детектор всегда ищет до 2 рук, а размерность модели решает,
        # сколько рук попадет в классификатор.
        self._requested_two_hands = bool(two_hands)
        self._detector_two_hands = True
        self._window: Deque[np.ndarray] = deque(maxlen=max(1, window))
        self._finger_count_window: Deque[int] = deque(maxlen=5)
        self._gesture_signatures: dict[str, dict[str, Any]] = {}

        model_path = model_path or (PROJECT_ROOT / "models" / "knn.pkl")
        classes_path = classes_path or (PROJECT_ROOT / "models" / "classes.json")
        feature_dim_path = feature_dim_path or (PROJECT_ROOT / "models" / "feature_dim.txt")
        gesture_signatures_path = gesture_signatures_path or (
            model_path.parent / "gesture_signatures.json"
        )

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
                model_feature_dim = int(getattr(self._clf, "n_features_in_", 0) or 0)
                if model_feature_dim > 0 and model_feature_dim != self._feature_dim:
                    print(
                        "[w] feature_dim.txt does not match knn.pkl: "
                        f"{self._feature_dim} -> {model_feature_dim}",
                        flush=True,
                    )
                    self._feature_dim = model_feature_dim
            except Exception as e:
                self._clf = None
                self._model_error = f"knn.pkl: {e}"
        else:
            self._clf = None
            if not self._model_error:
                self._model_error = "Нет models/knn.pkl"
        self._classifier_two_hands = self._feature_dim == 84
        self._detector_two_hands = True
        self._gesture_signatures = self._load_gesture_signatures(gesture_signatures_path)

        try:
            task_path = str(resolve_hand_landmarker_task_path())
            self._detector = HandLandmarkerVideo(
                num_hands=2,
                task_path=task_path,
            )
        except Exception as e:
            self._init_error = str(e)
            self._detector = None

    def _load_gesture_signatures(self, signatures_path: Path) -> dict[str, dict[str, Any]]:
        metadata = load_signature_metadata(signatures_path) if signatures_path.exists() else {}
        if not metadata and self._classes:
            metadata = build_signature_metadata_from_data_root(
                PROJECT_ROOT / "data" / "gestures",
                self._classes,
                max_hands=2 if self._classifier_two_hands else 1,
            )

        classes_raw = metadata.get("classes") if isinstance(metadata, dict) else None
        if not isinstance(classes_raw, dict):
            return {}

        signatures: dict[str, dict[str, Any]] = {}
        for label, raw in classes_raw.items():
            if not isinstance(raw, dict):
                continue
            try:
                count = int(raw["non_thumb_count"])
                stability = float(raw.get("stability", 1.0))
            except (KeyError, TypeError, ValueError):
                continue
            if stability < 0.55:
                continue
            signature = {"non_thumb_count": count, "stability": stability}
            signatures[str(label)] = signature
            signatures[str(label).lower()] = signature
        return signatures

    def _stable_non_thumb_count(self, hands: List[DetectedHand]) -> int | None:
        count = hands_non_thumb_count(
            (h.landmarks for h in hands),
            max_hands=2 if self._classifier_two_hands else 1,
        )
        if count is None:
            self._finger_count_window.clear()
            return None

        self._finger_count_window.append(int(count))
        dominant, support = Counter(self._finger_count_window).most_common(1)[0]
        if support >= max(1, (len(self._finger_count_window) + 1) // 2):
            return int(dominant)
        return int(count)

    def _pose_matches_prediction(self, label: str, current_count: int | None) -> bool:
        if current_count is None or not label:
            return True

        signature = self._gesture_signatures.get(label) or self._gesture_signatures.get(
            label.lower()
        )
        if not signature:
            return True

        expected_count = int(signature["non_thumb_count"])
        if int(current_count) == expected_count:
            return True

        print(
            "[i] gesture rejected by finger count: "
            f"label={label!r} expected={expected_count} current={current_count}",
            flush=True,
        )
        return False

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
        Legacy setter для старого UI.

        Сейчас инференс работает в auto-hand режиме: детектор всегда ищет до
        двух рук, а классификатор получает 42 или 84 признака по своей
        фактической размерности. Поэтому переключатель больше не пересоздаёт
        MediaPipe, а только сбрасывает сглаживающее окно.
        """
        target = bool(enabled)
        if target == self._requested_two_hands:
            return

        self._requested_two_hands = target
        self._window.clear()
        self._finger_count_window.clear()

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
        self._window.clear()
        self._finger_count_window.clear()

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

    def _ordered_hands(self, hands: List[DetectedHand]) -> List[DetectedHand]:
        def sort_key(hand: DetectedHand) -> tuple[int, float]:
            handedness = (hand.handedness or "").strip().lower()
            if handedness == "right":
                side_rank = 0
            elif handedness == "left":
                side_rank = 1
            else:
                side_rank = 2
            return side_rank, -float(hand.score or 0.0)

        return sorted(hands, key=sort_key)

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

        hands = self._ordered_hands(self._detector.detect_for_video_rgb(frame_rgb))
        landmarks_json = self._build_overlay_payload(hands)

        # Нормализуем точки рук для классификатора.
        normalized: List[np.ndarray] = []
        for h in hands:
            try:
                normalized.append(normalize_landmarks(h.landmarks))
            except Exception:
                continue

        if not normalized:
            self._window.clear()
            self._finger_count_window.clear()
            return {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": landmarks_json,
            }

        current_finger_count = self._stable_non_thumb_count(hands)

        if self._classifier_two_hands:
            if len(normalized) >= 2:
                frame_vec = np.concatenate(normalized[:2], axis=0)
            elif len(normalized) == 1:
                frame_vec = np.concatenate(
                    [normalized[0], np.zeros((21, 2), dtype=np.float32)], axis=0
                )
        else:
            frame_vec = normalized[0]

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
            try:
                pred_idx = int(self._clf.predict(avg_feat)[0])
                label = (
                    self._classes[pred_idx]
                    if 0 <= pred_idx < len(self._classes)
                    else str(pred_idx)
                )
                proba = self._clf.predict_proba(avg_feat)[0]
                confidence = float(np.max(proba))
            except Exception as e:
                print(f"[!] classifier prediction failed: {e}", flush=True)
                self._window.clear()
                return {
                    "label": "",
                    "confidence": 0.0,
                    "landmarks_json": landmarks_json,
                }
        elif normalized:
            confidence = min(0.35 + 0.02 * len(self._window), 0.55)

        if label and not self._pose_matches_prediction(label, current_finger_count):
            self._window.clear()
            return {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": landmarks_json,
            }

        return {
            "label": label,
            "confidence": confidence,
            "landmarks_json": landmarks_json,
        }
