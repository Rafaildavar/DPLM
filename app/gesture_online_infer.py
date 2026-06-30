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

from cv.gesture_features import (
    DYNAMIC_SEQUENCE_TARGET_FRAMES,
    DYNAMIC_SEQUENCE_LONG_TARGET_FRAMES,
    FEATURE_DYNAMIC_SEQUENCE,
    FEATURE_DYNAMIC_SEQUENCE_72,
    FEATURE_DYNAMIC_STATS,
    FEATURE_HYBRID_STATS,
    FEATURE_STATIC_MEAN,
    FEATURE_STATIC_STATS,
    build_feature_vector,
    infer_raw_dim_from_feature_size,
    trajectory_features,
)
from cv.intent_gate import (
    DEFAULT_INTENT_TARGET_DIM,
    build_intent_feature_vector,
)
from cv.dynamic_direction import classify_swipe_direction
from cv.dynamic_motion import DynamicMotionSegmenter
from cv.dynamic_prototype import (
    load_dynamic_prototype_model,
    predict_dynamic_prototype,
)
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
from app.services.gesture_taxonomy import (
    GESTURE_TYPE_NEGATIVE,
    GestureTaxonomy,
    load_gesture_taxonomy,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DYNAMIC_GATE_MIN_PATH_LENGTH = 0.08
DYNAMIC_GATE_MIN_DISPLACEMENT = 0.04
DYNAMIC_GATE_DIRECTION_THRESHOLD = 0.035
DYNAMIC_INTENT_MIN_PATH_LENGTH = 0.04
DYNAMIC_INTENT_MIN_DISPLACEMENT = 0.025
DYNAMIC_DIRECTION_DOMINANCE_RATIO = 1.15
DYNAMIC_NEGATIVE_REJECT_THRESHOLD = 0.72
DYNAMIC_COMPLEX_MODEL_MIN_CONFIDENCE = 0.60
DYNAMIC_COMPLEX_MODEL_MIN_MARGIN = 0.08
DYNAMIC_COMPLEX_MODEL_NEAR_TOP_MIN_CONFIDENCE = 0.48
DYNAMIC_COMPLEX_MODEL_NEAR_TOP_MAX_GAP = 0.12
DYNAMIC_COMPOUND_DIRECTION_MAX_AXIS_RATIO = 3.0
DYNAMIC_PROTOTYPE_OVERRIDE_MIN_CONFIDENCE = 0.68
DYNAMIC_MODEL_PROTOTYPE_OVERRIDE_MIN_CONFIDENCE = 0.88
DYNAMIC_MODEL_PROTOTYPE_OVERRIDE_MIN_MARGIN = 0.18
DYNAMIC_SEGMENT_PRE_ROLL_FRAMES = 3
DYNAMIC_SEGMENT_ONSET_PATH = 0.015
DYNAMIC_SEGMENT_ONSET_DISPLACEMENT = 0.012
DYNAMIC_SEGMENT_MIN_ACTIVE_FRAMES = 3
STATIC_REJECTION_NEGATIVE_CLASSES = "negative_classes"
STATIC_REJECTION_CONFIDENCE_THRESHOLD = "confidence_threshold"
STATIC_REJECTION_OPEN_SET_POLICY = "open_set_policy"
STATIC_REJECTION_ONE_VS_REST_LOGREG = "one_vs_rest_logreg"
STATIC_REJECTION_ONE_CLASS_SVM = "one_class_svm"
STATIC_REJECTION_ISOLATION_FOREST = "isolation_forest"
STATIC_REJECTION_LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
STATIC_REJECTION_METRIC_NCA_CENTROID = "metric_nca_centroid"
STATIC_REJECTION_MLP_NEGATIVE_CLASSES = "mlp_negative_classes"
DEFAULT_STATIC_REJECTION_METHOD = STATIC_REJECTION_OPEN_SET_POLICY
MIN_FINGER_SIGNATURE_STABILITY = 0.85
STATIC_REJECTION_METHODS = (
    STATIC_REJECTION_NEGATIVE_CLASSES,
    STATIC_REJECTION_CONFIDENCE_THRESHOLD,
    STATIC_REJECTION_OPEN_SET_POLICY,
    STATIC_REJECTION_ONE_VS_REST_LOGREG,
    STATIC_REJECTION_ONE_CLASS_SVM,
    STATIC_REJECTION_ISOLATION_FOREST,
    STATIC_REJECTION_LOCAL_OUTLIER_FACTOR,
    STATIC_REJECTION_METRIC_NCA_CENTROID,
    STATIC_REJECTION_MLP_NEGATIVE_CLASSES,
)


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
        feature_mode_path: Optional[Path] = None,
        feature_mode: Optional[str] = None,
        gesture_signatures_path: Optional[Path] = None,
        gesture_rejection_path: Optional[Path] = None,
        static_rejection_verifier_path: Optional[Path] = None,
        dynamic_prototypes_path: Optional[Path] = None,
        static_rejection_method: str = DEFAULT_STATIC_REJECTION_METHOD,
        window: int = 30,
        two_hands: bool = False,
        initialize_detector: bool = True,
    ) -> None:
        self._init_error = ""
        self._model_error = ""
        self._detector: Optional[HandLandmarkerVideo] = None
        self._clf: Any = None
        self._classes: List[str] = []
        self._feature_dim = 42
        self._feature_mode = FEATURE_STATIC_MEAN
        self._raw_feature_dim = 42
        # Режим классификатора (фиксируется обученной моделью):
        self._classifier_two_hands = False
        # Режим детектора рук (что мы реально ловим/рисуем). В auto-hand
        # режиме детектор всегда ищет до 2 рук, а размерность модели решает,
        # сколько рук попадет в классификатор.
        self._requested_two_hands = bool(two_hands)
        self._detector_two_hands = True
        self._window: Deque[np.ndarray] = deque(maxlen=max(1, window))
        self._intent_window: Deque[np.ndarray] = deque(
            maxlen=max(self._dynamic_sequence_target_frames(), int(window))
        )
        self._finger_count_window: Deque[int] = deque(maxlen=5)
        self._gesture_signatures: dict[str, dict[str, Any]] = {}
        self._gesture_rejection: dict[str, Any] = {}
        self._static_rejection_method = self._normalize_static_rejection_method(
            static_rejection_method
        )
        self._static_rejection_verifiers: dict[str, Any] = {}
        self._dynamic_prototypes: dict[str, Any] = {}
        self._dynamic_segmenter: DynamicMotionSegmenter | None = None
        self._pending_dynamic_prediction: tuple[str, float] | None = None
        self._pending_dynamic_repeats = 0
        self._pending_dynamic_motion_scale = 0.0
        self._last_dynamic_decision: dict[str, Any] = {}
        self._last_static_decision: dict[str, Any] = {}
        self._gesture_taxonomy: GestureTaxonomy | None = None

        model_path = model_path or (PROJECT_ROOT / "models" / "knn.pkl")
        classes_path = classes_path or (PROJECT_ROOT / "models" / "classes.json")
        feature_dim_path = feature_dim_path or (PROJECT_ROOT / "models" / "feature_dim.txt")
        feature_mode_path = feature_mode_path or (model_path.parent / "feature_mode.txt")
        gesture_signatures_path = gesture_signatures_path or (
            model_path.parent / "gesture_signatures.json"
        )
        gesture_rejection_path = gesture_rejection_path or (
            model_path.parent / "gesture_rejection.json"
        )
        static_rejection_verifier_path = static_rejection_verifier_path or (
            model_path.parent / "static_rejection_verifiers.pkl"
        )
        dynamic_prototypes_path = dynamic_prototypes_path or (
            model_path.parent / "dynamic_prototypes.json"
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
        if feature_mode:
            self._feature_mode = str(feature_mode).strip() or FEATURE_STATIC_MEAN
        elif feature_mode_path.exists():
            try:
                self._feature_mode = (
                    feature_mode_path.read_text(encoding="utf-8").strip()
                    or FEATURE_STATIC_MEAN
                )
            except Exception:
                self._feature_mode = FEATURE_STATIC_MEAN

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
        self._ensure_dynamic_sequence_feature_mode(model_path)
        self._raw_feature_dim = self._infer_raw_feature_dim()
        target_frames = self._dynamic_sequence_target_frames()
        if self._uses_temporal_features() and int(self._window.maxlen or 0) < target_frames:
            self._window = deque(self._window, maxlen=target_frames)
        if int(self._intent_window.maxlen or 0) < target_frames:
            self._intent_window = deque(self._intent_window, maxlen=target_frames)
        self._classifier_two_hands = self._is_two_hand_feature_dim(self._raw_feature_dim)
        if self._uses_global_dynamic_motion():
            self._dynamic_segmenter = self._create_dynamic_segmenter(
                target_frames=max(2, int(self._window.maxlen or window)),
            )
        self._detector_two_hands = True
        self._gesture_signatures = self._load_gesture_signatures(gesture_signatures_path)
        self._gesture_rejection = self._load_gesture_rejection(gesture_rejection_path)
        if self._uses_global_dynamic_motion():
            self._static_rejection_verifiers = {}
        else:
            self._static_rejection_verifiers = self._load_static_rejection_verifiers(
                static_rejection_verifier_path
            )
        self._dynamic_prototypes = self._load_dynamic_prototypes(
            dynamic_prototypes_path
        )
        try:
            self._gesture_taxonomy = load_gesture_taxonomy()
        except Exception:
            self._gesture_taxonomy = None

        if initialize_detector:
            try:
                task_path = str(resolve_hand_landmarker_task_path())
                self._detector = HandLandmarkerVideo(
                    num_hands=2,
                    task_path=task_path,
                )
            except Exception as e:
                self._init_error = str(e)
                self._detector = None

    def _ensure_dynamic_sequence_feature_mode(self, model_path: Path) -> None:
        model_name = str(getattr(model_path, "name", "") or "").lower()
        if not model_name.startswith("dynamic_sequence"):
            return
        feature_dim = int(getattr(self, "_feature_dim", 0) or 0)
        if feature_dim >= DYNAMIC_SEQUENCE_LONG_TARGET_FRAMES * 42:
            expected_mode = FEATURE_DYNAMIC_SEQUENCE_72
        elif feature_dim >= DYNAMIC_SEQUENCE_TARGET_FRAMES * 42:
            expected_mode = FEATURE_DYNAMIC_SEQUENCE
        else:
            return
        if str(getattr(self, "_feature_mode", "") or "") == expected_mode:
            return
        print(
            "[w] dynamic sequence artifact loaded with non-sequence feature_mode: "
            f"{self._feature_mode!r} -> {expected_mode}",
            flush=True,
        )
        self._feature_mode = expected_mode

    def _normalize_static_rejection_method(self, method: str) -> str:
        clean = str(method or "").strip().lower()
        return clean if clean in STATIC_REJECTION_METHODS else DEFAULT_STATIC_REJECTION_METHOD

    def _dynamic_sequence_target_frames(self) -> int:
        mode = str(
            getattr(self, "_feature_mode", FEATURE_STATIC_MEAN) or FEATURE_STATIC_MEAN
        )
        if mode == FEATURE_DYNAMIC_SEQUENCE_72:
            return DYNAMIC_SEQUENCE_LONG_TARGET_FRAMES
        return DYNAMIC_SEQUENCE_TARGET_FRAMES

    @property
    def static_rejection_method(self) -> str:
        return self._static_rejection_method

    def set_static_rejection_method(self, method: str) -> None:
        target = self._normalize_static_rejection_method(method)
        if target == self._static_rejection_method:
            return
        self._static_rejection_method = target
        self._last_static_decision = {}
        self._window.clear()
        self._finger_count_window.clear()

    def _infer_raw_feature_dim(self) -> int:
        return max(
            1,
            int(
                infer_raw_dim_from_feature_size(
                    str(self._feature_mode or FEATURE_STATIC_MEAN),
                    int(self._feature_dim),
                )
            ),
        )

    def _is_two_hand_feature_dim(self, raw_feature_dim: int) -> bool:
        if raw_feature_dim in {84, 88}:
            return True
        if raw_feature_dim % 2 != 0:
            return False
        per_hand_dim = raw_feature_dim // 2
        return per_hand_dim in {42, 44}

    def _hand_frame_feature(
        self,
        hand: DetectedHand,
        normalized: np.ndarray,
        *,
        target_dim: int,
    ) -> np.ndarray:
        pose = normalized.reshape(-1).astype(np.float32, copy=False)
        target = int(target_dim)
        if target <= pose.shape[0]:
            return pose[:target]

        pts = np.asarray(hand.landmarks, dtype=np.float32)
        if pts.shape == (21, 2):
            wrist = pts[0]
            center = pts.mean(axis=0)
            size = pts.max(axis=0) - pts.min(axis=0)
            extras = np.concatenate([wrist, center, size], axis=0)
        else:
            extras = np.zeros(0, dtype=np.float32)

        missing = target - pose.shape[0]
        if extras.shape[0] < missing:
            extras = np.concatenate(
                [extras, np.zeros(missing - extras.shape[0], dtype=np.float32)],
                axis=0,
            )
        return np.concatenate([pose, extras[:missing]], axis=0).astype(
            np.float32,
            copy=False,
        )

    def _build_model_feature(self) -> np.ndarray:
        sequence = np.stack(tuple(self._window), axis=0)
        raw_dim = int(getattr(self, "_raw_feature_dim", self._feature_dim))
        mode = str(
            getattr(self, "_feature_mode", FEATURE_STATIC_MEAN) or FEATURE_STATIC_MEAN
        )
        try:
            feat = build_feature_vector(sequence, mode=mode, target_dim=raw_dim)
        except ValueError:
            feat = build_feature_vector(
                sequence,
                mode=FEATURE_STATIC_MEAN,
                target_dim=raw_dim,
            )
        if feat.shape[0] != self._feature_dim:
            if feat.shape[0] > self._feature_dim:
                feat = feat[: self._feature_dim]
            else:
                pad = np.zeros(self._feature_dim - feat.shape[0], dtype=feat.dtype)
                feat = np.concatenate([feat, pad], axis=0)
        return feat.astype(np.float32, copy=False)

    def _uses_temporal_features(self) -> bool:
        mode = getattr(self, "_feature_mode", FEATURE_STATIC_MEAN)
        return mode in {
            FEATURE_DYNAMIC_SEQUENCE,
            FEATURE_DYNAMIC_SEQUENCE_72,
            FEATURE_DYNAMIC_STATS,
            FEATURE_HYBRID_STATS,
        }

    def _window_ready_for_prediction(self) -> bool:
        if not self._uses_temporal_features():
            return len(self._window) > 0
        target = int(self._window.maxlen or 1)
        return len(self._window) >= target

    def _uses_global_dynamic_motion(self) -> bool:
        return self._uses_temporal_features() and int(self._raw_feature_dim) >= 44

    def _dynamic_motion_gate(self) -> tuple[bool, dict[str, float]]:
        if not self._uses_global_dynamic_motion():
            return True, {}
        if not self._window_ready_for_prediction():
            return False, {}

        sequence = np.stack(tuple(self._window), axis=0)
        dx, dy, abs_dx, abs_dy, path_length, direction_cos, direction_sin = [
            float(value)
            for value in trajectory_features(
                sequence,
                target_dim=int(self._raw_feature_dim),
            )
        ]
        displacement = float(np.hypot(dx, dy))
        motion = {
            "dx": dx,
            "dy": dy,
            "abs_dx": abs_dx,
            "abs_dy": abs_dy,
            "path_length": path_length,
            "displacement": displacement,
            "direction_cos": direction_cos,
            "direction_sin": direction_sin,
        }
        if path_length < DYNAMIC_GATE_MIN_PATH_LENGTH:
            return False, motion
        if displacement < DYNAMIC_GATE_MIN_DISPLACEMENT:
            return False, motion
        return True, motion

    def _dynamic_temporal_state(self) -> dict[str, Any]:
        if not self._uses_temporal_features():
            return {}

        frames = len(self._window)
        required_frames = int(self._window.maxlen or 1)
        state: dict[str, Any] = {
            "enabled": True,
            "phase": "warming_up",
            "frames": frames,
            "required_frames": required_frames,
        }
        if not self._uses_global_dynamic_motion() or frames < 2:
            if frames >= required_frames:
                state["phase"] = "idle"
            return state

        sequence = np.stack(tuple(self._window), axis=0)
        dx, dy, abs_dx, abs_dy, path_length, direction_cos, direction_sin = [
            float(value)
            for value in trajectory_features(
                sequence,
                target_dim=int(self._raw_feature_dim),
            )
        ]
        displacement = float(np.hypot(dx, dy))
        state.update(
            {
                "dx": dx,
                "dy": dy,
                "abs_dx": abs_dx,
                "abs_dy": abs_dy,
                "path_length": path_length,
                "displacement": displacement,
                "direction_cos": direction_cos,
                "direction_sin": direction_sin,
            }
        )
        if (
            path_length >= DYNAMIC_INTENT_MIN_PATH_LENGTH
            and displacement >= DYNAMIC_INTENT_MIN_DISPLACEMENT
        ):
            state["phase"] = "active"
        elif frames >= required_frames:
            state["phase"] = "idle"
        return state

    def _dynamic_label_matches_motion(
        self,
        label: str,
        motion: dict[str, float],
    ) -> bool:
        if not motion:
            return True
        clean = str(label or "").strip().lower()
        dx = float(motion.get("dx") or 0.0)
        dy = float(motion.get("dy") or 0.0)
        horizontal = abs(dx)
        vertical = abs(dy)
        wants_left = "left" in clean
        wants_right = "right" in clean
        wants_up = "up" in clean
        wants_down = "down" in clean
        wants_horizontal = wants_left or wants_right
        wants_vertical = wants_up or wants_down
        horizontal_ok = (
            (wants_left and dx <= -DYNAMIC_GATE_DIRECTION_THRESHOLD)
            or (wants_right and dx >= DYNAMIC_GATE_DIRECTION_THRESHOLD)
        )
        vertical_ok = (
            (wants_up and dy <= -DYNAMIC_GATE_DIRECTION_THRESHOLD)
            or (wants_down and dy >= DYNAMIC_GATE_DIRECTION_THRESHOLD)
        )

        if wants_horizontal and wants_vertical:
            axis_ratio = max(horizontal, vertical) / max(min(horizontal, vertical), 1e-6)
            return (
                horizontal_ok
                and vertical_ok
                and axis_ratio <= DYNAMIC_COMPOUND_DIRECTION_MAX_AXIS_RATIO
            )
        if "left" in clean:
            return (
                horizontal_ok
                and horizontal >= vertical * DYNAMIC_DIRECTION_DOMINANCE_RATIO
            )
        if "right" in clean:
            return (
                horizontal_ok
                and horizontal >= vertical * DYNAMIC_DIRECTION_DOMINANCE_RATIO
            )
        if "up" in clean:
            return (
                vertical_ok
                and vertical >= horizontal * DYNAMIC_DIRECTION_DOMINANCE_RATIO
            )
        if "down" in clean:
            return (
                vertical_ok
                and vertical >= horizontal * DYNAMIC_DIRECTION_DOMINANCE_RATIO
            )
        return True

    def _dynamic_prediction(
        self,
        model_feat: np.ndarray,
        motion: dict[str, float],
        sequence: np.ndarray | None = None,
    ) -> tuple[str, float]:
        self._last_dynamic_decision = {}
        motion_decision = classify_swipe_direction(
            motion,
            self._classes,
            min_path_length=DYNAMIC_GATE_MIN_PATH_LENGTH,
            min_displacement=DYNAMIC_GATE_MIN_DISPLACEMENT,
            min_axis_ratio=DYNAMIC_DIRECTION_DOMINANCE_RATIO,
        )
        model_label = ""
        model_confidence = 0.0
        probabilities = np.asarray([], dtype=float)
        estimator_classes: list[Any] = []
        if self._clf is not None:
            try:
                predicted = self._clf.predict(model_feat)
                if len(predicted) > 0:
                    model_label = self._label_from_estimator_class(predicted[0])
            except Exception:
                model_label = ""
            try:
                probabilities = np.asarray(
                    self._clf.predict_proba(model_feat)[0],
                    dtype=float,
                )
                estimator_classes = list(
                    getattr(self._clf, "classes_", range(len(probabilities)))
                )
            except Exception:
                probabilities = np.asarray([], dtype=float)
                estimator_classes = []

        ranked: list[tuple[float, str]] = []
        for column, probability in enumerate(probabilities):
            raw_class = (
                estimator_classes[column]
                if column < len(estimator_classes)
                else column
            )
            label = self._label_from_estimator_class(raw_class)
            ranked.append((float(probability), label))
            if label == model_label:
                model_confidence = max(model_confidence, float(probability))

        negative_label, negative_confidence = self._best_negative_prediction(ranked)
        prototype_decision = self._dynamic_prototype_decision(sequence)
        prototype_fields = self._dynamic_prototype_fields(prototype_decision)
        if prototype_decision and not bool(prototype_decision.get("accepted")):
            (
                model_override_label,
                model_override_confidence,
                model_override_margin,
            ) = self._best_model_prototype_override(
                ranked,
                prototype_decision,
                motion_decision_label=motion_decision.label
                if motion_decision.accepted
                else "",
            )
            if (
                motion_decision.accepted
                and float(motion_decision.confidence) >= DYNAMIC_PROTOTYPE_OVERRIDE_MIN_CONFIDENCE
                and negative_confidence < DYNAMIC_NEGATIVE_REJECT_THRESHOLD
                and self._dynamic_motion_can_override_prototype_reject(
                    motion_decision.label,
                    prototype_decision,
                )
            ):
                self._last_dynamic_decision = {
                    "source": "motion_over_prototype_reject",
                    "motion_label": motion_decision.label,
                    "motion_confidence": float(motion_decision.confidence),
                    "model_label": model_label,
                    "model_confidence": model_confidence,
                    "negative_label": negative_label,
                    "negative_confidence": negative_confidence,
                    "negative_threshold": DYNAMIC_NEGATIVE_REJECT_THRESHOLD,
                    **prototype_fields,
                    **motion_decision.as_dict(),
                }
                return motion_decision.label, float(motion_decision.confidence)
            if (
                model_override_label
                and negative_confidence < DYNAMIC_NEGATIVE_REJECT_THRESHOLD
            ):
                self._last_dynamic_decision = {
                    "source": "model_over_prototype_reject",
                    "motion_label": motion_decision.label
                    if motion_decision.accepted
                    else "",
                    "motion_confidence": float(motion_decision.confidence)
                    if motion_decision.accepted
                    else 0.0,
                    "model_label": model_label,
                    "model_confidence": model_confidence,
                    "negative_label": negative_label,
                    "negative_confidence": negative_confidence,
                    "negative_threshold": DYNAMIC_NEGATIVE_REJECT_THRESHOLD,
                    "model_override_label": model_override_label,
                    "model_override_confidence": model_override_confidence,
                    "model_override_margin": model_override_margin,
                    **prototype_fields,
                    **motion_decision.as_dict(),
                }
                return model_override_label, model_override_confidence
            self._last_dynamic_decision = {
                "source": "prototype_rejected",
                "motion_label": motion_decision.label if motion_decision.accepted else "",
                "motion_confidence": float(motion_decision.confidence)
                if motion_decision.accepted
                else 0.0,
                "model_label": model_label,
                "model_confidence": model_confidence,
                "negative_label": negative_label,
                "negative_confidence": negative_confidence,
                "negative_threshold": DYNAMIC_NEGATIVE_REJECT_THRESHOLD,
                **prototype_fields,
                **motion_decision.as_dict(),
            }
            return "", 0.0

        prototype_label = str(prototype_decision.get("label") or "") if prototype_decision else ""
        prototype_confidence = (
            float(prototype_decision.get("confidence") or 0.0)
            if prototype_decision
            else 0.0
        )
        compatible_label = ""
        compatible_confidence = 0.0
        for probability, label in sorted(ranked, reverse=True):
            if self._is_negative_label(label):
                continue
            if self._dynamic_label_matches_motion(label, motion):
                compatible_label = label
                compatible_confidence = float(probability)
                break
        (
            complex_model_label,
            complex_model_confidence,
            complex_model_margin,
        ) = self._best_complex_model_prediction(ranked)

        if motion_decision.accepted:
            motion_confidence = float(motion_decision.confidence)
            model_for_motion = max(
                (probability for probability, label in ranked if label == motion_decision.label),
                default=0.0,
            )
            if (
                prototype_label
                and prototype_label != motion_decision.label
                and self._is_complex_dynamic_label(prototype_label)
                and negative_confidence < DYNAMIC_NEGATIVE_REJECT_THRESHOLD
            ):
                model_for_prototype = max(
                    (probability for probability, label in ranked if label == prototype_label),
                    default=0.0,
                )
                self._last_dynamic_decision = {
                    "source": "complex_prototype_over_motion",
                    "motion_label": motion_decision.label,
                    "motion_confidence": motion_confidence,
                    "model_label": model_label,
                    "model_confidence": model_confidence,
                    "negative_label": negative_label,
                    "negative_confidence": negative_confidence,
                    "negative_threshold": DYNAMIC_NEGATIVE_REJECT_THRESHOLD,
                    "compatible_model_label": compatible_label,
                    "compatible_model_confidence": compatible_confidence,
                    "model_confidence_for_motion": float(model_for_motion),
                    "model_confidence_for_prototype": float(model_for_prototype),
                    **prototype_fields,
                    **motion_decision.as_dict(),
                }
                return prototype_label, max(
                    prototype_confidence,
                    float(model_for_prototype),
                )
            if (
                complex_model_label
                and complex_model_label != motion_decision.label
                and negative_confidence < DYNAMIC_NEGATIVE_REJECT_THRESHOLD
                and self._complex_model_prediction_confident(
                    complex_model_confidence,
                    complex_model_margin,
                )
            ):
                self._last_dynamic_decision = {
                    "source": "complex_model_over_motion",
                    "motion_label": motion_decision.label,
                    "motion_confidence": motion_confidence,
                    "model_label": model_label,
                    "model_confidence": model_confidence,
                    "negative_label": negative_label,
                    "negative_confidence": negative_confidence,
                    "negative_threshold": DYNAMIC_NEGATIVE_REJECT_THRESHOLD,
                    "compatible_model_label": compatible_label,
                    "compatible_model_confidence": compatible_confidence,
                    "complex_model_label": complex_model_label,
                    "complex_model_confidence": complex_model_confidence,
                    "complex_model_margin": complex_model_margin,
                    "model_confidence_for_motion": float(model_for_motion),
                    **prototype_fields,
                    **motion_decision.as_dict(),
                }
                return complex_model_label, complex_model_confidence
            if (
                complex_model_label
                and complex_model_label != motion_decision.label
                and negative_confidence < DYNAMIC_NEGATIVE_REJECT_THRESHOLD
                and self._dynamic_label_matches_motion(complex_model_label, motion)
                and self._complex_model_prediction_near_top(
                    complex_model_confidence,
                    complex_model_margin,
                )
            ):
                self._last_dynamic_decision = {
                    "source": "complex_model_near_top_over_motion",
                    "motion_label": motion_decision.label,
                    "motion_confidence": motion_confidence,
                    "model_label": model_label,
                    "model_confidence": model_confidence,
                    "negative_label": negative_label,
                    "negative_confidence": negative_confidence,
                    "negative_threshold": DYNAMIC_NEGATIVE_REJECT_THRESHOLD,
                    "compatible_model_label": compatible_label,
                    "compatible_model_confidence": compatible_confidence,
                    "complex_model_label": complex_model_label,
                    "complex_model_confidence": complex_model_confidence,
                    "complex_model_margin": complex_model_margin,
                    "model_confidence_for_motion": float(model_for_motion),
                    **prototype_fields,
                    **motion_decision.as_dict(),
                }
                return complex_model_label, complex_model_confidence
            if prototype_label and prototype_label != motion_decision.label:
                self._last_dynamic_decision = {
                    "source": "prototype_motion_conflict",
                    "motion_label": motion_decision.label,
                    "motion_confidence": motion_confidence,
                    "model_label": model_label,
                    "model_confidence": model_confidence,
                    "negative_label": negative_label,
                    "negative_confidence": negative_confidence,
                    "negative_threshold": DYNAMIC_NEGATIVE_REJECT_THRESHOLD,
                    "compatible_model_label": compatible_label,
                    "compatible_model_confidence": compatible_confidence,
                    "model_confidence_for_motion": float(model_for_motion),
                    **prototype_fields,
                    **motion_decision.as_dict(),
                }
                return "", 0.0
            source = "motion_first"
            motion_supported_by_model = (
                model_label == motion_decision.label
                or compatible_label == motion_decision.label
            )
            motion_supported_by_prototype = prototype_label == motion_decision.label
            if motion_supported_by_model:
                source = "motion_and_model_agree"
            if motion_supported_by_prototype:
                source = "motion_and_prototype_agree"
            if negative_confidence >= DYNAMIC_NEGATIVE_REJECT_THRESHOLD:
                self._last_dynamic_decision = {
                    "source": "negative_rejected",
                    "motion_label": motion_decision.label,
                    "motion_confidence": motion_confidence,
                    "model_label": model_label,
                    "model_confidence": model_confidence,
                    "negative_label": negative_label,
                    "negative_confidence": negative_confidence,
                    "negative_threshold": DYNAMIC_NEGATIVE_REJECT_THRESHOLD,
                    "compatible_model_label": compatible_label,
                    "compatible_model_confidence": compatible_confidence,
                    "model_confidence_for_motion": float(model_for_motion),
                    **prototype_fields,
                    **motion_decision.as_dict(),
                }
                return "", 0.0
            if (
                self._has_custom_dynamic_labels()
                and not motion_supported_by_model
                and not motion_supported_by_prototype
            ):
                self._last_dynamic_decision = {
                    "source": "motion_fallback_suppressed_for_custom_labels",
                    "motion_label": motion_decision.label,
                    "motion_confidence": motion_confidence,
                    "model_label": model_label,
                    "model_confidence": model_confidence,
                    "negative_label": negative_label,
                    "negative_confidence": negative_confidence,
                    "negative_threshold": DYNAMIC_NEGATIVE_REJECT_THRESHOLD,
                    "compatible_model_label": compatible_label,
                    "compatible_model_confidence": compatible_confidence,
                    "model_confidence_for_motion": float(model_for_motion),
                    **prototype_fields,
                    **motion_decision.as_dict(),
                }
                return "", 0.0
            self._last_dynamic_decision = {
                "source": source,
                "motion_label": motion_decision.label,
                "motion_confidence": motion_confidence,
                "model_label": model_label,
                "model_confidence": model_confidence,
                "negative_label": negative_label,
                "negative_confidence": negative_confidence,
                "negative_threshold": DYNAMIC_NEGATIVE_REJECT_THRESHOLD,
                "compatible_model_label": compatible_label,
                "compatible_model_confidence": compatible_confidence,
                "model_confidence_for_motion": float(model_for_motion),
                **prototype_fields,
                **motion_decision.as_dict(),
            }
            return motion_decision.label, max(
                motion_confidence,
                float(model_for_motion),
                prototype_confidence,
            )

        if prototype_label:
            model_for_prototype = max(
                (probability for probability, label in ranked if label == prototype_label),
                default=0.0,
            )
            self._last_dynamic_decision = {
                "source": "prototype_accepted",
                "motion_label": "",
                "motion_confidence": 0.0,
                "model_label": model_label,
                "model_confidence": model_confidence,
                "negative_label": negative_label,
                "negative_confidence": negative_confidence,
                "negative_threshold": DYNAMIC_NEGATIVE_REJECT_THRESHOLD,
                "compatible_model_label": compatible_label,
                "compatible_model_confidence": compatible_confidence,
                "model_confidence_for_prototype": float(model_for_prototype),
                **prototype_fields,
                **motion_decision.as_dict(),
            }
            return prototype_label, max(prototype_confidence, float(model_for_prototype))

        if compatible_label:
            self._last_dynamic_decision = {
                "source": "model_fallback",
                "motion_label": "",
                "motion_confidence": 0.0,
                "model_label": model_label,
                "model_confidence": model_confidence,
                "negative_label": negative_label,
                "negative_confidence": negative_confidence,
                "negative_threshold": DYNAMIC_NEGATIVE_REJECT_THRESHOLD,
                "compatible_model_label": compatible_label,
                "compatible_model_confidence": compatible_confidence,
                **prototype_fields,
                **motion_decision.as_dict(),
            }
            return compatible_label, compatible_confidence

        if negative_confidence >= DYNAMIC_NEGATIVE_REJECT_THRESHOLD:
            self._last_dynamic_decision = {
                "source": "negative_rejected",
                "motion_label": "",
                "motion_confidence": 0.0,
                "model_label": model_label,
                "model_confidence": model_confidence,
                "negative_label": negative_label,
                "negative_confidence": negative_confidence,
                "negative_threshold": DYNAMIC_NEGATIVE_REJECT_THRESHOLD,
                **prototype_fields,
                **motion_decision.as_dict(),
            }
            return "", 0.0

        self._last_dynamic_decision = {
            "source": "rejected",
            "motion_label": "",
            "motion_confidence": 0.0,
            "model_label": model_label,
            "model_confidence": model_confidence,
            "negative_label": negative_label,
            "negative_confidence": negative_confidence,
            "negative_threshold": DYNAMIC_NEGATIVE_REJECT_THRESHOLD,
            **prototype_fields,
            **motion_decision.as_dict(),
        }
        return "", 0.0

    def _best_complex_model_prediction(
        self,
        ranked: list[tuple[float, str]],
    ) -> tuple[str, float, float]:
        ordered = sorted(ranked, key=lambda item: item[0], reverse=True)
        if not ordered:
            return "", 0.0, 0.0

        complex_candidates = [
            (float(confidence), str(label))
            for confidence, label in ordered
            if self._is_complex_dynamic_label(label)
        ]
        if not complex_candidates:
            return "", 0.0, 0.0

        top_confidence, top_label = complex_candidates[0]
        competing_confidence = max(
            (
                float(confidence)
                for confidence, label in ordered
                if str(label) != str(top_label)
            ),
            default=0.0,
        )
        return (
            str(top_label),
            float(top_confidence),
            float(top_confidence - competing_confidence),
        )

    def _best_model_prototype_override(
        self,
        ranked: list[tuple[float, str]],
        prototype_decision: dict[str, Any],
        *,
        motion_decision_label: str = "",
    ) -> tuple[str, float, float]:
        reason = str(prototype_decision.get("reason") or "").strip().lower()
        if reason != "far_from_prototype":
            return "", 0.0, 0.0

        nearest_type = str(prototype_decision.get("nearest_type") or "").strip().lower()
        if nearest_type and nearest_type != "positive":
            return "", 0.0, 0.0

        ordered = sorted(ranked, key=lambda item: item[0], reverse=True)
        if not ordered:
            return "", 0.0, 0.0

        confidence, label = ordered[0]
        clean = str(label or "").strip().lower()
        if not clean or self._is_negative_label(clean):
            return "", 0.0, 0.0

        motion_label = str(motion_decision_label or "").strip().lower()
        if motion_label and motion_label != clean:
            return "", 0.0, 0.0

        nearest_label = str(
            prototype_decision.get("nearest_label")
            or prototype_decision.get("label")
            or ""
        ).strip().lower()
        positive_labels = {
            str(item or "").strip().lower()
            for item in (self._dynamic_prototypes or {}).get("positive_labels", [])
            if str(item or "").strip()
        }
        if nearest_label and nearest_label != clean:
            return "", 0.0, 0.0
        if positive_labels and clean not in positive_labels:
            return "", 0.0, 0.0

        second_confidence = float(ordered[1][0]) if len(ordered) > 1 else 0.0
        margin = float(confidence - second_confidence)
        if confidence < DYNAMIC_MODEL_PROTOTYPE_OVERRIDE_MIN_CONFIDENCE:
            return "", 0.0, 0.0
        if margin < DYNAMIC_MODEL_PROTOTYPE_OVERRIDE_MIN_MARGIN:
            return "", 0.0, 0.0
        return str(label), float(confidence), margin

    @staticmethod
    def _complex_model_prediction_confident(
        confidence: float,
        margin: float,
    ) -> bool:
        return (
            float(confidence) >= DYNAMIC_COMPLEX_MODEL_MIN_CONFIDENCE
            and float(margin) >= DYNAMIC_COMPLEX_MODEL_MIN_MARGIN
        )

    @staticmethod
    def _complex_model_prediction_near_top(
        confidence: float,
        margin: float,
    ) -> bool:
        return (
            float(confidence) >= DYNAMIC_COMPLEX_MODEL_NEAR_TOP_MIN_CONFIDENCE
            and float(margin) >= -DYNAMIC_COMPLEX_MODEL_NEAR_TOP_MAX_GAP
        )

    def _is_complex_dynamic_label(self, label: str) -> bool:
        clean = str(label or "").strip().lower()
        if not clean:
            return False
        if clean.startswith("swipe_") or self._is_negative_label(clean):
            return False
        return True

    def _has_custom_dynamic_labels(self) -> bool:
        return any(self._is_complex_dynamic_label(label) for label in self._classes)

    def _dynamic_motion_can_override_prototype_reject(
        self,
        label: str,
        prototype_decision: dict[str, Any],
    ) -> bool:
        clean = str(label or "").strip().lower()
        if not clean or not prototype_decision:
            return False

        positive_labels = {
            str(item or "").strip().lower()
            for item in (self._dynamic_prototypes or {}).get("positive_labels", [])
            if str(item or "").strip()
        }
        if not positive_labels:
            positive_labels = {
                str(item.get("label") or "").strip().lower()
                for item in (self._dynamic_prototypes or {}).get("prototypes", [])
                if isinstance(item, dict)
                and not bool(item.get("is_negative"))
                and str(item.get("label") or "").strip()
            }
        if clean not in positive_labels:
            return False

        reason = str(prototype_decision.get("reason") or "").strip().lower()
        return reason == "far_from_prototype"

    def _dynamic_prototype_decision(
        self,
        sequence: np.ndarray | None,
    ) -> dict[str, Any]:
        payload = getattr(self, "_dynamic_prototypes", {}) or {}
        if sequence is None or not payload:
            return {}
        try:
            decision = predict_dynamic_prototype(payload, sequence)
        except Exception as exc:
            print(f"[w] dynamic prototype decision failed: {exc}", flush=True)
            return {}
        return decision if isinstance(decision, dict) else {}

    @staticmethod
    def _dynamic_prototype_fields(decision: dict[str, Any]) -> dict[str, Any]:
        if not decision:
            return {}
        return {
            "prototype_method": str(decision.get("method") or ""),
            "prototype_label": str(
                decision.get("label")
                or decision.get("nearest_label")
                or ""
            ),
            "prototype_nearest_type": str(decision.get("nearest_type") or ""),
            "prototype_confidence": float(decision.get("confidence") or 0.0),
            "prototype_distance": float(decision.get("distance") or 0.0),
            "prototype_threshold": float(decision.get("threshold") or 0.0),
            "prototype_margin": float(decision.get("margin") or 0.0),
            "prototype_reason": str(decision.get("reason") or ""),
        }

    def _label_from_estimator_class(self, raw_class: Any) -> str:
        try:
            class_index = int(raw_class)
        except (TypeError, ValueError):
            return str(raw_class)
        return (
            self._classes[class_index]
            if 0 <= class_index < len(self._classes)
            else str(raw_class)
        )

    def _best_negative_prediction(
        self,
        ranked: list[tuple[float, str]],
    ) -> tuple[str, float]:
        best_label = ""
        best_probability = 0.0
        for probability, label in ranked:
            if probability > best_probability and self._is_negative_label(label):
                best_label = label
                best_probability = float(probability)
        return best_label, best_probability

    def _static_prediction(self, model_feat: np.ndarray) -> tuple[str, float]:
        self._last_static_decision = {}
        if self._clf is None:
            return "", 0.0

        model_label = ""
        try:
            predicted = self._clf.predict(model_feat)
            if len(predicted) > 0:
                model_label = self._label_from_estimator_class(predicted[0])
        except Exception:
            raise

        ranked: list[tuple[float, str]] = []
        try:
            probabilities = np.asarray(
                self._clf.predict_proba(model_feat)[0],
                dtype=float,
            )
            estimator_classes = list(
                getattr(self._clf, "classes_", range(len(probabilities)))
            )
            for column, probability in enumerate(probabilities):
                raw_class = (
                    estimator_classes[column]
                    if column < len(estimator_classes)
                    else column
                )
                ranked.append(
                    (float(probability), self._label_from_estimator_class(raw_class))
                )
        except Exception:
            ranked = []

        ranked = sorted(ranked, key=lambda item: item[0], reverse=True)
        if ranked:
            confidence, label = ranked[0]
        else:
            label = model_label
            confidence = 1.0 if model_label else 0.0

        top2_label = ranked[1][1] if len(ranked) > 1 else ""
        top2_confidence = float(ranked[1][0]) if len(ranked) > 1 else 0.0
        margin = float(confidence - top2_confidence)
        rejection = getattr(self, "_gesture_rejection", {}) or {}
        thresholds = rejection.get("thresholds") if isinstance(rejection, dict) else {}
        if not isinstance(thresholds, dict):
            thresholds = {}
        negative_threshold = float(thresholds.get("negative_confidence", 0.65))
        min_margin = float(thresholds.get("min_top1_top2_margin", 0.0))
        distance_multiplier = float(thresholds.get("distance_multiplier", 0.0))

        negative_label, negative_confidence = self._best_negative_prediction(ranked)
        method = self._normalize_static_rejection_method(
            getattr(self, "_static_rejection_method", DEFAULT_STATIC_REJECTION_METHOD)
        )
        decision: dict[str, Any] = {
            "source": "accepted",
            "rejection_reason": "",
            "rejection_method": method,
            "model_label": str(label or model_label or ""),
            "model_confidence": float(confidence),
            "top2_label": top2_label,
            "top2_confidence": top2_confidence,
            "margin": margin,
            "min_margin": min_margin,
            "negative_label": negative_label,
            "negative_confidence": float(negative_confidence),
            "negative_threshold": negative_threshold,
        }

        if method == STATIC_REJECTION_NEGATIVE_CLASSES:
            if self._is_negative_label(label):
                return self._reject_static_prediction(
                    decision,
                    source="negative_rejected",
                    reason="negative_class",
                )
            self._last_static_decision = decision
            return str(label), float(confidence)

        if method == STATIC_REJECTION_CONFIDENCE_THRESHOLD:
            positive_threshold = float(
                thresholds.get(
                    "positive_confidence",
                    thresholds.get("negative_confidence", 0.65),
                )
            )
            decision["positive_threshold"] = positive_threshold
            if self._is_negative_label(label):
                return self._reject_static_prediction(
                    decision,
                    source="negative_rejected",
                    reason="negative_class",
                )
            if confidence < positive_threshold:
                return self._reject_static_prediction(
                    decision,
                    source="confidence_rejected",
                    reason="low_confidence",
                )
            self._last_static_decision = decision
            return str(label), float(confidence)

        if method in {
            STATIC_REJECTION_ONE_VS_REST_LOGREG,
            STATIC_REJECTION_ONE_CLASS_SVM,
            STATIC_REJECTION_ISOLATION_FOREST,
            STATIC_REJECTION_LOCAL_OUTLIER_FACTOR,
            STATIC_REJECTION_METRIC_NCA_CENTROID,
            STATIC_REJECTION_MLP_NEGATIVE_CLASSES,
        }:
            if self._is_negative_label(label):
                return self._reject_static_prediction(
                    decision,
                    source="negative_rejected",
                    reason="negative_class",
                )
            return self._apply_static_verifier_method(
                method,
                str(label),
                float(confidence),
                model_feat.reshape(-1),
                decision,
            )

        if self._is_negative_label(label) or negative_confidence >= negative_threshold:
            return self._reject_static_prediction(
                decision,
                source="negative_rejected",
                reason="negative_class",
            )

        if len(ranked) > 1 and min_margin > 0.0 and margin < min_margin:
            return self._reject_static_prediction(
                decision,
                source="margin_rejected",
                reason="low_margin",
            )

        prototype = self._static_prototype_distance(
            label,
            model_feat.reshape(-1),
            distance_multiplier=distance_multiplier,
        )
        decision.update(prototype)
        if prototype.get("prototype_rejected"):
            return self._reject_static_prediction(
                decision,
                source="prototype_rejected",
                reason="far_from_prototype",
            )

        self._last_static_decision = decision
        return str(label), float(confidence)

    def _reject_static_prediction(
        self,
        decision: dict[str, Any],
        *,
        source: str,
        reason: str,
    ) -> tuple[str, float]:
        decision["source"] = str(source)
        decision["rejection_reason"] = str(reason)
        self._last_static_decision = decision
        return "", 0.0

    def _static_verifier_payload(self, method: str) -> dict[str, Any]:
        payload = getattr(self, "_static_rejection_verifiers", {}) or {}
        methods = payload.get("methods") if isinstance(payload, dict) else {}
        if not isinstance(methods, dict):
            return {}
        raw = methods.get(method)
        return raw if isinstance(raw, dict) else {}

    def _predict_positive_probability(self, model: Any, feature: np.ndarray) -> float:
        if not hasattr(model, "predict_proba"):
            prediction = model.predict(feature.reshape(1, -1))
            return 1.0 if int(prediction[0]) == 1 else 0.0
        probabilities = np.asarray(model.predict_proba(feature.reshape(1, -1))[0], dtype=float)
        classes = list(getattr(model, "classes_", range(len(probabilities))))
        positive_index = 1 if len(probabilities) > 1 else 0
        for index, raw_class in enumerate(classes):
            try:
                if int(raw_class) == 1:
                    positive_index = index
                    break
            except (TypeError, ValueError):
                if str(raw_class).lower() in {"1", "true", "positive"}:
                    positive_index = index
                    break
        return float(probabilities[positive_index])

    def _apply_static_verifier_method(
        self,
        method: str,
        label: str,
        confidence: float,
        feature: np.ndarray,
        decision: dict[str, Any],
    ) -> tuple[str, float]:
        payload = self._static_verifier_payload(method)
        decision["verifier_method"] = method
        if not payload:
            decision["source"] = "verifier_missing_accepted"
            decision["rejection_reason"] = ""
            self._last_static_decision = decision
            return str(label), float(confidence)

        if method == STATIC_REJECTION_ONE_VS_REST_LOGREG:
            verifiers = payload.get("verifiers")
            verifier = verifiers.get(label) if isinstance(verifiers, dict) else None
            if verifier is None:
                decision["source"] = "verifier_missing_accepted"
                self._last_static_decision = decision
                return str(label), float(confidence)
            probability = self._predict_positive_probability(verifier, feature)
            threshold = float(payload.get("threshold", 0.50))
            decision.update(
                {
                    "verifier_probability": probability,
                    "verifier_threshold": threshold,
                    "verifier_label": label,
                }
            )
            if probability < threshold:
                return self._reject_static_prediction(
                    decision,
                    source="verifier_rejected",
                    reason="one_vs_rest_low_probability",
                )
            self._last_static_decision = decision
            return str(label), float(confidence)

        if method in {
            STATIC_REJECTION_ONE_CLASS_SVM,
            STATIC_REJECTION_ISOLATION_FOREST,
            STATIC_REJECTION_LOCAL_OUTLIER_FACTOR,
        }:
            models = payload.get("models")
            verifier = models.get(label) if isinstance(models, dict) else None
            if verifier is None:
                decision["source"] = "verifier_missing_accepted"
                self._last_static_decision = decision
                return str(label), float(confidence)
            prediction = int(verifier.predict(feature.reshape(1, -1))[0])
            score = 0.0
            scorer = getattr(verifier, "decision_function", None)
            if callable(scorer):
                try:
                    score = float(np.asarray(scorer(feature.reshape(1, -1))).reshape(-1)[0])
                except Exception:
                    score = 0.0
            decision.update(
                {
                    "verifier_label": label,
                    "verifier_prediction": prediction,
                    "verifier_score": score,
                }
            )
            if prediction != 1:
                return self._reject_static_prediction(
                    decision,
                    source="outlier_rejected",
                    reason=f"{method}_outlier",
                )
            self._last_static_decision = decision
            return str(label), float(confidence)

        if method == STATIC_REJECTION_METRIC_NCA_CENTROID:
            transformer = payload.get("transformer")
            prototypes = payload.get("prototypes")
            prototype = prototypes.get(label) if isinstance(prototypes, dict) else None
            if transformer is None or not isinstance(prototype, dict):
                decision["source"] = "verifier_missing_accepted"
                self._last_static_decision = decision
                return str(label), float(confidence)
            embedded = np.asarray(
                transformer.transform(feature.reshape(1, -1))[0],
                dtype=np.float32,
            )
            centroid = np.asarray(prototype.get("centroid", []), dtype=np.float32)
            threshold = float(prototype.get("threshold", 0.0))
            distance = float(np.linalg.norm(embedded - centroid)) if centroid.size else 0.0
            decision.update(
                {
                    "verifier_label": label,
                    "verifier_distance": distance,
                    "verifier_threshold": threshold,
                }
            )
            if threshold > 0.0 and distance > threshold:
                return self._reject_static_prediction(
                    decision,
                    source="metric_distance_rejected",
                    reason="metric_centroid_distance",
                )
            self._last_static_decision = decision
            return str(label), float(confidence)

        if method == STATIC_REJECTION_MLP_NEGATIVE_CLASSES:
            model = payload.get("model")
            if model is None:
                decision["source"] = "verifier_missing_accepted"
                self._last_static_decision = decision
                return str(label), float(confidence)
            ranked = self._ranked_labels_from_model(model, feature)
            verifier_confidence, verifier_label = ranked[0] if ranked else (0.0, "")
            negative_label, negative_confidence = self._best_negative_prediction(ranked)
            threshold = float(payload.get("negative_threshold", 0.65))
            decision.update(
                {
                    "verifier_label": verifier_label,
                    "verifier_confidence": float(verifier_confidence),
                    "verifier_negative_label": negative_label,
                    "verifier_negative_confidence": float(negative_confidence),
                    "verifier_threshold": threshold,
                }
            )
            if self._is_negative_label(verifier_label) or negative_confidence >= threshold:
                return self._reject_static_prediction(
                    decision,
                    source="mlp_negative_rejected",
                    reason="mlp_negative_class",
                )
            self._last_static_decision = decision
            return str(label), float(confidence)

        self._last_static_decision = decision
        return str(label), float(confidence)

    def _ranked_labels_from_model(
        self,
        model: Any,
        feature: np.ndarray,
    ) -> list[tuple[float, str]]:
        if not hasattr(model, "predict_proba"):
            prediction = model.predict(feature.reshape(1, -1))
            label = self._label_from_estimator_class(prediction[0]) if len(prediction) else ""
            return [(1.0, label)] if label else []
        probabilities = np.asarray(model.predict_proba(feature.reshape(1, -1))[0], dtype=float)
        classes = list(getattr(model, "classes_", range(len(probabilities))))
        ranked: list[tuple[float, str]] = []
        for column, probability in enumerate(probabilities):
            raw_class = classes[column] if column < len(classes) else column
            ranked.append((float(probability), self._label_from_estimator_class(raw_class)))
        return sorted(ranked, key=lambda item: item[0], reverse=True)

    def _static_prototype_distance(
        self,
        label: str,
        feature: np.ndarray,
        *,
        distance_multiplier: float,
    ) -> dict[str, Any]:
        rejection = getattr(self, "_gesture_rejection", {}) or {}
        classes = rejection.get("classes") if isinstance(rejection, dict) else {}
        if not isinstance(classes, dict) or not label:
            return {}
        raw = classes.get(label) or classes.get(str(label).lower())
        if not isinstance(raw, dict):
            return {}
        centroid_raw = raw.get("centroid")
        if not isinstance(centroid_raw, list):
            return {}
        centroid = np.asarray(centroid_raw, dtype=np.float32)
        candidate = np.asarray(feature, dtype=np.float32).reshape(-1)
        if centroid.shape != candidate.shape:
            return {}
        radius = float(raw.get("prototype_radius") or 0.0)
        multiplier = float(distance_multiplier or 0.0)
        threshold = radius * multiplier if radius > 0.0 and multiplier > 0.0 else 0.0
        distance = float(np.linalg.norm(candidate - centroid))
        return {
            "prototype_distance": distance,
            "prototype_radius": radius,
            "prototype_threshold": threshold,
            "prototype_rejected": bool(threshold > 0.0 and distance > threshold),
        }

    def _is_negative_label(self, label: str) -> bool:
        clean = str(label or "").strip().lower()
        if not clean:
            return False
        taxonomy = getattr(self, "_gesture_taxonomy", None)
        if taxonomy is not None:
            try:
                return taxonomy.gesture_type_for_label(clean) == GESTURE_TYPE_NEGATIVE
            except Exception:
                pass
        return (
            clean.startswith("negative_")
            or clean.startswith("no_gesture")
            or clean.startswith("background_")
            or clean.startswith("random_")
            or clean.startswith("partial_")
            or clean.startswith("return_")
            or clean.startswith("wrong_axis_")
        )

    def _load_gesture_rejection(self, rejection_path: Path) -> dict[str, Any]:
        if not rejection_path.exists():
            return {}
        try:
            metadata = json.loads(rejection_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[w] gesture_rejection.json ignored: {exc}", flush=True)
            return {}
        if not isinstance(metadata, dict):
            return {}
        if not isinstance(metadata.get("classes"), dict):
            return {}
        return metadata

    def _load_static_rejection_verifiers(self, verifier_path: Path) -> dict[str, Any]:
        if not verifier_path.exists():
            return {}
        try:
            import warnings

            import joblib

            try:
                from sklearn.exceptions import InconsistentVersionWarning
            except ImportError:
                InconsistentVersionWarning = UserWarning  # type: ignore[misc,assignment]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InconsistentVersionWarning)
                payload = joblib.load(str(verifier_path))
        except Exception as exc:
            print(f"[w] static_rejection_verifiers.pkl ignored: {exc}", flush=True)
            return {}
        if not isinstance(payload, dict):
            return {}
        methods = payload.get("methods")
        if not isinstance(methods, dict):
            return {}
        return payload

    def _load_dynamic_prototypes(self, prototypes_path: Path) -> dict[str, Any]:
        if not prototypes_path.exists():
            return {}
        try:
            payload = load_dynamic_prototype_model(prototypes_path)
        except Exception as exc:
            print(f"[w] dynamic_prototypes.json ignored: {exc}", flush=True)
            return {}
        if not isinstance(payload, dict):
            return {}
        prototypes = payload.get("prototypes")
        if not isinstance(prototypes, list) or not prototypes:
            return {}
        return payload

    def reset_temporal_state(self) -> None:
        self._window.clear()
        self._ensure_intent_window().clear()
        self._finger_count_window.clear()
        segmenter = getattr(self, "_dynamic_segmenter", None)
        if segmenter is not None:
            segmenter.reset()
        self._pending_dynamic_prediction = None
        self._pending_dynamic_repeats = 0
        self._pending_dynamic_motion_scale = 0.0
        self._last_dynamic_decision = {}
        self._last_static_decision = {}

    def acknowledge_dynamic_event(self) -> None:
        """Clear emitted prediction while preserving return-motion cooldown."""
        self._window.clear()
        self._ensure_intent_window().clear()
        self._finger_count_window.clear()
        self._pending_dynamic_prediction = None
        self._pending_dynamic_repeats = 0
        self._pending_dynamic_motion_scale = 0.0
        self._last_dynamic_decision = {}
        self._last_static_decision = {}

    def _segmenter(self) -> DynamicMotionSegmenter:
        segmenter = getattr(self, "_dynamic_segmenter", None)
        if segmenter is None:
            segmenter = self._create_dynamic_segmenter(
                target_frames=int(self._window.maxlen or 36),
            )
            self._dynamic_segmenter = segmenter
        return segmenter

    @staticmethod
    def _create_dynamic_segmenter(*, target_frames: int) -> DynamicMotionSegmenter:
        return DynamicMotionSegmenter(
            target_frames=max(2, int(target_frames)),
            pre_roll_frames=DYNAMIC_SEGMENT_PRE_ROLL_FRAMES,
            onset_path=DYNAMIC_SEGMENT_ONSET_PATH,
            onset_displacement=DYNAMIC_SEGMENT_ONSET_DISPLACEMENT,
            min_active_frames=DYNAMIC_SEGMENT_MIN_ACTIVE_FRAMES,
            max_active_frames=max(60, int(target_frames)),
        )

    def _temporal_state_from_segment_update(
        self,
        update: Any,
        *,
        motion_scale: float,
    ) -> dict[str, Any]:
        return {
            "enabled": True,
            "phase": update.phase,
            "frames": update.frames,
            "required_frames": int(self._window.maxlen or 36),
            "motion_scale": float(motion_scale),
            "end_reason": str(getattr(update, "end_reason", "") or ""),
        }

    def _ensure_intent_window(self) -> Deque[np.ndarray]:
        window = getattr(self, "_intent_window", None)
        if window is None:
            maxlen = max(
                self._dynamic_sequence_target_frames(),
                int(getattr(getattr(self, "_window", None), "maxlen", 0) or 0),
            )
            window = deque(maxlen=maxlen)
            self._intent_window = window
        return window

    def _intent_feature_payload(
        self,
        sequence: np.ndarray | None = None,
    ) -> list[float]:
        try:
            if sequence is None:
                intent_window = self._ensure_intent_window()
                if not intent_window:
                    return []
                sequence = np.stack(tuple(intent_window), axis=0)
            feature = build_intent_feature_vector(
                sequence,
                target_dim=DEFAULT_INTENT_TARGET_DIM,
            )
        except Exception:
            return []
        return [float(value) for value in feature.astype(float).tolist()]

    def _classify_completed_dynamic_update(
        self,
        update: Any,
        landmarks_json: str,
        *,
        motion_scale: float,
    ) -> Dict[str, Any]:
        temporal_state = self._temporal_state_from_segment_update(
            update,
            motion_scale=motion_scale,
        )
        if update.completed_sequence is None:
            return {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": landmarks_json,
                "temporal": temporal_state,
                "intent_features": self._intent_feature_payload(),
            }

        self._window.clear()
        self._window.extend(update.completed_sequence)
        motion_ok, motion = self._dynamic_motion_gate()
        if not motion_ok or not self._classes:
            return {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": landmarks_json,
                "temporal": temporal_state,
                "intent_features": self._intent_feature_payload(
                    update.completed_sequence
                ),
            }

        try:
            model_feat = (
                self._build_model_feature().reshape(1, -1)
                if self._clf is not None
                else np.empty((1, 0), dtype=np.float32)
            )
            label, confidence = self._dynamic_prediction(
                model_feat,
                motion,
                sequence=update.completed_sequence,
            )
        except Exception as exc:
            print(f"[!] segmented dynamic prediction failed: {exc}", flush=True)
            self.reset_temporal_state()
            return {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": landmarks_json,
                "temporal": temporal_state,
                "intent_features": self._intent_feature_payload(
                    update.completed_sequence
                ),
            }

        if label:
            self._pending_dynamic_prediction = (label, confidence)
            self._pending_dynamic_repeats = 1
            self._pending_dynamic_motion_scale = float(motion_scale)
        return {
            "label": label,
            "confidence": confidence,
            "landmarks_json": landmarks_json,
            "dynamic_decision": dict(
                getattr(self, "_last_dynamic_decision", {}) or {}
            ),
            "temporal": temporal_state,
            "intent_features": self._intent_feature_payload(update.completed_sequence),
        }

    def _process_segmented_dynamic_frame(
        self,
        feat: np.ndarray,
        landmarks_json: str,
        *,
        motion_scale: float,
    ) -> Dict[str, Any]:
        pending = getattr(self, "_pending_dynamic_prediction", None)
        repeats = int(getattr(self, "_pending_dynamic_repeats", 0) or 0)
        if pending is not None and repeats > 0:
            self._pending_dynamic_repeats = repeats - 1
            return {
                "label": pending[0],
                "confidence": pending[1],
                "landmarks_json": landmarks_json,
                "dynamic_decision": dict(
                    getattr(self, "_last_dynamic_decision", {}) or {}
                ),
                "temporal": {
                    "enabled": True,
                    "phase": "completed",
                    "frames": int(self._window.maxlen or 36),
                    "required_frames": int(self._window.maxlen or 36),
                    "motion_scale": float(
                        getattr(self, "_pending_dynamic_motion_scale", 0.0)
                    ),
                    "end_reason": "pending_repeat",
                },
                "intent_features": self._intent_feature_payload(),
            }

        update = self._segmenter().update(feat, motion_scale=motion_scale)
        if update.completed_sequence is None:
            return {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": landmarks_json,
                "temporal": self._temporal_state_from_segment_update(
                    update,
                    motion_scale=motion_scale,
                ),
                "intent_features": self._intent_feature_payload(),
            }

        return self._classify_completed_dynamic_update(
            update,
            landmarks_json,
            motion_scale=motion_scale,
        )

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
        if self._uses_temporal_features():
            return True

        signature = self._gesture_signatures.get(label) or self._gesture_signatures.get(
            label.lower()
        )
        if not signature:
            return True

        expected_count = int(signature["non_thumb_count"])
        stability = float(signature.get("stability", 1.0))
        if stability < MIN_FINGER_SIGNATURE_STABILITY:
            return True
        if expected_count <= 0 or int(current_count) <= 0:
            return True
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

    def detect_hands(self, frame_rgb: np.ndarray) -> List[DetectedHand]:
        """Run MediaPipe once and return hands in stable classifier order."""
        if self._detector is None or frame_rgb is None or frame_rgb.size == 0:
            return []
        return self._ordered_hands(self._detector.detect_for_video_rgb(frame_rgb))

    def process_detected_hands(
        self,
        hands: List[DetectedHand],
    ) -> Dict[str, Any]:
        """Build model features from hands detected by this or a shared detector."""
        hands = self._ordered_hands(list(hands or []))
        landmarks_json = self._build_overlay_payload(hands)

        if not hands:
            if self._uses_global_dynamic_motion():
                update = self._segmenter().finish_due_to_hand_lost()
                if update.completed_sequence is not None:
                    return self._classify_completed_dynamic_update(
                        update,
                        landmarks_json,
                        motion_scale=0.0,
                    )
                if update.phase == "cooldown":
                    return {
                        "label": "",
                        "confidence": 0.0,
                        "landmarks_json": landmarks_json,
                        "temporal": self._temporal_state_from_segment_update(
                            update,
                            motion_scale=0.0,
                        ),
                    }
            self.reset_temporal_state()
            return {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": landmarks_json,
            }

        # Нормализуем точки рук для классификатора.
        normalized: List[np.ndarray] = []
        normalized_hands: List[DetectedHand] = []
        for hand in hands:
            try:
                normalized.append(normalize_landmarks(hand.landmarks))
                normalized_hands.append(hand)
            except Exception:
                continue

        hands = normalized_hands
        if not normalized:
            self.reset_temporal_state()
            return {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": landmarks_json,
            }

        current_finger_count = self._stable_non_thumb_count(hands)

        raw_feature_dim = int(getattr(self, "_raw_feature_dim", self._feature_dim))
        if self._classifier_two_hands:
            per_hand_dim = max(1, raw_feature_dim // 2)
            if len(normalized) >= 2:
                hand_features = [
                    self._hand_frame_feature(
                        hands[0],
                        normalized[0],
                        target_dim=per_hand_dim,
                    ),
                    self._hand_frame_feature(
                        hands[1],
                        normalized[1],
                        target_dim=per_hand_dim,
                    ),
                ]
            elif len(normalized) == 1:
                hand_features = [
                    self._hand_frame_feature(
                        hands[0],
                        normalized[0],
                        target_dim=per_hand_dim,
                    ),
                    np.zeros(per_hand_dim, dtype=np.float32),
                ]
            else:
                hand_features = [
                    np.zeros(per_hand_dim, dtype=np.float32),
                    np.zeros(per_hand_dim, dtype=np.float32),
                ]
            feat = np.concatenate(hand_features, axis=0)
        else:
            feat = self._hand_frame_feature(
                hands[0],
                normalized[0],
                target_dim=raw_feature_dim,
            )

        if feat.shape[0] != raw_feature_dim:
            if feat.shape[0] > raw_feature_dim:
                feat = feat[:raw_feature_dim]
            else:
                pad = np.zeros(raw_feature_dim - feat.shape[0], dtype=feat.dtype)
                feat = np.concatenate([feat, pad], axis=0)

        if self._uses_global_dynamic_motion():
            self._ensure_intent_window().append(feat.astype(np.float32, copy=False))
            hand_scales = []
            for hand in hands:
                points = np.asarray(hand.landmarks, dtype=np.float32)
                if points.shape == (21, 2):
                    size = points.max(axis=0) - points.min(axis=0)
                    hand_scales.append(float(np.linalg.norm(size)))
            motion_scale = (
                float(np.median(hand_scales))
                if hand_scales
                else 0.0
            )
            return self._process_segmented_dynamic_frame(
                feat,
                landmarks_json,
                motion_scale=motion_scale,
            )

        self._window.append(feat)
        temporal_state = self._dynamic_temporal_state()

        label = ""
        confidence = 0.0
        static_decision: dict[str, Any] = {}

        if (
            self._clf is not None
            and self._classes
            and self._window_ready_for_prediction()
        ):
            motion_ok, motion = self._dynamic_motion_gate()
            if not motion_ok:
                return {
                    "label": "",
                    "confidence": 0.0,
                    "landmarks_json": landmarks_json,
                    "temporal": temporal_state,
                    "intent_features": self._intent_feature_payload(),
                }
            model_feat = self._build_model_feature().reshape(1, -1)
            try:
                if self._uses_global_dynamic_motion():
                    label, confidence = self._dynamic_prediction(
                        model_feat,
                        motion,
                        sequence=np.stack(tuple(self._window), axis=0),
                    )
                else:
                    label, confidence = self._static_prediction(model_feat)
                    static_decision = dict(
                        getattr(self, "_last_static_decision", {}) or {}
                    )
            except Exception as e:
                print(f"[!] classifier prediction failed: {e}", flush=True)
                self._window.clear()
                return {
                    "label": "",
                    "confidence": 0.0,
                    "landmarks_json": landmarks_json,
                    "intent_features": self._intent_feature_payload(),
                }
        elif normalized and not self._uses_temporal_features():
            confidence = min(0.35 + 0.02 * len(self._window), 0.55)

        if label and not self._pose_matches_prediction(label, current_finger_count):
            static_decision = dict(
                static_decision
                or getattr(self, "_last_static_decision", {})
                or {}
            )
            static_decision["source"] = "finger_count_rejected"
            static_decision["rejection_reason"] = "finger_count_mismatch"
            self._window.clear()
            return {
                "label": "",
                "confidence": 0.0,
                "landmarks_json": landmarks_json,
                "static_decision": static_decision,
            }

        return {
            "label": label,
            "confidence": confidence,
            "landmarks_json": landmarks_json,
            "temporal": temporal_state,
            "static_decision": static_decision,
            "intent_features": self._intent_feature_payload(),
        }

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

        return self.process_detected_hands(self.detect_hands(frame_rgb))
