"""
GUI-агностичный контроллер для Flet-версии GestureBind.

Делает то же, что прежний :class:`app.main.AppController` (PySide6), но без
зависимостей от Qt: вместо ``Signal/Slot`` — обычные списки коллбэков.

Контроллер инкапсулирует:
    * захват камеры (OpenCV, тот же ``open_default_capture`` что и в QML версии);
    * встроенный CV-пайплайн (MediaPipe + KNN через :class:`GestureOnlineInfer`);
    * subprocess-распознавание (``cv/realtime_infer.py``) для фонового режима;
    * выполнение команд (``CommandExecutor``);
    * (лениво) связку с БД и привязками жестов.

Кадр камеры отдаётся UI как JPEG-байты, но Flet-preview отделён от ML:
захват и распознавание идут по свежему кадру, а UI-доставка работает отдельно.
"""
from __future__ import annotations

import os
import csv
import html
import signal
import subprocess
import sys
import tempfile
import threading
import time
import json
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Optional

from app.services.app_config import (
    AppConfig,
    ConfigStore,
    DEFAULT_CONFIG_PATH,
    database_url_from_config,
    resolve_config_path,
)
from app.services.gesture_taxonomy import (
    DEFAULT_TAXONOMY_PATH,
    GESTURE_TYPE_DYNAMIC,
    GESTURE_TYPE_NEGATIVE,
    labels_for_gesture_types,
    load_gesture_taxonomy,
    parse_gesture_type_scope,
)
from app.services.live_gesture_state import (
    LiveGestureSnapshot,
    LiveGestureState,
)
from cv.gesture_dataset_files import (
    augmented_sample_path,
    augmented_sample_paths,
    gesture_sample_paths,
    real_sample_paths,
    sample_source_from_path,
)


# ---- Опциональные зависимости (как в исходном app/main.py) -------------------

try:
    from app.services.command_executor import get_executor

    SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Сервисы недоступны: {e}")
    SERVICES_AVAILABLE = False
    get_executor = None  # type: ignore[assignment]

VoiceAssistant = None  # type: ignore[assignment]
create_voice_assistant = None  # type: ignore[assignment]
VOICE_ASSISTANT_AVAILABLE = False


try:
    from app.models.database import (
        Command as DbCommand,
        Gesture as DbGesture,
        GestureHistory as DbGestureHistory,
        GestureSample as DbGestureSample,
        get_db_session,
        init_database,
        reset_database_manager,
    )
    from app.services.binding_settings import (
        load_binding_settings,
        save_binding_settings,
    )
    from app.services.gesture_command_bridge import (
        GestureCommandBridge,
        save_gesture_binding,
    )
    from app.services.gesture_samples import (
        ensure_gesture,
        record_gesture_sample,
        resolve_project_path,
        sample_index_from_path,
        sample_shape_metadata,
    )
    from app.services.user_command_sync import (
        ACTION_SPEC_SCHEMA,
        CATEGORY_LABELS,
        DANGEROUS_ACTIONS,
        category_for_action,
        sync_db_commands_to_executor,
        validate_action_spec as _validate_action_spec,
    )

    BINDING_SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Сервис привязок недоступен: {e}")
    BINDING_SERVICES_AVAILABLE = False
    DbCommand = None  # type: ignore[assignment]
    DbGesture = None  # type: ignore[assignment]
    DbGestureHistory = None  # type: ignore[assignment]
    DbGestureSample = None  # type: ignore[assignment]
    init_database = None  # type: ignore[assignment]
    get_db_session = None  # type: ignore[assignment]
    reset_database_manager = None  # type: ignore[assignment]
    GestureCommandBridge = None  # type: ignore[assignment]
    save_gesture_binding = None  # type: ignore[assignment]
    ensure_gesture = None  # type: ignore[assignment]
    record_gesture_sample = None  # type: ignore[assignment]
    resolve_project_path = None  # type: ignore[assignment]
    sample_index_from_path = None  # type: ignore[assignment]
    sample_shape_metadata = None  # type: ignore[assignment]
    sync_db_commands_to_executor = None  # type: ignore[assignment]
    load_binding_settings = None  # type: ignore[assignment]
    save_binding_settings = None  # type: ignore[assignment]
    ACTION_SPEC_SCHEMA = {}  # type: ignore[assignment]
    CATEGORY_LABELS = {}  # type: ignore[assignment]
    DANGEROUS_ACTIONS = set()  # type: ignore[assignment]
    category_for_action = lambda _a: ""  # noqa: E731
    _validate_action_spec = lambda _s: "binding services unavailable"  # noqa: E731


try:
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


try:
    from app.services.pointer_control import PointerControlService, parse_landmarks_json

    POINTER_CONTROL_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Pointer control недоступен: {e}")
    POINTER_CONTROL_AVAILABLE = False
    PointerControlService = None  # type: ignore[assignment]
    parse_landmarks_json = None  # type: ignore[assignment]


# ---- Сигналы как простые подписки -------------------------------------------


class _Event:
    """Минимальный pub-sub: ``event.connect(cb)`` / ``event.emit(*args)``.

    Все коллбэки вызываются синхронно в том потоке, который сделал ``emit``.
    UI-слой (Flet) должен сам решить, нужно ли откладывать обновление в
    основной поток через ``page.update()``.
    """

    def __init__(self) -> None:
        self._handlers: list[Callable[..., None]] = []

    def connect(self, handler: Callable[..., None]) -> None:
        if handler not in self._handlers:
            self._handlers.append(handler)

    def disconnect(self, handler: Callable[..., None]) -> None:
        try:
            self._handlers.remove(handler)
        except ValueError:
            pass

    def emit(self, *args, **kwargs) -> None:
        for h in list(self._handlers):
            try:
                h(*args, **kwargs)
            except Exception as e:  # pragma: no cover - защитный лог
                print(f"[!] event handler error: {e}")


# ---- Контроллер -------------------------------------------------------------


GESTURE_CONFIRM_FRAMES = 6
AUTO_STATIC_GESTURE_CONFIRM_FRAMES = 15
RECOGNITION_MODEL_AUTO = "auto"
RECOGNITION_MODEL_STATIC = "static"
RECOGNITION_MODEL_DYNAMIC = "dynamic"
DYNAMIC_MODEL_PROFILE_KNN = "knn"
DYNAMIC_MODEL_PROFILE_SVM = "svm"
DYNAMIC_MODEL_PROFILE_EXTRA_TREES = "extra_trees"
DYNAMIC_MODEL_PROFILE_SEQUENCE_KNN = "sequence_knn"
DYNAMIC_MODEL_PROFILE_SEQUENCE_MLP = "sequence_mlp"
DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET = "sequence_rocket"
DYNAMIC_MODEL_PROFILE_SEQUENCE_MULTIROCKET = "sequence_multirocket"
DYNAMIC_MODEL_PROFILE_SEQUENCE_SPROCKET = "sequence_sprocket"
DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET = "sequence_shapelet"
DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET_72 = "sequence_shapelet_72"
DYNAMIC_MODEL_PROFILE_SEQUENCE_PHASE_HMM = "sequence_phase_hmm"
DYNAMIC_MODEL_PROFILE_SEQUENCE_ENSEMBLE = "sequence_ensemble"
DYNAMIC_MODEL_PROFILE_SEQUENCE_GRU_BACKBONE = "sequence_gru_backbone"
DYNAMIC_MODEL_PROFILE_SEQUENCE_LSTM_BACKBONE = "sequence_lstm_backbone"
DYNAMIC_MODEL_PROFILE_DYNAMIC_LANDMARK_LSTM_BACKBONE = (
    "dynamic_landmark_lstm_backbone"
)
DYNAMIC_MODEL_PROFILE_DYNAMIC_LANDMARK_CNN = "dynamic_landmark_cnn"
DYNAMIC_MODEL_PROFILE_PRODUCTION = DYNAMIC_MODEL_PROFILE_DYNAMIC_LANDMARK_LSTM_BACKBONE
DYNAMIC_MODEL_PROFILES = (
    DYNAMIC_MODEL_PROFILE_PRODUCTION,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_MLP,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_MULTIROCKET,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SPROCKET,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET_72,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_PHASE_HMM,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_ENSEMBLE,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_GRU_BACKBONE,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_LSTM_BACKBONE,
    DYNAMIC_MODEL_PROFILE_DYNAMIC_LANDMARK_CNN,
)
DYNAMIC_MODEL_FILENAMES = {
    DYNAMIC_MODEL_PROFILE_PRODUCTION: "dynamic_landmark_lstm_backbone.pkl",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_MLP: "dynamic_sequence_mlp.pkl",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET: "dynamic_sequence_rocket.pkl",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_MULTIROCKET: "dynamic_sequence_multirocket.pkl",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SPROCKET: "dynamic_sequence_sprocket.pkl",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET: "dynamic_sequence_shapelet.pkl",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET_72: "dynamic_sequence_shapelet_72.pkl",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_PHASE_HMM: "dynamic_sequence_phase_hmm.pkl",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_ENSEMBLE: "dynamic_sequence_ensemble.pkl",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_GRU_BACKBONE: "dynamic_sequence_gru_backbone.pkl",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_LSTM_BACKBONE: "dynamic_sequence_lstm_backbone.pkl",
    DYNAMIC_MODEL_PROFILE_DYNAMIC_LANDMARK_CNN: "dynamic_landmark_cnn.pkl",
}
DYNAMIC_METADATA_PREFIXES = {
    DYNAMIC_MODEL_PROFILE_PRODUCTION: "dynamic_landmark_lstm_backbone",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_MLP: "dynamic_sequence_mlp",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET: "dynamic_sequence_rocket",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_MULTIROCKET: "dynamic_sequence_multirocket",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SPROCKET: "dynamic_sequence_sprocket",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET: "dynamic_sequence_shapelet",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET_72: "dynamic_sequence_shapelet_72",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_PHASE_HMM: "dynamic_sequence_phase_hmm",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_ENSEMBLE: "dynamic_sequence_ensemble",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_GRU_BACKBONE: "dynamic_sequence_gru_backbone",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_LSTM_BACKBONE: "dynamic_sequence_lstm_backbone",
    DYNAMIC_MODEL_PROFILE_DYNAMIC_LANDMARK_CNN: "dynamic_landmark_cnn",
}
DYNAMIC_PROTOTYPE_FILENAMES = {
    DYNAMIC_MODEL_PROFILE_PRODUCTION: (
        "dynamic_landmark_lstm_backbone_prototypes.json"
    ),
    DYNAMIC_MODEL_PROFILE_SEQUENCE_MLP: "dynamic_sequence_mlp_prototypes.json",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET: "dynamic_sequence_rocket_prototypes.json",
    DYNAMIC_MODEL_PROFILE_SEQUENCE_MULTIROCKET: (
        "dynamic_sequence_multirocket_prototypes.json"
    ),
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SPROCKET: (
        "dynamic_sequence_sprocket_prototypes.json"
    ),
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET: (
        "dynamic_sequence_shapelet_prototypes.json"
    ),
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET_72: (
        "dynamic_sequence_shapelet_72_prototypes.json"
    ),
    DYNAMIC_MODEL_PROFILE_SEQUENCE_PHASE_HMM: (
        "dynamic_sequence_phase_hmm_prototypes.json"
    ),
    DYNAMIC_MODEL_PROFILE_SEQUENCE_ENSEMBLE: (
        "dynamic_sequence_ensemble_prototypes.json"
    ),
    DYNAMIC_MODEL_PROFILE_SEQUENCE_GRU_BACKBONE: (
        "dynamic_sequence_gru_backbone_prototypes.json"
    ),
    DYNAMIC_MODEL_PROFILE_SEQUENCE_LSTM_BACKBONE: (
        "dynamic_sequence_lstm_backbone_prototypes.json"
    ),
    DYNAMIC_MODEL_PROFILE_DYNAMIC_LANDMARK_CNN: (
        "dynamic_landmark_cnn_prototypes.json"
    ),
}
INTENT_GATE_MODEL_FILENAME = "intent_gate_mlp.pkl"
MODEL_VARIANT_PRODUCTION = "production"
MODEL_VARIANT_DIRS = {
    MODEL_VARIANT_PRODUCTION: "models",
    "baseline_internal": "models/experiments/external_negative/baseline_internal",
    "ipn_external": "models/experiments/external_negative/ipn_external",
    "combined_external": "models/experiments/external_negative/combined_external",
    "static_landmark_cnn": "models/experiments/static_landmark_cnn",
    "static_landmark_image_extra_trees": (
        "models/experiments/static_landmark_image_extra_trees"
    ),
    "prototype_distance": "models/experiments/dynamic_prototype/prototype_distance",
    "prototype_dtw": "models/experiments/dynamic_prototype/prototype_dtw",
}
MODEL_VARIANT_STATIC_MODEL_FILENAMES = {
    "static_landmark_cnn": "static_landmark_cnn.pkl",
}
DYNAMIC_ONLY_MODEL_VARIANTS = {
    "prototype_distance",
    "prototype_dtw",
}
MODEL_VARIANT_LABELS = {
    MODEL_VARIANT_PRODUCTION: "production",
    "baseline_internal": "baseline_internal",
    "ipn_external": "ipn_external",
    "combined_external": "combined_external",
    "static_landmark_cnn": "static_landmark_cnn",
    "static_landmark_image_extra_trees": "static_landmark_image_extra_trees",
    "prototype_distance": "prototype_distance",
    "prototype_dtw": "prototype_dtw",
}


def _static_model_filename_for_variant(variant: str) -> str:
    clean = str(variant or MODEL_VARIANT_PRODUCTION).strip()
    return MODEL_VARIANT_STATIC_MODEL_FILENAMES.get(clean, "knn.pkl")


STATIC_REJECTION_NEGATIVE_CLASSES = "negative_classes"
STATIC_REJECTION_CONFIDENCE_THRESHOLD = "confidence_threshold"
STATIC_REJECTION_OPEN_SET_POLICY = "open_set_policy"
STATIC_REJECTION_ONE_VS_REST_LOGREG = "one_vs_rest_logreg"
STATIC_REJECTION_ONE_CLASS_SVM = "one_class_svm"
STATIC_REJECTION_ISOLATION_FOREST = "isolation_forest"
STATIC_REJECTION_LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
STATIC_REJECTION_METRIC_NCA_CENTROID = "metric_nca_centroid"
STATIC_REJECTION_MLP_NEGATIVE_CLASSES = "mlp_negative_classes"
STATIC_REJECTION_METHODS = (
    STATIC_REJECTION_OPEN_SET_POLICY,
    STATIC_REJECTION_ONE_VS_REST_LOGREG,
    STATIC_REJECTION_NEGATIVE_CLASSES,
    STATIC_REJECTION_CONFIDENCE_THRESHOLD,
    STATIC_REJECTION_ONE_CLASS_SVM,
    STATIC_REJECTION_ISOLATION_FOREST,
    STATIC_REJECTION_LOCAL_OUTLIER_FACTOR,
    STATIC_REJECTION_METRIC_NCA_CENTROID,
    STATIC_REJECTION_MLP_NEGATIVE_CLASSES,
)
DYNAMIC_RECOGNITION_WINDOW = 36
DYNAMIC_RECOGNITION_LONG_WINDOW = 72
DYNAMIC_PROTOTYPE_DEFAULT_TARGET_DIM = 65
DYNAMIC_GESTURE_CONFIRM_FRAMES = 1
LIVE_EVAL_NO_COMMAND_LABEL = "no_command"
LIVE_GESTURE_IDLE_HOLD_SECONDS = 1.15
LIVE_GESTURE_INSPECTOR_LIMIT = 64
LIVE_GESTURE_INSPECTOR_EXPORT_FIELDS = [
    "sequence",
    "recordedAt",
    "phase",
    "label",
    "confidence",
    "progress",
    "frames",
    "requiredFrames",
    "mode",
    "route",
    "model",
    "staticReject",
    "reason",
    "displayText",
]
SYSTEM_REFERENCE_GESTURE_LABELS: set[str] = set()
USER_RECORDED_SAMPLE_SOURCES = {"camera", "user", "recording"}
DYNAMIC_POST_EVENT_SUPPRESS_SECONDS = 0.35
DYNAMIC_RETURN_SUPPRESS_SECONDS = 1.15
DYNAMIC_OPPOSITE_LABELS = {
    "swipe_down": {"swipe_up"},
    "swipe_up": {"swipe_down"},
    "swipe_left": {"swipe_right"},
    "swipe_right": {"swipe_left"},
}


def _is_dynamic_rejection_label(label: str) -> bool:
    clean = str(label or "").strip().lower()
    return (
        clean.startswith("negative_")
        or clean.startswith("background_")
        or clean.startswith("no_gesture")
        or clean.startswith("random_")
        or clean.startswith("partial_")
        or clean.startswith("return_")
        or clean.startswith("wrong_axis_")
    )


SAMPLE_RECORDING_READY_FRAMES = 6
SAMPLE_RECORDING_COUNTDOWN_SECONDS = 0.8
SAMPLE_RECORDING_STABILITY_THRESHOLD = 0.055
SAMPLE_RECORDING_STATIC_AUGMENTATIONS = 1
SAMPLE_RECORDING_DYNAMIC_AUGMENTATIONS = 2
DYNAMIC_SAMPLE_MIN_MOTION_ENERGY = 0.015
DYNAMIC_SAMPLE_DIRECTION_THRESHOLD = 0.05
LIVE_EVAL_DEFAULT_ATTEMPTS = 10
LIVE_EVAL_DEFAULT_TIMEOUT_SECONDS = 0.0
LIVE_EVAL_DEFAULT_MIN_CONFIDENCE = 0.60
LIVE_EVAL_ATTEMPT_COOLDOWN_SECONDS = 0.85
CAMERA_CAPTURE_WIDTH = 960
CAMERA_CAPTURE_HEIGHT = 540
CAMERA_INFERENCE_MAX_WIDTH = 480
CAMERA_INFERENCE_MAX_FPS = 30.0
CAMERA_PREVIEW_MAX_WIDTH = 960
CAMERA_PREVIEW_MAX_FPS = 30.0
CAMERA_PREVIEW_JPEG_QUALITY = 62
CAMERA_PREVIEW_DISABLE_ENV = "DPLM_DISABLE_CAMERA_PREVIEW"
RUNTIME_PERFORMANCE_FLUSH_SECONDS = 5.0
LIVE_USAGE_SUMMARY_WINDOW_SECONDS = 300.0
LIVE_USAGE_EVENTS_LIMIT = 1000


@dataclass(frozen=True)
class _CameraFrame:
    sequence: int
    frame_bgr: Any
    captured_at: float


def _camera_preview_enabled() -> bool:
    value = str(os.environ.get(CAMERA_PREVIEW_DISABLE_ENV, "")).strip().lower()
    return value not in {"1", "true", "yes", "on"}


def _resize_frame_to_max_width(frame: Any, max_width: int) -> Any:
    if frame is None or int(max_width) <= 0:
        return frame
    try:
        height, width = frame.shape[:2]
    except (AttributeError, ValueError):
        return frame
    if width <= int(max_width):
        return frame

    import cv2

    scale = float(max_width) / float(width)
    return cv2.resize(
        frame,
        (int(max_width), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


class AppController:
    """
    Главный контроллер Flet-версии.

    Атрибуты-события (используются View'ами для подписки на изменения):
        status_changed(str)
        recognizing_changed(bool)
        camera_active_changed(bool)
        camera_frame_updated()       # новый JPEG в `latest_jpeg_b64`
        gesture_detected(str)
        command_executed(str)
        recognition_event_recorded()
        confidence_changed(float)
        gesture_state_changed(dict)  # pending/confirmed/rejected/cooldown UI state
        landmarks_changed(str)       # JSON со списком ландмарок
        gesture_mode_changed(bool)
        pointer_mode_changed(bool)
        pointer_state_changed(dict)
        landmark_overlay_changed(bool)
        voice_assistant_state_changed(str)
        two_hands_changed(bool)
        recognition_model_mode_changed(str)
        dynamic_model_profile_changed(str)
        static_rejection_method_changed(str)
        model_variant_changed(str)
        sample_recording_changed(dict)
        live_evaluation_changed(dict)
    """

    def __init__(self) -> None:
        # Локальная техническая конфигурация -------------------------------
        self._config_store = ConfigStore(DEFAULT_CONFIG_PATH)
        try:
            self._config_store.ensure_exists()
        except Exception as e:
            print(f"[w] config ensure failed: {e}")
        self._config_file: AppConfig = self._config_store.load(include_env=False)
        self._config: AppConfig = self._config_store.load(include_env=True)

        # Состояние ----------------------------------------------------------
        self._status: str = "Idle"
        self._is_recognizing: bool = False
        self._is_camera_active: bool = False
        self._embedded_active: bool = False
        self._recognition_model_mode: str = RECOGNITION_MODEL_AUTO
        self._dynamic_model_profile: str = DYNAMIC_MODEL_PROFILE_PRODUCTION
        self._static_rejection_method: str = STATIC_REJECTION_OPEN_SET_POLICY
        self._two_hands_mode: bool = bool(self._config.recognition.two_hands_mode)
        self._gesture_mode: bool = True
        self._show_landmark_overlay: bool = True
        self._pointer_mode: bool = False
        self._pointer_state_payload: dict[str, Any] = {
            "enabled": False,
            "state": "idle",
            "moved": False,
            "clicked": False,
            "tabSwitched": "",
            "error": "",
        }
        self._confidence: float = 0.0
        self._landmarks_json: str = "[]"
        self._last_label: str = ""
        self._pending_label: str = ""
        self._pending_frames: int = 0
        self._pending_confidence_total: float = 0.0
        self._live_gesture_state = LiveGestureState()
        self._last_live_gesture_state_payload: dict[str, Any] | None = None
        self._live_gesture_idle_token: int = 0
        self._live_gesture_idle_timer: threading.Timer | None = None
        self._live_gesture_inspector_history: list[dict[str, Any]] = []
        self._live_gesture_inspector_sequence: int = 0
        self._last_execute_info: str = ""
        self._dynamic_return_guard: dict[str, Any] = {}
        self._live_evaluation: dict[str, Any] | None = None
        self._last_live_evaluation_snapshot: dict[str, Any] | None = None
        self._live_evaluation_lock = threading.RLock()
        self._gesture_taxonomy_cache: Any | None = None

        # События ------------------------------------------------------------
        self.status_changed = _Event()
        self.recognizing_changed = _Event()
        self.camera_active_changed = _Event()
        self.camera_frame_updated = _Event()
        self.gesture_detected = _Event()
        self.command_executed = _Event()
        self.recognition_event_recorded = _Event()
        self.confidence_changed = _Event()
        self.gesture_state_changed = _Event()
        self.landmarks_changed = _Event()
        self.gesture_mode_changed = _Event()
        self.pointer_mode_changed = _Event()
        self.pointer_state_changed = _Event()
        self.landmark_overlay_changed = _Event()
        self.voice_assistant_state_changed = _Event()
        self.two_hands_changed = _Event()
        self.recognition_model_mode_changed = _Event()
        self.dynamic_model_profile_changed = _Event()
        self.static_rejection_method_changed = _Event()
        self.model_variant_changed = _Event()
        self.sample_recording_changed = _Event()
        self.live_evaluation_changed = _Event()

        # Камера -------------------------------------------------------------
        self._camera_cap: Any | None = None
        self._camera_thread: threading.Thread | None = None
        self._camera_preview_thread: threading.Thread | None = None
        self._camera_stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_jpeg_bytes: bytes = b""
        self._frame_w = 0
        self._frame_h = 0
        self._camera_frame_seq = 0
        self._preview_condition = threading.Condition()
        self._latest_preview_frame: _CameraFrame | None = None
        self._target_fps = int(self._config.recognition.target_fps)
        self._runtime_inference_samples: list[dict[str, Any]] = []
        self._runtime_perf_last_flush = time.monotonic()
        self._live_usage_events: list[dict[str, Any]] = []
        self._live_usage_latest_runtime: dict[str, Any] = {}

        # Встроенный CV ------------------------------------------------------
        self._embedded_infer: Any | None = None
        self._pointer_control: Any | None = None

        # Subprocess realtime_infer (фоновое распознавание) ------------------
        self._recognition_process: Optional[subprocess.Popen] = None
        log_dir = self._configured_log_dir()
        self._recognition_pid_file = log_dir / "gesture_infer.pid"
        self._recognition_log_file = log_dir / "gesture_infer.log"
        self._recognition_pid_file.parent.mkdir(parents=True, exist_ok=True)

        # Сервисы ------------------------------------------------------------
        if SERVICES_AVAILABLE and get_executor:
            self._command_executor: Any = get_executor()
        else:
            self._command_executor = None
        self._voice_assistant: Optional[Any] = None

        # БД-мост (lazy) -----------------------------------------------------
        self._db_initialized = False
        self._gesture_command_bridge: Any | None = None
        # Если включено — после детекции жеста автоматически вызывается
        # ``execute_for_gesture(label, conf)``. Можно выключить в UI.
        self._auto_execute_on_gesture: bool = bool(
            self._config.recognition.auto_execute_on_gesture
        )

        # Subprocess для записи/обучения (CLI-обёртка) ---------------------
        self._training_proc: Optional[subprocess.Popen] = None
        self._sample_recording: dict[str, Any] | None = None
        self._sample_recording_detector: Any | None = None
        self._sample_recording_lock = threading.Lock()
        self._sample_recording_detector_lock = threading.RLock()

        # Подхватить уже запущенный фоновый процесс ---------------------------
        if self._is_recognition_pid_active():
            self._is_recognizing = True
            self._status = "Recognizing in background"

    def _active_training_process(self) -> Optional[subprocess.Popen]:
        proc = self._training_proc
        if proc is None:
            return None
        try:
            if proc.poll() is None:
                return proc
        except Exception:
            return proc
        self._training_proc = None
        return None

    def _clear_training_process(self, proc: subprocess.Popen) -> None:
        if self._training_proc is proc:
            self._training_proc = None

    def _live_usage_log_dir(self) -> Path | None:
        try:
            return self._configured_log_dir()
        except Exception:
            return None

    # ----------------------------------------------------------------------
    # Свойства / геттеры
    # ----------------------------------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_recognizing(self) -> bool:
        return self._is_recognizing

    @property
    def is_camera_active(self) -> bool:
        return self._is_camera_active

    @property
    def live_recognition_active(self) -> bool:
        return bool(self._embedded_active or self._is_recognizing)

    @property
    def two_hands_mode(self) -> bool:
        return self._two_hands_mode

    @property
    def recognition_model_mode(self) -> str:
        return str(getattr(self, "_recognition_model_mode", RECOGNITION_MODEL_AUTO))

    @property
    def dynamic_model_profile(self) -> str:
        profile = str(
            getattr(self, "_dynamic_model_profile", DYNAMIC_MODEL_PROFILE_PRODUCTION)
            or DYNAMIC_MODEL_PROFILE_PRODUCTION
        ).strip()
        return (
            profile
            if profile in DYNAMIC_MODEL_PROFILES
            else DYNAMIC_MODEL_PROFILE_PRODUCTION
        )

    @property
    def static_rejection_method(self) -> str:
        method = str(
            getattr(self, "_static_rejection_method", STATIC_REJECTION_OPEN_SET_POLICY)
            or STATIC_REJECTION_OPEN_SET_POLICY
        ).strip()
        return (
            method
            if method in STATIC_REJECTION_METHODS
            else STATIC_REJECTION_OPEN_SET_POLICY
        )

    @property
    def gesture_mode(self) -> bool:
        return self._gesture_mode

    def get_live_gesture_inspector_history(self) -> list[dict[str, Any]]:
        history = getattr(self, "_live_gesture_inspector_history", [])
        return [dict(item) for item in list(history)]

    def clear_live_gesture_inspector_history(self) -> None:
        self._live_gesture_inspector_history = []
        self._live_gesture_inspector_sequence = 0
        self._last_live_gesture_state_payload = None

    def export_live_gesture_inspector_history(
        self,
        fmt: str = "jsonl",
        path: str | Path | None = None,
    ) -> Path:
        clean_format = str(fmt or "jsonl").strip().lower()
        if clean_format not in {"jsonl", "csv"}:
            raise ValueError("fmt must be 'jsonl' or 'csv'")

        target = (
            Path(path).expanduser()
            if path is not None
            else self._configured_log_dir() / f"live_gesture_inspector.{clean_format}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = list(reversed(self.get_live_gesture_inspector_history()))

        if clean_format == "jsonl":
            with target.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                    fh.write("\n")
            return target

        with target.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=LIVE_GESTURE_INSPECTOR_EXPORT_FIELDS,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return target

    @property
    def show_landmark_overlay(self) -> bool:
        return self._show_landmark_overlay

    @property
    def pointer_mode(self) -> bool:
        return self._pointer_mode

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def landmarks_json(self) -> str:
        return self._landmarks_json

    @property
    def latest_jpeg_bytes(self) -> bytes:
        """Последний кадр в формате JPEG (для ``ft.Image(src=...)``)."""
        with self._frame_lock:
            return self._latest_jpeg_bytes

    @property
    def frame_size(self) -> tuple[int, int]:
        with self._frame_lock:
            return (self._frame_w, self._frame_h)

    @property
    def sample_recording_state(self) -> dict[str, Any]:
        return self._sample_recording_snapshot()

    @property
    def cv2_available(self) -> bool:
        return _CV2_AVAILABLE

    @property
    def command_executor(self) -> Any | None:
        return self._command_executor

    @property
    def config_path(self) -> Path:
        return self._config_store.path

    def get_app_config(self) -> dict[str, Any]:
        """Вернуть настройки, записанные именно в config.json (без env override)."""
        self._config_file = self._config_store.load(include_env=False)
        return self._config_file.to_dict()

    def get_effective_app_config(self) -> dict[str, Any]:
        """Вернуть фактически применяемые настройки с учётом окружения."""
        self._config = self._config_store.load(include_env=True)
        return self._config.to_dict()

    @property
    def model_variant(self) -> str:
        self._config_file = self._config_store.load(include_env=False)
        return self._model_variant_for_dir(self._config_file.paths.models_dir)

    def list_model_variants(self) -> list[dict[str, Any]]:
        current = self.model_variant
        variants: list[dict[str, Any]] = []
        for key, rel_dir in MODEL_VARIANT_DIRS.items():
            models_dir = resolve_config_path(rel_dir)
            static_model_filename = _static_model_filename_for_variant(key)
            variants.append(
                {
                    "key": key,
                    "label": MODEL_VARIANT_LABELS.get(key, key),
                    "models_dir": str(models_dir),
                    "exists": models_dir.exists(),
                    "selected": key == current,
                    "static_model_exists": (
                        models_dir / static_model_filename
                    ).exists(),
                    "static_model_filename": static_model_filename,
                    "dynamic_model_exists": any(
                        (models_dir / filename).exists()
                        for filename in DYNAMIC_MODEL_FILENAMES.values()
                    ),
                }
            )
        return variants

    def list_dynamic_model_profiles(self) -> list[dict[str, Any]]:
        dataset_labels = self._dataset_labels_with_real_samples()
        profiles: list[dict[str, Any]] = []
        for profile in DYNAMIC_MODEL_PROFILES:
            model_path = self._dynamic_model_path_for_profile(profile)
            classes_path = self._dynamic_classes_path_for_profile(profile)
            feature_dim_path = self._dynamic_feature_dim_path_for_profile(profile)
            feature_mode_path = self._dynamic_feature_mode_path_for_profile(profile)
            classes = self._read_json_string_list(classes_path)
            positive_labels = [
                label for label in classes if not _is_dynamic_rejection_label(label)
            ]
            missing_dataset_labels = [
                label
                for label in positive_labels
                if dataset_labels and label.lower() not in dataset_labels
            ]
            exists = (
                model_path.exists()
                and classes_path.exists()
                and feature_dim_path.exists()
                and feature_mode_path.exists()
            )
            stale = bool(missing_dataset_labels)
            profiles.append(
                {
                    "key": profile,
                    "label": self._dynamic_model_profile_label(
                        profile,
                        exists=exists,
                        stale=stale,
                    ),
                    "model_path": str(model_path),
                    "classes_path": str(classes_path),
                    "exists": exists,
                    "selected": profile == self.dynamic_model_profile,
                    "classes": classes,
                    "positive_labels": positive_labels,
                    "missing_dataset_labels": missing_dataset_labels,
                    "stale": stale,
                    "quick_selectable": bool(exists and not stale),
                }
            )
        return profiles

    def apply_model_variant(self, variant: str) -> tuple[bool, list[str], list[str]]:
        target = str(variant or "").strip()
        if target not in MODEL_VARIANT_DIRS:
            target = MODEL_VARIANT_PRODUCTION
        for env_key in (
            "DPLM_MODELS_DIR",
            "DPLM_MODEL_PATH",
            "DPLM_CLASSES_PATH",
            "DPLM_FEATURE_DIM_PATH",
        ):
            os.environ.pop(env_key, None)
        models_dir = MODEL_VARIANT_DIRS[target]
        static_models_dir = (
            MODEL_VARIANT_DIRS[MODEL_VARIANT_PRODUCTION]
            if target in DYNAMIC_ONLY_MODEL_VARIANTS
            else models_dir
        )
        static_model_filename = (
            "knn.pkl"
            if target in DYNAMIC_ONLY_MODEL_VARIANTS
            else _static_model_filename_for_variant(target)
        )
        raw = self.get_app_config()
        paths = dict(raw.get("paths") or {})
        paths.update(
            {
                "models_dir": models_dir,
                "model_path": f"{static_models_dir}/{static_model_filename}",
                "classes_path": f"{static_models_dir}/classes.json",
                "feature_dim_path": f"{static_models_dir}/feature_dim.txt",
            }
        )
        raw["paths"] = paths
        ok, errors, warnings = self.save_app_config(raw)
        if not ok:
            return ok, errors, warnings

        self._reset_embedded_infer_after_model_change()
        event = getattr(self, "model_variant_changed", None)
        if event is not None:
            event.emit(target)
        self._set_status(f"Model variant: {target}")
        return ok, errors, warnings

    def get_config_env_overrides(self) -> dict[str, str]:
        return self._config_store.env_overrides()

    def save_app_config(self, raw: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
        """Сохранить технические настройки в config.json и применить runtime flags."""
        config = AppConfig.from_dict(raw)
        result = self._config_store.validate(config)
        if not result.ok:
            return False, result.errors, result.warnings

        try:
            self._config_store.save(config)
            self._config_file = self._config_store.load(include_env=False)
            self._config = self._config_store.load(include_env=True)
            self._apply_runtime_config()
            self._reset_db_bridge()
        except Exception as e:
            return False, [str(e)], result.warnings
        return True, [], result.warnings

    def get_system_status(self) -> dict[str, Any]:
        """Лёгкая сводка для экрана настроек."""
        self._config = self._config_store.load(include_env=True)
        validation = self._config_store.validate(self._config)
        db_url = database_url_from_config(self._config)
        data_dir = self._configured_data_dir()
        models_dir = self._configured_models_dir()
        model_path = self._configured_model_path()
        classes_path = self._configured_classes_path()
        feature_dim_path = self._configured_feature_dim_path()
        log_dir = self._configured_log_dir()

        db_ok = False
        db_error = ""
        if BINDING_SERVICES_AVAILABLE:
            try:
                if not self._db_initialized:
                    init_database()
                    self._db_initialized = True
                from app.models import database as database_module

                manager = getattr(database_module, "db_manager", None)
                db_ok = bool(manager.healthcheck()) if manager is not None else False
            except Exception as e:
                db_error = str(e)

        return {
            "configPath": str(self._config_store.path),
            "configExists": self._config_store.path.exists(),
            "envOverrides": self.get_config_env_overrides(),
            "databaseUrl": self._mask_database_url(db_url),
            "databaseOk": db_ok,
            "databaseError": db_error,
            "cv2Available": _CV2_AVAILABLE,
            "cameraIndex": int(self._config.recognition.camera_index),
            "targetFps": int(self._config.recognition.target_fps),
            "dataDir": str(data_dir),
            "dataDirExists": data_dir.exists(),
            "modelsDir": str(models_dir),
            "modelsDirExists": models_dir.exists(),
            "modelPath": str(model_path),
            "modelExists": model_path.exists(),
            "classesPath": str(classes_path),
            "classesExists": classes_path.exists(),
            "featureDimPath": str(feature_dim_path),
            "featureDimExists": feature_dim_path.exists(),
            "logDir": str(log_dir),
            "logDirExists": log_dir.exists(),
            "recognitionLog": str(self._recognition_log_file),
            "voiceEnabled": bool(self._config.assistant.voice_enabled),
            "voiceAvailable": VOICE_ASSISTANT_AVAILABLE,
            "validationErrors": validation.errors,
            "validationWarnings": validation.warnings,
        }

    def run_self_test(self, *, check_camera: bool = False) -> dict[str, Any]:
        from app.services.diagnostics import run_diagnostics

        report = run_diagnostics(
            config=self._config_store.load(include_env=True),
            config_store=self._config_store,
            check_database=True,
            check_camera=check_camera,
        )
        if report.get("ok"):
            self._set_status("Diagnostics: OK")
        else:
            failures = int((report.get("summary") or {}).get("fail", 0))
            self._set_status(f"Diagnostics: {failures} failures")
        return report

    def _apply_runtime_config(self) -> None:
        self._target_fps = int(self._config.recognition.target_fps)
        self._auto_execute_on_gesture = bool(
            self._config.recognition.auto_execute_on_gesture
        )
        log_dir = self._configured_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        self._recognition_pid_file = log_dir / "gesture_infer.pid"
        self._recognition_log_file = log_dir / "gesture_infer.log"
        self.set_two_hands_mode(bool(self._config.recognition.two_hands_mode))

    def _reset_db_bridge(self) -> None:
        self._gesture_command_bridge = None
        self._db_initialized = False
        if reset_database_manager is not None:
            try:
                reset_database_manager()
            except Exception as e:
                print(f"[w] reset_database_manager: {e}")

    def _configured_path(self, value: str | Path) -> Path:
        return resolve_config_path(value)

    def _production_models_dir(self) -> Path:
        return self._configured_path(MODEL_VARIANT_DIRS[MODEL_VARIANT_PRODUCTION])

    def _configured_static_artifact_path(
        self,
        configured_value: str | Path,
        production_filename: str,
    ) -> Path:
        configured = self._configured_path(configured_value)
        if configured.exists():
            return configured
        fallback = self._production_models_dir() / production_filename
        return fallback if fallback.exists() else configured

    def _configured_dynamic_artifact_path(self, filename: str) -> Path:
        configured = self._configured_models_dir() / filename
        if configured.exists():
            return configured
        fallback = self._production_models_dir() / filename
        return fallback if fallback.exists() else configured

    def _dynamic_model_path_for_profile(self, profile: str) -> Path:
        filename = DYNAMIC_MODEL_FILENAMES.get(
            str(profile or "").strip(),
            DYNAMIC_MODEL_FILENAMES[DYNAMIC_MODEL_PROFILE_PRODUCTION],
        )
        return self._configured_dynamic_artifact_path(filename)

    def _dynamic_metadata_prefix_for_profile(self, profile: str) -> str:
        return DYNAMIC_METADATA_PREFIXES.get(str(profile or "").strip(), "dynamic")

    def _dynamic_classes_path_for_profile(self, profile: str) -> Path:
        prefix = self._dynamic_metadata_prefix_for_profile(profile)
        return self._configured_dynamic_artifact_path(f"{prefix}_classes.json")

    def _dynamic_feature_dim_path_for_profile(self, profile: str) -> Path:
        prefix = self._dynamic_metadata_prefix_for_profile(profile)
        return self._configured_dynamic_artifact_path(f"{prefix}_feature_dim.txt")

    def _dynamic_feature_mode_path_for_profile(self, profile: str) -> Path:
        prefix = self._dynamic_metadata_prefix_for_profile(profile)
        return self._configured_dynamic_artifact_path(f"{prefix}_feature_mode.txt")

    def _dynamic_model_profile_label(
        self,
        profile: str,
        *,
        exists: bool,
        stale: bool,
    ) -> str:
        if profile == DYNAMIC_MODEL_PROFILE_PRODUCTION:
            suffix = "актуальная"
        elif stale:
            suffix = "устаревшая"
        elif not exists:
            suffix = "нет модели"
        else:
            suffix = "готова"
        return f"{profile} ({suffix})"

    def _read_json_string_list(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for item in raw:
            clean = str(item or "").strip()
            if clean:
                out.append(clean)
        return out

    def _dataset_labels_with_real_samples(self) -> set[str]:
        root = self._configured_data_dir()
        if not root.exists():
            return set()
        labels: set[str] = set()
        for label_dir in root.iterdir():
            if not label_dir.is_dir():
                continue
            try:
                if real_sample_paths(label_dir):
                    labels.add(label_dir.name.lower())
            except Exception:
                continue
        return labels

    def _configured_data_dir(self) -> Path:
        return self._configured_path(self._config.paths.data_dir)

    def _configured_models_dir(self) -> Path:
        return self._configured_path(self._config.paths.models_dir)

    def _configured_model_path(self) -> Path:
        return self._configured_static_artifact_path(
            self._config.paths.model_path,
            "knn.pkl",
        )

    def _configured_classes_path(self) -> Path:
        return self._configured_static_artifact_path(
            self._config.paths.classes_path,
            "classes.json",
        )

    def _configured_feature_dim_path(self) -> Path:
        return self._configured_static_artifact_path(
            self._config.paths.feature_dim_path,
            "feature_dim.txt",
        )

    def _configured_log_dir(self) -> Path:
        return self._configured_path(self._config.paths.log_dir)

    def _configured_feature_mode_path(self) -> Path:
        return self._configured_static_artifact_path(
            self._configured_models_dir() / "feature_mode.txt",
            "feature_mode.txt",
        )

    def _static_rejection_verifier_path(self) -> Path:
        return self._configured_static_artifact_path(
            self._configured_models_dir() / "static_rejection_verifiers.pkl",
            "static_rejection_verifiers.pkl",
        )

    def _configured_taxonomy_path(self) -> Path:
        return DEFAULT_TAXONOMY_PATH

    def _model_variant_for_dir(self, models_dir: str | Path) -> str:
        try:
            current = resolve_config_path(models_dir).resolve()
        except Exception:
            current = resolve_config_path(models_dir)
        for key, rel_dir in MODEL_VARIANT_DIRS.items():
            try:
                candidate = resolve_config_path(rel_dir).resolve()
            except Exception:
                candidate = resolve_config_path(rel_dir)
            if current == candidate:
                return key
        return "custom"

    def _reset_embedded_infer_after_model_change(self) -> None:
        try:
            self._reset_gesture_confirmation(immediate_ui=True)
        except Exception:
            pass
        self._set_confidence(0.0)
        if getattr(self, "_last_label", ""):
            self._last_label = ""
            try:
                self.gesture_detected.emit("")
            except Exception:
                pass

        infer = getattr(self, "_embedded_infer", None)
        self._embedded_infer = None
        if infer is not None:
            try:
                infer.close()
            except Exception:
                pass
        if getattr(self, "_embedded_active", False):
            self._set_status(self._live_recognition_status())

    def _dynamic_model_path(self) -> Path:
        return self._dynamic_model_path_for_profile(self.dynamic_model_profile)

    def _dynamic_classes_path(self) -> Path:
        return self._dynamic_classes_path_for_profile(self.dynamic_model_profile)

    def _dynamic_feature_dim_path(self) -> Path:
        return self._dynamic_feature_dim_path_for_profile(self.dynamic_model_profile)

    def _dynamic_feature_mode_path(self) -> Path:
        return self._dynamic_feature_mode_path_for_profile(self.dynamic_model_profile)

    def _dynamic_prototypes_path(self) -> Path:
        filename = DYNAMIC_PROTOTYPE_FILENAMES.get(
            self.dynamic_model_profile,
            "dynamic_prototypes.json",
        )
        return self._configured_dynamic_artifact_path(filename)

    def _intent_gate_model_path(self) -> Path:
        configured = self._configured_models_dir() / INTENT_GATE_MODEL_FILENAME
        if configured.exists():
            return configured
        fallback = self._production_models_dir() / INTENT_GATE_MODEL_FILENAME
        return fallback if fallback.exists() else configured

    def _embedded_model_paths(self) -> tuple[Path, Path, Path, Path]:
        if self.recognition_model_mode == RECOGNITION_MODEL_DYNAMIC:
            return (
                self._dynamic_model_path(),
                self._dynamic_classes_path(),
                self._dynamic_feature_dim_path(),
                self._dynamic_feature_mode_path(),
            )
        return (
            self._configured_model_path(),
            self._configured_classes_path(),
            self._configured_feature_dim_path(),
            self._configured_feature_mode_path(),
        )

    def _embedded_recognition_window(self) -> int:
        if self.recognition_model_mode == RECOGNITION_MODEL_DYNAMIC:
            return self._dynamic_recognition_window()
        return 30

    def _dynamic_recognition_window(self) -> int:
        if self.dynamic_model_profile in {
            DYNAMIC_MODEL_PROFILE_PRODUCTION,
            DYNAMIC_MODEL_PROFILE_DYNAMIC_LANDMARK_CNN,
            DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET_72,
        }:
            return DYNAMIC_RECOGNITION_LONG_WINDOW
        return DYNAMIC_RECOGNITION_WINDOW

    def _mask_database_url(self, url: str) -> str:
        if "://" not in url or "@" not in url:
            return url
        scheme, rest = url.split("://", 1)
        creds, host = rest.split("@", 1)
        if ":" in creds:
            user = creds.split(":", 1)[0]
            return f"{scheme}://{user}:***@{host}"
        return f"{scheme}://***@{host}"

    # ----------------------------------------------------------------------
    # Внутренние сеттеры с событиями
    # ----------------------------------------------------------------------

    def _set_status(self, value: str) -> None:
        if value != self._status:
            self._status = value
            self.status_changed.emit(value)

    def _set_recognizing(self, value: bool) -> None:
        if value != self._is_recognizing:
            self._is_recognizing = value
            self.recognizing_changed.emit(value)

    def _set_camera_active(self, value: bool) -> None:
        if value != self._is_camera_active:
            self._is_camera_active = value
            self.camera_active_changed.emit(value)

    def _set_confidence(self, value: float) -> None:
        if abs(value - self._confidence) > 1e-6:
            self._confidence = value
            self.confidence_changed.emit(value)

    def _set_landmarks(self, value: str) -> None:
        if value != self._landmarks_json:
            self._landmarks_json = value
            self.landmarks_changed.emit(value)

    def _reset_gesture_confirmation(self, *, immediate_ui: bool = False) -> None:
        snapshot = self._ensure_live_gesture_state().reset()
        self._sync_live_gesture_pending_fields()
        self._emit_live_gesture_state(snapshot, immediate_idle=immediate_ui)

    def _ensure_live_gesture_state(self) -> LiveGestureState:
        state = getattr(self, "_live_gesture_state", None)
        if state is None:
            state = LiveGestureState()
            self._live_gesture_state = state
        return state

    def _sync_live_gesture_pending_fields(self) -> None:
        state = self._ensure_live_gesture_state()
        self._pending_label = state.pending_label
        self._pending_frames = state.pending_frames
        self._pending_confidence_total = state.pending_confidence_total

    def _cancel_live_gesture_idle_timer(self) -> None:
        self._live_gesture_idle_token = int(
            getattr(self, "_live_gesture_idle_token", 0)
        ) + 1
        timer = getattr(self, "_live_gesture_idle_timer", None)
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass
        self._live_gesture_idle_timer = None

    def _live_gesture_route_fallback(self) -> str:
        mode = self.recognition_model_mode
        if mode == RECOGNITION_MODEL_DYNAMIC:
            return "dynamic"
        if mode == RECOGNITION_MODEL_STATIC:
            return "static"
        return "auto"

    def _live_gesture_model_label(self, route: str) -> str:
        clean_route = str(route or "").strip().lower()
        if clean_route == "dynamic":
            return self.dynamic_model_profile
        if clean_route == "static":
            return f"static:{self.static_rejection_method}"
        if self.recognition_model_mode == RECOGNITION_MODEL_DYNAMIC:
            return self.dynamic_model_profile
        if self.recognition_model_mode == RECOGNITION_MODEL_STATIC:
            return f"static:{self.static_rejection_method}"
        return self.dynamic_model_profile

    def _enrich_live_gesture_state_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        enriched = dict(payload)
        route = str(enriched.get("route") or "").strip().lower()
        if not route:
            route = self._live_gesture_route_fallback()
        enriched["route"] = route
        enriched["mode"] = self.recognition_model_mode
        enriched["model"] = self._live_gesture_model_label(route)
        enriched["staticReject"] = self.static_rejection_method
        return enriched

    def _record_live_gesture_inspector_entry(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        history = getattr(self, "_live_gesture_inspector_history", None)
        if history is None:
            history = []
            self._live_gesture_inspector_history = history
        sequence = int(getattr(self, "_live_gesture_inspector_sequence", 0)) + 1
        self._live_gesture_inspector_sequence = sequence
        entry = dict(payload)
        entry["sequence"] = sequence
        entry["recordedAt"] = time.time()
        history.insert(0, entry)
        del history[LIVE_GESTURE_INSPECTOR_LIMIT:]
        return entry

    def _publish_live_gesture_state_payload(self, payload: dict[str, Any]) -> None:
        if payload == getattr(self, "_last_live_gesture_state_payload", None):
            return
        self._last_live_gesture_state_payload = payload
        event_payload = self._record_live_gesture_inspector_entry(payload)
        event = getattr(self, "gesture_state_changed", None)
        if event is not None:
            try:
                event.emit(event_payload)
            except Exception:
                pass

    def _schedule_live_gesture_idle(self, payload: dict[str, Any]) -> None:
        self._cancel_live_gesture_idle_timer()
        token = int(getattr(self, "_live_gesture_idle_token", 0))

        def publish_later() -> None:
            if int(getattr(self, "_live_gesture_idle_token", 0)) != token:
                return
            self._live_gesture_idle_timer = None
            self._publish_live_gesture_state_payload(payload)

        timer = threading.Timer(LIVE_GESTURE_IDLE_HOLD_SECONDS, publish_later)
        timer.daemon = True
        self._live_gesture_idle_timer = timer
        timer.start()

    def _emit_live_gesture_state(
        self,
        snapshot: LiveGestureSnapshot,
        *,
        immediate_idle: bool = False,
    ) -> None:
        self._sync_live_gesture_pending_fields()
        payload = self._enrich_live_gesture_state_payload(snapshot.as_dict())
        phase = str(payload.get("phase") or "")
        if phase != "idle":
            self._cancel_live_gesture_idle_timer()
            self._publish_live_gesture_state_payload(payload)
            return

        previous = getattr(self, "_last_live_gesture_state_payload", None) or {}
        previous_phase = str(previous.get("phase") or "")
        if (
            not immediate_idle
            and previous_phase
            and previous_phase != "idle"
            and LIVE_GESTURE_IDLE_HOLD_SECONDS > 0
        ):
            self._schedule_live_gesture_idle(payload)
            return

        self._cancel_live_gesture_idle_timer()
        self._publish_live_gesture_state_payload(payload)

    def _is_no_command_label(self, label: str) -> bool:
        return str(label or "").strip().lower() == LIVE_EVAL_NO_COMMAND_LABEL

    def _gesture_type_for_label(self, label: str) -> str:
        if self._is_no_command_label(label):
            return GESTURE_TYPE_NEGATIVE
        taxonomy = getattr(self, "_gesture_taxonomy_cache", None)
        if taxonomy is None:
            taxonomy = load_gesture_taxonomy(self._configured_taxonomy_path())
            self._gesture_taxonomy_cache = taxonomy
        return taxonomy.gesture_type_for_label(label)

    def _is_negative_label(self, label: str) -> bool:
        try:
            return self._gesture_type_for_label(label) == GESTURE_TYPE_NEGATIVE
        except Exception:
            return False

    def _gesture_confirm_frames(
        self,
        label: str = "",
        route_metadata: dict[str, Any] | None = None,
    ) -> int:
        if label and self._is_negative_label(label):
            return DYNAMIC_GESTURE_CONFIRM_FRAMES
        route = str((route_metadata or {}).get("route") or "").strip().lower()
        if route == "dynamic":
            return DYNAMIC_GESTURE_CONFIRM_FRAMES
        if route == "static":
            return AUTO_STATIC_GESTURE_CONFIRM_FRAMES
        if self.recognition_model_mode == RECOGNITION_MODEL_DYNAMIC:
            return DYNAMIC_GESTURE_CONFIRM_FRAMES
        if self.recognition_model_mode == RECOGNITION_MODEL_STATIC:
            return AUTO_STATIC_GESTURE_CONFIRM_FRAMES
        if self.recognition_model_mode == RECOGNITION_MODEL_AUTO and label:
            try:
                if self._gesture_type_for_label(label) == GESTURE_TYPE_DYNAMIC:
                    return DYNAMIC_GESTURE_CONFIRM_FRAMES
                return AUTO_STATIC_GESTURE_CONFIRM_FRAMES
            except Exception as e:
                print(f"[w] dynamic confirmation taxonomy: {e}", flush=True)
        return GESTURE_CONFIRM_FRAMES

    def _update_gesture_confirmation(
        self,
        label: str,
        confidence: float,
        route_metadata: dict[str, Any] | None = None,
    ) -> tuple[bool, float]:
        clean = (label or "").strip()
        if not clean:
            self._reset_gesture_confirmation()
            return False, 0.0

        metadata = route_metadata or {}
        route = str(metadata.get("route") or "")
        reason = str(metadata.get("selected_reason") or "")
        state = self._ensure_live_gesture_state()
        confirmed, avg_conf, snapshot = state.observe(
            clean,
            confidence,
            required_frames=self._gesture_confirm_frames(
                clean,
                route_metadata=metadata,
            ),
            route=route,
            reason=reason,
        )
        self._emit_live_gesture_state(snapshot)
        return confirmed, avg_conf

    def _dynamic_return_guard_active(
        self,
        label: str,
        route_metadata: dict[str, Any] | None,
        *,
        now: float | None = None,
    ) -> str:
        route = str((route_metadata or {}).get("route") or "").strip().lower()
        if route != "dynamic":
            try:
                if self._gesture_type_for_label(label) != GESTURE_TYPE_DYNAMIC:
                    return ""
            except Exception:
                return ""

        guard = getattr(self, "_dynamic_return_guard", {}) or {}
        monotonic_now = time.monotonic() if now is None else float(now)
        if monotonic_now > float(guard.get("until") or 0.0):
            return ""

        if monotonic_now <= float(guard.get("all_until") or 0.0):
            return "post_dynamic_cooldown"

        previous = str(guard.get("label") or "").strip().lower()
        current = str(label or "").strip().lower()
        if current in DYNAMIC_OPPOSITE_LABELS.get(previous, set()):
            return "opposite_return_motion"
        return ""

    def _mark_dynamic_event_accepted(
        self,
        label: str,
        route_metadata: dict[str, Any] | None,
        *,
        now: float | None = None,
    ) -> None:
        route = str((route_metadata or {}).get("route") or "").strip().lower()
        if route != "dynamic":
            return
        monotonic_now = time.monotonic() if now is None else float(now)
        self._dynamic_return_guard = {
            "label": str(label or "").strip().lower(),
            "all_until": monotonic_now + DYNAMIC_POST_EVENT_SUPPRESS_SECONDS,
            "until": monotonic_now + DYNAMIC_RETURN_SUPPRESS_SECONDS,
        }

    def _live_evaluation_snapshot_from_session(
        self,
        session: dict[str, Any] | None,
        *,
        message: str = "",
    ) -> dict[str, Any]:
        if session is None:
            return {
                "active": False,
                "expectedLabel": "",
                "targetAttempts": 0,
                "attemptIndex": 0,
                "total": 0,
                "correct": 0,
                "wrong": 0,
                "missed": 0,
                "accuracy": 0.0,
                "progress": 0.0,
                "recognitionModelMode": self.recognition_model_mode,
                "dynamicModelProfile": self.dynamic_model_profile,
                "staticRejectionMethod": self.static_rejection_method,
                "minConfidence": LIVE_EVAL_DEFAULT_MIN_CONFIDENCE,
                "timeoutSeconds": LIVE_EVAL_DEFAULT_TIMEOUT_SECONDS,
                "lastPrediction": "",
                "lastConfidence": 0.0,
                "lastResult": "",
                "message": message,
                "attempts": [],
            }

        target = max(1, int(session.get("target_attempts") or 1))
        total = max(0, int(session.get("total") or 0))
        correct = max(0, int(session.get("correct") or 0))
        wrong = max(0, int(session.get("wrong") or 0))
        missed = max(0, int(session.get("missed") or 0))
        active = bool(session.get("active"))
        attempt_index = min(target, total + 1) if active and total < target else total
        accuracy = correct / total if total else 0.0
        return {
            "active": active,
            "expectedLabel": str(session.get("expected_label") or ""),
            "targetAttempts": target,
            "attemptIndex": attempt_index,
            "total": total,
            "correct": correct,
            "wrong": wrong,
            "missed": missed,
            "accuracy": accuracy,
            "progress": min(1.0, total / float(target)),
            "recognitionModelMode": str(
                session.get("recognition_model_mode") or self.recognition_model_mode
            ),
            "dynamicModelProfile": str(
                session.get("dynamic_model_profile") or self.dynamic_model_profile
            ),
            "staticRejectionMethod": str(
                session.get("static_rejection_method") or self.static_rejection_method
            ),
            "minConfidence": float(session.get("min_confidence") or 0.0),
            "timeoutSeconds": float(session.get("timeout_seconds") or 0.0),
            "lastPrediction": str(session.get("last_prediction") or ""),
            "lastConfidence": float(session.get("last_confidence") or 0.0),
            "lastResult": str(session.get("last_result") or ""),
            "message": message or str(session.get("message") or ""),
            "attempts": list(session.get("attempts") or []),
        }

    def _emit_live_evaluation_changed(
        self,
        session: dict[str, Any] | None = None,
        *,
        message: str = "",
    ) -> None:
        if session is None:
            with self._live_evaluation_lock:
                session = self._live_evaluation
        event = getattr(self, "live_evaluation_changed", None)
        if event is not None:
            event.emit(
                self._live_evaluation_snapshot_from_session(
                    session,
                    message=message,
                )
            )

    def current_live_evaluation(self) -> dict[str, Any]:
        with self._live_evaluation_lock:
            session = self._live_evaluation
            if session is None and self._last_live_evaluation_snapshot is not None:
                return dict(self._last_live_evaluation_snapshot)
        return self._live_evaluation_snapshot_from_session(session)

    def live_evaluation_active(self) -> bool:
        with self._live_evaluation_lock:
            return bool(self._live_evaluation and self._live_evaluation.get("active"))

    def list_recognition_labels(
        self,
        *,
        include_negative: bool = False,
        include_no_command: bool = True,
        include_model_classes: bool = False,
    ) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            clean = str(value or "").strip()
            key = clean.lower()
            if clean and key not in seen:
                if (
                    not include_negative
                    and not self._is_no_command_label(clean)
                    and self._is_negative_label(clean)
                ):
                    return
                labels.append(clean)
                seen.add(key)

        if include_model_classes:
            for path in (
                self._dynamic_classes_path(),
                self._configured_classes_path(),
            ):
                try:
                    if path.exists():
                        data = json.loads(path.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            for item in data:
                                add(str(item))
                except Exception as e:
                    print(f"[w] list_recognition_labels {path}: {e}", flush=True)

        for row in self.list_recorded_gestures():
            add(str(row.get("label") or ""))

        if include_no_command:
            add(LIVE_EVAL_NO_COMMAND_LABEL)

        return sorted(labels, key=lambda x: (x.lower() != LIVE_EVAL_NO_COMMAND_LABEL, x.lower()))

    def start_live_evaluation(
        self,
        expected_label: str,
        *,
        attempts: int = LIVE_EVAL_DEFAULT_ATTEMPTS,
        timeout_seconds: float = LIVE_EVAL_DEFAULT_TIMEOUT_SECONDS,
        min_confidence: float = LIVE_EVAL_DEFAULT_MIN_CONFIDENCE,
    ) -> bool:
        clean = str(expected_label or "").strip()
        if not clean:
            return False

        target_attempts = max(1, min(int(attempts or LIVE_EVAL_DEFAULT_ATTEMPTS), 100))
        timeout = max(0.0, min(float(timeout_seconds), 15.0))
        threshold = max(0.0, min(float(min_confidence), 1.0))
        now = time.monotonic()
        session = {
            "active": True,
            "expected_label": clean,
            "target_attempts": target_attempts,
            "timeout_seconds": timeout,
            "min_confidence": threshold,
            "recognition_model_mode": self.recognition_model_mode,
            "dynamic_model_profile": self.dynamic_model_profile,
            "static_rejection_method": self.static_rejection_method,
            "cooldown_seconds": LIVE_EVAL_ATTEMPT_COOLDOWN_SECONDS,
            "attempt_started_at": now,
            "next_ready_at": now,
            "total": 0,
            "correct": 0,
            "wrong": 0,
            "missed": 0,
            "attempts": [],
            "last_prediction": "",
            "last_confidence": 0.0,
            "last_result": "",
            "message": f"Тест {clean}: попытка 1/{target_attempts}",
            "started_at": time.time(),
            "auto_execute_was_enabled": bool(self._auto_execute_on_gesture),
        }
        with self._live_evaluation_lock:
            self._live_evaluation = session
            self._last_live_evaluation_snapshot = None

        if self._auto_execute_on_gesture:
            self._auto_execute_on_gesture = False
        self._reset_gesture_confirmation(immediate_ui=True)
        self._last_label = ""
        self._emit_live_evaluation_changed(session, message=session["message"])
        self._set_status(f"Live eval: {clean} 1/{target_attempts}")
        self._ensure_embedded_recognition_for_live_controls()
        return True

    def cancel_live_evaluation(self) -> bool:
        with self._live_evaluation_lock:
            session = self._live_evaluation
        if session is None:
            return False
        self._finish_live_evaluation("stopped")
        return True

    def mark_live_evaluation_missed(self) -> bool:
        with self._live_evaluation_lock:
            session = self._live_evaluation
            if session is None or not bool(session.get("active")):
                return False
            if time.monotonic() < float(session.get("next_ready_at") or 0.0):
                return False
            expected = str(session.get("expected_label") or "")
            result = (
                "correct"
                if self._gesture_type_for_label(expected) == GESTURE_TYPE_NEGATIVE
                else "missed"
            )
            self._record_live_evaluation_attempt(
                session,
                result=result,
                predicted_label="",
                confidence=0.0,
                route_metadata={"route": "none"},
            )
        return True

    def _finish_live_evaluation(self, reason: str = "completed") -> None:
        with self._live_evaluation_lock:
            session = self._live_evaluation
            self._live_evaluation = None
        if session is None:
            return
        session["active"] = False
        session["finished_at"] = time.time()
        session["message"] = self._live_evaluation_finish_message(session, reason)
        if bool(session.get("auto_execute_was_enabled")):
            self._auto_execute_on_gesture = True
        self._append_live_evaluation_jsonl(session, event_type=f"run_{reason}")
        self._log_live_evaluation_mlflow(session, reason=reason)
        self._last_live_evaluation_snapshot = self._live_evaluation_snapshot_from_session(
            session,
            message=session["message"],
        )
        self._emit_live_evaluation_changed(session, message=session["message"])
        if self._status.startswith("Live eval"):
            self._set_status(self._live_recognition_status())

    def _live_evaluation_finish_message(self, session: dict[str, Any], reason: str) -> str:
        total = int(session.get("total") or 0)
        target = int(session.get("target_attempts") or 0)
        correct = int(session.get("correct") or 0)
        wrong = int(session.get("wrong") or 0)
        missed = int(session.get("missed") or 0)
        prefix = "Завершено" if reason == "completed" else "Остановлено"
        return (
            f"{prefix}: {session.get('expected_label')} "
            f"{correct}/{total or target} correct, wrong={wrong}, missed={missed}"
        )

    def _append_live_evaluation_jsonl(
        self,
        payload: dict[str, Any],
        *,
        event_type: str,
    ) -> None:
        try:
            path = self._configured_log_dir() / "live_evaluation.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "event_type": event_type,
                "recorded_at": time.time(),
                "expected_label": payload.get("expected_label"),
                "target_attempts": payload.get("target_attempts"),
                "total": payload.get("total"),
                "correct": payload.get("correct"),
                "wrong": payload.get("wrong"),
                "missed": payload.get("missed"),
                "min_confidence": payload.get("min_confidence"),
                "timeout_seconds": payload.get("timeout_seconds"),
                "recognition_model_mode": payload.get("recognition_model_mode"),
                "dynamic_model_profile": payload.get("dynamic_model_profile"),
                "static_rejection_method": payload.get("static_rejection_method"),
                "attempts": payload.get("attempts", []),
                "route_counts": self._live_evaluation_route_counts(
                    payload.get("attempts", [])
                ),
            }
            if event_type == "attempt":
                row.update(
                    {
                        "attempt": payload.get("attempt"),
                        "expected": payload.get("expected"),
                        "predicted": payload.get("predicted"),
                        "confidence": payload.get("confidence"),
                        "result": payload.get("result"),
                        "elapsed_seconds": payload.get("elapsed_seconds"),
                        "route": payload.get("route"),
                        "selected_reason": payload.get("selected_reason"),
                        "static_label": payload.get("static_label"),
                        "static_confidence": payload.get("static_confidence"),
                        "static_type": payload.get("static_type"),
                        "static_reject_reason": payload.get("static_reject_reason"),
                        "static_decision_source": payload.get(
                            "static_decision_source"
                        ),
                        "static_rejection_method": payload.get(
                            "static_rejection_method"
                        ),
                        "static_model_label": payload.get("static_model_label"),
                        "static_model_confidence": payload.get(
                            "static_model_confidence"
                        ),
                        "static_top2_label": payload.get("static_top2_label"),
                        "static_top2_confidence": payload.get(
                            "static_top2_confidence"
                        ),
                        "static_margin": payload.get("static_margin"),
                        "static_min_margin": payload.get("static_min_margin"),
                        "static_negative_label": payload.get("static_negative_label"),
                        "static_negative_confidence": payload.get(
                            "static_negative_confidence"
                        ),
                        "static_negative_threshold": payload.get(
                            "static_negative_threshold"
                        ),
                        "static_prototype_distance": payload.get(
                            "static_prototype_distance"
                        ),
                        "static_prototype_radius": payload.get(
                            "static_prototype_radius"
                        ),
                        "static_prototype_threshold": payload.get(
                            "static_prototype_threshold"
                        ),
                        "static_verifier_method": payload.get("static_verifier_method"),
                        "static_verifier_label": payload.get("static_verifier_label"),
                        "static_verifier_probability": payload.get(
                            "static_verifier_probability"
                        ),
                        "static_verifier_confidence": payload.get(
                            "static_verifier_confidence"
                        ),
                        "static_verifier_score": payload.get("static_verifier_score"),
                        "static_verifier_distance": payload.get(
                            "static_verifier_distance"
                        ),
                        "static_verifier_threshold": payload.get(
                            "static_verifier_threshold"
                        ),
                        "dynamic_label": payload.get("dynamic_label"),
                        "dynamic_confidence": payload.get("dynamic_confidence"),
                        "dynamic_type": payload.get("dynamic_type"),
                        "dynamic_reject_reason": payload.get("dynamic_reject_reason"),
                        "dynamic_phase": payload.get("dynamic_phase"),
                        "dynamic_end_reason": payload.get("dynamic_end_reason"),
                        "dynamic_segment_frames": payload.get(
                            "dynamic_segment_frames"
                        ),
                        "dynamic_motion_scale": payload.get(
                            "dynamic_motion_scale"
                        ),
                        "dynamic_decision_source": payload.get(
                            "dynamic_decision_source"
                        ),
                        "dynamic_motion_label": payload.get("dynamic_motion_label"),
                        "dynamic_motion_confidence": payload.get(
                            "dynamic_motion_confidence"
                        ),
                        "dynamic_model_label": payload.get("dynamic_model_label"),
                        "dynamic_model_confidence": payload.get(
                            "dynamic_model_confidence"
                        ),
                        "dynamic_model_confidence_for_motion": payload.get(
                            "dynamic_model_confidence_for_motion"
                        ),
                        "dynamic_negative_label": payload.get(
                            "dynamic_negative_label"
                        ),
                        "dynamic_negative_confidence": payload.get(
                            "dynamic_negative_confidence"
                        ),
                        "dynamic_negative_threshold": payload.get(
                            "dynamic_negative_threshold"
                        ),
                        "dynamic_axis": payload.get("dynamic_axis"),
                        "dynamic_direction": payload.get("dynamic_direction"),
                        "dynamic_axis_ratio": payload.get("dynamic_axis_ratio"),
                        "dynamic_straightness": payload.get("dynamic_straightness"),
                    }
                )
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print(f"[w] live evaluation log write failed: {e}", flush=True)

    def _live_evaluation_mlflow_tracking_uri(self) -> str:
        tracking_uri = str(os.getenv("MLFLOW_TRACKING_URI") or "").strip()
        if tracking_uri:
            return tracking_uri
        project_root = Path(__file__).resolve().parents[2]
        return f"sqlite:///{project_root / 'mlflow.db'}"

    def _live_evaluation_metric_suffix(self, value: Any) -> str:
        text = str(value or "unknown").strip().lower()
        chars = [ch if ch.isalnum() else "_" for ch in text]
        suffix = "_".join(part for part in "".join(chars).split("_") if part)
        return suffix or "unknown"

    def _live_evaluation_mlflow_metrics(
        self,
        session: dict[str, Any],
    ) -> dict[str, float]:
        attempts = [
            item
            for item in session.get("attempts", [])
            if isinstance(item, dict)
        ]
        target = max(1, int(session.get("target_attempts") or 1))
        total = max(0, int(session.get("total") or len(attempts) or 0))
        correct = max(0, int(session.get("correct") or 0))
        wrong = max(0, int(session.get("wrong") or 0))
        missed = max(0, int(session.get("missed") or 0))
        expected = str(session.get("expected_label") or "")
        try:
            expected_type = self._gesture_type_for_label(expected)
        except Exception:
            expected_type = ""

        metrics: dict[str, float] = {
            "live_total": float(total),
            "live_target_attempts": float(target),
            "live_correct": float(correct),
            "live_wrong": float(wrong),
            "live_missed": float(missed),
            "live_accuracy": float(correct / total) if total else 0.0,
            "live_recall": float(correct / target),
            "live_completion_rate": float(total / target),
            "live_error_rate": float(wrong / total) if total else 0.0,
            "live_miss_rate": float(missed / target),
        }

        latencies = [
            float(item.get("elapsed_seconds"))
            for item in attempts
            if item.get("elapsed_seconds") is not None
        ]
        latencies = sorted(value for value in latencies if value >= 0.0)
        if latencies:
            p50_index = min(len(latencies) - 1, int(round((len(latencies) - 1) * 0.50)))
            p95_index = min(len(latencies) - 1, int(round((len(latencies) - 1) * 0.95)))
            metrics.update(
                {
                    "live_latency_avg_s": float(sum(latencies) / len(latencies)),
                    "live_latency_p50_s": float(latencies[p50_index]),
                    "live_latency_p95_s": float(latencies[p95_index]),
                }
            )

        route_counts = self._live_evaluation_route_counts(attempts)
        for route, count in route_counts.items():
            metrics[f"live_route_{self._live_evaluation_metric_suffix(route)}_count"] = float(count)

        decision_counts: dict[str, int] = {}
        end_reason_counts: dict[str, int] = {}
        static_decision_counts: dict[str, int] = {}
        static_reject_reason_counts: dict[str, int] = {}
        static_rejection_method_counts: dict[str, int] = {}
        dynamic_prototype_method_counts: dict[str, int] = {}
        dynamic_prototype_reason_counts: dict[str, int] = {}
        static_hijack_count = 0
        static_accept_count = 0
        static_reject_count = 0
        static_false_positive_count = 0
        wrong_dynamic_direction_count = 0
        negative_rejected_count = 0
        for item in attempts:
            route = str(item.get("route") or "none").strip() or "none"
            decision = str(item.get("dynamic_decision_source") or "").strip()
            if decision:
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
            end_reason = str(item.get("dynamic_end_reason") or "").strip()
            if end_reason:
                end_reason_counts[end_reason] = end_reason_counts.get(end_reason, 0) + 1
            if decision == "negative_rejected":
                negative_rejected_count += 1
            prototype_method = str(item.get("dynamic_prototype_method") or "").strip()
            if prototype_method:
                dynamic_prototype_method_counts[prototype_method] = (
                    dynamic_prototype_method_counts.get(prototype_method, 0) + 1
                )
            prototype_reason = str(item.get("dynamic_prototype_reason") or "").strip()
            if prototype_reason:
                dynamic_prototype_reason_counts[prototype_reason] = (
                    dynamic_prototype_reason_counts.get(prototype_reason, 0) + 1
                )
            static_decision = str(item.get("static_decision_source") or "").strip()
            if static_decision:
                static_decision_counts[static_decision] = (
                    static_decision_counts.get(static_decision, 0) + 1
                )
            static_reject_reason = str(
                item.get("static_reject_reason") or ""
            ).strip()
            if static_reject_reason:
                static_reject_reason_counts[static_reject_reason] = (
                    static_reject_reason_counts.get(static_reject_reason, 0) + 1
                )
            static_method = str(item.get("static_rejection_method") or "").strip()
            if static_method:
                static_rejection_method_counts[static_method] = (
                    static_rejection_method_counts.get(static_method, 0) + 1
                )
            if route == "static":
                static_accept_count += 1
            if route == "none" or static_reject_reason:
                static_reject_count += 1
            if expected_type == GESTURE_TYPE_NEGATIVE and route == "static":
                static_false_positive_count += 1
            if expected_type == GESTURE_TYPE_DYNAMIC and route == "static":
                static_hijack_count += 1
            predicted = str(item.get("predicted") or "")
            motion_label = str(item.get("dynamic_motion_label") or "")
            if (
                expected_type == GESTURE_TYPE_DYNAMIC
                and route == "dynamic"
                and str(item.get("result") or "") == "wrong"
                and (
                    predicted.startswith("swipe_")
                    or motion_label.startswith("swipe_")
                )
            ):
                wrong_dynamic_direction_count += 1

        for source, count in decision_counts.items():
            suffix = self._live_evaluation_metric_suffix(source)
            metrics[f"live_decision_{suffix}_count"] = float(count)
        for reason, count in end_reason_counts.items():
            suffix = self._live_evaluation_metric_suffix(reason)
            metrics[f"live_end_reason_{suffix}_count"] = float(count)
        for source, count in static_decision_counts.items():
            suffix = self._live_evaluation_metric_suffix(source)
            metrics[f"live_static_decision_{suffix}_count"] = float(count)
        for reason, count in static_reject_reason_counts.items():
            suffix = self._live_evaluation_metric_suffix(reason)
            metrics[f"live_static_rejection_reason_{suffix}_count"] = float(count)
        for method, count in static_rejection_method_counts.items():
            suffix = self._live_evaluation_metric_suffix(method)
            metrics[f"live_static_rejection_method_{suffix}_count"] = float(count)
        for method, count in dynamic_prototype_method_counts.items():
            suffix = self._live_evaluation_metric_suffix(method)
            metrics[f"live_dynamic_prototype_method_{suffix}_count"] = float(count)
        for reason, count in dynamic_prototype_reason_counts.items():
            suffix = self._live_evaluation_metric_suffix(reason)
            metrics[f"live_dynamic_prototype_reason_{suffix}_count"] = float(count)

        metrics["live_dynamic_recall"] = (
            float(correct / target) if expected_type == GESTURE_TYPE_DYNAMIC else 0.0
        )
        metrics["live_static_hijack_count"] = float(static_hijack_count)
        metrics["live_static_hijack_rate"] = (
            float(static_hijack_count / total)
            if expected_type == GESTURE_TYPE_DYNAMIC and total
            else 0.0
        )
        metrics["live_static_accept_count"] = float(static_accept_count)
        metrics["live_static_accept_rate"] = (
            float(static_accept_count / total) if total else 0.0
        )
        metrics["live_static_reject_count"] = float(static_reject_count)
        metrics["live_static_reject_rate"] = (
            float(static_reject_count / total) if total else 0.0
        )
        metrics["live_static_false_positive_count"] = float(
            static_false_positive_count
        )
        metrics["live_static_false_positive_rate"] = (
            float(static_false_positive_count / total)
            if expected_type == GESTURE_TYPE_NEGATIVE and total
            else 0.0
        )
        metrics["live_wrong_dynamic_direction_count"] = float(
            wrong_dynamic_direction_count
        )
        metrics["live_wrong_dynamic_direction_rate"] = (
            float(wrong_dynamic_direction_count / total)
            if expected_type == GESTURE_TYPE_DYNAMIC and total
            else 0.0
        )
        metrics["live_negative_false_positive_count"] = (
            float(wrong) if expected_type == GESTURE_TYPE_NEGATIVE else 0.0
        )
        metrics["live_negative_false_positive_rate"] = (
            float(wrong / total)
            if expected_type == GESTURE_TYPE_NEGATIVE and total
            else 0.0
        )
        metrics["live_negative_rejected_count"] = float(negative_rejected_count)
        return metrics

    def _live_evaluation_runtime_rows(
        self,
        session: dict[str, Any],
    ) -> list[dict[str, Any]]:
        started_at = float(session.get("started_at") or 0.0)
        finished_at = float(session.get("finished_at") or time.time())
        mode = str(session.get("recognition_model_mode") or "")
        profile = str(session.get("dynamic_model_profile") or "")
        path = self._configured_log_dir() / "runtime_performance.jsonl"
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                recorded_at = float(row.get("recorded_at") or 0.0)
                if started_at and recorded_at < started_at - 5.0:
                    continue
                if finished_at and recorded_at > finished_at + 5.0:
                    continue
                if mode and str(row.get("recognition_model_mode") or "") != mode:
                    continue
                if profile and str(row.get("dynamic_model_profile") or "") != profile:
                    continue
                rows.append(row)
        except Exception as e:
            print(f"[w] runtime rows read failed: {e}", flush=True)
            return []
        return rows[-24:]

    def _live_evaluation_runtime_metrics(
        self,
        rows: list[dict[str, Any]],
    ) -> dict[str, float]:
        if not rows:
            return {}

        def _float(row: dict[str, Any], key: str) -> float | None:
            try:
                value = row.get(key)
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        def _weighted_mean(key: str) -> float:
            weighted_sum = 0.0
            weight_total = 0.0
            fallback_values: list[float] = []
            for row in rows:
                value = _float(row, key)
                if value is None:
                    continue
                fallback_values.append(value)
                weight = max(1.0, float(row.get("samples") or 1.0))
                weighted_sum += value * weight
                weight_total += weight
            if weight_total > 0.0:
                return weighted_sum / weight_total
            return sum(fallback_values) / len(fallback_values) if fallback_values else 0.0

        samples_total = sum(max(0, int(row.get("samples") or 0)) for row in rows)
        p95_values = [
            value
            for value in (_float(row, "inference_ms_p95") for row in rows)
            if value is not None
        ]
        return {
            "system_runtime_windows": float(len(rows)),
            "system_runtime_samples_total": float(samples_total),
            "system_runtime_inference_ms_avg": _weighted_mean("inference_ms_avg"),
            "system_runtime_inference_ms_p95_max": max(p95_values) if p95_values else 0.0,
            "system_runtime_detection_ms_avg": _weighted_mean("detection_ms_avg"),
            "system_runtime_fps_capacity_avg": _weighted_mean("inference_fps_capacity"),
            "system_runtime_shared_detection_rate_avg": _weighted_mean(
                "shared_detection_rate"
            ),
            "system_mediapipe_detection_ms_avg": _weighted_mean(
                "mediapipe_detection_ms_avg"
            ),
            "system_mediapipe_real_timestamp_rate_avg": _weighted_mean(
                "mediapipe_real_timestamp_rate"
            ),
            "system_hand_detected_rate_avg": _weighted_mean("hand_detected_rate"),
            "system_hand_count_avg": _weighted_mean("hand_count_avg"),
            "system_handedness_score_avg": _weighted_mean(
                "handedness_score_max_avg"
            ),
            "system_landmark_z_available_rate_avg": _weighted_mean(
                "landmark_z_available_rate"
            ),
            "system_world_landmarks_available_rate_avg": _weighted_mean(
                "world_landmarks_available_rate"
            ),
            "system_landmark_z_range_avg": _weighted_mean("landmark_z_range_avg"),
            "system_world_z_range_avg": _weighted_mean("world_z_range_avg"),
            "system_hand_bbox_area_avg": _weighted_mean("hand_bbox_area_avg"),
            "system_hand_bbox_diag_avg": _weighted_mean("hand_bbox_diag_avg"),
            "system_primary_wrist_step_avg": _weighted_mean(
                "primary_wrist_step_avg"
            ),
            "system_hand_lost_streak_max": max(
                (
                    float(row.get("hand_lost_streak_max") or 0.0)
                    for row in rows
                ),
                default=0.0,
            ),
        }

    def _live_evaluation_attempt_metrics(
        self,
        attempt: dict[str, Any],
        *,
        expected_type: str,
    ) -> dict[str, float]:
        def _float(value: Any) -> float | None:
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        result = str(attempt.get("result") or "")
        route = str(attempt.get("route") or "none")
        predicted = str(attempt.get("predicted") or "")
        metrics = {
            "attempt_is_correct": 1.0 if result == "correct" else 0.0,
            "attempt_is_wrong": 1.0 if result == "wrong" else 0.0,
            "attempt_is_missed": 1.0 if result == "missed" else 0.0,
            "attempt_route_dynamic": 1.0 if route == "dynamic" else 0.0,
            "attempt_route_static": 1.0 if route == "static" else 0.0,
            "attempt_route_none": 1.0 if route in ("", "none") else 0.0,
            "attempt_confidence": float(attempt.get("confidence") or 0.0),
            "attempt_wrong_direction": (
                1.0
                if expected_type == GESTURE_TYPE_DYNAMIC
                and route == "dynamic"
                and result == "wrong"
                and predicted.startswith("swipe_")
                else 0.0
            ),
            "attempt_static_hijack": (
                1.0
                if expected_type == GESTURE_TYPE_DYNAMIC and route == "static"
                else 0.0
            ),
        }
        optional_fields = {
            "attempt_latency_s": "elapsed_seconds",
            "attempt_static_margin": "static_margin",
            "attempt_static_negative_confidence": "static_negative_confidence",
            "attempt_static_prototype_distance": "static_prototype_distance",
            "attempt_static_prototype_threshold": "static_prototype_threshold",
            "attempt_static_verifier_probability": "static_verifier_probability",
            "attempt_static_verifier_confidence": "static_verifier_confidence",
            "attempt_static_verifier_score": "static_verifier_score",
            "attempt_static_verifier_distance": "static_verifier_distance",
            "attempt_static_verifier_threshold": "static_verifier_threshold",
            "attempt_axis_ratio": "dynamic_axis_ratio",
            "attempt_straightness": "dynamic_straightness",
            "attempt_motion_scale": "dynamic_motion_scale",
            "attempt_segment_frames": "dynamic_segment_frames",
            "attempt_motion_confidence": "dynamic_motion_confidence",
            "attempt_model_confidence": "dynamic_model_confidence",
            "attempt_dynamic_prototype_confidence": "dynamic_prototype_confidence",
            "attempt_dynamic_prototype_distance": "dynamic_prototype_distance",
            "attempt_dynamic_prototype_threshold": "dynamic_prototype_threshold",
            "attempt_dynamic_prototype_margin": "dynamic_prototype_margin",
            "attempt_mediapipe_min_detection_confidence": (
                "mediapipe_min_detection_confidence"
            ),
            "attempt_mediapipe_min_presence_confidence": (
                "mediapipe_min_presence_confidence"
            ),
            "attempt_mediapipe_min_tracking_confidence": (
                "mediapipe_min_tracking_confidence"
            ),
            "attempt_mediapipe_smoothing_alpha": "mediapipe_smoothing_alpha",
            "attempt_mediapipe_detection_ms": "mediapipe_detection_ms",
            "attempt_hand_detected": "hand_detected",
            "attempt_hand_count": "hand_count",
            "attempt_handedness_score_max": "handedness_score_max",
            "attempt_landmark_z_available": "landmark_z_available",
            "attempt_world_landmarks_available": "world_landmarks_available",
            "attempt_landmark_z_range": "landmark_z_range",
            "attempt_world_z_range": "world_z_range",
            "attempt_hand_bbox_area": "hand_bbox_area",
            "attempt_hand_bbox_diag": "hand_bbox_diag",
            "attempt_primary_wrist_step": "primary_wrist_step",
            "attempt_hand_lost_streak": "hand_lost_streak",
            "attempt_hand_lost_grace_frames": "hand_lost_grace_frames",
        }
        for metric_name, field_name in optional_fields.items():
            value = _float(attempt.get(field_name))
            if value is not None:
                metrics[metric_name] = value
        return metrics

    def _write_live_evaluation_artifact_bundle(
        self,
        artifact_dir: Path,
        *,
        payload: dict[str, Any],
        metrics: dict[str, float],
        runtime_rows: list[dict[str, Any]],
    ) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        charts_dir = artifact_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        attempts = [
            item
            for item in (payload.get("session") or {}).get("attempts", [])
            if isinstance(item, dict)
        ]

        (artifact_dir / "live_evaluation_run.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (artifact_dir / "metrics.csv").write_text(
            self._live_evaluation_metrics_csv(metrics),
            encoding="utf-8",
        )
        (artifact_dir / "attempts.csv").write_text(
            self._live_evaluation_attempts_csv(attempts),
            encoding="utf-8",
        )
        if runtime_rows:
            (artifact_dir / "runtime_performance.json").write_text(
                json.dumps(runtime_rows, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        (charts_dir / "quality.svg").write_text(
            self._render_live_bar_svg(
                "Live quality metrics",
                [
                    ("accuracy", metrics.get("live_accuracy", 0.0), "good"),
                    ("recall", metrics.get("live_recall", 0.0), "good"),
                    (
                        "wrong direction",
                        metrics.get("live_wrong_dynamic_direction_rate", 0.0),
                        "bad",
                    ),
                    ("static hijack", metrics.get("live_static_hijack_rate", 0.0), "bad"),
                    (
                        "static false positive",
                        metrics.get("live_static_false_positive_rate", 0.0),
                        "bad",
                    ),
                    (
                        "static reject",
                        metrics.get("live_static_reject_rate", 0.0),
                        "warn",
                    ),
                    ("miss rate", metrics.get("live_miss_rate", 0.0), "bad"),
                ],
                max_value=1.0,
                value_suffix="",
            ),
            encoding="utf-8",
        )
        (charts_dir / "outcomes.svg").write_text(
            self._render_live_bar_svg(
                "Attempt outcomes",
                [
                    ("correct", metrics.get("live_correct", 0.0), "good"),
                    ("wrong", metrics.get("live_wrong", 0.0), "bad"),
                    ("missed", metrics.get("live_missed", 0.0), "warn"),
                ],
            ),
            encoding="utf-8",
        )
        route_items = [
            (
                key.replace("live_route_", "").replace("_count", ""),
                value,
                "good" if "dynamic" in key else "warn",
            )
            for key, value in sorted(metrics.items())
            if key.startswith("live_route_") and key.endswith("_count")
        ]
        (charts_dir / "routes.svg").write_text(
            self._render_live_bar_svg("Router decisions", route_items or [("none", 0.0, "warn")]),
            encoding="utf-8",
        )
        static_rejection_items = [
            (
                "method:"
                + key.replace("live_static_rejection_method_", "").replace("_count", ""),
                value,
                "good",
            )
            for key, value in sorted(metrics.items())
            if key.startswith("live_static_rejection_method_") and key.endswith("_count")
        ]
        static_rejection_items.extend(
            [
                (
                    "decision:"
                    + key.replace("live_static_decision_", "").replace("_count", ""),
                    value,
                    "warn" if "rejected" in key else "good",
                )
                for key, value in sorted(metrics.items())
                if key.startswith("live_static_decision_") and key.endswith("_count")
            ]
        )
        static_rejection_items.extend(
            [
                (
                    "reason:"
                    + key.replace("live_static_rejection_reason_", "").replace("_count", ""),
                    value,
                    "bad" if "negative" in key or "low" in key else "warn",
                )
                for key, value in sorted(metrics.items())
                if key.startswith("live_static_rejection_reason_") and key.endswith("_count")
            ]
        )
        (charts_dir / "static_rejection.svg").write_text(
            self._render_live_bar_svg(
                "Static rejection methods and reasons",
                static_rejection_items or [("none", 0.0, "warn")],
            ),
            encoding="utf-8",
        )
        verifier_items = self._live_static_verifier_signal_items(attempts)
        (charts_dir / "static_verifier_signals.svg").write_text(
            self._render_live_bar_svg(
                "Static verifier signals",
                verifier_items or [("none", 0.0, "warn")],
            ),
            encoding="utf-8",
        )
        end_reason_items = [
            (
                key.replace("live_end_reason_", "").replace("_count", ""),
                value,
                "good",
            )
            for key, value in sorted(metrics.items())
            if key.startswith("live_end_reason_") and key.endswith("_count")
        ]
        (charts_dir / "end_reasons.svg").write_text(
            self._render_live_bar_svg(
                "Dynamic end reasons",
                end_reason_items or [("none", 0.0, "warn")],
            ),
            encoding="utf-8",
        )
        (charts_dir / "attempt_timeline.svg").write_text(
            self._render_live_attempt_timeline_svg(attempts),
            encoding="utf-8",
        )
        runtime_items = [
            (
                "inference avg ms",
                metrics.get("system_runtime_inference_ms_avg", 0.0),
                "good",
            ),
            (
                "inference p95 max ms",
                metrics.get("system_runtime_inference_ms_p95_max", 0.0),
                "warn",
            ),
            (
                "detection avg ms",
                metrics.get("system_runtime_detection_ms_avg", 0.0),
                "good",
            ),
            (
                "fps capacity avg",
                metrics.get("system_runtime_fps_capacity_avg", 0.0),
                "good",
            ),
        ]
        (charts_dir / "runtime.svg").write_text(
            self._render_live_bar_svg("Runtime system metrics", runtime_items),
            encoding="utf-8",
        )
        mediapipe_items = [
            (
                "real timestamp rate",
                metrics.get("system_mediapipe_real_timestamp_rate_avg", 0.0),
                "good",
            ),
            (
                "hand detected rate",
                metrics.get("system_hand_detected_rate_avg", 0.0),
                "good",
            ),
            (
                "world landmarks rate",
                metrics.get("system_world_landmarks_available_rate_avg", 0.0),
                "good",
            ),
            (
                "z landmarks rate",
                metrics.get("system_landmark_z_available_rate_avg", 0.0),
                "good",
            ),
            (
                "hand lost streak max",
                metrics.get("system_hand_lost_streak_max", 0.0),
                "warn",
            ),
            (
                "wrist step avg",
                metrics.get("system_primary_wrist_step_avg", 0.0),
                "warn",
            ),
            (
                "bbox diag avg",
                metrics.get("system_hand_bbox_diag_avg", 0.0),
                "good",
            ),
        ]
        (charts_dir / "mediapipe_quality.svg").write_text(
            self._render_live_bar_svg(
                "MediaPipe landmark quality",
                mediapipe_items,
            ),
            encoding="utf-8",
        )
        (artifact_dir / "index.html").write_text(
            self._render_live_evaluation_artifact_html(
                payload=payload,
                metrics=metrics,
                runtime_rows=runtime_rows,
            ),
            encoding="utf-8",
        )

    def _live_static_verifier_signal_items(
        self,
        attempts: list[dict[str, Any]],
    ) -> list[tuple[str, float, str]]:
        def _values(field: str) -> list[float]:
            values: list[float] = []
            for attempt in attempts:
                try:
                    value = attempt.get(field)
                    if value is not None:
                        values.append(float(value))
                except (TypeError, ValueError):
                    continue
            return values

        items: list[tuple[str, float, str]] = []
        for label, field, kind in (
            ("avg margin", "static_margin", "good"),
            ("avg negative prob", "static_negative_confidence", "bad"),
            ("avg prototype dist", "static_prototype_distance", "warn"),
            ("avg prototype threshold", "static_prototype_threshold", "good"),
            ("avg verifier prob", "static_verifier_probability", "good"),
            ("avg verifier confidence", "static_verifier_confidence", "good"),
            ("avg verifier score", "static_verifier_score", "good"),
            ("avg verifier dist", "static_verifier_distance", "warn"),
            ("avg verifier threshold", "static_verifier_threshold", "good"),
        ):
            values = _values(field)
            if values:
                items.append((label, float(sum(values) / len(values)), kind))
        return items

    def _live_evaluation_metrics_csv(self, metrics: dict[str, float]) -> str:
        lines = ["metric,value"]
        for key, value in sorted(metrics.items()):
            lines.append(f"{key},{value}")
        return "\n".join(lines) + "\n"

    def _live_evaluation_attempts_csv(self, attempts: list[dict[str, Any]]) -> str:
        if not attempts:
            return "attempt,expected,predicted,result,route\n"
        fieldnames = sorted(
            {
                key
                for attempt in attempts
                for key in attempt.keys()
            }
        )
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for attempt in attempts:
                writer.writerow(attempt)
            fh.seek(0)
            return fh.read()

    def _render_live_bar_svg(
        self,
        title: str,
        items: list[tuple[str, float, str]],
        *,
        max_value: float | None = None,
        value_suffix: str = "",
    ) -> str:
        width = 920
        row_height = 48
        top = 74
        height = max(180, top + row_height * max(1, len(items)) + 28)
        max_item_value = max([abs(float(value)) for _label, value, _kind in items] + [1.0])
        scale_max = float(max_value if max_value is not None else max_item_value)
        scale_max = max(scale_max, 1e-9)
        colors = {
            "good": "#18c7b8",
            "bad": "#ff5b5f",
            "warn": "#f7b955",
        }
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" rx="18" fill="#101720"/>',
            f'<text x="28" y="42" fill="#f5f7fb" font-size="24" font-family="Inter, Arial" font-weight="700">{html.escape(title)}</text>',
        ]
        x0 = 220
        bar_max = width - x0 - 110
        for index, (label, value, kind) in enumerate(items):
            y = top + index * row_height
            bar_width = max(0.0, min(1.0, float(value) / scale_max)) * bar_max
            color = colors.get(kind, "#77a8ff")
            display_value = f"{float(value):.3g}{value_suffix}"
            parts.extend(
                [
                    f'<text x="28" y="{y + 25}" fill="#aab7c4" font-size="17" font-family="Inter, Arial">{html.escape(str(label))}</text>',
                    f'<rect x="{x0}" y="{y}" width="{bar_max}" height="28" rx="8" fill="#1c2733"/>',
                    f'<rect x="{x0}" y="{y}" width="{bar_width:.1f}" height="28" rx="8" fill="{color}"/>',
                    f'<text x="{x0 + bar_max + 18}" y="{y + 21}" fill="#f5f7fb" font-size="16" font-family="Inter, Arial" font-weight="700">{html.escape(display_value)}</text>',
                ]
            )
        parts.append("</svg>")
        return "\n".join(parts)

    def _render_live_attempt_timeline_svg(self, attempts: list[dict[str, Any]]) -> str:
        count = max(1, len(attempts))
        width = max(920, 92 * count + 80)
        height = 250
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" rx="18" fill="#101720"/>',
            '<text x="28" y="42" fill="#f5f7fb" font-size="24" font-family="Inter, Arial" font-weight="700">Attempt timeline</text>',
            f'<line x1="60" y1="110" x2="{width - 60}" y2="110" stroke="#2d3a46" stroke-width="4"/>',
        ]
        colors = {"correct": "#18c7b8", "wrong": "#ff5b5f", "missed": "#f7b955"}
        for index, attempt in enumerate(attempts):
            x = 60 + index * ((width - 120) / max(1, count - 1))
            result = str(attempt.get("result") or "")
            predicted = str(attempt.get("predicted") or "miss")
            color = colors.get(result, "#8aa0b4")
            parts.extend(
                [
                    f'<circle cx="{x:.1f}" cy="110" r="18" fill="{color}"/>',
                    f'<text x="{x:.1f}" y="116" text-anchor="middle" fill="#101720" font-size="14" font-family="Inter, Arial" font-weight="800">{index + 1}</text>',
                    f'<text x="{x:.1f}" y="158" text-anchor="middle" fill="#d7e1ea" font-size="13" font-family="Inter, Arial">{html.escape(predicted)}</text>',
                    f'<text x="{x:.1f}" y="182" text-anchor="middle" fill="#8fa2b4" font-size="12" font-family="Inter, Arial">{html.escape(result)}</text>',
                ]
            )
        parts.append("</svg>")
        return "\n".join(parts)

    def _render_live_evaluation_artifact_html(
        self,
        *,
        payload: dict[str, Any],
        metrics: dict[str, float],
        runtime_rows: list[dict[str, Any]],
    ) -> str:
        session = payload.get("session") or {}
        attempts = [
            item
            for item in session.get("attempts", [])
            if isinstance(item, dict)
        ]
        expected = html.escape(str(session.get("expected_label") or "unknown"))
        score = (
            f"{int(session.get('correct') or 0)}/"
            f"{int(session.get('total') or 0)}"
        )

        def metric_card(key: str, label: str) -> str:
            value = metrics.get(key, 0.0)
            return (
                '<div class="card">'
                f'<div class="muted">{html.escape(label)}</div>'
                f'<div class="value">{value:.3g}</div>'
                f'<div class="key">{html.escape(key)}</div>'
                '</div>'
            )

        attempts_rows = []
        for attempt in attempts:
            attempts_rows.append(
                "<tr>"
                f"<td>{html.escape(str(attempt.get('attempt') or ''))}</td>"
                f"<td>{html.escape(str(attempt.get('result') or ''))}</td>"
                f"<td>{html.escape(str(attempt.get('predicted') or ''))}</td>"
                f"<td>{html.escape(str(attempt.get('route') or ''))}</td>"
                f"<td>{html.escape(str(attempt.get('static_rejection_method') or ''))}</td>"
                f"<td>{html.escape(str(attempt.get('static_reject_reason') or ''))}</td>"
                f"<td>{html.escape(str(attempt.get('static_verifier_probability') or attempt.get('static_verifier_confidence') or attempt.get('static_verifier_distance') or ''))}</td>"
                f"<td>{html.escape(str(attempt.get('dynamic_end_reason') or ''))}</td>"
                f"<td>{html.escape(str(attempt.get('dynamic_axis') or ''))}</td>"
                f"<td>{html.escape(str(attempt.get('dynamic_direction') or ''))}</td>"
                f"<td>{html.escape(str(attempt.get('dynamic_straightness') or ''))}</td>"
                f"<td>{html.escape(str(attempt.get('mediapipe_profile') or ''))}</td>"
                f"<td>{html.escape(str(attempt.get('mediapipe_timestamp_source') or ''))}</td>"
                f"<td>{html.escape(str(attempt.get('hand_lost_streak') or ''))}</td>"
                "</tr>"
            )

        runtime_note = (
            f"{len(runtime_rows)} runtime windows captured"
            if runtime_rows
            else "No runtime windows matched this live run"
        )
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GestureBind live evaluation - {expected}</title>
  <style>
    body {{ margin: 0; padding: 32px; background: #0d1218; color: #f5f7fb; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    h2 {{ margin: 32px 0 16px; font-size: 22px; }}
    .muted {{ color: #94a6b8; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 24px 0; }}
    .card {{ background: #131c25; border: 1px solid #263544; border-radius: 14px; padding: 18px; }}
    .value {{ margin-top: 8px; font-size: 32px; font-weight: 800; color: #18c7b8; }}
    .key {{ margin-top: 8px; font-size: 12px; color: #6f8294; }}
    img {{ width: 100%; max-width: 980px; display: block; margin: 16px 0; }}
    table {{ width: 100%; border-collapse: collapse; background: #131c25; border-radius: 14px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #263544; text-align: left; font-size: 14px; }}
    th {{ color: #9fcaef; background: #182431; }}
  </style>
</head>
<body>
  <h1>Live evaluation: {expected}</h1>
  <div class="muted">Score {html.escape(score)} · {html.escape(str(session.get("recognition_model_mode") or ""))} / {html.escape(str(session.get("dynamic_model_profile") or ""))} · {html.escape(runtime_note)}</div>
  <div class="grid">
    {metric_card("live_accuracy", "Accuracy")}
    {metric_card("live_recall", "Recall")}
    {metric_card("live_wrong_dynamic_direction_rate", "Wrong direction")}
    {metric_card("live_static_hijack_rate", "Static hijack")}
    {metric_card("live_static_false_positive_rate", "Static false positive")}
    {metric_card("live_static_reject_rate", "Static reject")}
    {metric_card("live_latency_avg_s", "Avg latency, s")}
    {metric_card("system_runtime_inference_ms_avg", "Runtime inference, ms")}
    {metric_card("system_hand_detected_rate_avg", "Hand detected")}
    {metric_card("system_world_landmarks_available_rate_avg", "World landmarks")}
    {metric_card("system_hand_lost_streak_max", "Hand lost streak")}
  </div>
	  <h2>Charts</h2>
	  <img src="charts/quality.svg" alt="quality metrics">
	  <img src="charts/attempt_timeline.svg" alt="attempt timeline">
	  <img src="charts/routes.svg" alt="routes">
	  <img src="charts/static_rejection.svg" alt="static rejection">
	  <img src="charts/static_verifier_signals.svg" alt="static verifier signals">
	  <img src="charts/end_reasons.svg" alt="end reasons">
	  <img src="charts/runtime.svg" alt="runtime metrics">
	  <img src="charts/mediapipe_quality.svg" alt="mediapipe quality">
	  <h2>Attempts</h2>
	  <table>
	    <thead><tr><th>#</th><th>Result</th><th>Predicted</th><th>Route</th><th>Static method</th><th>Static reject</th><th>Verifier signal</th><th>End</th><th>Axis</th><th>Direction</th><th>Straightness</th><th>MP profile</th><th>Timestamp</th><th>Lost</th></tr></thead>
	    <tbody>{''.join(attempts_rows)}</tbody>
	  </table>
</body>
</html>
"""

    def _log_live_evaluation_mlflow(
        self,
        session: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        experiment = str(
            os.getenv("GESTUREBIND_MLFLOW_EXPERIMENT")
            or os.getenv("GESTUREFLOW_MLFLOW_EXPERIMENT")
            or "GestureBind"
        ).strip()
        if not experiment:
            return
        try:
            import mlflow
        except Exception as e:
            print(f"[w] MLflow недоступен, live tracking пропущен: {e}", flush=True)
            return

        try:
            tracking_uri = self._live_evaluation_mlflow_tracking_uri()
            runtime_rows = self._live_evaluation_runtime_rows(session)
            metrics = self._live_evaluation_mlflow_metrics(session)
            metrics.update(self._live_evaluation_runtime_metrics(runtime_rows))
            expected = str(session.get("expected_label") or "unknown")
            mode = str(session.get("recognition_model_mode") or self.recognition_model_mode)
            profile = str(session.get("dynamic_model_profile") or self.dynamic_model_profile)
            static_rejection_method = str(
                session.get("static_rejection_method") or self.static_rejection_method
            )
            run_name = (
                f"live-{self._live_evaluation_metric_suffix(expected)}-"
                f"{self._live_evaluation_metric_suffix(mode)}-"
                f"{self._live_evaluation_metric_suffix(profile)}-"
                f"{self._live_evaluation_metric_suffix(static_rejection_method)}"
            )
            params = {
                "expected_label": expected,
                "target_attempts": int(session.get("target_attempts") or 0),
                "min_confidence": float(session.get("min_confidence") or 0.0),
                "timeout_seconds": float(session.get("timeout_seconds") or 0.0),
                "recognition_model_mode": mode,
                "dynamic_model_profile": profile,
                "static_rejection_method": static_rejection_method,
                "finish_reason": str(reason or ""),
            }
            try:
                params["expected_type"] = self._gesture_type_for_label(expected)
            except Exception:
                params["expected_type"] = ""
            payload = {
                "session": {
                    "expected_label": expected,
                    "target_attempts": session.get("target_attempts"),
                    "total": session.get("total"),
                    "correct": session.get("correct"),
                    "wrong": session.get("wrong"),
                    "missed": session.get("missed"),
                    "min_confidence": session.get("min_confidence"),
                    "timeout_seconds": session.get("timeout_seconds"),
                    "recognition_model_mode": mode,
                    "dynamic_model_profile": profile,
                    "static_rejection_method": static_rejection_method,
                    "started_at": session.get("started_at"),
                    "finished_at": session.get("finished_at"),
                    "attempts": session.get("attempts", []),
                    "route_counts": self._live_evaluation_route_counts(
                        session.get("attempts", [])
                    ),
                },
                "metrics": metrics,
                "runtime_performance": runtime_rows,
            }

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)
            with mlflow.start_run(run_name=run_name, log_system_metrics=True):
                mlflow.set_tags(
                    {
                        "run_kind": "live_evaluation",
                        "source": "gesturebind_flet",
                        "finish_reason": str(reason or ""),
                        "artifact_bundle": "live_evaluation/index.html",
                    }
                )
                mlflow.log_params(params)
                try:
                    expected_type = str(params.get("expected_type") or "")
                    for attempt in session.get("attempts", []):
                        if not isinstance(attempt, dict):
                            continue
                        step = int(attempt.get("attempt") or 0)
                        if step <= 0:
                            continue
                        mlflow.log_metrics(
                            self._live_evaluation_attempt_metrics(
                                attempt,
                                expected_type=expected_type,
                            ),
                            step=step,
                        )
                except Exception as e:
                    print(f"[w] MLflow attempt metrics failed: {e}", flush=True)
                mlflow.log_metrics(metrics)
                mlflow.log_dict(payload, "live_evaluation_run.json")
                with tempfile.TemporaryDirectory() as tmpdir:
                    artifact_dir = Path(tmpdir)
                    self._write_live_evaluation_artifact_bundle(
                        artifact_dir,
                        payload=payload,
                        metrics=metrics,
                        runtime_rows=runtime_rows,
                    )
                    mlflow.log_artifacts(
                        str(artifact_dir),
                        artifact_path="live_evaluation",
                    )
            print(
                f"[✓] MLflow live run logged: expected={expected!r}, "
                f"uri={tracking_uri}",
                flush=True,
            )
        except Exception as e:
            print(f"[w] MLflow live tracking failed: {e}", flush=True)

    def _live_evaluation_route_fields(
        self,
        route_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(route_metadata, dict):
            route_metadata = {}
        route = str(route_metadata.get("route") or "").strip()
        if not route:
            return {}

        def _float_or_none(value: Any) -> float | None:
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        return {
            "route": route,
            "selected_reason": str(route_metadata.get("selected_reason") or ""),
            "intent_gate_enabled": bool(route_metadata.get("intent_gate_enabled")),
            "intent_gate_label": str(route_metadata.get("intent_gate_label") or ""),
            "intent_gate_confidence": _float_or_none(
                route_metadata.get("intent_gate_confidence")
            ),
            "intent_gate_margin": _float_or_none(
                route_metadata.get("intent_gate_margin")
            ),
            "intent_gate_threshold": _float_or_none(
                route_metadata.get("intent_gate_threshold")
            ),
            "intent_gate_accepted": bool(route_metadata.get("intent_gate_accepted")),
            "intent_gate_reason": str(route_metadata.get("intent_gate_reason") or ""),
            "static_label": str(route_metadata.get("static_label") or ""),
            "static_confidence": _float_or_none(
                route_metadata.get("static_confidence")
            ),
            "static_type": str(route_metadata.get("static_type") or ""),
            "static_reject_reason": str(
                route_metadata.get("static_reject_reason") or ""
            ),
            "static_decision_source": str(
                route_metadata.get("static_decision_source") or ""
            ),
            "static_rejection_method": str(
                route_metadata.get("static_rejection_method") or ""
            ),
            "static_model_label": str(
                route_metadata.get("static_model_label") or ""
            ),
            "static_model_confidence": _float_or_none(
                route_metadata.get("static_model_confidence")
            ),
            "static_top2_label": str(route_metadata.get("static_top2_label") or ""),
            "static_top2_confidence": _float_or_none(
                route_metadata.get("static_top2_confidence")
            ),
            "static_margin": _float_or_none(route_metadata.get("static_margin")),
            "static_min_margin": _float_or_none(
                route_metadata.get("static_min_margin")
            ),
            "static_negative_label": str(
                route_metadata.get("static_negative_label") or ""
            ),
            "static_negative_confidence": _float_or_none(
                route_metadata.get("static_negative_confidence")
            ),
            "static_negative_threshold": _float_or_none(
                route_metadata.get("static_negative_threshold")
            ),
            "static_prototype_distance": _float_or_none(
                route_metadata.get("static_prototype_distance")
            ),
            "static_prototype_radius": _float_or_none(
                route_metadata.get("static_prototype_radius")
            ),
            "static_prototype_threshold": _float_or_none(
                route_metadata.get("static_prototype_threshold")
            ),
            "static_verifier_method": str(
                route_metadata.get("static_verifier_method") or ""
            ),
            "static_verifier_label": str(
                route_metadata.get("static_verifier_label") or ""
            ),
            "static_verifier_probability": _float_or_none(
                route_metadata.get("static_verifier_probability")
            ),
            "static_verifier_confidence": _float_or_none(
                route_metadata.get("static_verifier_confidence")
            ),
            "static_verifier_score": _float_or_none(
                route_metadata.get("static_verifier_score")
            ),
            "static_verifier_distance": _float_or_none(
                route_metadata.get("static_verifier_distance")
            ),
            "static_verifier_threshold": _float_or_none(
                route_metadata.get("static_verifier_threshold")
            ),
            "dynamic_label": str(route_metadata.get("dynamic_label") or ""),
            "dynamic_confidence": _float_or_none(
                route_metadata.get("dynamic_confidence")
            ),
            "dynamic_type": str(route_metadata.get("dynamic_type") or ""),
            "dynamic_reject_reason": str(
                route_metadata.get("dynamic_reject_reason") or ""
            ),
            "dynamic_phase": str(route_metadata.get("dynamic_phase") or ""),
            "dynamic_end_reason": str(route_metadata.get("dynamic_end_reason") or ""),
            "dynamic_segment_frames": int(
                route_metadata.get("dynamic_segment_frames") or 0
            ),
            "dynamic_motion_scale": _float_or_none(
                route_metadata.get("dynamic_motion_scale")
            ),
            "dynamic_decision_source": str(
                route_metadata.get("dynamic_decision_source") or ""
            ),
            "dynamic_motion_label": str(
                route_metadata.get("dynamic_motion_label") or ""
            ),
            "dynamic_motion_confidence": _float_or_none(
                route_metadata.get("dynamic_motion_confidence")
            ),
            "dynamic_model_label": str(
                route_metadata.get("dynamic_model_label") or ""
            ),
            "dynamic_model_confidence": _float_or_none(
                route_metadata.get("dynamic_model_confidence")
            ),
            "dynamic_model_confidence_for_motion": _float_or_none(
                route_metadata.get("dynamic_model_confidence_for_motion")
            ),
            "dynamic_prototype_method": str(
                route_metadata.get("dynamic_prototype_method") or ""
            ),
            "dynamic_prototype_label": str(
                route_metadata.get("dynamic_prototype_label") or ""
            ),
            "dynamic_prototype_nearest_type": str(
                route_metadata.get("dynamic_prototype_nearest_type") or ""
            ),
            "dynamic_prototype_confidence": _float_or_none(
                route_metadata.get("dynamic_prototype_confidence")
            ),
            "dynamic_prototype_distance": _float_or_none(
                route_metadata.get("dynamic_prototype_distance")
            ),
            "dynamic_prototype_threshold": _float_or_none(
                route_metadata.get("dynamic_prototype_threshold")
            ),
            "dynamic_prototype_margin": _float_or_none(
                route_metadata.get("dynamic_prototype_margin")
            ),
            "dynamic_prototype_reason": str(
                route_metadata.get("dynamic_prototype_reason") or ""
            ),
            "dynamic_negative_label": str(
                route_metadata.get("dynamic_negative_label") or ""
            ),
            "dynamic_negative_confidence": _float_or_none(
                route_metadata.get("dynamic_negative_confidence")
            ),
            "dynamic_negative_threshold": _float_or_none(
                route_metadata.get("dynamic_negative_threshold")
            ),
            "dynamic_axis": str(route_metadata.get("dynamic_axis") or ""),
            "dynamic_direction": str(route_metadata.get("dynamic_direction") or ""),
            "dynamic_axis_ratio": _float_or_none(
                route_metadata.get("dynamic_axis_ratio")
            ),
            "dynamic_straightness": _float_or_none(
                route_metadata.get("dynamic_straightness")
            ),
            "mediapipe_profile": str(route_metadata.get("mediapipe_profile") or ""),
            "mediapipe_timestamp_source": str(
                route_metadata.get("mediapipe_timestamp_source") or ""
            ),
            "mediapipe_min_detection_confidence": _float_or_none(
                route_metadata.get("mediapipe_min_detection_confidence")
            ),
            "mediapipe_min_presence_confidence": _float_or_none(
                route_metadata.get("mediapipe_min_presence_confidence")
            ),
            "mediapipe_min_tracking_confidence": _float_or_none(
                route_metadata.get("mediapipe_min_tracking_confidence")
            ),
            "mediapipe_smoothing_alpha": _float_or_none(
                route_metadata.get("mediapipe_smoothing_alpha")
            ),
            "mediapipe_detection_ms": _float_or_none(
                route_metadata.get("mediapipe_detection_ms")
            ),
            "hand_detected": (
                1.0 if bool(route_metadata.get("hand_detected")) else 0.0
            ),
            "hand_count": _float_or_none(route_metadata.get("hand_count")),
            "handedness_score_max": _float_or_none(
                route_metadata.get("handedness_score_max")
            ),
            "landmark_z_available": (
                1.0 if bool(route_metadata.get("landmark_z_available")) else 0.0
            ),
            "world_landmarks_available": (
                1.0
                if bool(route_metadata.get("world_landmarks_available"))
                else 0.0
            ),
            "landmark_z_range": _float_or_none(
                route_metadata.get("landmark_z_range")
            ),
            "world_z_range": _float_or_none(route_metadata.get("world_z_range")),
            "hand_bbox_area": _float_or_none(route_metadata.get("hand_bbox_area")),
            "hand_bbox_diag": _float_or_none(route_metadata.get("hand_bbox_diag")),
            "primary_wrist_step": _float_or_none(
                route_metadata.get("primary_wrist_step")
            ),
            "hand_lost_streak": _float_or_none(
                route_metadata.get("hand_lost_streak")
            ),
            "hand_lost_grace_frames": _float_or_none(
                route_metadata.get("hand_lost_grace_frames")
            ),
        }

    def _live_evaluation_route_counts(self, attempts: Any) -> dict[str, int]:
        if not isinstance(attempts, list):
            return {}
        counts: dict[str, int] = {}
        for item in attempts:
            if not isinstance(item, dict):
                continue
            route = str(item.get("route") or "").strip() or "unknown"
            counts[route] = counts.get(route, 0) + 1
        return dict(sorted(counts.items()))

    def _record_live_evaluation_attempt(
        self,
        session: dict[str, Any],
        *,
        result: str,
        predicted_label: str = "",
        confidence: float = 0.0,
        route_metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> None:
        timestamp = time.time()
        monotonic_now = time.monotonic() if now is None else float(now)
        attempt_no = int(session.get("total") or 0) + 1
        expected = str(session.get("expected_label") or "")
        elapsed = max(
            0.0,
            monotonic_now - float(session.get("attempt_started_at") or monotonic_now),
        )
        row = {
            "attempt": attempt_no,
            "expected": expected,
            "predicted": str(predicted_label or ""),
            "confidence": float(confidence or 0.0),
            "result": result,
            "elapsed_seconds": round(elapsed, 3),
            "recorded_at": timestamp,
        }
        row.update(self._live_evaluation_route_fields(route_metadata))
        attempts = session.setdefault("attempts", [])
        attempts.append(row)
        session["total"] = attempt_no
        if result == "correct":
            session["correct"] = int(session.get("correct") or 0) + 1
        elif result == "wrong":
            session["wrong"] = int(session.get("wrong") or 0) + 1
        elif result == "missed":
            session["missed"] = int(session.get("missed") or 0) + 1
        session["last_prediction"] = row["predicted"]
        session["last_confidence"] = row["confidence"]
        session["last_result"] = result
        attempt_payload = dict(row)
        attempt_payload.update(
            {
                "expected_label": expected,
                "target_attempts": session.get("target_attempts"),
                "total": session.get("total"),
                "correct": session.get("correct"),
                "wrong": session.get("wrong"),
                "missed": session.get("missed"),
                "min_confidence": session.get("min_confidence"),
                "timeout_seconds": session.get("timeout_seconds"),
                "recognition_model_mode": session.get("recognition_model_mode"),
                "dynamic_model_profile": session.get("dynamic_model_profile"),
                "static_rejection_method": session.get("static_rejection_method"),
            }
        )
        self._append_live_evaluation_jsonl(attempt_payload, event_type="attempt")

        target = int(session.get("target_attempts") or 1)
        if attempt_no >= target:
            session["message"] = self._live_evaluation_finish_message(session, "completed")
            self._finish_live_evaluation("completed")
            return

        cooldown = float(session.get("cooldown_seconds") or LIVE_EVAL_ATTEMPT_COOLDOWN_SECONDS)
        session["next_ready_at"] = monotonic_now + cooldown
        session["attempt_started_at"] = monotonic_now + cooldown
        session["message"] = (
            f"Тест {expected}: попытка {attempt_no + 1}/{target}"
        )
        self._emit_live_evaluation_changed(session, message=session["message"])
        self._set_status(f"Live eval: {expected} {attempt_no + 1}/{target}")

    def _consume_live_evaluation_prediction(
        self,
        label: str,
        confidence: float,
        *,
        route_metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> None:
        monotonic_now = time.monotonic() if now is None else float(now)
        with self._live_evaluation_lock:
            session = self._live_evaluation
            if session is None or not bool(session.get("active")):
                return
            if monotonic_now < float(session.get("next_ready_at") or 0.0):
                return

            clean_label = str(label or "").strip()
            conf = float(confidence or 0.0)
            session["last_prediction"] = clean_label
            session["last_confidence"] = conf
            if conf < float(session.get("min_confidence") or 0.0):
                session["last_result"] = "below_threshold"
                self._emit_live_evaluation_changed(
                    session,
                    message=f"{clean_label}: confidence {conf:.2f} ниже порога",
                )
                return

            expected = str(session.get("expected_label") or "")
            expected_type = self._gesture_type_for_label(expected)
            predicted_type = self._gesture_type_for_label(clean_label)
            route = str((route_metadata or {}).get("route") or "")
            if expected_type == GESTURE_TYPE_NEGATIVE:
                result = (
                    "correct"
                    if predicted_type == GESTURE_TYPE_NEGATIVE
                    else "wrong"
                )
                self._record_live_evaluation_attempt(
                    session,
                    result=result,
                    predicted_label=clean_label,
                    confidence=conf,
                    route_metadata=route_metadata,
                    now=monotonic_now,
                )
                return
            if expected_type == GESTURE_TYPE_DYNAMIC and route != "dynamic":
                self._record_live_evaluation_attempt(
                    session,
                    result="wrong",
                    predicted_label=clean_label,
                    confidence=conf,
                    route_metadata=route_metadata,
                    now=monotonic_now,
                )
                return
            result = "correct" if clean_label == expected else "wrong"
            self._record_live_evaluation_attempt(
                session,
                result=result,
                predicted_label=clean_label,
                confidence=conf,
                route_metadata=route_metadata,
                now=monotonic_now,
            )

    def _update_live_evaluation_timeout(self, *, now: float | None = None) -> None:
        monotonic_now = time.monotonic() if now is None else float(now)
        with self._live_evaluation_lock:
            session = self._live_evaluation
            if session is None or not bool(session.get("active")):
                return
            if monotonic_now < float(session.get("next_ready_at") or 0.0):
                return
            started = float(session.get("attempt_started_at") or monotonic_now)
            timeout = float(session.get("timeout_seconds") or LIVE_EVAL_DEFAULT_TIMEOUT_SECONDS)
            if timeout <= 0.0:
                return
            if monotonic_now - started < timeout:
                return
            expected = str(session.get("expected_label") or "")
            result = (
                "correct"
                if self._gesture_type_for_label(expected) == GESTURE_TYPE_NEGATIVE
                else "missed"
            )
            self._record_live_evaluation_attempt(
                session,
                result=result,
                predicted_label="",
                confidence=0.0,
                route_metadata={"route": "none"},
                now=monotonic_now,
            )

    def _ensure_pointer_control(self) -> Any | None:
        if not POINTER_CONTROL_AVAILABLE or PointerControlService is None:
            return None
        if self._pointer_control is None:
            self._pointer_control = PointerControlService(
                smoothing=float(self._config.recognition.pointer_smoothing)
            )
        return self._pointer_control

    def _reset_pointer_control(self) -> None:
        if self._pointer_control is not None:
            try:
                self._pointer_control.reset()
            except Exception:
                pass
        self._emit_pointer_state(state="idle", moved=False, clicked=False)

    def _emit_pointer_state(
        self,
        *,
        state: str,
        moved: bool = False,
        clicked: bool = False,
        tab_switched: str = "",
        error: str = "",
    ) -> None:
        payload = {
            "enabled": bool(getattr(self, "_pointer_mode", False)),
            "state": str(state or "idle"),
            "moved": bool(moved),
            "clicked": bool(clicked),
            "tabSwitched": str(tab_switched or ""),
            "error": str(error or ""),
        }
        if payload == getattr(self, "_pointer_state_payload", None):
            return
        self._pointer_state_payload = payload
        event = getattr(self, "pointer_state_changed", None)
        if event is not None:
            try:
                event.emit(dict(payload))
            except Exception:
                pass

    def _update_pointer_from_landmarks(self, landmarks_json: str) -> None:
        if not self._pointer_mode:
            return
        pointer = self._ensure_pointer_control()
        if pointer is None:
            self._emit_pointer_state(
                state="disabled",
                error="pyautogui недоступен",
            )
            self._set_status("Pointer: pyautogui недоступен")
            return
        result = pointer.update(landmarks_json)
        self._emit_pointer_state(
            state=str(getattr(result, "state", "") or "tracking"),
            moved=bool(getattr(result, "moved", False)),
            clicked=bool(getattr(result, "clicked", False)),
            tab_switched=str(getattr(result, "tab_switched", "") or ""),
            error=str(getattr(result, "error", "") or ""),
        )
        if not result.ok and result.error:
            self._set_status(f"Pointer: {result.error}")

    def _draw_landmarks_on_frame(self, frame_bgr: Any, landmarks_json: str) -> None:
        if (
            not self._show_landmark_overlay
            or not landmarks_json
            or landmarks_json == "[]"
            or parse_landmarks_json is None
        ):
            return
        try:
            import cv2
            from cv.hand_landmarker import HAND_CONNECTIONS

            hands = parse_landmarks_json(landmarks_json)
            height, width = frame_bgr.shape[:2]
            for hand in hands:
                points = hand.get("landmarks") or []
                if not isinstance(points, list):
                    continue

                pts_px: list[tuple[int, int]] = []
                for point in points:
                    try:
                        x = max(0.0, min(1.0, float(point[0])))
                        y = max(0.0, min(1.0, float(point[1])))
                    except (TypeError, ValueError, IndexError):
                        continue
                    pts_px.append(
                        (
                            max(0, min(width - 1, int(round(x * width)))),
                            max(0, min(height - 1, int(round(y * height)))),
                        )
                    )

                if len(pts_px) < 21:
                    continue
                for a, b in HAND_CONNECTIONS:
                    cv2.line(
                        frame_bgr,
                        pts_px[a],
                        pts_px[b],
                        (0, 215, 255),
                        2,
                        cv2.LINE_AA,
                    )
                for x, y in pts_px:
                    cv2.circle(frame_bgr, (x, y), 4, (0, 255, 80), -1, cv2.LINE_AA)
        except Exception as e:
            print(f"[w] draw_landmarks_on_frame: {e}", flush=True)

    def _sample_recording_snapshot_from_session(
        self,
        session: dict[str, Any] | None,
        *,
        active: bool,
        message: str = "",
    ) -> dict[str, Any]:
        if session is None:
            return {
                "active": False,
                "label": "",
                "saved": 0,
                "target": 0,
                "sampleIndex": 0,
                "currentFrames": 0,
                "targetFrames": 0,
                "progress": 0.0,
                "twoHands": False,
                "message": message,
            }

        target = max(1, int(session.get("target") or 1))
        saved = max(0, int(session.get("saved") or 0))
        target_frames = max(1, int(session.get("frames") or 1))
        current_frames = len(session.get("frames_buf") or [])
        if saved >= target:
            sample_index = target
            progress = 1.0
            current_frames = target_frames
        else:
            sample_index = saved + 1
            done_steps = saved * target_frames + current_frames
            progress = min(1.0, done_steps / float(target * target_frames))

        return {
            "active": bool(active),
            "label": str(session.get("label") or ""),
            "saved": saved,
            "target": target,
            "sampleIndex": sample_index,
            "currentFrames": current_frames,
            "targetFrames": target_frames,
            "progress": progress,
            "twoHands": bool(session.get("two_hands")),
            "message": message or str(session.get("last_message") or ""),
        }

    def _sample_recording_snapshot(self) -> dict[str, Any]:
        with self._sample_recording_lock:
            session = self._sample_recording
        return self._sample_recording_snapshot_from_session(
            session,
            active=session is not None,
        )

    def _emit_sample_recording_changed(
        self,
        session: dict[str, Any] | None = None,
        *,
        active: bool | None = None,
        message: str = "",
    ) -> None:
        if session is None:
            with self._sample_recording_lock:
                session = self._sample_recording
        snapshot = self._sample_recording_snapshot_from_session(
            session,
            active=(session is not None if active is None else active),
            message=message,
        )
        event = getattr(self, "sample_recording_changed", None)
        if event is not None:
            event.emit(snapshot)

    def _draw_sample_recording_overlay_on_frame(self, frame_bgr: Any) -> None:
        with self._sample_recording_lock:
            session = self._sample_recording
        if session is None:
            return

        try:
            import cv2

            snap = self._sample_recording_snapshot_from_session(
                session,
                active=True,
            )
            height, width = frame_bgr.shape[:2]
            box_w = min(width - 24, 430)
            cv2.rectangle(frame_bgr, (12, 12), (12 + box_w, 104), (16, 16, 28), -1)
            cv2.rectangle(frame_bgr, (12, 12), (12 + box_w, 104), (64, 64, 92), 1)
            cv2.circle(frame_bgr, (34, 36), 9, (40, 40, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame_bgr,
                "REC",
                (52, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                snap["label"][:28],
                (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (235, 245, 255),
                2,
                cv2.LINE_AA,
            )
            detail = (
                f"sample {snap['sampleIndex']}/{snap['target']}  "
                f"frames {snap['currentFrames']}/{snap['targetFrames']}"
            )
            cv2.putText(
                frame_bgr,
                detail,
                (20, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (170, 230, 255),
                1,
                cv2.LINE_AA,
            )
        except Exception as e:
            print(f"[w] draw_sample_recording_overlay_on_frame: {e}", flush=True)

    def _ensure_embedded_recognition_for_live_controls(self) -> None:
        if self._embedded_active:
            return
        self._set_recognizing(True)
        self.start_embedded_recognition()
        if not self._embedded_active:
            self._set_recognizing(False)

    def _live_recognition_status(self) -> str:
        if self.recognition_model_mode == RECOGNITION_MODEL_DYNAMIC:
            model_suffix = f" (dynamic:{self.dynamic_model_profile})"
        elif self.recognition_model_mode == RECOGNITION_MODEL_AUTO:
            model_suffix = (
                f" (auto:{self.dynamic_model_profile}, reject:{self.static_rejection_method})"
            )
        else:
            model_suffix = f" (reject:{self.static_rejection_method})"
        if self._gesture_mode and self._pointer_mode:
            return f"Жесты и указатель включены{model_suffix}"
        if self._pointer_mode:
            return "Указатель: указательный палец двигает курсор"
        if self._gesture_mode:
            return f"Распознавание жестов включено{model_suffix}"
        if self._show_landmark_overlay:
            return "Точки руки включены"
        return "Камера включена"

    def _create_embedded_infer(self) -> Any | None:
        err = ""
        infer: Any | None = None
        try:
            from app.gesture_online_infer import GestureOnlineInfer

            if self.recognition_model_mode == RECOGNITION_MODEL_AUTO:
                from app.services.recognition_router import GestureRecognitionRouter

                print("[ctrl.embedded] creating GestureRecognitionRouter in camera thread…")
                dynamic_model_path = self._dynamic_model_path()
                dynamic_classes_path = self._dynamic_classes_path()
                dynamic_feature_dim_path = self._dynamic_feature_dim_path()
                dynamic_feature_mode_path = self._dynamic_feature_mode_path()
                dynamic_prototypes_path = self._dynamic_prototypes_path()
                print(
                    "[ctrl.embedded] dynamic profile="
                    f"{self.dynamic_model_profile} model={dynamic_model_path} "
                    f"classes={dynamic_classes_path} "
                    f"feature_dim={dynamic_feature_dim_path} "
                    f"feature_mode={dynamic_feature_mode_path} "
                    f"prototypes={dynamic_prototypes_path}",
                    flush=True,
                )
                static_infer = GestureOnlineInfer(
                    model_path=self._configured_model_path(),
                    classes_path=self._configured_classes_path(),
                    feature_dim_path=self._configured_feature_dim_path(),
                    feature_mode_path=self._configured_feature_mode_path(),
                    static_rejection_verifier_path=self._static_rejection_verifier_path(),
                    static_rejection_method=self.static_rejection_method,
                    window=30,
                    two_hands=self._two_hands_mode,
                )
                dynamic_infer = GestureOnlineInfer(
                    model_path=dynamic_model_path,
                    classes_path=dynamic_classes_path,
                    feature_dim_path=dynamic_feature_dim_path,
                    feature_mode_path=dynamic_feature_mode_path,
                    dynamic_prototypes_path=dynamic_prototypes_path,
                    window=self._dynamic_recognition_window(),
                    two_hands=self._two_hands_mode,
                    initialize_detector=False,
                )
                infer = GestureRecognitionRouter(
                    static_infer=static_infer,
                    dynamic_infer=dynamic_infer,
                    intent_gate_path=str(self._intent_gate_model_path()),
                )
            else:
                model_path, classes_path, feature_dim_path, feature_mode_path = (
                    self._embedded_model_paths()
                )
                print("[ctrl.embedded] creating GestureOnlineInfer in camera thread…")
                infer = GestureOnlineInfer(
                    model_path=model_path,
                    classes_path=classes_path,
                    feature_dim_path=feature_dim_path,
                    feature_mode_path=feature_mode_path,
                    static_rejection_verifier_path=self._static_rejection_verifier_path(),
                    dynamic_prototypes_path=self._dynamic_prototypes_path(),
                    static_rejection_method=self.static_rejection_method,
                    window=self._embedded_recognition_window(),
                    two_hands=self._two_hands_mode,
                )
            print(
                "[ctrl.embedded] infer created; "
                f"init_error={getattr(infer, 'init_error', '')!r}",
                flush=True,
            )
            if getattr(infer, "init_error", ""):
                err = infer.init_error
                infer.close()
                infer = None
        except Exception as e:
            err = str(e)
            print(f"[!] embedded infer init: {e}", flush=True)
            import traceback

            traceback.print_exc()
            infer = None

        if infer is None:
            self._embedded_active = False
            self._set_recognizing(False)
            self._set_status(f"CV init error: {err}" if err else "CV init failed")
            return None

        self._set_status(self._live_recognition_status())
        return infer

    # ----------------------------------------------------------------------
    # Камера (поток-цикл — отдельный thread, не QTimer)
    # ----------------------------------------------------------------------

    def start_camera(self) -> None:
        print(f"[ctrl.start_camera] active={self._is_camera_active} cv2={_CV2_AVAILABLE}")
        if self._is_camera_active:
            return
        if not _CV2_AVAILABLE:
            self._set_status("Camera: OpenCV не установлен")
            return
        try:
            from app.cv_camera import open_default_capture

            cap = open_default_capture(
                int(self._config.recognition.camera_index),
                width=CAMERA_CAPTURE_WIDTH,
                height=CAMERA_CAPTURE_HEIGHT,
                fps=int(self._config.recognition.target_fps),
            )
        except Exception as e:
            print(f"[!] Ошибка открытия камеры: {e}")
            import traceback
            traceback.print_exc()
            self._set_status(f"Camera error: {e}")
            return
        print(f"[ctrl.start_camera] cap.isOpened()={cap.isOpened()}")
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            self._set_status("Camera: open failed (Privacy → Camera?)")
            return

        self._camera_cap = cap
        self._camera_frame_seq = 0
        with self._preview_condition:
            self._latest_preview_frame = None
        self._camera_stop.clear()
        self._camera_thread = threading.Thread(
            target=self._camera_capture_loop, name="dplm-camera-capture", daemon=True
        )
        if _camera_preview_enabled():
            self._camera_preview_thread = threading.Thread(
                target=self._camera_preview_loop,
                name="dplm-camera-preview",
                daemon=True,
            )
        else:
            self._camera_preview_thread = None
        self._camera_thread.start()
        if self._camera_preview_thread is not None:
            self._camera_preview_thread.start()
        self._set_camera_active(True)
        self._set_status("Camera: streaming")
        print("[✓] Flet: камера открыта")

    def stop_camera(self) -> None:
        if (
            not self._is_camera_active
            and not self._camera_thread
            and not self._camera_preview_thread
        ):
            return
        self._camera_stop.set()
        with self._preview_condition:
            self._preview_condition.notify_all()
        cap = self._camera_cap
        self._camera_cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        current_thread = threading.current_thread()
        for t in (
            self._camera_thread,
            self._camera_preview_thread,
        ):
            if t is not None and t is not current_thread:
                t.join(timeout=2.0)
        self._camera_thread = None
        self._camera_preview_thread = None
        with self._preview_condition:
            self._latest_preview_frame = None
        with self._frame_lock:
            self._latest_jpeg_bytes = b""
            self._frame_w = 0
            self._frame_h = 0
        self._set_camera_active(False)
        if self._is_recognition_pid_active():
            self._set_status("Recognizing in background")
        elif self._status.startswith("Camera:") or self._status.startswith("CV:"):
            self._set_status("Stopped")
        print("[i] Flet: камера остановлена")

    def _publish_preview_frame(self, frame: _CameraFrame) -> None:
        with self._preview_condition:
            self._latest_preview_frame = frame
            self._preview_condition.notify()

    def _camera_capture_loop(self) -> None:
        """Read fresh camera frames and run ML immediately; Flet preview is separate."""
        import cv2

        frame_interval = 1.0 / max(1.0, float(self._target_fps))
        next_t = time.monotonic()

        while not self._camera_stop.is_set():
            cap = self._camera_cap
            if cap is None:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            self._camera_frame_seq += 1
            frame = _CameraFrame(
                sequence=int(self._camera_frame_seq),
                frame_bgr=cv2.flip(frame_bgr, 1),
                captured_at=time.monotonic(),
            )
            self._process_camera_frame_for_ml(frame)
            self._publish_preview_frame(frame)

            after_work = time.monotonic()
            next_t += frame_interval
            sleep = next_t - after_work
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = after_work

    def _process_camera_frame_for_ml(
        self,
        frame: _CameraFrame,
    ) -> None:
        import cv2
        import numpy as np

        frame_bgr = frame.frame_bgr
        if self._sample_recording is not None:
            try:
                recording_landmarks = self._process_sample_recording_frame(frame_bgr)
                if recording_landmarks:
                    self._set_landmarks(recording_landmarks)
            except Exception as e:
                print(f"[!] embedded recording frame error: {e}")
                self._finish_sample_recording(1, f"[!] Ошибка записи сэмпла: {e}")
            return

        if not self._embedded_active:
            self._set_landmarks("[]")
            return

        try:
            if self._embedded_infer is None:
                self._embedded_infer = self._create_embedded_infer()
            if self._embedded_infer is None:
                return

            camera_h, camera_w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb = _resize_frame_to_max_width(
                rgb,
                CAMERA_INFERENCE_MAX_WIDTH,
            )
            rgb = np.ascontiguousarray(rgb)
            inference_started_at = time.monotonic()
            timestamp_ms = int(round(float(frame.captured_at) * 1000.0))
            try:
                out = self._embedded_infer.process_frame_rgb(
                    rgb,
                    timestamp_ms=timestamp_ms,
                )
            except TypeError as exc:
                if "timestamp" not in str(exc):
                    raise
                out = self._embedded_infer.process_frame_rgb(rgb)
            if out is None:
                return
            perf = out.get("performance")
            if isinstance(perf, dict):
                inference_h, inference_w = rgb.shape[:2]
                perf.update(
                    {
                        "camera_frame_width": int(camera_w),
                        "camera_frame_height": int(camera_h),
                        "camera_frame_sequence": int(frame.sequence),
                        "inference_frame_width": int(inference_w),
                        "inference_frame_height": int(inference_h),
                        "preview_max_fps": float(CAMERA_PREVIEW_MAX_FPS),
                        "preview_enabled": bool(_camera_preview_enabled()),
                        "inference_max_fps": float(
                            max(float(self._target_fps), CAMERA_INFERENCE_MAX_FPS)
                        ),
                        "inference_frame_policy": "fresh_capture_loop",
                        "inference_queue_depth": 0,
                        "capture_to_inference_ms": round(
                            max(0.0, inference_started_at - frame.captured_at) * 1000.0,
                            3,
                        ),
                        "dynamic_window_frames": int(self._dynamic_recognition_window()),
                    }
                )
            self._dispatch_infer_result(out)
        except Exception as e:
            print(f"[!] embedded CV frame error: {e}")

    def _camera_preview_loop(self) -> None:
        """Encode latest camera frame for Flet; skipped preview frames do not hit ML."""
        import cv2

        preview_interval = 1.0 / max(
            1.0,
            min(float(self._target_fps), CAMERA_PREVIEW_MAX_FPS),
        )
        next_preview_t = time.monotonic()
        last_sequence = 0

        while not self._camera_stop.is_set():
            with self._preview_condition:
                self._preview_condition.wait_for(
                    lambda: (
                        self._camera_stop.is_set()
                        or (
                            self._latest_preview_frame is not None
                            and self._latest_preview_frame.sequence != last_sequence
                        )
                    ),
                    timeout=preview_interval,
                )
                if self._camera_stop.is_set():
                    break

            now = time.monotonic()
            sleep = next_preview_t - now
            if sleep > 0:
                time.sleep(min(sleep, preview_interval))

            with self._preview_condition:
                frame = self._latest_preview_frame
            if frame is None or frame.sequence == last_sequence:
                continue

            try:
                preview_frame = _resize_frame_to_max_width(
                    frame.frame_bgr.copy(),
                    CAMERA_PREVIEW_MAX_WIDTH,
                )
                self._draw_landmarks_on_frame(preview_frame, self._landmarks_json)
                self._draw_sample_recording_overlay_on_frame(preview_frame)

                ok2, buf = cv2.imencode(
                    ".jpg",
                    preview_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), CAMERA_PREVIEW_JPEG_QUALITY],
                )
                if ok2:
                    data = buf.tobytes()
                    with self._frame_lock:
                        self._latest_jpeg_bytes = data
                        self._frame_h, self._frame_w = preview_frame.shape[:2]
                    self.camera_frame_updated.emit()
                    last_sequence = frame.sequence
            except Exception as e:
                print(f"[!] Flet preview frame error: {e}")

            next_preview_t = time.monotonic() + preview_interval

    def _hands_landmarks_json(self, hands: list[Any]) -> str:
        payload: list[list[list[float]]] = []
        for hand in hands:
            landmarks = getattr(hand, "landmarks", None) or []
            payload.append([[float(x), float(y)] for (x, y) in landmarks])
        return json.dumps(payload, separators=(",", ":"))

    def _sample_frame_from_hands(
        self,
        hands: list[Any],
        *,
        two_hands: bool,
        include_global_motion: bool = False,
        include_landmark_z: bool = False,
    ) -> Any | None:
        import numpy as np

        from cv.gesture_features import normalize_landmark_z_with_xy
        from cv.hand_landmarker import normalize_landmarks

        def hand_feature(hand: Any) -> Any:
            normalized = normalize_landmarks(hand.landmarks)
            pts = np.asarray(hand.landmarks, dtype=np.float32)
            wrist = (
                pts[0]
                if pts.shape == (21, 2)
                else np.zeros(2, dtype=np.float32)
            )
            if include_landmark_z:
                xyz = np.asarray(
                    getattr(hand, "landmarks_xyz", None) or [],
                    dtype=np.float32,
                )
                if xyz.shape == (21, 3):
                    z = normalize_landmark_z_with_xy(pts, xyz[:, 2:3])
                else:
                    z = np.zeros((21, 1), dtype=np.float32)
                pose_xyz = np.concatenate([normalized, z], axis=1)
                if not include_global_motion:
                    return pose_xyz.astype(np.float32, copy=False)
                return np.concatenate([pose_xyz.reshape(-1), wrist], axis=0).astype(
                    np.float32,
                    copy=False,
                )
            if not include_global_motion:
                return normalized

            pose = normalized.reshape(-1)
            return np.concatenate([pose, wrist], axis=0).astype(np.float32, copy=False)

        features: list[Any] = []
        for hand in hands:
            try:
                features.append(hand_feature(hand))
            except Exception:
                continue

        if two_hands:
            if include_global_motion:
                per_hand_dim = 65 if include_landmark_z else 44
                if len(features) >= 2:
                    return np.concatenate(features[:2], axis=0)
                if len(features) == 1:
                    return np.concatenate(
                        [features[0], np.zeros(per_hand_dim, dtype=np.float32)],
                        axis=0,
                    )
            else:
                if len(features) >= 2:
                    return np.concatenate(features[:2], axis=0)
                if len(features) == 1:
                    pad_shape = (21, 3) if include_landmark_z else (21, 2)
                    return np.concatenate(
                        [features[0], np.zeros(pad_shape, dtype=np.float32)],
                        axis=0,
                    )
            return None

        if features:
            return features[0]
        return None

    def _ensure_sample_recording_detector(self, *, two_hands: bool) -> Any:
        if self._sample_recording_detector is not None:
            return self._sample_recording_detector

        from cv.hand_landmarker import HandLandmarkerVideo

        self._sample_recording_detector = HandLandmarkerVideo(
            num_hands=2 if two_hands else 1,
            min_detection_confidence=0.6,
            min_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        return self._sample_recording_detector

    def _next_sample_path(self, label_dir: Path) -> Path:
        existing = real_sample_paths(label_dir)
        used: set[int] = set()
        for path in existing:
            try:
                used.add(int(path.stem.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        index = 0
        while index in used:
            index += 1
        return label_dir / f"sample_{index:04d}.npy"

    def _reset_sample_recording_ready_gate(self, session: dict[str, Any]) -> None:
        session["ready_buffer"] = []
        session["countdown_until"] = 0.0
        session["ready_stability"] = None

    def _sample_recording_ready_gate(
        self,
        session: dict[str, Any],
        frame_vec: Any,
        now: float,
        on_line: Optional[Callable[[str], None]],
    ) -> bool:
        import numpy as np

        required = max(
            1,
            int(session.get("ready_required_frames", SAMPLE_RECORDING_READY_FRAMES)),
        )
        threshold = float(
            session.get("stability_threshold", SAMPLE_RECORDING_STABILITY_THRESHOLD)
        )
        countdown_seconds = max(
            0.0,
            float(session.get("countdown_seconds", SAMPLE_RECORDING_COUNTDOWN_SECONDS)),
        )
        vector = np.asarray(frame_vec, dtype=np.float32).reshape(-1)
        if vector.size <= 0:
            return False

        buffer = session.setdefault("ready_buffer", [])
        buffer.append(vector)
        if len(buffer) > required:
            del buffer[:-required]

        message: str
        if len(buffer) < required:
            session["countdown_until"] = 0.0
            session["ready_stability"] = None
            message = f"Зафиксируй стартовую позу: {len(buffer)}/{required}"
        else:
            deltas = [
                float(np.linalg.norm(buffer[i] - buffer[i - 1]) / np.sqrt(vector.size))
                for i in range(1, len(buffer))
            ]
            stability = max(deltas) if deltas else 0.0
            session["ready_stability"] = stability
            if stability > threshold:
                buffer[:] = [vector]
                session["countdown_until"] = 0.0
                message = f"Зафиксируй стартовую позу: jitter={stability:.3f}"
            else:
                countdown_until = float(session.get("countdown_until", 0.0) or 0.0)
                if countdown_until <= 0.0:
                    countdown_until = now + countdown_seconds
                    session["countdown_until"] = countdown_until
                    if on_line:
                        on_line(
                            "[i] Рука стабильна — запись начнётся после короткого отсчета"
                        )

                remaining = max(0.0, countdown_until - now)
                if remaining > 0.0:
                    message = f"Запись через {remaining:.1f} с"
                else:
                    self._reset_sample_recording_ready_gate(session)
                    session["last_message"] = "Идет запись сэмпла"
                    return True

        session["last_message"] = message
        last_emit = float(session.get("last_progress_emit", 0.0) or 0.0)
        if now - last_emit >= 0.15:
            session["last_progress_emit"] = now
            self._emit_sample_recording_changed(
                session,
                active=True,
                message=message,
            )

        last_ready_log = float(session.get("last_ready_log", 0.0) or 0.0)
        if on_line and now - last_ready_log >= 1.1 and not message.startswith("Запись"):
            session["last_ready_log"] = now
            on_line(f"[i] {message}")

        return False

    def _sample_quality_report(
        self,
        sequence: Any,
        *,
        label: str,
        include_global_motion: bool,
    ) -> dict[str, Any]:
        from cv.gesture_features import (
            sequence_displacement,
            sequence_motion_energy,
            sequence_to_matrix,
            trajectory_features,
        )

        seq = sequence_to_matrix(sequence)
        motion_energy = sequence_motion_energy(seq)
        displacement = sequence_displacement(seq)
        dx: float | None = None
        dy: float | None = None
        warnings: list[str] = []

        has_global_motion = bool(include_global_motion and seq.shape[1] >= 44)
        if has_global_motion:
            motion = trajectory_features(seq)
            dx = float(motion[0])
            dy = float(motion[1])
            if motion_energy < DYNAMIC_SAMPLE_MIN_MOTION_ENERGY:
                warnings.append("low_motion")

            clean_label = str(label or "").strip().lower()
            direction_specs = (
                ("left", "dx", -1.0),
                ("right", "dx", 1.0),
                ("up", "dy", -1.0),
                ("down", "dy", 1.0),
            )
            for token, axis, sign in direction_specs:
                if token not in clean_label:
                    continue
                value = dx if axis == "dx" else dy
                if value is None or (value * sign) < DYNAMIC_SAMPLE_DIRECTION_THRESHOLD:
                    warnings.append(f"expected_{token}")
                break

        return {
            "ok": not warnings,
            "shape": tuple(int(x) for x in seq.shape),
            "motion_energy": motion_energy,
            "displacement": displacement,
            "dx": dx,
            "dy": dy,
            "warnings": warnings,
        }

    def _format_sample_quality_line(self, path: Path, report: dict[str, Any]) -> str:
        parts = [
            f"motion={float(report['motion_energy']):.4f}",
            f"disp={float(report['displacement']):.4f}",
        ]
        if report.get("dx") is not None and report.get("dy") is not None:
            parts.append(f"dx={float(report['dx']):+.3f}")
            parts.append(f"dy={float(report['dy']):+.3f}")
        if report.get("projected_hand_scale") is not None:
            parts.append(f"hand_scale={float(report['projected_hand_scale']):.3f}")
        verdict = "OK" if report.get("ok") else "WARN: " + ",".join(report["warnings"])
        return f"[✓] Сохранено: {path} | {' '.join(parts)} | {verdict}"

    def _sample_recording_seed(self, label: str, path: Path) -> int:
        payload = f"{label}:{path.name}".encode("utf-8", errors="ignore")
        return int(zlib.crc32(payload) & 0xFFFFFFFF)

    def _json_safe_sample_report(self, report: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in report.items():
            if isinstance(value, tuple):
                out[key] = [int(item) for item in value]
            elif isinstance(value, list):
                out[key] = [str(item) for item in value]
            elif isinstance(value, bool):
                out[key] = bool(value)
            elif isinstance(value, int):
                out[key] = int(value)
            elif isinstance(value, float):
                out[key] = float(value)
            elif value is None:
                out[key] = None
            else:
                out[key] = str(value)
        return out

    def _write_sample_metadata(
        self,
        path: Path,
        metadata: dict[str, Any],
        on_line: Optional[Callable[[str], None]],
    ) -> None:
        try:
            path.with_suffix(".meta.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            if on_line:
                on_line(f"[w] Не удалось сохранить metadata: {exc}")

    def _sample_metadata_payload(
        self,
        *,
        session: dict[str, Any],
        path: Path,
        arr: Any,
        kind: str,
        report: dict[str, Any],
        projected_hand_scale: float | None,
        source_path: Path | None = None,
        transform: str = "",
        transform_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "label": str(session.get("label") or ""),
            "sample": path.name,
            "kind": kind,
            "source": "camera" if kind == "real" else "augmented",
            "source_sample": source_path.name if source_path is not None else "",
            "transform": transform,
            "transform_metadata": transform_metadata or {},
            "frames": int(arr.shape[0]),
            "raw_feature_dim": int(arr.reshape(arr.shape[0], -1).shape[1]),
            "two_hands": bool(session.get("two_hands")),
            "include_global_motion": bool(session.get("include_global_motion")),
            "include_landmark_z": bool(session.get("include_landmark_z")),
            "sample_feature_format": (
                "landmark_xyz_wrist_xy"
                if bool(session.get("include_global_motion"))
                and bool(session.get("include_landmark_z"))
                else "landmark_xyz"
                if bool(session.get("include_landmark_z"))
                else "landmark_xy_wrist_xy"
                if bool(session.get("include_global_motion"))
                else "landmark_xy"
            ),
            "projected_hand_scale_median": projected_hand_scale,
            "quality": self._json_safe_sample_report(report),
            "recorded_at": time.time(),
        }

    def _write_augmented_recording_samples(
        self,
        *,
        session: dict[str, Any],
        source_path: Path,
        arr: Any,
        projected_hand_scale: float | None,
        on_line: Optional[Callable[[str], None]],
    ) -> int:
        import numpy as np

        from cv.gesture_augmentation import augment_gislr_landmark_sequence

        count = max(0, int(session.get("augment_count", 0) or 0))
        if count <= 0:
            return 0

        saved = 0
        include_global_motion = bool(session.get("include_global_motion"))
        for variant_index in range(count):
            seed = self._sample_recording_seed(
                str(session.get("label") or ""),
                source_path.with_name(f"{source_path.name}:{variant_index}"),
            )
            try:
                augmented, transform_metadata = augment_gislr_landmark_sequence(
                    arr,
                    seed=seed,
                    include_global_motion=include_global_motion,
                )
            except Exception as exc:
                if on_line:
                    on_line(f"[w] GISLR-аугментация пропущена: {exc}")
                continue

            out_path = augmented_sample_path(source_path, variant_index)
            try:
                np.save(out_path, augmented.astype(np.float32, copy=False))
            except OSError as exc:
                if on_line:
                    on_line(f"[w] Не удалось сохранить augmented sample: {exc}")
                continue

            report = self._sample_quality_report(
                augmented,
                label=str(session.get("label") or ""),
                include_global_motion=include_global_motion,
            )
            report["projected_hand_scale"] = projected_hand_scale
            self._write_sample_metadata(
                out_path,
                self._sample_metadata_payload(
                    session=session,
                    path=out_path,
                    arr=augmented,
                    kind="augmented",
                    report=report,
                    projected_hand_scale=projected_hand_scale,
                    source_path=source_path,
                    transform="gislr_landmark_v1",
                    transform_metadata=transform_metadata,
                ),
                on_line,
            )
            saved += 1

        return saved

    def _process_sample_recording_frame(self, frame_bgr: Any) -> str:
        import cv2
        import numpy as np

        with self._sample_recording_lock:
            session = self._sample_recording
        if session is None:
            return ""

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w > 640:
            scale = 640.0 / w
            rgb = cv2.resize(
                rgb,
                (640, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            rgb = np.ascontiguousarray(rgb)

        with self._sample_recording_detector_lock:
            detector = self._ensure_sample_recording_detector(
                two_hands=bool(session.get("two_hands"))
            )
            hands = detector.detect_for_video_rgb(rgb)
        landmarks_json = self._hands_landmarks_json(hands)
        frame_vec = self._sample_frame_from_hands(
            hands,
            two_hands=bool(session.get("two_hands")),
            include_global_motion=bool(session.get("include_global_motion")),
            include_landmark_z=bool(session.get("include_landmark_z")),
        )
        hand_scales: list[float] = []
        for hand in hands:
            points = np.asarray(getattr(hand, "landmarks", []), dtype=np.float32)
            if points.shape == (21, 2):
                size = points.max(axis=0) - points.min(axis=0)
                hand_scales.append(float(np.linalg.norm(size)))
        current_hand_scale = (
            float(np.median(hand_scales)) if hand_scales else 0.0
        )

        now = time.monotonic()
        on_line = session.get("on_line")
        if frame_vec is None:
            if bool(session.get("quality_gate")):
                self._reset_sample_recording_ready_gate(session)
                if session.get("frames_buf"):
                    session["frames_buf"] = []
                    session["frame_scales"] = []
                    if on_line:
                        on_line("[w] Сэмпл сброшен: рука пропала из кадра")
            last_no_hand = float(session.get("last_no_hand_log", 0.0) or 0.0)
            if on_line and now - last_no_hand > 1.25:
                session["last_no_hand_log"] = now
                session["last_message"] = "Держи руку в кадре"
                on_line("[i] Держи руку в кадре — сэмпл начнёт собираться автоматически")
                self._emit_sample_recording_changed(
                    session,
                    active=True,
                    message="Держи руку в кадре",
                )
            return landmarks_json

        if now < float(session.get("next_allowed_at", 0.0) or 0.0):
            return landmarks_json

        warmup_until = float(session.get("warmup_until", 0.0) or 0.0)
        if now < warmup_until:
            remaining = max(0.0, warmup_until - now)
            message = f"Подготовка к записи: {remaining:.1f} с"
            session["last_message"] = message
            last_progress_emit = float(session.get("last_progress_emit", 0.0) or 0.0)
            if now - last_progress_emit >= 0.15:
                session["last_progress_emit"] = now
                self._emit_sample_recording_changed(
                    session,
                    active=True,
                    message=message,
                )
            return landmarks_json

        frames = session.setdefault("frames_buf", [])
        if bool(session.get("quality_gate")) and not frames:
            if not self._sample_recording_ready_gate(session, frame_vec, now, on_line):
                return landmarks_json

        frames.append(frame_vec)
        session.setdefault("frame_scales", []).append(current_hand_scale)
        session["last_message"] = "Идет запись сэмпла"
        target_frames = int(session["frames"])
        progress_step = max(1, target_frames // 3)
        if on_line and len(frames) in {1, progress_step, progress_step * 2}:
            on_line(
                f"[i] Сбор сэмпла {int(session['saved']) + 1}/"
                f"{int(session['target'])}: {len(frames)}/{target_frames} кадров"
            )
        last_progress_emit = float(session.get("last_progress_emit", 0.0) or 0.0)
        if now - last_progress_emit >= 0.12 or len(frames) >= target_frames:
            session["last_progress_emit"] = now
            self._emit_sample_recording_changed(
                session,
                active=True,
                message="Идет запись сэмпла",
            )

        if len(frames) < target_frames:
            return landmarks_json

        label_dir = Path(session["out_dir"])
        label_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._next_sample_path(label_dir)
        arr = np.asarray(frames[:target_frames], dtype=np.float32)
        np.save(out_path, arr)
        projected_hand_scale: float | None = None
        if bool(session.get("include_global_motion")):
            scales = [
                float(value)
                for value in session.get("frame_scales", [])[:target_frames]
                if float(value) > 0.0
            ]
            projected_hand_scale = (
                float(np.median(scales)) if scales else None
            )
        report = self._sample_quality_report(
            arr,
            label=str(session.get("label") or ""),
            include_global_motion=bool(session.get("include_global_motion")),
        )
        report["projected_hand_scale"] = projected_hand_scale
        self._write_sample_metadata(
            out_path,
            self._sample_metadata_payload(
                session=session,
                path=out_path,
                arr=arr,
                kind="real",
                report=report,
                projected_hand_scale=projected_hand_scale,
            ),
            on_line,
        )
        augmented_saved = self._write_augmented_recording_samples(
            session=session,
            source_path=out_path,
            arr=arr,
            projected_hand_scale=projected_hand_scale,
            on_line=on_line,
        )
        session["saved"] = int(session["saved"]) + 1
        session["frames_buf"] = []
        session["frame_scales"] = []
        self._reset_sample_recording_ready_gate(session)
        reports = session.setdefault("quality_reports", [])
        reports.append({"path": out_path.name, "augmented": augmented_saved, **report})
        session["next_allowed_at"] = now + 0.45
        verdict = "OK" if report["ok"] else "WARN"
        aug_suffix = f", +{augmented_saved} aug" if augmented_saved else ""
        session["last_message"] = f"Сохранено: {out_path.name} ({verdict}{aug_suffix})"
        if on_line:
            on_line(self._format_sample_quality_line(out_path, report))
        self._emit_sample_recording_changed(
            session,
            active=True,
            message=f"Сохранено: {out_path.name} ({verdict}{aug_suffix})",
        )

        if int(session["saved"]) >= int(session["target"]):
            self._finish_sample_recording(
                0,
                f"[✓] Запись «{session['label']}» завершена: "
                f"{session['saved']} сэмплов",
            )

        return landmarks_json

    def _finish_sample_recording(self, code: int, message: str = "") -> None:
        with self._sample_recording_lock:
            session = self._sample_recording
            self._sample_recording = None

        with self._sample_recording_detector_lock:
            detector = self._sample_recording_detector
            self._sample_recording_detector = None
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass

        if session is None:
            return

        on_line = session.get("on_line")
        on_done = session.get("on_done")
        if message and on_line:
            try:
                on_line(message)
            except Exception:
                pass

        if int(code) == 0:
            try:
                summary = self.sync_dataset_to_db()
                if on_line:
                    on_line(
                        f"[✓] БД жестов синхронизирована: "
                        f"добавлено {summary['created']}, "
                        f"обновлено {summary['updated']}, "
                        f"классов: {summary['total']}, "
                        f"сэмплов: {summary['samples']}"
                    )
            except Exception as e:
                if on_line:
                    on_line(f"[w] sync_dataset_to_db: {e}")

        self._emit_sample_recording_changed(
            session,
            active=False,
            message=message or "Запись завершена",
        )

        if on_done:
            try:
                on_done(int(code))
            except Exception:
                pass

        if bool(session.get("started_camera_for_recording")) and self._is_camera_active:
            self.stop_camera()
        elif self._status.startswith("Запись"):
            self._set_status("Camera: streaming" if self._is_camera_active else "Idle")

    def _append_jsonl_log(self, filename: str, row: dict[str, Any]) -> None:
        log_dir = self._live_usage_log_dir()
        if log_dir is None:
            return
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / filename).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                fh.write("\n")
        except Exception as exc:
            print(f"[w] {filename} write failed: {exc}", flush=True)

    def _write_json_log(self, filename: str, payload: dict[str, Any]) -> None:
        log_dir = self._live_usage_log_dir()
        if log_dir is None:
            return
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / filename).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[w] {filename} write failed: {exc}", flush=True)

    @staticmethod
    def _live_usage_counter(events: list[dict[str, Any]], key: str) -> dict[str, int]:
        counter = Counter(str(item.get(key) or "unknown") for item in events)
        return dict(sorted(counter.items()))

    def _write_live_usage_summary(self) -> None:
        events = list(getattr(self, "_live_usage_events", []) or [])
        now = time.time()
        window_start = now - LIVE_USAGE_SUMMARY_WINDOW_SECONDS
        window_events = [
            event
            for event in events
            if float(event.get("recorded_at") or 0.0) >= window_start
        ]
        if window_events != events:
            self._live_usage_events = window_events[-LIVE_USAGE_EVENTS_LIMIT:]
            events = list(self._live_usage_events)

        executed = sum(1 for item in events if bool(item.get("executed")))
        command_events = [
            item
            for item in events
            if str(item.get("event_type") or "")
            in {"command_executed", "command_rejected", "command_cooldown"}
        ]
        rejected = sum(
            1
            for item in events
            if str(item.get("event_type") or "") in {
                "gesture_rejected",
                "command_rejected",
            }
        )
        suppressed = sum(
            1
            for item in events
            if str(item.get("event_type") or "") == "gesture_suppressed"
        )
        cooldown = sum(
            1
            for item in events
            if str(item.get("event_type") or "") == "command_cooldown"
        )
        confidence_values = [
            float(item.get("confidence") or 0.0)
            for item in events
            if float(item.get("confidence") or 0.0) > 0.0
        ]
        total = len(events)
        payload = {
            "generated_at": now,
            "window_seconds": LIVE_USAGE_SUMMARY_WINDOW_SECONDS,
            "total_events": total,
            "command_attempts": len(command_events),
            "executed_events": executed,
            "rejected_events": rejected,
            "suppressed_events": suppressed,
            "cooldown_events": cooldown,
            "command_success_rate": (
                executed / len(command_events) if command_events else 0.0
            ),
            "avg_confidence": (
                sum(confidence_values) / len(confidence_values)
                if confidence_values
                else 0.0
            ),
            "label_counts": self._live_usage_counter(events, "label"),
            "route_counts": self._live_usage_counter(events, "route"),
            "event_type_counts": self._live_usage_counter(events, "event_type"),
            "latest_event": events[-1] if events else {},
            "latest_runtime": dict(
                getattr(self, "_live_usage_latest_runtime", {}) or {}
            ),
        }
        self._write_json_log("live_usage_summary.json", payload)

    def _record_live_usage_event(
        self,
        event_type: str,
        label: str,
        confidence: float,
        *,
        executed: bool = False,
        route_metadata: dict[str, Any] | None = None,
        reason: str = "",
        command_info: str = "",
    ) -> None:
        metadata = route_metadata if isinstance(route_metadata, dict) else {}
        route_fields = self._live_evaluation_route_fields(metadata)
        now = time.time()
        row: dict[str, Any] = {
            "recorded_at": now,
            "event_type": str(event_type or ""),
            "label": str(label or "").strip(),
            "confidence": float(confidence or 0.0),
            "executed": bool(executed),
            "reason": str(reason or ""),
            "command_info": str(command_info or ""),
            "recognition_model_mode": self.recognition_model_mode,
            "dynamic_model_profile": self.dynamic_model_profile,
            "static_rejection_method": self.static_rejection_method,
            "gesture_mode": bool(getattr(self, "_gesture_mode", False)),
            "pointer_mode": bool(getattr(self, "_pointer_mode", False)),
            "auto_execute": bool(
                getattr(self, "_auto_execute_on_gesture", False)
            ),
        }
        row.update(route_fields)
        events = getattr(self, "_live_usage_events", None)
        if not isinstance(events, list):
            events = []
            self._live_usage_events = events
        events.append(row)
        del events[:-LIVE_USAGE_EVENTS_LIMIT]
        self._append_jsonl_log("live_usage_events.jsonl", row)
        self._write_live_usage_summary()

    def _dispatch_infer_result(self, out: dict[str, Any]) -> None:
        self._update_live_evaluation_timeout()
        performance = out.get("performance")
        if isinstance(performance, dict):
            self._record_runtime_performance(performance)
        label = (out.get("label") or "").strip()
        conf = float(out.get("confidence") or 0.0)
        route_metadata = out.get("router") if isinstance(out.get("router"), dict) else {}
        if not route_metadata and out.get("route"):
            route_metadata = {"route": out.get("route")}
        lj = out.get("landmarks_json") or "[]"
        self._set_landmarks(lj)
        self._update_pointer_from_landmarks(lj)
        if not self._gesture_mode:
            self._set_confidence(0.0)
            self._reset_gesture_confirmation()
            if self._last_label:
                self._last_label = ""
                self.gesture_detected.emit("")
            return
        self._set_confidence(conf)
        if not label:
            self._reset_gesture_confirmation()
            if self._last_label:
                self._last_label = ""
                self.gesture_detected.emit("")
            return
        confirmed, stable_conf = self._update_gesture_confirmation(
            label,
            conf,
            route_metadata=route_metadata,
        )
        if not confirmed:
            return
        if label and label != self._last_label:
            evaluation_active = self.live_evaluation_active()
            route = str(route_metadata.get("route") or "")
            if self._is_negative_label(label):
                self._consume_live_evaluation_prediction(
                    label,
                    stable_conf,
                    route_metadata=route_metadata,
                )
                self._last_label = label
                self._emit_live_gesture_state(
                    self._ensure_live_gesture_state().mark_rejected(
                        label,
                        stable_conf,
                        reason="negative_label",
                        route=route,
                    )
                )
                self._set_status(f"Rejected gesture evidence: {label}")
                self._record_recognition_event(label, stable_conf, False)
                self._record_live_usage_event(
                    "gesture_rejected",
                    label,
                    stable_conf,
                    executed=False,
                    route_metadata=route_metadata,
                    reason="negative_label",
                )
                return

            suppressed_reason = self._dynamic_return_guard_active(
                label,
                route_metadata,
            )
            if suppressed_reason:
                self._last_label = label
                self._reset_gesture_confirmation()
                self._emit_live_gesture_state(
                    self._ensure_live_gesture_state().mark_suppressed(
                        label,
                        stable_conf,
                        reason=suppressed_reason,
                        route=route,
                    )
                )
                print(
                    f"[ctrl.gesture] suppressed={label!r} "
                    f"reason={suppressed_reason}",
                    flush=True,
                )
                self._set_status(f"Suppressed return motion: {label}")
                self._record_live_usage_event(
                    "gesture_suppressed",
                    label,
                    stable_conf,
                    executed=False,
                    route_metadata=route_metadata,
                    reason=suppressed_reason,
                )
                return

            print(
                f"[ctrl.gesture] detected={label!r} conf={stable_conf:.3f} "
                f"frames={self._pending_frames} prev={self._last_label!r}",
                flush=True,
            )
            self._consume_live_evaluation_prediction(
                label,
                stable_conf,
                route_metadata=route_metadata,
            )
            if str(route_metadata.get("route") or "") == "dynamic":
                acknowledger = getattr(
                    getattr(self, "_embedded_infer", None),
                    "acknowledge_dynamic_event",
                    None,
                )
                if callable(acknowledger):
                    acknowledger()
                self._mark_dynamic_event_accepted(label, route_metadata)
            self._last_label = label
            self.gesture_detected.emit(label)
            # Главное: при детекции жеста сразу запускаем команду через БД-
            # бридж (R4/R5/R6 политика — порог уверенности, cooldown, фильтр
            # опасных действий). Это даёт «жест → команда ОС» — главную фичу
            # диплома.
            if self._auto_execute_on_gesture and not evaluation_active:
                executed = self.execute_for_gesture(label, stable_conf)
                info = str(getattr(self, "_last_execute_info", "") or "")
                if not executed:
                    marker = (
                        self._ensure_live_gesture_state().mark_cooldown
                        if info == "cooldown"
                        else self._ensure_live_gesture_state().mark_rejected
                    )
                    self._emit_live_gesture_state(
                        marker(
                            label,
                            stable_conf,
                            reason=info or "command_not_executed",
                            route=route,
                        )
                    )
                event_type = (
                    "command_executed"
                    if executed
                    else "command_cooldown"
                    if info == "cooldown"
                    else "command_rejected"
                )
                self._record_live_usage_event(
                    event_type,
                    label,
                    stable_conf,
                    executed=executed,
                    route_metadata=route_metadata,
                    reason=info or "",
                    command_info=info,
                )
            else:
                executed = False
                reason = (
                    "live-evaluation активен"
                    if evaluation_active
                    else "auto-execute выключен"
                )
                print(f"[ctrl.gesture] {reason} — команда не запускается", flush=True)
                self._record_live_usage_event(
                    "gesture_confirmed",
                    label,
                    stable_conf,
                    executed=False,
                    route_metadata=route_metadata,
                    reason=reason,
                )
            self._record_recognition_event(label, stable_conf, executed)

    def _record_runtime_performance(self, performance: dict[str, Any]) -> None:
        try:
            total_ms = float(performance.get("total_inference_ms") or 0.0)
            detection_ms = float(performance.get("detection_ms") or 0.0)
        except (TypeError, ValueError):
            return
        if total_ms <= 0.0:
            return

        samples = getattr(self, "_runtime_inference_samples", None)
        if not isinstance(samples, list):
            samples = []
            self._runtime_inference_samples = samples
        samples.append(
            {
                "total_inference_ms": total_ms,
                "detection_ms": max(0.0, detection_ms),
                "shared_detection": bool(performance.get("shared_detection")),
                "camera_frame_width": int(performance.get("camera_frame_width") or 0),
                "camera_frame_height": int(performance.get("camera_frame_height") or 0),
                "inference_frame_width": int(
                    performance.get("inference_frame_width") or 0
                ),
                "inference_frame_height": int(
                    performance.get("inference_frame_height") or 0
                ),
                "camera_frame_sequence": int(
                    performance.get("camera_frame_sequence") or 0
                ),
                "preview_max_fps": float(
                    performance.get("preview_max_fps") or CAMERA_PREVIEW_MAX_FPS
                ),
                "preview_enabled": bool(performance.get("preview_enabled", True)),
                "inference_max_fps": float(
                    performance.get("inference_max_fps") or CAMERA_INFERENCE_MAX_FPS
                ),
                "inference_frame_policy": str(
                    performance.get("inference_frame_policy") or ""
                ),
                "inference_queue_depth": int(
                    performance.get("inference_queue_depth") or 0
                ),
                "capture_to_inference_ms": float(
                    performance.get("capture_to_inference_ms") or 0.0
                ),
                "dynamic_window_frames": int(
                    performance.get("dynamic_window_frames")
                    or DYNAMIC_RECOGNITION_WINDOW
                ),
                "mediapipe_profile": str(performance.get("mediapipe_profile") or ""),
                "mediapipe_min_detection_confidence": float(
                    performance.get("mediapipe_min_detection_confidence") or 0.0
                ),
                "mediapipe_min_presence_confidence": float(
                    performance.get("mediapipe_min_presence_confidence") or 0.0
                ),
                "mediapipe_min_tracking_confidence": float(
                    performance.get("mediapipe_min_tracking_confidence") or 0.0
                ),
                "mediapipe_smoothing_alpha": float(
                    performance.get("mediapipe_smoothing_alpha") or 0.0
                ),
                "mediapipe_timestamp_source": str(
                    performance.get("mediapipe_timestamp_source") or ""
                ),
                "mediapipe_detection_ms": float(
                    performance.get("mediapipe_detection_ms") or 0.0
                ),
                "hand_detected": bool(performance.get("hand_detected", False)),
                "hand_count": int(performance.get("hand_count") or 0),
                "handedness_score_max": float(
                    performance.get("handedness_score_max") or 0.0
                ),
                "landmark_z_available": bool(
                    performance.get("landmark_z_available", False)
                ),
                "world_landmarks_available": bool(
                    performance.get("world_landmarks_available", False)
                ),
                "landmark_z_range": float(
                    performance.get("landmark_z_range") or 0.0
                ),
                "world_z_range": float(performance.get("world_z_range") or 0.0),
                "hand_bbox_area": float(performance.get("hand_bbox_area") or 0.0),
                "hand_bbox_diag": float(performance.get("hand_bbox_diag") or 0.0),
                "primary_wrist_step": float(
                    performance.get("primary_wrist_step") or 0.0
                ),
                "hand_lost_streak": int(performance.get("hand_lost_streak") or 0),
                "hand_lost_grace_frames": int(
                    performance.get("hand_lost_grace_frames") or 0
                ),
            }
        )
        if len(samples) > 300:
            del samples[:-300]

        now = time.monotonic()
        last_flush = float(getattr(self, "_runtime_perf_last_flush", now))
        if now - last_flush < RUNTIME_PERFORMANCE_FLUSH_SECONDS:
            return
        self._runtime_perf_last_flush = now

        total_values = sorted(item["total_inference_ms"] for item in samples)
        detection_values = sorted(item["detection_ms"] for item in samples)
        capture_latency_values = sorted(
            item["capture_to_inference_ms"] for item in samples
        )
        p95_index = min(
            len(total_values) - 1,
            max(0, int(round((len(total_values) - 1) * 0.95))),
        )
        average_ms = sum(total_values) / len(total_values)
        last_sample = samples[-1]

        def _mean(key: str) -> float:
            values = [float(item.get(key) or 0.0) for item in samples]
            return sum(values) / len(values) if values else 0.0

        def _rate(key: str) -> float:
            return (
                sum(1 for item in samples if bool(item.get(key))) / len(samples)
                if samples
                else 0.0
            )

        row = {
            "recorded_at": time.time(),
            "recognition_model_mode": self.recognition_model_mode,
            "dynamic_model_profile": self.dynamic_model_profile,
            "target_fps": int(getattr(self, "_target_fps", 0) or 0),
            "preview_max_fps": round(float(last_sample.get("preview_max_fps") or 0.0), 2),
            "preview_enabled": bool(last_sample.get("preview_enabled", True)),
            "inference_max_fps": round(
                float(last_sample.get("inference_max_fps") or 0.0),
                2,
            ),
            "dynamic_window_frames": int(
                last_sample.get("dynamic_window_frames") or DYNAMIC_RECOGNITION_WINDOW
            ),
            "mediapipe_profile": str(last_sample.get("mediapipe_profile") or ""),
            "mediapipe_min_detection_confidence": round(
                float(last_sample.get("mediapipe_min_detection_confidence") or 0.0),
                3,
            ),
            "mediapipe_min_presence_confidence": round(
                float(last_sample.get("mediapipe_min_presence_confidence") or 0.0),
                3,
            ),
            "mediapipe_min_tracking_confidence": round(
                float(last_sample.get("mediapipe_min_tracking_confidence") or 0.0),
                3,
            ),
            "mediapipe_smoothing_alpha": round(
                float(last_sample.get("mediapipe_smoothing_alpha") or 0.0),
                3,
            ),
            "mediapipe_timestamp_source": str(
                last_sample.get("mediapipe_timestamp_source") or ""
            ),
            "mediapipe_real_timestamp_rate": round(
                sum(
                    1
                    for item in samples
                    if str(item.get("mediapipe_timestamp_source") or "").startswith(
                        "real_"
                    )
                )
                / len(samples),
                4,
            ),
            "camera_frame_width": int(last_sample.get("camera_frame_width") or 0),
            "camera_frame_height": int(last_sample.get("camera_frame_height") or 0),
            "inference_frame_width": int(last_sample.get("inference_frame_width") or 0),
            "inference_frame_height": int(last_sample.get("inference_frame_height") or 0),
            "camera_frame_sequence": int(last_sample.get("camera_frame_sequence") or 0),
            "inference_frame_policy": str(
                last_sample.get("inference_frame_policy") or ""
            ),
            "inference_queue_depth_max": int(
                max(item["inference_queue_depth"] for item in samples)
            ),
            "capture_to_inference_ms_avg": round(
                sum(capture_latency_values) / len(capture_latency_values),
                3,
            ),
            "capture_to_inference_ms_p95": round(
                capture_latency_values[p95_index],
                3,
            ),
            "samples": len(samples),
            "shared_detection_rate": round(
                sum(1 for item in samples if item["shared_detection"]) / len(samples),
                4,
            ),
            "inference_ms_avg": round(average_ms, 3),
            "inference_ms_p95": round(total_values[p95_index], 3),
            "detection_ms_avg": round(
                sum(detection_values) / len(detection_values),
                3,
            ),
            "mediapipe_detection_ms_avg": round(
                _mean("mediapipe_detection_ms"),
                3,
            ),
            "hand_detected_rate": round(_rate("hand_detected"), 4),
            "hand_count_avg": round(_mean("hand_count"), 3),
            "handedness_score_max_avg": round(_mean("handedness_score_max"), 4),
            "landmark_z_available_rate": round(_rate("landmark_z_available"), 4),
            "world_landmarks_available_rate": round(
                _rate("world_landmarks_available"),
                4,
            ),
            "landmark_z_range_avg": round(_mean("landmark_z_range"), 6),
            "world_z_range_avg": round(_mean("world_z_range"), 6),
            "hand_bbox_area_avg": round(_mean("hand_bbox_area"), 6),
            "hand_bbox_diag_avg": round(_mean("hand_bbox_diag"), 6),
            "primary_wrist_step_avg": round(_mean("primary_wrist_step"), 6),
            "hand_lost_streak_max": int(
                max(int(item.get("hand_lost_streak") or 0) for item in samples)
            ),
            "hand_lost_grace_frames": int(
                last_sample.get("hand_lost_grace_frames") or 0
            ),
            "inference_fps_capacity": round(1000.0 / average_ms, 2),
        }
        self._live_usage_latest_runtime = dict(row)
        try:
            path = self._configured_log_dir() / "runtime_performance.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[w] runtime performance log write failed: {exc}", flush=True)
        self._write_live_usage_summary()
        samples.clear()

    # ----------------------------------------------------------------------
    # Встроенный пайплайн распознавания
    # ----------------------------------------------------------------------

    def start_embedded_recognition(self) -> None:
        print(f"[ctrl.start_embedded] already_active={self._embedded_active}", flush=True)
        if self._embedded_active:
            return
        self.stop_recognition()
        self._set_recognizing(True)
        self._set_status("CV: загрузка MediaPipe…")
        self._embedded_infer = None
        self._embedded_active = True
        self._last_label = ""
        self._reset_gesture_confirmation(immediate_ui=True)
        self._set_confidence(0.0)
        self._set_landmarks("[]")
        if not self._is_camera_active:
            self.start_camera()
        if not self._is_camera_active:
            self._embedded_active = False
            self._set_recognizing(False)

    def stop_embedded_recognition(self) -> None:
        if not self._embedded_active and self._embedded_infer is None:
            return
        self._embedded_active = False
        self._last_label = ""
        self._reset_gesture_confirmation(immediate_ui=True)
        self._reset_pointer_control()
        self._set_confidence(0.0)
        self._set_landmarks("[]")
        infer = self._embedded_infer
        self._embedded_infer = None
        if infer is not None:
            try:
                infer.close()
            except Exception:
                pass
        if (
            self._status.startswith("CV:")
            or self._status.startswith("Распознавание")
            or self._status.startswith("Указатель:")
        ):
            self._set_status("Idle")
        if not self._is_recognition_pid_active():
            self._set_recognizing(False)

    # ----------------------------------------------------------------------
    # Subprocess realtime_infer (фоновое распознавание для глобальных команд)
    # ----------------------------------------------------------------------

    def start_recognition(self) -> None:
        self.stop_camera()
        if self._is_recognition_pid_active():
            self._set_recognizing(True)
            self._set_status("Recognizing in background")
            return

        infer_script = Path(__file__).resolve().parent.parent.parent / "cv" / "realtime_infer.py"
        cmd = [
            sys.executable,
            str(infer_script),
            "--tts",
            "--model",
            str(self._configured_model_path()),
            "--classes",
            str(self._configured_classes_path()),
            "--feature-dim-file",
            str(self._configured_feature_dim_path()),
            "--camera-index",
            str(int(self._config.recognition.camera_index)),
            "--fps",
            str(int(self._config.recognition.target_fps)),
        ]
        if self._two_hands_mode:
            cmd.append("--two-hands")
        try:
            log_handle = self._recognition_log_file.open("a", encoding="utf-8")
            self._recognition_process = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._recognition_pid_file.write_text(
                str(self._recognition_process.pid), encoding="utf-8"
            )
            self._set_recognizing(True)
            self._set_status("Recognizing in background")
            print(f"[✓] Фоновое распознавание запущено, PID={self._recognition_process.pid}")
        except Exception as e:
            print(f"[!] Ошибка запуска распознавания: {e}")
            self._set_recognizing(False)
            self._set_status("Recognition start failed")

    def stop_recognition(self) -> None:
        pid = self._read_recognition_pid()
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"[!] Ошибка остановки PID={pid}: {e}")
        if self._recognition_pid_file.exists():
            try:
                self._recognition_pid_file.unlink()
            except OSError:
                pass
        self._recognition_process = None
        self._set_recognizing(False)
        if self._status.startswith("Recognizing"):
            self._set_status("Stopped")

    def _read_recognition_pid(self) -> Optional[int]:
        if not self._recognition_pid_file.exists():
            return None
        try:
            return int(self._recognition_pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            return None

    def _is_recognition_pid_active(self) -> bool:
        pid = self._read_recognition_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            try:
                self._recognition_pid_file.unlink()
            except OSError:
                pass
            return False

    # ----------------------------------------------------------------------
    # Режим двух рук
    # ----------------------------------------------------------------------

    def set_two_hands_mode(self, enabled: bool) -> None:
        target = bool(enabled)
        if target == self._two_hands_mode:
            return
        self._two_hands_mode = target
        self.two_hands_changed.emit(target)
        if self._embedded_infer is not None:
            try:
                self._embedded_infer.set_two_hands(target)
            except Exception as e:
                print(f"[!] set_two_hands: {e}")
        if self._is_recognition_pid_active():
            self.stop_recognition()
            self.start_recognition()

    def set_recognition_model_mode(self, mode: str) -> None:
        target = str(mode or "").strip().lower()
        if target not in {
            RECOGNITION_MODEL_AUTO,
            RECOGNITION_MODEL_STATIC,
            RECOGNITION_MODEL_DYNAMIC,
        }:
            target = RECOGNITION_MODEL_AUTO
        if target == self.recognition_model_mode:
            return

        self._recognition_model_mode = target
        self._reset_gesture_confirmation(immediate_ui=True)
        self._set_confidence(0.0)
        if self._last_label:
            self._last_label = ""
            self.gesture_detected.emit("")

        infer = self._embedded_infer
        self._embedded_infer = None
        if infer is not None:
            try:
                infer.close()
            except Exception:
                pass

        event = getattr(self, "recognition_model_mode_changed", None)
        if event is not None:
            event.emit(target)
        if self._embedded_active:
            self._set_status(self._live_recognition_status())

    def set_dynamic_model_profile(self, profile: str) -> None:
        target = str(profile or "").strip().lower()
        if target not in DYNAMIC_MODEL_PROFILES:
            target = DYNAMIC_MODEL_PROFILE_PRODUCTION
        if target == self.dynamic_model_profile:
            return

        self._dynamic_model_profile = target
        self._reset_gesture_confirmation(immediate_ui=True)
        self._set_confidence(0.0)
        if self._last_label:
            self._last_label = ""
            self.gesture_detected.emit("")

        infer = self._embedded_infer
        self._embedded_infer = None
        if infer is not None:
            try:
                infer.close()
            except Exception:
                pass

        event = getattr(self, "dynamic_model_profile_changed", None)
        if event is not None:
            event.emit(target)
        if self._embedded_active:
            self._set_status(self._live_recognition_status())

    def set_static_rejection_method(self, method: str) -> None:
        target = str(method or "").strip().lower()
        if target not in STATIC_REJECTION_METHODS:
            target = STATIC_REJECTION_OPEN_SET_POLICY
        if target == self.static_rejection_method:
            return

        self._static_rejection_method = target
        self._reset_gesture_confirmation(immediate_ui=True)
        self._set_confidence(0.0)
        if self._last_label:
            self._last_label = ""
            self.gesture_detected.emit("")

        infer = self._embedded_infer
        self._embedded_infer = None
        if infer is not None:
            try:
                infer.close()
            except Exception:
                pass

        event = getattr(self, "static_rejection_method_changed", None)
        if event is not None:
            event.emit(target)
        if self._embedded_active:
            self._set_status(self._live_recognition_status())

    def set_show_landmark_overlay(self, enabled: bool) -> None:
        target = bool(enabled)
        if target == self._show_landmark_overlay:
            return
        self._show_landmark_overlay = target
        self.landmark_overlay_changed.emit(target)
        if self._embedded_active:
            self._set_status(self._live_recognition_status())

    def set_gesture_mode(self, enabled: bool) -> None:
        target = bool(enabled)
        if target == self._gesture_mode:
            return
        self._gesture_mode = target
        self._reset_gesture_confirmation(immediate_ui=True)
        if not target:
            self._set_confidence(0.0)
            if self._last_label:
                self._last_label = ""
                self.gesture_detected.emit("")
        self.gesture_mode_changed.emit(target)
        if self._embedded_active:
            self._set_status(self._live_recognition_status())

    def set_pointer_mode(self, enabled: bool) -> None:
        target = bool(enabled)
        if target == self._pointer_mode:
            return
        self._pointer_mode = target
        self._reset_pointer_control()
        self._emit_pointer_state(state="idle", moved=False, clicked=False)
        self.pointer_mode_changed.emit(target)
        if self._embedded_active:
            self._set_status(self._live_recognition_status())
        elif self._status.startswith("Указатель:"):
            self._set_status("Idle")

    # ----------------------------------------------------------------------
    # Команды
    # ----------------------------------------------------------------------

    def execute_command(self, name: str) -> bool:
        if not name:
            return False
        if not self._command_executor:
            print("[w] CommandExecutor недоступен")
            return False
        ok = bool(self._command_executor.execute(name))
        if ok:
            self.command_executed.emit(name)
        return ok

    def list_commands(self) -> list[dict[str, Any]]:
        if not self._command_executor:
            return []
        items: list[dict[str, Any]] = []
        for n, cfg in self._command_executor.commands_registry.items():
            action = str(cfg.get("action") or "")
            items.append(
                {
                    "name": n,
                    "description": cfg.get("description", ""),
                    "platform": cfg.get("platform", "all"),
                    "action": action,
                    "category": category_for_action(action),
                    "config": dict(cfg),
                }
            )
        return sorted(items, key=lambda item: str(item.get("name") or ""))

    def list_db_commands(self) -> list[dict[str, Any]]:
        """Вернуть пользовательские команды, сохранённые в таблице commands."""
        if not BINDING_SERVICES_AVAILABLE or DbCommand is None:
            return []
        try:
            if not self._db_initialized:
                init_database()
                self._db_initialized = True
            session = get_db_session()
        except Exception as e:
            print(f"[!] list_db_commands: {e}")
            return []
        try:
            import json as _json

            rows = session.query(DbCommand).order_by(DbCommand.name).all()
            out: list[dict[str, Any]] = []
            for cmd in rows:
                spec: dict[str, Any] = {}
                raw_spec = (cmd.action_spec or "").strip()
                if raw_spec:
                    try:
                        parsed = _json.loads(raw_spec)
                        if isinstance(parsed, dict):
                            spec = parsed
                    except _json.JSONDecodeError:
                        spec = {}
                action = str(spec.get("action") or "")
                gesture = getattr(cmd, "gesture", None)
                out.append(
                    {
                        "id": int(cmd.id or 0),
                        "name": str(cmd.name or ""),
                        "platform": str(cmd.platform or "all"),
                        "isActive": bool(cmd.is_active),
                        "gestureLabel": str(getattr(gesture, "label", "") or ""),
                        "action": action,
                        "category": category_for_action(action),
                        "actionSpec": spec,
                    }
                )
            return out
        finally:
            session.close()

    def _forget_executor_command(self, name: str) -> None:
        registry = getattr(self._command_executor, "commands_registry", None)
        if isinstance(registry, dict):
            registry.pop((name or "").strip().lower(), None)

    def delete_db_command(self, command_id: int) -> dict[str, Any]:
        """Удалить пользовательскую команду из БД и убрать её из executor registry."""
        summary: dict[str, Any] = {
            "ok": False,
            "deleted": 0,
            "historyDeleted": 0,
            "name": "",
            "error": "",
        }
        if (
            not BINDING_SERVICES_AVAILABLE
            or DbCommand is None
            or DbGestureHistory is None
        ):
            summary["error"] = "Сервис команд недоступен"
            return summary
        try:
            if not self._db_initialized:
                init_database()
                self._db_initialized = True
            session = get_db_session()
        except Exception as e:
            summary["error"] = f"БД недоступна: {e}"
            return summary
        try:
            cmd = session.query(DbCommand).filter(DbCommand.id == int(command_id)).first()
            if cmd is None:
                summary["error"] = "Команда не найдена"
                return summary
            command_name = str(cmd.name or "")
            history_deleted = (
                session.query(DbGestureHistory)
                .filter(DbGestureHistory.command_id == cmd.id)
                .delete(synchronize_session=False)
            )
            session.delete(cmd)
            session.commit()

            self._forget_executor_command(command_name)
            if self._command_executor and sync_db_commands_to_executor:
                try:
                    sync_db_commands_to_executor(session, self._command_executor)
                except Exception as e:
                    print(f"[w] sync_db_commands_to_executor after delete: {e}")

            summary.update(
                {
                    "ok": True,
                    "deleted": 1,
                    "historyDeleted": int(history_deleted or 0),
                    "name": command_name,
                }
            )
            return summary
        except Exception as e:
            try:
                session.rollback()
            except Exception:
                pass
            summary["error"] = f"Ошибка удаления: {e}"
            return summary
        finally:
            session.close()

    # ----------------------------------------------------------------------
    # Главная связка: один тумблер «запустить/остановить» (embedded)
    # ----------------------------------------------------------------------

    def toggle_recognition(self) -> None:
        """Главный метод для UI: запускает или останавливает встроенное
        распознавание (камера + MediaPipe + KNN + БД-бридж).

        В отличие от ``start_recognition`` (запускающего subprocess
        ``cv/realtime_infer.py``), здесь всё работает прямо в процессе GUI.
        """
        if (
            self._is_recognizing
            or self._embedded_active
            or self._is_recognition_pid_active()
        ):
            self.cancel_live_evaluation()
            self.stop_embedded_recognition()
            self.stop_recognition()
            self.stop_camera()
            self._set_recognizing(False)
        else:
            self.start_embedded_recognition()

    # ----------------------------------------------------------------------
    # Автозапуск команды и UI-помощники для экрана привязок
    # ----------------------------------------------------------------------

    def set_auto_execute(self, enabled: bool) -> None:
        self._auto_execute_on_gesture = bool(enabled)

    @property
    def auto_execute(self) -> bool:
        return self._auto_execute_on_gesture

    @property
    def auto_start_recognition(self) -> bool:
        return bool(self._config.recognition.auto_start_recognition)

    def _ensure_db_bridge(self) -> Any | None:
        """Лениво инициализирует БД и ``GestureCommandBridge``. ``None`` при ошибке."""
        if not BINDING_SERVICES_AVAILABLE:
            return None
        if self._gesture_command_bridge is not None:
            return self._gesture_command_bridge
        try:
            if not self._db_initialized:
                init_database()
                self._db_initialized = True
            self._gesture_command_bridge = GestureCommandBridge(
                session_factory=get_db_session,
                executor=self._command_executor,
            )
            return self._gesture_command_bridge
        except Exception as e:
            print(f"[!] db bridge init failed: {e}")
            return None

    def execute_for_gesture(self, gesture_label: str, confidence: float) -> bool:
        """
        Главная фича: метка жеста + уверенность → проверка R4/R5 → команда ОС.

        Применяется автоматически из ``_dispatch_infer_result``. Можно
        вызывать и вручную из UI (например, кнопкой «Тестовый запуск»).
        """
        bridge = self._ensure_db_bridge()
        if bridge is None:
            print(f"[ctrl.exec] {gesture_label!r}: BRIDGE_UNAVAILABLE", flush=True)
            self._last_execute_info = "bridge_unavailable"
            return False
        try:
            ok, info = bridge.execute(
                gesture_label,
                record_history=True,
                confidence=float(confidence),
                apply_policy=True,
            )
        except Exception as e:
            print(f"[!] execute_for_gesture {gesture_label!r}: {e}", flush=True)
            self._last_execute_info = str(e)
            return False
        self._last_execute_info = str(info or "")
        print(
            f"[ctrl.exec] {gesture_label!r} conf={confidence:.3f} -> ok={ok} info={info!r}",
            flush=True,
        )
        if ok:
            self.command_executed.emit(info)
        return bool(ok)

    def get_recent_recognition_events(self, limit: int = 8) -> list[dict[str, Any]]:
        if not BINDING_SERVICES_AVAILABLE:
            return []
        try:
            if not self._db_initialized:
                init_database()
                self._db_initialized = True
            session = get_db_session()
        except Exception as e:
            print(f"[!] get_recent_recognition_events: {e}")
            return []
        try:
            from app.services.recognition_events import list_recent_recognition_events

            return list_recent_recognition_events(session, limit=limit)
        except Exception as e:
            print(f"[!] list_recent_recognition_events: {e}")
            return []
        finally:
            session.close()

    def _record_recognition_event(
        self,
        label: str,
        confidence: float,
        executed: bool,
    ) -> None:
        if not BINDING_SERVICES_AVAILABLE:
            return
        try:
            if not self._db_initialized:
                init_database()
                self._db_initialized = True
            session = get_db_session()
        except Exception as e:
            print(f"[!] record_recognition_event init: {e}")
            return
        try:
            from app.services.recognition_events import record_recognition_event

            record_recognition_event(
                session,
                label=label,
                confidence=confidence,
                executed=executed,
            )
            self.recognition_event_recorded.emit()
        except Exception as e:
            try:
                session.rollback()
            except Exception:
                pass
            print(f"[!] record_recognition_event: {e}")
        finally:
            session.close()

    # ---- Методы, нужные экрану «Привязки» --------------------------------

    def get_db_gestures(self) -> list[dict[str, Any]]:
        if not BINDING_SERVICES_AVAILABLE:
            return []
        try:
            if not self._db_initialized:
                init_database()
                self._db_initialized = True
            session = get_db_session()
        except Exception as e:
            print(f"[!] get_db_gestures: {e}")
            return []
        try:
            def _read_classes(path: Path) -> set[str]:
                if not path.exists():
                    return set()
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    return set()
                if not isinstance(raw, list):
                    return set()
                return {
                    str(item).strip().lower()
                    for item in raw
                    if str(item).strip()
                }

            static_classes = _read_classes(self._configured_classes_path())
            dynamic_classes = _read_classes(self._dynamic_classes_path())
            rows = (
                session.query(DbGesture)
                .filter(DbGesture.is_active.is_(True))
                .order_by(DbGesture.label)
                .all()
            )
            current = {
                row.gesture_id: row
                for row in session.query(DbCommand)
                .filter(DbCommand.gesture_id.isnot(None))
                .all()
            }
            out: list[dict[str, Any]] = []
            for g in rows:
                label_value = str(g.label or "").strip()
                system_class = self._is_system_internal_gesture_label(label_value)
                if system_class:
                    continue
                samples = list(getattr(g, "samples", []) or [])
                user_samples = [
                    sample
                    for sample in samples
                    if self._db_sample_is_user_recorded(sample)
                ]
                try:
                    sample_count = len(user_samples)
                except Exception:
                    sample_count = 0
                visible_sample_count = sample_count
                samples_path = str(getattr(g, "samples_path", "") or "").strip()
                preview_path = ""
                if user_samples:
                    preview_path = str(
                        getattr(user_samples[0], "features_path", "") or ""
                    )
                elif samples:
                    preview_path = str(getattr(samples[0], "features_path", "") or "")
                    visible_sample_count = len(samples)
                legacy_user_samples: list[Path] = []
                legacy_visible_samples: list[Path] = []
                if samples_path:
                    try:
                        sample_dir = self._configured_path(samples_path)
                        if sample_count <= 0:
                            legacy_user_samples = (
                                self._legacy_user_sample_paths_for_gesture(
                                    g,
                                    sample_dir,
                                )
                            )
                            sample_count = len(legacy_user_samples)
                            visible_sample_count = max(
                                visible_sample_count,
                                sample_count,
                            )
                            if visible_sample_count <= 0:
                                legacy_visible_samples = gesture_sample_paths(sample_dir)
                                visible_sample_count = len(legacy_visible_samples)
                        if not preview_path and legacy_user_samples:
                            preview_path = str(legacy_user_samples[0])
                        if not preview_path and legacy_visible_samples:
                            preview_path = str(legacy_visible_samples[0])
                    except Exception:
                        pass
                if visible_sample_count <= 0:
                    continue
                command = current.get(g.id)
                action_spec: dict[str, Any] = {}
                raw_spec = (
                    str(getattr(command, "action_spec", "") or "").strip()
                    if command is not None
                    else ""
                )
                if raw_spec:
                    try:
                        loaded = json.loads(raw_spec)
                        if isinstance(loaded, dict):
                            action_spec = loaded
                    except json.JSONDecodeError:
                        action_spec = {}
                out.append(
                    {
                        "id": int(g.id),
                        "label": label_value,
                        "description": str(g.description or ""),
                        "samplesPath": str(getattr(g, "samples_path", "") or ""),
                        "samplePreviewPath": preview_path,
                        "sampleCount": int(visible_sample_count),
                        "userRecordedSamples": int(sample_count),
                        "standardClass": bool(sample_count <= 0),
                        "systemClass": False,
                        "canDelete": bool(sample_count > 0),
                        "gestureType": (
                            GESTURE_TYPE_DYNAMIC
                            if label_value.lower() in dynamic_classes
                            else ""
                        ),
                        "isTwoHands": bool(
                            getattr(g, "is_two_hands", False) or False
                        ),
                        "boundCommandName": str(getattr(command, "name", "") or ""),
                        "boundCommandDescription": str(
                            getattr(command, "description", "") or ""
                        ),
                        "boundCommandPlatform": str(
                            getattr(command, "platform", "") or "all"
                        ),
                        "boundCommandScriptPath": str(
                            getattr(command, "script_path", "") or ""
                        ),
                        "boundCommandActionSpec": action_spec,
                    }
                )
            return out
        finally:
            session.close()

    def _is_auto_imported_gesture(self, gesture: Any) -> bool:
        description = str(getattr(gesture, "description", "") or "").strip()
        return description.startswith("Auto-imported from ")

    def _is_system_internal_gesture_label(self, label: str) -> bool:
        clean = str(label or "").strip().lower()
        if not clean:
            return True
        if clean in SYSTEM_REFERENCE_GESTURE_LABELS:
            return True
        if self._is_no_command_label(clean):
            return True
        try:
            if self._is_negative_label(clean):
                return True
        except Exception:
            pass
        return False

    def _db_sample_path(self, sample: Any) -> Path | None:
        raw = str(getattr(sample, "features_path", "") or "").strip()
        if not raw:
            return None
        try:
            return self._configured_path(raw)
        except Exception:
            path = Path(raw).expanduser()
            if path.is_absolute():
                return path
            return Path.cwd() / path

    def _sample_metadata(self, sample_path: Path) -> dict[str, Any]:
        meta_path = sample_path.with_suffix(".meta.json")
        if not meta_path.exists():
            return {}
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return raw if isinstance(raw, dict) else {}

    def _sample_source_for_db(self, sample_path: Path) -> str:
        metadata = self._sample_metadata(sample_path)
        source = str(metadata.get("source") or "").strip().lower()
        if source:
            return source[:32]
        return sample_source_from_path(sample_path)

    def _sample_path_is_legacy_user_recorded(self, sample_path: Path) -> bool:
        if sample_path.name.startswith("sample_auto_"):
            return False
        metadata = self._sample_metadata(sample_path)
        if str(metadata.get("generated_by") or "").strip():
            return False
        source = str(metadata.get("source") or "").strip().lower()
        if source and source not in USER_RECORDED_SAMPLE_SOURCES:
            return False
        return True

    def _user_recorded_sample_paths_for_label(
        self,
        label: str,
        sample_dir: Path,
    ) -> list[Path]:
        if self._is_system_internal_gesture_label(label):
            return []
        return [
            path
            for path in real_sample_paths(sample_dir)
            if self._sample_path_is_legacy_user_recorded(path)
        ]

    def _legacy_user_sample_paths_for_gesture(
        self,
        gesture: Any,
        sample_dir: Path,
    ) -> list[Path]:
        label = str(getattr(gesture, "label", "") or "")
        return self._user_recorded_sample_paths_for_label(label, sample_dir)

    def _db_sample_is_user_recorded(self, sample: Any) -> bool:
        source = str(getattr(sample, "source", "") or "").strip().lower()
        if source in USER_RECORDED_SAMPLE_SOURCES:
            return True
        sample_path = self._db_sample_path(sample)
        if sample_path is None:
            return False
        metadata = self._sample_metadata(sample_path)
        metadata_source = str(metadata.get("source") or "").strip().lower()
        kind = str(metadata.get("kind") or "real").strip().lower()
        if metadata_source in USER_RECORDED_SAMPLE_SOURCES and kind in {"", "real"}:
            return True
        if not source and self._sample_path_is_legacy_user_recorded(sample_path):
            return True
        return False

    def get_action_categories(self) -> list[dict[str, str]]:
        if not BINDING_SERVICES_AVAILABLE:
            return []
        return [{"id": cid, "label": label} for cid, label in CATEGORY_LABELS.items()]

    def get_actions_for_category(self, category_id: str) -> list[dict[str, Any]]:
        if not BINDING_SERVICES_AVAILABLE:
            return []
        cid = (category_id or "").strip()
        out: list[dict[str, Any]] = []
        for action_name, schema in ACTION_SPEC_SCHEMA.items():
            if schema.get("category") == cid:
                out.append(
                    {
                        "action": action_name,
                        "fields": list((schema.get("fields") or {}).keys()),
                        "fieldHints": dict(schema.get("fields") or {}),
                        "example": schema.get("example") or {},
                    }
                )
        return out

    def is_action_dangerous(self, action: str) -> bool:
        return bool(BINDING_SERVICES_AVAILABLE and action in DANGEROUS_ACTIONS)

    def validate_command_name(self, name: str) -> str:
        if not BINDING_SERVICES_AVAILABLE:
            return ""
        clean = (name or "").strip()
        if not clean:
            return "Имя команды не может быть пустым"
        try:
            if not self._db_initialized:
                init_database()
                self._db_initialized = True
            session = get_db_session()
        except Exception as e:
            return f"БД недоступна: {e}"
        try:
            exists = (
                session.query(DbCommand)
                .filter(DbCommand.name == clean)
                .first()
                is not None
            )
        finally:
            session.close()
        if exists:
            return f"Команда с именем «{clean}» уже существует"
        return ""

    def validate_action_spec_json(self, spec_json: str) -> str:
        if not BINDING_SERVICES_AVAILABLE:
            return ""
        import json as _json

        raw = (spec_json or "").strip()
        if not raw:
            return "action_spec не задан"
        try:
            spec = _json.loads(raw)
        except _json.JSONDecodeError as e:
            return f"action_spec: некорректный JSON: {e}"
        msg = _validate_action_spec(spec)
        return msg or ""

    def save_binding(
        self,
        gesture_label: str,
        command_name: str,
        action_spec_json: str,
    ) -> str:
        """Сохранить привязку «жест → команда» в БД. Пустая строка = ок."""
        if not BINDING_SERVICES_AVAILABLE or save_gesture_binding is None:
            return "Сервис привязок не доступен"

        import json as _json

        label = (gesture_label or "").strip()
        name = (command_name or "").strip()
        if not label:
            return "Не выбран жест"
        if not name:
            return "Имя команды не может быть пустым"

        raw = (action_spec_json or "").strip()
        if not raw:
            return "action_spec не задан"
        try:
            spec = _json.loads(raw)
        except _json.JSONDecodeError as e:
            return f"action_spec: некорректный JSON: {e}"
        err = _validate_action_spec(spec)
        if err:
            return err

        try:
            if not self._db_initialized:
                init_database()
                self._db_initialized = True
            session = get_db_session()
        except Exception as e:
            return f"БД недоступна: {e}"
        try:
            save_gesture_binding(session, label, name, spec, platform="macos")
            if self._command_executor and sync_db_commands_to_executor:
                try:
                    sync_db_commands_to_executor(session, self._command_executor)
                except Exception as e:
                    print(f"[w] sync_db_commands_to_executor: {e}")
            return ""
        except ValueError as e:
            return str(e)
        except Exception as e:
            try:
                session.rollback()
            except Exception:
                pass
            return f"Ошибка сохранения: {e}"
        finally:
            session.close()

    # ---- Настройки политики R4/R5/R6 -------------------------------------

    def get_binding_policy(self) -> dict[str, Any]:
        if not BINDING_SERVICES_AVAILABLE or load_binding_settings is None:
            return {
                "confidenceThreshold": 0.65,
                "cooldownMs": 1500,
                "warnTwoHands": True,
            }
        try:
            if not self._db_initialized:
                init_database()
                self._db_initialized = True
            session = get_db_session()
        except Exception:
            return {
                "confidenceThreshold": 0.65,
                "cooldownMs": 1500,
                "warnTwoHands": True,
            }
        try:
            s = load_binding_settings(session)
            return {
                "confidenceThreshold": float(s.confidence_threshold),
                "cooldownMs": int(s.cooldown_ms),
                "warnTwoHands": bool(s.warn_two_hands),
            }
        finally:
            session.close()

    def set_binding_policy(
        self,
        confidence_threshold: float,
        cooldown_ms: int,
        warn_two_hands: bool,
    ) -> bool:
        if not BINDING_SERVICES_AVAILABLE or save_binding_settings is None:
            return False
        try:
            if not self._db_initialized:
                init_database()
                self._db_initialized = True
            session = get_db_session()
        except Exception:
            return False
        try:
            save_binding_settings(
                session,
                confidence_threshold=float(confidence_threshold),
                cooldown_ms=int(cooldown_ms),
                warn_two_hands=bool(warn_two_hands),
            )
        finally:
            session.close()
        bridge = self._gesture_command_bridge
        if bridge is not None:
            try:
                bridge.reload_policy()
            except Exception as e:
                print(f"[w] reload_policy: {e}")
        return True

    # ---- Голосовой помощник ----------------------------------------------

    def start_voice_assistant(
        self,
        language: str = "ru",
        wake_word: bool = True,
        tts: bool = True,
    ) -> bool:
        return False

    def stop_voice_assistant(self) -> None:
        if self._voice_assistant is None:
            return
        try:
            self._voice_assistant.stop_listening_loop()
        except Exception:
            pass
        self._voice_assistant = None
        self.voice_assistant_state_changed.emit("idle")

    @property
    def is_voice_assistant_active(self) -> bool:
        return self._voice_assistant is not None

    def process_voice_command(self, text: str) -> str:
        return "Голосовой помощник временно отключён"

    # ----------------------------------------------------------------------
    # Обучение: запись примеров и тренировка KNN (обёртки над CLI)
    # ----------------------------------------------------------------------

    def sync_dataset_to_db(self) -> dict[str, int]:
        """
        Просканировать ``data/gestures/<label>/`` и для каждой папки с непустым
        набором ``sample_*.npy`` создать/обновить строки в таблицах
        ``gestures`` и ``gesture_samples``.

        После этого экран «Привязки» сразу увидит словарь жестов (правило R3
        требует ``model_class_id IS NOT NULL`` и ``is_active=True``).
        Если рядом существует ``models/classes.json`` (его пишет
        ``cv/train_classifier.py``), индексы классов проставляются из него.
        Новые, ещё не обученные жесты всё равно попадают в БД, но без
        ``model_class_id``; следующий запуск обучения подхватит их из БД и
        после успешной тренировки сделает доступными для привязок.

        Returns:
            Сводка ``{"created": N, "updated": M, "total": K, "samples": S}``.
        """
        empty_summary = {"created": 0, "updated": 0, "total": 0, "samples": 0}
        if (
            not BINDING_SERVICES_AVAILABLE
            or ensure_gesture is None
            or record_gesture_sample is None
            or sample_index_from_path is None
            or sample_shape_metadata is None
        ):
            return empty_summary

        data_root = self._configured_data_dir()
        if not data_root.exists():
            return empty_summary

        # ``classes.json`` — список меток в порядке индексов классификатора.
        classes_path = self._configured_classes_path()
        class_idx_by_lower: dict[str, int] = {}
        class_label_by_lower: dict[str, str] = {}
        if classes_path.exists():
            try:
                import json as _json

                names = _json.loads(classes_path.read_text())
                for idx, name in enumerate(names):
                    clean_name = str(name).strip()
                    if not clean_name:
                        continue
                    lower = clean_name.lower()
                    class_idx_by_lower.setdefault(lower, idx)
                    class_label_by_lower.setdefault(lower, clean_name)
            except Exception as e:
                print(f"[w] classes.json read failed: {e}")

        try:
            if not self._db_initialized:
                init_database()
                self._db_initialized = True
            session = get_db_session()
        except Exception as e:
            print(f"[!] sync_dataset_to_db: {e}")
            return empty_summary

        created = updated = total = samples_synced = 0
        seen_labels: set[str] = set()
        try:
            for label_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
                samples = gesture_sample_paths(label_dir)
                if not samples:
                    continue
                raw_label = label_dir.name
                label_key = raw_label.lower()
                # Если модель уже обучена, берём точное имя класса из
                # classes.json. До обучения сохраняем пользовательское имя
                # папки, чтобы жесты и команды не зависели от общих алиасов.
                label = class_label_by_lower.get(label_key, raw_label)
                model_class_id = class_idx_by_lower.get(label_key)

                if label not in seen_labels:
                    total += 1
                    seen_labels.add(label)

                row = session.query(DbGesture).filter(DbGesture.label == label).first()
                if row is None:
                    row = (
                        session.query(DbGesture)
                        .filter(DbGesture.label.ilike(label))
                        .first()
                    )
                    if row is None:
                        created += 1
                    else:
                        row.label = label
                        session.flush()
                        updated += 1
                else:
                    updated += 1

                _, first_hand_count = sample_shape_metadata(samples[0])
                is_two_hands = first_hand_count >= 2 if first_hand_count is not None else None
                is_system_label = self._is_system_internal_gesture_label(label)
                sample_sources: list[str] = []
                for sample_path in samples:
                    source = self._sample_source_for_db(sample_path)
                    if (
                        source == "dataset"
                        and not is_system_label
                        and self._sample_path_is_legacy_user_recorded(sample_path)
                    ):
                        source = "user"
                    sample_sources.append(source)
                has_user_recorded_samples = any(
                    source in {"camera", "user", "recording"}
                    for source in sample_sources
                )
                gesture = ensure_gesture(
                    session,
                    label=label,
                    samples_path=label_dir,
                    model_class_id=model_class_id,
                    is_two_hands=is_two_hands,
                    commit=False,
                )
                if has_user_recorded_samples:
                    if self._is_auto_imported_gesture(gesture):
                        gesture.description = ""
                else:
                    gesture.description = (
                        gesture.description or f"Auto-imported from {label_dir}"
                    )
                if classes_path.exists():
                    gesture.model_class_id = model_class_id
                gesture.is_active = True

                for sample_path, sample_source in zip(samples, sample_sources):
                    frames, hand_count = sample_shape_metadata(sample_path)
                    record_gesture_sample(
                        session,
                        label=label,
                        sample_index=sample_index_from_path(sample_path),
                        features_path=sample_path,
                        frames=frames,
                        hand_count=hand_count,
                        source=sample_source,
                        samples_path=label_dir,
                        is_two_hands=is_two_hands,
                        commit=False,
                    )
                    samples_synced += 1
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"[!] sync_dataset_to_db commit: {e}")
        finally:
            session.close()
        return {
            "created": created,
            "updated": updated,
            "total": total,
            "samples": samples_synced,
        }

    def list_recorded_gestures(self) -> list[dict[str, Any]]:
        """Возвращает список папок в ``data/gestures/<label>/`` с числом
        сэмплов в каждой. Используется на вкладке «Обучение»."""
        root = self._configured_data_dir()
        out: list[dict[str, Any]] = []
        if not root.exists():
            return out
        for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            real_samples = real_sample_paths(label_dir)
            augmented_samples = augmented_sample_paths(label_dir)
            if not real_samples:
                continue
            user_recorded_samples = self._user_recorded_sample_paths_for_label(
                label_dir.name,
                label_dir,
            )
            can_delete = bool(user_recorded_samples)
            delete_reason = (
                ""
                if can_delete
                else "Можно удалять только классы, записанные пользователем"
            )
            out.append(
                {
                    "label": label_dir.name,
                    "samples": len(real_samples),
                    "realSamples": len(real_samples),
                    "augmentedSamples": len(augmented_samples),
                    "userRecordedSamples": len(user_recorded_samples),
                    "canDelete": can_delete,
                    "deleteReason": delete_reason,
                    "systemClass": self._is_system_internal_gesture_label(label_dir.name),
                }
            )
        return out

    def _label_dir_for_dataset_label(self, label: str) -> Path | None:
        root = self._configured_data_dir()
        raw = (label or "").strip()
        if not raw or not root.exists():
            return None
        direct = root / raw
        if direct.is_dir():
            return direct
        label_key = raw.lower()
        for item in root.iterdir():
            if item.is_dir() and item.name.lower() == label_key:
                return item
        return None

    def _safe_dataset_npy(self, path: Path, data_root: Path) -> bool:
        try:
            resolved = path.resolve()
            root = data_root.resolve()
            return resolved.is_file() and resolved.suffix == ".npy" and resolved.is_relative_to(root)
        except OSError:
            return False

    def delete_recorded_samples(self, label: str) -> dict[str, Any]:
        """Удалить записанные NPY-семплы для жеста и синхронно почистить ORM."""
        summary: dict[str, Any] = {
            "ok": False,
            "label": (label or "").strip(),
            "filesDeleted": 0,
            "sampleRowsDeleted": 0,
            "commandsUnbound": 0,
            "directoryRemoved": False,
            "error": "",
        }
        raw_label = (label or "").strip()
        if not raw_label:
            summary["error"] = "Не указан жест"
            return summary
        if (
            not BINDING_SERVICES_AVAILABLE
            or DbCommand is None
            or DbGesture is None
            or DbGestureSample is None
        ):
            summary["error"] = "Сервис жестов недоступен"
            return summary

        data_root = self._configured_data_dir()
        label_dir = self._label_dir_for_dataset_label(raw_label)
        file_paths: set[Path] = (
            set(self._user_recorded_sample_paths_for_label(raw_label, label_dir))
            if label_dir
            else set()
        )

        try:
            if not self._db_initialized:
                init_database()
                self._db_initialized = True
            session = get_db_session()
        except Exception as e:
            summary["error"] = f"БД недоступна: {e}"
            return summary

        try:
            label_key = raw_label.lower()
            gesture = session.query(DbGesture).filter(DbGesture.label == label_key).first()
            if gesture is None:
                gesture = (
                    session.query(DbGesture)
                    .filter(DbGesture.label.ilike(raw_label))
                    .first()
                )

            sample_rows: list[Any] = []
            user_sample_rows: list[Any] = []
            if gesture is not None:
                sample_rows = (
                    session.query(DbGestureSample)
                    .filter(DbGestureSample.gesture_id == gesture.id)
                    .all()
                )
                if not self._is_system_internal_gesture_label(str(gesture.label or "")):
                    user_sample_rows = [
                        sample
                        for sample in sample_rows
                        if self._db_sample_is_user_recorded(sample)
                    ]
                if resolve_project_path is not None:
                    for sample in user_sample_rows:
                        stored_path = (sample.features_path or "").strip()
                        if stored_path:
                            file_paths.add(resolve_project_path(stored_path))

                for sample in user_sample_rows:
                    session.delete(sample)

                summary["sampleRowsDeleted"] = len(user_sample_rows)

            if not file_paths and not user_sample_rows:
                summary["error"] = "Можно удалять только классы, записанные пользователем"
                session.rollback()
                return summary

            deleted_files = 0
            for sample_file in sorted(file_paths):
                if not self._safe_dataset_npy(sample_file, data_root):
                    continue
                try:
                    sample_file.unlink()
                    metadata_file = sample_file.with_suffix(".meta.json")
                    if metadata_file.exists():
                        metadata_file.unlink()
                    deleted_files += 1
                except OSError as e:
                    session.rollback()
                    summary["error"] = f"Не удалось удалить файл {sample_file}: {e}"
                    return summary

            if gesture is not None:
                user_sample_set = set(user_sample_rows)
                protected_rows = [
                    sample for sample in sample_rows if sample not in user_sample_set
                ]
                protected_files = set()
                if label_dir is not None:
                    protected_files = set(gesture_sample_paths(label_dir)) - file_paths
                if not protected_rows and not protected_files:
                    commands = (
                        session.query(DbCommand)
                        .filter(DbCommand.gesture_id == gesture.id)
                        .all()
                    )
                    for command in commands:
                        command.gesture_id = None

                    gesture.model_class_id = None
                    gesture.accuracy = None
                    gesture.is_active = False
                    if label_dir is not None:
                        gesture.samples_path = str(label_dir)

                    summary["commandsUnbound"] = len(commands)

            if label_dir is not None:
                try:
                    label_dir.rmdir()
                    summary["directoryRemoved"] = True
                except OSError:
                    summary["directoryRemoved"] = False

            session.commit()
            summary["filesDeleted"] = deleted_files
            summary["ok"] = True
            return summary
        except Exception as e:
            try:
                session.rollback()
            except Exception:
                pass
            summary["error"] = f"Ошибка удаления: {e}"
            return summary
        finally:
            session.close()

    def _active_training_labels_from_db(self) -> list[str]:
        if not BINDING_SERVICES_AVAILABLE or DbGesture is None:
            return []
        try:
            if not getattr(self, "_db_initialized", False):
                init_database()
                self._db_initialized = True
            session = get_db_session()
        except Exception as e:
            print(f"[w] training labels from DB: {e}")
            return []
        try:
            labels: list[str] = []
            seen: set[str] = set()
            rows = (
                session.query(DbGesture.label)
                .filter(DbGesture.is_active.is_(True))
                .order_by(DbGesture.label)
                .all()
            )
            for row in rows:
                label = str(row[0]).strip()
                key = label.lower()
                if not label or key in seen:
                    continue
                labels.append(label)
                seen.add(key)
            return labels
        finally:
            session.close()

    def _active_training_label_profiles_from_db(self) -> list[tuple[str, str]]:
        if (
            not BINDING_SERVICES_AVAILABLE
            or DbGesture is None
            or DbGestureSample is None
            or not getattr(self, "_db_initialized", False)
        ):
            return []
        try:
            session = get_db_session()
        except Exception as e:
            print(f"[w] training label profiles from DB: {e}")
            return []

        try:
            profiles: list[tuple[str, str]] = []
            gestures = (
                session.query(DbGesture)
                .filter(DbGesture.is_active.is_(True))
                .order_by(DbGesture.label)
                .all()
            )
            for gesture in gestures:
                label = str(getattr(gesture, "label", "") or "").strip()
                if not label:
                    continue

                sample_paths: list[Path] = []
                rows = (
                    session.query(DbGestureSample.features_path)
                    .filter(DbGestureSample.gesture_id == gesture.id)
                    .order_by(DbGestureSample.sample_index)
                    .limit(8)
                    .all()
                )
                for row in rows:
                    raw_path = str(row[0] or "").strip()
                    if raw_path:
                        sample_paths.append(resolve_config_path(raw_path))

                samples_path = str(getattr(gesture, "samples_path", "") or "").strip()
                if samples_path and len(sample_paths) < 8:
                    sample_dir = resolve_config_path(samples_path)
                    if sample_dir.exists():
                        for sample_path in gesture_sample_paths(sample_dir)[:8]:
                            if sample_path not in sample_paths:
                                sample_paths.append(sample_path)

                profiles.append(
                    (label, self._infer_training_gesture_type(label, sample_paths))
                )
            return profiles
        finally:
            session.close()

    def _infer_training_gesture_type(self, label: str, sample_paths: list[Path]) -> str:
        clean = str(label or "").strip().lower()
        if _is_dynamic_rejection_label(clean):
            return GESTURE_TYPE_NEGATIVE

        for sample_path in sample_paths:
            inferred = self._infer_training_gesture_type_from_sample(sample_path)
            if inferred:
                return inferred

        try:
            return self._gesture_type_for_label(label)
        except Exception:
            return "static"

    @staticmethod
    def _infer_training_gesture_type_from_sample(sample_path: Path) -> str:
        meta_path = sample_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                scope = str(
                    meta.get("source_scope") or meta.get("gesture_type") or ""
                ).strip().lower()
                if scope in {
                    "static",
                    "quasi_static",
                    GESTURE_TYPE_DYNAMIC,
                    GESTURE_TYPE_NEGATIVE,
                }:
                    return scope
                if bool(meta.get("include_global_motion")):
                    return GESTURE_TYPE_DYNAMIC
                sample_format = str(meta.get("sample_feature_format") or "").lower()
                if "wrist_xy" in sample_format:
                    return GESTURE_TYPE_DYNAMIC
                raw_dim = int(meta.get("raw_feature_dim") or 0)
                if raw_dim in {44, 65, 88, 130}:
                    return GESTURE_TYPE_DYNAMIC
            except Exception:
                pass

        raw_dim = AppController._sample_raw_feature_dim(sample_path)
        if raw_dim in {44, 65, 88, 130}:
            return GESTURE_TYPE_DYNAMIC
        return ""

    @staticmethod
    def _sample_raw_feature_dim(sample_path: Path) -> int:
        meta_path = sample_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                raw_dim = int(meta.get("raw_feature_dim") or 0)
                if raw_dim > 0:
                    return raw_dim
            except Exception:
                pass

        try:
            import numpy as np

            arr = np.load(sample_path, mmap_mode="r", allow_pickle=False)
            if arr.ndim >= 2:
                return int(np.prod(arr.shape[1:]))
        except Exception:
            pass
        return 0

    def _dynamic_prototype_target_dim(
        self,
        data_root: str | Path,
        include_labels: list[str] | None = None,
    ) -> int:
        root = resolve_config_path(data_root)
        include_keys = {
            str(label or "").strip().lower()
            for label in (include_labels or [])
            if str(label or "").strip()
        }
        if not root.exists():
            return DYNAMIC_PROTOTYPE_DEFAULT_TARGET_DIM

        supported_dims = {44, 65, 88, 130}
        for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if include_keys and label_dir.name.strip().lower() not in include_keys:
                continue
            for sample_path in gesture_sample_paths(label_dir)[:8]:
                raw_dim = self._sample_raw_feature_dim(sample_path)
                if raw_dim in supported_dims:
                    return raw_dim
        return DYNAMIC_PROTOTYPE_DEFAULT_TARGET_DIM

    def _training_labels_for_scope(self, training_scope: str = "") -> list[str]:
        labels = self._active_training_labels_from_db()
        scope = str(training_scope or "").strip()
        if not scope:
            return labels
        selected = set(parse_gesture_type_scope(scope))
        profiles = self._active_training_label_profiles_from_db()
        if profiles:
            out: list[str] = []
            seen: set[str] = set()
            for label, gesture_type in profiles:
                clean = str(label or "").strip()
                key = clean.lower()
                if not clean or key in seen or gesture_type not in selected:
                    continue
                out.append(clean)
                seen.add(key)
            return out
        return labels_for_gesture_types(
            labels,
            selected,
            taxonomy_path=self._configured_taxonomy_path(),
        )

    def start_recording(
        self,
        label: str,
        num_samples: int = 20,
        frames: int = 30,
        two_hands: bool = False,
        include_global_motion: bool = False,
        include_landmark_z: bool = False,
        on_line: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """Запустить встроенную запись сэмплов через Flet-камеру.

        Раньше кнопка открывала отдельное окно OpenCV через
        ``cv.record_gestures``. Для пользовательского UI это выглядело как
        неработающая кнопка: запись уходила в другое окно и требовала
        клавиатурного управления. Теперь сэмплы собираются в текущем camera
        loop, а прогресс выводится в журнал вкладки «Обучение».
        """
        if self._active_training_process() is not None:
            return False
        with self._sample_recording_lock:
            if self._sample_recording is not None:
                return False

        clean = (label or "").strip()
        if not clean:
            return False

        camera_was_active = bool(self._is_camera_active)
        self.stop_recognition()
        self.stop_embedded_recognition()

        data_dir = self._configured_data_dir()
        out_dir = data_dir / clean
        target_samples = max(1, int(num_samples))
        target_frames = max(1, int(frames))
        use_landmark_z = bool(include_landmark_z)
        augment_count = (
            SAMPLE_RECORDING_DYNAMIC_AUGMENTATIONS
            if include_global_motion
            else SAMPLE_RECORDING_STATIC_AUGMENTATIONS
        )
        session = {
            "label": clean,
            "target": target_samples,
            "frames": target_frames,
            "two_hands": bool(two_hands),
            "include_global_motion": bool(include_global_motion),
            "include_landmark_z": use_landmark_z,
            "augment_count": augment_count,
            "started_camera_for_recording": False,
            "saved": 0,
            "frames_buf": [],
            "frame_scales": [],
            "out_dir": out_dir,
            "on_line": on_line,
            "on_done": on_done,
            "next_allowed_at": 0.0,
            "last_no_hand_log": 0.0,
            "last_ready_log": 0.0,
            "last_progress_emit": 0.0,
            "warmup_until": time.monotonic() + 1.5,
            "quality_gate": True,
            "ready_required_frames": SAMPLE_RECORDING_READY_FRAMES,
            "ready_buffer": [],
            "countdown_until": 0.0,
            "countdown_seconds": SAMPLE_RECORDING_COUNTDOWN_SECONDS,
            "stability_threshold": SAMPLE_RECORDING_STABILITY_THRESHOLD,
            "quality_reports": [],
            "last_message": "Подготовка камеры",
        }
        with self._sample_recording_lock:
            self._sample_recording = session
            self._sample_recording_detector = None
        self._emit_sample_recording_changed(
            session,
            active=True,
            message="Подготовка камеры",
        )

        if on_line:
            on_line(
                f"[i] Встроенная запись «{clean}»: {target_samples} сэмплов, "
                f"{target_frames} кадров"
                + (" (две руки)" if two_hands else "")
                + (" + глобальное движение" if include_global_motion else "")
                + (" + landmark z" if use_landmark_z else "")
            )
            on_line(
                "[i] Держи стартовую позу неподвижно — после отсчета записывай движение"
            )

        if not self._is_camera_active:
            self.start_camera()
            with self._sample_recording_lock:
                if self._sample_recording is session:
                    session["started_camera_for_recording"] = not camera_was_active
        if not self._is_camera_active:
            self._finish_sample_recording(1, "[!] Камера не открылась, запись остановлена")
            return False

        self._set_status(f"Запись жеста: {clean}")
        return True

    def _ensure_dynamic_label_in_taxonomy(
        self,
        label: str,
        *,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> bool:
        clean = str(label or "").strip()
        if not clean:
            return False

        taxonomy_path = self._configured_taxonomy_path()
        if not taxonomy_path.exists():
            return False

        raw = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return False

        types = raw.setdefault("types", {})
        if not isinstance(types, dict):
            return False

        dynamic = types.setdefault(GESTURE_TYPE_DYNAMIC, [])
        if not isinstance(dynamic, list):
            dynamic = []
            types[GESTURE_TYPE_DYNAMIC] = dynamic

        key = clean.lower()
        changed = False
        for gesture_type in ("static", "quasi_static"):
            labels = types.get(gesture_type)
            if not isinstance(labels, list):
                continue
            filtered = [
                item
                for item in labels
                if str(item or "").strip().lower() != key
            ]
            if len(filtered) != len(labels):
                types[gesture_type] = filtered
                changed = True

        if not any(str(item or "").strip().lower() == key for item in dynamic):
            dynamic.append(clean)
            changed = True

        if not changed:
            return False

        taxonomy_path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._gesture_taxonomy_cache = None
        if on_line:
            on_line(f"[i] Taxonomy: «{clean}» помечен как dynamic")
        return True

    def start_training(
        self,
        data_root: str = "",
        out_path: str = "",
        neighbors: int = 5,
        expect_dim: Optional[int] = None,
        feature_mode: str = "static_mean",
        model_type: str = "knn",
        classes_out_path: str = "",
        feature_dim_out_path: str = "",
        feature_mode_out_path: str = "",
        training_scope: str = "",
        extra_args: Optional[list[str]] = None,
        on_line: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """Запустить ``cv/train_classifier.py`` как subprocess."""
        if self._active_training_process() is not None:
            return False
        project_root = Path(__file__).resolve().parents[2]
        try:
            self.sync_dataset_to_db()
        except Exception as e:
            if on_line:
                on_line(f"[w] sync_dataset_to_db before training: {e}")
        scope = str(training_scope or "").strip()
        if scope:
            try:
                scoped_labels = self._training_labels_for_scope(scope)
            except Exception as e:
                if on_line:
                    on_line(f"[!] Не удалось прочитать taxonomy для scope={scope}: {e}")
                return False
            if not scoped_labels:
                if on_line:
                    on_line(
                        f"[!] Нет активных жестов для training scope={scope}. "
                        "Проверь configs/gesture_taxonomy.json и список жестов."
                    )
                return False
            if on_line:
                on_line(
                    f"[i] Training scope={scope}: "
                    + ", ".join(scoped_labels)
                )
        cmd = self._build_training_command(
            data_root=data_root,
            out_path=out_path,
            neighbors=neighbors,
            expect_dim=expect_dim,
            feature_mode=feature_mode,
            model_type=model_type,
            classes_out_path=classes_out_path,
            feature_dim_out_path=feature_dim_out_path,
            feature_mode_out_path=feature_mode_out_path,
            training_scope=training_scope,
            extra_args=extra_args,
        )

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
            )
        except Exception as e:
            if on_line:
                on_line(f"[!] Не удалось запустить процесс: {e}")
            return False

        self._training_proc = proc
        if on_line:
            on_line(f"[i] PID={proc.pid}: {' '.join(cmd)}")

        def reader() -> None:
            final_code = 1
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    if on_line:
                        try:
                            on_line(line.rstrip())
                        except Exception:
                            pass
            except Exception:
                pass
            finally:
                code = proc.wait()
                final_code = int(code)
                self._clear_training_process(proc)
                # После успешного обучения синхронизируем словарь жестов в БД,
                # чтобы экран «Привязки» сразу увидел новые/обновлённые классы.
                if final_code == 0:
                    try:
                        summary = self.sync_dataset_to_db()
                        if on_line:
                            on_line(
                                f"[✓] БД жестов синхронизирована: "
                                f"добавлено {summary['created']}, "
                                f"обновлено {summary['updated']}, "
                                f"классов: {summary['total']}, "
                                f"сэмплов: {summary['samples']}"
                            )
                    except Exception as e:
                        if on_line:
                            on_line(f"[w] sync_dataset_to_db: {e}")
                if final_code == 0 and self._should_train_dynamic_prototypes(training_scope):
                    final_code = self._run_dynamic_prototype_training(
                        data_root=data_root,
                        dynamic_model_out_path=out_path,
                        on_line=on_line,
                    )
                if on_done:
                    try:
                        on_done(int(final_code))
                    except Exception:
                        pass

        Thread(target=reader, daemon=True).start()
        return True

    def _should_train_dynamic_prototypes(self, training_scope: str = "") -> bool:
        try:
            return GESTURE_TYPE_DYNAMIC in parse_gesture_type_scope(training_scope)
        except ValueError:
            return False

    def _run_dynamic_prototype_training(
        self,
        *,
        data_root: str = "",
        dynamic_model_out_path: str = "",
        on_line: Optional[Callable[[str], None]] = None,
    ) -> int:
        project_root = Path(__file__).resolve().parents[2]
        cmd = self._build_dynamic_prototype_training_command(
            data_root=data_root,
            dynamic_model_out_path=dynamic_model_out_path,
        )
        if on_line:
            on_line(
                "[i] Обучение dynamic prototype/rejection layer "
                "(external negatives + conflict filter)"
            )
            on_line(f"[i] PID=pending: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
            )
        except Exception as e:
            if on_line:
                on_line(f"[!] Не удалось запустить dynamic prototype training: {e}")
            return 1

        self._training_proc = proc
        if on_line:
            on_line(f"[i] PID={proc.pid}: {' '.join(cmd)}")
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if on_line:
                    try:
                        on_line(line.rstrip())
                    except Exception:
                        pass
        except Exception:
            pass
        code = int(proc.wait())
        self._clear_training_process(proc)
        if code == 0:
            try:
                self._reset_embedded_infer_after_model_change()
            except Exception:
                pass
            if on_line:
                on_line("[✓] Dynamic prototype/rejection layer обновлен")
        elif on_line:
            on_line(f"[!] Dynamic prototype training завершился с кодом {code}")
        return code

    def start_negative_generation(
        self,
        *,
        samples_per_label: int = 20,
        seed: int = 42,
        on_line: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """Generate reproducible synthetic negative samples as a subprocess."""
        if self._active_training_process() is not None:
            return False
        with self._sample_recording_lock:
            if self._sample_recording is not None:
                return False

        project_root = Path(__file__).resolve().parents[2]
        cmd = self._build_negative_generation_command(
            samples_per_label=samples_per_label,
            seed=seed,
        )
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
            )
        except Exception as e:
            if on_line:
                on_line(f"[!] Не удалось запустить negative sampler: {e}")
            return False

        self._training_proc = proc
        if on_line:
            on_line(f"[i] PID={proc.pid}: {' '.join(cmd)}")

        def reader() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    if on_line:
                        try:
                            on_line(line.rstrip())
                        except Exception:
                            pass
            except Exception:
                pass
            finally:
                code = proc.wait()
                self._clear_training_process(proc)
                if int(code) == 0:
                    try:
                        summary = self.sync_dataset_to_db()
                        if on_line:
                            on_line(
                                f"[✓] БД жестов синхронизирована: "
                                f"добавлено {summary['created']}, "
                                f"обновлено {summary['updated']}, "
                                f"классов: {summary['total']}, "
                                f"сэмплов: {summary['samples']}"
                            )
                    except Exception as e:
                        if on_line:
                            on_line(f"[w] sync_dataset_to_db: {e}")
                if on_done:
                    try:
                        on_done(int(code))
                    except Exception:
                        pass

        Thread(target=reader, daemon=True).start()
        return True

    def _build_negative_generation_command(
        self,
        *,
        samples_per_label: int = 20,
        seed: int = 42,
    ) -> list[str]:
        project_root = Path(__file__).resolve().parents[2]
        return [
            sys.executable,
            "-u",
            "-m",
            "scripts.generate_negative_samples",
            "--data-root",
            str(self._configured_data_dir()),
            "--taxonomy",
            str(self._configured_taxonomy_path()),
            "--samples-per-label",
            str(max(1, int(samples_per_label))),
            "--target-frames",
            str(self._dynamic_recognition_window()),
            "--seed",
            str(int(seed)),
            "--manifest-out",
            str(project_root / "docs" / "experiments" / "negative_sampling_manifest.json"),
        ]

    def _build_training_command(
        self,
        *,
        data_root: str = "",
        out_path: str = "",
        neighbors: int = 5,
        expect_dim: Optional[int] = None,
        feature_mode: str = "static_mean",
        model_type: str = "knn",
        classes_out_path: str = "",
        feature_dim_out_path: str = "",
        feature_mode_out_path: str = "",
        training_scope: str = "",
        extra_args: Optional[list[str]] = None,
    ) -> list[str]:
        actual_data_root = data_root or str(self._configured_data_dir())
        actual_out_path = out_path or str(self._configured_model_path())
        actual_classes_out = classes_out_path or str(self._configured_classes_path())
        actual_feature_dim_out = feature_dim_out_path or str(self._configured_feature_dim_path())
        actual_feature_mode_out = (
            feature_mode_out_path
            or str(Path(actual_out_path).with_name("feature_mode.txt"))
        )
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "cv.train_classifier",
            "--data-root",
            actual_data_root,
            "--out",
            actual_out_path,
            "--classes-out",
            actual_classes_out,
            "--feature-dim-out",
            actual_feature_dim_out,
            "--feature-mode-out",
            actual_feature_mode_out,
            "--feature-mode",
            str(feature_mode or "static_mean"),
            "--model-type",
            str(model_type or "knn"),
            "--neighbors",
            str(int(neighbors)),
        ]
        for label in self._training_labels_for_scope(training_scope):
            cmd += ["--include-label", label]
        if expect_dim is not None:
            cmd += ["--expect-dim", str(int(expect_dim))]
        for item in extra_args or []:
            clean = str(item).strip()
            if clean:
                cmd.append(clean)
        return cmd

    def _build_dynamic_prototype_training_command(
        self,
        *,
        data_root: str = "",
        dynamic_model_out_path: str = "",
    ) -> list[str]:
        project_root = Path(__file__).resolve().parents[2]
        actual_data_root = data_root or str(self._configured_data_dir())
        dynamic_model_path = Path(dynamic_model_out_path or self._dynamic_model_path())
        if dynamic_model_path.name not in set(DYNAMIC_MODEL_FILENAMES.values()):
            dynamic_model_path = dynamic_model_path.with_name(
                DYNAMIC_MODEL_FILENAMES[DYNAMIC_MODEL_PROFILE_PRODUCTION]
            )
        dynamic_model_dir = dynamic_model_path.parent if dynamic_model_path.parent else Path(".")
        production_prototypes = dynamic_model_dir / "dynamic_prototypes.json"
        long_window_filenames = {
            DYNAMIC_MODEL_FILENAMES[DYNAMIC_MODEL_PROFILE_PRODUCTION],
            DYNAMIC_MODEL_FILENAMES[DYNAMIC_MODEL_PROFILE_DYNAMIC_LANDMARK_CNN],
            DYNAMIC_MODEL_FILENAMES[DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET_72],
        }
        target_frames = (
            DYNAMIC_RECOGNITION_LONG_WINDOW
            if dynamic_model_path.name in long_window_filenames
            else DYNAMIC_RECOGNITION_WINDOW
        )
        for profile, filename in DYNAMIC_MODEL_FILENAMES.items():
            if dynamic_model_path.name == filename:
                production_prototypes = dynamic_model_dir / DYNAMIC_PROTOTYPE_FILENAMES.get(
                    profile,
                    "dynamic_prototypes.json",
                )
                break
        else:
            if dynamic_model_path.stem.startswith("dynamic_sequence"):
                production_prototypes = (
                    dynamic_model_dir / f"{dynamic_model_path.stem}_prototypes.json"
                )
        include_labels: list[str] = []
        if getattr(self, "_db_initialized", False):
            try:
                include_labels = self._training_labels_for_scope("dynamic,negative")
            except Exception as e:
                print(f"[w] dynamic prototype include labels: {e}")
                include_labels = []
        target_dim = self._dynamic_prototype_target_dim(
            actual_data_root,
            include_labels,
        )
        external_root = project_root / "data" / "external"
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "scripts.dynamic_prototype_experiments",
            "--data-root",
            actual_data_root,
            "--external-negative-root",
            str(external_root),
            "--include-external-negatives",
            "--methods",
            "prototype_distance",
            "--target-frames",
            str(target_frames),
            "--target-dim",
            str(target_dim),
            "--base-models-dir",
            str(dynamic_model_dir),
            "--variant-root",
            str(project_root / "models" / "experiments" / "dynamic_prototype"),
            "--write-production",
            "--production-out",
            str(production_prototypes),
            "--report-json",
            str(project_root / "docs" / "experiments" / "dynamic_prototype_ui_training.json"),
            "--report-md",
            str(project_root / "docs" / "experiments" / "dynamic_prototype_ui_training.md"),
            "--mlflow-tracking-uri",
            self._live_evaluation_mlflow_tracking_uri(),
        ]
        for label in include_labels:
            cmd += ["--include-label", label]
        return cmd

    def cancel_training(self) -> None:
        """Прервать текущий тренировочный subprocess (если запущен)."""
        if self.cancel_sample_recording():
            return
        proc = self._active_training_process()
        if proc is None:
            return
        try:
            proc.terminate()
        except Exception:
            pass

    @property
    def is_training_active(self) -> bool:
        return (
            self._active_training_process() is not None
            or self.is_sample_recording_active
        )

    @property
    def is_sample_recording_active(self) -> bool:
        with self._sample_recording_lock:
            return self._sample_recording is not None

    def cancel_sample_recording(self) -> bool:
        with self._sample_recording_lock:
            active = self._sample_recording is not None
        if not active:
            return False
        self._finish_sample_recording(130, "[i] Запись сэмплов остановлена")
        return True

    # ----------------------------------------------------------------------
    # Завершение
    # ----------------------------------------------------------------------

    def shutdown(self) -> None:
        """Аккуратное завершение всех ресурсов."""
        try:
            self.stop_embedded_recognition()
        except Exception:
            pass
        try:
            self.stop_recognition()
        except Exception:
            pass
        try:
            self.stop_camera()
        except Exception:
            pass
        try:
            self.cancel_sample_recording()
        except Exception:
            pass
        if self._voice_assistant is not None:
            try:
                self._voice_assistant.stop_listening_loop()
            except Exception:
                pass
            self._voice_assistant = None
