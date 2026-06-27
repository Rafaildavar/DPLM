"""
GUI-агностичный контроллер для Flet-версии DPLM.

Делает то же, что прежний :class:`app.main.AppController` (PySide6), но без
зависимостей от Qt: вместо ``Signal/Slot`` — обычные списки коллбэков.

Контроллер инкапсулирует:
    * захват камеры (OpenCV, тот же ``open_default_capture`` что и в QML версии);
    * встроенный CV-пайплайн (MediaPipe + KNN через :class:`GestureOnlineInfer`);
    * subprocess-распознавание (``cv/realtime_infer.py``) для фонового режима;
    * выполнение команд (``CommandExecutor``);
    * (лениво) связку с БД и привязками жестов.

Кадр камеры отдаётся UI как JPEG-байты (Flet ``Image`` умеет ``src_base64``).
Цикл захвата работает в отдельном потоке — Flet UI обновляется через
``page.update()`` из callback'а.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import json
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
DYNAMIC_MODEL_PROFILES = (
    DYNAMIC_MODEL_PROFILE_KNN,
    DYNAMIC_MODEL_PROFILE_SVM,
    DYNAMIC_MODEL_PROFILE_EXTRA_TREES,
)
DYNAMIC_MODEL_FILENAMES = {
    DYNAMIC_MODEL_PROFILE_KNN: "dynamic_knn.pkl",
    DYNAMIC_MODEL_PROFILE_SVM: "dynamic_svm.pkl",
    DYNAMIC_MODEL_PROFILE_EXTRA_TREES: "dynamic_extra_trees.pkl",
}
DYNAMIC_RECOGNITION_WINDOW = 36
DYNAMIC_GESTURE_CONFIRM_FRAMES = 2
SAMPLE_RECORDING_READY_FRAMES = 6
SAMPLE_RECORDING_COUNTDOWN_SECONDS = 0.8
SAMPLE_RECORDING_STABILITY_THRESHOLD = 0.055
DYNAMIC_SAMPLE_MIN_MOTION_ENERGY = 0.015
DYNAMIC_SAMPLE_DIRECTION_THRESHOLD = 0.05
LIVE_EVAL_DEFAULT_ATTEMPTS = 10
LIVE_EVAL_DEFAULT_TIMEOUT_SECONDS = 0.0
LIVE_EVAL_DEFAULT_MIN_CONFIDENCE = 0.60
LIVE_EVAL_ATTEMPT_COOLDOWN_SECONDS = 0.85
CAMERA_PREVIEW_MAX_FPS = 20.0
RUNTIME_PERFORMANCE_FLUSH_SECONDS = 5.0


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
        landmarks_changed(str)       # JSON со списком ландмарок
        gesture_mode_changed(bool)
        pointer_mode_changed(bool)
        landmark_overlay_changed(bool)
        voice_assistant_state_changed(str)
        two_hands_changed(bool)
        recognition_model_mode_changed(str)
        dynamic_model_profile_changed(str)
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
        self._dynamic_model_profile: str = DYNAMIC_MODEL_PROFILE_KNN
        self._two_hands_mode: bool = bool(self._config.recognition.two_hands_mode)
        self._gesture_mode: bool = True
        self._show_landmark_overlay: bool = True
        self._pointer_mode: bool = False
        self._confidence: float = 0.0
        self._landmarks_json: str = "[]"
        self._last_label: str = ""
        self._pending_label: str = ""
        self._pending_frames: int = 0
        self._pending_confidence_total: float = 0.0
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
        self.landmarks_changed = _Event()
        self.gesture_mode_changed = _Event()
        self.pointer_mode_changed = _Event()
        self.landmark_overlay_changed = _Event()
        self.voice_assistant_state_changed = _Event()
        self.two_hands_changed = _Event()
        self.recognition_model_mode_changed = _Event()
        self.dynamic_model_profile_changed = _Event()
        self.sample_recording_changed = _Event()
        self.live_evaluation_changed = _Event()

        # Камера -------------------------------------------------------------
        self._camera_cap: Any | None = None
        self._camera_thread: threading.Thread | None = None
        self._camera_stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_jpeg_bytes: bytes = b""
        self._frame_w = 0
        self._frame_h = 0
        self._target_fps = int(self._config.recognition.target_fps)
        self._runtime_inference_samples: list[dict[str, Any]] = []
        self._runtime_perf_last_flush = time.monotonic()

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
            getattr(self, "_dynamic_model_profile", DYNAMIC_MODEL_PROFILE_KNN)
            or DYNAMIC_MODEL_PROFILE_KNN
        ).strip()
        return (
            profile
            if profile in DYNAMIC_MODEL_PROFILES
            else DYNAMIC_MODEL_PROFILE_KNN
        )

    @property
    def gesture_mode(self) -> bool:
        return self._gesture_mode

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

    def _configured_data_dir(self) -> Path:
        return self._configured_path(self._config.paths.data_dir)

    def _configured_models_dir(self) -> Path:
        return self._configured_path(self._config.paths.models_dir)

    def _configured_model_path(self) -> Path:
        return self._configured_path(self._config.paths.model_path)

    def _configured_classes_path(self) -> Path:
        return self._configured_path(self._config.paths.classes_path)

    def _configured_feature_dim_path(self) -> Path:
        return self._configured_path(self._config.paths.feature_dim_path)

    def _configured_log_dir(self) -> Path:
        return self._configured_path(self._config.paths.log_dir)

    def _configured_feature_mode_path(self) -> Path:
        return self._configured_models_dir() / "feature_mode.txt"

    def _configured_taxonomy_path(self) -> Path:
        return DEFAULT_TAXONOMY_PATH

    def _dynamic_model_path(self) -> Path:
        filename = DYNAMIC_MODEL_FILENAMES.get(
            self.dynamic_model_profile,
            DYNAMIC_MODEL_FILENAMES[DYNAMIC_MODEL_PROFILE_KNN],
        )
        return self._configured_models_dir() / filename

    def _dynamic_classes_path(self) -> Path:
        return self._configured_models_dir() / "dynamic_classes.json"

    def _dynamic_feature_dim_path(self) -> Path:
        return self._configured_models_dir() / "dynamic_feature_dim.txt"

    def _dynamic_feature_mode_path(self) -> Path:
        return self._configured_models_dir() / "dynamic_feature_mode.txt"

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
            return DYNAMIC_RECOGNITION_WINDOW
        return 30

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

    def _reset_gesture_confirmation(self) -> None:
        self._pending_label = ""
        self._pending_frames = 0
        self._pending_confidence_total = 0.0

    def _gesture_type_for_label(self, label: str) -> str:
        taxonomy = getattr(self, "_gesture_taxonomy_cache", None)
        if taxonomy is None:
            taxonomy = load_gesture_taxonomy(self._configured_taxonomy_path())
            self._gesture_taxonomy_cache = taxonomy
        return taxonomy.gesture_type_for_label(label)

    def _gesture_confirm_frames(self, label: str = "") -> int:
        if self.recognition_model_mode == RECOGNITION_MODEL_DYNAMIC:
            return DYNAMIC_GESTURE_CONFIRM_FRAMES
        if self.recognition_model_mode == RECOGNITION_MODEL_AUTO and label:
            try:
                if self._gesture_type_for_label(label) == GESTURE_TYPE_DYNAMIC:
                    return DYNAMIC_GESTURE_CONFIRM_FRAMES
                return AUTO_STATIC_GESTURE_CONFIRM_FRAMES
            except Exception as e:
                print(f"[w] dynamic confirmation taxonomy: {e}", flush=True)
        return GESTURE_CONFIRM_FRAMES

    def _update_gesture_confirmation(self, label: str, confidence: float) -> tuple[bool, float]:
        clean = (label or "").strip()
        if not clean:
            self._reset_gesture_confirmation()
            return False, 0.0

        if clean != getattr(self, "_pending_label", ""):
            self._pending_label = clean
            self._pending_frames = 1
            self._pending_confidence_total = float(confidence)
        else:
            self._pending_frames = int(getattr(self, "_pending_frames", 0)) + 1
            self._pending_confidence_total = float(
                getattr(self, "_pending_confidence_total", 0.0)
            ) + float(confidence)

        avg_conf = self._pending_confidence_total / max(1, self._pending_frames)
        return self._pending_frames >= self._gesture_confirm_frames(clean), avg_conf

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

    def list_recognition_labels(self) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            clean = str(value or "").strip()
            key = clean.lower()
            if clean and key not in seen:
                labels.append(clean)
                seen.add(key)

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

        return sorted(labels, key=lambda x: x.lower())

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
        self._reset_gesture_confirmation()
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
        session["message"] = self._live_evaluation_finish_message(session, reason)
        if bool(session.get("auto_execute_was_enabled")):
            self._auto_execute_on_gesture = True
        self._append_live_evaluation_jsonl(session, event_type=f"run_{reason}")
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
                        "dynamic_label": payload.get("dynamic_label"),
                        "dynamic_confidence": payload.get("dynamic_confidence"),
                        "dynamic_type": payload.get("dynamic_type"),
                        "dynamic_reject_reason": payload.get("dynamic_reject_reason"),
                        "dynamic_phase": payload.get("dynamic_phase"),
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
            "static_label": str(route_metadata.get("static_label") or ""),
            "static_confidence": _float_or_none(
                route_metadata.get("static_confidence")
            ),
            "static_type": str(route_metadata.get("static_type") or ""),
            "static_reject_reason": str(
                route_metadata.get("static_reject_reason") or ""
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
            route = str((route_metadata or {}).get("route") or "")
            if expected_type == GESTURE_TYPE_DYNAMIC and route != "dynamic":
                session["last_result"] = "ignored_wrong_route"
                self._emit_live_evaluation_changed(
                    session,
                    message=f"{clean_label}: static-route игнорируется в dynamic-тесте",
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

    def _update_pointer_from_landmarks(self, landmarks_json: str) -> None:
        if not self._pointer_mode:
            return
        pointer = self._ensure_pointer_control()
        if pointer is None:
            self._set_status("Pointer: pyautogui недоступен")
            return
        result = pointer.update(landmarks_json)
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
            model_suffix = f" (auto:{self.dynamic_model_profile})"
        else:
            model_suffix = ""
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
                static_infer = GestureOnlineInfer(
                    model_path=self._configured_model_path(),
                    classes_path=self._configured_classes_path(),
                    feature_dim_path=self._configured_feature_dim_path(),
                    feature_mode_path=self._configured_feature_mode_path(),
                    window=30,
                    two_hands=self._two_hands_mode,
                )
                dynamic_infer = GestureOnlineInfer(
                    model_path=self._dynamic_model_path(),
                    classes_path=self._dynamic_classes_path(),
                    feature_dim_path=self._dynamic_feature_dim_path(),
                    feature_mode_path=self._dynamic_feature_mode_path(),
                    window=DYNAMIC_RECOGNITION_WINDOW,
                    two_hands=self._two_hands_mode,
                    initialize_detector=False,
                )
                infer = GestureRecognitionRouter(
                    static_infer=static_infer,
                    dynamic_infer=dynamic_infer,
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
        self._camera_stop.clear()
        self._camera_thread = threading.Thread(
            target=self._camera_loop, name="dplm-camera", daemon=True
        )
        self._camera_thread.start()
        self._set_camera_active(True)
        self._set_status("Camera: streaming")
        print("[✓] Flet: камера открыта")

    def stop_camera(self) -> None:
        if not self._is_camera_active and not self._camera_thread:
            return
        self._camera_stop.set()
        cap = self._camera_cap
        self._camera_cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        t = self._camera_thread
        if t is not None:
            t.join(timeout=2.0)
        self._camera_thread = None
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

    def _camera_loop(self) -> None:
        """Фоновый поток: читает кадры, кодирует в JPEG, кладёт base64."""
        import cv2
        import numpy as np

        frame_interval = 1.0 / float(self._target_fps)
        preview_interval = 1.0 / max(
            1.0,
            min(float(self._target_fps), CAMERA_PREVIEW_MAX_FPS),
        )
        next_t = time.monotonic()
        next_preview_t = next_t

        while not self._camera_stop.is_set():
            cap = self._camera_cap
            if cap is None:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame_bgr = cv2.flip(frame_bgr, 1)
            landmarks_json = "[]"

            if self._sample_recording is not None:
                try:
                    recording_landmarks = self._process_sample_recording_frame(frame_bgr)
                    if recording_landmarks:
                        landmarks_json = recording_landmarks
                except Exception as e:
                    print(f"[!] embedded recording frame error: {e}")
                    self._finish_sample_recording(1, f"[!] Ошибка записи сэмпла: {e}")

            # Встроенный CV (MediaPipe + KNN) — синхронно в этом же потоке;
            # как в исходной версии, это безопасно потому что MediaPipe-объекты
            # создаются и используются в одном потоке.
            if self._embedded_active and self._sample_recording is None:
                try:
                    if self._embedded_infer is None:
                        self._embedded_infer = self._create_embedded_infer()
                    if self._embedded_infer is None:
                        continue
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
                    out = self._embedded_infer.process_frame_rgb(rgb)
                    if out is not None:
                        landmarks_json = out.get("landmarks_json") or "[]"
                        self._dispatch_infer_result(out)
                except Exception as e:
                    print(f"[!] embedded CV frame error: {e}")

            preview_now = time.monotonic()
            if preview_now >= next_preview_t:
                self._draw_landmarks_on_frame(frame_bgr, landmarks_json)
                self._draw_sample_recording_overlay_on_frame(frame_bgr)

                # Keep ML at target FPS, but cap JPEG/base64/Flet preview work.
                ok2, buf = cv2.imencode(
                    ".jpg",
                    frame_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 75],
                )
                if ok2:
                    data = buf.tobytes()
                    with self._frame_lock:
                        self._latest_jpeg_bytes = data
                        self._frame_h, self._frame_w = frame_bgr.shape[:2]
                    self.camera_frame_updated.emit()
                next_preview_t = preview_now + preview_interval

            after_work = time.monotonic()
            next_t += frame_interval
            sleep = next_t - after_work
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = after_work  # отстаём — сбрасываем расписание

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
    ) -> Any | None:
        import numpy as np

        from cv.hand_landmarker import normalize_landmarks

        def hand_feature(hand: Any) -> Any:
            normalized = normalize_landmarks(hand.landmarks)
            if not include_global_motion:
                return normalized

            pts = np.asarray(hand.landmarks, dtype=np.float32)
            pose = normalized.reshape(-1)
            if pts.shape == (21, 2):
                wrist = pts[0]
            else:
                wrist = np.zeros(2, dtype=np.float32)
            return np.concatenate([pose, wrist], axis=0).astype(np.float32, copy=False)

        features: list[Any] = []
        for hand in hands:
            try:
                features.append(hand_feature(hand))
            except Exception:
                continue

        if two_hands:
            if include_global_motion:
                per_hand_dim = 44
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
                    return np.concatenate(
                        [features[0], np.zeros((21, 2), dtype=np.float32)],
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
        existing = sorted(label_dir.glob("sample_*.npy"))
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
        )

        seq = sequence_to_matrix(sequence)
        motion_energy = sequence_motion_energy(seq)
        displacement = sequence_displacement(seq)
        dx: float | None = None
        dy: float | None = None
        warnings: list[str] = []

        has_global_motion = bool(include_global_motion and seq.shape[1] >= 44)
        if has_global_motion:
            dx = float(seq[-1, -2] - seq[0, -2])
            dy = float(seq[-1, -1] - seq[0, -1])
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
            metadata = {
                "schema_version": 1,
                "label": str(session.get("label") or ""),
                "sample": out_path.name,
                "frames": int(arr.shape[0]),
                "raw_feature_dim": int(arr.shape[1]),
                "projected_hand_scale_median": projected_hand_scale,
                "projected_hand_scale_min": min(scales) if scales else None,
                "projected_hand_scale_max": max(scales) if scales else None,
                "recorded_at": time.time(),
            }
            try:
                out_path.with_suffix(".meta.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError as exc:
                if on_line:
                    on_line(f"[w] Не удалось сохранить metadata: {exc}")
        report = self._sample_quality_report(
            arr,
            label=str(session.get("label") or ""),
            include_global_motion=bool(session.get("include_global_motion")),
        )
        report["projected_hand_scale"] = projected_hand_scale
        session["saved"] = int(session["saved"]) + 1
        session["frames_buf"] = []
        session["frame_scales"] = []
        self._reset_sample_recording_ready_gate(session)
        reports = session.setdefault("quality_reports", [])
        reports.append({"path": out_path.name, **report})
        session["next_allowed_at"] = now + 0.45
        verdict = "OK" if report["ok"] else "WARN"
        session["last_message"] = f"Сохранено: {out_path.name} ({verdict})"
        if on_line:
            on_line(self._format_sample_quality_line(out_path, report))
        self._emit_sample_recording_changed(
            session,
            active=True,
            message=f"Сохранено: {out_path.name} ({verdict})",
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

        if self._status.startswith("Запись"):
            self._set_status("Camera: streaming" if self._is_camera_active else "Idle")

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
        confirmed, stable_conf = self._update_gesture_confirmation(label, conf)
        if not confirmed:
            return
        if label and label != self._last_label:
            print(
                f"[ctrl.gesture] detected={label!r} conf={stable_conf:.3f} "
                f"frames={self._pending_frames} prev={self._last_label!r}",
                flush=True,
            )
            evaluation_active = self.live_evaluation_active()
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
            self._last_label = label
            self.gesture_detected.emit(label)
            # Главное: при детекции жеста сразу запускаем команду через БД-
            # бридж (R4/R5/R6 политика — порог уверенности, cooldown, фильтр
            # опасных действий). Это даёт «жест → команда ОС» — главную фичу
            # диплома.
            if self._auto_execute_on_gesture and not evaluation_active:
                executed = self.execute_for_gesture(label, stable_conf)
            else:
                executed = False
                reason = (
                    "live-evaluation активен"
                    if evaluation_active
                    else "auto-execute выключен"
                )
                print(f"[ctrl.gesture] {reason} — команда не запускается", flush=True)
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
        p95_index = min(
            len(total_values) - 1,
            max(0, int(round((len(total_values) - 1) * 0.95))),
        )
        average_ms = sum(total_values) / len(total_values)
        row = {
            "recorded_at": time.time(),
            "recognition_model_mode": self.recognition_model_mode,
            "dynamic_model_profile": self.dynamic_model_profile,
            "target_fps": int(getattr(self, "_target_fps", 0) or 0),
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
            "inference_fps_capacity": round(1000.0 / average_ms, 2),
        }
        try:
            path = self._configured_log_dir() / "runtime_performance.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[w] runtime performance log write failed: {exc}", flush=True)
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
        self._reset_gesture_confirmation()
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
        self._reset_gesture_confirmation()
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
        self._reset_gesture_confirmation()
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
            target = DYNAMIC_MODEL_PROFILE_KNN
        if target == self.dynamic_model_profile:
            return

        self._dynamic_model_profile = target
        self._reset_gesture_confirmation()
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

    def set_show_landmark_overlay(self, enabled: bool) -> None:
        target = bool(enabled)
        if target == self._show_landmark_overlay:
            return
        self._show_landmark_overlay = target
        self.landmark_overlay_changed.emit(target)
        if target:
            self._ensure_embedded_recognition_for_live_controls()
        elif self._embedded_active:
            self._set_status(self._live_recognition_status())

    def set_gesture_mode(self, enabled: bool) -> None:
        target = bool(enabled)
        if target == self._gesture_mode:
            return
        self._gesture_mode = target
        self._reset_gesture_confirmation()
        if not target:
            self._set_confidence(0.0)
            if self._last_label:
                self._last_label = ""
                self.gesture_detected.emit("")
        self.gesture_mode_changed.emit(target)
        if target:
            self._ensure_embedded_recognition_for_live_controls()
        if self._embedded_active:
            self._set_status(self._live_recognition_status())

    def set_pointer_mode(self, enabled: bool) -> None:
        target = bool(enabled)
        if target == self._pointer_mode:
            return
        self._pointer_mode = target
        if not target:
            self._reset_pointer_control()
        self.pointer_mode_changed.emit(target)
        if target:
            self._ensure_embedded_recognition_for_live_controls()
            self._set_status(self._live_recognition_status())
        elif self._embedded_active:
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
            return False
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
            rows = (
                session.query(DbGesture)
                .filter(DbGesture.is_active.is_(True))
                .filter(DbGesture.model_class_id.isnot(None))
                .order_by(DbGesture.label)
                .all()
            )
            current = {
                row.gesture_id: row.name
                for row in session.query(DbCommand)
                .filter(DbCommand.gesture_id.isnot(None))
                .all()
            }
            return [
                {
                    "id": int(g.id),
                    "label": str(g.label or ""),
                    "description": str(g.description or ""),
                    "isTwoHands": bool(getattr(g, "is_two_hands", False) or False),
                    "boundCommandName": current.get(g.id, ""),
                }
                for g in rows
            ]
        finally:
            session.close()

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
                samples = sorted(label_dir.glob("sample_*.npy"))
                if not samples:
                    continue
                raw_label = label_dir.name
                label_key = raw_label.lower()
                # Тренировка всегда запускается с --lowercase-labels, поэтому
                # БД тоже хранит каноническую метку в нижнем регистре. Так
                # распознанный класс "ctrlz" совпадает с жестом, записанным в
                # папку "CTRLZ".
                label = class_label_by_lower.get(label_key, label_key)
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
                gesture = ensure_gesture(
                    session,
                    label=label,
                    samples_path=label_dir,
                    model_class_id=model_class_id,
                    is_two_hands=is_two_hands,
                    commit=False,
                )
                gesture.description = gesture.description or f"Auto-imported from {label_dir}"
                if classes_path.exists():
                    gesture.model_class_id = model_class_id
                gesture.is_active = True

                for sample_path in samples:
                    frames, hand_count = sample_shape_metadata(sample_path)
                    record_gesture_sample(
                        session,
                        label=label,
                        sample_index=sample_index_from_path(sample_path),
                        features_path=sample_path,
                        frames=frames,
                        hand_count=hand_count,
                        source="dataset",
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
            samples = sorted(label_dir.glob("sample_*.npy"))
            if not samples:
                continue
            out.append({"label": label_dir.name, "samples": len(samples)})
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
        file_paths: set[Path] = set(label_dir.glob("sample_*.npy")) if label_dir else set()

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
            if gesture is not None:
                sample_rows = (
                    session.query(DbGestureSample)
                    .filter(DbGestureSample.gesture_id == gesture.id)
                    .all()
                )
                if resolve_project_path is not None:
                    for sample in sample_rows:
                        stored_path = (sample.features_path or "").strip()
                        if stored_path:
                            file_paths.add(resolve_project_path(stored_path))

                for sample in sample_rows:
                    session.delete(sample)

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

                summary["sampleRowsDeleted"] = len(sample_rows)
                summary["commandsUnbound"] = len(commands)

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

    def _training_labels_for_scope(self, training_scope: str = "") -> list[str]:
        labels = self._active_training_labels_from_db()
        scope = str(training_scope or "").strip()
        if not scope:
            return labels
        return labels_for_gesture_types(
            labels,
            parse_gesture_type_scope(scope),
            taxonomy_path=self._configured_taxonomy_path(),
        )

    def start_recording(
        self,
        label: str,
        num_samples: int = 20,
        frames: int = 30,
        two_hands: bool = False,
        include_global_motion: bool = False,
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
        if self._training_proc is not None and self._training_proc.poll() is None:
            return False
        with self._sample_recording_lock:
            if self._sample_recording is not None:
                return False

        clean = (label or "").strip()
        if not clean:
            return False

        self.stop_recognition()
        self.stop_embedded_recognition()

        data_dir = self._configured_data_dir()
        out_dir = data_dir / clean
        target_samples = max(1, int(num_samples))
        target_frames = max(1, int(frames))
        session = {
            "label": clean,
            "target": target_samples,
            "frames": target_frames,
            "two_hands": bool(two_hands),
            "include_global_motion": bool(include_global_motion),
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
            )
            on_line(
                "[i] Держи стартовую позу неподвижно — после отсчета записывай движение"
            )

        if not self._is_camera_active:
            self.start_camera()
        if not self._is_camera_active:
            self._finish_sample_recording(1, "[!] Камера не открылась, запись остановлена")
            return False

        self._set_status(f"Запись жеста: {clean}")
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
        on_line: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """Запустить ``cv/train_classifier.py`` как subprocess."""
        if self._training_proc is not None and self._training_proc.poll() is None:
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
                # После успешного обучения синхронизируем словарь жестов в БД,
                # чтобы экран «Привязки» сразу увидел новые/обновлённые классы.
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

    def start_negative_generation(
        self,
        *,
        samples_per_label: int = 20,
        seed: int = 42,
        on_line: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """Generate reproducible synthetic negative samples as a subprocess."""
        if self._training_proc is not None and self._training_proc.poll() is None:
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
            str(DYNAMIC_RECOGNITION_WINDOW),
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
            "--lowercase-labels",
        ]
        for label in self._training_labels_for_scope(training_scope):
            cmd += ["--include-label", label]
        if expect_dim is not None:
            cmd += ["--expect-dim", str(int(expect_dim))]
        return cmd

    def cancel_training(self) -> None:
        """Прервать текущий тренировочный subprocess (если запущен)."""
        if self.cancel_sample_recording():
            return
        proc = self._training_proc
        if proc is None:
            return
        if proc.poll() is not None:
            self._training_proc = None
            return
        try:
            proc.terminate()
        except Exception:
            pass

    @property
    def is_training_active(self) -> bool:
        return (
            self._training_proc is not None and self._training_proc.poll() is None
        ) or self.is_sample_recording_active

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
