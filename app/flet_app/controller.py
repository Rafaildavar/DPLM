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
    init_database = None  # type: ignore[assignment]
    get_db_session = None  # type: ignore[assignment]
    reset_database_manager = None  # type: ignore[assignment]
    GestureCommandBridge = None  # type: ignore[assignment]
    save_gesture_binding = None  # type: ignore[assignment]
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
        confidence_changed(float)
        landmarks_changed(str)       # JSON со списком ландмарок
        voice_assistant_state_changed(str)
        two_hands_changed(bool)
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
        self._two_hands_mode: bool = bool(self._config.recognition.two_hands_mode)
        self._confidence: float = 0.0
        self._landmarks_json: str = "[]"
        self._last_label: str = ""

        # События ------------------------------------------------------------
        self.status_changed = _Event()
        self.recognizing_changed = _Event()
        self.camera_active_changed = _Event()
        self.camera_frame_updated = _Event()
        self.gesture_detected = _Event()
        self.command_executed = _Event()
        self.confidence_changed = _Event()
        self.landmarks_changed = _Event()
        self.voice_assistant_state_changed = _Event()
        self.two_hands_changed = _Event()

        # Камера -------------------------------------------------------------
        self._camera_cap: Any | None = None
        self._camera_thread: threading.Thread | None = None
        self._camera_stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_jpeg_bytes: bytes = b""
        self._frame_w = 0
        self._frame_h = 0
        self._target_fps = int(self._config.recognition.target_fps)

        # Встроенный CV ------------------------------------------------------
        self._embedded_infer: Any | None = None

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
    def two_hands_mode(self) -> bool:
        return self._two_hands_mode

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
        t = self._camera_thread
        if t is not None:
            t.join(timeout=2.0)
        self._camera_thread = None
        cap = self._camera_cap
        self._camera_cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
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
        next_t = time.monotonic()

        while not self._camera_stop.is_set():
            cap = self._camera_cap
            if cap is None:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame_bgr = cv2.flip(frame_bgr, 1)

            # Встроенный CV (MediaPipe + KNN) — синхронно в этом же потоке;
            # как в исходной версии, это безопасно потому что MediaPipe-объекты
            # создаются и используются в одном потоке.
            if self._embedded_active and self._embedded_infer is not None:
                try:
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
                        self._dispatch_infer_result(out)
                except Exception as e:
                    print(f"[!] embedded CV frame error: {e}")

            # Кодирование в JPEG для Flet ``Image(src=bytes)``.
            # 720p @ q=75 → 30-70 КБ на кадр, на M-серии тянет 30 fps без
            # заметной нагрузки.
            ok2, buf = cv2.imencode(
                ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            )
            if not ok2:
                continue
            data = buf.tobytes()
            with self._frame_lock:
                self._latest_jpeg_bytes = data
                self._frame_h, self._frame_w = frame_bgr.shape[:2]
            self.camera_frame_updated.emit()

            now = time.monotonic()
            next_t += frame_interval
            sleep = next_t - now
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = now  # отстаём — сбрасываем расписание

    def _dispatch_infer_result(self, out: dict[str, Any]) -> None:
        label = (out.get("label") or "").strip()
        conf = float(out.get("confidence") or 0.0)
        lj = out.get("landmarks_json") or "[]"
        self._set_confidence(conf)
        self._set_landmarks(lj)
        if label and label != self._last_label:
            print(f"[ctrl.gesture] detected={label!r} conf={conf:.3f} prev={self._last_label!r}", flush=True)
            self._last_label = label
            self.gesture_detected.emit(label)
            # Главное: при детекции жеста сразу запускаем команду через БД-
            # бридж (R4/R5/R6 политика — порог уверенности, cooldown, фильтр
            # опасных действий). Это даёт «жест → команда ОС» — главную фичу
            # диплома.
            if self._auto_execute_on_gesture:
                self.execute_for_gesture(label, conf)
            else:
                print("[ctrl.gesture] auto-execute выключен — команда не запускается", flush=True)

    # ----------------------------------------------------------------------
    # Встроенный пайплайн распознавания
    # ----------------------------------------------------------------------

    def start_embedded_recognition(self) -> None:
        import traceback as _tb

        print(
            f"[ctrl.start_embedded] already_active={self._embedded_active}; "
            f"called from:\n  {_tb.format_stack(limit=6)[-2].strip()}",
            flush=True,
        )
        if self._embedded_active:
            return
        self.stop_recognition()
        self._set_status("CV: загрузка MediaPipe…")
        err = ""
        infer: Any | None = None
        try:
            from app.gesture_online_infer import GestureOnlineInfer

            print("[ctrl.start_embedded] creating GestureOnlineInfer…")
            infer = GestureOnlineInfer(
                model_path=self._configured_model_path(),
                classes_path=self._configured_classes_path(),
                feature_dim_path=self._configured_feature_dim_path(),
                two_hands=self._two_hands_mode,
            )
            print(f"[ctrl.start_embedded] infer created; init_error={getattr(infer,'init_error','')!r}")
            if (
                getattr(infer, "classifier_requires_two_hands", False)
                and not self._two_hands_mode
            ):
                self._two_hands_mode = True
                self.two_hands_changed.emit(True)
            if getattr(infer, "init_error", ""):
                err = infer.init_error
                infer.close()
                infer = None
        except Exception as e:
            err = str(e)
            print(f"[!] start_embedded: {e}")
            import traceback
            traceback.print_exc()
            infer = None

        if infer is None:
            self._set_status(f"CV init error: {err}" if err else "CV init failed")
            return

        self._embedded_infer = infer
        self._embedded_active = True
        self._last_label = ""
        self._set_confidence(0.0)
        self._set_landmarks("[]")
        if not self._is_camera_active:
            self.start_camera()
        self._set_status("CV: embedded gesture recognition")

    def stop_embedded_recognition(self) -> None:
        if not self._embedded_active and self._embedded_infer is None:
            return
        self._embedded_active = False
        self._last_label = ""
        self._set_confidence(0.0)
        self._set_landmarks("[]")
        infer = self._embedded_infer
        self._embedded_infer = None
        if infer is not None:
            try:
                infer.close()
            except Exception:
                pass
        if self._status.startswith("CV:"):
            self._set_status("Idle")

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
            items.append(
                {
                    "name": n,
                    "description": cfg.get("description", ""),
                    "platform": cfg.get("platform", "all"),
                }
            )
        return items

    # ----------------------------------------------------------------------
    # Главная связка: один тумблер «запустить/остановить» (embedded)
    # ----------------------------------------------------------------------

    def toggle_recognition(self) -> None:
        """Главный метод для UI: запускает или останавливает встроенное
        распознавание (камера + MediaPipe + KNN + БД-бридж).

        В отличие от ``start_recognition`` (запускающего subprocess
        ``cv/realtime_infer.py``), здесь всё работает прямо в процессе GUI.
        """
        if self._is_recognizing or self._embedded_active:
            self.stop_embedded_recognition()
            self.stop_camera()
            self._set_recognizing(False)
        else:
            self._set_recognizing(True)
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
        набором ``sample_*.npy`` создать/обновить строку в таблице ``gestures``.

        После этого экран «Привязки» сразу увидит словарь жестов (правило R3
        требует ``model_class_id IS NOT NULL`` и ``is_active=True``).
        Если рядом существует ``models/classes.json`` (его пишет
        ``cv/train_classifier.py``), индексы классов проставляются из него.

        Returns:
            Сводка ``{"created": N, "updated": M, "total": K}``.
        """
        if not BINDING_SERVICES_AVAILABLE:
            return {"created": 0, "updated": 0, "total": 0}

        data_root = self._configured_data_dir()
        if not data_root.exists():
            return {"created": 0, "updated": 0, "total": 0}

        # ``classes.json`` — список меток в порядке индексов классификатора.
        classes_path = self._configured_classes_path()
        class_idx_map: dict[str, int] = {}
        if classes_path.exists():
            try:
                import json as _json

                names = _json.loads(classes_path.read_text())
                class_idx_map = {name: idx for idx, name in enumerate(names)}
            except Exception as e:
                print(f"[w] classes.json read failed: {e}")

        try:
            if not self._db_initialized:
                init_database()
                self._db_initialized = True
            session = get_db_session()
        except Exception as e:
            print(f"[!] sync_dataset_to_db: {e}")
            return {"created": 0, "updated": 0, "total": 0}

        created = updated = total = 0
        try:
            for label_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
                samples = list(label_dir.glob("sample_*.npy"))
                if not samples:
                    continue
                total += 1
                label = label_dir.name
                row = (
                    session.query(DbGesture)
                    .filter(DbGesture.label == label)
                    .first()
                )
                model_class_id = class_idx_map.get(label)
                if row is None:
                    row = DbGesture(
                        label=label,
                        description=f"Auto-imported from {data_root / label}",
                        samples_path=str(label_dir),
                        model_class_id=model_class_id,
                        is_active=True,
                    )
                    session.add(row)
                    created += 1
                else:
                    row.samples_path = str(label_dir)
                    if model_class_id is not None:
                        row.model_class_id = model_class_id
                    row.is_active = True
                    updated += 1
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"[!] sync_dataset_to_db commit: {e}")
        finally:
            session.close()
        return {"created": created, "updated": updated, "total": total}

    def list_recorded_gestures(self) -> list[dict[str, Any]]:
        """Возвращает список папок в ``data/gestures/<label>/`` с числом
        сэмплов в каждой. Используется на вкладке «Обучение»."""
        root = self._configured_data_dir()
        out: list[dict[str, Any]] = []
        if not root.exists():
            return out
        for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            samples = sorted(label_dir.glob("sample_*.npy"))
            out.append({"label": label_dir.name, "samples": len(samples)})
        return out

    def start_recording(
        self,
        label: str,
        num_samples: int = 20,
        frames: int = 30,
        two_hands: bool = False,
        on_line: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """Запустить ``cv/record_gestures.py`` как subprocess.

        ``on_line`` будет вызываться из фонового потока с каждой строкой
        stdout/stderr CLI-скрипта. ``on_done(exit_code)`` — по завершению.
        """
        if self._training_proc is not None and self._training_proc.poll() is None:
            return False
        clean = (label or "").strip()
        if not clean:
            return False

        project_root = Path(__file__).resolve().parents[2]
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "cv.record_gestures",
            "--label",
            clean,
            "--num-samples",
            str(int(num_samples)),
            "--frames",
            str(int(frames)),
            "--data-root",
            str(self._configured_data_dir()),
            "--camera-index",
            str(int(self._config.recognition.camera_index)),
            "--fps",
            str(int(self._config.recognition.target_fps)),
        ]
        if two_hands:
            cmd.append("--two-hands")

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
                if on_done:
                    try:
                        on_done(int(code))
                    except Exception:
                        pass

        Thread(target=reader, daemon=True).start()
        return True

    def start_training(
        self,
        data_root: str = "",
        out_path: str = "",
        neighbors: int = 5,
        expect_dim: Optional[int] = None,
        on_line: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """Запустить ``cv/train_classifier.py`` как subprocess."""
        if self._training_proc is not None and self._training_proc.poll() is None:
            return False
        project_root = Path(__file__).resolve().parents[2]
        actual_data_root = data_root or str(self._configured_data_dir())
        actual_out_path = out_path or str(self._configured_model_path())
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
            str(self._configured_classes_path()),
            "--feature-dim-out",
            str(self._configured_feature_dim_path()),
            "--neighbors",
            str(int(neighbors)),
        ]
        if expect_dim is not None:
            cmd += ["--expect-dim", str(int(expect_dim))]

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
                                f"всего классов: {summary['total']}"
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

    def cancel_training(self) -> None:
        """Прервать текущий тренировочный subprocess (если запущен)."""
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
        return self._training_proc is not None and self._training_proc.poll() is None

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
        if self._voice_assistant is not None:
            try:
                self._voice_assistant.stop_listening_loop()
            except Exception:
                pass
            self._voice_assistant = None
