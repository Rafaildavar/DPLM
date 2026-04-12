#!/usr/bin/env python3
"""
Главный файл запуска приложения DPLM
Main entry point for DPLM application

Запуск / Launch:
    python -m app.main
"""
import sys
import os
import site
import signal
import subprocess
from pathlib import Path
from typing import Any, Optional, Tuple


def _resolve_pyside6_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Вернуть (plugins_path, platforms_path) для установленного PySide6 (без импорта PySide6)."""
    candidates = []
    for site_dir in site.getsitepackages():
        candidates.append(Path(site_dir) / "PySide6")
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site) / "PySide6")
    for base in candidates:
        plugins = base / "Qt" / "plugins"
        platforms = plugins / "platforms"
        if plugins.exists():
            return plugins.resolve(), (platforms.resolve() if platforms.exists() else None)
    return None, None


def _configure_qt_environment_for_macos() -> None:
    """Настроить Qt plugin paths для macOS до импорта PySide6."""
    if sys.platform != "darwin":
        return
    plugins_path, platforms_path = _resolve_pyside6_paths()
    if plugins_path:
        os.environ["QT_PLUGIN_PATH"] = str(plugins_path)
        print(f"[i] Qt plugins path: {plugins_path}")
    else:
        print("[!] Warning: Qt plugins path not found (PySide6/Qt/plugins)")
    if platforms_path:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_path)
        print(f"[i] Qt platform plugins path: {platforms_path}")
    os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")


# ВАЖНО: Настройка Qt окружения ДО импорта PySide6
_configure_qt_environment_for_macos()

# Теперь можно импортировать PySide6
from PySide6.QtCore import QCoreApplication, QUrl, QObject, Slot, Signal, Property, QTimer, QSize
from PySide6.QtGui import QGuiApplication, QIcon, QImage
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickImageProvider

# Установка путей к библиотекам Qt после импорта QCoreApplication
if sys.platform == "darwin":
    plugins_path, _ = _resolve_pyside6_paths()
    if plugins_path:
        QCoreApplication.setLibraryPaths([str(plugins_path)])
        print(f"[i] Qt library paths set: {QCoreApplication.libraryPaths()}")

# Импорт сервисов / Import services
try:
    from app.services.voice_assistant import VoiceAssistant, Language, create_voice_assistant
    from app.services.command_executor import get_executor
    SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Сервисы не доступны: {e}")
    SERVICES_AVAILABLE = False
    VoiceAssistant = None
    Language = None
    create_voice_assistant = None
    get_executor = None

try:
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class DplmCameraImageProvider(QQuickImageProvider):
    """Поставляет последний кадр превью камеры в QML (image://dplmcam/preview)."""

    def __init__(self, controller: "AppController"):
        super().__init__(QQuickImageProvider.Image)
        self._controller = controller

    def requestImage(self, path_id: str, size: QSize, requested_size: QSize) -> QImage:
        # Сигнатура Qt 6 / PySide6: (id, size*, requestedSize) -> QImage; size — out-параметр
        img = self._controller.camera_preview_for_provider()
        if img.isNull():
            # Пустой QImage даёт в QML "Failed to get image from provider"
            placeholder = QImage(2, 2, QImage.Format_RGB32)
            placeholder.fill(0xFF16161E)
            size.setWidth(placeholder.width())
            size.setHeight(placeholder.height())
            return placeholder
        size.setWidth(img.width())
        size.setHeight(img.height())
        return img


class AppController(QObject):
    """
    Контроллер приложения - связь между Python бэкендом и QML UI
    Application controller - bridge between Python backend and QML UI
    """
    
    # Сигналы для UI / Signals for UI
    statusChanged = Signal(str)  # Статус распознавания / Recognition status
    gestureDetected = Signal(str)  # Обнаружен жест / Gesture detected
    commandExecuted = Signal(str)  # Команда выполнена / Command executed
    voiceAssistantStateChanged = Signal(str)  # Состояние голосового помощника / Voice assistant state
    voiceCommandReceived = Signal(str)  # Получена голосовая команда / Voice command received
    cameraActiveChanged = Signal()
    cameraPreviewRevisionChanged = Signal()
    isRecognizingChanged = Signal()
    isVoiceAssistantActiveChanged = Signal()
    embeddedRecognitionConfidenceChanged = Signal()
    embeddedLandmarksJsonChanged = Signal()

    def __init__(self):
        super().__init__()
        self._status = "Idle"
        self._is_recognizing = False
        self._recognition_process: Optional[subprocess.Popen] = None
        self._recognition_pid_file = Path.home() / ".dplm" / "gesture_infer.pid"
        self._recognition_log_file = Path.home() / ".dplm" / "gesture_infer.log"
        self._recognition_pid_file.parent.mkdir(parents=True, exist_ok=True)
        if self._is_recognition_pid_active():
            self._is_recognizing = True
            self._status = "Recognizing in background"
            self.isRecognizingChanged.emit()

        # Голосовой помощник / Voice assistant
        self._voice_assistant: Optional[VoiceAssistant] = None
        self._voice_assistant_enabled = False
        
        # Исполнитель команд / Command executor
        if SERVICES_AVAILABLE and get_executor:
            self._command_executor = get_executor()
        else:
            self._command_executor = None

        # Превью камеры (OpenCV → QML, те же настройки, что cv/realtime_infer.py)
        self._camera_cap: Optional[object] = None
        self._preview_qimage: Optional[QImage] = None
        self._is_camera_active = False
        self._camera_preview_revision = 0
        self._camera_timer = QTimer(self)
        self._camera_timer.setInterval(33)
        self._camera_timer.timeout.connect(self._update_camera_frame)

        # Встроенное CV на экране распознавания (MediaPipe + KNN, без subprocess)
        self._embedded_infer: Optional[Any] = None
        self._embedded_infer_active = False
        self._last_embedded_label = ""
        self._embedded_recognition_confidence = 0.0
        self._embedded_landmarks_json = "[]"

    @Property(str, notify=statusChanged)
    def status(self):
        """Текущий статус системы / Current system status"""
        return self._status
    
    @status.setter
    def status(self, value):
        if self._status != value:
            self._status = value
            self.statusChanged.emit(value)
    
    @Property(bool, notify=isRecognizingChanged)
    def isRecognizing(self):
        """Активно ли распознавание / Is recognition active"""
        return self._is_recognizing

    @Property(bool, notify=cameraActiveChanged)
    def isCameraActive(self) -> bool:
        return self._is_camera_active

    @Property(int, notify=cameraPreviewRevisionChanged)
    def cameraPreviewRevision(self) -> int:
        return self._camera_preview_revision

    @Property(float, notify=embeddedRecognitionConfidenceChanged)
    def embeddedRecognitionConfidence(self) -> float:
        return self._embedded_recognition_confidence

    @Property(str, notify=embeddedLandmarksJsonChanged)
    def embeddedLandmarksJson(self) -> str:
        return self._embedded_landmarks_json

    def camera_preview_for_provider(self) -> QImage:
        if self._preview_qimage is None or self._preview_qimage.isNull():
            return QImage()
        return self._preview_qimage

    @Slot()
    def startCamera(self) -> None:
        """Открыть камеру для превью в QML (тот же захват, что в cv/realtime_infer.py)."""
        if self._is_camera_active:
            return
        if not _CV2_AVAILABLE:
            self.status = "Camera: OpenCV not installed"
            print("[!] OpenCV (cv2) недоступен — превью камеры отключено")
            return
        if self._is_recognizing:
            print("[w] Запущено фоновое распознавание (realtime_infer.py); оно может занять камеру 0.")
        try:
            from app.cv_camera import open_default_capture

            cap = open_default_capture()
        except Exception as e:
            print(f"[!] Ошибка открытия камеры: {e}")
            self.status = f"Camera error: {e}"
            return
        if not cap.isOpened():
            self.status = "Camera: open failed (check macOS Privacy → Camera)"
            print("[!] cv2.VideoCapture не открыл устройство")
            try:
                cap.release()
            except Exception:
                pass
            return
        self._camera_cap = cap
        self._is_camera_active = True
        self.cameraActiveChanged.emit()
        self._camera_timer.start()
        self.status = "Camera: streaming (QML preview)"
        print("[✓] Камера открыта для превью в интерфейсе")

    @Slot()
    def stopCamera(self) -> None:
        """Остановить превью и освободить камеру."""
        self._camera_timer.stop()
        self._preview_qimage = None
        if self._camera_cap is not None:
            try:
                self._camera_cap.release()
            except Exception:
                pass
            self._camera_cap = None
        if self._is_camera_active:
            self._is_camera_active = False
            self.cameraActiveChanged.emit()
        self._camera_preview_revision += 1
        self.cameraPreviewRevisionChanged.emit()
        if self._is_recognition_pid_active():
            self.status = "Recognizing in background"
        elif self._status.startswith("Camera:"):
            self.status = "Stopped"
        print("[i] Превью камеры остановлено")

    def _update_camera_frame(self) -> None:
        if not self._camera_cap or not self._is_camera_active:
            return
        ok, frame_bgr = self._camera_cap.read()
        if not ok:
            return
        import cv2
        import numpy as np

        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        frame_rgb = np.ascontiguousarray(frame_rgb)
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self._preview_qimage = qimg

        if self._embedded_infer_active and self._embedded_infer is not None:
            try:
                out = self._embedded_infer.process_frame_rgb(frame_rgb)
            except Exception as e:
                print(f"[!] embedded CV frame error: {e}")
                out = {"label": "", "confidence": 0.0, "landmarks_json": "[]"}
            label = (out.get("label") or "").strip()
            conf = float(out.get("confidence") or 0.0)
            lj = out.get("landmarks_json") or "[]"

            if label and label != self._last_embedded_label:
                self._last_embedded_label = label
                self.gestureDetected.emit(label)

            if self._embedded_recognition_confidence != conf:
                self._embedded_recognition_confidence = conf
                self.embeddedRecognitionConfidenceChanged.emit()
            if self._embedded_landmarks_json != lj:
                self._embedded_landmarks_json = lj
                self.embeddedLandmarksJsonChanged.emit()

        self._camera_preview_revision += 1
        self.cameraPreviewRevisionChanged.emit()

    @Slot()
    def startRecognition(self):
        """
        Запустить распознавание жестов в реальном времени
        Start real-time gesture recognition
        """
        self.stopCamera()
        if self._is_recognition_pid_active():
            print("[i] Распознавание уже запущено в фоне")
            self._is_recognizing = True
            self.isRecognizingChanged.emit()
            self.status = "Recognizing in background"
            return

        print("[i] Запуск фонового распознавания жестов...")
        infer_script = Path(__file__).parent.parent / "cv" / "realtime_infer.py"
        cmd = [
            sys.executable,
            str(infer_script.resolve()),
            "--tts",
        ]

        try:
            log_handle = self._recognition_log_file.open("a", encoding="utf-8")
            self._recognition_process = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._save_recognition_pid(self._recognition_process.pid)
            self._is_recognizing = True
            self.isRecognizingChanged.emit()
            self.status = "Recognizing in background"
            print(f"[✓] Фоновое распознавание запущено, PID={self._recognition_process.pid}")
        except Exception as e:
            print(f"[!] Ошибка запуска распознавания: {e}")
            self._is_recognizing = False
            self.isRecognizingChanged.emit()
            self.status = "Recognition start failed"
    
    @Slot()
    def startEmbeddedGestureRecognition(self) -> None:
        """
        Режим «камера в приложении + MediaPipe/KNN» для экрана распознавания.
        Не использует отдельный процесс realtime_infer.py (освобождает камеру 0).
        """
        if self._embedded_infer_active:
            return
        self.stopRecognition()
        try:
            from app.gesture_online_infer import GestureOnlineInfer

            self._embedded_infer = GestureOnlineInfer()
        except Exception as e:
            print(f"[!] Встроенное CV: не удалось создать движок: {e}")
            self.status = f"CV init error: {e}"
            self._embedded_infer = None
            return

        if self._embedded_infer.init_error:
            print(f"[!] Встроенное CV: {self._embedded_infer.init_error}")
            self.status = f"CV: {self._embedded_infer.init_error}"
            self._embedded_infer.close()
            self._embedded_infer = None
            return

        if self._embedded_infer.model_error:
            print(f"[i] Встроенное CV (без классификатора): {self._embedded_infer.model_error}")

        self._embedded_infer_active = True
        self._last_embedded_label = ""
        self._embedded_recognition_confidence = 0.0
        self._embedded_landmarks_json = "[]"
        self.embeddedRecognitionConfidenceChanged.emit()
        self.embeddedLandmarksJsonChanged.emit()
        if not self._is_camera_active:
            self.startCamera()
        self.status = "CV: embedded gesture recognition"
        print("[✓] Встроенное распознавание жестов (камера + MediaPipe) включено")

    @Slot()
    def stopEmbeddedGestureRecognition(self) -> None:
        """Выключить встроенный пайплайн (камеру не останавливаем — ей управляет UI)."""
        if not self._embedded_infer_active and self._embedded_infer is None:
            return
        self._embedded_infer_active = False
        self._last_embedded_label = ""
        if self._embedded_recognition_confidence != 0.0:
            self._embedded_recognition_confidence = 0.0
            self.embeddedRecognitionConfidenceChanged.emit()
        if self._embedded_landmarks_json != "[]":
            self._embedded_landmarks_json = "[]"
            self.embeddedLandmarksJsonChanged.emit()
        if self._embedded_infer is not None:
            self._embedded_infer.close()
            self._embedded_infer = None
        if self._status.startswith("CV: embedded"):
            self.status = "Idle"
        print("[i] Встроенное распознавание жестов выключено")

    @Slot()
    def shutdownCvPipeline(self) -> None:
        """Полная остановка камеры, встроенного CV и фонового infer при выходе."""
        self.stopEmbeddedGestureRecognition()
        self.stopRecognition()
        self.stopCamera()

    @Slot()
    def stopRecognition(self):
        """
        Остановить распознавание жестов
        Stop gesture recognition
        """
        print("[i] Остановка распознавания жестов...")
        pid = self._read_recognition_pid()
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"[✓] Процесс распознавания остановлен, PID={pid}")
            except ProcessLookupError:
                print(f"[w] Процесс PID={pid} уже завершён")
            except Exception as e:
                print(f"[!] Ошибка остановки PID={pid}: {e}")

        self._remove_recognition_pid_file()
        self._recognition_process = None
        self._is_recognizing = False
        self.isRecognizingChanged.emit()
        self.status = "Stopped"

    def _save_recognition_pid(self, pid: int):
        self._recognition_pid_file.write_text(str(pid), encoding="utf-8")

    def _read_recognition_pid(self) -> Optional[int]:
        if not self._recognition_pid_file.exists():
            return None
        try:
            return int(self._recognition_pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            return None

    def _remove_recognition_pid_file(self):
        if self._recognition_pid_file.exists():
            self._recognition_pid_file.unlink()

    def _is_recognition_pid_active(self) -> bool:
        pid = self._read_recognition_pid()
        if pid is None:
            return False

        try:
            os.kill(pid, 0)
            return True
        except OSError:
            self._remove_recognition_pid_file()
            return False
    
    @Slot(str)
    def startGestureTraining(self, gesture_label):
        """
        Запустить обучение нового жеста
        Start training a new gesture
        
        Args:
            gesture_label: название жеста / gesture name
        """
        print(f"[i] Начало обучения жесту: {gesture_label}")
        self.status = f"Training: {gesture_label}"
        # TODO: интеграция с cv/record_gestures.py
        # TODO: integration with cv/record_gestures.py
    
    @Slot(str, result=bool)
    def executeCommand(self, command_name):
        """
        Выполнить команду по имени
        Execute command by name
        
        Args:
            command_name: название команды / command name
        
        Returns:
            bool: успешно ли выполнена команда / whether command succeeded
        """
        print(f"[i] Выполнение команды: {command_name}")
        
        # Выполнение через CommandExecutor / Execute via CommandExecutor
        success = False
        if self._command_executor:
            success = self._command_executor.execute(command_name)
        else:
            print("[w] CommandExecutor не доступен")
            success = True  # Fallback для демо
        
        if success:
            self.commandExecuted.emit(command_name)
        
        return success
    
    @Slot(result=list)
    def getAvailableCommands(self):
        """
        Получить список доступных команд
        Get list of available commands
        
        Returns:
            list: список команд из БД / list of commands from DB
        """
        # TODO: загрузка из БД через app/backend/command_manager.py
        # TODO: load from DB via app/backend/command_manager.py
        
        # Получение команд из CommandExecutor / Get commands from CommandExecutor
        if self._command_executor:
            commands = []
            for name, config in self._command_executor.commands_registry.items():
                commands.append({
                    "name": name,
                    "description": config.get("description", ""),
                    "platform": config.get("platform", "all")
                })
            return commands
        
        # Fallback для демо / Fallback for demo
        return [
            {"name": "Open Browser", "gesture": "swipe_right"},
            {"name": "Volume Up", "gesture": "thumbs_up"},
            {"name": "Close Window", "gesture": "palm"},
        ]
    
    # ============================================================================
    # Голосовой помощник / Voice Assistant
    # ============================================================================
    
    @Slot(str, bool, bool, result=bool)
    def startVoiceAssistant(self, language: str = "ru", wake_word: bool = True, tts: bool = True):
        """
        Запустить голосового помощника
        Start voice assistant
        
        Args:
            language: язык ("ru", "en", "de") / language code
            wake_word: включить wake word / enable wake word
            tts: включить TTS / enable TTS
        
        Returns:
            bool: успешно ли запущен / whether started successfully
        """
        if not SERVICES_AVAILABLE or not create_voice_assistant:
            print("[!] Голосовой помощник не доступен")
            return False
        
        try:
            self._voice_assistant = create_voice_assistant(
                language=language,
                wake_word=wake_word,
                tts=tts,
                stt_engine="auto"
            )
            
            if self._voice_assistant:
                # Регистрация обработчика команд для голосового помощника
                # Register command handler for voice assistant
                def voice_command_handler(command_name: str):
                    """Обработчик голосовых команд / Voice command handler"""
                    self.voiceCommandReceived.emit(command_name)
                    # Выполнение команды / Execute command
                    success = self.executeCommand(command_name)
                    if success and self._voice_assistant:
                        self._voice_assistant.speak("Команда выполнена" if language == "ru" else "Command executed")
                
                # Регистрация команд из CommandExecutor в голосовом помощнике
                # Register commands from CommandExecutor in voice assistant
                if self._command_executor:
                    for cmd_name in self._command_executor.commands_registry.keys():
                        self._voice_assistant.register_command(
                            cmd_name,
                            lambda text, name=cmd_name: voice_command_handler(name)
                        )
                
                # Запуск цикла прослушивания / Start listening loop
                self._voice_assistant.start_listening_loop()
                self._voice_assistant_enabled = True
                self.voiceAssistantStateChanged.emit("active")
                self.isVoiceAssistantActiveChanged.emit()
                self.status = "Voice Assistant Active"
                return True
            else:
                return False
        
        except Exception as e:
            print(f"[!] Ошибка запуска голосового помощника: {e}")
            return False
    
    @Slot()
    def stopVoiceAssistant(self):
        """
        Остановить голосового помощника
        Stop voice assistant
        """
        if self._voice_assistant:
            self._voice_assistant.stop_listening_loop()
            self._voice_assistant = None
            self._voice_assistant_enabled = False
            self.voiceAssistantStateChanged.emit("idle")
            self.isVoiceAssistantActiveChanged.emit()
            self.status = "Voice Assistant Stopped"
    
    @Property(bool, notify=isVoiceAssistantActiveChanged)
    def isVoiceAssistantActive(self):
        """Активен ли голосовой помощник / Is voice assistant active"""
        return self._voice_assistant_enabled and self._voice_assistant is not None
    
    @Slot(str, result=str)
    def processVoiceCommand(self, text: str) -> str:
        """
        Обработать голосовую команду напрямую
        Process voice command directly
        
        Args:
            text: текст команды / command text
        
        Returns:
            Ответ помощника / Assistant response
        """
        if not self._voice_assistant:
            return "Голосовой помощник не активен"
        
        response = self._voice_assistant.process_command(text)
        return response if response else "Команда не распознана"
    
    @Slot(result=str)
    def getVoiceAssistantState(self) -> str:
        """
        Получить состояние голосового помощника
        Get voice assistant state
        
        Returns:
            Состояние / State
        """
        if not self._voice_assistant:
            return "not_initialized"
        return self._voice_assistant.state.value
    
    @Slot(result=list)
    def getAvailableVoices(self):
        """
        Получить список доступных голосов
        Get list of available voices
        
        Returns:
            Список голосов / List of voices
        """
        if not self._voice_assistant:
            return []
        return self._voice_assistant.get_available_voices()
    
    @Slot(str, result=list)
    def getAvailableVoicesByLanguage(self, language: str):
        """
        Получить список голосов по языку
        Get voices by language
        
        Args:
            language: код языка ("ru", "en", "de") / language code
        
        Returns:
            Список голосов / List of voices
        """
        # Если помощник не создан, создаем временный экземпляр для получения голосов
        # If assistant not created, create temporary instance to get voices
        if not self._voice_assistant:
            if SERVICES_AVAILABLE and create_voice_assistant:
                try:
                    temp_assistant = create_voice_assistant(
                        language=language,
                        wake_word=False,
                        tts=True,
                        stt_engine="auto"
                    )
                    if temp_assistant:
                        voices = temp_assistant.get_available_voices(language_filter=language)
                        # Очистка временного помощника / Cleanup temporary assistant
                        del temp_assistant
                        return voices
                except Exception as e:
                    print(f"[!] Ошибка получения голосов: {e}")
            return []
        return self._voice_assistant.get_available_voices(language_filter=language)
    
    @Slot(str, result=bool)
    def setVoiceAssistantVoice(self, voice_id: str) -> bool:
        """
        Установить голос помощника
        Set assistant voice
        
        Args:
            voice_id: ID голоса / voice ID
        
        Returns:
            True если голос установлен / True if voice set
        """
        if not self._voice_assistant:
            return False
        return self._voice_assistant.set_voice(voice_id)
    
    @Slot(result=dict)
    def getCurrentVoice(self):
        """
        Получить текущий голос
        Get current voice
        
        Returns:
            Информация о голосе или пустой dict / Voice info or empty dict
        """
        if not self._voice_assistant:
            return {}
        voice = self._voice_assistant.get_current_voice()
        return voice if voice else {}
    
    @Slot(float)
    def setVoiceAssistantSpeechRate(self, rate: float):
        """
        Установить скорость речи помощника
        Set assistant speech rate
        
        Args:
            rate: скорость (50-300) / rate (50-300)
        """
        if self._voice_assistant:
            self._voice_assistant.set_speech_rate(rate)
    
    @Slot(float)
    def setVoiceAssistantVolume(self, volume: float):
        """
        Установить громкость помощника
        Set assistant volume
        
        Args:
            volume: громкость (0.0-1.0) / volume (0.0-1.0)
        """
        if self._voice_assistant:
            self._voice_assistant.set_volume(volume)
    
    @Slot(str)
    def previewVoiceAssistantVoice(self, text: str = "Привет, это тест голоса"):
        """
        Предпросмотр голоса помощника
        Preview assistant voice
        
        Args:
            text: текст для предпросмотра / preview text
        """
        if self._voice_assistant:
            self._voice_assistant.preview_voice(text)


def main():
    """
    Главная функция запуска приложения
    Main application launch function
    """
    # Пути Qt заданы до импорта PySide6 и при загрузке модуля
    # Создание приложения / Create application
    app = QGuiApplication(sys.argv)
    app.setApplicationName("DPLM")
    app.setApplicationVersion("0.7.0")
    app.setOrganizationName("GUAP")
    app.setOrganizationDomain("guap.ru")
    
    # Создание QML движка / Create QML engine
    engine = QQmlApplicationEngine()
    
    # Создание контроллера и регистрация в QML
    # Create controller and register in QML
    controller = AppController()
    engine.addImageProvider("dplmcam", DplmCameraImageProvider(controller))
    engine.rootContext().setContextProperty("appController", controller)
    app.aboutToQuit.connect(controller.shutdownCvPipeline)
    
    # Загрузка главного QML файла / Load main QML file
    qml_file = Path(__file__).parent / "qml" / "MainWindow.qml"
    qml_url = QUrl.fromLocalFile(str(qml_file.resolve()))
    
    # Обработка ошибок загрузки QML / Handle QML loading errors
    def on_object_created(obj, url):
        if obj is None:
            print(f"[!] Ошибка загрузки QML: {url}")
            sys.exit(-1)
    
    engine.objectCreated.connect(on_object_created)
    engine.load(qml_url)
    
    # Проверка успешной загрузки / Check successful loading
    if not engine.rootObjects():
        print("[!] Не удалось загрузить QML интерфейс")
        sys.exit(-1)
    
    print("[✓] Приложение DPLM запущено")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
