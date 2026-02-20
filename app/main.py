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
from pathlib import Path
from typing import Optional


def _resolve_pyside6_paths() -> tuple[Optional[Path], Optional[Path]]:
    """Вернуть (plugins_path, platforms_path) для установленного PySide6."""
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

    # Явно указываем платформу, если не задано пользователем.
    os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")


# ВАЖНО: Настройка Qt окружения ДО импорта PySide6
# IMPORTANT: Setup Qt environment BEFORE importing PySide6
_configure_qt_environment_for_macos()

# Теперь можно импортировать PySide6
# Now we can import PySide6
from PySide6.QtCore import QCoreApplication, QUrl, QObject, Slot, Signal, Property
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtQml import QQmlApplicationEngine

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
    
    def __init__(self):
        super().__init__()
        self._status = "Idle"
        self._is_recognizing = False
        
        # Голосовой помощник / Voice assistant
        self._voice_assistant: Optional[VoiceAssistant] = None
        self._voice_assistant_enabled = False
        
        # Исполнитель команд / Command executor
        if SERVICES_AVAILABLE and get_executor:
            self._command_executor = get_executor()
        else:
            self._command_executor = None
    
    @Property(str, notify=statusChanged)
    def status(self):
        """Текущий статус системы / Current system status"""
        return self._status
    
    @status.setter
    def status(self, value):
        if self._status != value:
            self._status = value
            self.statusChanged.emit(value)
    
    @Property(bool)
    def isRecognizing(self):
        """Активно ли распознавание / Is recognition active"""
        return self._is_recognizing
    
    @Slot()
    def startRecognition(self):
        """
        Запустить распознавание жестов в реальном времени
        Start real-time gesture recognition
        """
        print("[i] Запуск распознавания жестов...")
        self._is_recognizing = True
        self.status = "Recognizing..."
        # TODO: интеграция с cv/realtime_infer.py
        # TODO: integration with cv/realtime_infer.py
    
    @Slot()
    def stopRecognition(self):
        """
        Остановить распознавание жестов
        Stop gesture recognition
        """
        print("[i] Остановка распознавания жестов...")
        self._is_recognizing = False
        self.status = "Stopped"
    
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
            self.status = "Voice Assistant Stopped"
    
    @Property(bool)
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
    # Установка путей к библиотекам Qt перед созданием приложения
    # Set Qt library paths before creating application
    if sys.platform == "darwin":
        plugins_path, _ = _resolve_pyside6_paths()
        if plugins_path:
            QCoreApplication.setLibraryPaths([str(plugins_path)])
            print(f"[i] Qt library paths set: {QCoreApplication.libraryPaths()}")
    
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
    engine.rootContext().setContextProperty("appController", controller)
    
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

