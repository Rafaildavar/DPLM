# -*- coding: utf-8 -*-
"""
Голосовой помощник GestureBind - полнофункциональный ассистент
Voice Assistant GestureBind - full-featured assistant

Функциональность:
- STT (Speech-to-Text): распознавание речи через SpeechRecognition/Vosk
- TTS (Text-to-Speech): синтез речи через pyttsx3
- NLU (Natural Language Understanding): понимание намерений пользователя
- Диалоговая система: контекстные ответы и поддержание разговора
- Выполнение команд: запуск действий по голосовому запросу
- Wake word detection: активация по ключевому слову
- Контекстные подсказки: помощь пользователю
- Многоязычность: поддержка русского и английского
"""

import re
import time
import threading
import queue
from typing import Optional, Dict, List, Callable, Any
from enum import Enum
from pathlib import Path
import platform

# STT библиотеки
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("[WARN] SpeechRecognition not installed - STT disabled")

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("[WARN] Vosk not installed - offline STT disabled")

# TTS библиотека
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[WARN] pyttsx3 not installed - TTS disabled")


class Language(Enum):
    """Поддерживаемые языки / Supported languages"""
    RUSSIAN = "ru-RU"
    ENGLISH = "en-US"
    GERMAN = "de-DE"


class AssistantState(Enum):
    """Состояния голосового помощника / Voice assistant states"""
    IDLE = "idle"  # Ожидание активации / Waiting for activation
    LISTENING = "listening"  # Слушает команду / Listening for command
    PROCESSING = "processing"  # Обрабатывает команду / Processing command
    SPEAKING = "speaking"  # Говорит ответ / Speaking response
    ERROR = "error"  # Ошибка / Error state


class VoiceAssistant:
    """
    Голосовой помощник с полным функционалом современных ассистентов.
    Voice assistant with full functionality of modern assistants.
    """
    
    # Ключевые слова активации / Wake words
    WAKE_WORDS_RU = ["ассистент", "помощник", "эй", "слушай"]
    WAKE_WORDS_EN = ["assistant", "hey", "listen", "okay"]
    
    def __init__(
        self,
        language: Language = Language.RUSSIAN,
        wake_word_enabled: bool = True,
        tts_enabled: bool = True,
        stt_engine: str = "auto",  # "auto", "google", "vosk", "sphinx"
    ):
        """
        Инициализация голосового помощника.
        Initialize voice assistant.
        
        Args:
            language: язык помощника / assistant language
            wake_word_enabled: включить активацию по ключевому слову / enable wake word
            tts_enabled: включить синтез речи / enable TTS
            stt_engine: движок распознавания речи / STT engine
        """
        self.language = language
        self.wake_word_enabled = wake_word_enabled
        self.tts_enabled = tts_enabled
        self.stt_engine = stt_engine
        
        # Состояние / State
        self.state = AssistantState.IDLE
        self.is_active = False
        self.conversation_context: List[Dict[str, Any]] = []
        
        # Инициализация TTS / TTS initialization
        self.tts_engine = None
        self.current_voice_id = None
        self.available_voices: List[Dict[str, str]] = []
        if tts_enabled and TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self._load_available_voices()
                self._configure_tts()
            except Exception as e:
                print(f"[!] Ошибка инициализации TTS: {e}")
                self.tts_enabled = False
        
        # Инициализация STT / STT initialization
        self.recognizer = None
        self.microphone = None
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                # Калибровка микрофона / Microphone calibration
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
            except Exception as e:
                print(f"[!] Ошибка инициализации микрофона: {e}")
        
        # Vosk модель (для офлайн распознавания) / Vosk model (for offline recognition)
        self.vosk_model = None
        if VOSK_AVAILABLE and stt_engine == "vosk":
            self._load_vosk_model()
        
        # Очередь команд / Command queue
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Поток обработки / Processing thread
        self.processing_thread = None
        self.listening_thread = None
        self._stop_listening = False
        
        # Обработчики команд / Command handlers
        self.command_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
        
        # Статистика / Statistics
        self.stats = {
            "commands_executed": 0,
            "errors": 0,
            "wake_word_detected": 0,
        }
    
    def _load_available_voices(self):
        """Загрузить список доступных голосов / Load available voices list"""
        if not self.tts_engine:
            return
        
        try:
            voices = self.tts_engine.getProperty('voices')
            if voices:
                self.available_voices = []
                for voice in voices:
                    self.available_voices.append({
                        'id': voice.id,
                        'name': voice.name,
                        'language': self._detect_voice_language(voice.name, voice.id)
                    })
        except Exception as e:
            print(f"[!] Ошибка загрузки голосов: {e}")
            self.available_voices = []
    
    def _detect_voice_language(self, name: str, voice_id: str) -> str:
        """Определить язык голоса / Detect voice language"""
        name_lower = name.lower()
        id_lower = voice_id.lower()
        
        if any(word in name_lower or word in id_lower for word in ['russian', 'ru', 'русск']):
            return "ru"
        elif any(word in name_lower or word in id_lower for word in ['english', 'en', 'eng', 'англ']):
            return "en"
        elif any(word in name_lower or word in id_lower for word in ['german', 'de', 'deutsch', 'немец']):
            return "de"
        else:
            return "unknown"
    
    def _configure_tts(self, voice_id: Optional[str] = None):
        """Настройка TTS движка / Configure TTS engine"""
        if not self.tts_engine:
            return
        
        # Выбор голоса / Select voice
        if voice_id:
            # Использовать указанный голос / Use specified voice
            try:
                self.tts_engine.setProperty('voice', voice_id)
                self.current_voice_id = voice_id
            except Exception as e:
                print(f"[!] Ошибка установки голоса {voice_id}: {e}")
                voice_id = None
        
        if not voice_id:
            # Автоматический выбор голоса по языку / Auto-select voice by language
            voices = self.tts_engine.getProperty('voices')
            if voices:
                target_lang = "russian" if self.language == Language.RUSSIAN else "english"
                for voice in voices:
                    if target_lang in voice.name.lower() or target_lang in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        self.current_voice_id = voice.id
                        break
        
        # Скорость речи / Speech rate
        self.tts_engine.setProperty('rate', 150)
        
        # Громкость / Volume
        self.tts_engine.setProperty('volume', 0.9)
    
    def get_available_voices(self, language_filter: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Получить список доступных голосов.
        Get list of available voices.
        
        Args:
            language_filter: фильтр по языку ("ru", "en", "de") / language filter
        
        Returns:
            Список голосов с полями id, name, language / List of voices with id, name, language
        """
        if language_filter:
            return [v for v in self.available_voices if v['language'] == language_filter]
        return self.available_voices.copy()
    
    def set_voice(self, voice_id: str) -> bool:
        """
        Установить голос по ID.
        Set voice by ID.
        
        Args:
            voice_id: ID голоса / voice ID
        
        Returns:
            True если голос установлен успешно / True if voice set successfully
        """
        if not self.tts_engine:
            return False
        
        # Проверка существования голоса / Check if voice exists
        voice_ids = [v['id'] for v in self.available_voices]
        if voice_id not in voice_ids:
            print(f"[!] Голос с ID '{voice_id}' не найден")
            return False
        
        try:
            self.tts_engine.setProperty('voice', voice_id)
            self.current_voice_id = voice_id
            return True
        except Exception as e:
            print(f"[!] Ошибка установки голоса: {e}")
            return False
    
    def get_current_voice(self) -> Optional[Dict[str, str]]:
        """
        Получить текущий голос.
        Get current voice.
        
        Returns:
            Информация о текущем голосе или None / Current voice info or None
        """
        if not self.current_voice_id:
            return None
        
        for voice in self.available_voices:
            if voice['id'] == self.current_voice_id:
                return voice.copy()
        
        return None
    
    def set_speech_rate(self, rate: float):
        """
        Установить скорость речи.
        Set speech rate.
        
        Args:
            rate: скорость (50-300, по умолчанию 150) / rate (50-300, default 150)
        """
        if self.tts_engine:
            self.tts_engine.setProperty('rate', max(50, min(300, int(rate))))
    
    def set_volume(self, volume: float):
        """
        Установить громкость.
        Set volume.
        
        Args:
            volume: громкость (0.0-1.0) / volume (0.0-1.0)
        """
        if self.tts_engine:
            self.tts_engine.setProperty('volume', max(0.0, min(1.0, volume)))
    
    def preview_voice(self, text: str = "Привет, это тест голоса"):
        """
        Предпросмотр голоса с заданным текстом.
        Preview voice with given text.
        
        Args:
            text: текст для предпросмотра / text for preview
        """
        if not self.tts_enabled or not self.tts_engine:
            print("[!] TTS не доступен")
            return
        
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"[!] Ошибка предпросмотра голоса: {e}")
    
    def _load_vosk_model(self):
        """Загрузка модели Vosk для офлайн распознавания / Load Vosk model for offline recognition"""
        if not VOSK_AVAILABLE:
            return
        
        # Путь к модели Vosk / Path to Vosk model
        model_path = Path("models/vosk")
        if not model_path.exists():
            print("[!] Vosk модель не найдена. Используйте Google STT или установите модель Vosk.")
            return
        
        try:
            self.vosk_model = vosk.Model(str(model_path))
            print("[✓] Vosk модель загружена")
        except Exception as e:
            print(f"[!] Ошибка загрузки Vosk модели: {e}")
    
    def _register_default_handlers(self):
        """Регистрация обработчиков команд по умолчанию / Register default command handlers"""
        
        # Системные команды / System commands
        self.register_command("привет", self._handle_greeting)
        self.register_command("hello", self._handle_greeting)
        self.register_command("помощь", self._handle_help)
        self.register_command("help", self._handle_help)
        self.register_command("спасибо", self._handle_thanks)
        self.register_command("thanks", self._handle_thanks)
        self.register_command("пока", self._handle_goodbye)
        self.register_command("goodbye", self._handle_goodbye)
        
        # Команды управления / Control commands
        self.register_command("стоп", self._handle_stop)
        self.register_command("stop", self._handle_stop)
        self.register_command("выход", self._handle_exit)
        self.register_command("exit", self._handle_exit)
        
        # Команды информации / Information commands
        self.register_command("статус", self._handle_status)
        self.register_command("status", self._handle_status)
        self.register_command("команды", self._handle_list_commands)
        self.register_command("commands", self._handle_list_commands)
    
    def register_command(self, pattern: str, handler: Callable):
        """
        Регистрация обработчика команды.
        Register command handler.
        
        Args:
            pattern: паттерн команды (может быть regex) / command pattern (can be regex)
            handler: функция-обработчик / handler function
        """
        self.command_handlers[pattern.lower()] = handler
    
    def speak(self, text: str, async_mode: bool = False):
        """
        Произнести текст / Speak text.
        
        Args:
            text: текст для произнесения / text to speak
            async_mode: асинхронный режим / async mode
        """
        if not self.tts_enabled or not self.tts_engine:
            print(f"[TTS] {text}")
            return
        
        self.state = AssistantState.SPEAKING
        
        def _speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                self.state = AssistantState.IDLE
            except Exception as e:
                print(f"[!] Ошибка TTS: {e}")
                self.state = AssistantState.ERROR
        
        if async_mode:
            threading.Thread(target=_speak, daemon=True).start()
        else:
            _speak()
    
    def listen(self, timeout: float = 5.0, phrase_time_limit: float = 5.0) -> Optional[str]:
        """
        Распознать речь с микрофона / Recognize speech from microphone.
        
        Args:
            timeout: таймаут ожидания речи / timeout for waiting speech
            phrase_time_limit: максимальная длина фразы / maximum phrase length
        
        Returns:
            Распознанный текст или None / Recognized text or None
        """
        if not self.recognizer or not self.microphone:
            return None
        
        self.state = AssistantState.LISTENING
        
        try:
            with self.microphone as source:
                # Слушаем с учетом шума / Listen with noise adjustment
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            # Распознавание в зависимости от движка / Recognition based on engine
            text = None
            
            if self.stt_engine == "auto" or self.stt_engine == "google":
                # Google Speech Recognition (требует интернет) / Google STT (requires internet)
                try:
                    lang_code = self.language.value
                    text = self.recognizer.recognize_google(audio, language=lang_code)
                except sr.UnknownValueError:
                    print("[i] Не удалось распознать речь / Could not recognize speech")
                except sr.RequestError as e:
                    print(f"[!] Ошибка сервиса распознавания: {e}")
                    # Fallback на Vosk если доступен / Fallback to Vosk if available
                    if VOSK_AVAILABLE:
                        text = self._recognize_vosk(audio)
            
            elif self.stt_engine == "vosk":
                text = self._recognize_vosk(audio)
            
            elif self.stt_engine == "sphinx":
                # CMU Sphinx (офлайн) / CMU Sphinx (offline)
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                except sr.UnknownValueError:
                    print("[i] Не удалось распознать речь / Could not recognize speech")
            
            self.state = AssistantState.IDLE
            return text
            
        except sr.WaitTimeoutError:
            print("[i] Таймаут ожидания речи / Speech timeout")
            self.state = AssistantState.IDLE
            return None
        except Exception as e:
            print(f"[!] Ошибка распознавания речи: {e}")
            self.state = AssistantState.ERROR
            return None
    
    def _recognize_vosk(self, audio) -> Optional[str]:
        """Распознавание через Vosk / Recognition via Vosk"""
        if not self.vosk_model or not VOSK_AVAILABLE:
            return None
        
        try:
            # Конвертация аудио для Vosk / Convert audio for Vosk
            import json
            audio_data = audio.get_wav_data()
            
            rec = vosk.KaldiRecognizer(self.vosk_model, 16000)
            rec.AcceptWaveform(audio_data)
            result = json.loads(rec.Result())
            
            return result.get("text", "").strip()
        except Exception as e:
            print(f"[!] Ошибка Vosk распознавания: {e}")
            return None
    
    def detect_wake_word(self, text: str) -> bool:
        """
        Определить наличие ключевого слова активации / Detect wake word presence.
        
        Args:
            text: распознанный текст / recognized text
        
        Returns:
            True если обнаружено ключевое слово / True if wake word detected
        """
        if not self.wake_word_enabled:
            return True  # Всегда активно если wake word отключен / Always active if disabled
        
        text_lower = text.lower()
        wake_words = self.WAKE_WORDS_RU if self.language == Language.RUSSIAN else self.WAKE_WORDS_EN
        
        for word in wake_words:
            if word in text_lower:
                self.stats["wake_word_detected"] += 1
                return True
        
        return False
    
    def process_command(self, text: str) -> Optional[str]:
        """
        Обработать команду и вернуть ответ / Process command and return response.
        
        Args:
            text: текст команды / command text
        
        Returns:
            Ответ помощника или None / Assistant response or None
        """
        if not text:
            return None
        
        self.state = AssistantState.PROCESSING
        text_lower = text.lower().strip()
        
        # Сохранение в контекст / Save to context
        self.conversation_context.append({
            "user": text,
            "timestamp": time.time()
        })
        
        # Поиск обработчика команды / Find command handler
        response = None
        
        # Проверка точных совпадений / Check exact matches
        if text_lower in self.command_handlers:
            handler = self.command_handlers[text_lower]
            response = handler(text)
        
        # Проверка паттернов (regex) / Check patterns (regex)
        else:
            for pattern, handler in self.command_handlers.items():
                if re.search(pattern, text_lower, re.IGNORECASE):
                    response = handler(text)
                    break
        
        # Если команда не найдена / If command not found
        if not response:
            response = self._handle_unknown_command(text)
        
        # Сохранение ответа в контекст / Save response to context
        if response:
            self.conversation_context.append({
                "assistant": response,
                "timestamp": time.time()
            })
            # Ограничение размера контекста / Limit context size
            if len(self.conversation_context) > 20:
                self.conversation_context = self.conversation_context[-20:]
        
        self.state = AssistantState.IDLE
        return response
    
    def start_listening_loop(self):
        """Запустить цикл прослушивания / Start listening loop"""
        if self.listening_thread and self.listening_thread.is_alive():
            return
        
        self._stop_listening = False
        self.is_active = True
        
        def _listen_loop():
            print("[✓] Голосовой помощник активирован / Voice assistant activated")
            self.speak("Готов к работе" if self.language == Language.RUSSIAN else "Ready to work")
            
            while not self._stop_listening:
                try:
                    # Слушаем команду / Listen for command
                    text = self.listen(timeout=1.0, phrase_time_limit=5.0)
                    
                    if text:
                        # Проверка wake word / Check wake word
                        if self.detect_wake_word(text):
                            # Удаление wake word из текста / Remove wake word from text
                            wake_words = self.WAKE_WORDS_RU if self.language == Language.RUSSIAN else self.WAKE_WORDS_EN
                            for word in wake_words:
                                text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE).strip()
                            
                            if text:  # Если есть команда после wake word / If command after wake word
                                response = self.process_command(text)
                                if response:
                                    self.speak(response, async_mode=False)
                        else:
                            # Если wake word не обнаружен, но помощник активен / If wake word not detected but assistant active
                            if self.is_active:
                                response = self.process_command(text)
                                if response:
                                    self.speak(response, async_mode=False)
                
                except Exception as e:
                    print(f"[!] Ошибка в цикле прослушивания: {e}")
                    self.stats["errors"] += 1
                    time.sleep(1)
        
        self.listening_thread = threading.Thread(target=_listen_loop, daemon=True)
        self.listening_thread.start()
    
    def stop_listening_loop(self):
        """Остановить цикл прослушивания / Stop listening loop"""
        self._stop_listening = True
        self.is_active = False
        print("[i] Голосовой помощник остановлен / Voice assistant stopped")
    
    # ============================================================================
    # Обработчики команд по умолчанию / Default command handlers
    # ============================================================================
    
    def _handle_greeting(self, text: str) -> str:
        """Обработка приветствия / Handle greeting"""
        greetings_ru = [
            "Привет! Чем могу помочь?",
            "Здравствуйте! Готов помочь.",
            "Привет! Слушаю вас."
        ]
        greetings_en = [
            "Hello! How can I help?",
            "Hi there! Ready to assist.",
            "Hello! I'm listening."
        ]
        greetings = greetings_ru if self.language == Language.RUSSIAN else greetings_en
        import random
        return random.choice(greetings)
    
    def _handle_help(self, text: str) -> str:
        """Обработка запроса помощи / Handle help request"""
        help_ru = (
            "Я могу помочь вам:\n"
            "- Выполнить команды по жестам\n"
            "- Управлять системой голосом\n"
            "- Обучить новые жесты\n"
            "- Показать список доступных команд\n"
            "Скажите 'команды' чтобы увидеть список."
        )
        help_en = (
            "I can help you:\n"
            "- Execute gesture commands\n"
            "- Control system by voice\n"
            "- Train new gestures\n"
            "- Show available commands\n"
            "Say 'commands' to see the list."
        )
        return help_ru if self.language == Language.RUSSIAN else help_en
    
    def _handle_thanks(self, text: str) -> str:
        """Обработка благодарности / Handle thanks"""
        thanks_ru = ["Пожалуйста!", "Рад помочь!", "Всегда к вашим услугам!"]
        thanks_en = ["You're welcome!", "Happy to help!", "Always at your service!"]
        thanks = thanks_ru if self.language == Language.RUSSIAN else thanks_en
        import random
        return random.choice(thanks)
    
    def _handle_goodbye(self, text: str) -> str:
        """Обработка прощания / Handle goodbye"""
        return "До свидания!" if self.language == Language.RUSSIAN else "Goodbye!"
    
    def _handle_stop(self, text: str) -> str:
        """Обработка команды остановки / Handle stop command"""
        self.stop_listening_loop()
        return "Остановлено" if self.language == Language.RUSSIAN else "Stopped"
    
    def _handle_exit(self, text: str) -> str:
        """Обработка команды выхода / Handle exit command"""
        self.stop_listening_loop()
        return "Выход" if self.language == Language.RUSSIAN else "Exit"
    
    def _handle_status(self, text: str) -> str:
        """Обработка запроса статуса / Handle status request"""
        status_ru = (
            f"Статус: {self.state.value}\n"
            f"Выполнено команд: {self.stats['commands_executed']}\n"
            f"Ошибок: {self.stats['errors']}"
        )
        status_en = (
            f"Status: {self.state.value}\n"
            f"Commands executed: {self.stats['commands_executed']}\n"
            f"Errors: {self.stats['errors']}"
        )
        return status_ru if self.language == Language.RUSSIAN else status_en
    
    def _handle_list_commands(self, text: str) -> str:
        """Обработка запроса списка команд / Handle list commands request"""
        commands_list = ", ".join(self.command_handlers.keys())
        return f"Доступные команды: {commands_list}" if self.language == Language.RUSSIAN else f"Available commands: {commands_list}"
    
    def _handle_unknown_command(self, text: str) -> str:
        """Обработка неизвестной команды / Handle unknown command"""
        unknown_ru = "Извините, я не понял команду. Скажите 'помощь' для списка команд."
        unknown_en = "Sorry, I didn't understand the command. Say 'help' for a list of commands."
        return unknown_ru if self.language == Language.RUSSIAN else unknown_en
    
    def execute_gesture_command(self, command_name: str) -> bool:
        """
        Выполнить команду по жесту (интеграция с системой команд).
        Execute gesture command (integration with command system).
        
        Args:
            command_name: название команды / command name
        
        Returns:
            True если команда выполнена успешно / True if command executed successfully
        """
        # TODO: интеграция с app/services/command_executor.py
        # TODO: integration with app/services/command_executor.py
        self.stats["commands_executed"] += 1
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику работы / Get work statistics"""
        return {
            **self.stats,
            "state": self.state.value,
            "is_active": self.is_active,
            "conversation_length": len(self.conversation_context),
        }


# ============================================================================
# Утилиты / Utilities
# ============================================================================

def create_voice_assistant(
    language: str = "ru",
    wake_word: bool = True,
    tts: bool = True,
    stt_engine: str = "auto"
) -> Optional[VoiceAssistant]:
    """
    Фабричная функция для создания голосового помощника.
    Factory function to create voice assistant.
    
    Args:
        language: язык ("ru", "en", "de") / language code
        wake_word: включить wake word / enable wake word
        tts: включить TTS / enable TTS
        stt_engine: движок STT / STT engine
    
    Returns:
        Экземпляр VoiceAssistant или None / VoiceAssistant instance or None
    """
    lang_map = {
        "ru": Language.RUSSIAN,
        "en": Language.ENGLISH,
        "de": Language.GERMAN,
    }
    
    lang = lang_map.get(language.lower(), Language.RUSSIAN)
    
    try:
        return VoiceAssistant(
            language=lang,
            wake_word_enabled=wake_word,
            tts_enabled=tts,
            stt_engine=stt_engine
        )
    except Exception as e:
        print(f"[!] Ошибка создания голосового помощника: {e}")
        return None
