# -*- coding: utf-8 -*-
"""
Тесты для голосового помощника
Tests for voice assistant
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from app.services.voice_assistant import (
        VoiceAssistant,
        Language,
        AssistantState,
        create_voice_assistant
    )
    VOICE_ASSISTANT_AVAILABLE = True
except ImportError:
    VOICE_ASSISTANT_AVAILABLE = False
    pytestmark = pytest.mark.skip("Voice assistant not available")


@pytest.mark.skipif(not VOICE_ASSISTANT_AVAILABLE, reason="Voice assistant not available")
class TestVoiceAssistant:
    """Тесты для класса VoiceAssistant / Tests for VoiceAssistant class"""
    
    @pytest.fixture
    def assistant(self):
        """Создать экземпляр помощника / Create assistant instance"""
        with patch('app.services.voice_assistant.pyttsx3') as mock_tts, \
             patch('app.services.voice_assistant.sr') as mock_sr:
            mock_tts.init.return_value = Mock()
            mock_sr.Recognizer.return_value = Mock()
            mock_sr.Microphone.return_value = Mock()
            
            return VoiceAssistant(
                language=Language.RUSSIAN,
                wake_word_enabled=False,  # Отключаем для тестов / Disable for tests
                tts_enabled=False,  # Отключаем для тестов / Disable for tests
                stt_engine="auto"
            )
    
    def test_assistant_initialization(self, assistant):
        """Тест инициализации помощника / Test assistant initialization"""
        assert assistant is not None
        assert assistant.state == AssistantState.IDLE
        assert assistant.language == Language.RUSSIAN
        assert not assistant.is_active
    
    def test_wake_word_detection(self, assistant):
        """Тест обнаружения ключевого слова / Test wake word detection"""
        assistant.wake_word_enabled = True
        
        # Тест с ключевым словом / Test with wake word
        assert assistant.detect_wake_word("ассистент открой браузер")
        assert assistant.detect_wake_word("помощник покажи команды")
        
        # Тест без ключевого слова / Test without wake word
        assistant.wake_word_enabled = False
        assert assistant.detect_wake_word("открой браузер")  # Всегда True если отключен
    
    def test_command_processing(self, assistant):
        """Тест обработки команд / Test command processing"""
        # Тест приветствия / Test greeting
        response = assistant.process_command("привет")
        assert response is not None
        assert len(response) > 0
        
        # Тест помощи / Test help
        response = assistant.process_command("помощь")
        assert response is not None
        assert "помочь" in response.lower() or "help" in response.lower()
        
        # Тест неизвестной команды / Test unknown command
        response = assistant.process_command("неизвестная команда 12345")
        assert response is not None
        assert "не понял" in response.lower() or "didn't understand" in response.lower()
    
    def test_command_registration(self, assistant):
        """Тест регистрации команд / Test command registration"""
        test_response = "Test response"
        
        def test_handler(text: str) -> str:
            return test_response
        
        assistant.register_command("тест", test_handler)
        response = assistant.process_command("тест")
        assert response == test_response
    
    def test_conversation_context(self, assistant):
        """Тест сохранения контекста разговора / Test conversation context saving"""
        initial_length = len(assistant.conversation_context)
        
        assistant.process_command("привет")
        assert len(assistant.conversation_context) > initial_length
        
        # Проверка структуры контекста / Check context structure
        last_entry = assistant.conversation_context[-1]
        assert "assistant" in last_entry or "user" in last_entry
        assert "timestamp" in last_entry
    
    def test_stats_tracking(self, assistant):
        """Тест отслеживания статистики / Test statistics tracking"""
        initial_commands = assistant.stats["commands_executed"]
        
        assistant.process_command("привет")
        # Статистика может не измениться для некоторых команд / Stats may not change for some commands
        
        # Проверка структуры статистики / Check stats structure
        stats = assistant.get_stats()
        assert "commands_executed" in stats
        assert "errors" in stats
        assert "state" in stats
        assert "is_active" in stats


@pytest.mark.skipif(not VOICE_ASSISTANT_AVAILABLE, reason="Voice assistant not available")
class TestVoiceAssistantFactory:
    """Тесты для фабричной функции / Tests for factory function"""
    
    @patch('app.services.voice_assistant.VoiceAssistant')
    def test_create_voice_assistant_ru(self, mock_assistant_class):
        """Тест создания помощника на русском / Test creating Russian assistant"""
        mock_assistant_class.return_value = Mock()
        
        assistant = create_voice_assistant(language="ru", wake_word=True, tts=True)
        assert assistant is not None
        mock_assistant_class.assert_called_once()
    
    @patch('app.services.voice_assistant.VoiceAssistant')
    def test_create_voice_assistant_en(self, mock_assistant_class):
        """Тест создания помощника на английском / Test creating English assistant"""
        mock_assistant_class.return_value = Mock()
        
        assistant = create_voice_assistant(language="en", wake_word=False, tts=False)
        assert assistant is not None


@pytest.mark.skipif(not VOICE_ASSISTANT_AVAILABLE, reason="Voice assistant not available")
class TestCommandHandlers:
    """Тесты обработчиков команд / Tests for command handlers"""
    
    @pytest.fixture
    def assistant(self):
        """Создать помощника для тестов / Create assistant for tests"""
        with patch('app.services.voice_assistant.pyttsx3'), \
             patch('app.services.voice_assistant.sr'):
            return VoiceAssistant(
                language=Language.RUSSIAN,
                wake_word_enabled=False,
                tts_enabled=False,
                stt_engine="auto"
            )
    
    def test_greeting_handler(self, assistant):
        """Тест обработчика приветствия / Test greeting handler"""
        response = assistant._handle_greeting("привет")
        assert response is not None
        assert len(response) > 0
    
    def test_help_handler(self, assistant):
        """Тест обработчика помощи / Test help handler"""
        response = assistant._handle_help("помощь")
        assert response is not None
        assert "помочь" in response.lower() or "help" in response.lower()
    
    def test_thanks_handler(self, assistant):
        """Тест обработчика благодарности / Test thanks handler"""
        response = assistant._handle_thanks("спасибо")
        assert response is not None
        assert len(response) > 0
    
    def test_stop_handler(self, assistant):
        """Тест обработчика остановки / Test stop handler"""
        response = assistant._handle_stop("стоп")
        assert response is not None
        assert assistant.state == AssistantState.IDLE or not assistant.is_active
