"""
Юнит-тесты для AppController
Unit tests for AppController

Тесты контроллера приложения (app/main.py)
Tests for application controller (app/main.py)
"""
import pytest
from PySide6.QtCore import QObject, Signal
from unittest.mock import Mock, patch


def test_app_controller_import():
    """
    Тест импорта AppController
    Test AppController import
    """
    from app.main import AppController
    assert AppController is not None


def test_app_controller_initialization():
    """
    Тест инициализации AppController
    Test AppController initialization
    """
    from app.main import AppController
    
    controller = AppController()
    assert controller is not None
    assert isinstance(controller, QObject)
    assert controller.status == "Idle"
    assert controller.isRecognizing == False


def test_app_controller_status_property():
    """
    Тест свойства status
    Test status property
    """
    from app.main import AppController
    
    controller = AppController()
    
    # Проверка начального значения / Check initial value
    assert controller.status == "Idle"
    
    # Изменение статуса / Change status
    controller.status = "Recognizing..."
    assert controller.status == "Recognizing..."


def test_app_controller_status_signal(qtbot):
    """
    Тест сигнала statusChanged
    Test statusChanged signal
    """
    from app.main import AppController
    
    controller = AppController()
    
    # Создаём мок для отслеживания сигнала
    # Create mock to track signal
    with qtbot.waitSignal(controller.statusChanged, timeout=1000):
        controller.status = "Testing"


def test_app_controller_start_recognition():
    """
    Тест метода startRecognition
    Test startRecognition method
    """
    from app.main import AppController
    
    controller = AppController()
    
    # Запуск распознавания / Start recognition
    controller.startRecognition()
    
    assert controller.isRecognizing == True
    assert controller.status == "Recognizing..."


def test_app_controller_stop_recognition():
    """
    Тест метода stopRecognition
    Test stopRecognition method
    """
    from app.main import AppController
    
    controller = AppController()
    
    # Сначала запускаем / First start
    controller.startRecognition()
    assert controller.isRecognizing == True
    
    # Затем останавливаем / Then stop
    controller.stopRecognition()
    assert controller.isRecognizing == False
    assert controller.status == "Stopped"


def test_app_controller_start_gesture_training():
    """
    Тест метода startGestureTraining
    Test startGestureTraining method
    """
    from app.main import AppController
    
    controller = AppController()
    
    gesture_label = "test_gesture"
    controller.startGestureTraining(gesture_label)
    
    assert controller.status == f"Training: {gesture_label}"


def test_app_controller_execute_command():
    """
    Тест метода executeCommand
    Test executeCommand method
    """
    from app.main import AppController
    
    controller = AppController()
    
    command_name = "test_command"
    result = controller.executeCommand(command_name)
    
    # TODO: Пока возвращает True (заглушка)
    # TODO: Currently returns True (stub)
    assert result == True


def test_app_controller_execute_command_signal(qtbot):
    """
    Тест сигнала commandExecuted при выполнении команды
    Test commandExecuted signal when executing command
    """
    from app.main import AppController
    
    controller = AppController()
    
    command_name = "test_command"
    
    # Проверяем что сигнал испускается
    # Check that signal is emitted
    with qtbot.waitSignal(controller.commandExecuted, timeout=1000):
        controller.executeCommand(command_name)


def test_app_controller_get_available_commands():
    """
    Тест метода getAvailableCommands
    Test getAvailableCommands method
    """
    from app.main import AppController
    
    controller = AppController()
    
    commands = controller.getAvailableCommands()
    
    # Проверяем что возвращается список
    # Check that list is returned
    assert isinstance(commands, list)
    
    # Проверяем что есть команды (демо данные)
    # Check that there are commands (demo data)
    assert len(commands) > 0
    
    # Проверяем структуру команды
    # Check command structure
    first_command = commands[0]
    assert "name" in first_command
    assert "gesture" in first_command


@pytest.mark.parametrize("gesture_label", [
    "swipe_right",
    "thumbs_up",
    "palm",
    "peace_sign",
])
def test_app_controller_gesture_training_various_labels(gesture_label):
    """
    Тест обучения с различными названиями жестов
    Test training with various gesture labels
    """
    from app.main import AppController
    
    controller = AppController()
    controller.startGestureTraining(gesture_label)
    
    assert gesture_label in controller.status

