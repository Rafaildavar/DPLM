"""
Юнит-тесты для модуля базы данных
Unit tests for database module

Тесты SQLAlchemy моделей и DatabaseManager
Tests for SQLAlchemy models and DatabaseManager
"""
import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def test_database_models_import():
    """
    Тест импорта моделей БД
    Test database models import
    """
    from app.models.database import Base, Command, Gesture, Settings, GestureHistory
    
    assert Base is not None
    assert Command is not None
    assert Gesture is not None
    assert Settings is not None
    assert GestureHistory is not None


def test_database_manager_import():
    """
    Тест импорта DatabaseManager
    Test DatabaseManager import
    """
    from app.models.database import DatabaseManager
    
    assert DatabaseManager is not None


def test_create_in_memory_database():
    """
    Тест создания БД в памяти (SQLite)
    Test creating in-memory database (SQLite)
    """
    from app.models.database import Base
    
    # Создаём БД в памяти / Create in-memory DB
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    # Проверяем что таблицы созданы / Check tables created
    tables = Base.metadata.tables.keys()
    assert "commands" in tables
    assert "gestures" in tables
    assert "settings" in tables
    assert "gesture_history" in tables


def test_command_model_creation():
    """
    Тест создания записи в таблице commands
    Test creating record in commands table
    """
    from app.models.database import Base, Command
    
    # Создаём БД в памяти / Create in-memory DB
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    # Создаём сессию / Create session
    session = Session(engine)
    
    # Создаём команду / Create command
    command = Command(
        name="Test Command",
        description="Test description",
        platform="macOS",
        script_path="/path/to/script.sh",
        gesture_id=None
    )
    
    session.add(command)
    session.commit()
    
    # Проверяем что команда создана / Check command created
    assert command.id is not None
    assert command.name == "Test Command"
    assert command.platform == "macOS"
    
    # Проверяем что можем загрузить из БД / Check can load from DB
    loaded_command = session.query(Command).filter_by(name="Test Command").first()
    assert loaded_command is not None
    assert loaded_command.name == "Test Command"
    
    session.close()


def test_gesture_model_creation():
    """
    Тест создания записи в таблице gestures
    Test creating record in gestures table
    """
    from app.models.database import Base, Gesture
    
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    
    # Создаём жест / Create gesture
    gesture = Gesture(
        label="swipe_right",
        samples_path="/path/to/samples",
        model_class_id=0,
        accuracy=0.95
    )
    
    session.add(gesture)
    session.commit()
    
    # Проверки / Checks
    assert gesture.id is not None
    assert gesture.label == "swipe_right"
    assert gesture.accuracy == 0.95
    
    # Загрузка из БД / Load from DB
    loaded_gesture = session.query(Gesture).filter_by(label="swipe_right").first()
    assert loaded_gesture is not None
    assert loaded_gesture.accuracy == 0.95
    
    session.close()


def test_setting_model_creation():
    """
    Тест создания записи в таблице settings
    Test creating record in settings table
    """
    from app.models.database import Base, Settings
    
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    
    # Создаём настройку / Create setting
    setting = Settings(
        key="camera_index",
        value="0"
    )
    
    session.add(setting)
    session.commit()
    
    # Проверки / Checks
    assert setting.key == "camera_index"
    assert setting.value == "0"
    
    # Загрузка из БД / Load from DB
    loaded_setting = session.query(Settings).filter_by(key="camera_index").first()
    assert loaded_setting is not None
    assert loaded_setting.value == "0"
    
    session.close()


def test_gesture_history_model_creation():
    """
    Тест создания записи в таблице gesture_history
    Test creating record in gesture_history table
    """
    from app.models.database import Base, Gesture, Command, GestureHistory
    from datetime import datetime
    
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    
    # Создаём жест и команду / Create gesture and command
    gesture = Gesture(label="test_gesture", samples_path="/path", model_class_id=0)
    command = Command(name="test_command", platform="macOS")
    
    session.add(gesture)
    session.add(command)
    session.commit()
    
    # Создаём запись истории / Create history record
    history = GestureHistory(
        gesture_id=gesture.id,
        command_id=command.id
    )
    
    session.add(history)
    session.commit()
    
    # Проверки / Checks
    assert history.id is not None
    assert history.gesture_id == gesture.id
    assert history.command_id == command.id
    assert history.executed_at is not None
    assert isinstance(history.executed_at, datetime)
    
    session.close()


def test_command_gesture_relationship():
    """
    Тест связи между Command и Gesture
    Test relationship between Command and Gesture
    """
    from app.models.database import Base, Command, Gesture
    
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    
    # Создаём жест / Create gesture
    gesture = Gesture(label="test_gesture", samples_path="/path", model_class_id=0)
    session.add(gesture)
    session.commit()
    
    # Создаём команду с привязкой к жесту / Create command linked to gesture
    command = Command(
        name="test_command",
        platform="macOS",
        gesture_id=gesture.id
    )
    session.add(command)
    session.commit()
    
    # Проверяем связь / Check relationship
    assert command.gesture_id == gesture.id
    
    # Загружаем и проверяем связь через relationship
    # Load and check relationship via SQLAlchemy relationship
    loaded_command = session.query(Command).filter_by(name="test_command").first()
    assert loaded_command.gesture_id == gesture.id
    
    session.close()


@pytest.mark.parametrize("platform", ["macOS", "Windows", "All"])
def test_command_various_platforms(platform):
    """
    Тест команд с различными платформами
    Test commands with various platforms
    """
    from app.models.database import Base, Command
    
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    
    command = Command(
        name=f"Command for {platform}",
        platform=platform
    )
    session.add(command)
    session.commit()
    
    assert command.platform == platform
    
    session.close()


def test_multiple_settings():
    """
    Тест хранения нескольких настроек
    Test storing multiple settings
    """
    from app.models.database import Base, Settings
    
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    
    settings_data = [
        ("camera_index", "0"),
        ("tts_enabled", "true"),
        ("tts_language", "ru"),
        ("smoothing_window", "30"),
    ]
    
    # Создаём настройки / Create settings
    for key, value in settings_data:
        setting = Settings(key=key, value=value)
        session.add(setting)
    
    session.commit()
    
    # Проверяем что все созданы / Check all created
    all_settings = session.query(Settings).all()
    assert len(all_settings) == len(settings_data)
    
    # Проверяем загрузку конкретной настройки / Check loading specific setting
    camera_setting = session.query(Settings).filter_by(key="camera_index").first()
    assert camera_setting.value == "0"
    
    session.close()

