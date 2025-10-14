# -*- coding: utf-8 -*-
"""
Модуль базы данных: PostgreSQL Embedded с SQLAlchemy ORM.
Использует pg_embed для упаковки PostgreSQL внутри приложения.

Таблицы:
- commands: пользовательские команды
- gestures: записанные жесты и привязка к моделям
- settings: настройки приложения
- gesture_history: история использования жестов для статистики
"""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
    Boolean,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    Session,
    relationship,
)
from sqlalchemy.pool import StaticPool

# pg_embed для упаковки PostgreSQL (будет использоваться в production)
try:
    from pg_embed import pg_embed
    PG_EMBED_AVAILABLE = True
except ImportError:
    PG_EMBED_AVAILABLE = False
    print("[WARN] pg_embed not installed - using SQLite fallback for development")


# SQLAlchemy Base для моделей
Base = declarative_base()


# ============================================================================
# Модели данных (ORM)
# ============================================================================

class Command(Base):
    """
    Модель команды: действие, привязанное к жесту.
    
    Поля:
    - id: уникальный идентификатор
    - name: название команды (например, "Открыть браузер")
    - description: описание действия
    - platform: целевая платформа ("macos", "windows", "all")
    - script_path: путь к скрипту/приложению для выполнения
    - gesture_id: привязка к жесту (FK на Gesture)
    - is_active: включена ли команда
    - created_at: дата создания
    """
    __tablename__ = "commands"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    platform = Column(String(50), nullable=False, default="all")  # "macos", "windows", "all"
    script_path = Column(Text, nullable=True)  # путь к приложению/скрипту
    gesture_id = Column(Integer, ForeignKey("gestures.id"), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Связь с жестом (один к одному)
    gesture = relationship("Gesture", back_populates="command", uselist=False)
    # Связь с историей выполнений
    history = relationship("GestureHistory", back_populates="command")

    def __repr__(self):
        return f"<Command(id={self.id}, name='{self.name}', platform='{self.platform}')>"


class Gesture(Base):
    """
    Модель жеста: обученный жест пользователя.
    
    Поля:
    - id: уникальный идентификатор
    - label: метка жеста (например, "zoom", "swipe_left")
    - samples_path: путь к семплам NPY для переобучения
    - model_class_id: индекс класса в обученной модели
    - accuracy: точность распознавания (из валидации)
    - is_two_hands: использует ли жест две руки
    - created_at: дата создания
    """
    __tablename__ = "gestures"

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String(255), nullable=False, unique=True)
    samples_path = Column(Text, nullable=True)  # путь к data/gestures/<label>/
    model_class_id = Column(Integer, nullable=True)  # индекс в models/classes.json
    accuracy = Column(Float, nullable=True)  # точность распознавания (0.0-1.0)
    is_two_hands = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Связь с командой
    command = relationship("Command", back_populates="gesture", uselist=False)
    # Связь с историей выполнений
    history = relationship("GestureHistory", back_populates="gesture")

    def __repr__(self):
        return f"<Gesture(id={self.id}, label='{self.label}', accuracy={self.accuracy})>"


class Settings(Base):
    """
    Модель настроек приложения (key-value хранилище).
    
    Поля:
    - key: ключ настройки (уникальный)
    - value: значение (JSON-совместимый текст)
    """
    __tablename__ = "settings"

    key = Column(String(255), primary_key=True)
    value = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Settings(key='{self.key}', value='{self.value}')>"


class GestureHistory(Base):
    """
    Модель истории использования жестов для статистики.
    
    Поля:
    - id: уникальный идентификатор
    - gesture_id: ссылка на жест (FK на Gesture)
    - command_id: ссылка на команду (FK на Command)
    - executed_at: дата/время выполнения
    """
    __tablename__ = "gesture_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    gesture_id = Column(Integer, ForeignKey("gestures.id"), nullable=False)
    command_id = Column(Integer, ForeignKey("commands.id"), nullable=False)
    executed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Связи
    gesture = relationship("Gesture", back_populates="history")
    command = relationship("Command", back_populates="history")

    def __repr__(self):
        return f"<GestureHistory(id={self.id}, gesture_id={self.gesture_id}, command_id={self.command_id})>"


# ============================================================================
# Конфигурация базы данных
# ============================================================================

class DatabaseManager:
    """
    Менеджер базы данных: инициализация, создание сессий, управление pg_embed.
    """

    def __init__(self, data_dir: Optional[Path] = None, use_pg_embed: bool = True):
        """
        Инициализация менеджера БД.
        
        Аргументы:
        - data_dir: директория для хранения данных PostgreSQL (по умолчанию .db_data/)
        - use_pg_embed: использовать ли pg_embed (если False, используется SQLite для разработки)
        """
        self.data_dir = data_dir or Path(".db_data")
        self.use_pg_embed = use_pg_embed and PG_EMBED_AVAILABLE
        self.pg_embed_instance: Optional[pg_embed] = None
        self.engine = None
        self.SessionLocal = None

        # Инициализация БД
        self._init_database()

    def _init_database(self):
        """Инициализация PostgreSQL embedded или SQLite fallback."""
        if self.use_pg_embed:
            try:
                self._init_pg_embed()
            except Exception as e:
                print(f"[ERROR] Failed to initialize pg_embed: {e}")
                print("[INFO] Falling back to SQLite")
                self._init_sqlite()
        else:
            self._init_sqlite()

        # Создание таблиц (если их нет)
        Base.metadata.create_all(bind=self.engine)
        print(f"[OK] Database initialized: {self.engine.url}")

    def _init_pg_embed(self):
        """Инициализация PostgreSQL Embedded через pg_embed."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Настройка pg_embed
        self.pg_embed_instance = pg_embed.PgEmbed(
            pg_ctl_settings={
                "data_dir": str(self.data_dir / "pgdata"),
                "log_dir": str(self.data_dir / "logs"),
            }
        )

        # Запуск PostgreSQL сервера
        self.pg_embed_instance.start()
        print("[OK] PostgreSQL Embedded started")

        # Создание подключения к БД
        # pg_embed создаёт БД с именем "test" по умолчанию
        pg_url = self.pg_embed_instance.get_dsn(database="dplm_db")
        
        # Создание engine
        self.engine = create_engine(
            pg_url,
            pool_pre_ping=True,  # проверка соединения перед использованием
            echo=False,  # логирование SQL-запросов (для отладки: echo=True)
        )

        # Создание фабрики сессий
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def _init_sqlite(self):
        """Инициализация SQLite fallback (для разработки без pg_embed)."""
        db_path = self.data_dir / "dplm_dev.sqlite"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        sqlite_url = f"sqlite:///{db_path}"
        print(f"[INFO] Using SQLite: {sqlite_url}")

        # Создание engine с StaticPool для SQLite
        self.engine = create_engine(
            sqlite_url,
            connect_args={"check_same_thread": False},  # для многопоточности
            poolclass=StaticPool,
            echo=False,
        )

        # Создание фабрики сессий
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def get_session(self) -> Session:
        """Получить новую сессию БД."""
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()

    def close(self):
        """Корректное закрытие БД и остановка pg_embed (если используется)."""
        if self.engine:
            self.engine.dispose()
            print("[INFO] Database engine disposed")

        if self.pg_embed_instance:
            self.pg_embed_instance.stop()
            print("[INFO] PostgreSQL Embedded stopped")


# ============================================================================
# Глобальный экземпляр менеджера БД (singleton)
# ============================================================================

# Используется по всему приложению для доступа к БД
db_manager: Optional[DatabaseManager] = None


def init_database(data_dir: Optional[Path] = None, use_pg_embed: bool = True) -> DatabaseManager:
    """
    Инициализация глобального менеджера БД.
    
    Вызывается один раз при старте приложения.
    
    Аргументы:
    - data_dir: директория для данных БД
    - use_pg_embed: использовать PostgreSQL Embedded (если False, используется SQLite)
    
    Возвращает:
    - DatabaseManager экземпляр
    """
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager(data_dir=data_dir, use_pg_embed=use_pg_embed)
    return db_manager


def get_db_session() -> Session:
    """
    Получить сессию БД (convenience функция).
    
    Использование:
    >>> session = get_db_session()
    >>> commands = session.query(Command).all()
    >>> session.close()
    """
    if db_manager is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return db_manager.get_session()


# ============================================================================
# Пример использования (для тестирования)
# ============================================================================

if __name__ == "__main__":
    # Инициализация БД (SQLite для разработки)
    db = init_database(use_pg_embed=False)

    # Создание тестовых данных
    session = get_db_session()
    try:
        # Создание жеста
        gesture = Gesture(
            label="test_zoom",
            samples_path="data/gestures/test_zoom",
            model_class_id=0,
            accuracy=0.95,
            is_two_hands=True,
        )
        session.add(gesture)
        session.commit()
        print(f"[OK] Created gesture: {gesture}")

        # Создание команды
        command = Command(
            name="Open Browser",
            description="Opens default web browser",
            platform="all",
            script_path="/usr/bin/open",
            gesture_id=gesture.id,
        )
        session.add(command)
        session.commit()
        print(f"[OK] Created command: {command}")

        # Запрос данных
        all_commands = session.query(Command).all()
        print(f"[INFO] Total commands: {len(all_commands)}")
        for cmd in all_commands:
            print(f"  - {cmd}")

    finally:
        session.close()

    # Закрытие БД
    db.close()
    print("[OK] Database closed")

