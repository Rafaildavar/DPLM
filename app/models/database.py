# -*- coding: utf-8 -*-
"""
Прикладной слой хранения данных системы распознавания жестов.

База данных в дипломе выполняет роль прикладного хранилища пользовательских
данных и не является «тяжёлым» центральным модулем. Она обеспечивает:

- хранение пользовательского словаря жестов (``gestures``);
- хранение обучающих примеров для каждого жеста (``gesture_samples``);
- хранение привязок жестов к действиям системы (``commands``);
- хранение сведений об обученных моделях (``recognition_models``);
- хранение результатов распознавания и истории работы приложения
  (``recognition_logs``, ``app_sessions``, ``gesture_history``);
- хранение глобальных настроек приложения (``settings``);
- идентификацию пользователей системы (``users``).

Реализация:
- ORM: SQLAlchemy 2.x (``declarative_base``).
- СУБД по умолчанию — PostgreSQL (через ``psycopg2``). Параметры подключения
  читаются из переменных окружения (``DATABASE_URL`` либо
  ``DPLM_DB_HOST/USER/PASSWORD/...``).
- Для разработки и тестов поддерживается резервный режим SQLite (включается
  явно: ``DPLM_DB_BACKEND=sqlite`` или ``DATABASE_URL=sqlite:///...``).

Совместимость: имена и поля существующих таблиц
(``commands``, ``gestures``, ``settings``, ``gesture_history``) сохранены,
чтобы не сломать текущие тесты и сервисы. Новые таблицы и столбцы
добавлены как опциональные.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse, urlunparse
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import (
    Session,
    declarative_base,
    relationship,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool


# ============================================================================
# Базовый класс ORM
# ============================================================================

Base = declarative_base()


# ============================================================================
# Модели данных (ORM)
# ============================================================================


class User(Base):
    """
    Пользователь системы.

    Используется для персонализации словаря жестов и истории действий
    (например, при использовании приложения несколькими людьми на одной
    рабочей станции). Если приложение работает в однопользовательском
    режиме, создаётся служебная запись ``default``.

    Поля:
    - ``id``         — суррогатный ключ;
    - ``username``   — уникальный логин (без зависимостей от ОС);
    - ``display_name`` — отображаемое имя;
    - ``created_at`` — дата создания записи.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), nullable=False, unique=True)
    display_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    gestures = relationship("Gesture", back_populates="user")
    sessions = relationship("AppSession", back_populates="user")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}')>"


class Gesture(Base):
    """
    Запись пользовательского словаря жестов.

    Поля:
    - ``id``             — суррогатный ключ;
    - ``label``          — метка жеста (используется как имя класса в модели);
    - ``description``    — человекочитаемое описание (для UI и отчётов);
    - ``samples_path``   — путь к каталогу с сырыми примерами (``data/gestures/<label>``);
    - ``model_class_id`` — индекс класса в обученной модели (см. ``models/classes.json``);
    - ``accuracy``       — оценка точности по валидации (0.0–1.0);
    - ``is_two_hands``   — признак «жест двумя руками»;
    - ``is_active``      — учитывается ли жест при инференсе;
    - ``user_id``        — владелец словарной записи;
    - ``created_at``     — дата добавления;
    - ``updated_at``     — дата последней правки.
    """

    __tablename__ = "gestures"

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    samples_path = Column(Text, nullable=True)
    model_class_id = Column(Integer, nullable=True)
    accuracy = Column(Float, nullable=True)
    is_two_hands = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    user = relationship("User", back_populates="gestures")
    samples = relationship(
        "GestureSample",
        back_populates="gesture",
        cascade="all, delete-orphan",
    )
    command = relationship("Command", back_populates="gesture", uselist=False)
    history = relationship("GestureHistory", back_populates="gesture")
    logs = relationship("RecognitionLog", back_populates="gesture")

    def __repr__(self) -> str:
        return f"<Gesture(id={self.id}, label='{self.label}', accuracy={self.accuracy})>"


class GestureSample(Base):
    """
    Один обучающий пример жеста (запись «кадра» признаков MediaPipe Hands).

    Большие массивы (NumPy) хранятся на диске; в БД сохраняется метаданные
    и относительный путь к файлу с признаками. По желанию можно хранить
    сериализованный JSON-вектор признаков непосредственно (поле
    ``features_json``) — тогда дисковый файл не обязателен.

    Поля:
    - ``id``            — суррогатный ключ;
    - ``gesture_id``    — внешний ключ на ``gestures.id``;
    - ``sample_index``  — порядковый номер примера в рамках жеста;
    - ``features_path`` — путь к ``.npy``/``.json`` с признаками (опционально);
    - ``features_json`` — сериализованный вектор признаков (опционально);
    - ``frames``        — количество кадров в примере;
    - ``hand_count``    — обнаружено рук в примере (1 или 2);
    - ``source``        — источник примера (``camera``, ``import``);
    - ``created_at``    — дата записи.
    """

    __tablename__ = "gesture_samples"
    __table_args__ = (
        UniqueConstraint("gesture_id", "sample_index", name="uq_gesture_sample_index"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    gesture_id = Column(
        Integer,
        ForeignKey("gestures.id", ondelete="CASCADE"),
        nullable=False,
    )
    sample_index = Column(Integer, nullable=False)
    features_path = Column(Text, nullable=True)
    features_json = Column(Text, nullable=True)
    frames = Column(Integer, nullable=True)
    hand_count = Column(Integer, nullable=True)
    source = Column(String(32), nullable=False, default="camera")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    gesture = relationship("Gesture", back_populates="samples")

    def __repr__(self) -> str:
        return (
            f"<GestureSample(id={self.id}, gesture_id={self.gesture_id}, "
            f"sample_index={self.sample_index})>"
        )


class RecognitionModel(Base):
    """
    Сведения об обученной модели распознавания.

    На каждом цикле дообучения создаётся новая запись; «активная» модель
    помечается флагом ``is_active`` (одновременно активной должна быть одна).

    Поля:
    - ``id``               — суррогатный ключ;
    - ``name``             — отображаемое имя версии (``knn-2026-04-19``);
    - ``algorithm``        — алгоритм (``knn``, ``mlp``, ``svm`` ...);
    - ``model_path``       — путь к сериализованной модели (``models/knn.pkl``);
    - ``classes_path``     — путь к ``classes.json``;
    - ``feature_dim``      — размерность вектора признаков;
    - ``n_classes``        — количество классов в модели;
    - ``n_samples``        — общее количество обучающих примеров;
    - ``accuracy``         — точность на валидации;
    - ``hyperparams_json`` — JSON с гиперпараметрами (k, метрика и т.п.);
    - ``is_active``        — отметка «текущая модель»;
    - ``created_at``       — дата обучения.
    """

    __tablename__ = "recognition_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False, unique=True)
    algorithm = Column(String(32), nullable=False, default="knn")
    model_path = Column(Text, nullable=True)
    classes_path = Column(Text, nullable=True)
    feature_dim = Column(Integer, nullable=True)
    n_classes = Column(Integer, nullable=True)
    n_samples = Column(Integer, nullable=True)
    accuracy = Column(Float, nullable=True)
    hyperparams_json = Column(Text, nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    logs = relationship("RecognitionLog", back_populates="model")

    def __repr__(self) -> str:
        return (
            f"<RecognitionModel(id={self.id}, name='{self.name}', "
            f"algorithm='{self.algorithm}', is_active={self.is_active})>"
        )


class Command(Base):
    """
    Команда: действие, привязанное к жесту.

    Поля:
    - ``id``           — суррогатный ключ;
    - ``name``         — уникальное название команды (для UI и голоса);
    - ``description``  — текст для подсказок;
    - ``platform``     — целевая платформа (``macos``, ``windows``, ``all``);
    - ``script_path``  — путь/спецификация исполняемого ресурса
      (поддерживаются префиксы ``open:`` и ``shell:`` —
      см. ``app.services.gesture_command_bridge``);
    - ``action_spec``  — JSON-описание «действия из UI без кода»
      (схема: ``app.services.user_command_sync.ACTION_SPEC_SCHEMA``);
    - ``gesture_id``   — внешний ключ на ``gestures.id`` (один к одному);
    - ``is_active``    — включена ли команда;
    - ``created_at``   — дата создания.
    """

    __tablename__ = "commands"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    platform = Column(String(50), nullable=False, default="all")
    script_path = Column(Text, nullable=True)
    action_spec = Column(Text, nullable=True)
    gesture_id = Column(Integer, ForeignKey("gestures.id"), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    gesture = relationship("Gesture", back_populates="command", uselist=False)
    history = relationship("GestureHistory", back_populates="command")

    def __repr__(self) -> str:
        return f"<Command(id={self.id}, name='{self.name}', platform='{self.platform}')>"


class Settings(Base):
    """
    Глобальные настройки приложения (key-value).

    Поля:
    - ``key``   — уникальный ключ настройки;
    - ``value`` — значение в виде строки (часто JSON).
    """

    __tablename__ = "settings"

    key = Column(String(255), primary_key=True)
    value = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<Settings(key='{self.key}', value='{self.value}')>"


class GestureHistory(Base):
    """
    История срабатываний пары «жест → команда».

    Используется для отчётов и статистики использования словаря.

    Поля:
    - ``id``          — суррогатный ключ;
    - ``gesture_id``  — внешний ключ на ``gestures.id``;
    - ``command_id``  — внешний ключ на ``commands.id``;
    - ``executed_at`` — момент выполнения команды.
    """

    __tablename__ = "gesture_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    gesture_id = Column(Integer, ForeignKey("gestures.id"), nullable=False)
    command_id = Column(Integer, ForeignKey("commands.id"), nullable=False)
    executed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    gesture = relationship("Gesture", back_populates="history")
    command = relationship("Command", back_populates="history")

    def __repr__(self) -> str:
        return (
            f"<GestureHistory(id={self.id}, gesture_id={self.gesture_id}, "
            f"command_id={self.command_id})>"
        )


class RecognitionLog(Base):
    """
    Журнал распознавания жестов в реальном времени.

    Хранит «сырые» события распознавания (включая жесты, которые не были
    привязаны к команде или были отброшены по порогу уверенности).
    Используется для аналитики качества модели и подсчёта статистики.

    Поля:
    - ``id``           — суррогатный ключ;
    - ``label``        — метка, выданная классификатором;
    - ``confidence``   — уверенность классификатора (0.0–1.0);
    - ``model_id``     — внешний ключ на ``recognition_models.id``;
    - ``gesture_id``   — внешний ключ на ``gestures.id`` (если метка нашлась
      в словаре пользователя; иначе ``NULL``);
    - ``session_id``   — внешний ключ на ``app_sessions.id``;
    - ``executed``     — была ли запущена команда (булево);
    - ``detected_at``  — момент распознавания.
    """

    __tablename__ = "recognition_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=True)
    model_id = Column(Integer, ForeignKey("recognition_models.id"), nullable=True)
    gesture_id = Column(Integer, ForeignKey("gestures.id"), nullable=True)
    session_id = Column(Integer, ForeignKey("app_sessions.id"), nullable=True)
    executed = Column(Boolean, default=False, nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    model = relationship("RecognitionModel", back_populates="logs")
    gesture = relationship("Gesture", back_populates="logs")
    session = relationship("AppSession", back_populates="logs")

    def __repr__(self) -> str:
        return (
            f"<RecognitionLog(id={self.id}, label='{self.label}', "
            f"confidence={self.confidence})>"
        )


class AppSession(Base):
    """
    История запусков приложения.

    Запись создаётся при старте приложения и обновляется при остановке
    (поле ``ended_at``). Используется для отчётов и поиска проблем
    («когда последний раз падало распознавание»).

    Поля:
    - ``id``         — суррогатный ключ;
    - ``user_id``    — текущий пользователь (опционально);
    - ``app_version`` — версия приложения;
    - ``platform``   — ОС (``macos``, ``windows``, ``linux``);
    - ``started_at`` — момент запуска;
    - ``ended_at``   — момент завершения (``NULL`` пока сессия активна);
    - ``notes``      — произвольные пометки/последняя ошибка.
    """

    __tablename__ = "app_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    app_version = Column(String(32), nullable=True)
    platform = Column(String(32), nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    user = relationship("User", back_populates="sessions")
    logs = relationship("RecognitionLog", back_populates="session")

    def __repr__(self) -> str:
        return (
            f"<AppSession(id={self.id}, started_at={self.started_at}, "
            f"ended_at={self.ended_at})>"
        )


# ============================================================================
# Конфигурация подключения
# ============================================================================


def _load_dotenv_if_present() -> None:
    """Минимальный парсер ``.env`` без зависимости от ``python-dotenv``.

    Подгружает только переменные, которых нет в окружении. Игнорирует
    комментарии и некорректные строки. Безопасен на любой ОС.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            key, _, value = s.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        return


def _append_pg_connect_timeout(url: str, seconds: int = 10) -> str:
    """Добавить ``connect_timeout`` в query-string для PostgreSQL (psycopg2).

    Без таймаута клиент при недоступном хосте может ждать десятки секунд
    или дольше (особенно заметно в ``alembic upgrade``).
    """
    parsed = urlparse(url)
    if not parsed.scheme.lower().startswith("postgres"):
        return url
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    keys = {k for k, _ in pairs}
    if "connect_timeout" in keys:
        return url
    pairs = list(pairs) + [("connect_timeout", str(int(seconds)))]
    new_query = urlencode(pairs)
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
    )


def resolve_database_url(prefer_sqlite: bool = False) -> str:
    """Сформировать SQLAlchemy URL подключения к БД.

    Порядок приоритетов:

    1. Явный параметр ``prefer_sqlite=True`` — SQLite-файл из
       ``DPLM_SQLITE_PATH`` (по умолчанию ``.db_data/dplm_dev.sqlite``).
    2. Переменная ``DATABASE_URL`` (любая поддерживаемая SQLAlchemy строка).
    3. ``DPLM_DB_BACKEND=sqlite`` — резервный SQLite.
    4. По умолчанию — PostgreSQL, собранный из переменных
       ``DPLM_DB_HOST/PORT/USER/PASSWORD/NAME`` (со значениями по
       умолчанию ``localhost/5432/dplm/dplm/dplm``).

    Для строк ``postgresql*`` в URL добавляется ``connect_timeout`` (10 с),
    если параметр ещё не задан — чтобы при недоступной БД соединение
    обрывалось предсказуемо, а не «висело» без вывода.

    Эта функция не открывает соединение и пригодна для использования
    в Alembic.
    """
    _load_dotenv_if_present()

    if prefer_sqlite:
        return _sqlite_url_from_env()

    explicit_url = os.environ.get("DATABASE_URL", "").strip()
    if explicit_url:
        return _append_pg_connect_timeout(explicit_url)

    backend = os.environ.get("DPLM_DB_BACKEND", "postgres").strip().lower()
    if backend in ("sqlite", "sqlite3"):
        return _sqlite_url_from_env()

    host = os.environ.get("DPLM_DB_HOST", "localhost")
    port = os.environ.get("DPLM_DB_PORT", "5432")
    user = quote_plus(os.environ.get("DPLM_DB_USER", "dplm"))
    password = quote_plus(os.environ.get("DPLM_DB_PASSWORD", "dplm"))
    name = os.environ.get("DPLM_DB_NAME", "dplm")
    base = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    return _append_pg_connect_timeout(base)


def _sqlite_url_from_env() -> str:
    raw = os.environ.get("DPLM_SQLITE_PATH", "").strip()
    if raw:
        return f"sqlite:///{raw}"
    default_path = Path(".db_data") / "dplm_dev.sqlite"
    return f"sqlite:///{default_path}"


# ============================================================================
# Менеджер БД
# ============================================================================


class DatabaseManager:
    """Менеджер БД: единая точка инициализации engine/Session.

    Параметры:
    - ``url``         — явный SQLAlchemy URL (если ``None`` — берётся из
      окружения через :func:`resolve_database_url`);
    - ``prefer_sqlite`` — принудительно использовать SQLite (для тестов
      и режима разработки без PostgreSQL);
    - ``data_dir``    — директория для SQLite-файла, если он используется;
    - ``echo``        — включить SQL-логирование (для отладки).
    """

    def __init__(
        self,
        url: Optional[str] = None,
        prefer_sqlite: bool = False,
        data_dir: Optional[Path] = None,
        echo: bool = False,
    ):
        self.data_dir = data_dir or Path(".db_data")
        if data_dir is not None and not str(self.data_dir).startswith(":"):
            self.data_dir.mkdir(parents=True, exist_ok=True)

        self.url = url or resolve_database_url(prefer_sqlite=prefer_sqlite)
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._init_engine(echo=echo)
        self._ensure_schema()

    # ---- Инициализация --------------------------------------------------

    def _init_engine(self, *, echo: bool) -> None:
        url_obj = make_url(self.url)
        backend = url_obj.get_backend_name()

        if backend == "sqlite":
            db_path = url_obj.database or ""
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(
                self.url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=echo,
                future=True,
            )
            _enable_sqlite_foreign_keys(self.engine)
        else:
            self.engine = create_engine(
                self.url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                echo=echo,
                future=True,
            )

        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )

    def _ensure_schema(self) -> None:
        """Создать недостающие таблицы.

        В production-режиме рекомендуется управлять схемой через Alembic
        (``alembic upgrade head``), но автосоздание необходимо для
        SQLite-разработки и для первого запуска без миграций.
        """
        assert self.engine is not None
        Base.metadata.create_all(bind=self.engine)
        print(f"[OK] Database initialized: {self.engine.url.render_as_string(hide_password=True)}")

    # ---- Публичные методы -----------------------------------------------

    def get_session(self) -> Session:
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()

    def healthcheck(self) -> bool:
        """Простая проверка доступности БД (``SELECT 1``)."""
        from sqlalchemy import text

        try:
            with self.engine.connect() as conn:  # type: ignore[union-attr]
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:  # pragma: no cover - диагностика
            print(f"[!] DB healthcheck failed: {exc}")
            return False

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
            print("[INFO] Database engine disposed")


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    """SQLite по умолчанию игнорирует ``ON DELETE CASCADE`` — включаем."""

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_conn, _record):  # type: ignore[no-untyped-def]
        cur = dbapi_conn.cursor()
        try:
            cur.execute("PRAGMA foreign_keys=ON")
        finally:
            cur.close()


# ============================================================================
# Глобальный singleton
# ============================================================================

db_manager: Optional[DatabaseManager] = None


def init_database(
    data_dir: Optional[Path] = None,
    use_pg_embed: bool = True,  # сохранено для обратной совместимости
    url: Optional[str] = None,
    prefer_sqlite: bool = False,
    echo: bool = False,
) -> DatabaseManager:
    """Инициализация глобального менеджера БД.

    Параметр ``use_pg_embed`` сохранён ради обратной совместимости с прежним
    API (PostgreSQL Embedded). Если он явно равен ``False``, используется
    SQLite-резерв; в остальных случаях работает обычный PostgreSQL/URL из
    окружения.
    """
    global db_manager
    if db_manager is not None:
        return db_manager

    if use_pg_embed is False and not prefer_sqlite and url is None:
        prefer_sqlite = True

    db_manager = DatabaseManager(
        url=url,
        prefer_sqlite=prefer_sqlite,
        data_dir=data_dir,
        echo=echo,
    )
    return db_manager


def get_db_session() -> Session:
    """Получить сессию глобального менеджера БД."""
    if db_manager is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return db_manager.get_session()


# ============================================================================
# Самопроверка модуля
# ============================================================================

if __name__ == "__main__":
    db = init_database(prefer_sqlite=True)
    session = get_db_session()
    try:
        if not session.query(User).filter_by(username="default").first():
            session.add(User(username="default", display_name="Default user"))
            session.commit()
        print("[OK] users:", session.query(User).count())
        print("[OK] gestures:", session.query(Gesture).count())
        print("[OK] commands:", session.query(Command).count())
    finally:
        session.close()
    db.close()
