"""Local technical configuration for the DPLM desktop app.

Domain data stays in the database. This module only owns machine-level
settings: database connection, local paths, camera/runtime flags, and logs.
The load order is:

1. built-in defaults
2. ``~/.dplm/config.json``
3. environment overrides
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote_plus


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path.home() / ".dplm" / "config.json"


def _default_sqlite_path() -> str:
    return str(Path.home() / ".dplm" / "dplm.sqlite")


def _default_log_dir() -> str:
    return str(Path.home() / ".dplm" / "logs")


@dataclass
class DatabaseConfig:
    backend: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    user: str = "dplm"
    password: str = "dplm"
    name: str = "dplm"
    sqlite_path: str = field(default_factory=_default_sqlite_path)
    url: str = ""


@dataclass
class PathConfig:
    data_dir: str = "data/gestures"
    models_dir: str = "models"
    model_path: str = "models/knn.pkl"
    classes_path: str = "models/classes.json"
    feature_dim_path: str = "models/feature_dim.txt"
    log_dir: str = field(default_factory=_default_log_dir)


@dataclass
class RecognitionConfig:
    camera_index: int = 0
    target_fps: int = 30
    two_hands_mode: bool = False
    auto_execute_on_gesture: bool = True
    auto_start_recognition: bool = False
    pointer_smoothing: float = 0.55


@dataclass
class AssistantConfig:
    voice_enabled: bool = False


@dataclass
class AppConfig:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    assistant: AssistantConfig = field(default_factory=AssistantConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "AppConfig":
        base = cls()
        if not isinstance(raw, dict):
            return base

        return cls(
            database=_merge_dataclass(DatabaseConfig, base.database, raw.get("database")),
            paths=_merge_dataclass(PathConfig, base.paths, raw.get("paths")),
            recognition=_merge_dataclass(
                RecognitionConfig, base.recognition, raw.get("recognition")
            ),
            assistant=_merge_dataclass(AssistantConfig, base.assistant, raw.get("assistant")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _merge_dataclass(cls: type, defaults: Any, raw: Any) -> Any:
    data = asdict(defaults)
    if isinstance(raw, dict):
        data.update({k: v for k, v in raw.items() if k in data})
    out: dict[str, Any] = {}
    for key, default_value in asdict(defaults).items():
        value = data.get(key, default_value)
        if isinstance(default_value, bool):
            out[key] = _coerce_bool(value, bool(default_value))
        elif isinstance(default_value, int):
            out[key] = _coerce_int(value, int(default_value))
        elif isinstance(default_value, float):
            out[key] = _coerce_float(value, float(default_value))
        else:
            out[key] = "" if value is None else str(value)
    return cls(**out)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return default


EnvSetter = Callable[[AppConfig, str], None]


def _set_db_url(config: AppConfig, value: str) -> None:
    config.database.url = value.strip()


def _set_db_backend(config: AppConfig, value: str) -> None:
    config.database.backend = value.strip().lower()


def _set_db_host(config: AppConfig, value: str) -> None:
    config.database.host = value.strip()


def _set_db_port(config: AppConfig, value: str) -> None:
    config.database.port = _coerce_int(value, config.database.port)


def _set_db_user(config: AppConfig, value: str) -> None:
    config.database.user = value.strip()


def _set_db_password(config: AppConfig, value: str) -> None:
    config.database.password = value


def _set_db_name(config: AppConfig, value: str) -> None:
    config.database.name = value.strip()


def _set_sqlite_path(config: AppConfig, value: str) -> None:
    config.database.sqlite_path = value.strip()


def _set_data_dir(config: AppConfig, value: str) -> None:
    config.paths.data_dir = value.strip()


def _set_models_dir(config: AppConfig, value: str) -> None:
    config.paths.models_dir = value.strip()


def _set_model_path(config: AppConfig, value: str) -> None:
    config.paths.model_path = value.strip()


def _set_classes_path(config: AppConfig, value: str) -> None:
    config.paths.classes_path = value.strip()


def _set_feature_dim_path(config: AppConfig, value: str) -> None:
    config.paths.feature_dim_path = value.strip()


def _set_log_dir(config: AppConfig, value: str) -> None:
    config.paths.log_dir = value.strip()


def _set_camera_index(config: AppConfig, value: str) -> None:
    config.recognition.camera_index = _coerce_int(value, config.recognition.camera_index)


def _set_target_fps(config: AppConfig, value: str) -> None:
    config.recognition.target_fps = _coerce_int(value, config.recognition.target_fps)


def _set_two_hands(config: AppConfig, value: str) -> None:
    config.recognition.two_hands_mode = _coerce_bool(
        value, config.recognition.two_hands_mode
    )


def _set_auto_execute(config: AppConfig, value: str) -> None:
    config.recognition.auto_execute_on_gesture = _coerce_bool(
        value, config.recognition.auto_execute_on_gesture
    )


def _set_auto_start(config: AppConfig, value: str) -> None:
    config.recognition.auto_start_recognition = _coerce_bool(
        value, config.recognition.auto_start_recognition
    )


def _set_pointer_smoothing(config: AppConfig, value: str) -> None:
    config.recognition.pointer_smoothing = _coerce_float(
        value, config.recognition.pointer_smoothing
    )


ENV_OVERRIDES: dict[str, tuple[str, EnvSetter]] = {
    "DATABASE_URL": ("database.url", _set_db_url),
    "DPLM_DB_BACKEND": ("database.backend", _set_db_backend),
    "DPLM_DB_HOST": ("database.host", _set_db_host),
    "DPLM_DB_PORT": ("database.port", _set_db_port),
    "DPLM_DB_USER": ("database.user", _set_db_user),
    "DPLM_DB_PASSWORD": ("database.password", _set_db_password),
    "DPLM_DB_NAME": ("database.name", _set_db_name),
    "DPLM_SQLITE_PATH": ("database.sqlite_path", _set_sqlite_path),
    "DPLM_DATA_DIR": ("paths.data_dir", _set_data_dir),
    "DPLM_MODELS_DIR": ("paths.models_dir", _set_models_dir),
    "DPLM_MODEL_PATH": ("paths.model_path", _set_model_path),
    "DPLM_CLASSES_PATH": ("paths.classes_path", _set_classes_path),
    "DPLM_FEATURE_DIM_PATH": ("paths.feature_dim_path", _set_feature_dim_path),
    "DPLM_LOG_DIR": ("paths.log_dir", _set_log_dir),
    "DPLM_CAMERA_INDEX": ("recognition.camera_index", _set_camera_index),
    "DPLM_TARGET_FPS": ("recognition.target_fps", _set_target_fps),
    "DPLM_TWO_HANDS_MODE": ("recognition.two_hands_mode", _set_two_hands),
    "DPLM_AUTO_EXECUTE": ("recognition.auto_execute_on_gesture", _set_auto_execute),
    "DPLM_AUTO_START_RECOGNITION": (
        "recognition.auto_start_recognition",
        _set_auto_start,
    ),
    "DPLM_POINTER_SMOOTHING": (
        "recognition.pointer_smoothing",
        _set_pointer_smoothing,
    ),
    "DPLM_POINTER_SHARPNESS": (
        "recognition.pointer_smoothing",
        _set_pointer_smoothing,
    ),
}


def load_project_dotenv(*, missing_only: bool = True) -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, value = s.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if missing_only and key in os.environ:
            continue
        os.environ[key] = value


class ConfigStore:
    def __init__(self, path: Path | str = DEFAULT_CONFIG_PATH):
        self.path = Path(path).expanduser()

    def load(
        self,
        *,
        include_env: bool = True,
        load_dotenv: bool = True,
    ) -> AppConfig:
        if load_dotenv:
            load_project_dotenv()
        config = AppConfig()
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {}
            config = AppConfig.from_dict(raw)
        if include_env:
            self._apply_env_overrides(config)
        return config

    def ensure_exists(self) -> None:
        if self.path.exists():
            return
        self.save(self.load(include_env=False))

    def save(self, config: AppConfig) -> None:
        result = self.validate(config)
        if not result.ok:
            raise ValueError("; ".join(result.errors))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def validate(self, config: AppConfig) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        db = config.database
        paths = config.paths
        rec = config.recognition

        backend = (db.backend or "").strip().lower()
        if db.url.strip():
            if "://" not in db.url:
                errors.append("DATABASE_URL должен быть SQLAlchemy URL")
        elif backend not in {"postgres", "postgresql", "sqlite", "sqlite3"}:
            errors.append("database.backend должен быть postgres или sqlite")

        if backend in {"postgres", "postgresql"} and not db.url.strip():
            if not db.host.strip():
                errors.append("database.host не может быть пустым для PostgreSQL")
            if not db.user.strip():
                errors.append("database.user не может быть пустым для PostgreSQL")
            if not db.name.strip():
                errors.append("database.name не может быть пустым для PostgreSQL")
            if db.port < 1 or db.port > 65535:
                errors.append("database.port должен быть в диапазоне 1..65535")

        if backend in {"sqlite", "sqlite3"} and not db.url.strip():
            if not db.sqlite_path.strip():
                errors.append("database.sqlite_path не может быть пустым")

        required_paths = {
            "paths.data_dir": paths.data_dir,
            "paths.models_dir": paths.models_dir,
            "paths.model_path": paths.model_path,
            "paths.classes_path": paths.classes_path,
            "paths.feature_dim_path": paths.feature_dim_path,
            "paths.log_dir": paths.log_dir,
        }
        for name, value in required_paths.items():
            if not str(value or "").strip():
                errors.append(f"{name} не может быть пустым")

        if rec.camera_index < 0:
            errors.append("recognition.camera_index должен быть >= 0")
        if rec.target_fps < 1 or rec.target_fps > 120:
            errors.append("recognition.target_fps должен быть в диапазоне 1..120")
        if rec.pointer_smoothing < 0.05 or rec.pointer_smoothing > 0.95:
            errors.append("recognition.pointer_smoothing должен быть в диапазоне 0.05..0.95")

        model_path = resolve_config_path(paths.model_path)
        classes_path = resolve_config_path(paths.classes_path)
        if not model_path.exists():
            warnings.append(f"Модель не найдена: {model_path}")
        if not classes_path.exists():
            warnings.append(f"classes.json не найден: {classes_path}")

        return ValidationResult(errors=errors, warnings=warnings)

    def env_overrides(self, *, load_dotenv: bool = True) -> dict[str, str]:
        if load_dotenv:
            load_project_dotenv()
        out: dict[str, str] = {}
        for key, (field_name, _) in ENV_OVERRIDES.items():
            if os.environ.get(key, "").strip():
                out[key] = field_name
        return out

    def _apply_env_overrides(self, config: AppConfig) -> None:
        for key, (_, setter) in ENV_OVERRIDES.items():
            value = os.environ.get(key)
            if value is not None and value.strip():
                setter(config, value)


def resolve_config_path(value: str | Path, *, base_dir: Path = PROJECT_ROOT) -> Path:
    text = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def database_url_from_config(config: AppConfig, *, prefer_sqlite: bool = False) -> str:
    db = config.database
    backend = (db.backend or "").strip().lower()
    if prefer_sqlite:
        backend = "sqlite"

    if db.url.strip() and not prefer_sqlite:
        return db.url.strip()

    if backend in {"sqlite", "sqlite3"}:
        sqlite_path = resolve_config_path(db.sqlite_path)
        return f"sqlite:///{sqlite_path}"

    user = quote_plus(db.user)
    password = quote_plus(db.password)
    host = db.host.strip() or "localhost"
    port = int(db.port or 5432)
    name = db.name.strip() or "dplm"
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


def load_app_config(
    *,
    include_env: bool = True,
    load_dotenv: bool = True,
    path: Path | str = DEFAULT_CONFIG_PATH,
) -> AppConfig:
    return ConfigStore(path).load(include_env=include_env, load_dotenv=load_dotenv)
