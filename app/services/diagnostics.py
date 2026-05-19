"""Self-test diagnostics for the Flet desktop assistant."""
from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.services.app_config import (
    AppConfig,
    ConfigStore,
    database_url_from_config,
    resolve_config_path,
)


STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_INFO = "info"


@dataclass
class DiagnosticItem:
    key: str
    label: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def run_diagnostics(
    *,
    config: AppConfig | None = None,
    config_store: ConfigStore | None = None,
    check_database: bool = True,
    check_camera: bool = False,
    load_dotenv: bool = True,
) -> dict[str, Any]:
    """Run a non-destructive self-test and return a JSON-serializable report."""
    store = config_store or ConfigStore()
    effective = config or store.load(include_env=True, load_dotenv=load_dotenv)
    items: list[DiagnosticItem] = []

    def add(key: str, label: str, status: str, detail: str) -> None:
        items.append(DiagnosticItem(key, label, status, detail))

    _check_config_file(store, add)
    _check_env_overrides(store, add, load_dotenv=load_dotenv)
    _check_validation(store, effective, add)
    _check_dependencies(add)
    _check_paths(effective, add)
    _check_model_metadata(effective, add)
    _check_dataset(effective, add)
    _check_log_dir(effective, add)

    if check_database:
        _check_database(effective, add)
    else:
        add("database", "База данных", STATUS_INFO, "Проверка пропущена")

    if check_camera:
        _check_camera(effective, add)
    else:
        add("camera_open", "Камера", STATUS_INFO, "Открытие камеры пропущено")

    counts = {
        STATUS_PASS: 0,
        STATUS_WARN: 0,
        STATUS_FAIL: 0,
        STATUS_INFO: 0,
    }
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1

    return {
        "ok": counts[STATUS_FAIL] == 0,
        "summary": counts,
        "databaseUrl": _mask_database_url(database_url_from_config(effective)),
        "items": [item.to_dict() for item in items],
    }


def _check_config_file(store: ConfigStore, add) -> None:
    if store.path.exists():
        add("config_file", "Конфиг", STATUS_PASS, str(store.path))
    else:
        add("config_file", "Конфиг", STATUS_WARN, f"Файл будет создан: {store.path}")


def _check_env_overrides(store: ConfigStore, add, *, load_dotenv: bool) -> None:
    overrides = store.env_overrides(load_dotenv=load_dotenv)
    if not overrides:
        add("env_overrides", "Env overrides", STATUS_PASS, "Нет активных переопределений")
        return
    pairs = ", ".join(f"{key}->{field}" for key, field in sorted(overrides.items()))
    add("env_overrides", "Env overrides", STATUS_WARN, pairs)


def _check_validation(store: ConfigStore, config: AppConfig, add) -> None:
    result = store.validate(config)
    if result.ok:
        add("config_validation", "Валидация", STATUS_PASS, "Ошибок нет")
    else:
        for idx, error in enumerate(result.errors, start=1):
            add(f"config_error_{idx}", "Валидация", STATUS_FAIL, error)
    for idx, warning in enumerate(result.warnings, start=1):
        add(f"config_warning_{idx}", "Валидация", STATUS_WARN, warning)


def _check_dependencies(add) -> None:
    dependencies = [
        ("flet", "Flet GUI"),
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("sklearn", "scikit-learn"),
        ("joblib", "joblib"),
        ("sqlalchemy", "SQLAlchemy"),
        ("psycopg2", "psycopg2"),
    ]
    for module_name, label in dependencies:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            add(f"dep_{module_name}", label, STATUS_PASS, str(version))
        except Exception as exc:
            add(f"dep_{module_name}", label, STATUS_FAIL, str(exc))


def _check_paths(config: AppConfig, add) -> None:
    path_specs = [
        ("data_dir", "Данные", resolve_config_path(config.paths.data_dir), True),
        ("models_dir", "Каталог моделей", resolve_config_path(config.paths.models_dir), True),
        ("model_path", "Модель", resolve_config_path(config.paths.model_path), False),
        ("classes_path", "classes.json", resolve_config_path(config.paths.classes_path), False),
        (
            "feature_dim_path",
            "feature_dim.txt",
            resolve_config_path(config.paths.feature_dim_path),
            False,
        ),
    ]
    for key, label, path, should_be_dir in path_specs:
        exists = path.exists()
        kind_ok = path.is_dir() if should_be_dir else path.is_file()
        if exists and kind_ok:
            add(key, label, STATUS_PASS, str(path))
        elif exists:
            expected = "каталог" if should_be_dir else "файл"
            add(key, label, STATUS_FAIL, f"Ожидался {expected}: {path}")
        else:
            add(key, label, STATUS_FAIL, f"Не найдено: {path}")


def _check_model_metadata(config: AppConfig, add) -> None:
    classes_path = resolve_config_path(config.paths.classes_path)
    feature_dim_path = resolve_config_path(config.paths.feature_dim_path)

    if classes_path.exists():
        try:
            classes = json.loads(classes_path.read_text(encoding="utf-8"))
            if isinstance(classes, list) and classes:
                add("classes_parse", "Классы", STATUS_PASS, f"{len(classes)} классов")
            else:
                add("classes_parse", "Классы", STATUS_FAIL, "classes.json должен быть непустым списком")
        except Exception as exc:
            add("classes_parse", "Классы", STATUS_FAIL, str(exc))

    if feature_dim_path.exists():
        try:
            feature_dim = int(feature_dim_path.read_text(encoding="utf-8").strip())
            if feature_dim in {42, 84}:
                add("feature_dim_parse", "Размерность", STATUS_PASS, str(feature_dim))
            else:
                add("feature_dim_parse", "Размерность", STATUS_WARN, str(feature_dim))
        except Exception as exc:
            add("feature_dim_parse", "Размерность", STATUS_FAIL, str(exc))


def _check_dataset(config: AppConfig, add) -> None:
    data_dir = resolve_config_path(config.paths.data_dir)
    if not data_dir.exists():
        return
    labels = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    sample_count = sum(len(list(label_dir.glob("sample_*.npy"))) for label_dir in labels)
    if labels and sample_count:
        add(
            "dataset_samples",
            "Датасет",
            STATUS_PASS,
            f"{len(labels)} жестов, {sample_count} sample_*.npy",
        )
    elif labels:
        add("dataset_samples", "Датасет", STATUS_WARN, f"{len(labels)} папок без sample_*.npy")
    else:
        add("dataset_samples", "Датасет", STATUS_WARN, "Папок жестов пока нет")


def _check_log_dir(config: AppConfig, add) -> None:
    log_dir = resolve_config_path(config.paths.log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        probe = log_dir / ".dplm_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        add("log_dir", "Логи", STATUS_PASS, str(log_dir))
    except Exception as exc:
        add("log_dir", "Логи", STATUS_FAIL, f"{log_dir}: {exc}")


def _check_database(config: AppConfig, add) -> None:
    url = database_url_from_config(config)
    try:
        from app.models.database import Base, _append_pg_connect_timeout
        from sqlalchemy import create_engine, text
        from sqlalchemy.engine import make_url
        from sqlalchemy.pool import StaticPool

        url = _append_pg_connect_timeout(url)
        url_obj = make_url(url)
        if url_obj.get_backend_name() == "sqlite":
            db_path = url_obj.database or ""
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(
                url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                future=True,
            )
        else:
            engine = create_engine(url, pool_pre_ping=True, future=True)
        try:
            Base.metadata.create_all(bind=engine)
            with engine.connect() as conn:
                conn.execute(text("select 1"))
                settings_count = conn.execute(text("select count(*) from settings")).scalar()
        finally:
            engine.dispose()
        add("database", "База данных", STATUS_PASS, f"OK, settings={settings_count}")
    except Exception as exc:
        add(
            "database",
            "База данных",
            STATUS_FAIL,
            f"{_mask_database_url(url)}: {exc}",
        )


def _check_camera(config: AppConfig, add) -> None:
    try:
        from app.cv_camera import open_default_capture

        cap = open_default_capture(
            int(config.recognition.camera_index),
            width=640,
            height=480,
            fps=int(config.recognition.target_fps),
        )
        try:
            if cap.isOpened():
                add(
                    "camera_open",
                    "Камера",
                    STATUS_PASS,
                    f"index={config.recognition.camera_index}",
                )
            else:
                add(
                    "camera_open",
                    "Камера",
                    STATUS_FAIL,
                    f"Не открылась: index={config.recognition.camera_index}",
                )
        finally:
            try:
                cap.release()
            except Exception:
                pass
    except Exception as exc:
        add("camera_open", "Камера", STATUS_FAIL, str(exc))


def _mask_database_url(url: str) -> str:
    if "://" not in url or "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.split("@", 1)
    if ":" in creds:
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return f"{scheme}://***@{host}"
