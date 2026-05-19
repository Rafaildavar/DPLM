from pathlib import Path

import pytest

from app.services.app_config import (
    AppConfig,
    ConfigStore,
    ENV_OVERRIDES,
    database_url_from_config,
)


@pytest.fixture(autouse=True)
def clear_dplm_env(monkeypatch):
    for key in ENV_OVERRIDES:
        monkeypatch.delenv(key, raising=False)


def test_config_defaults_load_without_file(tmp_path):
    store = ConfigStore(tmp_path / "config.json")

    config = store.load(include_env=False, load_dotenv=False)

    assert config.database.backend == "sqlite"
    assert config.database.sqlite_path
    assert config.paths.model_path == "models/knn.pkl"
    assert config.recognition.camera_index == 0
    assert config.recognition.target_fps == 30
    assert config.assistant.voice_enabled is False


def test_config_save_and_load_round_trip(tmp_path):
    store = ConfigStore(tmp_path / "config.json")
    config = AppConfig()
    config.database.backend = "postgres"
    config.database.host = "db.local"
    config.database.port = 5433
    config.paths.data_dir = str(tmp_path / "gestures")
    config.recognition.camera_index = 2
    config.recognition.auto_execute_on_gesture = False

    store.save(config)
    loaded = store.load(include_env=False, load_dotenv=False)

    assert loaded.database.backend == "postgres"
    assert loaded.database.host == "db.local"
    assert loaded.database.port == 5433
    assert loaded.paths.data_dir == str(tmp_path / "gestures")
    assert loaded.recognition.camera_index == 2
    assert loaded.recognition.auto_execute_on_gesture is False


def test_env_overrides_config_values(tmp_path, monkeypatch):
    store = ConfigStore(tmp_path / "config.json")
    config = AppConfig()
    config.database.backend = "sqlite"
    config.database.host = "from-config"
    store.save(config)

    monkeypatch.setenv("DPLM_DB_BACKEND", "postgres")
    monkeypatch.setenv("DPLM_DB_HOST", "from-env")
    monkeypatch.setenv("DPLM_DB_PORT", "5544")
    monkeypatch.setenv("DPLM_CAMERA_INDEX", "3")

    loaded = store.load(include_env=True, load_dotenv=False)

    assert loaded.database.backend == "postgres"
    assert loaded.database.host == "from-env"
    assert loaded.database.port == 5544
    assert loaded.recognition.camera_index == 3
    assert store.env_overrides(load_dotenv=False)["DPLM_DB_HOST"] == "database.host"


def test_validation_rejects_invalid_database_and_camera_fields(tmp_path):
    store = ConfigStore(tmp_path / "config.json")
    config = AppConfig()
    config.database.backend = "postgres"
    config.database.port = 99999
    config.database.host = ""
    config.recognition.camera_index = -1
    config.recognition.target_fps = 0

    result = store.validate(config)

    assert not result.ok
    assert any("database.host" in item for item in result.errors)
    assert any("database.port" in item for item in result.errors)
    assert any("camera_index" in item for item in result.errors)
    assert any("target_fps" in item for item in result.errors)


def test_database_url_generation_from_config(tmp_path):
    config = AppConfig()
    config.database.backend = "postgres"
    config.database.host = "db.example"
    config.database.port = 5433
    config.database.user = "user@host"
    config.database.password = "p&w=1"
    config.database.name = "dplm"

    url = database_url_from_config(config)

    assert url == (
        "postgresql+psycopg2://user%40host:p%26w%3D1@db.example:5433/dplm"
    )

    sqlite_config = AppConfig()
    sqlite_config.database.sqlite_path = str(tmp_path / "dplm.sqlite")
    assert database_url_from_config(sqlite_config) == f"sqlite:///{tmp_path / 'dplm.sqlite'}"


def test_database_resolver_uses_config_when_env_absent(monkeypatch, tmp_path):
    import app.models.database as db
    from app.services import app_config

    config = AppConfig()
    config.database.backend = "sqlite"
    config.database.sqlite_path = str(tmp_path / "local.sqlite")

    monkeypatch.setattr(db, "_load_dotenv_if_present", lambda: None)
    monkeypatch.setattr(
        app_config,
        "load_app_config",
        lambda **_kwargs: config,
    )

    assert db.resolve_database_url() == f"sqlite:///{Path(tmp_path / 'local.sqlite')}"
