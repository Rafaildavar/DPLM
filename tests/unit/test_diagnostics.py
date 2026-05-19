import json

import pytest

from app.services.app_config import AppConfig, ConfigStore, ENV_OVERRIDES
from app.services.diagnostics import STATUS_FAIL, STATUS_PASS, run_diagnostics


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for key in ENV_OVERRIDES:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def reset_db():
    from app.models.database import reset_database_manager

    reset_database_manager()
    yield
    reset_database_manager()


def _healthy_config(tmp_path) -> AppConfig:
    data_dir = tmp_path / "data" / "gestures"
    label_dir = data_dir / "palm"
    models_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    label_dir.mkdir(parents=True)
    models_dir.mkdir()
    log_dir.mkdir()

    (label_dir / "sample_0000.npy").write_bytes(b"test")
    (models_dir / "knn.pkl").write_bytes(b"model")
    (models_dir / "classes.json").write_text(json.dumps(["palm"]), encoding="utf-8")
    (models_dir / "feature_dim.txt").write_text("42", encoding="utf-8")

    config = AppConfig()
    config.database.backend = "sqlite"
    config.database.sqlite_path = str(tmp_path / "dplm.sqlite")
    config.paths.data_dir = str(data_dir)
    config.paths.models_dir = str(models_dir)
    config.paths.model_path = str(models_dir / "knn.pkl")
    config.paths.classes_path = str(models_dir / "classes.json")
    config.paths.feature_dim_path = str(models_dir / "feature_dim.txt")
    config.paths.log_dir = str(log_dir)
    return config


def test_run_diagnostics_healthy_sqlite_config(tmp_path):
    store = ConfigStore(tmp_path / "config.json")
    config = _healthy_config(tmp_path)
    store.save(config)

    report = run_diagnostics(
        config=config,
        config_store=store,
        check_database=True,
        check_camera=False,
        load_dotenv=False,
    )

    assert report["ok"] is True
    assert report["summary"][STATUS_FAIL] == 0
    assert report["summary"][STATUS_PASS] > 0
    labels = {item["key"]: item for item in report["items"]}
    assert labels["database"]["status"] == STATUS_PASS
    assert labels["dataset_samples"]["status"] == STATUS_PASS
    assert labels["camera_open"]["status"] == "info"


def test_run_diagnostics_reports_missing_model(tmp_path):
    store = ConfigStore(tmp_path / "config.json")
    config = _healthy_config(tmp_path)
    missing_model = tmp_path / "models" / "missing.pkl"
    config.paths.model_path = str(missing_model)
    store.save(config)

    report = run_diagnostics(
        config=config,
        config_store=store,
        check_database=False,
        check_camera=False,
        load_dotenv=False,
    )

    assert report["ok"] is False
    assert any(
        item["key"] == "model_path" and item["status"] == STATUS_FAIL
        for item in report["items"]
    )
