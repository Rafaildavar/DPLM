from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _launcher_module():
    path = Path("packaging/macos/gesturebind_launcher.py").resolve()
    spec = importlib.util.spec_from_file_location("gesturebind_launcher_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_launcher_applies_embedded_telemetry_destination(monkeypatch, tmp_path):
    monkeypatch.delenv("DPLM_TELEMETRY_ENDPOINT", raising=False)
    monkeypatch.delenv("DPLM_TELEMETRY_PROJECT_KEY", raising=False)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "release_telemetry.json").write_text(
        json.dumps(
            {
                "endpoint": "https://telemetry.example.test/v1/telemetry/daily",
                "project_key": "public-key",
            }
        ),
        encoding="utf-8",
    )

    module = _launcher_module()
    module._apply_release_telemetry_config(tmp_path)

    assert (
        module.os.environ["DPLM_TELEMETRY_ENDPOINT"]
        == "https://telemetry.example.test/v1/telemetry/daily"
    )
    assert module.os.environ["DPLM_TELEMETRY_PROJECT_KEY"] == "public-key"
