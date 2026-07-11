"""PyInstaller entrypoint for the macOS GestureBind bundle."""
from __future__ import annotations

import os
import json
import sys
from pathlib import Path


def _runtime_root() -> Path:
    bundled_root = getattr(sys, "_MEIPASS", None)
    if bundled_root:
        return Path(bundled_root)
    return Path(__file__).resolve().parents[2]


def _apply_release_telemetry_config(runtime_root: Path) -> None:
    path = runtime_root / "configs" / "release_telemetry.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    endpoint = str(payload.get("endpoint") or "").strip()
    project_key = str(payload.get("project_key") or "").strip()
    if endpoint:
        os.environ.setdefault("DPLM_TELEMETRY_ENDPOINT", endpoint)
    if project_key:
        os.environ.setdefault("DPLM_TELEMETRY_PROJECT_KEY", project_key)


def main() -> None:
    runtime_root = _runtime_root()
    os.chdir(runtime_root)
    _apply_release_telemetry_config(runtime_root)
    state_root = Path.home() / ".dplm"
    os.environ.setdefault("DPLM_DB_BACKEND", "sqlite")
    os.environ.setdefault("DPLM_SQLITE_PATH", str(state_root / "dplm.sqlite"))
    os.environ.setdefault("DPLM_DATA_DIR", str(state_root / "data" / "gestures"))
    os.environ.setdefault("DPLM_LOG_DIR", str(state_root / "logs"))
    (state_root / "data" / "gestures").mkdir(parents=True, exist_ok=True)
    (state_root / "logs").mkdir(parents=True, exist_ok=True)

    import flet as ft

    from app.flet_app.main import main as flet_main

    ft.run(flet_main, view=ft.AppView.FLET_APP)


if __name__ == "__main__":
    main()
