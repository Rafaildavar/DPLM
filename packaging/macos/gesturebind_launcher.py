"""PyInstaller entrypoint for the macOS GestureBind bundle."""
from __future__ import annotations

import os
import json
import subprocess
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


def _is_running_from_disk_image(executable: Path | None = None) -> bool:
    path = (executable or Path(sys.executable)).resolve()
    parts = path.parts
    return len(parts) > 2 and parts[0] == "/" and parts[1] == "Volumes"


def _show_installation_required() -> None:
    script = (
        'display dialog "GestureBind запущен из установочного образа. '
        'Перетащите GestureBind в папку Applications, извлеките DMG и '
        'запустите установленное приложение." '
        'with title "Установите GestureBind" buttons {"OK"} '
        'default button "OK" with icon caution'
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _configure_bundled_flet_view(runtime_root: Path) -> None:
    view_dir = runtime_root / "gesturebind_flet"
    if view_dir.is_dir() and any(view_dir.glob("*.app")):
        os.environ["FLET_VIEW_PATH"] = str(view_dir)


def main() -> None:
    if _is_running_from_disk_image():
        _show_installation_required()
        return

    runtime_root = _runtime_root()
    os.chdir(runtime_root)
    _apply_release_telemetry_config(runtime_root)
    _configure_bundled_flet_view(runtime_root)
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
