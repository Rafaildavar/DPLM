"""PyInstaller entrypoint for the macOS GestureBind bundle."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _runtime_root() -> Path:
    bundled_root = getattr(sys, "_MEIPASS", None)
    if bundled_root:
        return Path(bundled_root)
    return Path(__file__).resolve().parents[2]


def main() -> None:
    os.chdir(_runtime_root())
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
