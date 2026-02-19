#!/usr/bin/env python3
"""Lightweight bootstrap launcher for DPLM.

Позволяет запускать проект даже в "пустом" окружении: показывает недостающие зависимости
и даёт понятную команду установки, вместо падения ImportError при импорте PySide6.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REQUIRED_MODULES = [
    "PySide6",
    "numpy",
    "cv2",
    "mediapipe",
    "sklearn",
    "sqlalchemy",
]

PIP_NAMES = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
}


def _missing_modules(modules: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    return missing


def _print_help(missing: list[str]) -> None:
    print("[!] Окружение неполное: отсутствуют зависимости для GUI/CV запуска.")
    print("[i] Missing modules:", ", ".join(missing))
    pkgs = [PIP_NAMES.get(m, m) for m in missing]
    print("[i] Install command:")
    print("    pip install " + " ".join(pkgs))
    print("[i] Или установите полный набор:")
    print("    pip install -r requirements.txt")


def _run_env_check() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_env.py"
    if script.exists():
        subprocess.run([sys.executable, str(script)], check=False)


def main() -> int:
    missing = _missing_modules(REQUIRED_MODULES)
    if missing:
        _print_help(missing)
        _run_env_check()
        return 0

    from app.main import main as gui_main

    return int(gui_main())


if __name__ == "__main__":
    raise SystemExit(main())
