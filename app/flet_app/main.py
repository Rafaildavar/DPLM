#!/usr/bin/env python3
"""
Точка входа Flet-версии GestureBind.

Запуск:
    python -m app.flet_app.main

Flet поставляет Flutter runtime внутри ``flet_desktop``. Это единственная
desktop-точка входа релизного приложения.
"""
from __future__ import annotations

import atexit
import sys

import flet as ft

from app.flet_app.controller import AppController
from app.flet_app.theme import COLOR_BG_TOP, build_theme
from app.flet_app.views.shell import build_shell


def _configure_page(page: ft.Page) -> None:
    page.title = "GestureBind — Gesture & Voice Assistant"
    page.padding = 0
    page.bgcolor = COLOR_BG_TOP
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = build_theme()

    page.window.width = 1040
    page.window.height = 760
    page.window.min_width = 880
    page.window.min_height = 640


def main(page: ft.Page) -> None:
    _configure_page(page)

    controller = AppController()
    # В Flet 0.85 ``page.on_close`` / ``page.window.prevent_close`` нестабильны
    # (часто срабатывают прямо при старте), поэтому регистрируем shutdown как
    # process-level хук — он отработает при выходе питон-процесса.
    atexit.register(controller.shutdown)

    page.add(build_shell(page, controller))
    if controller.auto_start_recognition:
        page.run_thread(controller.toggle_recognition)
    print("[✓] Flet GestureBind запущен")


if __name__ == "__main__":
    # ``ft.AppView.FLET_APP`` — нативное окно Flutter, без браузера.
    ft.run(main, view=ft.AppView.FLET_APP)
    sys.exit(0)
