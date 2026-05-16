"""
Цветовая палитра и общие стили для Flet-версии DPLM.

Повторяет визуальную айдентику исходного QML-интерфейса
(тёмная Material 3 тема с акцентом cyan + градиенты #1a1a2e → #16213e).
"""
from __future__ import annotations

import flet as ft


# Базовая палитра — отражает QML Material.Cyan на тёмном фоне.
COLOR_BG_TOP = "#1a1a2e"
COLOR_BG_BOTTOM = "#16213e"
COLOR_SURFACE = "#2a2a3e"
COLOR_SURFACE_HIGH = "#363650"
COLOR_ACCENT = "#00bcd4"   # Material Cyan 500
COLOR_ACCENT_DEEP = "#0097a7"
COLOR_ON_SURFACE = "#ECEFF1"
COLOR_MUTED = "#90A4AE"
COLOR_DANGER = "#f44336"
COLOR_SUCCESS = "#4CAF50"


def build_theme() -> ft.Theme:
    """Тёмная Material 3 тема с акцентом cyan."""
    return ft.Theme(
        color_scheme_seed=ft.Colors.CYAN,
        use_material3=True,
        font_family="SF Pro Display",
    )


def app_background() -> ft.LinearGradient:
    """Градиентный фон главного контейнера (как в QML MainWindow)."""
    return ft.LinearGradient(
        begin=ft.Alignment.TOP_CENTER,
        end=ft.Alignment.BOTTOM_CENTER,
        colors=[COLOR_BG_TOP, COLOR_BG_BOTTOM],
    )


def surface_card(content: ft.Control, *, padding: int = 16, radius: int = 16) -> ft.Container:
    """Карточка-поверхность с округлёнными углами (стиль `#2a2a3e` + radius)."""
    return ft.Container(
        content=content,
        padding=padding,
        bgcolor=COLOR_SURFACE,
        border_radius=radius,
    )
