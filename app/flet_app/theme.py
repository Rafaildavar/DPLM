"""
Цветовая палитра и общие стили для Flet-версии GestureBind.

Тёмная desktop-палитра: графитовые поверхности, тёплый текст,
приглушённый cyan-акцент и семантические состояния.
"""
from __future__ import annotations

import flet as ft


# Graphite classic palette. Keep the interface dark, but not blue-heavy.
COLOR_BG_TOP = "#111315"
COLOR_BG_BOTTOM = "#17191D"
COLOR_SURFACE = "#202328"
COLOR_SURFACE_HIGH = "#2B2F36"
COLOR_ACCENT = "#52C7D8"
COLOR_ACCENT_DEEP = "#2D8D9C"
COLOR_ON_SURFACE = "#F0EEE7"
COLOR_MUTED = "#A5A8A1"
COLOR_DANGER = "#E65A4F"
COLOR_SUCCESS = "#67B36B"
COLOR_WARNING = "#D8A84E"


def build_theme() -> ft.Theme:
    """Тёмная Material 3 тема с приглушённым cyan-акцентом."""
    return ft.Theme(
        color_scheme_seed=ft.Colors.CYAN,
        use_material3=True,
        font_family="SF Pro Display",
    )


def app_background() -> ft.LinearGradient:
    """Сдержанный фон главного контейнера."""
    return ft.LinearGradient(
        begin=ft.Alignment.TOP_CENTER,
        end=ft.Alignment.BOTTOM_CENTER,
        colors=[COLOR_BG_TOP, COLOR_BG_BOTTOM],
    )


def _border_all(color: str, width: float = 1) -> ft.Border:
    side = ft.BorderSide(width, color)
    return ft.Border(top=side, right=side, bottom=side, left=side)


def surface_card(content: ft.Control, *, padding: int = 16, radius: int = 8) -> ft.Container:
    """Карточка-поверхность с классическим graphite border."""
    return ft.Container(
        content=content,
        padding=padding,
        bgcolor=COLOR_SURFACE,
        border_radius=radius,
        border=_border_all(COLOR_SURFACE_HIGH),
    )
