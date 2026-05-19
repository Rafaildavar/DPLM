"""
Экран «Настройки» — параметры распознавания и привязок.

Главные настройки — это политика R4/R5/R6 (порог уверенности, cooldown,
предупреждение о двуручных жестах). Они хранятся в БД (таблица ``settings``)
и сразу же подхватываются ``BindingPolicy`` через ``reload_policy``.

Аналог QML ``SystemSettingsScreen.qml`` + ``SettingsPanel.qml`` (упрощённо,
без полей, которые не были реализованы в бэкенде QML версии).
"""
from __future__ import annotations

import flet as ft

from app.flet_app.controller import AppController
from app.flet_app.theme import (
    COLOR_ACCENT,
    COLOR_MUTED,
    COLOR_ON_SURFACE,
    COLOR_SUCCESS,
    COLOR_SURFACE_HIGH,
    surface_card,
)


class SettingsView:
    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller

        policy = controller.get_binding_policy()
        self._threshold_slider = ft.Slider(
            value=float(policy["confidenceThreshold"]),
            min=0.30,
            max=0.99,
            divisions=69,
            active_color=COLOR_ACCENT,
            inactive_color=COLOR_SURFACE_HIGH,
            on_change=self._on_threshold_change,
        )
        self._threshold_label = ft.Text(
            self._threshold_text(), size=14, color=COLOR_ON_SURFACE
        )

        self._cooldown_slider = ft.Slider(
            value=float(policy["cooldownMs"]),
            min=200,
            max=5000,
            divisions=48,
            active_color=COLOR_ACCENT,
            inactive_color=COLOR_SURFACE_HIGH,
            on_change=self._on_cooldown_change,
        )
        self._cooldown_label = ft.Text(
            self._cooldown_text(), size=14, color=COLOR_ON_SURFACE
        )

        self._warn_switch = ft.Switch(
            value=bool(policy["warnTwoHands"]),
            label="Предупреждать о привязке опасных действий к одноручным жестам (R6)",
            active_color=COLOR_ACCENT,
        )

        self._save_btn = ft.FilledButton(
            content=ft.Text("Сохранить настройки", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.SAVE,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=24, vertical=14),
            ),
            on_click=self._on_save,
        )
        self._status_text = ft.Text("", color=COLOR_SUCCESS, size=13, visible=False)

    def on_show(self) -> None:
        # Перечитать из БД на случай, если меняли в другом месте.
        policy = self._controller.get_binding_policy()
        self._threshold_slider.value = float(policy["confidenceThreshold"])
        self._cooldown_slider.value = float(policy["cooldownMs"])
        self._warn_switch.value = bool(policy["warnTwoHands"])
        self._threshold_label.value = self._threshold_text()
        self._cooldown_label.value = self._cooldown_text()
        try:
            self._page.update()
        except Exception:
            pass

    def on_hide(self) -> None:
        pass

    def _threshold_text(self) -> str:
        v = float(self._threshold_slider.value) if hasattr(self, "_threshold_slider") else 0.65
        return f"Порог уверенности (R4): {int(round(v * 100))}%"

    def _cooldown_text(self) -> str:
        v = int(self._cooldown_slider.value) if hasattr(self, "_cooldown_slider") else 1500
        return f"Cooldown между срабатываниями (R5): {v} мс"

    def _on_threshold_change(self, _e) -> None:
        self._threshold_label.value = self._threshold_text()
        try:
            self._threshold_label.update()
        except Exception:
            pass

    def _on_cooldown_change(self, _e) -> None:
        self._cooldown_label.value = self._cooldown_text()
        try:
            self._cooldown_label.update()
        except Exception:
            pass

    def _on_save(self, _e) -> None:
        ok = self._controller.set_binding_policy(
            confidence_threshold=float(self._threshold_slider.value),
            cooldown_ms=int(self._cooldown_slider.value),
            warn_two_hands=bool(self._warn_switch.value),
        )
        self._status_text.value = (
            "Сохранено и применено к политике распознавания"
            if ok
            else "Не удалось сохранить — БД недоступна"
        )
        self._status_text.color = COLOR_SUCCESS if ok else "#f44336"
        self._status_text.visible = True
        try:
            self._status_text.update()
        except Exception:
            pass

    def build(self) -> ft.Control:
        header = ft.Row(
            controls=[
                ft.Icon(ft.Icons.SETTINGS, color=COLOR_ACCENT, size=28),
                ft.Text(
                    "Настройки",
                    size=20,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_ON_SURFACE,
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        )
        return ft.Column(
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            controls=[
                surface_card(header, padding=16, radius=16),
                surface_card(
                    ft.Column(
                        spacing=14,
                        controls=[
                            ft.Text(
                                "Политика распознавания",
                                size=14,
                                weight=ft.FontWeight.W_600,
                                color=COLOR_ON_SURFACE,
                            ),
                            self._threshold_label,
                            self._threshold_slider,
                            ft.Text(
                                "Жесты с уверенностью ниже порога не вызывают команду.",
                                size=11,
                                color=COLOR_MUTED,
                            ),
                            ft.Divider(color=COLOR_SURFACE_HIGH, thickness=1),
                            self._cooldown_label,
                            self._cooldown_slider,
                            ft.Text(
                                "Минимальный интервал между двумя выполнениями одного жеста.",
                                size=11,
                                color=COLOR_MUTED,
                            ),
                            ft.Divider(color=COLOR_SURFACE_HIGH, thickness=1),
                            self._warn_switch,
                            ft.Container(height=4),
                            self._save_btn,
                            self._status_text,
                        ],
                    ),
                    padding=20,
                    radius=16,
                ),
            ],
        )
