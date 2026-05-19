"""
Экран «Голосовой помощник» — Старт/Стоп + ввод текстовой команды.

Аналог QML ``VoiceAssistantPanel.qml``.
"""
from __future__ import annotations

import flet as ft

from app.flet_app.controller import AppController
from app.flet_app.theme import (
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_MUTED,
    COLOR_ON_SURFACE,
    COLOR_SUCCESS,
    COLOR_SURFACE_HIGH,
    surface_card,
)


class VoiceView:
    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller

        self._status_chip = ft.Container(
            width=56,
            height=56,
            border_radius=28,
            bgcolor=COLOR_SUCCESS if controller.is_voice_assistant_active else "#607d8b",
            alignment=ft.Alignment.CENTER,
            content=ft.Icon(
                ft.Icons.MIC if controller.is_voice_assistant_active else ft.Icons.MIC_OFF,
                color=ft.Colors.WHITE,
                size=28,
            ),
        )
        self._status_title = ft.Text(
            "Голосовой помощник",
            size=16,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_SURFACE,
        )
        self._status_subtitle = ft.Text(
            self._subtitle(),
            color=COLOR_MUTED,
            size=12,
        )
        self._start_btn = ft.FilledButton(
            content=ft.Text("Запустить", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.PLAY_ARROW,
            style=ft.ButtonStyle(
                bgcolor="#00695C",
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=24, vertical=14),
            ),
            on_click=self._on_start,
            disabled=controller.is_voice_assistant_active,
        )
        self._stop_btn = ft.OutlinedButton(
            content=ft.Text("Остановить"),
            icon=ft.Icons.STOP,
            on_click=self._on_stop,
            disabled=not controller.is_voice_assistant_active,
        )
        self._command_input = ft.TextField(
            hint_text="Например: привет, команды, статус, открыть браузер",
            border_color=COLOR_SURFACE_HIGH,
        )
        self._send_btn = ft.FilledButton(
            content=ft.Text("Отправить текстовую команду", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.SEND,
            style=ft.ButtonStyle(bgcolor=COLOR_ACCENT, color=ft.Colors.WHITE),
            on_click=self._on_send,
        )
        self._response_text = ft.Text(
            "Ответ появится здесь",
            color="#E1F5FE",
            size=13,
        )

        controller.voice_assistant_state_changed.connect(self._on_state)

    def on_show(self) -> None:
        pass

    def on_hide(self) -> None:
        pass

    def _subtitle(self) -> str:
        if self._controller.is_voice_assistant_active:
            return "Активен: можно говорить или отправить текстовую команду"
        return "Не активен. Нажмите «Запустить» — wake-word: «ассистент»."

    def _apply_state(self) -> None:
        active = self._controller.is_voice_assistant_active
        self._status_chip.bgcolor = COLOR_SUCCESS if active else "#607d8b"
        self._status_chip.content = ft.Icon(
            ft.Icons.MIC if active else ft.Icons.MIC_OFF,
            color=ft.Colors.WHITE,
            size=28,
        )
        self._status_subtitle.value = self._subtitle()
        self._start_btn.disabled = active
        self._stop_btn.disabled = not active
        try:
            self._page.update()
        except Exception:
            pass

    def _on_state(self, _v: str) -> None:
        self._page.run_thread(self._apply_state)

    def _on_start(self, _e) -> None:
        ok = self._controller.start_voice_assistant(
            language="ru", wake_word=True, tts=True
        )
        if not ok:
            self._response_text.value = (
                "Не удалось запустить голосового помощника "
                "(см. логи: возможно, отсутствует Vosk-модель или микрофон)."
            )
            self._response_text.color = COLOR_DANGER
            try:
                self._response_text.update()
            except Exception:
                pass
        else:
            self._apply_state()

    def _on_stop(self, _e) -> None:
        self._controller.stop_voice_assistant()
        self._apply_state()

    def _on_send(self, _e) -> None:
        text = (self._command_input.value or "").strip()
        if not text:
            return
        resp = self._controller.process_voice_command(text)
        self._response_text.value = f"Ответ: {resp}"
        self._response_text.color = "#E1F5FE"
        try:
            self._response_text.update()
        except Exception:
            pass

    def build(self) -> ft.Control:
        return ft.Column(
            spacing=14,
            expand=True,
            scroll=ft.ScrollMode.AUTO,
            controls=[
                surface_card(
                    ft.Row(
                        spacing=14,
                        controls=[
                            self._status_chip,
                            ft.Column(
                                spacing=4,
                                expand=True,
                                controls=[self._status_title, self._status_subtitle],
                            ),
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    padding=16,
                    radius=16,
                ),
                surface_card(
                    ft.Row(
                        spacing=12,
                        controls=[self._start_btn, self._stop_btn],
                    ),
                    padding=14,
                    radius=16,
                ),
                surface_card(
                    ft.Column(
                        spacing=10,
                        controls=[
                            ft.Text(
                                "Текстовая команда",
                                size=14,
                                weight=ft.FontWeight.W_600,
                                color=COLOR_ON_SURFACE,
                            ),
                            self._command_input,
                            self._send_btn,
                            self._response_text,
                        ],
                    ),
                    padding=16,
                    radius=16,
                ),
            ],
        )
