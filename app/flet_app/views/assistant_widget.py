"""Compact assistant widget view for the Flet app."""
from __future__ import annotations

import flet as ft

from app.flet_app.controller import AppController
from app.flet_app.theme import (
    COLOR_ACCENT,
    COLOR_ACCENT_DEEP,
    COLOR_DANGER,
    COLOR_MUTED,
    COLOR_ON_SURFACE,
    COLOR_SUCCESS,
    COLOR_SURFACE_HIGH,
    COLOR_WARNING,
    surface_card,
)


class AssistantWidgetView:
    """A compact, always-visible style control surface for background use."""

    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller
        self._gesture_state_phase = "idle"

        self._status_text = ft.Text(controller.status, size=13, color=COLOR_MUTED)
        self._mode_text = ft.Text(self._mode_text_value(), size=12, color=COLOR_MUTED)
        self._gesture_text = ft.Text(
            "—",
            size=34,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_SURFACE,
            no_wrap=True,
        )
        self._confidence_text = ft.Text("0%", size=13, color=COLOR_MUTED)
        self._gesture_state_text = ft.Text(
            "ожидание",
            size=12,
            color=COLOR_MUTED,
            no_wrap=True,
        )
        self._confidence_bar = ft.ProgressBar(
            value=0.0,
            color=COLOR_ACCENT,
            bgcolor=COLOR_SURFACE_HIGH,
        )
        self._command_text = ft.Text(
            "—",
            size=15,
            color=COLOR_ON_SURFACE,
            weight=ft.FontWeight.W_600,
            no_wrap=True,
        )
        self._activity_list = ft.Column(
            spacing=6,
            scroll=ft.ScrollMode.AUTO,
            controls=[ft.Text("Пока нет событий", size=12, color=COLOR_MUTED)],
        )

        self._toggle_btn = ft.FilledButton(
            content=ft.Text(self._toggle_label(), weight=ft.FontWeight.BOLD),
            icon=self._toggle_icon(),
            style=self._toggle_style(),
            on_click=lambda _e: self._toggle_recognition(),
        )
        self._compact_btn = ft.OutlinedButton(
            content=ft.Text("Компактное окно"),
            icon=ft.Icons.SMART_TOY,
            on_click=lambda _e: self._apply_compact_window(),
        )
        self._full_btn = ft.OutlinedButton(
            content=ft.Text("Полное окно"),
            icon=ft.Icons.HOME,
            on_click=lambda _e: self._apply_full_window(),
        )

        controller.status_changed.connect(self._on_status)
        controller.recognizing_changed.connect(self._on_recognizing)
        controller.gesture_detected.connect(self._on_gesture)
        controller.confidence_changed.connect(self._on_confidence)
        controller.gesture_state_changed.connect(self._on_gesture_state)
        controller.command_executed.connect(self._on_command)
        controller.recognition_event_recorded.connect(self._on_activity_changed)

    def on_show(self) -> None:
        self._refresh_activity()

    def on_hide(self) -> None:
        pass

    def _mode_text_value(self) -> str:
        if self._controller.is_recognizing:
            return "Фоновое распознавание активно"
        return "Ассистент ожидает запуска"

    def _toggle_label(self) -> str:
        return "Остановить" if self._controller.is_recognizing else "Запустить"

    def _toggle_icon(self) -> str:
        return ft.Icons.STOP if self._controller.is_recognizing else ft.Icons.PLAY_ARROW

    def _toggle_style(self) -> ft.ButtonStyle:
        return ft.ButtonStyle(
            bgcolor=COLOR_DANGER if self._controller.is_recognizing else COLOR_SUCCESS,
            color=ft.Colors.WHITE,
            padding=ft.Padding.symmetric(horizontal=22, vertical=14),
        )

    def _toggle_recognition(self) -> None:
        runner = getattr(self._page, "run_thread", None)
        if callable(runner):
            runner(self._controller.toggle_recognition)
        else:
            self._controller.toggle_recognition()

    def _apply_compact_window(self) -> None:
        page = self._page
        if page is None:
            return
        try:
            page.window.width = 430
            page.window.height = 620
            page.window.min_width = 390
            page.window.min_height = 540
            page.window.always_on_top = True
            page.window.center()
            page.update()
        except Exception as e:
            print(f"[w] compact window failed: {e}")

    def _apply_full_window(self) -> None:
        page = self._page
        if page is None:
            return
        try:
            page.window.always_on_top = False
            page.window.width = 1040
            page.window.height = 760
            page.window.min_width = 880
            page.window.min_height = 640
            page.window.center()
            page.update()
        except Exception as e:
            print(f"[w] full window failed: {e}")

    def _on_status(self, value: str) -> None:
        self._run_ui(self._apply_status, value)

    def _apply_status(self, value: str) -> None:
        self._status_text.value = value
        self._safe_update(self._status_text)

    def _on_recognizing(self, _value: bool) -> None:
        self._run_ui(self._apply_recognizing)

    def _apply_recognizing(self) -> None:
        self._mode_text.value = self._mode_text_value()
        self._toggle_btn.content = ft.Text(self._toggle_label(), weight=ft.FontWeight.BOLD)
        self._toggle_btn.icon = self._toggle_icon()
        self._toggle_btn.style = self._toggle_style()
        self._safe_update(self._mode_text)
        self._safe_update(self._toggle_btn)

    def _on_gesture(self, label: str) -> None:
        self._run_ui(self._apply_gesture, label)

    def _apply_gesture(self, label: str) -> None:
        self._gesture_text.value = label or "—"
        self._safe_update(self._gesture_text)

    def _on_confidence(self, value: float) -> None:
        self._run_ui(self._apply_confidence, value)

    def _apply_confidence(self, value: float) -> None:
        safe_value = max(0.0, min(1.0, float(value or 0.0)))
        if self._gesture_state_phase != "idle" and safe_value <= 0.0:
            return
        self._confidence_bar.value = safe_value
        self._confidence_bar.color = COLOR_ACCENT
        self._confidence_text.value = f"{int(round(safe_value * 100))}%"
        self._safe_update(self._confidence_bar)
        self._safe_update(self._confidence_text)

    def _on_gesture_state(self, payload: dict) -> None:
        self._run_ui(self._apply_gesture_state, payload)

    def _apply_gesture_state(self, payload: dict) -> None:
        data = payload if isinstance(payload, dict) else {}
        phase = str(data.get("phase") or "idle")
        self._gesture_state_phase = phase
        label = str(data.get("label") or "").strip()
        reason = str(data.get("reason") or "").strip()
        confidence = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
        progress = max(0.0, min(1.0, float(data.get("progress") or 0.0)))
        frames = int(data.get("frames") or 0)
        required = int(data.get("requiredFrames") or 0)

        if phase == "pending":
            self._gesture_text.value = label or "—"
            self._confidence_bar.value = progress
            self._confidence_bar.color = COLOR_WARNING
            self._confidence_text.value = f"{frames}/{max(1, required)}"
            self._gesture_state_text.value = f"{int(round(confidence * 100))}% · подтверждение"
            self._gesture_state_text.color = COLOR_WARNING
        elif phase == "confirmed":
            self._confidence_bar.value = confidence
            self._confidence_bar.color = COLOR_SUCCESS
            self._confidence_text.value = f"{int(round(confidence * 100))}%"
            self._gesture_state_text.value = "готово"
            self._gesture_state_text.color = COLOR_SUCCESS
        elif phase == "rejected":
            self._gesture_text.value = label or "—"
            self._confidence_bar.value = 1.0
            self._confidence_bar.color = COLOR_DANGER
            self._confidence_text.value = f"{int(round(confidence * 100))}%"
            self._gesture_state_text.value = reason or "отклонено"
            self._gesture_state_text.color = COLOR_DANGER
        elif phase == "suppressed":
            self._gesture_text.value = label or "—"
            self._confidence_bar.value = 1.0
            self._confidence_bar.color = COLOR_WARNING
            self._confidence_text.value = f"{int(round(confidence * 100))}%"
            self._gesture_state_text.value = reason or "подавлено"
            self._gesture_state_text.color = COLOR_WARNING
        elif phase == "cooldown":
            self._gesture_text.value = label or "—"
            self._confidence_bar.value = 1.0
            self._confidence_bar.color = COLOR_WARNING
            self._confidence_text.value = "cooldown"
            self._gesture_state_text.value = reason or "cooldown"
            self._gesture_state_text.color = COLOR_WARNING
        else:
            self._confidence_bar.value = 0.0
            self._confidence_bar.color = COLOR_ACCENT
            self._confidence_text.value = "0%"
            self._gesture_state_text.value = "ожидание"
            self._gesture_state_text.color = COLOR_MUTED

        self._safe_update(self._gesture_text)
        self._safe_update(self._confidence_bar)
        self._safe_update(self._confidence_text)
        self._safe_update(self._gesture_state_text)

    def _on_command(self, name: str) -> None:
        self._run_ui(self._apply_command, name)

    def _apply_command(self, name: str) -> None:
        self._command_text.value = name or "—"
        self._safe_update(self._command_text)

    def _on_activity_changed(self) -> None:
        self._run_ui(self._refresh_activity)

    def _refresh_activity(self) -> None:
        rows = self._controller.get_recent_recognition_events(limit=5)
        if not rows:
            self._activity_list.controls = [
                ft.Text("Пока нет событий", size=12, color=COLOR_MUTED)
            ]
        else:
            self._activity_list.controls = [self._activity_row(row) for row in rows]
        self._safe_update(self._activity_list)

    def _activity_row(self, row: dict) -> ft.Control:
        executed = bool(row.get("executed"))
        label = str(row.get("label") or "—")
        confidence = int(round(float(row.get("confidence") or 0.0) * 100))
        detected_at = str(row.get("detectedAt") or "")
        command = str(row.get("commandName") or "")
        detail = f"{confidence}%"
        if executed and command:
            detail = f"{detail} · {command}"
        elif not executed:
            detail = f"{detail} · без команды"
        return ft.Row(
            controls=[
                ft.Icon(
                    ft.Icons.CHECK_CIRCLE if executed else ft.Icons.INFO_OUTLINE,
                    color=COLOR_SUCCESS if executed else COLOR_MUTED,
                    size=16,
                ),
                ft.Column(
                    spacing=0,
                    expand=True,
                    controls=[
                        ft.Text(
                            label,
                            size=12,
                            color=COLOR_ON_SURFACE,
                            weight=ft.FontWeight.W_600,
                            no_wrap=True,
                        ),
                        ft.Text(detail, size=11, color=COLOR_MUTED, no_wrap=True),
                    ],
                ),
                ft.Text(detected_at, size=11, color=COLOR_MUTED),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _run_ui(self, fn, *args) -> None:
        if self._page is not None:
            try:
                self._page.run_thread(fn, *args)
                return
            except Exception:
                pass
        fn(*args)

    def _safe_update(self, control: ft.Control) -> None:
        try:
            control.update()
        except Exception:
            pass

    def build(self) -> ft.Control:
        avatar = ft.Container(
            width=184,
            height=184,
            border_radius=92,
            gradient=ft.LinearGradient(
                begin=ft.Alignment.TOP_LEFT,
                end=ft.Alignment.BOTTOM_RIGHT,
                colors=[COLOR_ACCENT, COLOR_ACCENT_DEEP, "#5c6bc0"],
            ),
            border=ft.Border.all(1, "#80deea"),
            alignment=ft.Alignment.CENTER,
            content=ft.Column(
                spacing=4,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.SMART_TOY, color=ft.Colors.WHITE, size=58),
                    ft.Text(
                        "GestureBind",
                        size=16,
                        color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD,
                    ),
                    ft.Text(
                        "assistant",
                        size=11,
                        color="#e0f7fa",
                    ),
                ],
            ),
        )

        left = ft.Column(
            spacing=14,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                avatar,
                self._mode_text,
                self._toggle_btn,
                ft.Row(
                    controls=[self._compact_btn, self._full_btn],
                    wrap=True,
                    spacing=8,
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
            ],
        )

        right = ft.Column(
            spacing=14,
            expand=True,
            controls=[
                surface_card(
                    ft.Column(
                        spacing=10,
                        controls=[
                            ft.Row(
                                controls=[
                                    ft.Icon(ft.Icons.GESTURE, color=COLOR_ACCENT, size=20),
                                    ft.Text("Жест", color=COLOR_MUTED, size=13),
                                    self._confidence_text,
                                ],
                                spacing=8,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            self._gesture_state_text,
                            self._gesture_text,
                            self._confidence_bar,
                        ],
                    ),
                    padding=16,
                    radius=16,
                ),
                surface_card(
                    ft.Column(
                        spacing=8,
                        controls=[
                            ft.Row(
                                controls=[
                                    ft.Icon(ft.Icons.TERMINAL, color=COLOR_SUCCESS, size=18),
                                    ft.Text("Команда", color=COLOR_MUTED, size=13),
                                ],
                                spacing=8,
                            ),
                            self._command_text,
                        ],
                    ),
                    padding=16,
                    radius=16,
                ),
                surface_card(
                    ft.Column(
                        spacing=8,
                        controls=[
                            ft.Row(
                                controls=[
                                    ft.Icon(ft.Icons.HISTORY, color=COLOR_ACCENT, size=18),
                                    ft.Text("Последние события", color=COLOR_MUTED, size=13),
                                ],
                                spacing=8,
                            ),
                            ft.Container(content=self._activity_list, height=160),
                        ],
                    ),
                    padding=16,
                    radius=16,
                ),
            ],
        )

        return ft.Column(
            spacing=14,
            expand=True,
            controls=[
                surface_card(
                    ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.SMART_TOY, color=COLOR_ACCENT, size=24),
                            ft.Column(
                                spacing=2,
                                controls=[
                                    ft.Text(
                                        "Ассистент-виджет",
                                        size=18,
                                        weight=ft.FontWeight.BOLD,
                                        color=COLOR_ON_SURFACE,
                                    ),
                                    self._status_text,
                                ],
                            ),
                        ],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    padding=16,
                    radius=16,
                ),
                ft.ResponsiveRow(
                    spacing=16,
                    run_spacing=16,
                    controls=[
                        ft.Container(content=left, col={"xs": 12, "md": 4}),
                        ft.Container(content=right, col={"xs": 12, "md": 8}),
                    ],
                ),
            ],
        )
