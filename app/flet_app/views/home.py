"""
Главный экран DPLM: превью камеры + жесты + последняя команда.

В отличие от старой версии (где главная только показывала список команд, а
распознавание жило на отдельной вкладке/в subprocess), здесь главная — это
полностью **встроенное распознавание**: одна большая кнопка «Старт/Стоп»
включает камеру + MediaPipe + KNN прямо в окне.

При детекции жеста ``AppController`` автоматически вызывает
``execute_for_gesture(label, conf)``, что через ``GestureCommandBridge``
поднимает привязку из БД и выполняет команду ОС (Safari, громкость и т.п.).
"""
from __future__ import annotations

import base64

import flet as ft

from app.flet_app.controller import AppController
from app.flet_app.theme import (
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_MUTED,
    COLOR_ON_SURFACE,
    COLOR_SUCCESS,
    COLOR_SURFACE_HIGH,
)


# 1×1 PNG-плейсхолдер до прихода первого кадра.
_PLACEHOLDER_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "2mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


class HomeView:
    """Главный экран (с состоянием камеры и подписками)."""

    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller

        self._camera_image = ft.Image(
            src=_PLACEHOLDER_DATA_URL,
            fit=ft.BoxFit.CONTAIN,
            gapless_playback=True,
            expand=True,
            visible=False,
        )

        self._gesture_text = ft.Text(
            "—",
            size=32,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_SURFACE,
        )
        self._confidence_text = ft.Text("0%", size=14, color=COLOR_MUTED)
        self._confidence_bar = ft.ProgressBar(
            value=0.0,
            color=COLOR_ACCENT,
            bgcolor=COLOR_SURFACE_HIGH,
        )
        self._command_text = ft.Text(
            "—",
            size=18,
            weight=ft.FontWeight.W_500,
            color=COLOR_ON_SURFACE,
        )
        self._activity_list = ft.Column(
            spacing=6,
            scroll=ft.ScrollMode.AUTO,
            controls=[ft.Text("Пока нет событий", size=12, color=COLOR_MUTED)],
        )

        self._toggle_btn = ft.FilledButton(
            content=ft.Text(self._btn_label(), size=15, weight=ft.FontWeight.BOLD),
            icon=self._btn_icon(),
            style=self._btn_style(),
            on_click=self._on_toggle,
            height=56,
        )

        self._two_hands_switch = ft.Switch(
            value=controller.two_hands_mode,
            label="Режим двух рук",
            active_color=COLOR_ACCENT,
            on_change=lambda e: controller.set_two_hands_mode(
                bool(self._two_hands_switch.value)
            ),
        )
        self._auto_exec_switch = ft.Switch(
            value=controller.auto_execute,
            label="Авто-команды",
            active_color=COLOR_ACCENT,
            on_change=lambda e: controller.set_auto_execute(
                bool(self._auto_exec_switch.value)
            ),
        )
        self._gesture_switch = ft.Switch(
            value=controller.gesture_mode,
            label="Жесты",
            active_color=COLOR_ACCENT,
            on_change=self._on_gesture_mode_toggle,
        )
        self._landmarks_switch = ft.Switch(
            value=controller.show_landmark_overlay,
            label="Точки руки",
            active_color=COLOR_ACCENT,
            on_change=self._on_landmarks_toggle,
        )
        self._pointer_switch = ft.Switch(
            value=controller.pointer_mode,
            label="Курсор",
            active_color=COLOR_ACCENT,
            on_change=self._on_pointer_toggle,
        )

        self._status_text = ft.Text(
            controller.status, size=12, color=COLOR_MUTED, italic=True
        )

        # Подписки. Все события могут прийти из фонового потока, поэтому
        # все обновления UI идут через ``page.run_thread``.
        controller.camera_frame_updated.connect(self._on_frame)
        controller.gesture_detected.connect(self._on_gesture)
        controller.command_executed.connect(self._on_command)
        controller.recognition_event_recorded.connect(self._on_activity_changed)
        controller.confidence_changed.connect(self._on_confidence)
        controller.status_changed.connect(self._on_status)
        controller.recognizing_changed.connect(self._on_recognizing)
        controller.two_hands_changed.connect(self._on_two_hands)
        controller.gesture_mode_changed.connect(self._on_gesture_mode)
        controller.pointer_mode_changed.connect(self._on_pointer_mode)
        controller.landmark_overlay_changed.connect(self._on_landmark_overlay)

    # ---- Жизненный цикл (вызывается shell при показе/скрытии) ------------

    def on_show(self) -> None:
        # Камера сама поднимется по «Старт»; ничего не делаем при простом
        # переключении на вкладку, чтобы зря не открывать устройство.
        self._refresh_activity()

    def on_hide(self) -> None:
        # При уходе с главной — НЕ останавливаем распознавание, потому что
        # «жест → команда ОС» должно работать в фоне (как в QML с subprocess).
        # Останавливать камеру/CV нужно только явной кнопкой «Стоп».
        pass

    # ---- Логика кнопки ----------------------------------------------------

    def _btn_label(self) -> str:
        return "Стоп" if self._controller.is_recognizing else "Старт"

    def _btn_icon(self) -> str:
        return (
            ft.Icons.STOP_CIRCLE
            if self._controller.is_recognizing
            else ft.Icons.PLAY_CIRCLE
        )

    def _btn_style(self) -> ft.ButtonStyle:
        return ft.ButtonStyle(
            bgcolor=COLOR_DANGER if self._controller.is_recognizing else COLOR_SUCCESS,
            color=ft.Colors.WHITE,
            padding=ft.Padding.symmetric(horizontal=24, vertical=14),
        )

    def _on_toggle(self, _e) -> None:
        # ВАЖНО: используем встроенный пайплайн (embedded), а не subprocess
        # ``realtime_infer.py``. Это требование пользователя — «камера
        # должна быть встроена в GUI».
        self._controller.toggle_recognition()

    def _on_landmarks_toggle(self, _e) -> None:
        self._controller.set_show_landmark_overlay(
            bool(self._landmarks_switch.value)
        )

    def _on_gesture_mode_toggle(self, _e) -> None:
        self._controller.set_gesture_mode(bool(self._gesture_switch.value))

    def _on_pointer_toggle(self, _e) -> None:
        self._controller.set_pointer_mode(bool(self._pointer_switch.value))

    # ---- Слушатели событий контроллера (приходят из фонового потока) ----

    def _on_frame(self) -> None:
        data = self._controller.latest_jpeg_bytes
        if not data:
            return
        b64 = base64.b64encode(data).decode("ascii")
        self._page.run_thread(self._apply_frame, f"data:image/jpeg;base64,{b64}")

    def _apply_frame(self, data_url: str) -> None:
        self._camera_image.src = data_url
        self._camera_image.visible = True
        try:
            self._page.update()
        except Exception:
            pass

    def _on_gesture(self, label: str) -> None:
        self._page.run_thread(self._apply_gesture, label)

    def _apply_gesture(self, label: str) -> None:
        self._gesture_text.value = label or "—"
        try:
            self._gesture_text.update()
        except Exception:
            pass

    def _on_command(self, name: str) -> None:
        self._page.run_thread(self._apply_command, name)

    def _apply_command(self, name: str) -> None:
        self._command_text.value = name or "—"
        try:
            self._command_text.update()
        except Exception:
            pass

    def _on_activity_changed(self) -> None:
        self._page.run_thread(self._refresh_activity)

    def _refresh_activity(self) -> None:
        rows = self._controller.get_recent_recognition_events(limit=6)
        if not rows:
            self._activity_list.controls = [
                ft.Text("Пока нет событий", size=12, color=COLOR_MUTED)
            ]
        else:
            self._activity_list.controls = [
                self._activity_row(row)
                for row in rows
            ]
        try:
            self._activity_list.update()
        except Exception:
            pass

    def _activity_row(self, row: dict) -> ft.Control:
        executed = bool(row.get("executed"))
        label = str(row.get("label") or "—")
        confidence = int(round(float(row.get("confidence") or 0.0) * 100))
        detected_at = str(row.get("detectedAt") or "")
        command = str(row.get("commandName") or "")
        detail = f"{confidence}%"
        if command and executed:
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

    def _on_confidence(self, value: float) -> None:
        self._page.run_thread(self._apply_confidence, value)

    def _apply_confidence(self, value: float) -> None:
        self._confidence_bar.value = max(0.0, min(1.0, float(value)))
        self._confidence_text.value = f"{int(round(value * 100))}%"
        try:
            self._confidence_bar.update()
            self._confidence_text.update()
        except Exception:
            pass

    def _on_status(self, value: str) -> None:
        self._page.run_thread(self._apply_status, value)

    def _apply_status(self, value: str) -> None:
        self._status_text.value = value
        try:
            self._status_text.update()
        except Exception:
            pass

    def _on_recognizing(self, _v: bool) -> None:
        self._page.run_thread(self._apply_button_state)

    def _apply_button_state(self) -> None:
        self._toggle_btn.content = ft.Text(
            self._btn_label(), size=15, weight=ft.FontWeight.BOLD
        )
        self._toggle_btn.icon = self._btn_icon()
        self._toggle_btn.style = self._btn_style()
        try:
            self._toggle_btn.update()
        except Exception:
            pass

    def _on_two_hands(self, value: bool) -> None:
        self._page.run_thread(self._apply_two_hands, value)

    def _apply_two_hands(self, value: bool) -> None:
        if self._two_hands_switch.value != value:
            self._two_hands_switch.value = value
            try:
                self._two_hands_switch.update()
            except Exception:
                pass

    def _on_gesture_mode(self, value: bool) -> None:
        self._page.run_thread(self._apply_gesture_mode, value)

    def _apply_gesture_mode(self, value: bool) -> None:
        if self._gesture_switch.value != value:
            self._gesture_switch.value = value
            try:
                self._gesture_switch.update()
            except Exception:
                pass

    def _on_pointer_mode(self, value: bool) -> None:
        self._page.run_thread(self._apply_pointer_mode, value)

    def _apply_pointer_mode(self, value: bool) -> None:
        if self._pointer_switch.value != value:
            self._pointer_switch.value = value
            try:
                self._pointer_switch.update()
            except Exception:
                pass

    def _on_landmark_overlay(self, value: bool) -> None:
        self._page.run_thread(self._apply_landmark_overlay, value)

    def _apply_landmark_overlay(self, value: bool) -> None:
        if self._landmarks_switch.value != value:
            self._landmarks_switch.value = value
        try:
            self._landmarks_switch.update()
        except Exception:
            pass

    # ---- Сборка дерева ---------------------------------------------------

    def build(self) -> ft.Control:
        toolbar_controls = ft.Row(
            spacing=18,
            wrap=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                self._toggle_btn,
                self._gesture_switch,
                self._pointer_switch,
                self._landmarks_switch,
                self._auto_exec_switch,
            ],
        )
        toolbar = ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Container(content=toolbar_controls, expand=True),
                ft.Container(
                    content=self._status_text,
                    alignment=ft.Alignment.CENTER_RIGHT,
                    width=420,
                ),
            ],
        )

        camera_card = ft.Container(
            content=ft.Stack(
                expand=True,
                controls=[
                    ft.Container(expand=True, bgcolor="#080812"),
                    ft.Container(
                        content=self._camera_image,
                        expand=True,
                        alignment=ft.Alignment.CENTER,
                    ),
                ],
            ),
            bgcolor="#0d0d18",
            border_radius=16,
            padding=8,
            height=420,
            alignment=ft.Alignment.CENTER,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )

        result_panel = ft.Container(
            height=126,
            padding=ft.Padding(22, 8, 22, 10),
            content=ft.Row(
                spacing=42,
                vertical_alignment=ft.CrossAxisAlignment.START,
                controls=[
                    ft.Column(
                        expand=True,
                        spacing=8,
                        controls=[
                            ft.Text("Жест", size=13, color=COLOR_MUTED),
                            self._gesture_text,
                            ft.Row(
                                controls=[
                                    ft.Text("Уверенность", color=COLOR_MUTED, size=12),
                                    self._confidence_text,
                                ],
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            ),
                            self._confidence_bar,
                        ],
                    ),
                    ft.Container(
                        width=420,
                        content=ft.Column(
                            spacing=8,
                            controls=[
                                ft.Text("Команда", size=13, color=COLOR_MUTED),
                                self._command_text,
                            ],
                        ),
                    ),
                ],
            ),
        )

        return ft.Column(
            spacing=18,
            expand=True,
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
                toolbar,
                camera_card,
                result_panel,
            ],
        )
