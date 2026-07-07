"""
Главный экран GestureBind: превью камеры + жесты + последняя команда.

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
import threading

import flet as ft

from app.flet_app.controller import (
    AppController,
    DYNAMIC_MODEL_PROFILE_PRODUCTION,
    LIVE_EVAL_NO_COMMAND_LABEL,
    STATIC_REJECTION_METHODS,
)
from app.flet_app.theme import (
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_MUTED,
    COLOR_ON_SURFACE,
    COLOR_SUCCESS,
    COLOR_SURFACE,
    COLOR_SURFACE_HIGH,
    COLOR_WARNING,
    surface_card,
)


# 1×1 PNG-плейсхолдер до прихода первого кадра.
_PLACEHOLDER_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "2mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
LIVE_PREDICTION_HOLD_SECONDS = 1.8


class HomeView:
    """Главный экран (с состоянием камеры и подписками)."""

    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller
        self._visible = True
        self._frame_update_lock = threading.Lock()
        self._frame_update_pending = False
        self._gesture_clear_lock = threading.Lock()
        self._gesture_clear_token = 0
        self._gesture_state_phase = "idle"
        self._recognition_inspector_paused = False
        self._recognition_inspector_snapshot: list[dict] = []
        self._recognition_inspector_last_seen_sequence = 0

        self._camera_image = ft.Image(
            src=_PLACEHOLDER_DATA_URL,
            fit=ft.BoxFit.CONTAIN,
            gapless_playback=True,
            expand=True,
            visible=False,
        )

        self._gesture_text = ft.Text(
            "—",
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_SURFACE,
        )
        self._confidence_text = ft.Text("0%", size=14, color=COLOR_MUTED)
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
            size=16,
            weight=ft.FontWeight.W_500,
            color=COLOR_ON_SURFACE,
        )
        self._activity_list = ft.Column(
            spacing=6,
            scroll=ft.ScrollMode.AUTO,
            controls=[ft.Text("Пока нет событий", size=12, color=COLOR_MUTED)],
        )
        self._recognition_inspector_list = ft.Column(
            spacing=5,
            scroll=ft.ScrollMode.AUTO,
            controls=[ft.Text("—", size=12, color=COLOR_MUTED)],
        )
        self._recognition_inspector_status_text = ft.Text(
            "live",
            size=11,
            color=COLOR_MUTED,
            no_wrap=True,
        )
        self._recognition_inspector_pause_btn = ft.IconButton(
            icon=ft.Icons.PAUSE,
            icon_size=17,
            icon_color=COLOR_MUTED,
            tooltip="Pause inspector",
            on_click=self._on_recognition_inspector_pause_toggle,
        )
        self._recognition_inspector_step_btn = ft.IconButton(
            icon=ft.Icons.SKIP_NEXT,
            icon_size=17,
            icon_color=COLOR_MUTED,
            tooltip="Step next inspector event",
            disabled=True,
            on_click=self._on_recognition_inspector_step,
        )
        self._recognition_inspector_clear_btn = ft.IconButton(
            icon=ft.Icons.CLEAR_ALL,
            icon_size=17,
            icon_color=COLOR_MUTED,
            tooltip="Clear inspector history",
            on_click=self._on_recognition_inspector_clear,
        )
        self._recognition_inspector_export_jsonl_btn = ft.IconButton(
            icon=ft.Icons.ARTICLE,
            icon_size=17,
            icon_color=COLOR_MUTED,
            tooltip="Export inspector JSONL",
            on_click=lambda _e: self._on_recognition_inspector_export("jsonl"),
        )
        self._recognition_inspector_export_csv_btn = ft.IconButton(
            icon=ft.Icons.FILE_DOWNLOAD,
            icon_size=17,
            icon_color=COLOR_MUTED,
            tooltip="Export inspector CSV",
            on_click=lambda _e: self._on_recognition_inspector_export("csv"),
        )
        eval_labels = self._recognition_label_options()
        default_eval_label = "swipe_down" if "swipe_down" in eval_labels else (
            eval_labels[0] if eval_labels else ""
        )
        self._eval_expected = ft.Dropdown(
            label="Ожидаем",
            value=default_eval_label,
            width=180,
            options=[
                ft.DropdownOption(key=label, text=label)
                for label in eval_labels
            ],
        )
        self._eval_attempts = ft.TextField(
            label="Попыток",
            value="10",
            width=104,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._eval_timeout = ft.TextField(
            label="Таймаут 0=нет",
            value="0",
            width=132,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._eval_threshold = ft.TextField(
            label="Порог",
            value="0.60",
            width=112,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._eval_start_btn = ft.FilledButton(
            content=ft.Text("Начать тест", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.PLAYLIST_PLAY,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=18, vertical=12),
            ),
            on_click=self._on_eval_start,
        )
        self._eval_stop_btn = ft.OutlinedButton(
            content=ft.Text("Стоп"),
            icon=ft.Icons.STOP,
            disabled=True,
            on_click=self._on_eval_stop,
        )
        self._eval_miss_btn = ft.OutlinedButton(
            content=ft.Text("Пропуск"),
            icon=ft.Icons.SKIP_NEXT,
            disabled=True,
            on_click=self._on_eval_miss,
        )
        self._eval_progress_text = ft.Text("0/10", size=13, color=COLOR_ON_SURFACE)
        self._eval_correct_text = ft.Text("Верно 0", size=13, color=COLOR_SUCCESS)
        self._eval_wrong_text = ft.Text("Ошибка 0", size=13, color=COLOR_DANGER)
        self._eval_missed_text = ft.Text("Пропуск 0", size=13, color=COLOR_MUTED)
        self._eval_accuracy_text = ft.Text("Accuracy —", size=13, color=COLOR_ON_SURFACE)
        self._eval_last_text = ft.Text("—", size=12, color=COLOR_MUTED)
        self._eval_progress_bar = ft.ProgressBar(
            value=0.0,
            color=COLOR_ACCENT,
            bgcolor=COLOR_SURFACE_HIGH,
        )
        self._eval_overlay_title = ft.Text(
            "Live evaluation",
            size=14,
            weight=ft.FontWeight.W_700,
            color=COLOR_ON_SURFACE,
        )
        self._eval_overlay = ft.Container(
            visible=False,
            width=540,
            padding=ft.Padding(14, 12, 14, 12),
            bgcolor="#111417",
            border_radius=12,
            content=ft.Column(
                spacing=10,
                controls=[
                    ft.Row(
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            self._eval_overlay_title,
                            ft.Container(expand=True),
                            self._eval_miss_btn,
                        ],
                    ),
                    ft.Row(
                        spacing=16,
                        wrap=True,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            self._eval_progress_text,
                            self._eval_correct_text,
                            self._eval_wrong_text,
                            self._eval_missed_text,
                            self._eval_accuracy_text,
                        ],
                    ),
                    self._eval_last_text,
                    self._eval_progress_bar,
                ],
            ),
        )

        self._toggle_btn = ft.FilledButton(
            content=ft.Text(self._btn_label(), size=15, weight=ft.FontWeight.BOLD),
            icon=self._btn_icon(),
            style=self._btn_style(),
            on_click=self._on_toggle,
            height=56,
        )

        self._model_mode_dd = ft.Dropdown(
            label="Модель",
            value=controller.recognition_model_mode,
            width=150,
            dense=True,
            options=[
                ft.DropdownOption(key="auto", text="auto"),
                ft.DropdownOption(key="static", text="static"),
                ft.DropdownOption(key="dynamic", text="dynamic"),
            ],
            on_select=self._on_model_mode_changed,
        )
        self._model_variant_dd = ft.Dropdown(
            label="Variant",
            value=controller.model_variant,
            width=190,
            dense=True,
            options=self._model_variant_options(),
            on_select=self._on_model_variant_changed,
        )
        self._dynamic_profile_dd = ft.Dropdown(
            label="Dynamic",
            value=controller.dynamic_model_profile,
            width=240,
            dense=True,
            visible=controller.recognition_model_mode in {"auto", "dynamic"},
            options=self._dynamic_profile_options(),
            on_select=self._on_dynamic_profile_changed,
        )
        self._static_rejection_dd = ft.Dropdown(
            label="Reject",
            value=controller.static_rejection_method,
            width=220,
            dense=True,
            visible=controller.recognition_model_mode in {"auto", "static"},
            options=[
                ft.DropdownOption(key=method, text=method)
                for method in STATIC_REJECTION_METHODS
            ],
            on_select=self._on_static_rejection_method_changed,
        )
        self._two_hands_switch = ft.Switch(
            value=controller.two_hands_mode,
            label="2 руки",
            active_color=COLOR_ACCENT,
            on_change=lambda e: controller.set_two_hands_mode(
                bool(self._two_hands_switch.value)
            ),
        )
        self._auto_exec_switch = ft.Switch(
            value=controller.auto_execute,
            label="Авто",
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
            label="Точки",
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
        self._pointer_state_text = ft.Text(
            "Pointer: off",
            size=12,
            color=COLOR_MUTED,
            no_wrap=True,
        )

        # Подписки. Все события могут прийти из фонового потока, поэтому
        # все обновления UI идут через ``page.run_thread``.
        controller.camera_frame_updated.connect(self._on_frame)
        controller.gesture_detected.connect(self._on_gesture)
        controller.command_executed.connect(self._on_command)
        controller.recognition_event_recorded.connect(self._on_activity_changed)
        controller.confidence_changed.connect(self._on_confidence)
        controller.gesture_state_changed.connect(self._on_gesture_state)
        controller.status_changed.connect(self._on_status)
        controller.recognizing_changed.connect(self._on_recognizing)
        controller.camera_active_changed.connect(self._on_camera_active)
        controller.two_hands_changed.connect(self._on_two_hands)
        controller.gesture_mode_changed.connect(self._on_gesture_mode)
        controller.pointer_mode_changed.connect(self._on_pointer_mode)
        controller.pointer_state_changed.connect(self._on_pointer_state)
        controller.landmark_overlay_changed.connect(self._on_landmark_overlay)
        controller.recognition_model_mode_changed.connect(self._on_model_mode)
        controller.dynamic_model_profile_changed.connect(self._on_dynamic_profile)
        controller.model_variant_changed.connect(self._on_model_variant)
        controller.static_rejection_method_changed.connect(self._on_static_rejection_method)
        controller.live_evaluation_changed.connect(self._on_live_evaluation)

    # ---- Жизненный цикл (вызывается shell при показе/скрытии) ------------

    def on_show(self) -> None:
        # Камера сама поднимется по «Старт»; ничего не делаем при простом
        # переключении на вкладку, чтобы зря не открывать устройство.
        self._visible = True
        self._sync_dynamic_profile_dropdown()
        self._apply_model_variant(self._controller.model_variant)
        self._refresh_eval_labels()
        self._refresh_activity()
        self._refresh_recognition_inspector()
        self._apply_live_evaluation(self._controller.current_live_evaluation())

    def on_hide(self) -> None:
        # При уходе с главной — НЕ останавливаем распознавание, потому что
        # «жест → команда ОС» должно работать в фоне (как в QML с subprocess).
        # Останавливать камеру/CV нужно только явной кнопкой «Стоп».
        self._visible = False

    def _recognition_label_options(self) -> list[str]:
        try:
            labels = self._controller.list_recognition_labels()
        except Exception:
            labels = []
        out: list[str] = []
        seen: set[str] = set()
        for label in [LIVE_EVAL_NO_COMMAND_LABEL, *labels]:
            clean = str(label or "").strip()
            key = clean.lower()
            if clean and key not in seen:
                out.append(clean)
                seen.add(key)
        return out

    def _model_variant_options(self) -> list[ft.DropdownOption]:
        try:
            variants = self._controller.list_model_variants()
        except Exception:
            variants = []
        return [
            ft.DropdownOption(
                key=str(item.get("key") or ""),
                text=str(item.get("label") or item.get("key") or ""),
            )
            for item in variants
            if str(item.get("key") or "").strip()
        ]

    def _dynamic_profile_rows(self) -> list[dict]:
        try:
            profiles = self._controller.list_dynamic_model_profiles()
        except Exception:
            profiles = []
        rows = [row for row in profiles if bool(row.get("quick_selectable"))]
        if not any(
            str(row.get("key") or "") == DYNAMIC_MODEL_PROFILE_PRODUCTION
            for row in rows
        ):
            production = next(
                (
                    row
                    for row in profiles
                    if str(row.get("key") or "") == DYNAMIC_MODEL_PROFILE_PRODUCTION
                ),
                None,
            )
            if production is not None:
                rows.insert(0, production)
        if not rows:
            rows = [
                {
                    "key": DYNAMIC_MODEL_PROFILE_PRODUCTION,
                    "label": DYNAMIC_MODEL_PROFILE_PRODUCTION,
                }
            ]
        return rows

    def _dynamic_profile_options(
        self,
        rows: list[dict] | None = None,
    ) -> list[ft.DropdownOption]:
        return [
            ft.DropdownOption(
                key=str(item.get("key") or ""),
                text=str(item.get("label") or item.get("key") or ""),
            )
            for item in (rows if rows is not None else self._dynamic_profile_rows())
            if str(item.get("key") or "").strip()
        ]

    def _sync_dynamic_profile_dropdown(self, *, update: bool = False) -> None:
        rows = self._dynamic_profile_rows()
        allowed = {str(row.get("key") or "") for row in rows}
        current = self._controller.dynamic_model_profile
        if current not in allowed:
            current = DYNAMIC_MODEL_PROFILE_PRODUCTION
            self._controller.set_dynamic_model_profile(current)
        self._dynamic_profile_dd.options = self._dynamic_profile_options(rows)
        if self._dynamic_profile_dd.value != current:
            self._dynamic_profile_dd.value = current
        if update:
            try:
                self._dynamic_profile_dd.update()
            except Exception:
                pass

    def _refresh_eval_labels(self) -> None:
        labels = self._recognition_label_options()
        current = str(self._eval_expected.value or "")
        self._eval_expected.options = [
            ft.DropdownOption(key=label, text=label)
            for label in labels
        ]
        if current not in labels:
            self._eval_expected.value = labels[0] if labels else ""
        try:
            self._eval_expected.update()
        except Exception:
            pass

    # ---- Логика кнопки ----------------------------------------------------

    def _is_running(self) -> bool:
        return bool(
            getattr(
                self._controller,
                "live_recognition_active",
                self._controller.is_recognizing,
            )
        )

    def _btn_label(self) -> str:
        return "Стоп" if self._is_running() else "Старт"

    def _btn_icon(self) -> str:
        return (
            ft.Icons.STOP_CIRCLE
            if self._is_running()
            else ft.Icons.PLAY_CIRCLE
        )

    def _btn_style(self) -> ft.ButtonStyle:
        return ft.ButtonStyle(
            bgcolor=COLOR_DANGER if self._is_running() else COLOR_SUCCESS,
            color=ft.Colors.WHITE,
            padding=ft.Padding.symmetric(horizontal=24, vertical=14),
        )

    def _on_toggle(self, _e) -> None:
        # ВАЖНО: используем встроенный пайплайн (embedded), а не subprocess
        # ``realtime_infer.py``. Это требование пользователя — «камера
        # должна быть встроена в GUI».
        runner = getattr(self._page, "run_thread", None)
        if callable(runner):
            runner(self._controller.toggle_recognition)
        else:
            self._controller.toggle_recognition()

    def _on_landmarks_toggle(self, _e) -> None:
        self._controller.set_show_landmark_overlay(
            bool(self._landmarks_switch.value)
        )

    def _on_gesture_mode_toggle(self, _e) -> None:
        self._controller.set_gesture_mode(bool(self._gesture_switch.value))

    def _on_pointer_toggle(self, _e) -> None:
        self._controller.set_pointer_mode(bool(self._pointer_switch.value))

    def _on_model_mode_changed(self, _e) -> None:
        self._controller.set_recognition_model_mode(
            str(self._model_mode_dd.value or "auto")
        )
        self._refresh_eval_labels()

    def _on_model_variant_changed(self, _e) -> None:
        ok, errors, warnings = self._controller.apply_model_variant(
            str(self._model_variant_dd.value or "production")
        )
        if not ok:
            self._status_text.value = "Variant error: " + "; ".join(errors)
            try:
                self._status_text.update()
            except Exception:
                pass
            return
        if warnings:
            self._status_text.value = "Variant warning: " + " · ".join(warnings)
            try:
                self._status_text.update()
            except Exception:
                pass
        self._refresh_eval_labels()

    def _on_dynamic_profile_changed(self, _e) -> None:
        self._controller.set_dynamic_model_profile(
            str(self._dynamic_profile_dd.value or DYNAMIC_MODEL_PROFILE_PRODUCTION)
        )

    def _on_static_rejection_method_changed(self, _e) -> None:
        self._controller.set_static_rejection_method(
            str(self._static_rejection_dd.value or "open_set_policy")
        )

    def _parse_int_field(self, field: ft.TextField, default: int) -> int:
        try:
            return int(str(field.value or "").strip())
        except (TypeError, ValueError):
            field.value = str(default)
            try:
                field.update()
            except Exception:
                pass
            return default

    def _parse_float_field(self, field: ft.TextField, default: float) -> float:
        try:
            return float(str(field.value or "").strip().replace(",", "."))
        except (TypeError, ValueError):
            field.value = f"{default:.2f}"
            try:
                field.update()
            except Exception:
                pass
            return default

    def _on_eval_start(self, _e) -> None:
        expected = str(self._eval_expected.value or "").strip()
        attempts = self._parse_int_field(self._eval_attempts, 10)
        timeout = self._parse_float_field(self._eval_timeout, 0.0)
        threshold = self._parse_float_field(self._eval_threshold, 0.60)
        if self._controller.start_live_evaluation(
            expected,
            attempts=attempts,
            timeout_seconds=timeout,
            min_confidence=threshold,
        ):
            self._apply_live_evaluation(self._controller.current_live_evaluation())

    def _on_eval_stop(self, _e) -> None:
        self._controller.cancel_live_evaluation()

    def _on_eval_miss(self, _e) -> None:
        self._controller.mark_live_evaluation_missed()

    # ---- Слушатели событий контроллера (приходят из фонового потока) ----

    def _on_frame(self) -> None:
        if not self._visible:
            return
        with self._frame_update_lock:
            if self._frame_update_pending:
                return
            self._frame_update_pending = True
        data = self._controller.latest_jpeg_bytes
        if not data:
            with self._frame_update_lock:
                self._frame_update_pending = False
            return
        b64 = base64.b64encode(data).decode("ascii")
        try:
            self._page.run_thread(
                self._apply_frame,
                f"data:image/jpeg;base64,{b64}",
            )
        except Exception:
            with self._frame_update_lock:
                self._frame_update_pending = False

    def _apply_frame(self, data_url: str) -> None:
        try:
            if self._visible:
                self._camera_image.src = data_url
                self._camera_image.visible = True
                try:
                    self._camera_image.update()
                except Exception:
                    self._page.update()
        except Exception:
            pass
        finally:
            with self._frame_update_lock:
                self._frame_update_pending = False

    def _on_gesture(self, label: str) -> None:
        self._page.run_thread(self._apply_gesture, label)

    def _apply_gesture(self, label: str) -> None:
        clean = str(label or "").strip()
        if clean:
            with self._gesture_clear_lock:
                self._gesture_clear_token += 1
            self._gesture_text.value = clean
            self._safe_update_gesture_text()
            return
        if self._is_running():
            self._schedule_gesture_clear()
        else:
            self._clear_gesture_now()

    def _safe_update_gesture_text(self) -> None:
        try:
            self._gesture_text.update()
        except Exception:
            pass

    def _schedule_gesture_clear(self) -> None:
        with self._gesture_clear_lock:
            self._gesture_clear_token += 1
            token = self._gesture_clear_token

        def clear_later() -> None:
            try:
                self._page.run_thread(self._clear_gesture_if_current, token)
            except Exception:
                pass

        timer = threading.Timer(LIVE_PREDICTION_HOLD_SECONDS, clear_later)
        timer.daemon = True
        timer.start()

    def _clear_gesture_if_current(self, token: int) -> None:
        with self._gesture_clear_lock:
            if token != self._gesture_clear_token:
                return
        self._gesture_text.value = "—"
        self._safe_update_gesture_text()

    def _clear_gesture_now(self) -> None:
        with self._gesture_clear_lock:
            self._gesture_clear_token += 1
        self._gesture_text.value = "—"
        self._safe_update_gesture_text()

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

    def _current_recognition_inspector_rows(self) -> list[dict]:
        try:
            rows = self._controller.get_live_gesture_inspector_history()
        except Exception:
            rows = []
        return [dict(row) for row in rows if isinstance(row, dict)]

    def _recognition_inspector_max_sequence(self, rows: list[dict]) -> int:
        out = 0
        for row in rows:
            try:
                out = max(out, int(row.get("sequence") or 0))
            except (TypeError, ValueError):
                pass
        return out

    def _recognition_inspector_unseen_rows(self) -> list[dict]:
        last_seen = int(getattr(self, "_recognition_inspector_last_seen_sequence", 0))
        out: list[dict] = []
        for row in self._current_recognition_inspector_rows():
            try:
                sequence = int(row.get("sequence") or 0)
            except (TypeError, ValueError):
                sequence = 0
            if sequence > last_seen:
                out.append(row)
        return sorted(out, key=lambda item: int(item.get("sequence") or 0))

    def _safe_update_control(self, control: ft.Control) -> None:
        try:
            control.update()
        except Exception:
            pass

    def _apply_recognition_inspector_controls(self) -> None:
        paused = bool(getattr(self, "_recognition_inspector_paused", False))
        rows = (
            list(getattr(self, "_recognition_inspector_snapshot", []))
            if paused
            else self._current_recognition_inspector_rows()
        )
        unseen_count = len(self._recognition_inspector_unseen_rows()) if paused else 0
        self._recognition_inspector_status_text.value = (
            f"paused +{unseen_count}" if paused and unseen_count else (
                "paused" if paused else "live"
            )
        )
        self._recognition_inspector_status_text.color = (
            COLOR_WARNING if paused else COLOR_MUTED
        )
        self._recognition_inspector_pause_btn.icon = (
            ft.Icons.PLAY_ARROW if paused else ft.Icons.PAUSE
        )
        self._recognition_inspector_pause_btn.tooltip = (
            "Resume inspector" if paused else "Pause inspector"
        )
        self._recognition_inspector_step_btn.disabled = not paused or unseen_count <= 0
        has_rows = bool(rows) or bool(self._current_recognition_inspector_rows())
        self._recognition_inspector_clear_btn.disabled = not has_rows
        self._recognition_inspector_export_jsonl_btn.disabled = not has_rows
        self._recognition_inspector_export_csv_btn.disabled = not has_rows
        for control in (
            self._recognition_inspector_status_text,
            self._recognition_inspector_pause_btn,
            self._recognition_inspector_step_btn,
            self._recognition_inspector_clear_btn,
            self._recognition_inspector_export_jsonl_btn,
            self._recognition_inspector_export_csv_btn,
        ):
            self._safe_update_control(control)

    def _refresh_recognition_inspector(self) -> None:
        if bool(getattr(self, "_recognition_inspector_paused", False)):
            rows = list(getattr(self, "_recognition_inspector_snapshot", []))
        else:
            rows = self._current_recognition_inspector_rows()
            self._recognition_inspector_last_seen_sequence = (
                self._recognition_inspector_max_sequence(rows)
            )
        if not rows:
            self._recognition_inspector_list.controls = [
                ft.Text("—", size=12, color=COLOR_MUTED)
            ]
        else:
            self._recognition_inspector_list.controls = [
                self._recognition_inspector_row(row)
                for row in rows[:6]
            ]
        try:
            self._recognition_inspector_list.update()
        except Exception:
            pass
        self._apply_recognition_inspector_controls()

    def _on_recognition_inspector_pause_toggle(self, _e) -> None:
        paused = not bool(getattr(self, "_recognition_inspector_paused", False))
        self._recognition_inspector_paused = paused
        if paused:
            self._recognition_inspector_snapshot = (
                self._current_recognition_inspector_rows()
            )
            self._recognition_inspector_last_seen_sequence = (
                self._recognition_inspector_max_sequence(
                    self._recognition_inspector_snapshot
                )
            )
        else:
            self._recognition_inspector_snapshot = []
        self._refresh_recognition_inspector()

    def _on_recognition_inspector_step(self, _e) -> None:
        if not bool(getattr(self, "_recognition_inspector_paused", False)):
            self._on_recognition_inspector_pause_toggle(_e)
            return
        unseen = self._recognition_inspector_unseen_rows()
        if not unseen:
            self._refresh_recognition_inspector()
            return
        next_row = dict(unseen[0])
        self._recognition_inspector_snapshot.insert(0, next_row)
        del self._recognition_inspector_snapshot[6:]
        try:
            self._recognition_inspector_last_seen_sequence = int(
                next_row.get("sequence") or self._recognition_inspector_last_seen_sequence
            )
        except (TypeError, ValueError):
            pass
        self._refresh_recognition_inspector()

    def _on_recognition_inspector_clear(self, _e) -> None:
        try:
            self._controller.clear_live_gesture_inspector_history()
        except Exception as e:
            self._recognition_inspector_status_text.value = f"clear failed: {e}"
            self._safe_update_control(self._recognition_inspector_status_text)
            return
        self._recognition_inspector_snapshot = []
        self._recognition_inspector_last_seen_sequence = 0
        self._refresh_recognition_inspector()

    def _on_recognition_inspector_export(self, fmt: str) -> None:
        try:
            path = self._controller.export_live_gesture_inspector_history(fmt)
        except Exception as e:
            self._recognition_inspector_status_text.value = f"export failed: {e}"
            self._recognition_inspector_status_text.color = COLOR_DANGER
        else:
            self._recognition_inspector_status_text.value = f"exported {path.name}"
            self._recognition_inspector_status_text.color = COLOR_SUCCESS
        self._safe_update_control(self._recognition_inspector_status_text)

    def _recognition_inspector_phase_color(self, phase: str) -> str:
        clean = str(phase or "").strip().lower()
        if clean == "confirmed":
            return COLOR_SUCCESS
        if clean in {"pending", "suppressed", "cooldown"}:
            return COLOR_WARNING
        if clean == "rejected":
            return COLOR_DANGER
        return COLOR_MUTED

    def _recognition_inspector_row(self, row: dict) -> ft.Control:
        data = row if isinstance(row, dict) else {}
        phase = str(data.get("phase") or "idle").strip()
        label = str(data.get("label") or "—").strip() or "—"
        route = str(data.get("route") or "—").strip() or "—"
        mode = str(data.get("mode") or "—").strip() or "—"
        model = str(data.get("model") or "—").strip() or "—"
        reason = str(data.get("reason") or "").strip()
        confidence = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
        frames = int(data.get("frames") or 0)
        required = int(data.get("requiredFrames") or 0)
        frame_text = f"{frames}/{max(1, required)}" if frames or required else "—"
        details = [f"{mode}:{route}", model, frame_text]
        if reason:
            details.append(reason)
        color = self._recognition_inspector_phase_color(phase)
        return ft.Container(
            bgcolor="#171A1D",
            border_radius=8,
            padding=ft.Padding.symmetric(horizontal=9, vertical=7),
            content=ft.Row(
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Text(
                        phase,
                        width=72,
                        size=11,
                        weight=ft.FontWeight.W_700,
                        color=color,
                        no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                    ft.Column(
                        spacing=1,
                        expand=True,
                        controls=[
                            ft.Text(
                                label,
                                size=12,
                                weight=ft.FontWeight.W_600,
                                color=COLOR_ON_SURFACE,
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Text(
                                " · ".join(details),
                                size=10,
                                color=COLOR_MUTED,
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ],
                    ),
                    ft.Text(
                        f"{int(round(confidence * 100))}%",
                        width=38,
                        size=11,
                        color=COLOR_MUTED,
                        text_align=ft.TextAlign.RIGHT,
                        no_wrap=True,
                    ),
                ],
            ),
        )

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
        return ft.Container(
            bgcolor="#171A1D",
            border_radius=8,
            padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            content=ft.Row(
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
            ),
        )

    def _on_confidence(self, value: float) -> None:
        self._page.run_thread(self._apply_confidence, value)

    def _apply_confidence(self, value: float) -> None:
        safe_value = max(0.0, min(1.0, float(value)))
        if self._gesture_state_phase != "idle" and safe_value <= 0.0:
            return
        self._confidence_bar.value = safe_value
        self._confidence_bar.color = COLOR_ACCENT
        self._confidence_text.value = f"{int(round(safe_value * 100))}%"
        try:
            self._confidence_bar.update()
            self._confidence_text.update()
        except Exception:
            pass

    def _on_gesture_state(self, payload: dict) -> None:
        self._page.run_thread(self._apply_gesture_state, payload)

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
            self._confidence_text.value = f"{frames}/{max(1, required)} · {int(round(confidence * 100))}%"
            self._gesture_state_text.value = "подтверждение"
            self._gesture_state_text.color = COLOR_WARNING
            self._safe_update_gesture_text()
        elif phase == "confirmed":
            self._confidence_bar.value = confidence
            self._confidence_bar.color = COLOR_SUCCESS
            self._confidence_text.value = f"{int(round(confidence * 100))}% · готово"
            self._gesture_state_text.value = "готово"
            self._gesture_state_text.color = COLOR_SUCCESS
        elif phase == "rejected":
            self._gesture_text.value = label or "—"
            self._confidence_bar.value = 1.0
            self._confidence_bar.color = COLOR_DANGER
            self._confidence_text.value = f"{int(round(confidence * 100))}%"
            self._gesture_state_text.value = reason or "отклонено"
            self._gesture_state_text.color = COLOR_DANGER
            self._safe_update_gesture_text()
        elif phase == "suppressed":
            self._gesture_text.value = label or "—"
            self._confidence_bar.value = 1.0
            self._confidence_bar.color = COLOR_WARNING
            self._confidence_text.value = f"{int(round(confidence * 100))}%"
            self._gesture_state_text.value = reason or "подавлено"
            self._gesture_state_text.color = COLOR_WARNING
            self._safe_update_gesture_text()
        elif phase == "cooldown":
            self._gesture_text.value = label or "—"
            self._confidence_bar.value = 1.0
            self._confidence_bar.color = COLOR_WARNING
            self._confidence_text.value = "cooldown"
            self._gesture_state_text.value = reason or "cooldown"
            self._gesture_state_text.color = COLOR_WARNING
            self._safe_update_gesture_text()
        else:
            self._confidence_bar.value = 0.0
            self._confidence_bar.color = COLOR_ACCENT
            self._confidence_text.value = "0%"
            self._gesture_state_text.value = "ожидание"
            self._gesture_state_text.color = COLOR_MUTED

        try:
            self._confidence_bar.update()
            self._confidence_text.update()
            self._gesture_state_text.update()
        except Exception:
            pass
        self._refresh_recognition_inspector()

    def _on_status(self, value: str) -> None:
        self._page.run_thread(self._apply_status, value)

    def _apply_status(self, value: str) -> None:
        self._status_text.value = value
        try:
            self._status_text.update()
        except Exception:
            pass

    def _on_recognizing(self, value: bool) -> None:
        self._page.run_thread(self._apply_recognizing_state, value)

    def _apply_recognizing_state(self, active: bool) -> None:
        self._apply_button_state()
        if not active:
            self._clear_gesture_now()

    def _on_camera_active(self, active: bool) -> None:
        self._page.run_thread(self._apply_camera_active, active)

    def _apply_camera_active(self, active: bool) -> None:
        if not active:
            self._camera_image.src = _PLACEHOLDER_DATA_URL
            self._camera_image.visible = False
        self._apply_button_state()

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
        if not value:
            self._clear_gesture_now()

    def _on_pointer_mode(self, value: bool) -> None:
        self._page.run_thread(self._apply_pointer_mode, value)

    def _apply_pointer_mode(self, value: bool) -> None:
        if self._pointer_switch.value != value:
            self._pointer_switch.value = value
            try:
                self._pointer_switch.update()
            except Exception:
                pass
        if not value:
            self._apply_pointer_state({"enabled": False, "state": "idle"})

    def _on_pointer_state(self, payload: dict) -> None:
        self._page.run_thread(self._apply_pointer_state, payload)

    def _apply_pointer_state(self, payload: dict) -> None:
        data = payload if isinstance(payload, dict) else {}
        enabled = bool(data.get("enabled"))
        state = str(data.get("state") or "idle").strip()
        error = str(data.get("error") or "").strip()
        tab_switched = str(data.get("tabSwitched") or "").strip()
        clicked = bool(data.get("clicked"))

        if not enabled:
            text = "Pointer: off"
            color = COLOR_MUTED
        elif error or state == "disabled":
            text = f"Pointer: {error or 'disabled'}"
            color = COLOR_DANGER
        elif tab_switched:
            text = f"Pointer: swipe {tab_switched}"
            color = COLOR_WARNING
        elif clicked:
            text = "Pointer: clicked"
            color = COLOR_SUCCESS
        elif state == "lost":
            text = "Pointer: lost hand"
            color = COLOR_WARNING
        elif state == "click-ready":
            text = "Pointer: click-ready"
            color = COLOR_WARNING
        elif state == "swipe-tracking":
            text = "Pointer: swipe"
            color = COLOR_WARNING
        elif state == "tracking":
            text = "Pointer: tracking"
            color = COLOR_SUCCESS
        else:
            text = "Pointer: idle"
            color = COLOR_MUTED

        self._pointer_state_text.value = text
        self._pointer_state_text.color = color
        try:
            self._pointer_state_text.update()
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

    def _on_model_mode(self, value: str) -> None:
        self._page.run_thread(self._apply_model_mode, value)

    def _apply_model_mode(self, value: str) -> None:
        if self._model_mode_dd.value != value:
            self._model_mode_dd.value = value
        self._dynamic_profile_dd.visible = value in {"auto", "dynamic"}
        self._static_rejection_dd.visible = value in {"auto", "static"}
        try:
            self._model_mode_dd.update()
        except Exception:
            pass
        try:
            self._dynamic_profile_dd.update()
        except Exception:
            pass
        try:
            self._static_rejection_dd.update()
        except Exception:
            pass
        self._refresh_eval_labels()

    def _on_dynamic_profile(self, value: str) -> None:
        self._page.run_thread(self._apply_dynamic_profile, value)

    def _apply_dynamic_profile(self, _value: str) -> None:
        self._sync_dynamic_profile_dropdown()
        current = self._controller.dynamic_model_profile
        if self._dynamic_profile_dd.value != current:
            self._dynamic_profile_dd.value = current
        self._dynamic_profile_dd.visible = (
            self._controller.recognition_model_mode in {"auto", "dynamic"}
        )
        try:
            self._dynamic_profile_dd.update()
        except Exception:
            pass

    def _on_model_variant(self, value: str) -> None:
        self._page.run_thread(self._apply_model_variant, value)

    def _apply_model_variant(self, value: str) -> None:
        self._model_variant_dd.options = self._model_variant_options()
        self._sync_dynamic_profile_dropdown()
        if self._model_variant_dd.value != value:
            self._model_variant_dd.value = value
        try:
            self._model_variant_dd.update()
        except Exception:
            pass

    def _on_static_rejection_method(self, value: str) -> None:
        self._page.run_thread(self._apply_static_rejection_method, value)

    def _apply_static_rejection_method(self, value: str) -> None:
        if self._static_rejection_dd.value != value:
            self._static_rejection_dd.value = value
        self._static_rejection_dd.visible = (
            self._controller.recognition_model_mode in {"auto", "static"}
        )
        try:
            self._static_rejection_dd.update()
        except Exception:
            pass

    def _on_live_evaluation(self, snapshot: dict) -> None:
        self._page.run_thread(self._apply_live_evaluation, snapshot)

    def _apply_live_evaluation(self, snapshot: dict) -> None:
        active = bool(snapshot.get("active"))
        target = int(snapshot.get("targetAttempts") or 0)
        total = int(snapshot.get("total") or 0)
        correct = int(snapshot.get("correct") or 0)
        wrong = int(snapshot.get("wrong") or 0)
        missed = int(snapshot.get("missed") or 0)
        accuracy = float(snapshot.get("accuracy") or 0.0)
        progress = float(snapshot.get("progress") or 0.0)
        expected = str(snapshot.get("expectedLabel") or "")
        attempt_index = int(snapshot.get("attemptIndex") or 0)
        last_result = str(snapshot.get("lastResult") or "")
        last_prediction = str(snapshot.get("lastPrediction") or "")
        last_conf = float(snapshot.get("lastConfidence") or 0.0)
        message = str(snapshot.get("message") or "")

        self._eval_progress_text.value = (
            f"{total}/{target}" if not active else f"{attempt_index}/{target}"
        )
        self._eval_overlay_title.value = (
            f"Тест: {expected}" if expected else "Live evaluation"
        )
        self._eval_correct_text.value = f"Верно {correct}"
        self._eval_wrong_text.value = f"Ошибка {wrong}"
        self._eval_missed_text.value = f"Пропуск {missed}"
        self._eval_accuracy_text.value = (
            f"Accuracy {accuracy * 100:.0f}%" if total else "Accuracy —"
        )
        if last_result == "below_threshold" and last_prediction:
            self._eval_last_text.value = (
                f"{last_prediction} {last_conf * 100:.0f}% < threshold"
            )
        elif last_result and last_prediction:
            self._eval_last_text.value = (
                f"{last_result}: {last_prediction} {last_conf * 100:.0f}%"
            )
        elif message:
            self._eval_last_text.value = message
        elif expected:
            self._eval_last_text.value = expected
        else:
            self._eval_last_text.value = "—"
        self._eval_progress_bar.value = max(0.0, min(1.0, progress))

        self._eval_expected.disabled = active
        self._eval_attempts.disabled = active
        self._eval_timeout.disabled = active
        self._eval_threshold.disabled = active
        self._model_mode_dd.disabled = active
        self._model_variant_dd.disabled = active
        self._dynamic_profile_dd.disabled = active
        self._static_rejection_dd.disabled = active
        self._eval_start_btn.disabled = active
        self._eval_stop_btn.disabled = not active
        self._eval_miss_btn.disabled = not active
        self._eval_overlay.visible = bool(active or total)
        if self._auto_exec_switch.value != self._controller.auto_execute:
            self._auto_exec_switch.value = self._controller.auto_execute

        for control in (
            self._eval_overlay,
            self._eval_overlay_title,
            self._eval_progress_text,
            self._eval_correct_text,
            self._eval_wrong_text,
            self._eval_missed_text,
            self._eval_accuracy_text,
            self._eval_last_text,
            self._eval_progress_bar,
            self._eval_miss_btn,
            self._eval_expected,
            self._eval_attempts,
            self._eval_timeout,
            self._eval_threshold,
            self._model_mode_dd,
            self._model_variant_dd,
            self._dynamic_profile_dd,
            self._static_rejection_dd,
            self._eval_start_btn,
            self._eval_stop_btn,
            self._auto_exec_switch,
        ):
            try:
                control.update()
            except Exception:
                pass

    # ---- Сборка дерева ---------------------------------------------------

    def _panel_title(
        self,
        icon: str,
        title: str,
        *,
        color: str = COLOR_ACCENT,
        trailing: ft.Control | None = None,
    ) -> ft.Row:
        controls: list[ft.Control] = [
            ft.Container(
                width=30,
                height=30,
                border_radius=8,
                bgcolor="#1A1E22",
                alignment=ft.Alignment.CENTER,
                content=ft.Icon(icon, size=17, color=color),
            ),
            ft.Text(
                title,
                size=14,
                weight=ft.FontWeight.W_600,
                color=COLOR_ON_SURFACE,
                expand=True,
            ),
        ]
        if trailing is not None:
            controls.append(trailing)
        return ft.Row(
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=controls,
        )

    def _status_chip(self, icon: str, label: str, color: str) -> ft.Container:
        return ft.Container(
            border_radius=8,
            bgcolor="#171A1D",
            padding=ft.Padding.symmetric(horizontal=10, vertical=7),
            content=ft.Row(
                spacing=7,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Icon(icon, size=14, color=color),
                    ft.Text(label, size=12, color=COLOR_ON_SURFACE, no_wrap=True),
                ],
            ),
        )

    def _compact_switches(self, controls: list[ft.Switch]) -> ft.Column:
        left = controls[::2]
        right = controls[1::2]
        return ft.Column(
            spacing=4,
            controls=[
                ft.Row(
                    spacing=6,
                    wrap=True,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=left,
                ),
                ft.Row(
                    spacing=6,
                    wrap=True,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=right,
                ),
            ],
        )

    def build(self) -> ft.Control:
        title_group = ft.Row(
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Container(
                    width=36,
                    height=36,
                    border_radius=8,
                    bgcolor="#1A1E22",
                    alignment=ft.Alignment.CENTER,
                    content=ft.Icon(ft.Icons.VIDEO_CAMERA_FRONT, color=COLOR_ACCENT, size=20),
                ),
                ft.Column(
                    spacing=1,
                    expand=True,
                    controls=[
                        ft.Text(
                            "Live Monitor",
                            size=19,
                            weight=ft.FontWeight.BOLD,
                            color=COLOR_ON_SURFACE,
                        ),
                        ft.Text(
                            "камера, распознавание, команды",
                            size=12,
                            color=COLOR_MUTED,
                        ),
                    ],
                ),
            ],
        )
        monitor_chips = ft.Row(
            spacing=8,
            wrap=True,
            controls=[
                self._status_chip(ft.Icons.PAN_TOOL_ALT, "жесты", COLOR_ACCENT),
                self._status_chip(ft.Icons.TOUCH_APP, "курсор", COLOR_WARNING),
                self._status_chip(ft.Icons.ROCKET_LAUNCH, "команды", COLOR_SUCCESS),
            ],
        )
        header = surface_card(
            ft.ResponsiveRow(
                spacing=10,
                run_spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Container(content=title_group, col={"xs": 12, "md": 4}),
                    ft.Container(content=monitor_chips, col={"xs": 12, "md": 4}),
                    ft.Container(
                        content=ft.Row(
                            spacing=12,
                            alignment=ft.MainAxisAlignment.END,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[
                                self._toggle_btn,
                                ft.Container(
                                    content=self._status_text,
                                    alignment=ft.Alignment.CENTER_RIGHT,
                                    width=210,
                                ),
                            ],
                        ),
                        col={"xs": 12, "md": 4},
                    ),
                ],
            ),
            padding=14,
            radius=8,
        )

        camera_stage = ft.Container(
            content=ft.Stack(
                expand=True,
                controls=[
                    ft.Container(expand=True, bgcolor="#07090B"),
                    ft.Container(
                        content=self._camera_image,
                        expand=True,
                        alignment=ft.Alignment.CENTER,
                    ),
                    ft.Container(
                        content=self._eval_overlay,
                        left=18,
                        top=18,
                    ),
                    ft.Container(
                        left=16,
                        bottom=16,
                        bgcolor="#101316",
                        border_radius=8,
                        padding=ft.Padding.symmetric(horizontal=10, vertical=7),
                        content=ft.Row(
                            spacing=8,
                            tight=True,
                            controls=[
                                ft.Icon(ft.Icons.CENTER_FOCUS_STRONG, size=14, color=COLOR_MUTED),
                                ft.Text("preview", size=12, color=COLOR_MUTED),
                            ],
                        ),
                    ),
                ],
            ),
            bgcolor="#0B0D10",
            border_radius=8,
            padding=8,
            height=340,
            alignment=ft.Alignment.CENTER,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
        camera_card = surface_card(
            ft.Column(
                spacing=12,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    self._panel_title(ft.Icons.VIDEO_CAMERA_FRONT, "Камера"),
                    camera_stage,
                ],
            ),
            padding=14,
            radius=8,
        )
        camera_card.width = float("inf")

        live_panel = surface_card(
            ft.Column(
                spacing=5,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    self._panel_title(
                        ft.Icons.RADAR,
                        "Live Recognition",
                        trailing=self._confidence_text,
                    ),
                    self._gesture_state_text,
                    self._gesture_text,
                    self._confidence_bar,
                    ft.Container(
                        bgcolor="#171A1D",
                        border_radius=8,
                        padding=ft.Padding.symmetric(horizontal=10, vertical=5),
                        content=ft.Row(
                            spacing=9,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[
                                ft.Icon(ft.Icons.TERMINAL, size=15, color=COLOR_SUCCESS),
                                ft.Column(
                                    spacing=2,
                                    expand=True,
                                    controls=[
                                        ft.Text("Команда", size=11, color=COLOR_MUTED),
                                        self._command_text,
                                    ],
                                ),
                            ],
                        ),
                    ),
                ],
            ),
            padding=8,
            radius=8,
        )
        live_panel.width = float("inf")

        inspector_panel = surface_card(
            ft.Column(
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    self._panel_title(
                        ft.Icons.QUERY_STATS,
                        "Recognition Inspector",
                        color=COLOR_ACCENT,
                        trailing=ft.Row(
                            spacing=0,
                            tight=True,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[
                                self._recognition_inspector_status_text,
                                self._recognition_inspector_pause_btn,
                                self._recognition_inspector_step_btn,
                                self._recognition_inspector_clear_btn,
                                self._recognition_inspector_export_jsonl_btn,
                                self._recognition_inspector_export_csv_btn,
                            ],
                        ),
                    ),
                    ft.Container(
                        content=self._recognition_inspector_list,
                        height=162,
                    ),
                ],
            ),
            padding=10,
            radius=8,
        )
        inspector_panel.width = float("inf")

        command_panel = surface_card(
            ft.Column(
                spacing=9,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    self._panel_title(
                        ft.Icons.FORMAT_LIST_BULLETED,
                        "Recent Actions",
                        color=COLOR_SUCCESS,
                    ),
                    ft.Container(content=self._activity_list, height=96),
                ],
            ),
            padding=12,
            radius=8,
        )
        command_panel.width = float("inf")

        quick_panel = surface_card(
            ft.Column(
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    self._panel_title(
                        ft.Icons.TUNE,
                        "Quick Settings",
                        color=COLOR_WARNING,
                    ),
                    ft.Row(
                        spacing=10,
                        wrap=True,
                        controls=[
                            self._model_mode_dd,
                            self._model_variant_dd,
                            self._dynamic_profile_dd,
                            self._static_rejection_dd,
                        ],
                    ),
                    self._compact_switches(
                        [
                            self._gesture_switch,
                            self._landmarks_switch,
                            self._auto_exec_switch,
                            self._pointer_switch,
                            self._two_hands_switch,
                        ],
                    ),
                    ft.Container(
                        bgcolor="#171A1D",
                        border_radius=8,
                        padding=ft.Padding.symmetric(horizontal=10, vertical=7),
                        content=ft.Row(
                            spacing=8,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[
                                ft.Icon(
                                    ft.Icons.TOUCH_APP,
                                    size=15,
                                    color=COLOR_WARNING,
                                ),
                                self._pointer_state_text,
                            ],
                        ),
                    ),
                ],
            ),
            padding=10,
            radius=8,
        )
        quick_panel.width = float("inf")

        eval_panel = surface_card(
            ft.Column(
                spacing=12,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    self._panel_title(
                        ft.Icons.FACT_CHECK,
                        "Live Evaluation",
                        color=COLOR_ACCENT,
                    ),
                    ft.Row(
                        spacing=12,
                        wrap=True,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            self._eval_expected,
                            self._eval_attempts,
                            self._eval_timeout,
                            self._eval_threshold,
                            self._eval_start_btn,
                            self._eval_stop_btn,
                        ],
                    ),
                ],
            ),
            padding=14,
            radius=8,
        )
        eval_panel.width = float("inf")

        main_row = ft.ResponsiveRow(
            spacing=14,
            run_spacing=14,
            controls=[
                ft.Container(content=camera_card, col={"xs": 12, "lg": 8}),
                ft.Container(
                    content=ft.Column(
                        spacing=14,
                        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                        controls=[
                            live_panel,
                            inspector_panel,
                            quick_panel,
                            command_panel,
                        ],
                    ),
                    col={"xs": 12, "lg": 4},
                ),
            ],
        )

        return ft.Column(
            spacing=14,
            expand=True,
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
                header,
                main_row,
                eval_panel,
            ],
        )
