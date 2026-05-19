"""
Экран «Настройки» для Flet-версии DPLM.

Технические настройки пишутся в ``~/.dplm/config.json``. Политика привязок
R4/R5/R6 остаётся в БД, потому что это доменная настройка жестов и команд.
"""
from __future__ import annotations

from typing import Any

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


class SettingsView:
    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller

        config = controller.get_app_config()
        db = config["database"]
        paths = config["paths"]
        recognition = config["recognition"]

        self._system_status = ft.Column(spacing=8)
        self._env_warning = ft.Text("", size=12, color="#ffca28", visible=False)
        self._tech_status = ft.Text("", size=13, color=COLOR_SUCCESS, visible=False)

        self._db_backend = ft.Dropdown(
            label="Тип БД",
            value=str(db.get("backend") or "sqlite"),
            border_color=COLOR_SURFACE_HIGH,
            editable=False,
            options=[
                ft.DropdownOption(key="sqlite", text="SQLite"),
                ft.DropdownOption(key="postgres", text="PostgreSQL"),
            ],
        )
        self._db_host = self._field("Host", db.get("host"))
        self._db_port = self._field("Port", db.get("port"))
        self._db_user = self._field("User", db.get("user"))
        self._db_password = self._field(
            "Password", db.get("password"), password=True
        )
        self._db_name = self._field("Database", db.get("name"))
        self._db_sqlite_path = self._field("SQLite path", db.get("sqlite_path"))
        self._db_url = self._field("DATABASE_URL", db.get("url"))

        self._data_dir = self._field("data_dir", paths.get("data_dir"))
        self._models_dir = self._field("models_dir", paths.get("models_dir"))
        self._model_path = self._field("model_path", paths.get("model_path"))
        self._classes_path = self._field("classes_path", paths.get("classes_path"))
        self._feature_dim_path = self._field(
            "feature_dim_path", paths.get("feature_dim_path")
        )
        self._log_dir = self._field("log_dir", paths.get("log_dir"))

        self._camera_index = self._field("Camera index", recognition.get("camera_index"))
        self._target_fps = self._field("FPS", recognition.get("target_fps"))
        self._two_hands_switch = ft.Switch(
            value=bool(recognition.get("two_hands_mode")),
            label="Режим двух рук",
            active_color=COLOR_ACCENT,
        )
        self._auto_execute_switch = ft.Switch(
            value=bool(recognition.get("auto_execute_on_gesture")),
            label="Авто-выполнение команд по жесту",
            active_color=COLOR_ACCENT,
        )
        self._auto_start_switch = ft.Switch(
            value=bool(recognition.get("auto_start_recognition")),
            label="Автозапуск распознавания при старте приложения",
            active_color=COLOR_ACCENT,
        )

        self._recognition_btn = ft.FilledButton(
            content=ft.Text(self._recognition_btn_text(), weight=ft.FontWeight.BOLD),
            icon=self._recognition_btn_icon(),
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=20, vertical=12),
            ),
            on_click=self._on_toggle_recognition,
        )
        self._refresh_btn = ft.OutlinedButton(
            content=ft.Text("Обновить статус"),
            icon=ft.Icons.REFRESH,
            on_click=lambda _e: self._refresh_status(update=True),
        )
        self._save_tech_btn = ft.FilledButton(
            content=ft.Text("Сохранить технические настройки", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.SAVE,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=20, vertical=12),
            ),
            on_click=self._on_save_tech,
        )

        self._threshold_slider = ft.Slider(
            value=0.65,
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
            value=1500,
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
            value=True,
            label="Предупреждать о привязке опасных действий к одноручным жестам (R6)",
            active_color=COLOR_ACCENT,
        )
        self._policy_status = ft.Text("", color=COLOR_SUCCESS, size=13, visible=False)
        self._save_policy_btn = ft.FilledButton(
            content=ft.Text("Сохранить политику привязок", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.SAVE,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=20, vertical=12),
            ),
            on_click=self._on_save_policy,
        )

    def on_show(self) -> None:
        self._refresh_config_fields()
        self._refresh_policy()
        self._refresh_status(update=False)
        self._safe_update(self._page)

    def on_hide(self) -> None:
        pass

    def _field(self, label: str, value: Any = "", *, password: bool = False) -> ft.TextField:
        return ft.TextField(
            label=label,
            value="" if value is None else str(value),
            border_color=COLOR_SURFACE_HIGH,
            password=password,
            can_reveal_password=password,
            text_size=13,
        )

    def _responsive(self, *controls: ft.Control) -> ft.ResponsiveRow:
        return ft.ResponsiveRow(
            spacing=12,
            run_spacing=12,
            controls=[
                ft.Container(content=control, col={"xs": 12, "md": 6})
                for control in controls
            ],
        )

    def _section_title(self, icon: str, text: str) -> ft.Row:
        return ft.Row(
            controls=[
                ft.Icon(icon, color=COLOR_ACCENT, size=22),
                ft.Text(text, size=15, weight=ft.FontWeight.W_600, color=COLOR_ON_SURFACE),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _status_line(self, label: str, value: str, ok: bool | None = None) -> ft.Row:
        color = COLOR_MUTED
        icon = ft.Icons.INFO_OUTLINE
        if ok is True:
            color = COLOR_SUCCESS
            icon = ft.Icons.CHECK_CIRCLE
        elif ok is False:
            color = COLOR_DANGER
            icon = ft.Icons.ERROR_OUTLINE
        return ft.Row(
            controls=[
                ft.Icon(icon, color=color, size=18),
                ft.Text(label, color=COLOR_ON_SURFACE, size=12, width=130),
                ft.Text(value, color=color, size=12, selectable=True, expand=True),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

    def _refresh_config_fields(self) -> None:
        config = self._controller.get_app_config()
        db = config["database"]
        paths = config["paths"]
        recognition = config["recognition"]

        self._db_backend.value = str(db.get("backend") or "sqlite")
        self._db_host.value = str(db.get("host") or "")
        self._db_port.value = str(db.get("port") or "")
        self._db_user.value = str(db.get("user") or "")
        self._db_password.value = str(db.get("password") or "")
        self._db_name.value = str(db.get("name") or "")
        self._db_sqlite_path.value = str(db.get("sqlite_path") or "")
        self._db_url.value = str(db.get("url") or "")

        self._data_dir.value = str(paths.get("data_dir") or "")
        self._models_dir.value = str(paths.get("models_dir") or "")
        self._model_path.value = str(paths.get("model_path") or "")
        self._classes_path.value = str(paths.get("classes_path") or "")
        self._feature_dim_path.value = str(paths.get("feature_dim_path") or "")
        self._log_dir.value = str(paths.get("log_dir") or "")

        self._camera_index.value = str(recognition.get("camera_index") or 0)
        self._target_fps.value = str(recognition.get("target_fps") or 30)
        self._two_hands_switch.value = bool(recognition.get("two_hands_mode"))
        self._auto_execute_switch.value = bool(
            recognition.get("auto_execute_on_gesture")
        )
        self._auto_start_switch.value = bool(recognition.get("auto_start_recognition"))

    def _refresh_status(self, *, update: bool) -> None:
        status = self._controller.get_system_status()
        env = status.get("envOverrides") or {}
        if env:
            pairs = ", ".join(f"{k} → {v}" for k, v in env.items())
            self._env_warning.value = f"Значения из окружения сейчас сильнее config.json: {pairs}"
            self._env_warning.visible = True
        else:
            self._env_warning.value = ""
            self._env_warning.visible = False

        validation_errors = status.get("validationErrors") or []
        validation_warnings = status.get("validationWarnings") or []
        controls: list[ft.Control] = [
            self._status_line(
                "Конфиг",
                str(status["configPath"]),
                bool(status["configExists"]),
            ),
            self._status_line(
                "База данных",
                "OK" if status["databaseOk"] else (status["databaseError"] or "не проверена"),
                bool(status["databaseOk"]),
            ),
            self._status_line("DB URL", str(status["databaseUrl"]), None),
            self._status_line(
                "Камера",
                f"index={status['cameraIndex']}, fps={status['targetFps']}, OpenCV={'OK' if status['cv2Available'] else 'нет'}",
                bool(status["cv2Available"]),
            ),
            self._status_line(
                "Data",
                str(status["dataDir"]),
                bool(status["dataDirExists"]),
            ),
            self._status_line(
                "Models",
                str(status["modelsDir"]),
                bool(status["modelsDirExists"]),
            ),
            self._status_line(
                "model_path",
                str(status["modelPath"]),
                bool(status["modelExists"]),
            ),
            self._status_line(
                "classes_path",
                str(status["classesPath"]),
                bool(status["classesExists"]),
            ),
            self._status_line(
                "feature_dim",
                str(status["featureDimPath"]),
                bool(status["featureDimExists"]),
            ),
            self._status_line(
                "Логи",
                str(status["recognitionLog"]),
                bool(status["logDirExists"]),
            ),
            self._status_line(
                "Голос",
                "временно отключён" if not status["voiceAvailable"] else "доступен",
                False if not status["voiceAvailable"] else True,
            ),
        ]
        for item in validation_errors:
            controls.append(self._status_line("Ошибка", str(item), False))
        for item in validation_warnings:
            controls.append(self._status_line("Предупреждение", str(item), None))

        self._system_status.controls = controls
        self._recognition_btn.content.value = self._recognition_btn_text()
        self._recognition_btn.icon = self._recognition_btn_icon()
        if update:
            self._safe_update(self._page)

    def _refresh_policy(self) -> None:
        policy = self._controller.get_binding_policy()
        self._threshold_slider.value = float(policy["confidenceThreshold"])
        self._cooldown_slider.value = float(policy["cooldownMs"])
        self._warn_switch.value = bool(policy["warnTwoHands"])
        self._threshold_label.value = self._threshold_text()
        self._cooldown_label.value = self._cooldown_text()

    def _payload_from_fields(self) -> dict[str, Any]:
        return {
            "database": {
                "backend": self._db_backend.value or "sqlite",
                "host": self._db_host.value or "",
                "port": self._to_int(self._db_port.value, 5432),
                "user": self._db_user.value or "",
                "password": self._db_password.value or "",
                "name": self._db_name.value or "",
                "sqlite_path": self._db_sqlite_path.value or "",
                "url": self._db_url.value or "",
            },
            "paths": {
                "data_dir": self._data_dir.value or "",
                "models_dir": self._models_dir.value or "",
                "model_path": self._model_path.value or "",
                "classes_path": self._classes_path.value or "",
                "feature_dim_path": self._feature_dim_path.value or "",
                "log_dir": self._log_dir.value or "",
            },
            "recognition": {
                "camera_index": self._to_int(self._camera_index.value, 0),
                "target_fps": self._to_int(self._target_fps.value, 30),
                "two_hands_mode": bool(self._two_hands_switch.value),
                "auto_execute_on_gesture": bool(self._auto_execute_switch.value),
                "auto_start_recognition": bool(self._auto_start_switch.value),
            },
            "assistant": {
                "voice_enabled": False,
            },
        }

    def _to_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _on_save_tech(self, _e) -> None:
        ok, errors, warnings = self._controller.save_app_config(
            self._payload_from_fields()
        )
        if ok:
            text = f"Сохранено: {self._controller.config_path}"
            if warnings:
                text += " · " + " · ".join(warnings)
            self._tech_status.value = text
            self._tech_status.color = COLOR_SUCCESS
        else:
            self._tech_status.value = "Не сохранено: " + "; ".join(errors)
            self._tech_status.color = COLOR_DANGER
        self._tech_status.visible = True
        self._refresh_status(update=True)

    def _on_toggle_recognition(self, _e) -> None:
        self._controller.toggle_recognition()
        self._refresh_status(update=True)

    def _recognition_btn_text(self) -> str:
        return "Остановить распознавание" if self._controller.is_recognizing else "Запустить распознавание"

    def _recognition_btn_icon(self) -> str:
        return ft.Icons.STOP if self._controller.is_recognizing else ft.Icons.PLAY_ARROW

    def _threshold_text(self) -> str:
        v = float(self._threshold_slider.value)
        return f"Порог уверенности (R4): {int(round(v * 100))}%"

    def _cooldown_text(self) -> str:
        v = int(self._cooldown_slider.value)
        return f"Cooldown между срабатываниями (R5): {v} мс"

    def _on_threshold_change(self, _e) -> None:
        self._threshold_label.value = self._threshold_text()
        self._safe_update(self._threshold_label)

    def _on_cooldown_change(self, _e) -> None:
        self._cooldown_label.value = self._cooldown_text()
        self._safe_update(self._cooldown_label)

    def _on_save_policy(self, _e) -> None:
        ok = self._controller.set_binding_policy(
            confidence_threshold=float(self._threshold_slider.value),
            cooldown_ms=int(self._cooldown_slider.value),
            warn_two_hands=bool(self._warn_switch.value),
        )
        self._policy_status.value = (
            "Сохранено и применено к политике распознавания"
            if ok
            else "Не удалось сохранить — БД недоступна"
        )
        self._policy_status.color = COLOR_SUCCESS if ok else COLOR_DANGER
        self._policy_status.visible = True
        self._safe_update(self._policy_status)

    def _safe_update(self, control: Any) -> None:
        try:
            control.update()
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

        system_card = surface_card(
            ft.Column(
                spacing=14,
                controls=[
                    self._section_title(ft.Icons.MONITOR_HEART, "Статус системы"),
                    self._env_warning,
                    self._system_status,
                    ft.Row(
                        controls=[self._refresh_btn, self._recognition_btn],
                        spacing=10,
                        wrap=True,
                    ),
                ],
            ),
            padding=20,
            radius=16,
        )

        db_card = surface_card(
            ft.Column(
                spacing=14,
                controls=[
                    self._section_title(ft.Icons.STORAGE, "Подключение к БД"),
                    self._responsive(self._db_backend, self._db_url),
                    self._responsive(self._db_host, self._db_port),
                    self._responsive(self._db_user, self._db_password),
                    self._responsive(self._db_name, self._db_sqlite_path),
                ],
            ),
            padding=20,
            radius=16,
        )

        paths_card = surface_card(
            ft.Column(
                spacing=14,
                controls=[
                    self._section_title(ft.Icons.FOLDER_OPEN, "Пути"),
                    self._responsive(self._data_dir, self._models_dir),
                    self._responsive(self._model_path, self._classes_path),
                    self._responsive(self._feature_dim_path, self._log_dir),
                ],
            ),
            padding=20,
            radius=16,
        )

        recognition_card = surface_card(
            ft.Column(
                spacing=14,
                controls=[
                    self._section_title(ft.Icons.VIDEOCAM, "Распознавание"),
                    self._responsive(self._camera_index, self._target_fps),
                    self._two_hands_switch,
                    self._auto_execute_switch,
                    self._auto_start_switch,
                    ft.Divider(color=COLOR_SURFACE_HIGH, thickness=1),
                    ft.Text(
                        "Голосовой ассистент временно отключён; точка подключения сохранена для следующего этапа.",
                        size=12,
                        color=COLOR_MUTED,
                    ),
                    ft.Row(
                        controls=[self._save_tech_btn],
                        spacing=10,
                        wrap=True,
                    ),
                    self._tech_status,
                ],
            ),
            padding=20,
            radius=16,
        )

        policy_card = surface_card(
            ft.Column(
                spacing=14,
                controls=[
                    self._section_title(ft.Icons.RULE, "Политика привязок"),
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
                    self._save_policy_btn,
                    self._policy_status,
                ],
            ),
            padding=20,
            radius=16,
        )

        return ft.Column(
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            controls=[
                surface_card(header, padding=16, radius=16),
                system_card,
                db_card,
                paths_card,
                recognition_card,
                policy_card,
            ],
        )
