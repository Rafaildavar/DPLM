"""Экран пользовательских настроек для Flet-версии GestureBind."""
from __future__ import annotations

from datetime import datetime
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
from app.services.llm_providers import (
    LLM_PROVIDER_PRESETS,
    get_llm_provider_preset,
    llm_provider_label,
)


class SettingsView:
    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller

        config = controller.get_app_config()
        db = config["database"]
        paths = config["paths"]
        recognition = config["recognition"]
        telemetry = dict(config.get("telemetry") or {})
        telemetry_status = self._telemetry_status_snapshot()
        llm_status = self._mas_llm_settings_snapshot(config)

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
        self._model_variant_dd = ft.Dropdown(
            label="Набор моделей",
            value=controller.model_variant,
            border_color=COLOR_SURFACE_HIGH,
            editable=False,
            options=self._model_variant_options(),
        )
        self._apply_model_variant_btn = ft.FilledButton(
            content=ft.Text("Применить набор", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.SWAP_HORIZ,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=20, vertical=12),
            ),
            on_click=self._on_apply_model_variant,
        )
        self._model_variant_status = ft.Text(
            "", size=13, color=COLOR_SUCCESS, visible=False
        )

        self._camera_index = self._field("Камера", recognition.get("camera_index"))
        self._target_fps = self._field("Частота камеры", recognition.get("target_fps"))
        self._two_hands_switch = ft.Switch(
            value=False,
            label="",
            active_color=COLOR_ACCENT,
            disabled=True,
            visible=False,
        )
        self._auto_execute_switch = ft.Switch(
            value=bool(recognition.get("auto_execute_on_gesture")),
            label="Выполнять команды по жесту",
            active_color=COLOR_ACCENT,
        )
        self._auto_start_switch = ft.Switch(
            value=bool(recognition.get("auto_start_recognition")),
            label="Запускать распознавание при старте",
            active_color=COLOR_ACCENT,
        )
        self._pointer_sharpness_slider = ft.Slider(
            value=self._to_float(recognition.get("pointer_smoothing"), 0.55),
            min=0.05,
            max=0.95,
            divisions=90,
            active_color=COLOR_ACCENT,
            inactive_color=COLOR_SURFACE_HIGH,
            on_change=self._on_pointer_sharpness_change,
        )
        self._pointer_sharpness_label = ft.Text(
            self._pointer_sharpness_text(), size=14, color=COLOR_ON_SURFACE
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
            content=ft.Text("Обновить"),
            icon=ft.Icons.REFRESH,
            on_click=lambda _e: self._refresh_status(update=True),
        )
        self._diagnostics_status = ft.Text("", size=13, color=COLOR_MUTED, visible=False)
        self._diagnostics_results = ft.Column(spacing=8)
        self._diagnostics_camera_switch = ft.Switch(
            value=False,
            label="Открывать камеру во время проверки",
            active_color=COLOR_ACCENT,
        )
        self._diagnostics_btn = ft.FilledButton(
            content=ft.Text("Запустить самопроверку", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.CHECK_CIRCLE,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=20, vertical=12),
            ),
            on_click=self._on_run_diagnostics,
        )
        self._save_tech_btn = ft.FilledButton(
            content=ft.Text("Сохранить настройки", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.SAVE,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=20, vertical=12),
            ),
            on_click=self._on_save_tech,
        )

        self._telemetry_switch = ft.Switch(
            value=bool(telemetry.get("enabled", False)),
            label="Отправлять анонимную статистику качества раз в день",
            active_color=COLOR_ACCENT,
            disabled=not bool(telemetry_status.get("configured")),
            on_change=self._on_telemetry_change,
        )
        self._telemetry_description = ft.Text(
            "Передаются только агрегаты распознавания и производительности. "
            "Кадры камеры, landmarks, названия жестов и команды остаются на устройстве.",
            size=12,
            color=COLOR_MUTED,
        )
        self._telemetry_status = ft.Text(
            self._telemetry_status_text(telemetry_status),
            size=12,
            color=COLOR_MUTED,
        )
        self._telemetry_send_btn = ft.OutlinedButton(
            content=ft.Text("Отправить сейчас"),
            icon=ft.Icons.CLOUD_UPLOAD_OUTLINED,
            disabled=not (
                bool(telemetry_status.get("configured"))
                and bool(telemetry.get("enabled", False))
            ),
            on_click=self._on_send_telemetry,
        )

        self._llm_provider = ft.Dropdown(
            label="LLM-провайдер",
            value=str(llm_status.get("provider") or "local"),
            border_color=COLOR_SURFACE_HIGH,
            editable=False,
            options=[
                ft.DropdownOption(key=item.key, text=item.label)
                for item in LLM_PROVIDER_PRESETS
            ],
            on_select=self._on_llm_provider_change,
        )
        self._llm_model = self._field(
            "Модель",
            llm_status.get("model") or "",
        )
        self._llm_api_url = self._field(
            "API endpoint",
            llm_status.get("apiUrl") or "",
        )
        self._llm_api_key = self._field("API-ключ", "", password=True)
        self._llm_model.on_change = self._on_llm_fields_change
        self._llm_api_url.on_change = self._on_llm_fields_change
        self._llm_api_key.on_change = self._on_llm_fields_change
        self._llm_api_key.hint_text = (
            "Ключ уже сохранён"
            if llm_status.get("hasApiKey")
            else "Введите ключ провайдера"
        )
        self._llm_status = ft.Text(
            self._mas_llm_status_text(llm_status),
            size=12,
            color=COLOR_MUTED,
        )
        self._llm_save_btn = ft.FilledButton(
            content=ft.Text("Сохранить", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.SAVE,
            style=ft.ButtonStyle(bgcolor=COLOR_ACCENT, color=ft.Colors.WHITE),
            on_click=self._on_save_mas_llm,
        )
        self._llm_test_btn = ft.OutlinedButton(
            content=ft.Text("Проверить"),
            icon=ft.Icons.CHECK_CIRCLE_OUTLINE,
            on_click=self._on_test_mas_llm,
        )
        self._llm_delete_btn = ft.IconButton(
            icon=ft.Icons.DELETE_OUTLINE,
            icon_color=COLOR_DANGER,
            tooltip="Удалить API-ключ",
            on_click=self._on_delete_mas_llm_key,
        )
        self._update_llm_controls(llm_status)

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
            label="Предупреждать об опасных действиях",
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
        telemetry = dict(config.get("telemetry") or {})
        llm_status = self._mas_llm_settings_snapshot(config)

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
        self._model_variant_dd.options = self._model_variant_options()
        self._model_variant_dd.value = self._controller.model_variant

        self._camera_index.value = str(recognition.get("camera_index") or 0)
        self._target_fps.value = str(recognition.get("target_fps") or 30)
        self._two_hands_switch.value = False
        self._auto_execute_switch.value = bool(
            recognition.get("auto_execute_on_gesture")
        )
        self._auto_start_switch.value = bool(recognition.get("auto_start_recognition"))
        self._pointer_sharpness_slider.value = self._to_float(
            recognition.get("pointer_smoothing"), 0.55
        )
        self._pointer_sharpness_label.value = self._pointer_sharpness_text()
        telemetry_status = self._telemetry_status_snapshot()
        self._telemetry_switch.value = bool(telemetry.get("enabled", False))
        self._telemetry_switch.disabled = not bool(
            telemetry_status.get("configured")
        )
        self._telemetry_send_btn.disabled = not (
            bool(telemetry_status.get("configured"))
            and bool(telemetry.get("enabled", False))
        )
        self._telemetry_status.value = self._telemetry_status_text(telemetry_status)
        self._llm_provider.value = str(llm_status.get("provider") or "local")
        self._llm_model.value = str(
            llm_status.get("model") or ""
        )
        self._llm_api_url.value = str(llm_status.get("apiUrl") or "")
        self._llm_api_key.value = ""
        self._llm_api_key.hint_text = (
            "Ключ уже сохранён"
            if llm_status.get("hasApiKey")
            else "Введите ключ провайдера"
        )
        self._llm_status.value = self._mas_llm_status_text(llm_status)
        self._llm_status.color = COLOR_MUTED
        self._update_llm_controls(llm_status)

    def _refresh_status(self, *, update: bool) -> None:
        status = self._controller.get_system_status()
        env = status.get("envOverrides") or {}
        if env:
            self._env_warning.value = (
                "Часть настроек задана окружением и может быть недоступна для изменения."
            )
            self._env_warning.visible = True
        else:
            self._env_warning.value = ""
            self._env_warning.visible = False

        validation_errors = status.get("validationErrors") or []
        validation_warnings = status.get("validationWarnings") or []
        model_ready = bool(
            status.get("modelExists")
            and status.get("classesExists")
            and status.get("featureDimExists")
        )
        controls: list[ft.Control] = [
            self._status_line(
                "Распознавание",
                "запущено" if self._controller.is_recognizing else "остановлено",
                True if self._controller.is_recognizing else None,
            ),
            self._status_line(
                "Камера",
                f"камера #{status['cameraIndex']}, {status['targetFps']} FPS"
                if status.get("cv2Available")
                else "камера недоступна",
                bool(status.get("cv2Available")),
            ),
            self._status_line(
                "Модель",
                "готова" if model_ready else "не найдена",
                model_ready,
            ),
            self._status_line(
                "База данных",
                "готова" if status.get("databaseOk") else "недоступна",
                bool(status.get("databaseOk")),
            ),
            self._status_line(
                "Голос",
                "доступен" if status.get("voiceAvailable") else "отключён",
                True if status.get("voiceAvailable") else None,
            ),
        ]
        if validation_errors:
            controls.append(
                self._status_line(
                    "Настройки", f"ошибок: {len(validation_errors)}", False
                )
            )
        elif validation_warnings:
            controls.append(
                self._status_line(
                    "Настройки", f"предупреждений: {len(validation_warnings)}", None
                )
            )

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
        config = self._controller.get_app_config()
        database = dict(config.get("database") or {})
        paths = dict(config.get("paths") or {})
        assistant = dict(config.get("assistant") or {})
        assistant["voice_enabled"] = bool(assistant.get("voice_enabled", False))
        telemetry = dict(config.get("telemetry") or {})
        telemetry["enabled"] = bool(self._telemetry_switch.value)
        llm = dict(config.get("llm") or {})
        return {
            "database": database,
            "paths": paths,
            "recognition": {
                "camera_index": self._to_int(self._camera_index.value, 0),
                "target_fps": self._to_int(self._target_fps.value, 30),
                "two_hands_mode": False,
                "auto_execute_on_gesture": bool(self._auto_execute_switch.value),
                "auto_start_recognition": bool(self._auto_start_switch.value),
                "pointer_smoothing": self._to_float(
                    self._pointer_sharpness_slider.value, 0.55
                ),
            },
            "assistant": assistant,
            "telemetry": telemetry,
            "llm": llm,
        }

    def _mas_llm_settings_snapshot(
        self, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        getter = getattr(self._controller, "get_mas_llm_settings", None)
        if callable(getter):
            try:
                value = getter()
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
        raw = config or self._controller.get_app_config()
        llm = dict(raw.get("llm") or {})
        provider = str(llm.get("provider") or "local")
        preset = get_llm_provider_preset(provider)
        return {
            "provider": provider,
            "model": str(llm.get("model") or ""),
            "apiUrl": str(llm.get("api_url") or ""),
            "providerLabel": llm_provider_label(llm.get("provider")),
            "hasApiKey": False,
            "keySource": "none",
            "keyEnvironment": "",
            "requiresApiKey": bool(preset and preset.requires_api_key),
            "credentialError": "",
        }

    @staticmethod
    def _mas_llm_status_text(status: dict[str, Any]) -> str:
        provider = str(status.get("provider") or "local")
        if provider == "local":
            return "Локальная обработка включена"
        label = str(status.get("providerLabel") or llm_provider_label(provider))
        error = str(status.get("credentialError") or "").strip()
        if error:
            return error
        if status.get("keySource") == "environment":
            env_name = str(status.get("keyEnvironment") or "окружение")
            return f"{label}: ключ задан через {env_name}"
        if status.get("hasApiKey"):
            return "API-ключ хранится в системном Keychain"
        if not status.get("requiresApiKey"):
            return "API-ключ необязателен для выбранного endpoint"
        return f"Для {label} нужен личный API-ключ"

    def _update_llm_controls(self, status: dict[str, Any] | None = None) -> None:
        current = status or self._mas_llm_settings_snapshot()
        external = str(self._llm_provider.value or "local") != "local"
        env_key = current.get("keySource") == "environment"
        has_key = bool(current.get("hasApiKey"))
        requires_key = bool(current.get("requiresApiKey"))
        self._llm_model.disabled = not external
        self._llm_api_url.disabled = not external
        self._llm_api_key.disabled = not external or env_key
        self._llm_test_btn.disabled = not external or (requires_key and not has_key)
        self._llm_delete_btn.disabled = not external or not has_key or env_key

    def _on_llm_provider_change(self, _event) -> None:
        provider = str(self._llm_provider.value or "local")
        preset = get_llm_provider_preset(provider)
        if preset is not None and provider != "local":
            self._llm_model.value = preset.default_model
            self._llm_api_url.value = preset.api_url
        self._llm_api_key.value = ""
        status = {
            "provider": provider,
            "providerLabel": llm_provider_label(provider),
            "hasApiKey": False,
            "keySource": "none",
            "requiresApiKey": bool(preset and preset.requires_api_key),
            "credentialError": "",
        }
        self._llm_status.value = self._mas_llm_status_text(status)
        self._llm_status.color = COLOR_MUTED
        self._update_llm_controls(status)
        self._llm_test_btn.disabled = True
        self._safe_update(self._page)

    def _on_llm_fields_change(self, _event) -> None:
        if str(self._llm_provider.value or "local") != "local":
            self._llm_test_btn.disabled = True
            self._llm_status.value = "Сохраните изменения перед проверкой"
            self._llm_status.color = COLOR_MUTED
            self._safe_update(self._page)

    def _on_save_mas_llm(self, _event) -> None:
        saver = getattr(self._controller, "save_mas_llm_settings", None)
        if not callable(saver):
            self._llm_status.value = "Настройка LLM недоступна"
            self._llm_status.color = COLOR_DANGER
            self._safe_update(self._page)
            return
        ok, errors = saver(
            provider=str(self._llm_provider.value or "local"),
            model=str(self._llm_model.value or "").strip(),
            api_url=str(self._llm_api_url.value or "").strip(),
            api_key=str(self._llm_api_key.value or "").strip(),
        )
        self._llm_api_key.value = ""
        if ok:
            status = self._mas_llm_settings_snapshot()
            self._llm_status.value = self._mas_llm_status_text(status)
            self._llm_status.color = COLOR_SUCCESS
            self._llm_api_key.hint_text = (
                "Ключ уже сохранён"
                if status.get("hasApiKey")
                else "Введите ключ провайдера"
            )
            self._update_llm_controls(status)
        else:
            self._llm_status.value = "Не сохранено: " + "; ".join(errors)
            self._llm_status.color = COLOR_DANGER
        self._safe_update(self._page)

    def _on_test_mas_llm(self, _event) -> None:
        self._llm_test_btn.disabled = True
        self._llm_status.value = "Проверяю подключение…"
        self._llm_status.color = COLOR_MUTED
        self._safe_update(self._page)
        runner = getattr(self._page, "run_thread", None)
        if callable(runner):
            runner(self._test_mas_llm)
        else:
            self._test_mas_llm()

    def _test_mas_llm(self) -> None:
        tester = getattr(self._controller, "test_mas_llm_connection", None)
        ok, message = (
            tester() if callable(tester) else (False, "Проверка LLM недоступна")
        )
        self._llm_status.value = str(message)
        self._llm_status.color = COLOR_SUCCESS if ok else COLOR_DANGER
        self._update_llm_controls(self._mas_llm_settings_snapshot())
        self._safe_update(self._page)

    def _on_delete_mas_llm_key(self, _event) -> None:
        deleter = getattr(self._controller, "delete_mas_llm_api_key", None)
        ok, message = (
            deleter() if callable(deleter) else (False, "Удаление ключа недоступно")
        )
        status = self._mas_llm_settings_snapshot()
        self._llm_api_key.value = ""
        self._llm_api_key.hint_text = "Введите ключ провайдера"
        self._llm_status.value = str(message)
        self._llm_status.color = COLOR_SUCCESS if ok else COLOR_DANGER
        self._update_llm_controls(status)
        self._safe_update(self._page)

    def _telemetry_status_snapshot(self) -> dict[str, Any]:
        getter = getattr(self._controller, "get_usage_telemetry_status", None)
        if callable(getter):
            try:
                value = getter()
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
        return {
            "enabled": False,
            "configured": False,
            "last_success_at": 0.0,
            "last_error": "",
        }

    @staticmethod
    def _telemetry_status_text(status: dict[str, Any]) -> str:
        error = str(status.get("last_error") or "").strip()
        if error:
            return "Последняя отправка не удалась; приложение повторит её автоматически"
        if not bool(status.get("configured")):
            return "Сервер статистики не настроен в этой сборке"
        last_success = float(status.get("last_success_at") or 0.0)
        if last_success > 0.0:
            stamp = datetime.fromtimestamp(last_success).strftime("%d.%m.%Y %H:%M")
            return f"Последняя отправка: {stamp}"
        if bool(status.get("enabled")):
            return "Статистика включена; первая отправка будет выполнена автоматически"
        return "Статистика выключена"

    def _on_telemetry_change(self, _event) -> None:
        config = self._controller.get_app_config()
        telemetry = dict(config.get("telemetry") or {})
        telemetry["enabled"] = bool(self._telemetry_switch.value)
        config["telemetry"] = telemetry
        ok, errors, _warnings = self._controller.save_app_config(config)
        if not ok:
            self._telemetry_switch.value = not bool(self._telemetry_switch.value)
            self._telemetry_status.value = "Не сохранено: " + "; ".join(errors)
            self._telemetry_status.color = COLOR_DANGER
        else:
            status = self._telemetry_status_snapshot()
            self._telemetry_status.value = self._telemetry_status_text(status)
            self._telemetry_status.color = COLOR_MUTED
            self._telemetry_send_btn.disabled = not (
                bool(status.get("configured"))
                and bool(self._telemetry_switch.value)
            )
        self._safe_update(self._page)

    def _on_send_telemetry(self, _event) -> None:
        self._telemetry_send_btn.disabled = True
        self._telemetry_status.value = "Отправка…"
        self._safe_update(self._page)
        runner = getattr(self._page, "run_thread", None)
        if callable(runner):
            runner(self._send_telemetry)
        else:
            self._send_telemetry()

    def _send_telemetry(self) -> None:
        sender = getattr(self._controller, "send_usage_telemetry_now", None)
        sent = False
        if callable(sender):
            try:
                sent = bool(sender())
            except Exception:
                sent = False
        status = self._telemetry_status_snapshot()
        self._telemetry_status.value = (
            self._telemetry_status_text(status)
            if sent
            else "Не удалось отправить; приложение повторит попытку автоматически"
        )
        self._telemetry_status.color = COLOR_MUTED if sent else COLOR_DANGER
        self._telemetry_send_btn.disabled = not (
            bool(status.get("configured"))
            and bool(self._telemetry_switch.value)
        )
        self._safe_update(self._page)

    def _to_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _to_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _on_save_tech(self, _e) -> None:
        ok, errors, warnings = self._controller.save_app_config(
            self._payload_from_fields()
        )
        policy_ok = self._controller.set_binding_policy(
            confidence_threshold=float(self._threshold_slider.value),
            cooldown_ms=int(self._cooldown_slider.value),
            warn_two_hands=bool(self._warn_switch.value),
        )
        if ok:
            text = "Настройки сохранены"
            if warnings:
                text += " · " + " · ".join(warnings)
            if not policy_ok:
                text += " · предупреждения команд не сохранены"
            self._tech_status.value = text
            self._tech_status.color = COLOR_SUCCESS if policy_ok else "#ffca28"
        else:
            self._tech_status.value = "Не сохранено: " + "; ".join(errors)
            self._tech_status.color = COLOR_DANGER
        self._tech_status.visible = True
        self._refresh_status(update=True)

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

    def _on_apply_model_variant(self, _e) -> None:
        ok, errors, warnings = self._controller.apply_model_variant(
            str(self._model_variant_dd.value or "production")
        )
        if ok:
            self._model_variant_status.value = "Набор моделей применен"
            if warnings:
                self._model_variant_status.value += " · " + " · ".join(warnings)
            self._model_variant_status.color = COLOR_SUCCESS
            self._refresh_config_fields()
        else:
            self._model_variant_status.value = "Не применено: " + "; ".join(errors)
            self._model_variant_status.color = COLOR_DANGER
        self._model_variant_status.visible = True
        self._refresh_status(update=True)

    def _on_toggle_recognition(self, _e) -> None:
        runner = getattr(self._page, "run_thread", None)
        if callable(runner):
            runner(self._controller.toggle_recognition)
        else:
            self._controller.toggle_recognition()
        self._refresh_status(update=True)

    def _on_run_diagnostics(self, _e) -> None:
        report = self._controller.run_self_test(
            check_camera=bool(self._diagnostics_camera_switch.value)
        )
        summary = report.get("summary") or {}
        failures = int(summary.get("fail", 0))
        warnings = int(summary.get("warn", 0))
        if failures:
            self._diagnostics_status.value = f"Самопроверка: ошибок {failures}, предупреждений {warnings}"
            self._diagnostics_status.color = COLOR_DANGER
        elif warnings:
            self._diagnostics_status.value = f"Самопроверка: OK, предупреждений {warnings}"
            self._diagnostics_status.color = "#ffca28"
        else:
            self._diagnostics_status.value = "Самопроверка: OK"
            self._diagnostics_status.color = COLOR_SUCCESS
        self._diagnostics_status.visible = True

        controls: list[ft.Control] = []
        for item in report.get("items") or []:
            status = str(item.get("status") or "")
            ok = True if status == "pass" else False if status == "fail" else None
            controls.append(
                self._status_line(
                    str(item.get("label") or item.get("key") or ""),
                    str(item.get("detail") or ""),
                    ok,
                )
            )
        self._diagnostics_results.controls = controls
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

    def _pointer_sharpness_text(self) -> str:
        v = float(self._pointer_sharpness_slider.value)
        return f"Резкость курсора: {int(round(v * 100))}%"

    def _on_pointer_sharpness_change(self, _e) -> None:
        self._pointer_sharpness_label.value = self._pointer_sharpness_text()
        self._safe_update(self._pointer_sharpness_label)

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

        status_card = surface_card(
            ft.Column(
                spacing=14,
                controls=[
                    self._section_title(ft.Icons.MONITOR_HEART, "Состояние"),
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

        recognition_card = surface_card(
            ft.Column(
                spacing=14,
                controls=[
                    self._section_title(ft.Icons.VIDEOCAM, "Камера и жесты"),
                    self._responsive(self._camera_index, self._target_fps),
                    self._auto_start_switch,
                    ft.Divider(color=COLOR_SURFACE_HIGH, thickness=1),
                    self._pointer_sharpness_label,
                    self._pointer_sharpness_slider,
                ],
            ),
            padding=20,
            radius=16,
        )

        commands_card = surface_card(
            ft.Column(
                spacing=14,
                controls=[
                    self._section_title(ft.Icons.TOUCH_APP, "Команды"),
                    self._auto_execute_switch,
                    self._warn_switch,
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

        llm_card = surface_card(
            ft.Column(
                spacing=14,
                controls=[
                    self._section_title(ft.Icons.PSYCHOLOGY, "MAS и LLM"),
                    self._llm_provider,
                    self._responsive(self._llm_model, self._llm_api_key),
                    self._llm_api_url,
                    ft.Row(
                        controls=[
                            self._llm_save_btn,
                            self._llm_test_btn,
                            self._llm_delete_btn,
                        ],
                        spacing=10,
                        wrap=True,
                    ),
                    self._llm_status,
                ],
            ),
            padding=20,
            radius=16,
        )

        privacy_card = surface_card(
            ft.Column(
                spacing=12,
                controls=[
                    self._section_title(ft.Icons.PRIVACY_TIP_OUTLINED, "Приватность"),
                    self._telemetry_switch,
                    self._telemetry_description,
                    self._telemetry_status,
                    ft.Row(controls=[self._telemetry_send_btn], wrap=True),
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
                status_card,
                recognition_card,
                commands_card,
                llm_card,
                privacy_card,
            ],
        )
