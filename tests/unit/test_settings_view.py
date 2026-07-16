# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

from app.flet_app.views.settings import SettingsView


class _Controller:
    model_variant = "production"
    config_path = "/tmp/dplm/config.json"

    def __init__(self) -> None:
        self.saved_config: dict[str, Any] | None = None
        self.saved_policy: dict[str, Any] | None = None
        self.saved_mas: dict[str, str] | None = None
        self.is_recognizing = False
        self._mas_key = False
        self._mas_provider = "local"
        self._mas_model = ""
        self._mas_api_url = ""

    def get_app_config(self) -> dict[str, Any]:
        return {
            "database": {
                "backend": "sqlite",
                "sqlite_path": "/tmp/dplm/app.db",
                "url": "",
            },
            "paths": {
                "data_dir": "/tmp/dplm/data",
                "models_dir": "/tmp/dplm/models",
                "model_path": "/tmp/dplm/models/model.pkl",
                "classes_path": "/tmp/dplm/models/classes.json",
                "feature_dim_path": "/tmp/dplm/models/feature_dim.txt",
                "log_dir": "/tmp/dplm/logs",
            },
            "recognition": {
                "camera_index": 0,
                "target_fps": 30,
                "two_hands_mode": True,
                "auto_execute_on_gesture": False,
                "auto_start_recognition": False,
                "pointer_smoothing": 0.55,
            },
            "assistant": {"voice_enabled": False},
            "telemetry": {
                "enabled": False,
                "endpoint": "",
                "project_key": "",
                "interval_hours": 24,
            },
            "llm": {
                "provider": self._mas_provider,
                "model": self._mas_model,
                "api_url": self._mas_api_url,
            },
        }

    def get_mas_llm_settings(self) -> dict[str, Any]:
        return {
            "provider": self._mas_provider,
            "model": self._mas_model,
            "apiUrl": self._mas_api_url,
            "providerLabel": "Mistral" if self._mas_provider == "mistral" else "Локальный агент",
            "hasApiKey": self._mas_key,
            "keySource": "keychain" if self._mas_key else "none",
            "keyEnvironment": "",
            "requiresApiKey": self._mas_provider not in {"local", "ollama", "custom"},
            "credentialError": "",
        }

    def save_mas_llm_settings(
        self,
        *,
        provider: str,
        model: str,
        api_url: str,
        api_key: str,
    ):
        self._mas_provider = provider
        self._mas_model = model
        self._mas_api_url = api_url
        self._mas_key = self._mas_key or bool(api_key)
        self.saved_mas = {
            "provider": provider,
            "model": model,
            "api_url": api_url,
            "api_key": api_key,
        }
        return True, []

    def delete_mas_llm_api_key(self):
        self._mas_key = False
        return True, "API-ключ удалён"

    def test_mas_llm_connection(self):
        return True, "Подключение работает"

    def get_usage_telemetry_status(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "configured": True,
            "last_success_at": 0.0,
            "last_error": "",
        }

    def send_usage_telemetry_now(self) -> bool:
        return True

    def get_binding_policy(self) -> dict[str, Any]:
        return {
            "confidenceThreshold": 0.72,
            "cooldownMs": 1600,
            "warnTwoHands": True,
        }

    def get_system_status(self) -> dict[str, Any]:
        return {
            "envOverrides": {},
            "validationErrors": [],
            "validationWarnings": [],
            "cameraIndex": 0,
            "targetFps": 30,
            "cv2Available": True,
            "modelExists": True,
            "classesExists": True,
            "featureDimExists": True,
            "databaseOk": True,
            "databaseError": "",
            "voiceAvailable": False,
        }

    def list_model_variants(self) -> list[dict[str, str]]:
        return [{"key": "production", "label": "Production"}]

    def apply_model_variant(self, _variant: str):
        return True, [], []

    def save_app_config(self, payload: dict[str, Any]):
        self.saved_config = payload
        return True, [], []

    def set_binding_policy(
        self,
        *,
        confidence_threshold: float,
        cooldown_ms: int,
        warn_two_hands: bool,
    ) -> bool:
        self.saved_policy = {
            "confidence_threshold": confidence_threshold,
            "cooldown_ms": cooldown_ms,
            "warn_two_hands": warn_two_hands,
        }
        return True

    def toggle_recognition(self) -> None:
        self.is_recognizing = not self.is_recognizing

    def run_self_test(self, *, check_camera: bool):
        return {"summary": {}, "items": []}


def _visible_text(control: Any) -> str:
    parts: list[str] = []
    seen: set[int] = set()

    def visit(obj: Any) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(obj, (str, int, float, bool)):
            parts.append(str(obj))
            return

        for attr in ("value", "label", "text", "tooltip", "key"):
            value = getattr(obj, attr, None)
            if isinstance(value, (str, int, float, bool)):
                parts.append(str(value))

        visit(getattr(obj, "content", None))

        for attr in ("controls", "options"):
            for child in getattr(obj, attr, []) or []:
                visit(child)

    visit(control)
    return "\n".join(parts)


def test_settings_screen_contains_only_user_level_sections():
    view = SettingsView(object(), _Controller())
    control = view.build()

    text = _visible_text(control)

    for expected in (
        "Настройки",
        "Состояние",
        "Камера и жесты",
        "Команды",
        "MAS и LLM",
        "LLM-провайдер",
        "OpenAI",
        "Anthropic Claude",
        "Google Gemini",
        "Mistral",
        "OpenRouter",
        "Свой OpenAI-совместимый API",
        "API endpoint",
        "API-ключ",
        "Приватность",
        "Отправлять анонимную статистику качества раз в день",
        "Сохранить настройки",
    ):
        assert expected in text

    for technical in (
        "Самопроверка",
        "Подключение к БД",
        "DATABASE_URL",
        "Пути",
        "Набор моделей",
        "model_path",
        "Порог уверенности",
        "Cooldown",
        "Сохранить технические настройки",
        "Политика привязок",
    ):
        assert technical not in text


def test_saving_user_settings_preserves_hidden_technical_config():
    controller = _Controller()
    view = SettingsView(object(), controller)
    view._refresh_policy()
    view._camera_index.value = "2"
    view._target_fps.value = "24"
    view._two_hands_switch.value = False
    view._auto_execute_switch.value = True
    view._auto_start_switch.value = True
    view._warn_switch.value = False

    view._on_save_tech(None)

    assert controller.saved_config is not None
    assert controller.saved_config["database"]["sqlite_path"] == "/tmp/dplm/app.db"
    assert controller.saved_config["paths"]["model_path"] == "/tmp/dplm/models/model.pkl"
    assert controller.saved_config["recognition"] == {
        "camera_index": 2,
        "target_fps": 24,
        "two_hands_mode": False,
        "auto_execute_on_gesture": True,
        "auto_start_recognition": True,
        "pointer_smoothing": 0.55,
    }
    assert controller.saved_config["telemetry"]["enabled"] is False
    assert controller.saved_config["llm"]["provider"] == "local"
    assert controller.saved_policy == {
        "confidence_threshold": 0.72,
        "cooldown_ms": 1600,
        "warn_two_hands": False,
    }


def test_user_can_save_mistral_model_and_hidden_api_key():
    controller = _Controller()
    view = SettingsView(object(), controller)
    view._llm_provider.value = "mistral"
    view._llm_model.value = "mistral-medium-latest"
    view._llm_api_url.value = "https://api.mistral.ai/v1/chat/completions"
    view._llm_api_key.value = "user-secret"

    view._on_save_mas_llm(None)

    assert controller.saved_mas == {
        "provider": "mistral",
        "model": "mistral-medium-latest",
        "api_url": "https://api.mistral.ai/v1/chat/completions",
        "api_key": "user-secret",
    }
    assert view._llm_api_key.value == ""
    assert view._llm_api_key.password is True
    assert view._llm_test_btn.disabled is False


def test_provider_selection_applies_openai_compatible_preset():
    view = SettingsView(object(), _Controller())

    view._llm_provider.value = "gemini"
    view._on_llm_provider_change(None)

    assert view._llm_model.value == "gemini-3.5-flash"
    assert view._llm_api_url.value == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert view._llm_model.disabled is False
    assert view._llm_api_url.disabled is False
    assert view._llm_test_btn.disabled is True
