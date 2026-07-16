from __future__ import annotations

import json

from app.flet_app.controller import AppController
from app.services.app_config import AppConfig, ConfigStore
from app.services.mas_credentials import MasCredentialStore


class _MemoryKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, account: str):
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.values[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        self.values.pop((service, account), None)


def _controller(tmp_path) -> tuple[AppController, _MemoryKeyring]:
    store = ConfigStore(tmp_path / "config.json")
    config = AppConfig()
    store.save(config)
    backend = _MemoryKeyring()
    controller = AppController.__new__(AppController)
    controller._config_store = store
    controller._config_file = config
    controller._config = config
    controller._mas_credentials = MasCredentialStore(backend)
    controller._apply_runtime_config = lambda: None
    controller._reset_db_bridge = lambda: None
    return controller, backend


def test_controller_saves_mas_key_outside_config_file(tmp_path, monkeypatch):
    for key in (
        "MISTRAL_API_KEY",
        "MISTRAL_MODEL",
        "MISTRAL_API_URL",
        "DPLM_BINDING_AGENT_PROVIDER",
        "BINDING_AGENT_PROVIDER",
    ):
        monkeypatch.setenv(key, "")
    controller, _backend = _controller(tmp_path)
    secret = "user-mistral-secret"

    ok, errors = controller.save_mas_llm_settings(
        provider="mistral",
        model="mistral-medium-latest",
        api_key=secret,
    )

    assert ok, errors
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["llm"] == {
        "provider": "mistral",
        "model": "mistral-medium-latest",
        "api_url": "https://api.mistral.ai/v1/chat/completions",
    }
    assert secret not in json.dumps(saved)
    status = controller.get_mas_llm_settings()
    assert status["hasApiKey"] is True
    assert status["keySource"] == "keychain"
    assert "apiKey" not in status
    assert controller.get_binding_agent_provider_label() == (
        "Mistral · mistral-medium-latest"
    )


def test_controller_passes_saved_mas_runtime_to_agent(tmp_path, monkeypatch):
    for key in (
        "MISTRAL_API_KEY",
        "MISTRAL_MODEL",
        "MISTRAL_API_URL",
        "DPLM_BINDING_AGENT_PROVIDER",
        "BINDING_AGENT_PROVIDER",
    ):
        monkeypatch.setenv(key, "")
    controller, _backend = _controller(tmp_path)
    ok, errors = controller.save_mas_llm_settings(
        provider="mistral",
        model="mistral-medium-latest",
        api_key="runtime-secret",
    )
    assert ok, errors

    import app.services.binding_agent as binding_agent

    captured = {}

    class _Result:
        def to_legacy_draft(self):
            return {"ok": True}

    class _Orchestrator:
        def __init__(self, *, mistral_agent):
            captured["agent"] = mistral_agent

        def run(self, *_args, **kwargs):
            captured["provider"] = kwargs["provider"]
            return _Result()

    monkeypatch.setattr(binding_agent, "BindingAgentOrchestrator", _Orchestrator)

    result = controller.build_agent_binding_draft("test", [])

    assert result == {"ok": True}
    assert captured["provider"] == "mistral"
    assert captured["agent"].api_key == "runtime-secret"
    assert captured["agent"].model == "mistral-medium-latest"


def test_legacy_mistral_env_key_still_enables_cloud_provider(tmp_path, monkeypatch):
    controller, _backend = _controller(tmp_path)
    monkeypatch.setenv("MISTRAL_API_KEY", "environment-secret")
    monkeypatch.setenv("MISTRAL_MODEL", "")
    monkeypatch.setenv("MISTRAL_API_URL", "")
    monkeypatch.setenv("DPLM_BINDING_AGENT_PROVIDER", "")
    monkeypatch.setenv("BINDING_AGENT_PROVIDER", "")

    status = controller.get_mas_llm_settings()

    assert status["provider"] == "mistral"
    assert status["hasApiKey"] is True
    assert status["keySource"] == "environment"
    assert "apiKey" not in status
