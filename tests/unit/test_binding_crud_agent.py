import os

from app.services.binding_agent import BindingAgentOrchestrator
from app.services.binding_agents.skills.runtime import DEFAULT_SKILL_RUNTIME


os.environ.setdefault("DPLM_BINDING_AGENT_MLFLOW", "0")
os.environ.setdefault("DPLM_BINDING_AGENT_PROVIDER", "local")
os.environ.setdefault("DPLM_BINDING_RESEARCH_WEB", "0")


GESTURES = [{"label": "palm"}, {"label": "swipe_left"}]
BINDINGS = [
    {
        "id": 7,
        "name": "open_safari",
        "gestureLabel": "palm",
        "action": "open_app",
        "actionSpec": {"action": "open_app", "app": "Safari"},
        "isActive": True,
    }
]


def _run(prompt: str):
    return BindingAgentOrchestrator().run(
        prompt,
        GESTURES,
        bindings=BINDINGS,
        provider="local",
    )


def test_inspect_binding_returns_read_only_answer():
    result = _run("покажи привязку palm")

    assert result.intent == "inspect_binding"
    assert result.mode == "answer"
    assert result.can_apply is False
    assert "open_safari" in result.response_text


def test_list_bindings_does_not_enter_binding_compiler():
    result = _run("покажи все привязки")

    assert result.intent == "inspect_binding"
    assert result.mode == "answer"
    assert "palm" in result.response_text


def test_delete_binding_returns_confirmation_only_mutation():
    result = _run("удали привязку palm")

    assert result.intent == "delete_binding"
    assert result.mode == "mutation"
    assert result.can_apply is True
    assert result.requires_confirmation is True
    assert result.mutation == {
        "operation": "delete_binding",
        "bindingId": 7,
        "gestureLabel": "palm",
        "commandName": "open_safari",
        "requiresConfirmation": True,
    }


def test_delete_unknown_binding_abstains_without_mutation():
    result = _run("удали привязку swipe_left")

    assert result.mode == "answer"
    assert result.can_apply is False
    assert result.mutation == {}
    assert result.missing == ["существующая привязка"]


def test_executable_skill_registry_selects_and_validates_contract():
    skill = DEFAULT_SKILL_RUNTIME.select(
        {"action": "quit_app", "platform": "macos", "app": "Telegram"}
    )

    assert skill is not None
    assert skill.manifest.skill_id == "macos.app.quit"
    assert skill.manifest.version == "1.0.0"
    assert skill.validate(
        {"action": "quit_app", "platform": "macos", "app": "Telegram"}
    ) == ()

