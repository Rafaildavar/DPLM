import os

from app.services.binding_agent import BindingAgentOrchestrator
from app.services.binding_agents.session_memory import (
    SessionMemoryAgent,
    SessionMemoryStore,
)


os.environ.setdefault("DPLM_BINDING_AGENT_MLFLOW", "0")
os.environ.setdefault("DPLM_BINDING_AGENT_PROVIDER", "local")
os.environ.setdefault("DPLM_BINDING_RESEARCH_WEB", "0")


GESTURES = [{"label": "hand"}, {"label": "palm"}, {"label": "zoom"}]


def _orchestrator() -> BindingAgentOrchestrator:
    return BindingAgentOrchestrator(
        session_memory_agent=SessionMemoryAgent(SessionMemoryStore())
    )


def test_explicit_continuation_inherits_only_active_gesture():
    orchestrator = _orchestrator()
    first = orchestrator.run(
        "привяжи palm к открытию Safari",
        GESTURES,
        session_id="continuation",
        provider="local",
    )
    second = orchestrator.run(
        "теперь command+z",
        GESTURES,
        session_id="continuation",
        provider="local",
    )

    assert first.can_apply is True
    assert second.gesture_label == "palm"
    assert second.action_spec == {
        "action": "key_combination",
        "platform": "macos",
        "keys": ["command", "z"],
    }
    assert second.telemetry["session_memory"]["relationship"] == "continuation"


def test_new_task_does_not_reuse_ready_gesture_from_previous_turn():
    orchestrator = _orchestrator()
    orchestrator.run(
        "привяжи palm к открытию Safari",
        GESTURES,
        session_id="new-task",
        provider="local",
    )
    result = orchestrator.run(
        "открой Telegram",
        GESTURES,
        session_id="new-task",
        provider="local",
    )

    assert result.gesture_label == ""
    assert result.action_spec.get("app") == "Telegram"
    assert result.missing == ["жест"]
    assert result.telemetry["session_memory"]["relationship"] == "new_task"


def test_new_incomplete_binding_does_not_reuse_previous_action():
    orchestrator = _orchestrator()
    orchestrator.run(
        "привяжи palm к открытию Safari",
        GESTURES,
        session_id="no-stale-action",
        provider="local",
    )
    result = orchestrator.run(
        "привяжи zoom",
        GESTURES,
        session_id="no-stale-action",
        provider="local",
    )

    assert result.gesture_label == "zoom"
    assert result.action_spec == {}
    assert result.missing == ["действие"]


def test_gesture_correction_prefers_target_after_negation():
    orchestrator = _orchestrator()
    orchestrator.run(
        "привяжи hand к открытию Safari",
        GESTURES,
        session_id="correction",
        provider="local",
    )
    result = orchestrator.run(
        "нет, не hand, а zoom",
        GESTURES,
        session_id="correction",
        provider="local",
    )

    assert result.gesture_label == "zoom"
    assert result.action_spec.get("app") == "Safari"
    assert result.telemetry["session_memory"]["relationship"] == "correction"
    gesture_step = next(step for step in result.steps if step.agent == "Gesture Agent")
    assert gesture_step.data["source"] == "correction"


def test_unrelated_answer_clears_executable_slots_for_next_turn():
    orchestrator = _orchestrator()
    orchestrator.run(
        "привяжи palm к открытию Safari",
        GESTURES,
        session_id="answer-reset",
        provider="local",
    )
    answer = orchestrator.run(
        "какая завтра погода?",
        GESTURES,
        session_id="answer-reset",
        provider="local",
    )
    result = orchestrator.run(
        "теперь открой Telegram",
        GESTURES,
        session_id="answer-reset",
        provider="local",
    )

    assert answer.mode == "answer"
    assert result.gesture_label == ""
    assert result.action_spec.get("app") == "Telegram"
    assert result.missing == ["жест"]

