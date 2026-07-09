import os

from app.services.binding_agent import BindingAgentOrchestrator
from app.services.binding_agents.action_semantics import (
    ActionCandidate,
    CandidateArbiter,
    RiskLevel,
)
from app.services.binding_agents.semantic_router import route_semantically
from app.services.user_command_sync import validate_action_spec


os.environ.setdefault("DPLM_BINDING_AGENT_MLFLOW", "0")
os.environ.setdefault("DPLM_BINDING_AGENT_PROVIDER", "local")
os.environ.setdefault("DPLM_BINDING_RESEARCH_WEB", "0")


GESTURES = [
    {"label": "hand", "aliases": ["ладонь"]},
    {"label": "palm"},
    {"label": "zoom"},
    {"label": "swipe_left"},
]


def _run(prompt: str):
    return BindingAgentOrchestrator().run(prompt, GESTURES, provider="local")


def test_exact_gesture_label_beats_colliding_normalized_alias():
    result = _run("привяжи palm к повышению звука")

    assert result.gesture_label == "palm"
    gesture_step = next(step for step in result.steps if step.agent == "Gesture Agent")
    assert gesture_step.data["source"] == "exact_label"
    assert gesture_step.data["confidence"] == 1.0


def test_close_app_goal_cannot_turn_into_open_app():
    result = _run("привяжи swipe_left к закрытию Telegram")

    assert result.can_apply is True
    assert result.action_spec == {
        "action": "quit_app",
        "platform": "macos",
        "app": "Telegram",
    }
    assert result.requires_confirmation is True
    assert validate_action_spec(result.action_spec) is None


def test_press_space_and_page_zoom_compile_to_executor_contracts():
    press = _run("привяжи palm к нажатию пробела")
    zoom = _run("привяжи palm к увеличению масштаба страницы")

    assert press.action_spec == {
        "action": "press",
        "platform": "macos",
        "key": "space",
    }
    assert zoom.action_spec == {
        "action": "key_combination",
        "platform": "macos",
        "keys": ["command", "+"],
    }


def test_unsupported_system_toggle_does_not_masquerade_as_application():
    result = _run("привяжи palm к включению bluetooth")

    assert result.can_apply is False
    assert result.action_spec == {}
    assert result.missing == ["действие"]


def test_update_without_action_does_not_open_gesture_named_zoom():
    result = _run("поменяй жест у текущей команды на zoom")

    assert result.intent == "update_binding"
    assert result.gesture_label == "zoom"
    assert result.action_spec == {}
    assert result.missing == ["действие"]


def test_question_shaped_binding_stays_in_binding_pipeline():
    result = _run("можешь ли ты просто открыть Safari жестом palm?")

    assert result.intent == "create_binding"
    assert result.can_apply is True
    assert result.action_spec["action"] == "open_app"


def test_sequence_compiler_preserves_open_quit_and_media_steps():
    result = _run(
        "сделай сценарий palm: открыть Safari, закрыть Telegram "
        "и поставить видео на паузу"
    )

    assert result.can_apply is True
    assert result.action_spec["steps"] == [
        {"action": "open_app", "app": "Safari"},
        {"action": "quit_app", "app": "Telegram"},
        {"action": "media_key", "kind": "pause"},
    ]
    assert result.requires_confirmation is True


def test_candidate_arbiter_abstains_when_different_candidates_are_too_close():
    candidates = (
        ActionCandidate(
            {"action": "volume_up"},
            "one",
            0.82,
            skill_id="macos.volume.control",
        ),
        ActionCandidate(
            {"action": "brightness_up"},
            "two",
            0.79,
            skill_id="macos.brightness.control",
        ),
    )

    decision = CandidateArbiter(min_margin=0.08).choose(candidates)

    assert decision.selected is None
    assert decision.ambiguous is True


def test_semantic_router_reports_ranked_embedding_candidates_and_margin():
    route = route_semantically("поменяй действие у свайпа")

    assert route.intent == "update_binding"
    assert route.method == "hybrid_embedding"
    assert route.margin > 0
    assert route.alternatives[0][0] == "update_binding"
