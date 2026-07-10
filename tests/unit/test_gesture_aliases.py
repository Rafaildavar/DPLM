import json

from app.flet_app.views.bindings import build_agent_binding_draft
from app.services.binding_agent import BindingAgentContext
from app.services.binding_agents.tools import resolve_gesture
from app.services.gesture_aliases import (
    GestureAliasRegistry,
    approve_gesture_alias_proposal,
)


def test_gesture_alias_registry_resolves_user_gesture_metadata(tmp_path):
    registry = GestureAliasRegistry(tmp_path / "aliases.json")

    match = registry.resolve(
        "привяжи лайк к открытию телеграм",
        labels=["user_gesture_001"],
        gestures=[
            {
                "label": "user_gesture_001",
                "displayName": "Лайк",
                "aliases": ["палец вверх", "thumbs up"],
            }
        ],
    )

    assert match is not None
    assert match.label == "user_gesture_001"
    assert match.source == "gesture_metadata:displayName"
    assert match.matched_alias == "Лайк"


def test_gesture_alias_registry_learns_and_reuses_user_alias(tmp_path):
    path = tmp_path / "aliases.json"
    GestureAliasRegistry(path).learn("custom_wave", "мой привет", source="user")

    match = GestureAliasRegistry(path).resolve(
        "привяжи мой привет к открытию Safari",
        labels=["custom_wave"],
        gestures=[{"label": "custom_wave"}],
    )

    assert match is not None
    assert match.label == "custom_wave"
    assert match.source == "memory:user"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["gestures"]["custom_wave"]["aliases"][0]["value"] == "мой привет"


def test_gesture_alias_registry_marks_shared_alias_as_ambiguous(tmp_path):
    registry = GestureAliasRegistry(tmp_path / "aliases.json")
    registry.learn("gesture_a", "мой жест", source="user")
    registry.learn("gesture_b", "мой жест", source="user")

    match = registry.resolve(
        "привяжи мой жест к открытию Safari",
        labels=["gesture_a", "gesture_b"],
        gestures=[{"label": "gesture_a"}, {"label": "gesture_b"}],
    )

    assert match is not None
    assert match.ambiguous is True
    assert match.label == ""
    assert match.suggestions == ["gesture_a", "gesture_b"]


def test_gesture_alias_registry_resolves_compact_swipe_aliases(tmp_path):
    registry = GestureAliasRegistry(tmp_path / "aliases.json")

    match = registry.resolve(
        "привяжи swipeleft к паузе видео",
        labels=["swipe_left"],
        gestures=[{"label": "swipe_left"}],
    )

    assert match is not None
    assert match.label == "swipe_left"
    assert match.matched_alias == "swipeleft"


def test_resolve_gesture_uses_alias_registry_env_memory(monkeypatch, tmp_path):
    path = tmp_path / "aliases.json"
    GestureAliasRegistry(path).learn("custom_wave", "мой привет", source="user")
    monkeypatch.setenv("GESTUREBIND_GESTURE_ALIASES", str(path))

    result = resolve_gesture(
        BindingAgentContext(
            "привяжи мой привет к открытию Safari",
            gestures=[{"label": "custom_wave"}],
        )
    )

    assert result.status == "ok"
    assert result.payload["gesture"] == "custom_wave"
    assert result.payload["source"] == "memory:user"


def test_resolve_gesture_uses_display_name_from_trained_gesture(monkeypatch, tmp_path):
    monkeypatch.setenv("GESTUREBIND_GESTURE_ALIASES", str(tmp_path / "aliases.json"))

    result = resolve_gesture(
        BindingAgentContext(
            "привяжи лайк к открытию Telegram",
            gestures=[
                {
                    "label": "custom_like_42",
                    "displayName": "Лайк",
                    "description": "Пользовательский жест: большой палец вверх",
                }
            ],
        )
    )

    assert result.status == "ok"
    assert result.payload["gesture"] == "custom_like_42"
    assert result.payload["matchedAlias"] == "Лайк"


def test_selected_user_gesture_alias_requires_approval_before_reuse(monkeypatch, tmp_path):
    path = tmp_path / "aliases.json"
    monkeypatch.setenv("GESTUREBIND_GESTURE_ALIASES", str(path))
    gestures = [{"label": "custom_like_42"}]

    first = build_agent_binding_draft(
        "привяжи лайк к открытию телеграм",
        gestures,
        current_gesture="custom_like_42",
    )
    second = build_agent_binding_draft(
        "привяжи лайк к открытию сафари",
        gestures,
    )

    assert first["ok"] is True
    assert first["gestureLabel"] == "custom_like_42"
    assert first["gestureAliasProposal"]["status"] == "pending"
    assert second["canApply"] is False
    assert not path.exists()

    approved = approve_gesture_alias_proposal(first["gestureAliasProposal"])
    third = build_agent_binding_draft(
        "привяжи лайк к открытию сафари",
        gestures,
    )

    assert approved is not None
    assert third["ok"] is True
    assert third["gestureLabel"] == "custom_like_42"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["gestures"]["custom_like_42"]["aliases"][0]["value"] == "лайк"
