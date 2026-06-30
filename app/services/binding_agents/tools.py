"""Tool contracts used by GestureFlow binding agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.binding_agent import (
    BindingAgentContext,
    _action_title,
    _extract_scenario_name,
    _gesture_query_from_text,
    _history_items,
    _known_gesture_labels,
    _match_gesture_label_in_text,
    _parse_action,
    _similar_gesture_labels,
    _split_sequence,
)


def _draft_action_spec(context: BindingAgentContext) -> dict[str, Any]:
    draft_spec = context.draft_state.get("actionSpec")
    if isinstance(draft_spec, dict) and draft_spec.get("action"):
        return dict(draft_spec)
    return {}


def _is_draft_action_reference(text: str) -> bool:
    lower = (text or "").lower().replace("ё", "е")
    reference = any(
        marker in lower
        for marker in (
            "это",
            "этот",
            "эту",
            "текущ",
            "черновик",
            "предыдущ",
            "тот сценар",
            "этот сценар",
        )
    )
    action_marker = any(
        marker in lower
        for marker in (
            "откр",
            "запуст",
            "нажм",
            "command",
            "cmd",
            "ctrl",
            "уведом",
            "подожд",
            "сайт",
            "url",
            "файл",
            "папк",
            "скрин",
            "ярк",
            "громк",
        )
    )
    return reference and not action_marker


@dataclass(frozen=True)
class AgentToolResult:
    tool: str
    status: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def resolve_gesture(context: BindingAgentContext) -> AgentToolResult:
    labels = _known_gesture_labels(context)
    query = _gesture_query_from_text(context.prompt)
    gesture = _match_gesture_label_in_text(context.prompt, labels)
    known_labels = {label.lower() for label in labels}
    if gesture:
        known = gesture.lower() in known_labels
        if known:
            return AgentToolResult(
                "resolve_gesture",
                "ok",
                f"Нашёл жест из текста: {gesture}.",
                {"gesture": gesture, "source": "prompt", "known": True},
            )
        return AgentToolResult(
            "resolve_gesture",
            "ok",
            f"Принял явно написанный жест: {gesture}.",
            {"gesture": gesture, "source": "typed", "known": False},
        )

    suggestions = _similar_gesture_labels(query, labels)
    if suggestions:
        return AgentToolResult(
            "resolve_gesture",
            "need_clarification",
            "Нашёл похожие жесты, но нужен выбор.",
            {
                "gesture": "",
                "source": "similar",
                "known": False,
                "suggestions": suggestions,
                "query": query,
            },
        )

    for item in reversed(_history_items(context.conversation_history)):
        if item.get("role") != "user":
            continue
        gesture = _match_gesture_label_in_text(item.get("text") or "", labels)
        if gesture:
            known = gesture.lower() in known_labels
            return AgentToolResult(
                "resolve_gesture",
                "ok",
                f"Взял жест из памяти диалога: {gesture}.",
                {"gesture": gesture, "source": "memory", "known": known},
            )

    draft_gesture = str(context.draft_state.get("gestureLabel") or "").strip()
    if draft_gesture:
        known = draft_gesture.lower() in known_labels
        return AgentToolResult(
            "resolve_gesture",
            "ok",
            f"Взял жест из текущего черновика: {draft_gesture}.",
            {"gesture": draft_gesture, "source": "draft_state", "known": known},
        )

    current = (context.current_gesture or "").strip()
    if current and current.lower() in known_labels:
        label = next(label for label in labels if label.lower() == current.lower())
        return AgentToolResult(
            "resolve_gesture",
            "ok",
            f"Использую выбранный жест: {label}.",
            {"gesture": label, "source": "selected", "known": True},
        )

    return AgentToolResult(
        "resolve_gesture",
        "need_clarification",
        "Жест не указан.",
        {"gesture": "", "source": "", "known": False},
    )


def parse_macos_action(context: BindingAgentContext) -> AgentToolResult:
    draft_spec = _draft_action_spec(context)
    if draft_spec and _is_draft_action_reference(context.prompt):
        return AgentToolResult(
            "parse_macos_action",
            "ok",
            f"Взял действие из текущего черновика: {_action_title(draft_spec)}.",
            {"action_spec": draft_spec, "source": "draft_state"},
        )

    spec = _parse_action(context.prompt)
    if spec is None:
        if draft_spec:
            spec = draft_spec
            return AgentToolResult(
                "parse_macos_action",
                "ok",
                f"Взял действие из текущего черновика: {_action_title(spec)}.",
                {"action_spec": spec, "source": "draft_state"},
            )
    if spec is None:
        for item in reversed(_history_items(context.conversation_history)):
            if item.get("role") != "user":
                continue
            spec = _parse_action(item.get("text") or "")
            if spec is not None:
                return AgentToolResult(
                    "parse_macos_action",
                    "ok",
                    f"Взял действие из памяти: {_action_title(spec)}.",
                    {"action_spec": spec, "source": "memory"},
                )
    if spec is None:
        return AgentToolResult(
            "parse_macos_action",
            "need_clarification",
            "Не понял действие.",
            {"action_spec": {}},
        )
    return AgentToolResult(
        "parse_macos_action",
        "ok",
        f"Собрал действие: {_action_title(spec)}.",
        {"action_spec": spec, "source": "prompt"},
    )


def build_sequence_action(context: BindingAgentContext) -> AgentToolResult:
    steps: list[dict[str, Any]] = []
    for clause in _split_sequence(context.prompt):
        step = _parse_action(clause)
        if step is None:
            continue
        step = dict(step)
        step.pop("platform", None)
        steps.append(step)
    if len(steps) < 2:
        return AgentToolResult(
            "build_sequence",
            "need_clarification",
            "Для сценария нужно минимум два понятных шага.",
            {"action_spec": {}, "steps_count": len(steps)},
        )
    spec: dict[str, Any] = {"action": "sequence", "platform": "macos", "steps": steps}
    scenario_name = _extract_scenario_name(context.prompt)
    if scenario_name:
        spec["name"] = scenario_name
    return AgentToolResult(
        "build_sequence",
        "ok",
        f"Собрал сценарий из {len(steps)} шагов.",
        {"action_spec": spec, "steps_count": len(steps)},
    )


def validate_binding_contract(
    gesture: str,
    action_spec: dict[str, Any],
) -> AgentToolResult:
    missing: list[str] = []
    if not gesture:
        missing.append("жест")
    if not action_spec:
        missing.append("действие")
    if missing:
        return AgentToolResult(
            "validate_binding_contract",
            "need_clarification",
            "Нужны уточнения: " + ", ".join(missing) + ".",
            {"missing": missing},
        )
    return AgentToolResult(
        "validate_binding_contract",
        "ok",
        "Правила локальной политики пройдены.",
        {"missing": []},
    )


def review_answer_contract(
    *,
    intent_block: str,
    response_text: str,
    can_apply: bool,
    action_spec: dict[str, Any],
) -> AgentToolResult:
    issues: list[str] = []
    if not response_text.strip():
        issues.append("empty_response")
    if intent_block in {"project_question", "unsupported_general"} and can_apply:
        issues.append("answer_block_can_apply")
    if intent_block == "unsupported_general" and action_spec:
        issues.append("unsupported_action_spec")
    status = "ok" if not issues else "blocked"
    return AgentToolResult(
        "review_answer_contract",
        status,
        "Контракт ответа проверен." if not issues else "Контракт ответа нарушен.",
        {"issues": issues},
    )
