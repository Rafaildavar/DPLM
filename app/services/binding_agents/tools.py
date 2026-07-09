"""Tool contracts used by GestureBind binding agents."""
from __future__ import annotations

import re
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
from app.services.binding_agents.action_semantics import (
    CandidateArbiter,
    compile_action_candidate,
)
from app.services.gesture_aliases import (
    GestureAliasRegistry,
    gesture_alias_query_from_text,
    normalize_gesture_alias,
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


def _selected_known_gesture(context: BindingAgentContext, labels: list[str]) -> str:
    current = (context.current_gesture or "").strip()
    if not current:
        return ""
    for label in labels:
        if label.lower() == current.lower():
            return label
    return ""


def _is_explicit_typed_gesture_query(prompt: str, query: str) -> bool:
    clean = (query or "").strip()
    if not clean:
        return False
    if clean == (prompt or "").strip():
        return False
    normalized = normalize_gesture_alias(clean)
    if normalized in {
        "this",
        "eto",
        "это",
        "этот",
        "эту",
        "текущий",
        "текущии",
        "выбранный",
            "выбранныи",
    }:
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{1,63}", clean):
        return False
    return 0 < len(normalized.split("_")) <= 5


def _learn_selected_gesture_alias(
    context: BindingAgentContext,
    gesture_label: str,
) -> str:
    alias = gesture_alias_query_from_text(context.prompt)
    clean = (alias or "").strip()
    if not clean:
        return ""
    if clean == (context.prompt or "").strip():
        return ""
    normalized = normalize_gesture_alias(clean)
    if not normalized or normalized == normalize_gesture_alias(gesture_label):
        return ""
    if normalized in {
        "this",
        "eto",
        "это",
        "этот",
        "эту",
        "текущий",
        "текущии",
        "выбранный",
        "выбранныи",
    }:
        return ""
    if len(normalized.split("_")) > 5:
        return ""
    if GestureAliasRegistry().learn(
        gesture_label,
        clean,
        source="selected_gesture",
        confidence=0.9,
    ):
        return clean
    return ""


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
    exact_label = _exact_known_label(query, context.prompt, labels)
    if exact_label:
        return AgentToolResult(
            "resolve_gesture",
            "ok",
            f"Нашёл точный label жеста: {exact_label}.",
            {
                "gesture": exact_label,
                "source": "exact_label",
                "known": True,
                "matchedAlias": exact_label,
                "confidence": 1.0,
            },
        )
    alias_match = GestureAliasRegistry().resolve(
        context.prompt,
        labels=labels,
        gestures=context.gestures,
    )
    fuzzy_alias_match = None
    if alias_match is not None:
        if alias_match.ambiguous:
            if alias_match.source == "alias_registry:fuzzy":
                fuzzy_alias_match = alias_match
            else:
                return AgentToolResult(
                    "resolve_gesture",
                    "need_clarification",
                    "Нашёл несколько похожих жестов, нужен выбор.",
                    {
                        "gesture": "",
                        "source": alias_match.source,
                        "known": False,
                        "suggestions": alias_match.suggestions,
                        "query": alias_match.query,
                        "matchedAlias": alias_match.matched_alias,
                        "confidence": alias_match.confidence,
                    },
                )
        else:
            return AgentToolResult(
                "resolve_gesture",
                "ok",
                f"Нашёл жест по алиасу: {alias_match.label}.",
                {
                    "gesture": alias_match.label,
                    "source": alias_match.source,
                    "known": True,
                    "matchedAlias": alias_match.matched_alias,
                    "confidence": alias_match.confidence,
                },
            )
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

    if fuzzy_alias_match is not None and fuzzy_alias_match.suggestions:
        return AgentToolResult(
            "resolve_gesture",
            "need_clarification",
            "Нашёл похожие жесты, но нужен выбор.",
            {
                "gesture": "",
                "source": fuzzy_alias_match.source,
                "known": False,
                "suggestions": fuzzy_alias_match.suggestions,
                "query": fuzzy_alias_match.query,
                "matchedAlias": fuzzy_alias_match.matched_alias,
                "confidence": fuzzy_alias_match.confidence,
            },
        )

    if _is_explicit_typed_gesture_query(context.prompt, query):
        return AgentToolResult(
            "resolve_gesture",
            "ok",
            f"Принял явно указанное имя жеста: {query}.",
            {
                "gesture": query,
                "source": "typed_query",
                "known": False,
                "query": query,
            },
        )

    for item in reversed(_history_items(context.conversation_history)):
        if item.get("role") != "user":
            continue
        alias_match = GestureAliasRegistry().resolve(
            item.get("text") or "",
            labels=labels,
            gestures=context.gestures,
        )
        if alias_match is not None and not alias_match.ambiguous:
            source = alias_match.source
            if source in {"label", "label_words", "gesture_metadata:label"}:
                source = "memory"
            else:
                source = f"memory:{source}"
            return AgentToolResult(
                "resolve_gesture",
                "ok",
                f"Взял жест из памяти диалога: {alias_match.label}.",
                {
                    "gesture": alias_match.label,
                    "source": source,
                    "known": True,
                    "matchedAlias": alias_match.matched_alias,
                    "confidence": alias_match.confidence,
                },
            )
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

    label = _selected_known_gesture(context, labels)
    if label:
        learned_alias = _learn_selected_gesture_alias(context, label)
        source = "selected_alias" if learned_alias else "selected"
        return AgentToolResult(
            "resolve_gesture",
            "ok",
            f"Использую выбранный жест: {label}.",
            {
                "gesture": label,
                "source": source,
                "known": True,
                "matchedAlias": learned_alias,
            },
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

    legacy_spec = _parse_action(context.prompt)
    goal, candidate = compile_action_candidate(
        context.prompt,
        legacy_spec,
        frame=context.task_frame,
    )
    arbitration = CandidateArbiter().choose((candidate,))
    spec = dict(arbitration.selected.action_spec) if arbitration.selected else None
    if candidate.issues and any(
        issue not in {"action_unresolved"} for issue in candidate.issues
    ):
        return AgentToolResult(
            "parse_macos_action",
            "need_clarification",
            "Действие не прошло семантическую проверку.",
            {
                "action_spec": {},
                "goal": goal.to_dict(),
                "candidates": [candidate.to_dict()],
                "semanticIssues": list(candidate.issues),
            },
        )
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
            history_text = item.get("text") or ""
            history_legacy = _parse_action(history_text)
            history_goal, history_candidate = compile_action_candidate(
                history_text,
                history_legacy,
                source="dialog_memory",
            )
            history_choice = CandidateArbiter().choose((history_candidate,)).selected
            if history_choice is not None:
                spec = dict(history_choice.action_spec)
                return AgentToolResult(
                    "parse_macos_action",
                    "ok",
                    f"Взял действие из памяти: {_action_title(spec)}.",
                    {
                        "action_spec": spec,
                        "source": "memory",
                        "goal": history_goal.to_dict(),
                        "candidate": history_choice.to_dict(),
                    },
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
        {
            "action_spec": spec,
            "source": "prompt",
            "goal": goal.to_dict(),
            "candidate": arbitration.selected.to_dict() if arbitration.selected else {},
        },
    )


def build_sequence_action(context: BindingAgentContext) -> AgentToolResult:
    steps: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    clauses = _split_sequence(context.prompt)
    for index, clause in enumerate(clauses, start=1):
        legacy_step = _parse_action(clause)
        _goal, candidate = compile_action_candidate(clause, legacy_step)
        selected = CandidateArbiter().choose((candidate,)).selected
        if selected is None:
            unresolved.append({"index": index, "text": clause})
            continue
        step = dict(selected.action_spec)
        step.pop("platform", None)
        steps.append(step)
    scenario_name = _extract_scenario_name(context.prompt)
    if unresolved:
        spec: dict[str, Any] = {
            "action": "sequence",
            "platform": "macos",
            "steps": steps,
        }
        if scenario_name:
            spec["name"] = scenario_name
        return AgentToolResult(
            "build_sequence",
            "need_clarification",
            (
                "Не понял шаги сценария: "
                + ", ".join(f"#{item['index']}" for item in unresolved)
                + "."
            ),
            {
                "action_spec": spec if steps else {},
                "steps_count": len(steps),
                "expected_steps_count": len(clauses),
                "unresolved_steps": unresolved,
                "requires_confirmation": any(
                    step.get("action") in {"quit_app", "lock_screen", "run_script"}
                    for step in steps
                ),
            },
        )
    if len(steps) < 2:
        return AgentToolResult(
            "build_sequence",
            "need_clarification",
            "Для сценария нужно минимум два понятных шага.",
            {
                "action_spec": {},
                "steps_count": len(steps),
                "expected_steps_count": len(clauses),
                "unresolved_steps": [],
            },
        )
    spec: dict[str, Any] = {"action": "sequence", "platform": "macos", "steps": steps}
    if scenario_name:
        spec["name"] = scenario_name
    return AgentToolResult(
        "build_sequence",
        "ok",
        f"Собрал сценарий из {len(steps)} шагов.",
        {
            "action_spec": spec,
            "steps_count": len(steps),
            "expected_steps_count": len(clauses),
            "unresolved_steps": [],
            "requires_confirmation": any(
                step.get("action") in {"quit_app", "lock_screen", "run_script"}
                for step in steps
            ),
        },
    )


def _exact_known_label(query: str, prompt: str, labels: list[str]) -> str:
    query_key = (query or "").strip().lower()
    by_lower = {label.lower(): label for label in labels}
    if query_key in by_lower:
        return by_lower[query_key]
    lower_prompt = (prompt or "").lower()
    for label in sorted(labels, key=len, reverse=True):
        if re.search(
            rf"(?<![a-zа-я0-9_]){re.escape(label.lower())}(?![a-zа-я0-9_])",
            lower_prompt,
            re.IGNORECASE,
        ):
            return label
    return ""


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
