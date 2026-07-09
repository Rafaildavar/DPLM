"""Contract-first task-frame extraction for binding requests."""
from __future__ import annotations

import re

from app.services.binding_agents.contracts import (
    BindingAgentContext,
    TaskDomain,
    TaskEvidence,
    TaskFrame,
    TaskOperation,
)
from app.services.binding_agents.semantic_router import (
    SEMANTIC_MIN_MARGIN,
    SEMANTIC_ROUTE_THRESHOLD,
    SEMANTIC_SEQUENCE_OVERRIDE_THRESHOLD,
    route_semantically,
)


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower().replace("ё", "е"))


def _contains_any(text: str, markers: tuple[str, ...]) -> str:
    return next((marker for marker in markers if marker in text), "")


def _question_like(text: str) -> bool:
    lower = _norm(text)
    return "?" in text or lower.startswith(
        (
            "что ",
            "как ",
            "почему ",
            "зачем ",
            "когда ",
            "где ",
            "какой ",
            "какая ",
            "какие ",
            "сколько ",
            "можно ли ",
            "можешь ли ",
            "правда ли ",
            "is ",
            "does ",
            "what ",
            "how ",
            "why ",
        )
    )


def _known_label_in_text(context: BindingAgentContext) -> str:
    lower = _norm(context.prompt)
    labels = [
        str(item.get("label") or "").strip()
        for item in context.gestures
        if str(item.get("label") or "").strip()
    ]
    for label in sorted(labels, key=len, reverse=True):
        pattern = rf"(?<![a-zа-я0-9_]){re.escape(_norm(label))}(?![a-zа-я0-9_])"
        if re.search(pattern, lower, re.IGNORECASE):
            return label
    return ""


def _gesture_phrase(text: str) -> str:
    raw = text or ""
    explicit = re.search(
        r"(?:^|\s)(?:жест(?:ом|а)?|gesture)\s*[:=]?\s+"
        r"(?P<value>.+?)(?=\s+(?:к|на|для|сценар|откр|закр|запуст|нажм|будет|должен)\b|[,.;:]|$)",
        raw,
        re.IGNORECASE,
    )
    if explicit:
        return explicit.group("value").strip()
    generic = re.search(
        r"(?:привяж\w*|назнач\w*|сохрани\w*)\s+"
        r"(?P<value>.+?)(?=\s+(?:к|на|для)\s+)",
        raw,
        re.IGNORECASE,
    )
    return generic.group("value").strip() if generic else ""


class TaskFrameExtractor:
    """Build a single intent/operation frame before any action is compiled."""

    _create_markers = (
        "привяж",
        "назнач",
        "сохрани",
        "создай привяз",
        "добавь привяз",
        "bind ",
    )
    _update_markers = (
        "измени",
        "поменяй",
        "замени",
        "обнови",
        "перепривяж",
        "переназнач",
    )
    _delete_markers = (
        "удали привяз",
        "удалить привяз",
        "убери привяз",
        "отвяжи",
        "remove binding",
        "delete binding",
    )
    _inspect_markers = (
        "покажи текущую привяз",
        "покажи привяз",
        "что привязано",
        "какая привязка",
        "список привяз",
        "list bindings",
        "show binding",
    )
    _sequence_markers = (
        "сценар",
        "серия команд",
        "серию команд",
        "последовательност",
        "несколько команд",
        "потом",
        "затем",
        "после этого",
    )
    _action_markers = (
        "откр",
        "закр",
        "запуст",
        "нажм",
        "постав",
        "увелич",
        "уменьш",
        "повыс",
        "пониз",
        "включ",
        "выключ",
        "заблок",
        "покаж",
        "перелист",
        "переключ",
        "open ",
        "close ",
        "quit ",
        "press ",
        "command+",
        "cmd+",
        "ctrl+",
        "hotkey",
    )
    _project_markers = (
        "gesturebind",
        "агент",
        "жест",
        "привяз",
        "команд",
        "сценар",
        "mlflow",
        "mistral",
        "llm",
        "распознаван",
        "датасет",
        "обуч",
    )
    _capability_markers = (
        "что ты умеешь",
        "что умеешь",
        "что можешь",
        "какие команды",
        "какие жесты",
        "как ты работаешь",
        "capabilities",
    )

    def extract(self, context: BindingAgentContext) -> TaskFrame:
        prompt = context.prompt
        lower = _norm(prompt)
        semantic = route_semantically(prompt)
        label = _known_label_in_text(context)
        gesture = _gesture_phrase(prompt) or label
        evidence: list[TaskEvidence] = []
        if label:
            evidence.append(TaskEvidence("known_gesture_label", label, 1.0))

        cancel = re.search(
            r"(?:^|\s)не\s+(?:привязывай|привязать|создавай|назначай|сохраняй)",
            lower,
        )
        if cancel:
            evidence.append(TaskEvidence("negation", cancel.group(0).strip(), 0.99))
            return self._frame(
                TaskDomain.BINDING,
                TaskOperation.CANCEL,
                "cancel_binding",
                "binding",
                "answer",
                0.99,
                "rule",
                gesture,
                prompt,
                True,
                evidence,
                semantic,
            )

        marker = _contains_any(lower, self._delete_markers)
        if marker:
            evidence.append(TaskEvidence("operation_rule", marker, 0.99))
            return self._frame(
                TaskDomain.BINDING,
                TaskOperation.DELETE,
                "delete_binding",
                "binding",
                "binding_crud",
                0.99,
                "rule",
                gesture,
                prompt,
                False,
                evidence,
                semantic,
            )

        marker = _contains_any(lower, self._inspect_markers)
        if marker:
            evidence.append(TaskEvidence("operation_rule", marker, 0.98))
            return self._frame(
                TaskDomain.BINDING,
                TaskOperation.INSPECT,
                "inspect_binding",
                "binding",
                "binding_crud",
                0.98,
                "rule",
                gesture,
                prompt,
                False,
                evidence,
                semantic,
            )

        if self._is_validation(lower):
            evidence.append(TaskEvidence("validation_rule", "question+command", 0.96))
            return self._frame(
                TaskDomain.PROJECT,
                TaskOperation.VALIDATE,
                "validate_command",
                "project_question",
                "answer",
                0.96,
                "rule",
                gesture,
                prompt,
                False,
                evidence,
                semantic,
            )

        sequence_marker = _contains_any(lower, self._sequence_markers)
        semantic_sequence = (
            semantic.intent == "build_sequence"
            and semantic.score >= SEMANTIC_SEQUENCE_OVERRIDE_THRESHOLD
            and self._has_multiple_actions(lower)
        )
        if sequence_marker or semantic_sequence:
            evidence.append(
                TaskEvidence(
                    "sequence_rule" if sequence_marker else "semantic_sequence",
                    sequence_marker or semantic.matched_example,
                    0.96 if sequence_marker else semantic.score,
                )
            )
            return self._frame(
                TaskDomain.BINDING,
                TaskOperation.BUILD_SEQUENCE,
                "build_sequence",
                "binding",
                "binding_pipeline",
                0.96 if sequence_marker else max(0.72, semantic.score),
                "rule" if sequence_marker else "semantic",
                gesture,
                prompt,
                False,
                evidence,
                semantic,
                sequence=True,
            )

        update_marker = _contains_any(lower, self._update_markers)
        if update_marker:
            evidence.append(TaskEvidence("operation_rule", update_marker, 0.94))
            return self._frame(
                TaskDomain.BINDING,
                TaskOperation.UPDATE,
                "update_binding",
                "binding",
                "binding_pipeline",
                0.94,
                "rule",
                gesture,
                prompt,
                False,
                evidence,
                semantic,
            )

        create_marker = _contains_any(lower, self._create_markers)
        action_marker = _contains_any(lower, self._action_markers)
        has_context = self._has_binding_context(context)
        explicit_binding = bool(
            create_marker
            or label
            or gesture
            or (action_marker and not _question_like(prompt))
            or (action_marker and has_context)
        )
        command_shaped_question = bool(action_marker and (label or "жест" in lower))
        if explicit_binding or command_shaped_question:
            matched = create_marker or action_marker or label or gesture
            evidence.append(TaskEvidence("binding_rule", matched, 0.92))
            return self._frame(
                TaskDomain.BINDING,
                TaskOperation.CREATE,
                "create_binding",
                "binding",
                "binding_pipeline",
                0.92,
                "rule",
                gesture,
                prompt,
                False,
                evidence,
                semantic,
            )

        capability = _contains_any(lower, self._capability_markers)
        if capability:
            evidence.append(TaskEvidence("project_rule", capability, 0.94))
            return self._answer_frame("project_question", evidence, semantic, 0.94)

        project_marker = _contains_any(lower, self._project_markers)
        if _question_like(prompt) and project_marker:
            evidence.append(TaskEvidence("project_question", project_marker, 0.88))
            return self._answer_frame("project_question", evidence, semantic, 0.88)

        if (
            semantic.score >= SEMANTIC_ROUTE_THRESHOLD
            and semantic.margin >= SEMANTIC_MIN_MARGIN
        ):
            evidence.append(TaskEvidence("semantic_route", semantic.matched_example, semantic.score))
            domain = (
                TaskDomain.BINDING
                if semantic.block == "binding"
                else TaskDomain.PROJECT
                if semantic.block == "project_question"
                else TaskDomain.GENERAL
            )
            operation = self._operation_for_intent(semantic.intent)
            return self._frame(
                domain,
                operation,
                semantic.intent,
                semantic.block,
                semantic.route,
                semantic.score,
                "semantic",
                gesture,
                prompt,
                False,
                evidence,
                semantic,
                sequence=semantic.intent == "build_sequence",
            )

        evidence.append(TaskEvidence("fallback", "unsupported_general", 0.7))
        return self._frame(
            TaskDomain.GENERAL,
            TaskOperation.ANSWER,
            "unsupported_general_question",
            "unsupported_general",
            "safe_redirect",
            0.7,
            "fallback",
            "",
            prompt,
            False,
            evidence,
            semantic,
        )

    def _answer_frame(
        self,
        intent: str,
        evidence: list[TaskEvidence],
        semantic,
        confidence: float,
    ) -> TaskFrame:
        return self._frame(
            TaskDomain.PROJECT,
            TaskOperation.ANSWER,
            intent,
            "project_question",
            "answer",
            confidence,
            "rule",
            "",
            "",
            False,
            evidence,
            semantic,
        )

    def _frame(
        self,
        domain: TaskDomain,
        operation: TaskOperation,
        intent: str,
        block: str,
        route: str,
        confidence: float,
        route_method: str,
        gesture: str,
        action_text: str,
        negated: bool,
        evidence: list[TaskEvidence],
        semantic,
        *,
        sequence: bool = False,
    ) -> TaskFrame:
        alternatives = tuple(
            (intent, score)
            for intent, score, _example in semantic.alternatives
        ) or (((semantic.intent, semantic.score),) if semantic.intent else ())
        return TaskFrame(
            domain=domain,
            operation=operation,
            intent=intent,
            block=block,
            route=route,
            confidence=max(0.0, min(1.0, float(confidence))),
            route_method=route_method,
            gesture_text=gesture,
            action_text=action_text,
            negated=negated,
            sequence_requested=sequence,
            evidence=tuple(evidence),
            alternatives=alternatives,
        )

    def _is_validation(self, lower: str) -> bool:
        question = _contains_any(
            lower,
            (
                "разве",
                "правильно ли",
                "верно ли",
                "подходит ли",
                "соответствует",
                "будет ли",
            ),
        )
        command = _contains_any(
            lower,
            ("command", "cmd", "ctrl", "сочет", "клав", "действ", "закр", "откр"),
        )
        return bool(question and command)

    def _has_multiple_actions(self, lower: str) -> bool:
        action_hits = sum(
            len(re.findall(re.escape(marker), lower))
            for marker in self._action_markers
            if len(marker.strip()) >= 4
        )
        separators = lower.count(";") + len(
            re.findall(r",\s*(?:откр|закр|запуст|включ|покаж|нажм|подожд)", lower)
        )
        return action_hits >= 2 or separators >= 1

    def _has_binding_context(self, context: BindingAgentContext) -> bool:
        if context.current_gesture:
            return True
        if context.draft_state.get("gestureLabel") or context.draft_state.get("actionSpec"):
            return True
        labels = {
            _norm(str(item.get("label") or ""))
            for item in context.gestures
            if str(item.get("label") or "").strip()
        }
        for item in reversed(context.conversation_history[-8:]):
            if item.get("role") != "user":
                continue
            text = _norm(item.get("text") or "")
            if any(
                re.search(
                    rf"(?<![a-zа-я0-9_]){re.escape(label)}(?![a-zа-я0-9_])",
                    text,
                )
                for label in labels
                if label
            ):
                return True
        return False

    def _operation_for_intent(self, intent: str) -> TaskOperation:
        return {
            "create_binding": TaskOperation.CREATE,
            "update_binding": TaskOperation.UPDATE,
            "build_sequence": TaskOperation.BUILD_SEQUENCE,
            "validate_command": TaskOperation.VALIDATE,
            "project_question": TaskOperation.ANSWER,
            "unsupported_general_question": TaskOperation.ANSWER,
        }.get(intent, TaskOperation.UNKNOWN)


__all__ = ["TaskFrameExtractor"]
