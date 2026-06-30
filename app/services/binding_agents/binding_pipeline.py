"""Specialized binding pipeline agents."""
from __future__ import annotations

from typing import Any

from app.services.binding_agent import (
    AgentStep,
    BindingAgentContext,
    _action_title,
    _extract_scenario_name,
    _history_items,
    _known_gesture_labels,
    _match_gesture_label_in_text,
    _parse_action,
    _similar_gesture_labels,
    _split_sequence,
    _gesture_query_from_text,
)

class GestureAgent:
    name = "Gesture Agent"

    def run(self, context: BindingAgentContext) -> AgentStep:
        labels = _known_gesture_labels(context)
        query = _gesture_query_from_text(context.prompt)
        gesture = _match_gesture_label_in_text(context.prompt, labels)
        if gesture:
            known = gesture.lower() in {label.lower() for label in labels}
            if known:
                return AgentStep(
                    self.name,
                    "ok",
                    f"Нашёл жест из текста: {gesture}.",
                    {"gesture": gesture, "source": "prompt", "known": True},
                )
            return AgentStep(
                self.name,
                "ok",
                f"Принял явно написанный жест: {gesture}.",
                    {"gesture": gesture, "source": "typed", "known": False},
                )

        suggestions = _similar_gesture_labels(query, labels)
        if suggestions:
            return AgentStep(
                self.name,
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
                known = gesture.lower() in {label.lower() for label in labels}
                return AgentStep(
                    self.name,
                    "ok",
                    f"Взял жест из памяти диалога: {gesture}.",
                    {"gesture": gesture, "source": "memory", "known": known},
                )

        current = (context.current_gesture or "").strip()
        if current and current.lower() in {label.lower() for label in labels}:
            label = next(label for label in labels if label.lower() == current.lower())
            return AgentStep(
                self.name,
                "ok",
                f"Использую выбранный жест: {label}.",
                {"gesture": label, "source": "selected", "known": True},
            )

        return AgentStep(
            self.name,
            "need_clarification",
            "Жест не указан.",
            {"gesture": "", "source": "", "known": False},
        )


class ActionAgent:
    name = "Action Agent"

    def run(self, context: BindingAgentContext) -> AgentStep:
        spec = _parse_action(context.prompt)
        if spec is None:
            for item in reversed(_history_items(context.conversation_history)):
                if item.get("role") != "user":
                    continue
                spec = _parse_action(item.get("text") or "")
                if spec is not None:
                    return AgentStep(
                        self.name,
                        "ok",
                        f"Взял действие из памяти: {_action_title(spec)}.",
                        {"action_spec": spec, "source": "memory"},
                    )
        if spec is None:
            return AgentStep(
                self.name,
                "need_clarification",
                "Не понял действие.",
                {"action_spec": {}},
            )
        return AgentStep(
            self.name,
            "ok",
            f"Собрал действие: {_action_title(spec)}.",
            {"action_spec": spec},
        )


class ScenarioAgent:
    name = "Scenario Agent"

    def run(self, context: BindingAgentContext, enabled: bool) -> AgentStep:
        if not enabled:
            return AgentStep(
                self.name,
                "skipped",
                "Сценарий не нужен для одиночной команды.",
            )
        steps: list[dict[str, Any]] = []
        for clause in _split_sequence(context.prompt):
            step = _parse_action(clause)
            if step is None:
                continue
            step = dict(step)
            step.pop("platform", None)
            steps.append(step)
        if len(steps) < 2:
            return AgentStep(
                self.name,
                "need_clarification",
                "Для сценария нужно минимум два понятных шага.",
                {"action_spec": {}},
            )
        spec = {"action": "sequence", "platform": "macos", "steps": steps}
        scenario_name = _extract_scenario_name(context.prompt)
        if scenario_name:
            spec["name"] = scenario_name
        return AgentStep(
            self.name,
            "ok",
            f"Собрал сценарий из {len(steps)} шагов.",
            {"action_spec": spec},
        )


class MemoryAgent:
    name = "Memory Agent"

    def run(self, context: BindingAgentContext, gesture: str) -> AgentStep:
        known = {
            str(item.get("label") or "").strip().lower()
            for item in context.gestures
            if str(item.get("label") or "").strip()
        }
        if gesture and gesture.lower() in known:
            message = "Жест есть в текущем списке."
        elif gesture:
            message = "Жест принят как typed label; БД проверит его при сохранении."
        else:
            message = "Нет жеста для проверки памяти."
        return AgentStep(
            self.name,
            "ok",
            message,
            {"known": bool(gesture and gesture.lower() in known)},
        )


class PolicyAgent:
    name = "Policy Agent"

    def run(self, gesture: str, action_spec: dict[str, Any]) -> AgentStep:
        missing: list[str] = []
        if not gesture:
            missing.append("жест")
        if not action_spec:
            missing.append("действие")
        if missing:
            return AgentStep(
                self.name,
                "need_clarification",
                "Нужны уточнения: " + ", ".join(missing) + ".",
                {"missing": missing},
            )
        return AgentStep(
            self.name,
            "ok",
            "Правила локальной политики пройдены.",
            {"missing": []},
        )


class ValidationAgent:
    name = "Validation Agent"

    def run(self, gesture: str, action_spec: dict[str, Any]) -> AgentStep:
        if not action_spec:
            return AgentStep(
                self.name,
                "blocked",
                "Нельзя подготовить привязку без действия.",
            )
        command_name = (
            f"{gesture}: {_action_title(action_spec)}"
            if gesture
            else _action_title(action_spec)
        )
        return AgentStep(
            self.name,
            "ok",
            "Предложение готово для UI.",
            {"command_name": command_name},
        )
