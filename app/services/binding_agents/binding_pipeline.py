"""Specialized binding pipeline agents."""
from __future__ import annotations

from typing import Any

from app.services.binding_agent import (
    AgentStep,
    BindingAgentContext,
    _action_title,
)
from app.services.binding_agents.tools import (
    build_sequence_action,
    parse_macos_action,
    resolve_gesture,
    validate_binding_contract,
)

class GestureAgent:
    name = "Gesture Agent"

    def run(self, context: BindingAgentContext) -> AgentStep:
        result = resolve_gesture(context)
        return AgentStep(
            self.name,
            result.status,
            result.message,
            result.payload,
        )


class ActionAgent:
    name = "Action Agent"

    def run(self, context: BindingAgentContext) -> AgentStep:
        result = parse_macos_action(context)
        return AgentStep(
            self.name,
            result.status,
            result.message,
            result.payload,
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
        result = build_sequence_action(context)
        return AgentStep(
            self.name,
            result.status,
            result.message,
            result.payload,
        )


class MemoryAgent:
    name = "Memory Agent"

    def run(self, context: BindingAgentContext, gesture: str) -> AgentStep:
        known = {
            str(item.get("label") or "").strip().lower()
            for item in context.gestures
            if str(item.get("label") or "").strip()
        }
        relationship = str(
            context.session_state.get("relationship") or "new_task"
        )
        if relationship == "correction":
            message = "Применяю коррекцию только к активной задаче."
        elif relationship == "continuation":
            message = "Продолжаю активную задачу из структурированной памяти."
        elif gesture and gesture.lower() in known:
            message = "Жест есть в текущем списке; начата новая задача."
        elif gesture:
            message = (
                "Жеста нет в активном словаре; "
                "привязка заблокирована до выбора или записи жеста."
            )
        else:
            message = "Новая задача не наследует жест из старого диалога."
        return AgentStep(
            self.name,
            "ok",
            message,
            {
                "known": bool(gesture and gesture.lower() in known),
                "relationship": relationship,
                "sessionState": dict(context.session_state),
            },
        )


class PolicyAgent:
    name = "Policy Agent"

    def run(
        self,
        gesture: str,
        action_spec: dict[str, Any],
        *,
        gesture_known: bool | None = None,
        requested_gesture: str = "",
    ) -> AgentStep:
        result = validate_binding_contract(
            gesture,
            action_spec,
            gesture_known=gesture_known,
            requested_gesture=requested_gesture,
        )
        return AgentStep(
            self.name,
            result.status,
            result.message,
            result.payload,
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
        from app.services.user_command_sync import validate_action_spec

        validation_error = validate_action_spec(action_spec)
        if validation_error:
            return AgentStep(
                self.name,
                "blocked",
                validation_error,
                {"error": validation_error},
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
