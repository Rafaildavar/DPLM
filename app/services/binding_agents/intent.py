"""Intent routing agent for binding MAS requests."""
from __future__ import annotations

from app.services.binding_agent import (
    AgentStep,
    BindingAgentContext,
    _is_out_of_scope_question,
    _is_project_question,
    _is_validation_question,
    _looks_like_binding_request,
    _norm,
)

class IntentAgent:
    name = "Intent Agent"

    def run(self, context: BindingAgentContext) -> AgentStep:
        lower = _norm(context.prompt)
        if not lower:
            return AgentStep(self.name, "need_input", "Жду текст запроса.")

        if _is_validation_question(context.prompt):
            return AgentStep(
                self.name,
                "ok",
                "Маршрут: проектный вопрос о соответствии команды действию.",
                {
                    "intent": "validate_command",
                    "block": "project_question",
                    "route": "answer",
                },
            )

        if _looks_like_binding_request(context):
            is_sequence = any(
                marker in lower
                for marker in (
                    "сценар",
                    "потом",
                    "затем",
                    "после этого",
                    "далее",
                    ";",
                )
            )
            is_update = any(
                marker in lower
                for marker in (
                    "измени",
                    "изменить",
                    "поменяй",
                    "замени",
                    "обнови",
                    "перепривяж",
                    "переназнач",
                )
            )
            intent = (
                "build_sequence"
                if is_sequence
                else "update_binding"
                if is_update
                else "create_binding"
            )
            message = (
                "Маршрут: изменение существующей привязки."
                if intent == "update_binding"
                else "Маршрут: сценарий из нескольких действий."
                if intent == "build_sequence"
                else "Маршрут: создание привязки."
            )
            return AgentStep(
                self.name,
                "ok",
                message,
                {
                    "intent": intent,
                    "block": "binding",
                    "route": "binding_pipeline",
                },
            )

        if _is_project_question(context.prompt):
            return AgentStep(
                self.name,
                "ok",
                "Маршрут: общий вопрос по проекту.",
                {
                    "intent": "project_question",
                    "block": "project_question",
                    "route": "answer",
                },
            )

        if _is_out_of_scope_question(context.prompt):
            return AgentStep(
                self.name,
                "need_clarification",
                "Маршрут: общий вопрос вне области агента.",
                {
                    "intent": "unsupported_general_question",
                    "block": "unsupported_general",
                    "route": "safe_redirect",
                },
            )

        return AgentStep(
            self.name,
            "need_clarification",
            "Маршрут не относится к привязкам GestureFlow.",
            {
                "intent": "unsupported_general_question",
                "block": "unsupported_general",
                "route": "safe_redirect",
            },
        )
