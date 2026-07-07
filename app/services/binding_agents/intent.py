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
from app.services.binding_agents.semantic_router import (
    SEMANTIC_ROUTE_THRESHOLD,
    SEMANTIC_SEQUENCE_OVERRIDE_THRESHOLD,
    SemanticRoute,
    route_semantically,
)

class IntentAgent:
    name = "Intent Agent"

    def run(self, context: BindingAgentContext) -> AgentStep:
        lower = _norm(context.prompt)
        if not lower:
            return AgentStep(self.name, "need_input", "Жду текст запроса.")

        semantic = route_semantically(context.prompt)

        if _is_validation_question(context.prompt):
            return AgentStep(
                self.name,
                "ok",
                "Маршрут: проектный вопрос о соответствии команды действию.",
                {
                    "intent": "validate_command",
                    "block": "project_question",
                    "route": "answer",
                    "routeMethod": "rule",
                    "ruleConfidence": 0.96,
                    "semanticIntent": semantic.intent,
                    "semanticScore": round(semantic.score, 4),
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
                    "routeMethod": "rule",
                    "ruleConfidence": 0.92,
                    "semanticIntent": semantic.intent,
                    "semanticScore": round(semantic.score, 4),
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
                    "routeMethod": "rule",
                    "ruleConfidence": 0.9,
                    "semanticIntent": semantic.intent,
                    "semanticScore": round(semantic.score, 4),
                },
            )

        if (
            semantic.intent == "build_sequence"
            and semantic.score >= SEMANTIC_SEQUENCE_OVERRIDE_THRESHOLD
        ):
            return self._semantic_step(
                semantic,
                "Маршрут: сценарий выбран по смыслу фразы.",
            )

        if _looks_like_binding_request(context):
            is_sequence = any(
                marker in lower
                for marker in (
                    "сценар",
                    "серия команд",
                    "серию команд",
                    "серии команд",
                    "последовательность команд",
                    "несколько команд",
                    "нескольких команд",
                    "последовательно",
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
                    "routeMethod": "rule",
                    "ruleConfidence": 0.88,
                    "semanticIntent": semantic.intent,
                    "semanticScore": round(semantic.score, 4),
                },
            )

        if semantic.score >= SEMANTIC_ROUTE_THRESHOLD:
            return self._semantic_step(
                semantic,
                "Маршрут выбран по semantic-router.",
            )

        return AgentStep(
            self.name,
            "need_clarification",
            "Маршрут не относится к привязкам GestureBind.",
            {
                "intent": "unsupported_general_question",
                "block": "unsupported_general",
                "route": "safe_redirect",
                "routeMethod": "fallback",
                "ruleConfidence": 0.0,
                "semanticIntent": semantic.intent,
                "semanticScore": round(semantic.score, 4),
                "semanticExample": semantic.matched_example,
            },
        )

    def _semantic_step(self, semantic: SemanticRoute, message: str) -> AgentStep:
        status = (
            "need_clarification"
            if semantic.block == "unsupported_general"
            else "ok"
        )
        return AgentStep(
            self.name,
            status,
            message,
            {
                "intent": semantic.intent,
                "block": semantic.block,
                "route": semantic.route,
                "routeMethod": "semantic",
                "semanticIntent": semantic.intent,
                "semanticScore": round(semantic.score, 4),
                "semanticExample": semantic.matched_example,
            },
        )
