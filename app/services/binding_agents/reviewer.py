"""Reviewer agent for relevance and intent-block fit."""
from __future__ import annotations

from app.services.binding_agent import (
    AgentStep,
    BindingAgentContext,
    BindingAgentResult,
    _norm,
    _short_text,
)

class RelevanceReviewerAgent:
    name = "Reviewer Agent"

    def run(
        self,
        context: BindingAgentContext,
        result: BindingAgentResult,
        intent_step: AgentStep,
    ) -> AgentStep:
        intent = str(intent_step.data.get("intent") or result.intent or "")
        block = str(intent_step.data.get("block") or result.intent_block or "")
        response = _norm(result.response_text)
        issues: list[str] = []

        if not result.response_text.strip():
            issues.append("empty_response")

        if block == "project_question":
            if result.can_apply or result.action_spec:
                issues.append("project_answer_contains_binding_draft")
            if not any(
                marker in response
                for marker in ("gestureflow", "жест", "привяз", "команд", "mlflow", "mistral")
            ):
                issues.append("project_context_missing")
        elif block == "binding":
            if result.can_apply and (not result.gesture_label or not result.action_spec):
                issues.append("binding_ready_without_contract")
            if result.missing and "уточ" not in response and "добав" not in response:
                issues.append("clarification_text_missing")
        elif block == "unsupported_general":
            if result.can_apply or result.action_spec:
                issues.append("unsupported_question_created_binding")
            if not any(marker in response for marker in ("gestureflow", "привяз", "жест", "проект")):
                issues.append("safe_redirect_missing")
        elif block == "guardrails":
            if result.can_apply or result.action_spec:
                issues.append("guardrail_output_created_binding")
            if not any(marker in response for marker in ("guardrails", "останов", "секрет", "инструкц")):
                issues.append("guardrail_explanation_missing")
        else:
            issues.append("unknown_intent_block")

        relevance = 1.0 if not issues else 0.0
        status = "ok" if not issues else "blocked"
        message = (
            "Ответ релевантен выбранному блоку интента."
            if not issues
            else "Ответ требует безопасной замены: " + ", ".join(issues) + "."
        )
        return AgentStep(
            self.name,
            status,
            message,
            {
                "intent": intent,
                "block": block,
                "relevance": relevance,
                "issues": issues,
                "prompt_preview": _short_text(context.prompt, 160),
            },
        )
