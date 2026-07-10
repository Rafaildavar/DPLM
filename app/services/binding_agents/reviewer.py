"""Reviewer agent for relevance and intent-block fit."""
from __future__ import annotations

from app.services.binding_agent import (
    AgentStep,
    BindingAgentContext,
    BindingAgentResult,
    _norm,
    _short_text,
)
from app.services.binding_agents.tools import review_answer_contract
from app.services.binding_agents.action_semantics import compile_action_candidate

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
        contract = review_answer_contract(
            intent_block=block,
            response_text=result.response_text,
            can_apply=result.can_apply,
            action_spec=result.action_spec,
        )
        issues: list[str] = list(contract.payload.get("issues") or [])
        semantic_candidate = None
        if block == "binding" and result.action_spec:
            _goal, semantic_candidate = compile_action_candidate(
                context.prompt,
                result.action_spec,
                frame=context.task_frame,
                source="reviewer",
            )
            if not semantic_candidate.valid:
                issues.append("action_semantic_mismatch")

        if block == "project_question":
            if result.can_apply or result.action_spec:
                issues.append("project_answer_contains_binding_draft")
            if not any(
                marker in response
                for marker in ("gesturebind", "жест", "привяз", "команд", "mlflow", "mistral")
            ):
                issues.append("project_context_missing")
        elif block == "binding":
            if result.mutation:
                if (
                    result.mutation.get("operation") != "delete_binding"
                    or not result.mutation.get("bindingId")
                    or not result.requires_confirmation
                ):
                    issues.append("binding_mutation_invalid")
            elif result.can_apply and (not result.gesture_label or not result.action_spec):
                issues.append("binding_ready_without_contract")
            if (
                result.missing
                and "уточ" not in response
                and "добав" not in response
                and "укаж" not in response
                and "обнов" not in response
            ):
                issues.append("clarification_text_missing")
        elif block == "unsupported_general":
            if result.can_apply or result.action_spec:
                issues.append("unsupported_question_created_binding")
            if not any(marker in response for marker in ("gesturebind", "привяз", "жест", "проект")):
                issues.append("safe_redirect_missing")
        elif block == "guardrails":
            if result.can_apply or result.action_spec:
                issues.append("guardrail_output_created_binding")
            if not any(marker in response for marker in ("guardrails", "останов", "секрет", "инструкц")):
                issues.append("guardrail_explanation_missing")
        else:
            issues.append("unknown_intent_block")

        safety_issues = {
            "guardrail_output_created_binding",
            "guardrail_explanation_missing",
            "unsupported_action_spec",
            "unsupported_question_created_binding",
            "secret_in_output",
            "answer_block_can_apply",
        }
        contract_issues = {
            "binding_ready_without_contract",
            "project_answer_contains_binding_draft",
            "unsupported_question_created_binding",
            "project_answer_contains_binding_draft",
            "empty_response",
        }
        clarification_ok = (
            not result.missing
            or "уточ" in response
            or "выберите" in response
            or "нужно" in response
            or "добав" in response
        )
        tone_ok = (
            "ответ остановлен guardrails" not in response
            and "dplm" not in response
            and bool(result.response_text.strip())
        )
        scores = {
            "relevance": 1.0 if not issues else 0.0,
            "contract_completeness": (
                0.0 if any(issue in contract_issues for issue in issues) else 1.0
            ),
            "safety": 0.0 if any(issue in safety_issues for issue in issues) else 1.0,
            "tone": 1.0 if tone_ok else 0.0,
            "clarification_quality": 1.0 if clarification_ok else 0.0,
        }
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
                "scores": scores,
                "issues": issues,
                "contractTool": contract.tool,
                "contractStatus": contract.status,
                "semanticCandidate": (
                    semantic_candidate.to_dict() if semantic_candidate else {}
                ),
                "prompt_preview": _short_text(context.prompt, 160),
            },
        )
