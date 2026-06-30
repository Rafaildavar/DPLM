"""Input and output guardrails for the binding MAS."""
from __future__ import annotations

from app.services.binding_agent import (
    AgentStep,
    BindingAgentContext,
    BindingAgentResult,
    _contains_prompt_injection,
    _contains_secret_like,
)
from app.services.binding_agents.skills import with_skills

class GuardrailsAgent:
    name = "Guardrails Agent"
    max_prompt_chars = 1200

    def run_input(self, context: BindingAgentContext) -> AgentStep:
        issues: list[str] = []
        if len(context.prompt) > self.max_prompt_chars:
            issues.append("prompt_too_long")
        if _contains_secret_like(context.prompt):
            issues.append("secret_detected")
        if _contains_prompt_injection(context.prompt):
            issues.append("prompt_injection")

        if issues:
            return AgentStep(
                self.name,
                "blocked",
                "Input guardrails остановили запрос: " + ", ".join(issues) + ".",
                with_skills(
                    {
                        "stage": "input",
                        "issues": issues,
                        "prompt_chars": len(context.prompt),
                        "blocked": True,
                        "decision": "block",
                        "guardrailVersion": "v2",
                    },
                    "guardrails.input_safety",
                ),
            )
        return AgentStep(
            self.name,
            "ok",
            "Input guardrails пройдены.",
            with_skills(
                {
                    "stage": "input",
                    "issues": [],
                    "prompt_chars": len(context.prompt),
                    "blocked": False,
                    "decision": "allow",
                    "guardrailVersion": "v2",
                },
                "guardrails.input_safety",
            ),
        )

    def run_output(
        self,
        context: BindingAgentContext,
        result: BindingAgentResult,
    ) -> AgentStep:
        _ = context
        issues: list[str] = []
        clarifications: list[str] = []
        if not result.response_text.strip():
            issues.append("empty_response")
        if _contains_secret_like(result.response_text):
            issues.append("secret_in_output")
        if result.intent_block == "unsupported_general" and result.action_spec:
            issues.append("unsupported_binding_output")
        if result.intent_block == "project_question" and result.can_apply:
            issues.append("project_answer_can_apply")
        if result.intent_block == "binding":
            if result.can_apply and (not result.gesture_label or not result.action_spec):
                issues.append("binding_contract_incomplete")
            elif not result.can_apply and result.missing:
                clarifications.extend(str(item) for item in result.missing if str(item))

        if issues:
            return AgentStep(
                self.name,
                "blocked",
                "Output guardrails остановили ответ: " + ", ".join(issues) + ".",
                with_skills(
                    {
                        "stage": "output",
                        "issues": issues,
                        "blocked": True,
                        "decision": "block",
                        "guardrailVersion": "v2",
                    },
                    "guardrails.output_safety",
                ),
            )
        if clarifications:
            return AgentStep(
                self.name,
                "need_clarification",
                "Output guardrails разрешили уточнение: "
                + ", ".join(clarifications)
                + ".",
                with_skills(
                    {
                        "stage": "output",
                        "issues": [],
                        "clarifications": clarifications,
                        "blocked": False,
                        "decision": "clarify",
                        "guardrailVersion": "v2",
                    },
                    "guardrails.output_safety",
                ),
            )
        return AgentStep(
            self.name,
            "ok",
            "Output guardrails пройдены.",
            with_skills(
                {
                    "stage": "output",
                    "issues": [],
                    "blocked": False,
                    "decision": "allow",
                    "guardrailVersion": "v2",
                },
                "guardrails.output_safety",
            ),
        )
