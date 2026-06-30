"""Guardrail skills used by the binding MAS."""
from __future__ import annotations

from app.services.binding_agents.skills.base import AgentSkill


GUARDRAILS_SKILLS: tuple[AgentSkill, ...] = (
    AgentSkill(
        "guardrails.input_safety",
        "Input Safety",
        "Проверяет длину, секреты и prompt-injection до запуска агентов.",
    ),
    AgentSkill(
        "guardrails.output_safety",
        "Output Safety",
        "Проверяет, что ответ не раскрывает секреты и не нарушает контракт блока.",
    ),
)
