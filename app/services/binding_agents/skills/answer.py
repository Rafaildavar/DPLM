"""Answering skills for non-binding dialog turns."""
from __future__ import annotations

from app.services.binding_agents.skills.base import AgentSkill


ANSWER_SKILLS: tuple[AgentSkill, ...] = (
    AgentSkill(
        "answer.project_context",
        "Project Answering",
        "Отвечает только в рамках GestureBind, MLflow и LLM-пайплайна.",
    ),
    AgentSkill(
        "answer.safe_redirect",
        "Safe Redirect",
        "Красиво отклоняет общие вопросы вне области проекта.",
    ),
)
