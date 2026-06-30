"""Review skills."""
from __future__ import annotations

from app.services.binding_agents.skills.base import AgentSkill


REVIEW_SKILLS: tuple[AgentSkill, ...] = (
    AgentSkill(
        "review.relevance",
        "Relevance Review",
        "Оценивает, соответствует ли ответ выбранному intent-блоку.",
    ),
)
