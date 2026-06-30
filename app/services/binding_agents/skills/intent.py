"""Intent routing skills."""
from __future__ import annotations

from app.services.binding_agents.skills.base import AgentSkill


INTENT_SKILLS: tuple[AgentSkill, ...] = (
    AgentSkill(
        "intent.route",
        "Intent Routing",
        "Разделяет запросы на вопросы по проекту, привязки и внешние темы.",
    ),
)
