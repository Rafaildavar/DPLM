"""Binding pipeline skills."""
from __future__ import annotations

from app.services.binding_agents.skills.base import AgentSkill


BINDING_SKILLS: tuple[AgentSkill, ...] = (
    AgentSkill(
        "gesture.resolve",
        "Gesture Resolve",
        "Ищет жесты по label, русским/английским синонимам и похожим вариантам.",
    ),
    AgentSkill(
        "memory.dialog_lookup",
        "Dialog Memory",
        "Достаёт жест или действие из предыдущих сообщений диалога.",
    ),
    AgentSkill(
        "action.parse",
        "Action Parse",
        "Преобразует обычную фразу в actionSpec для macOS-команды.",
    ),
    AgentSkill(
        "scenario.compose",
        "Scenario Compose",
        "Собирает sequence из нескольких действий.",
    ),
    AgentSkill(
        "policy.required_fields",
        "Required Fields Policy",
        "Проверяет обязательные поля: жест и действие.",
    ),
    AgentSkill(
        "validation.contract",
        "Draft Contract",
        "Готовит commandName и проверяет, можно ли переносить draft в UI.",
    ),
)
