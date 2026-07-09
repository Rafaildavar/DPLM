"""Binding pipeline skills."""
from __future__ import annotations

from app.services.binding_agents.skills.base import AgentSkill


BINDING_SKILLS: tuple[AgentSkill, ...] = (
    AgentSkill(
        "binding.create",
        "Create Binding",
        "Готовит новую привязку жеста к проверенному actionSpec.",
        intents=("create_binding",),
    ),
    AgentSkill(
        "binding.update",
        "Update Binding",
        "Изменяет поля существующей привязки, сохраняя незатронутые слоты.",
        intents=("update_binding",),
    ),
    AgentSkill(
        "binding.delete",
        "Delete Binding",
        "Создаёт подтверждаемую mutation удаления существующей привязки.",
        intents=("delete_binding",),
        risk="high",
    ),
    AgentSkill(
        "binding.inspect",
        "Inspect Binding",
        "Показывает одну или несколько сохранённых привязок без изменения БД.",
        intents=("inspect_binding",),
    ),
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
        "Преобразует обычную фразу в actionSpec через словари действий и macOS-онтологию.",
    ),
    AgentSkill(
        "action.research",
        "Action Research",
        "Ищет проверенный macOS-рецепт, когда локальные навыки не знают команду.",
    ),
    AgentSkill(
        "action.remember_research",
        "Remember Research",
        "Запоминает пользовательски одобренный рецепт как skill для будущих запросов.",
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
