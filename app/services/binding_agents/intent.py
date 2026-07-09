"""Intent routing agent backed by a normalized TaskFrame."""
from __future__ import annotations

from app.services.binding_agents.contracts import (
    AgentStep,
    BindingAgentContext,
    TaskFrame,
)
from app.services.binding_agents.task_frame import TaskFrameExtractor


class IntentAgent:
    name = "Intent Agent"

    def __init__(self, extractor: TaskFrameExtractor | None = None) -> None:
        self.extractor = extractor or TaskFrameExtractor()

    def build_frame(self, context: BindingAgentContext) -> TaskFrame:
        return self.extractor.extract(context)

    def run(
        self,
        context: BindingAgentContext,
        *,
        frame: TaskFrame | None = None,
    ) -> AgentStep:
        if not context.prompt.strip():
            return AgentStep(self.name, "need_input", "Жду текст запроса.")
        task = frame or context.task_frame or self.build_frame(context)
        status = (
            "need_clarification"
            if task.block == "unsupported_general"
            else "ok"
        )
        message = {
            "create_binding": "Маршрут: создание привязки.",
            "update_binding": "Маршрут: изменение существующей привязки.",
            "delete_binding": "Маршрут: удаление существующей привязки.",
            "inspect_binding": "Маршрут: просмотр существующей привязки.",
            "cancel_binding": "Маршрут: отмена создания привязки.",
            "build_sequence": "Маршрут: сценарий из нескольких действий.",
            "validate_command": "Маршрут: проверка команды и действия.",
            "project_question": "Маршрут: вопрос по GestureBind.",
            "unsupported_general_question": "Маршрут: общий вопрос вне GestureBind.",
        }.get(task.intent, "Маршрут выбран по task-frame.")
        data = task.to_dict()
        data.update(
            {
                "intent": task.intent,
                "block": task.block,
                "route": task.route,
                "routeMethod": task.route_method,
                "ruleConfidence": round(task.confidence, 4),
                "taskFrame": task.to_dict(),
            }
        )
        if task.alternatives:
            data["semanticIntent"] = task.alternatives[0][0]
            data["semanticScore"] = round(task.alternatives[0][1], 4)
        return AgentStep(self.name, status, message, data)


__all__ = ["IntentAgent"]
