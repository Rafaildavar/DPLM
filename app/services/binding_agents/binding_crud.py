"""Read/delete binding operations exposed as explicit MAS mutations."""
from __future__ import annotations

from typing import Any

from app.services.binding_agents.contracts import AgentStep, BindingAgentContext
from app.services.binding_agents.skills import with_skills
from app.services.binding_agents.tools import resolve_gesture


class BindingCrudAgent:
    name = "Binding CRUD Agent"

    def run(self, context: BindingAgentContext, *, operation: str) -> AgentStep:
        list_all = operation == "inspect" and any(
            marker in context.prompt.lower()
            for marker in ("список", "все привяз", "list bindings")
        )
        if list_all:
            return self._list_bindings(context)

        gesture_result = resolve_gesture(context)
        gesture = str(gesture_result.payload.get("gesture") or "")
        binding = self._find_binding(context, gesture)
        if not gesture and binding:
            gesture = str(binding.get("gestureLabel") or "")
        if not gesture:
            return AgentStep(
                self.name,
                "need_clarification",
                "Для операции с привязкой нужно указать жест или имя команды.",
                with_skills(
                    {
                        "operation": operation,
                        "missing": ["жест или команда"],
                        "gestureTool": gesture_result.payload,
                    },
                    f"binding.{operation}",
                ),
            )
        if not binding:
            return AgentStep(
                self.name,
                "need_clarification",
                f"Для жеста «{gesture}» сохранённая привязка не найдена.",
                with_skills(
                    {
                        "operation": operation,
                        "gesture": gesture,
                        "missing": ["существующая привязка"],
                    },
                    f"binding.{operation}",
                ),
            )

        if operation == "inspect":
            response = self._binding_markdown(binding)
            return AgentStep(
                self.name,
                "ok",
                "Нашёл текущую привязку.",
                with_skills(
                    {
                        "operation": operation,
                        "gesture": gesture,
                        "binding": binding,
                        "response": response,
                        "canApply": False,
                    },
                    "binding.inspect",
                ),
            )

        binding_id = int(binding.get("id") or 0)
        if not binding_id:
            return AgentStep(
                self.name,
                "need_clarification",
                "Привязка видна, но её идентификатор не загружен. Обновите список.",
                with_skills(
                    {
                        "operation": operation,
                        "gesture": gesture,
                        "binding": binding,
                        "missing": ["идентификатор привязки"],
                    },
                    "binding.delete",
                ),
            )
        mutation = {
            "operation": "delete_binding",
            "bindingId": binding_id,
            "gestureLabel": gesture,
            "commandName": str(binding.get("name") or ""),
            "requiresConfirmation": True,
        }
        return AgentStep(
            self.name,
            "needs_approval",
            "Подготовил удаление привязки; требуется подтверждение пользователя.",
            with_skills(
                {
                    "operation": operation,
                    "gesture": gesture,
                    "binding": binding,
                    "mutation": mutation,
                    "response": (
                        f"Нашёл привязку **{gesture} → {binding.get('name') or 'команда'}**. "
                        "Удаление изменит базу GestureBind, поэтому выполню его только после "
                        "явного подтверждения."
                    ),
                    "canApply": True,
                },
                "binding.delete",
            ),
        )

    def _list_bindings(self, context: BindingAgentContext) -> AgentStep:
        rows = [dict(item) for item in context.bindings if isinstance(item, dict)]
        if not rows:
            return AgentStep(
                self.name,
                "ok",
                "Сохранённых привязок пока нет.",
                with_skills(
                    {
                        "operation": "inspect",
                        "response": "**Сохранённых привязок пока нет.**",
                        "bindings": [],
                        "canApply": False,
                    },
                    "binding.inspect",
                ),
            )
        lines = ["**Текущие привязки**", ""]
        for item in rows[:12]:
            lines.append(
                f"- `{item.get('gestureLabel') or 'без жеста'}` → "
                f"{item.get('name') or item.get('action') or 'команда'}"
            )
        return AgentStep(
            self.name,
            "ok",
            f"Показал {len(rows)} сохранённых привязок.",
            with_skills(
                {
                    "operation": "inspect",
                    "response": "\n".join(lines),
                    "bindings": rows,
                    "canApply": False,
                },
                "binding.inspect",
            ),
        )

    def _find_binding(
        self,
        context: BindingAgentContext,
        gesture: str,
    ) -> dict[str, Any]:
        gesture_key = gesture.lower()
        prompt = context.prompt.lower()
        for item in context.bindings:
            if not isinstance(item, dict):
                continue
            label = str(item.get("gestureLabel") or "").strip()
            name = str(item.get("name") or "").strip()
            if gesture_key and label.lower() == gesture_key:
                return dict(item)
            if name and name.lower() in prompt:
                return dict(item)
        return {}

    def _binding_markdown(self, binding: dict[str, Any]) -> str:
        gesture = str(binding.get("gestureLabel") or "не указан")
        name = str(binding.get("name") or "Без имени")
        action = str(binding.get("action") or "неизвестно")
        active = "да" if binding.get("isActive", True) else "нет"
        return (
            f"**Привязка `{gesture}`**\n\n"
            f"- Команда: **{name}**\n"
            f"- Действие: `{action}`\n"
            f"- Активна: {active}"
        )


__all__ = ["BindingCrudAgent"]
