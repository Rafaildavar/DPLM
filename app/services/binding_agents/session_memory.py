"""Structured, task-scoped dialog memory for the binding MAS."""
from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from app.services.binding_agents.contracts import (
    AgentStep,
    BindingAgentContext,
    BindingAgentResult,
)
from app.services.binding_agents.skills import with_skills


@dataclass(frozen=True)
class BindingSessionState:
    task_id: str
    status: str = "active"
    intent: str = "create_binding"
    gesture_label: str = ""
    action_spec: dict[str, Any] = field(default_factory=dict)
    command_name: str = ""
    source_binding_id: int = 0
    missing: tuple[str, ...] = ()
    turn_index: int = 0
    last_user_text: str = ""
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "status": self.status,
            "intent": self.intent,
            "gestureLabel": self.gesture_label,
            "actionSpec": dict(self.action_spec),
            "commandName": self.command_name,
            "sourceBindingId": self.source_binding_id,
            "missing": list(self.missing),
            "turnIndex": self.turn_index,
            "lastUserText": self.last_user_text,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BindingSessionState | None":
        if not isinstance(value, dict):
            return None
        action_spec = value.get("actionSpec")
        clean_action = (
            dict(action_spec)
            if isinstance(action_spec, dict) and action_spec.get("action")
            else {}
        )
        gesture = str(value.get("gestureLabel") or "").strip()
        missing = tuple(str(item) for item in value.get("missing") or [] if str(item))
        if not gesture and not clean_action and not missing:
            return None
        return cls(
            task_id=str(value.get("taskId") or uuid.uuid4().hex[:12]),
            status=str(value.get("status") or "active"),
            intent=str(value.get("intent") or "create_binding"),
            gesture_label=gesture,
            action_spec=clean_action,
            command_name=str(value.get("commandName") or ""),
            source_binding_id=int(value.get("sourceBindingId") or 0),
            missing=missing,
            turn_index=int(value.get("turnIndex") or 0),
            last_user_text=str(value.get("lastUserText") or ""),
            updated_at=float(value.get("updatedAt") or time.monotonic()),
        )


class SessionMemoryStore:
    def __init__(self, *, ttl_seconds: float = 1800.0, max_sessions: int = 128) -> None:
        self.ttl_seconds = max(60.0, float(ttl_seconds))
        self.max_sessions = max(8, int(max_sessions))
        self._items: dict[str, BindingSessionState] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str) -> BindingSessionState | None:
        if not session_id:
            return None
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            return self._items.get(session_id)

    def put(self, session_id: str, state: BindingSessionState) -> None:
        if not session_id:
            return
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            self._items[session_id] = replace(state, updated_at=now)
            if len(self._items) > self.max_sessions:
                oldest = min(self._items, key=lambda key: self._items[key].updated_at)
                self._items.pop(oldest, None)

    def clear(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            self._items.pop(session_id, None)

    def _evict(self, now: float) -> None:
        expired = [
            key
            for key, value in self._items.items()
            if now - value.updated_at > self.ttl_seconds
        ]
        for key in expired:
            self._items.pop(key, None)


DEFAULT_SESSION_MEMORY_STORE = SessionMemoryStore()


class SessionMemoryAgent:
    name = "Session Memory Agent"

    _continuation_markers = (
        "теперь",
        "а теперь",
        "это",
        "этот",
        "эту",
        "этому",
        "текущ",
        "предыдущ",
        "продолж",
        "добавь",
        "к нему",
        "к ней",
        "вместо",
        "нет,",
    )
    _new_task_markers = (
        "привяж",
        "назнач",
        "создай привяз",
        "сделай сценар",
        "удали привяз",
        "покажи привяз",
    )
    _action_markers = (
        "откр",
        "закр",
        "запуст",
        "наж",
        "command",
        "cmd",
        "ctrl",
        "громк",
        "ярк",
        "пау",
        "уведом",
        "сценар",
    )

    def __init__(self, store: SessionMemoryStore | None = None) -> None:
        self.store = store or DEFAULT_SESSION_MEMORY_STORE

    def prepare(self, context: BindingAgentContext) -> BindingAgentContext:
        stored = self.store.get(context.session_id)
        draft = self._state_from_draft(context.draft_state)
        active = stored or draft
        if active is None and self._has_continuation_marker(context.prompt):
            active = self._bootstrap_from_recent_history(context)

        relationship = self._relationship(context, active)
        if relationship == "new_task":
            active = None
            clean_draft: dict[str, Any] = {}
        else:
            clean_draft = self._draft_for_active_task(active, context.draft_state)

        state_payload = active.to_dict() if active else {}
        state_payload.update(
            {
                "relationship": relationship,
                "inherited": bool(active and relationship != "new_task"),
            }
        )
        return replace(
            context,
            draft_state=clean_draft,
            session_state=state_payload,
        )

    def remember(
        self,
        context: BindingAgentContext,
        result: BindingAgentResult,
    ) -> dict[str, Any]:
        previous = BindingSessionState.from_dict(context.session_state)
        if result.intent_block != "binding" or result.mode in {"answer", "mutation"}:
            if context.session_state.get("relationship") == "new_task":
                self.store.clear(context.session_id)
                return {}
            return previous.to_dict() if previous else {}
        if not (result.gesture_label or result.action_spec or result.missing):
            return previous.to_dict() if previous else {}

        relationship = str(context.session_state.get("relationship") or "new_task")
        task_id = (
            previous.task_id
            if previous is not None and relationship != "new_task"
            else uuid.uuid4().hex[:12]
        )
        state = BindingSessionState(
            task_id=task_id,
            status="ready" if result.can_apply else "clarify",
            intent=result.intent or (previous.intent if previous else "create_binding"),
            gesture_label=result.gesture_label,
            action_spec=dict(result.action_spec),
            command_name=result.command_name,
            source_binding_id=int(context.draft_state.get("sourceBindingId") or 0),
            missing=tuple(result.missing),
            turn_index=(previous.turn_index + 1) if previous else 1,
            last_user_text=context.prompt,
            updated_at=time.monotonic(),
        )
        self.store.put(context.session_id, state)
        payload = state.to_dict()
        payload["relationship"] = relationship
        return payload

    def trace_data(self, context: BindingAgentContext) -> dict[str, Any]:
        return with_skills(
            {
                "known": bool(context.session_state),
                "sessionState": dict(context.session_state),
                "relationship": str(
                    context.session_state.get("relationship") or "new_task"
                ),
            },
            "memory.dialog_lookup",
        )

    def _state_from_draft(self, draft: dict[str, Any]) -> BindingSessionState | None:
        session_state = draft.get("sessionState")
        state = BindingSessionState.from_dict(session_state)
        if state is not None:
            return state
        if str(draft.get("mode") or "") in {"answer", "mutation"}:
            return None
        return BindingSessionState.from_dict(draft)

    def _relationship(
        self,
        context: BindingAgentContext,
        active: BindingSessionState | None,
    ) -> str:
        lower = self._norm(context.prompt)
        if self._is_correction(lower):
            return "correction" if active else "new_task"
        if active is None:
            return "new_task"
        if self._has_continuation_marker(lower):
            return "continuation"
        explicit_new = any(marker in lower for marker in self._new_task_markers)
        if explicit_new:
            return "new_task"
        missing = set(active.missing)
        if "действие" in missing and any(marker in lower for marker in self._action_markers):
            return "continuation"
        if "жест" in missing and self._contains_known_gesture(lower, context):
            return "continuation"
        return "new_task"

    def _draft_for_active_task(
        self,
        active: BindingSessionState | None,
        raw_draft: dict[str, Any],
    ) -> dict[str, Any]:
        if active is None:
            return {}
        return {
            "gestureLabel": active.gesture_label,
            "actionSpec": dict(active.action_spec),
            "commandName": active.command_name,
            "sourceBindingId": active.source_binding_id,
            "missing": list(active.missing),
            "intent": active.intent,
            "mode": str(raw_draft.get("mode") or "single"),
            "sessionState": active.to_dict(),
        }

    def _bootstrap_from_recent_history(
        self,
        context: BindingAgentContext,
    ) -> BindingSessionState | None:
        labels = [
            str(item.get("label") or "").strip()
            for item in context.gestures
            if str(item.get("label") or "").strip()
        ]
        for item in reversed(context.conversation_history[-4:]):
            if item.get("role") != "user":
                continue
            text = self._norm(item.get("text") or "")
            for label in sorted(labels, key=len, reverse=True):
                if re.search(
                    rf"(?<![a-zа-я0-9_]){re.escape(label.lower())}(?![a-zа-я0-9_])",
                    text,
                ):
                    return BindingSessionState(
                        task_id=uuid.uuid4().hex[:12],
                        gesture_label=label,
                        missing=("действие",),
                        last_user_text=text,
                        updated_at=time.monotonic(),
                    )
        return None

    def _contains_known_gesture(
        self,
        lower: str,
        context: BindingAgentContext,
    ) -> bool:
        return any(
            re.search(
                rf"(?<![a-zа-я0-9_]){re.escape(str(item.get('label') or '').lower())}"
                rf"(?![a-zа-я0-9_])",
                lower,
            )
            for item in context.gestures
            if str(item.get("label") or "").strip()
        )

    def _has_continuation_marker(self, value: str) -> bool:
        lower = self._norm(value)
        return any(marker in lower for marker in self._continuation_markers)

    def _is_correction(self, lower: str) -> bool:
        return bool(
            re.search(r"(?:^|[,.]\s*)не\s+[^,]+[, ]+а\s+", lower)
            or "вместо" in lower
            or lower.startswith("нет ")
        )

    def _norm(self, value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip().lower().replace("ё", "е"))


__all__ = [
    "BindingSessionState",
    "DEFAULT_SESSION_MEMORY_STORE",
    "SessionMemoryAgent",
    "SessionMemoryStore",
]
