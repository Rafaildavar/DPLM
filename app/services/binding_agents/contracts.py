"""Typed contracts shared by the GestureBind binding MAS."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class AgentStatus(_ValueEnum):
    OK = "ok"
    NEED_INPUT = "need_input"
    NEED_CLARIFICATION = "need_clarification"
    NEEDS_APPROVAL = "needs_approval"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    FALLBACK = "fallback"
    ERROR = "error"


class ResultStatus(_ValueEnum):
    READY = "ready"
    CLARIFY = "clarify"
    ANSWER = "answer"
    BLOCKED = "blocked"
    ERROR = "error"
    EMPTY = "empty"


class TaskDomain(_ValueEnum):
    BINDING = "binding"
    PROJECT = "project"
    GENERAL = "general"
    GUARDRAILS = "guardrails"


class TaskOperation(_ValueEnum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    INSPECT = "inspect"
    VALIDATE = "validate"
    BUILD_SEQUENCE = "build_sequence"
    ANSWER = "answer"
    CANCEL = "cancel"
    UNKNOWN = "unknown"


class RiskLevel(_ValueEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class TaskEvidence:
    source: str
    value: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "value": self.value,
            "confidence": round(float(self.confidence), 4),
        }


@dataclass(frozen=True)
class TaskFrame:
    """One normalized interpretation of a user turn.

    The frame is descriptive only. It never authorizes command execution.
    """

    domain: TaskDomain
    operation: TaskOperation
    intent: str
    block: str
    route: str
    confidence: float
    route_method: str
    gesture_text: str = ""
    action_text: str = ""
    target_text: str = ""
    negated: bool = False
    sequence_requested: bool = False
    evidence: tuple[TaskEvidence, ...] = ()
    alternatives: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "operation": self.operation.value,
            "intent": self.intent,
            "block": self.block,
            "route": self.route,
            "confidence": round(float(self.confidence), 4),
            "routeMethod": self.route_method,
            "gestureText": self.gesture_text,
            "actionText": self.action_text,
            "targetText": self.target_text,
            "negated": self.negated,
            "sequenceRequested": self.sequence_requested,
            "evidence": [item.to_dict() for item in self.evidence],
            "alternatives": [
                {"intent": intent, "score": round(float(score), 4)}
                for intent, score in self.alternatives
            ],
        }


@dataclass(frozen=True)
class ActionCandidate:
    action_spec: dict[str, Any]
    source: str
    confidence: float
    evidence: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    skill_id: str = ""
    skill_version: str = ""
    risk: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = False

    @property
    def valid(self) -> bool:
        return bool(self.action_spec) and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "actionSpec": dict(self.action_spec),
            "source": self.source,
            "confidence": round(float(self.confidence), 4),
            "evidence": list(self.evidence),
            "issues": list(self.issues),
            "skillId": self.skill_id,
            "skillVersion": self.skill_version,
            "risk": self.risk.value,
            "requiresConfirmation": self.requires_confirmation,
        }


@dataclass(frozen=True)
class BindingAgentContext:
    prompt: str
    gestures: list[dict[str, Any]] = field(default_factory=list)
    current_gesture: str = ""
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    draft_state: dict[str, Any] = field(default_factory=dict)
    bindings: list[dict[str, Any]] = field(default_factory=list)
    task_frame: TaskFrame | None = None
    session_id: str = ""
    session_state: dict[str, Any] = field(default_factory=dict)
    request_started_at: float = 0.0
    request_deadline: float = 0.0


@dataclass(frozen=True)
class AgentStep:
    agent: str
    status: str | AgentStatus
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.status, AgentStatus):
            object.__setattr__(self, "status", self.status.value)


@dataclass(frozen=True)
class BindingAgentResult:
    ok: bool
    can_apply: bool
    error: str
    missing: list[str]
    gesture_label: str
    command_name: str
    mode: str
    action_spec: dict[str, Any]
    summary: list[str]
    response_text: str
    steps: list[AgentStep]
    intent: str = ""
    intent_block: str = ""
    telemetry: dict[str, Any] = field(default_factory=dict)
    research: dict[str, Any] = field(default_factory=dict)
    status: ResultStatus | str = ""
    task_frame: TaskFrame | None = None
    mutation: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.status, ResultStatus):
            object.__setattr__(self, "status", self.status.value)
        elif not self.status:
            object.__setattr__(self, "status", self._inferred_status().value)

    def _inferred_status(self) -> ResultStatus:
        if self.error:
            return ResultStatus.ERROR
        if self.intent_block == "guardrails":
            return ResultStatus.BLOCKED
        if self.mode == "answer":
            return ResultStatus.ANSWER
        if self.can_apply:
            return ResultStatus.READY
        if self.missing:
            return ResultStatus.CLARIFY
        return ResultStatus.EMPTY

    def to_legacy_draft(self) -> dict[str, Any]:
        from app.services.binding_agents.skill_packs import skill_pack_cards
        from app.services.binding_agents.skills import (
            skill_registry_cards,
            step_data_with_skills,
        )

        unresolved_steps: list[dict[str, Any]] = []
        alias_proposal: dict[str, Any] = {}
        for step in self.steps:
            raw_unresolved = step.data.get("unresolved_steps")
            if isinstance(raw_unresolved, list):
                unresolved_steps = [
                    dict(item) for item in raw_unresolved if isinstance(item, dict)
                ]
            raw_alias = step.data.get("aliasProposal")
            if isinstance(raw_alias, dict) and raw_alias:
                alias_proposal = dict(raw_alias)
        return {
            "ok": self.ok,
            "canApply": self.can_apply,
            "status": str(self.status),
            "error": self.error,
            "missing": list(self.missing),
            "gestureLabel": self.gesture_label,
            "commandName": self.command_name,
            "mode": self.mode,
            "actionSpec": dict(self.action_spec),
            "summary": list(self.summary),
            "agentReply": self.response_text,
            "intent": self.intent,
            "intentBlock": self.intent_block,
            "taskFrame": self.task_frame.to_dict() if self.task_frame else {},
            "mutation": dict(self.mutation),
            "requiresConfirmation": self.requires_confirmation,
            "researchProposal": dict(self.research),
            "gestureAliasProposal": alias_proposal,
            "unresolvedSteps": unresolved_steps,
            "agentSkills": skill_registry_cards(),
            "agentSkillPacks": skill_pack_cards(),
            "agentTrace": [
                {
                    "agent": step.agent,
                    "status": str(step.status),
                    "message": step.message,
                    "data": step_data_with_skills(step),
                }
                for step in self.steps
            ],
            "telemetry": dict(self.telemetry),
            "sessionState": dict(self.telemetry.get("session_memory") or {}),
            "mlflowRunId": str(self.telemetry.get("mlflow_run_id") or ""),
            "mlflowTrackingUri": str(
                self.telemetry.get("mlflow_tracking_uri") or ""
            ),
            "mlflowTraceId": str(self.telemetry.get("mlflow_trace_id") or ""),
        }


__all__ = [
    "ActionCandidate",
    "AgentStatus",
    "AgentStep",
    "BindingAgentContext",
    "BindingAgentResult",
    "ResultStatus",
    "RiskLevel",
    "TaskDomain",
    "TaskEvidence",
    "TaskFrame",
    "TaskOperation",
]
