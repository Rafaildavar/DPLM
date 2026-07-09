"""Skill contract for the binding MAS."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSkill:
    skill_id: str
    title: str
    description: str
    version: str = "1.0.0"
    intents: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    positive_examples: tuple[str, ...] = ()
    negative_examples: tuple[str, ...] = ()
    input_schema: str = "TaskFrame"
    output_schema: str = "AgentToolResult"
    supported_os: tuple[str, ...] = ("macos",)
    required_permissions: tuple[str, ...] = ()
    risk: str = "low"
    timeout_ms: int = 100
