"""Skill contract for the binding MAS."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSkill:
    skill_id: str
    title: str
    description: str
