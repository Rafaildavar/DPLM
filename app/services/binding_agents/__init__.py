"""Binding multi-agent system components.

The package exposes concrete agent classes lazily so the shared facade
``app.services.binding_agent`` can import ``binding_agents.skills`` without
triggering circular imports.
"""
from __future__ import annotations

from typing import Any


__all__ = [
    "ActionAgent",
    "GestureAgent",
    "GuardrailsAgent",
    "IntentAgent",
    "MemoryAgent",
    "MistralBindingAgent",
    "PolicyAgent",
    "RelevanceReviewerAgent",
    "ScenarioAgent",
    "ValidationAgent",
]


def __getattr__(name: str) -> Any:
    if name in {
        "ActionAgent",
        "GestureAgent",
        "MemoryAgent",
        "PolicyAgent",
        "ScenarioAgent",
        "ValidationAgent",
    }:
        from app.services.binding_agents.binding_pipeline import (
            ActionAgent,
            GestureAgent,
            MemoryAgent,
            PolicyAgent,
            ScenarioAgent,
            ValidationAgent,
        )

        return {
            "ActionAgent": ActionAgent,
            "GestureAgent": GestureAgent,
            "MemoryAgent": MemoryAgent,
            "PolicyAgent": PolicyAgent,
            "ScenarioAgent": ScenarioAgent,
            "ValidationAgent": ValidationAgent,
        }[name]
    if name == "GuardrailsAgent":
        from app.services.binding_agents.guardrails import GuardrailsAgent

        return GuardrailsAgent
    if name == "IntentAgent":
        from app.services.binding_agents.intent import IntentAgent

        return IntentAgent
    if name == "MistralBindingAgent":
        from app.services.binding_agents.mistral import MistralBindingAgent

        return MistralBindingAgent
    if name == "RelevanceReviewerAgent":
        from app.services.binding_agents.reviewer import RelevanceReviewerAgent

        return RelevanceReviewerAgent
    raise AttributeError(name)
