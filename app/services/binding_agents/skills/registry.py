"""Skill registry and trace helpers for the binding MAS."""
from __future__ import annotations

from typing import Any

from app.services.binding_agents.skills.answer import ANSWER_SKILLS
from app.services.binding_agents.skills.base import AgentSkill
from app.services.binding_agents.skills.binding import BINDING_SKILLS
from app.services.binding_agents.skills.guardrails import GUARDRAILS_SKILLS
from app.services.binding_agents.skills.intent import INTENT_SKILLS
from app.services.binding_agents.skills.review import REVIEW_SKILLS
from app.services.binding_agents.skills.runtime import DEFAULT_SKILL_RUNTIME


AGENT_SKILLS: dict[str, AgentSkill] = {
    skill.skill_id: skill
    for skill in (
        *GUARDRAILS_SKILLS,
        *INTENT_SKILLS,
        *ANSWER_SKILLS,
        *BINDING_SKILLS,
        *REVIEW_SKILLS,
        *DEFAULT_SKILL_RUNTIME.manifests,
    )
}

AGENT_DEFAULT_SKILLS: dict[str, list[str]] = {
    "Guardrails Agent": ["guardrails.input_safety", "guardrails.output_safety"],
    "Intent Agent": ["intent.route"],
    "Conversation Agent": ["answer.project_context", "answer.safe_redirect"],
    "Mistral Agent": ["action.parse", "gesture.resolve"],
    "Gesture Agent": ["gesture.resolve"],
    "Memory Agent": ["memory.dialog_lookup"],
    "Session Memory Agent": ["memory.dialog_lookup"],
    "Scenario Agent": ["scenario.compose"],
    "Action Agent": ["action.parse"],
    "Binding CRUD Agent": ["binding.inspect", "binding.delete"],
    "Research Agent": ["action.research", "action.remember_research"],
    "Policy Agent": ["policy.required_fields"],
    "Validation Agent": ["validation.contract"],
    "Reviewer Agent": ["review.relevance"],
}


def skill_card(skill_id: str) -> dict[str, str]:
    skill = AGENT_SKILLS.get(skill_id)
    if skill is None:
        return {
            "id": skill_id,
            "title": skill_id,
            "description": "Внутренний навык агента.",
        }
    return {
        "id": skill.skill_id,
        "title": skill.title,
        "description": skill.description,
        "version": skill.version,
        "risk": skill.risk,
        "supportedOs": ",".join(skill.supported_os),
    }


def skill_registry_cards() -> list[dict[str, str]]:
    return [skill_card(skill_id) for skill_id in sorted(AGENT_SKILLS)]


def step_data_with_skills(step: Any) -> dict[str, Any]:
    data = dict(getattr(step, "data", {}) or {})
    skill_ids = data.pop("skill_ids", None)
    if not isinstance(skill_ids, list):
        skill_ids = AGENT_DEFAULT_SKILLS.get(str(getattr(step, "agent", "")), [])
    skill_ids = [str(item) for item in skill_ids if str(item)]
    data.setdefault("skillIds", skill_ids)
    data.setdefault("skills", [skill_card(skill_id) for skill_id in skill_ids])
    return data


def with_skills(data: dict[str, Any] | None, *skill_ids: str) -> dict[str, Any]:
    payload = dict(data or {})
    payload["skill_ids"] = [skill_id for skill_id in skill_ids if skill_id]
    return payload
