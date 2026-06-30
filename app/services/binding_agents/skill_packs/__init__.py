"""Filesystem skill packs for the GestureFlow binding MAS."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


SKILL_PACK_ROOT = Path(__file__).resolve().parent

EXPECTED_SKILL_PACKS: tuple[str, ...] = (
    "intent-routing",
    "gesture-taxonomy",
    "macos-actions",
    "scenario-planning",
    "guardrails-policy",
    "reviewer-rubric",
    "tone-of-voice",
)


@dataclass(frozen=True)
class SkillPack:
    pack_id: str
    title: str
    description: str
    path: str
    content: str


def _title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        clean = line.strip()
        if clean.startswith("# "):
            return clean[2:].strip() or fallback
    return fallback


def _description_from_markdown(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        if line.lower() == "## purpose":
            for candidate in lines[index + 1 :]:
                if candidate and not candidate.startswith("#"):
                    return candidate.lstrip("- ").strip()
    for line in lines:
        if line and not line.startswith("#"):
            return line.lstrip("- ").strip()
    return "GestureFlow MAS skill pack."


@lru_cache(maxsize=1)
def list_skill_packs() -> tuple[SkillPack, ...]:
    packs: list[SkillPack] = []
    for pack_id in EXPECTED_SKILL_PACKS:
        skill_path = SKILL_PACK_ROOT / pack_id / "SKILL.md"
        content = skill_path.read_text(encoding="utf-8")
        packs.append(
            SkillPack(
                pack_id=pack_id,
                title=_title_from_markdown(content, pack_id),
                description=_description_from_markdown(content),
                path=str(skill_path),
                content=content,
            )
        )
    return tuple(packs)


def read_skill_pack(pack_id: str) -> SkillPack:
    for pack in list_skill_packs():
        if pack.pack_id == pack_id:
            return pack
    raise KeyError(pack_id)


def skill_pack_cards() -> list[dict[str, str]]:
    return [
        {
            "id": pack.pack_id,
            "title": pack.title,
            "description": pack.description,
            "path": pack.path,
        }
        for pack in list_skill_packs()
    ]


__all__ = [
    "EXPECTED_SKILL_PACKS",
    "SKILL_PACK_ROOT",
    "SkillPack",
    "list_skill_packs",
    "read_skill_pack",
    "skill_pack_cards",
]

