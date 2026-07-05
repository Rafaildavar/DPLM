"""Research agent and user-approved action skill memory."""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from app.services.binding_agent import (
    AgentStep,
    BindingAgentContext,
    PROJECT_ROOT,
    _action_title,
    _norm,
)


RESEARCH_MEMORY_ENV = "DPLM_BINDING_RESEARCH_MEMORY"
RESEARCH_SKILL_ENV = "DPLM_BINDING_RESEARCH_SKILL"
RESEARCH_WEB_ENV = "DPLM_BINDING_RESEARCH_WEB"
DEFAULT_MEMORY_PATH = PROJECT_ROOT / "data" / "binding_agent" / "researched_actions.json"
DEFAULT_SKILL_PATH = (
    PROJECT_ROOT
    / "data"
    / "binding_agent"
    / "skill_packs"
    / "researched-actions"
    / "SKILL.md"
)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _memory_path() -> Path:
    raw = os.getenv(RESEARCH_MEMORY_ENV)
    return Path(raw).expanduser() if raw else DEFAULT_MEMORY_PATH


def _skill_path() -> Path:
    raw = os.getenv(RESEARCH_SKILL_ENV)
    return Path(raw).expanduser() if raw else DEFAULT_SKILL_PATH


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zа-я0-9]+", _norm(text), re.IGNORECASE)
        if len(token) > 2
    }


def _recipe_id(query: str, action_spec: dict[str, Any]) -> str:
    payload = json.dumps(
        {"query": _norm(query), "actionSpec": action_spec},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ResearchRecipe:
    title: str
    query: str
    platform: str
    action_spec: dict[str, Any]
    source_title: str = ""
    source_url: str = ""
    source_excerpt: str = ""
    confidence: float = 0.72
    learned: bool = False
    approved: bool = False
    recipe_id: str = ""

    def to_proposal(self, *, approval_required: bool) -> dict[str, Any]:
        recipe_id = self.recipe_id or _recipe_id(self.query, self.action_spec)
        return {
            "id": recipe_id,
            "title": self.title,
            "query": self.query,
            "platform": self.platform,
            "actionSpec": dict(self.action_spec),
            "actionTitle": _action_title(self.action_spec),
            "sourceTitle": self.source_title,
            "sourceUrl": self.source_url,
            "sourceExcerpt": self.source_excerpt,
            "confidence": float(self.confidence),
            "learned": bool(self.learned),
            "approved": bool(self.approved),
            "approvalRequired": bool(approval_required),
            "rememberOnApproval": bool(approval_required),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchRecipe":
        action_spec = data.get("actionSpec") or data.get("action_spec") or {}
        if not isinstance(action_spec, dict):
            action_spec = {}
        return cls(
            title=str(data.get("title") or ""),
            query=str(data.get("query") or ""),
            platform=str(data.get("platform") or "macos"),
            action_spec=dict(action_spec),
            source_title=str(data.get("sourceTitle") or data.get("source_title") or ""),
            source_url=str(data.get("sourceUrl") or data.get("source_url") or ""),
            source_excerpt=str(
                data.get("sourceExcerpt") or data.get("source_excerpt") or ""
            ),
            confidence=float(data.get("confidence") or 0.72),
            learned=bool(data.get("learned")),
            approved=bool(data.get("approved")),
            recipe_id=str(data.get("id") or data.get("recipe_id") or ""),
        )


@dataclass(frozen=True)
class ResearchQueryPlan:
    query: str
    platform: str
    intent: str
    allow_web: bool
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "platform": self.platform,
            "intent": self.intent,
            "allowWeb": self.allow_web,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class ResearchProvider(Protocol):
    def research(self, context: BindingAgentContext) -> ResearchRecipe | None:
        """Return a source-backed recipe or None when research cannot help."""


def _step_dict(step: AgentStep) -> dict[str, Any]:
    return {
        "agent": step.agent,
        "status": step.status,
        "message": step.message,
        "data": dict(step.data),
    }


class ResearchQueryPlanner:
    name = "Research Query Planner"

    def run(self, context: BindingAgentContext) -> tuple[AgentStep, ResearchQueryPlan]:
        lower = _norm(context.prompt)
        allow_web = str(os.getenv(RESEARCH_WEB_ENV) or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        intent = "macos_action_lookup"
        reason = "Запрос похож на действие macOS, которого нет в локальном parser."
        if any(marker in lower for marker in ("сайт", "url", "website", "адрес")):
            intent = "website_lookup"
            reason = "Нужен URL или проверенный способ открыть сайт."
        elif any(
            marker in lower
            for marker in ("экран", "space", "desktop", "рабочий стол")
        ):
            intent = "desktop_navigation_lookup"
            reason = "Нужна команда навигации macOS."
        plan = ResearchQueryPlan(
            query=context.prompt,
            platform="macos",
            intent=intent,
            allow_web=allow_web,
            confidence=0.74,
            reason=reason,
        )
        return (
            AgentStep(
                self.name,
                "ok",
                "Собрал план поиска рецепта.",
                {"plan": plan.to_dict()},
            ),
            plan,
        )


class ResearchRecipeValidator:
    name = "Research Recipe Validator"

    def run(
        self,
        recipe: ResearchRecipe | None,
        *,
        plan: ResearchQueryPlan,
        approval_required: bool,
        source: str,
    ) -> AgentStep:
        issues: list[str] = []
        action_spec = dict(recipe.action_spec) if recipe else {}
        if not recipe:
            issues.append("recipe_missing")
        if recipe and recipe.platform and recipe.platform != plan.platform:
            issues.append("platform_mismatch")
        if not action_spec.get("action"):
            issues.append("action_missing")
        if approval_required and recipe and not (
            recipe.source_title or recipe.source_url or recipe.source_excerpt
        ):
            issues.append("source_missing")
        status = "ok" if not issues else "blocked"
        return AgentStep(
            self.name,
            status,
            (
                "Проверил research-рецепт."
                if not issues
                else "Research-рецепт не прошёл проверку."
            ),
            {
                "issues": issues,
                "source": source,
                "approvalRequired": approval_required,
                "action": str(action_spec.get("action") or ""),
                "plan": plan.to_dict(),
            },
        )


class SkillMemoryWriterAgent:
    name = "Skill Memory Writer"

    def __init__(self, memory: ResearchMemoryStore) -> None:
        self.memory = memory

    def approve(self, proposal: dict[str, Any]) -> dict[str, Any]:
        saved = self.memory.approve(proposal)
        return {
            **saved,
            "writerAgent": self.name,
            "memoryPath": str(self.memory.memory_path),
            "skillPath": str(self.memory.skill_path),
        }


class ResearchMemoryStore:
    """Stores user-approved research recipes and renders them as a skill file."""

    def __init__(
        self,
        *,
        memory_path: Path | None = None,
        skill_path: Path | None = None,
    ) -> None:
        self.memory_path = memory_path or _memory_path()
        self.skill_path = skill_path or _skill_path()

    def load(self) -> list[ResearchRecipe]:
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, OSError):
            return []
        items = data.get("recipes") if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        return [
            ResearchRecipe.from_dict(item)
            for item in items
            if isinstance(item, dict)
            and isinstance(item.get("actionSpec") or item.get("action_spec"), dict)
        ]

    def find(self, prompt: str) -> ResearchRecipe | None:
        prompt_tokens = _token_set(prompt)
        if not prompt_tokens:
            return None
        best: tuple[float, ResearchRecipe] | None = None
        for recipe in self.load():
            if not recipe.approved:
                continue
            recipe_tokens = _token_set(recipe.query + " " + recipe.title)
            if not recipe_tokens:
                continue
            overlap = len(prompt_tokens & recipe_tokens) / max(1, len(recipe_tokens))
            if _norm(recipe.query) and _norm(recipe.query) in _norm(prompt):
                overlap = max(overlap, 1.0)
            if overlap >= 0.55 and (best is None or overlap > best[0]):
                best = (overlap, recipe)
        if best is None:
            return None
        recipe = best[1]
        return ResearchRecipe(
            title=recipe.title,
            query=recipe.query,
            platform=recipe.platform,
            action_spec=recipe.action_spec,
            source_title=recipe.source_title,
            source_url=recipe.source_url,
            source_excerpt=recipe.source_excerpt,
            confidence=recipe.confidence,
            learned=True,
            approved=True,
            recipe_id=recipe.recipe_id,
        )

    def approve(self, proposal: dict[str, Any]) -> dict[str, Any]:
        recipe = ResearchRecipe.from_dict(
            {
                **proposal,
                "approved": True,
                "learned": True,
                "id": str(proposal.get("id") or ""),
            }
        )
        recipe_id = recipe.recipe_id or _recipe_id(recipe.query, recipe.action_spec)
        saved = {
            "id": recipe_id,
            "title": recipe.title or _action_title(recipe.action_spec),
            "query": recipe.query,
            "platform": recipe.platform or "macos",
            "actionSpec": dict(recipe.action_spec),
            "sourceTitle": recipe.source_title,
            "sourceUrl": recipe.source_url,
            "sourceExcerpt": recipe.source_excerpt,
            "confidence": recipe.confidence,
            "approved": True,
            "learned": True,
            "approvedAt": datetime.now(timezone.utc).isoformat(),
        }
        current = [
            item.to_proposal(approval_required=False)
            for item in self.load()
            if (item.recipe_id or _recipe_id(item.query, item.action_spec)) != recipe_id
        ]
        current.append(saved)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(
            json.dumps({"recipes": current}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_skill_markdown(current)
        return saved

    def _write_skill_markdown(self, recipes: list[dict[str, Any]]) -> None:
        lines = [
            "# User Researched Actions",
            "",
            "## Purpose",
            "",
            "Remember user-approved action recipes discovered by the Research Agent.",
            "",
            "## Rules",
            "",
            "- Use only recipes approved by the user.",
            "- Keep `platform` and `actionSpec` exactly as approved unless validation fails.",
            "- If a stored recipe is ambiguous for a new phrase, ask for clarification.",
            "",
            "## Approved Recipes",
            "",
        ]
        for item in sorted(recipes, key=lambda row: str(row.get("title") or "")):
            spec = item.get("actionSpec") if isinstance(item, dict) else {}
            lines.extend(
                [
                    f"### {item.get('title') or _action_title(spec or {})}",
                    "",
                    f"- Query: `{item.get('query') or ''}`",
                    f"- Source: {item.get('sourceTitle') or 'user approved'}"
                    + (
                        f" ({item.get('sourceUrl')})"
                        if item.get("sourceUrl")
                        else ""
                    ),
                    "- actionSpec:",
                    "```json",
                    json.dumps(spec or {}, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
        self.skill_path.parent.mkdir(parents=True, exist_ok=True)
        self.skill_path.write_text("\n".join(lines), encoding="utf-8")


class SourceBackedResearchProvider:
    """Small allowlisted recipe provider used before live web search is added."""

    _recipes: tuple[ResearchRecipe, ...] = (
        ResearchRecipe(
            title="Перейти на рабочий стол слева",
            query="перелистнуть экран налево macos space desktop",
            platform="macos",
            action_spec={
                "action": "key_combination",
                "platform": "macos",
                "keys": ["ctrl", "left"],
            },
            source_title="Apple User Guide: Work in multiple spaces on Mac",
            source_url="https://support.apple.com/guide/mac-help/work-in-multiple-spaces-mh14112/mac",
            source_excerpt="Move between spaces with Control + Left or Right Arrow.",
            confidence=0.88,
        ),
        ResearchRecipe(
            title="Перейти на рабочий стол справа",
            query="перелистнуть экран направо macos space desktop",
            platform="macos",
            action_spec={
                "action": "key_combination",
                "platform": "macos",
                "keys": ["ctrl", "right"],
            },
            source_title="Apple User Guide: Work in multiple spaces on Mac",
            source_url="https://support.apple.com/guide/mac-help/work-in-multiple-spaces-mh14112/mac",
            source_excerpt="Move between spaces with Control + Left or Right Arrow.",
            confidence=0.88,
        ),
    )

    def research(self, context: BindingAgentContext) -> ResearchRecipe | None:
        prompt_tokens = _token_set(context.prompt)
        if not prompt_tokens:
            return None
        best: tuple[float, ResearchRecipe] | None = None
        for recipe in self._recipes:
            recipe_tokens = _token_set(recipe.query + " " + recipe.title)
            overlap = len(prompt_tokens & recipe_tokens) / max(1, len(recipe_tokens))
            if overlap >= 0.42 and (best is None or overlap > best[0]):
                best = (overlap, recipe)
        return best[1] if best else None


class AppleWebResearchProvider:
    """Allowlisted web adapter for official Apple action recipes."""

    def __init__(self, *, enabled: bool | None = None, timeout: float = 4.0) -> None:
        self.enabled = (
            str(os.getenv(RESEARCH_WEB_ENV) or "").strip().lower()
            in {"1", "true", "yes", "on"}
            if enabled is None
            else enabled
        )
        self.timeout = timeout

    def research(self, context: BindingAgentContext) -> ResearchRecipe | None:
        if not self.enabled:
            return None
        query = (
            f"site:support.apple.com macOS keyboard shortcut {context.prompt}"
        )
        for url in self._search_urls(query)[:3]:
            if not self._is_allowed_url(url):
                continue
            page = self._fetch_text(url)
            if not page:
                continue
            recipe = self._recipe_from_page(context, url, page)
            if recipe:
                return recipe
        return None

    def _search_urls(self, query: str) -> list[str]:
        search_url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode(
            {"q": query}
        )
        text = self._fetch_text(search_url)
        urls: list[str] = []
        for raw in re.findall(r"href=[\"']([^\"']+)[\"']", text):
            value = html.unescape(raw)
            parsed = urllib.parse.urlparse(value)
            if parsed.query:
                params = urllib.parse.parse_qs(parsed.query)
                if params.get("uddg"):
                    value = params["uddg"][0]
            if value.startswith("http") and value not in urls:
                urls.append(value)
        return urls

    def _fetch_text(self, url: str) -> str:
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "GestureFlowResearchAgent/1.0"},
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(700_000)
        except Exception:
            return ""
        text = raw.decode("utf-8", errors="ignore")
        text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.IGNORECASE | re.S)
        text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.IGNORECASE | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    def _is_allowed_url(self, url: str) -> bool:
        host = urllib.parse.urlparse(url).netloc.lower()
        return host.endswith("support.apple.com") or host.endswith("developer.apple.com")

    def _recipe_from_page(
        self,
        context: BindingAgentContext,
        url: str,
        page_text: str,
    ) -> ResearchRecipe | None:
        prompt = _norm(context.prompt)
        page = _norm(page_text)
        keys: list[str] = []
        title = ""
        if "mission control" in prompt and (
            "control-up arrow" in page
            or "control up arrow" in page
            or "control-up" in page
        ):
            keys = ["ctrl", "up"]
            title = "Открыть Mission Control"
        elif any(marker in prompt for marker in ("окна приложения", "application windows")) and (
            "control-down arrow" in page
            or "control down arrow" in page
            or "control-down" in page
        ):
            keys = ["ctrl", "down"]
            title = "Показать окна приложения"
        elif any(marker in prompt for marker in ("влево", "налево", "left")) and (
            "control-left arrow" in page or "control left arrow" in page
        ):
            keys = ["ctrl", "left"]
            title = "Перейти на рабочий стол слева"
        elif any(marker in prompt for marker in ("вправо", "направо", "right")) and (
            "control-right arrow" in page or "control right arrow" in page
        ):
            keys = ["ctrl", "right"]
            title = "Перейти на рабочий стол справа"
        if not keys:
            return None
        return ResearchRecipe(
            title=title,
            query=context.prompt,
            platform="macos",
            action_spec={
                "action": "key_combination",
                "platform": "macos",
                "keys": keys,
            },
            source_title="Apple Support",
            source_url=url,
            source_excerpt="Рецепт извлечён Research Agent с allowlisted Apple page.",
            confidence=0.76,
        )


class CompositeResearchProvider:
    def __init__(self, providers: tuple[ResearchProvider, ...]) -> None:
        self.providers = providers

    def research(self, context: BindingAgentContext) -> ResearchRecipe | None:
        for provider in self.providers:
            recipe = provider.research(context)
            if recipe:
                return recipe
        return None


class ResearchAgent:
    name = "Research Agent"

    def __init__(
        self,
        *,
        memory: ResearchMemoryStore | None = None,
        provider: ResearchProvider | None = None,
        planner: ResearchQueryPlanner | None = None,
        validator: ResearchRecipeValidator | None = None,
    ) -> None:
        self.memory = memory or ResearchMemoryStore()
        self.planner = planner or ResearchQueryPlanner()
        self.validator = validator or ResearchRecipeValidator()
        self.provider = provider or CompositeResearchProvider(
            (
                SourceBackedResearchProvider(),
                AppleWebResearchProvider(),
            )
        )

    def run(self, context: BindingAgentContext) -> AgentStep:
        started = time.perf_counter()
        planner_step, plan = self.planner.run(context)
        substeps = [_step_dict(planner_step)]
        learned = self.memory.find(context.prompt)
        if learned:
            validator_step = self.validator.run(
                learned,
                plan=plan,
                approval_required=False,
                source="user_skill_memory",
            )
            substeps.append(_step_dict(validator_step))
            if validator_step.status == "blocked":
                return AgentStep(
                    self.name,
                    "need_clarification",
                    "Одобренный skill-рецепт не прошёл проверку.",
                    {
                        "action_spec": {},
                        "source": "user_skill_memory_blocked",
                        "researchPipeline": substeps,
                        "durationMs": _elapsed_ms(started),
                    },
                )
            return AgentStep(
                self.name,
                "ok",
                "Нашёл одобренный пользователем skill-рецепт.",
                {
                    "action_spec": dict(learned.action_spec),
                    "source": "user_skill_memory",
                    "research": learned.to_proposal(approval_required=False),
                    "researchPipeline": substeps,
                    "durationMs": _elapsed_ms(started),
                },
            )

        recipe = self.provider.research(context)
        validator_step = self.validator.run(
            recipe,
            plan=plan,
            approval_required=True,
            source="source_backed_research" if recipe else "research_miss",
        )
        substeps.append(_step_dict(validator_step))
        if recipe is None:
            return AgentStep(
                self.name,
                "need_clarification",
                "Research Agent не нашёл проверенный рецепт.",
                {
                    "action_spec": {},
                    "source": "research_miss",
                    "query": context.prompt,
                    "researchPipeline": substeps,
                    "durationMs": _elapsed_ms(started),
                },
            )
        if validator_step.status == "blocked":
            return AgentStep(
                self.name,
                "need_clarification",
                "Research Agent нашёл рецепт, но validator его остановил.",
                {
                    "action_spec": {},
                    "source": "research_blocked",
                    "query": context.prompt,
                    "researchPipeline": substeps,
                    "durationMs": _elapsed_ms(started),
                },
            )

        return AgentStep(
            self.name,
            "needs_approval",
            "Нашёл source-backed рецепт; нужно одобрение пользователя.",
            {
                "action_spec": dict(recipe.action_spec),
                "source": "source_backed_research",
                "research": recipe.to_proposal(approval_required=True),
                "researchPipeline": substeps,
                "durationMs": _elapsed_ms(started),
            },
        )


def approve_research_proposal(
    proposal: dict[str, Any],
    *,
    memory_path: Path | None = None,
    skill_path: Path | None = None,
) -> dict[str, Any] | None:
    if not isinstance(proposal, dict) or not proposal:
        return None
    if not proposal.get("rememberOnApproval"):
        return None
    action_spec = proposal.get("actionSpec")
    if not isinstance(action_spec, dict) or not action_spec.get("action"):
        return None
    memory = ResearchMemoryStore(
        memory_path=memory_path,
        skill_path=skill_path,
    )
    return SkillMemoryWriterAgent(memory).approve(proposal)
