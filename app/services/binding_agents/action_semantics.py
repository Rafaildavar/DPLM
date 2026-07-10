"""Semantic action goals, candidate compilation, and arbitration."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.services.binding_agents.contracts import (
    ActionCandidate,
    RiskLevel,
    TaskFrame,
    TaskOperation,
)
from app.services.binding_agents.skills.runtime import DEFAULT_SKILL_RUNTIME


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower().replace("ё", "е"))


def _action_segment(text: str) -> str:
    raw = text or ""
    match = re.search(
        r"(?:привяж\w*|назнач\w*|сохрани\w*)\s+.+?\s+(?:к|на|для)\s+(?P<action>.+)$",
        raw,
        re.IGNORECASE,
    )
    return match.group("action").strip() if match else raw.strip()


@dataclass(frozen=True)
class ActionGoal:
    kind: str
    verb: str = ""
    target: str = ""
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    negated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "verb": self.verb,
            "target": self.target,
            "confidence": round(float(self.confidence), 4),
            "evidence": list(self.evidence),
            "negated": self.negated,
        }


KEY_ALIASES: dict[str, str] = {
    "пробел": "space",
    "space": "space",
    "spacebar": "space",
    "ввод": "enter",
    "enter": "enter",
    "return": "enter",
    "escape": "escape",
    "esc": "escape",
    "таб": "tab",
    "tab": "tab",
    "удаление": "delete",
    "delete": "delete",
    "backspace": "backspace",
    "вверх": "up",
    "вниз": "down",
    "влево": "left",
    "вправо": "right",
}


def extract_action_goal(text: str, *, frame: TaskFrame | None = None) -> ActionGoal:
    segment = _norm(_action_segment(text))
    negated = bool(frame and frame.negated)
    if negated:
        return ActionGoal("cancelled", confidence=1.0, negated=True, evidence=("task_frame.negated",))

    if any(marker in segment for marker in ("громк", "volume", "звук")):
        if any(marker in segment for marker in ("увелич", "повыс", "прибав", "громче", "volume up")):
            return ActionGoal("volume_up", "increase", "volume", 0.98, ("volume+increase",))
        if any(marker in segment for marker in ("уменьш", "пониз", "убав", "тише", "volume down")):
            return ActionGoal("volume_down", "decrease", "volume", 0.98, ("volume+decrease",))

    close = next(
        (
            marker
            for marker in ("закрыт", "закрой", "закрыть", "заверш", "выйти из", "quit", "close")
            if marker in segment
        ),
        "",
    )
    if close:
        return ActionGoal("quit_app", close, confidence=0.97, evidence=(close,))

    if any(marker in segment for marker in ("масштаб", "zoom in", "zoom out")):
        if any(marker in segment for marker in ("увелич", "приблиз", "zoom in", "крупнее")):
            return ActionGoal("zoom_in", "increase", "page", 0.96, ("zoom+increase",))
        if any(marker in segment for marker in ("уменьш", "отдал", "zoom out", "мельче")):
            return ActionGoal("zoom_out", "decrease", "page", 0.96, ("zoom+decrease",))

    press = next(
        (marker for marker in ("нажати", "нажать", "нажми", "press") if marker in segment),
        "",
    )
    if press:
        for alias, key in KEY_ALIASES.items():
            matched = (
                alias in segment
                if re.search(r"[а-я]", alias)
                else re.search(
                    rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                    segment,
                )
                is not None
            )
            if matched:
                return ActionGoal("press", press, key, 0.97, (f"key:{alias}",))

    if any(marker in segment for marker in ("bluetooth", "блютуз", "wi-fi", "wifi")) and any(
        marker in segment for marker in ("включ", "выключ", "toggle", "enable", "disable")
    ):
        return ActionGoal("system_toggle", "toggle", segment, 0.96, ("system_setting",))

    if re.search(r"https?://|www\.", segment):
        return ActionGoal("open_url", "open", confidence=0.99, evidence=("url",))

    open_verb = next(
        (
            marker
            for marker in ("откры", "открой", "запуст", "open", "launch", "start")
            if marker in segment
        ),
        "",
    )
    if open_verb:
        return ActionGoal("open", open_verb, confidence=0.94, evidence=(open_verb,))

    if re.search(r"(?:command|cmd|ctrl|control|⌘|⌃)\s*\+", segment):
        return ActionGoal("hotkey", confidence=0.97, evidence=("hotkey",))

    return ActionGoal("unknown", confidence=0.0)


def compile_action_candidate(
    text: str,
    legacy_spec: dict[str, Any] | None,
    *,
    frame: TaskFrame | None = None,
    source: str = "local_parser",
) -> tuple[ActionGoal, ActionCandidate]:
    goal = extract_action_goal(text, frame=frame)
    spec = dict(legacy_spec or {})
    issues: list[str] = []
    confidence = 0.0
    skill_id = ""
    risk = RiskLevel.LOW
    confirmation = False

    if goal.kind == "cancelled":
        issues.append("negated_request")
        spec = {}
    elif spec.get("action") == "sequence" and frame and frame.sequence_requested:
        confidence = 0.96
        skill_id = "scenario.compose"
        confirmation = any(
            isinstance(step, dict) and step.get("action") in {"quit_app", "lock_screen", "run_script"}
            for step in spec.get("steps") or []
        )
        risk = RiskLevel.MEDIUM if confirmation else RiskLevel.LOW
    elif goal.kind == "press":
        spec = {"action": "press", "platform": "macos", "key": goal.target}
        confidence = goal.confidence
        skill_id = "macos.keyboard.press"
    elif goal.kind in {"zoom_in", "zoom_out"}:
        spec = {
            "action": "key_combination",
            "platform": "macos",
            "keys": ["command", "+" if goal.kind == "zoom_in" else "-"],
        }
        confidence = goal.confidence
        skill_id = "macos.page.zoom"
    elif goal.kind == "quit_app":
        app = str(spec.get("app") or "").strip()
        if spec.get("action") == "open_app" and app:
            spec = {"action": "quit_app", "platform": "macos", "app": app}
            confidence = goal.confidence
            skill_id = "macos.app.quit"
            risk = RiskLevel.MEDIUM
            confirmation = True
        elif spec.get("action") == "quit_app" and app:
            confidence = goal.confidence
            skill_id = "macos.app.quit"
            risk = RiskLevel.MEDIUM
            confirmation = True
        else:
            issues.append("quit_target_missing")
            spec = {}
    elif goal.kind == "system_toggle":
        issues.append("unsupported_system_toggle")
        spec = {}
    elif not spec:
        issues.append("action_unresolved")
    else:
        action = str(spec.get("action") or "")
        expected = {
            "volume_up": {"volume_up"},
            "volume_down": {"volume_down"},
            "open_url": {"open_url"},
            "hotkey": {"key_combination"},
            "open": {"open_app", "open_url", "open_path"},
        }.get(goal.kind)
        if expected is not None and action not in expected:
            issues.append(f"semantic_mismatch:{goal.kind}:{action}")
        elif (
            frame is not None
            and frame.operation == TaskOperation.UPDATE
            and goal.kind == "unknown"
            and action == "open_app"
        ):
            issues.append("update_has_no_action_change")
        else:
            confidence = max(0.72, goal.confidence)
            skill_id = _skill_for_action(action)

    if not issues and spec:
        return goal, DEFAULT_SKILL_RUNTIME.prepare_candidate(
            spec,
            frame=frame,
            preferred_skill=skill_id,
            source=source,
            confidence=confidence,
            evidence=goal.evidence,
            risk=risk,
            requires_confirmation=confirmation,
        )
    return goal, ActionCandidate(
        action_spec=spec,
        source=source,
        confidence=confidence,
        evidence=goal.evidence,
        issues=tuple(issues),
        skill_id=skill_id,
        skill_version="1.0.0" if skill_id else "",
        risk=risk,
        requires_confirmation=confirmation,
    )


def _skill_for_action(action: str) -> str:
    return {
        "open_app": "macos.app.open",
        "quit_app": "macos.app.quit",
        "open_url": "browser.open_url",
        "open_path": "macos.path.open",
        "key_combination": "macos.keyboard.hotkey",
        "press": "macos.keyboard.press",
        "scroll": "macos.page.scroll",
        "media_key": "macos.media.control",
        "volume_up": "macos.volume.control",
        "volume_down": "macos.volume.control",
        "mute_toggle": "macos.volume.control",
        "brightness_up": "macos.brightness.control",
        "brightness_down": "macos.brightness.control",
        "lock_screen": "macos.screen.lock",
        "screenshot": "macos.screen.capture",
        "notify": "macos.notification.show",
        "wait": "scenario.wait",
        "run_script": "macos.script.run",
        "sequence": "scenario.compose",
    }.get(action, "")


@dataclass(frozen=True)
class ArbitrationResult:
    selected: ActionCandidate | None
    candidates: tuple[ActionCandidate, ...]
    ambiguous: bool = False
    margin: float = 0.0


class CandidateArbiter:
    def __init__(self, *, min_confidence: float = 0.68, min_margin: float = 0.08) -> None:
        self.min_confidence = min_confidence
        self.min_margin = min_margin

    def choose(self, candidates: Iterable[ActionCandidate]) -> ArbitrationResult:
        items = tuple(candidates)
        valid = sorted(
            (item for item in items if item.valid),
            key=lambda item: item.confidence,
            reverse=True,
        )
        if not valid or valid[0].confidence < self.min_confidence:
            return ArbitrationResult(None, items)
        top = valid[0]
        second = next(
            (
                item
                for item in valid[1:]
                if json.dumps(item.action_spec, sort_keys=True)
                != json.dumps(top.action_spec, sort_keys=True)
            ),
            None,
        )
        margin = top.confidence - second.confidence if second else top.confidence
        if second is not None and margin < self.min_margin:
            return ArbitrationResult(None, items, ambiguous=True, margin=margin)
        return ArbitrationResult(top, items, margin=margin)


__all__ = [
    "ActionGoal",
    "ArbitrationResult",
    "CandidateArbiter",
    "compile_action_candidate",
    "extract_action_goal",
]
