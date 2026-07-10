"""Gesture alias memory and semantic-ish label resolver.

The binding agent receives short, human phrases such as "лайк" or
"свайп налево", while the ML model usually exposes stable class labels like
``thumbs_up`` or ``swipe_left``.  This module keeps that translation outside of
the prompt parser so user-trained gestures can bring their own aliases without
code changes.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any, Iterable


GESTURE_ALIASES_ENV = "GESTUREBIND_GESTURE_ALIASES"
LEGACY_GESTURE_ALIASES_ENV = "DPLM_GESTURE_ALIASES"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GESTURE_ALIASES_PATH = (
    PROJECT_ROOT / "data" / "binding_agent" / "gesture_aliases.json"
)

GESTURE_TOKEN_ALIASES: dict[str, str] = {
    "gesture": "",
    "жест": "",
    "свайп": "swipe",
    "сваип": "swipe",
    "смахивание": "swipe",
    "мах": "swipe",
    "swipe": "swipe",
    "вверх": "up",
    "верх": "up",
    "ап": "up",
    "up": "up",
    "вниз": "down",
    "низ": "down",
    "даун": "down",
    "down": "down",
    "влево": "left",
    "налево": "left",
    "лево": "left",
    "левый": "left",
    "left": "left",
    "вправо": "right",
    "направо": "right",
    "право": "right",
    "правый": "right",
    "right": "right",
    "ладонь": "palm",
    "ладонью": "palm",
    "пальма": "palm",
    "palm": "palm",
    "кулак": "fist",
    "кулаком": "fist",
    "fist": "fist",
    "ган": "gun",
    "пистолет": "gun",
    "gun": "gun",
    "щипок": "pinch",
    "pinch": "pinch",
    "ок": "ok",
    "okay": "ok",
    "лайк": "like",
    "лайка": "like",
    "лайком": "like",
    "like": "like",
    "палец": "finger",
    "большой": "thumb",
    "thumb": "thumb",
    "thumbs": "thumb",
}

SYSTEM_GESTURE_ALIASES: dict[str, tuple[str, ...]] = {
    "like": (
        "лайк",
        "лайка",
        "лайком",
        "thumb up",
        "thumbs up",
        "палец вверх",
        "большой палец вверх",
    ),
    "thumb_up": (
        "лайк",
        "лайка",
        "лайком",
        "like",
        "thumb up",
        "thumbs up",
        "палец вверх",
        "большой палец вверх",
    ),
    "thumbs_up": (
        "лайк",
        "лайка",
        "лайком",
        "like",
        "thumb up",
        "thumbs up",
        "палец вверх",
        "большой палец вверх",
    ),
    "swipe_left": (
        "свайп влево",
        "свайп налево",
        "мах влево",
        "swipe left",
        "swipeleft",
    ),
    "swipe_right": (
        "свайп вправо",
        "свайп направо",
        "мах вправо",
        "swipe right",
        "swiperight",
    ),
    "swipe_up": ("свайп вверх", "мах вверх", "swipe up", "swipeup"),
    "swipe_down": ("свайп вниз", "мах вниз", "swipe down", "swipedown"),
}


def _norm(value: str) -> str:
    return (value or "").strip().lower().replace("ё", "е")


@dataclass(frozen=True)
class GestureAliasCandidate:
    label: str
    alias: str
    normalized: str
    source: str
    confidence: float = 1.0


@dataclass(frozen=True)
class GestureAliasMatch:
    label: str
    confidence: float
    source: str
    matched_alias: str
    query: str
    ambiguous: bool = False
    suggestions: list[str] = field(default_factory=list)


def gesture_alias_registry_path() -> Path:
    raw = os.getenv(GESTURE_ALIASES_ENV) or os.getenv(LEGACY_GESTURE_ALIASES_ENV)
    return Path(raw).expanduser() if raw else DEFAULT_GESTURE_ALIASES_PATH


def normalize_gesture_alias(value: str) -> str:
    raw = _norm(value)
    raw = raw.replace("_", " ").replace("-", " ")
    tokens = re.findall(r"[a-zа-я0-9]+", raw, re.IGNORECASE)
    normalized: list[str] = []
    for token in tokens:
        mapped = GESTURE_TOKEN_ALIASES.get(token, token)
        if mapped:
            normalized.append(mapped)
    return "_".join(normalized)


def gesture_alias_query_from_text(text: str) -> str:
    return _gesture_query_from_text(text)


def _split_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [
            part.strip()
            for part in re.split(r"[,;\n]+", value)
            if part.strip()
        ]
    if isinstance(value, Iterable):
        out: list[str] = []
        for item in value:
            out.extend(_split_aliases(item))
        return out
    return []


def _gesture_metadata_aliases(gesture: dict[str, Any]) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    for key in (
        "label",
        "displayName",
        "display_name",
        "name",
        "title",
        "description",
    ):
        value = str(gesture.get(key) or "").strip()
        if value:
            aliases.append((value, key))
    for key in ("aliases", "gestureAliases", "alias"):
        for value in _split_aliases(gesture.get(key)):
            aliases.append((value, key))
    return aliases


def _add_candidate(
    items: list[GestureAliasCandidate],
    *,
    label: str,
    alias: str,
    source: str,
    confidence: float = 1.0,
) -> None:
    clean_label = str(label or "").strip()
    clean_alias = str(alias or "").strip()
    normalized = normalize_gesture_alias(clean_alias)
    if not clean_label or not normalized:
        return
    candidate = GestureAliasCandidate(
        label=clean_label,
        alias=clean_alias,
        normalized=normalized,
        source=source,
        confidence=float(confidence),
    )
    if candidate not in items:
        items.append(candidate)


class GestureAliasRegistry:
    """JSON-backed memory for labels and their user-facing aliases."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else gesture_alias_registry_path()

    def resolve(
        self,
        text: str,
        *,
        labels: Iterable[str],
        gestures: Iterable[dict[str, Any]] | None = None,
        threshold: float = 0.78,
    ) -> GestureAliasMatch | None:
        clean_labels = [str(label or "").strip() for label in labels if str(label or "").strip()]
        candidates = self.candidates(clean_labels, gestures=gestures)
        if not candidates:
            return None

        query = _gesture_query_from_text(text)
        normalized_query = normalize_gesture_alias(query)
        normalized_text = normalize_gesture_alias(text)

        direct = self._direct_match(
            normalized_query,
            normalized_text,
            candidates,
        )
        if direct is not None:
            return direct

        scored = self._fuzzy_matches(normalized_query or normalized_text, candidates)
        if not scored:
            return None
        best = scored[0]
        if best.confidence < threshold:
            suggestions = _unique_labels(item.label for item in scored[:3])
            if suggestions:
                return GestureAliasMatch(
                    label="",
                    confidence=best.confidence,
                    source="alias_registry:fuzzy",
                    matched_alias=best.alias,
                    query=query,
                    ambiguous=True,
                    suggestions=suggestions,
                )
            return None

        same_score = [
            item
            for item in scored
            if item.label != best.label and abs(item.confidence - best.confidence) <= 0.04
        ]
        if same_score:
            return GestureAliasMatch(
                label="",
                confidence=best.confidence,
                source="alias_registry:ambiguous",
                matched_alias=best.alias,
                query=query,
                ambiguous=True,
                suggestions=_unique_labels([best.label, *[item.label for item in same_score]][:3]),
            )
        self.touch(best.label, best.alias, source=best.source)
        return GestureAliasMatch(
            label=best.label,
            confidence=best.confidence,
            source=best.source,
            matched_alias=best.alias,
            query=query,
        )

    def candidates(
        self,
        labels: Iterable[str],
        *,
        gestures: Iterable[dict[str, Any]] | None = None,
    ) -> list[GestureAliasCandidate]:
        clean_labels = [str(label or "").strip() for label in labels if str(label or "").strip()]
        by_lower = {label.lower(): label for label in clean_labels}
        items: list[GestureAliasCandidate] = []

        for label in clean_labels:
            _add_candidate(items, label=label, alias=label, source="label", confidence=1.0)
            label_words = label.replace("_", " ").replace("-", " ")
            _add_candidate(items, label=label, alias=label_words, source="label_words", confidence=0.96)
            for alias in SYSTEM_GESTURE_ALIASES.get(normalize_gesture_alias(label), ()):
                _add_candidate(items, label=label, alias=alias, source="system_alias", confidence=0.94)

        for gesture in gestures or ():
            if not isinstance(gesture, dict):
                continue
            label = str(gesture.get("label") or "").strip()
            if not label:
                continue
            canonical = by_lower.get(label.lower(), label)
            for alias, source in _gesture_metadata_aliases(gesture):
                _add_candidate(
                    items,
                    label=canonical,
                    alias=alias,
                    source=f"gesture_metadata:{source}",
                    confidence=0.97 if source != "description" else 0.82,
                )

        for label, aliases in self._load_aliases().items():
            canonical = by_lower.get(label.lower())
            if canonical is None:
                continue
            for alias_data in aliases:
                alias = str(alias_data.get("value") or "").strip()
                source = str(alias_data.get("source") or "memory")
                confidence = float(alias_data.get("confidence") or 0.9)
                _add_candidate(
                    items,
                    label=canonical,
                    alias=alias,
                    source=f"memory:{source}",
                    confidence=confidence,
                )
        return items

    def learn(
        self,
        label: str,
        alias: str,
        *,
        source: str = "user",
        confidence: float = 0.92,
    ) -> bool:
        clean_label = str(label or "").strip()
        clean_alias = str(alias or "").strip()
        normalized = normalize_gesture_alias(clean_alias)
        if not clean_label or not normalized:
            return False
        data = self._load_raw()
        gestures = data.setdefault("gestures", {})
        item = gestures.setdefault(clean_label, {"label": clean_label, "aliases": []})
        aliases = item.setdefault("aliases", [])
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for entry in aliases:
            if normalize_gesture_alias(str(entry.get("value") or "")) == normalized:
                entry["value"] = clean_alias
                entry["normalized"] = normalized
                entry["source"] = str(entry.get("source") or source)
                entry["confidence"] = max(float(entry.get("confidence") or 0.0), float(confidence))
                entry["lastUsedAt"] = now
                entry["uses"] = int(entry.get("uses") or 0) + 1
                item["updatedAt"] = now
                self._save_raw(data)
                return True
        aliases.append(
            {
                "value": clean_alias,
                "normalized": normalized,
                "source": source,
                "confidence": float(confidence),
                "createdAt": now,
                "lastUsedAt": now,
                "uses": 1,
            }
        )
        item["updatedAt"] = now
        self._save_raw(data)
        return True

    def touch(self, label: str, alias: str, *, source: str = "runtime") -> None:
        if str(source or "").startswith(("label", "gesture_metadata", "system_alias")):
            return
        self.learn(label, alias, source=source.replace("memory:", "") or "runtime")

    def _direct_match(
        self,
        normalized_query: str,
        normalized_text: str,
        candidates: list[GestureAliasCandidate],
    ) -> GestureAliasMatch | None:
        for haystack in (normalized_query, normalized_text):
            if not haystack:
                continue
            exact = [item for item in candidates if item.normalized == haystack]
            if exact:
                return _best_or_ambiguous(exact, normalized_query)
            contained = [
                item
                for item in candidates
                if _contains_alias(haystack, item.normalized)
            ]
            if contained:
                return _best_or_ambiguous(contained, normalized_query)
        return None
    def _fuzzy_matches(
        self,
        normalized_query: str,
        candidates: list[GestureAliasCandidate],
    ) -> list[GestureAliasCandidate]:
        if not normalized_query:
            return []
        by_alias = {item.normalized: item for item in candidates}
        close = get_close_matches(
            normalized_query,
            list(by_alias),
            n=5,
            cutoff=0.62,
        )
        scored: list[GestureAliasCandidate] = []
        for alias in close:
            item = by_alias[alias]
            score = SequenceMatcher(None, normalized_query, alias).ratio()
            scored.append(
                GestureAliasCandidate(
                    label=item.label,
                    alias=item.alias,
                    normalized=item.normalized,
                    source=f"{item.source}:fuzzy",
                    confidence=round(score * item.confidence, 3),
                )
            )
        scored.sort(key=lambda item: item.confidence, reverse=True)
        return scored

    def _load_aliases(self) -> dict[str, list[dict[str, Any]]]:
        raw = self._load_raw()
        gestures = raw.get("gestures")
        if not isinstance(gestures, dict):
            return {}
        out: dict[str, list[dict[str, Any]]] = {}
        for label, item in gestures.items():
            clean_label = str(label or "").strip()
            if not clean_label:
                continue
            aliases = item.get("aliases") if isinstance(item, dict) else item
            rows: list[dict[str, Any]] = []
            if isinstance(aliases, list):
                for alias in aliases:
                    if isinstance(alias, dict):
                        rows.append(dict(alias))
                    elif str(alias or "").strip():
                        rows.append({"value": str(alias).strip(), "source": "legacy"})
            out[clean_label] = rows
        return out

    def _load_raw(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "gestures": {}}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "gestures": {}}
        if not isinstance(loaded, dict):
            return {"version": 1, "gestures": {}}
        if "gestures" not in loaded:
            loaded = {"version": 1, "gestures": loaded}
        gestures = loaded.get("gestures")
        if not isinstance(gestures, dict):
            loaded["gestures"] = {}
        return loaded

    def _save_raw(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
        self.path.write_text(payload + "\n", encoding="utf-8")


def gesture_alias_proposal(
    label: str,
    alias: str,
    *,
    source: str = "selected_gesture",
    confidence: float = 0.9,
) -> dict[str, Any]:
    """Build a non-persistent alias proposal for explicit user approval."""
    clean_label = str(label or "").strip()
    clean_alias = str(alias or "").strip()
    normalized = normalize_gesture_alias(clean_alias)
    if not clean_label or not normalized:
        return {}
    if normalized == normalize_gesture_alias(clean_label):
        return {}
    return {
        "label": clean_label,
        "alias": clean_alias,
        "normalized": normalized,
        "source": source,
        "confidence": float(confidence),
        "status": "pending",
        "approvalRequired": True,
        "rememberOnApproval": True,
    }


def approve_gesture_alias_proposal(
    proposal: dict[str, Any],
    *,
    path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Persist an alias only after the binding carrying it was approved."""
    if not isinstance(proposal, dict) or not proposal.get("rememberOnApproval"):
        return None
    label = str(proposal.get("label") or "").strip()
    alias = str(proposal.get("alias") or "").strip()
    normalized = normalize_gesture_alias(alias)
    if (
        not label
        or not normalized
        or normalized != str(proposal.get("normalized") or normalized)
        or len(normalized.split("_")) > 5
    ):
        return None
    source = str(proposal.get("source") or "user_approved")
    confidence = min(1.0, max(0.0, float(proposal.get("confidence") or 0.9)))
    if not GestureAliasRegistry(path=path).learn(
        label,
        alias,
        source=source,
        confidence=confidence,
    ):
        return None
    return {
        **proposal,
        "status": "approved",
        "approved": True,
        "rememberOnApproval": False,
    }


def _contains_alias(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return re.search(
        rf"(?<![a-zа-я0-9]){re.escape(needle)}(?![a-zа-я0-9])",
        haystack,
        re.IGNORECASE,
    ) is not None


def _best_or_ambiguous(
    candidates: list[GestureAliasCandidate],
    query: str,
) -> GestureAliasMatch:
    candidates = sorted(candidates, key=lambda item: item.confidence, reverse=True)
    best = candidates[0]
    labels = _unique_labels(item.label for item in candidates)
    if len(labels) > 1:
        return GestureAliasMatch(
            label="",
            confidence=best.confidence,
            source="alias_registry:ambiguous",
            matched_alias=best.alias,
            query=query,
            ambiguous=True,
            suggestions=labels[:3],
        )
    return GestureAliasMatch(
        label=best.label,
        confidence=best.confidence,
        source=best.source,
        matched_alias=best.alias,
        query=query,
    )


def _unique_labels(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or "").strip()
        key = clean.lower()
        if clean and key not in seen:
            out.append(clean)
            seen.add(key)
    return out


def _gesture_query_from_text(text: str) -> str:
    raw = text or ""
    match = re.search(
        (
            r"(?:^|\s)(?:жест(?:ом|а)?|gesture)\s*[:=]?\s+"
            r"(?P<value>.+?)(?=\s+(?:привяж|привяз|откр|закр|запуст|"
            r"нажм|к\s|как\s|для\s|сделай|будет|должен)|[,.;]|$)"
        ),
        raw,
        re.IGNORECASE,
    )
    if match:
        return match.group("value").strip()
    generic_bind_match = re.search(
        (
            r"(?:привяж\w*|привяз\w*|сохрани\w*|назнач\w*)\s+"
            r"(?P<value>.+?)(?=\s+(?:к|на|для)\s+)"
        ),
        raw,
        re.IGNORECASE,
    )
    if generic_bind_match:
        value = generic_bind_match.group("value").strip()
        if normalize_gesture_alias(value) not in {
            "this",
            "eto",
            "это",
            "этот",
            "эту",
            "текущий",
            "текущии",
            "выбранный",
            "выбранныи",
        }:
            return value
    bind_match = re.search(
        (
            r"(?:привяж\w*|сохрани\w*|назнач\w*)\s+"
            r"(?P<value>.+?)(?=\s+(?:к|на)\s+(?:откр|закр|запуск|команд|действ|сценар))"
        ),
        raw,
        re.IGNORECASE,
    )
    if bind_match:
        return bind_match.group("value").strip()
    return raw.strip()
