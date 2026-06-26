"""Gesture taxonomy helpers for routing and model training.

The taxonomy separates gesture labels into three ML scopes:
``static``, ``quasi_static`` and ``dynamic``. Training code uses the same helper
as runtime code so a dynamic model cannot accidentally learn static labels.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "configs" / "gesture_taxonomy.json"

GESTURE_TYPE_STATIC = "static"
GESTURE_TYPE_QUASI_STATIC = "quasi_static"
GESTURE_TYPE_DYNAMIC = "dynamic"
SUPPORTED_GESTURE_TYPES = (
    GESTURE_TYPE_STATIC,
    GESTURE_TYPE_QUASI_STATIC,
    GESTURE_TYPE_DYNAMIC,
)


def _normalize_label(label: str) -> str:
    return str(label or "").strip().lower()


def parse_gesture_type_scope(scope: str | Iterable[str] | None) -> tuple[str, ...]:
    if scope is None:
        return ()
    if isinstance(scope, str):
        raw_items = scope.replace(";", ",").split(",")
    else:
        raw_items = [str(item) for item in scope]

    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        clean = item.strip().lower()
        if not clean:
            continue
        if clean not in SUPPORTED_GESTURE_TYPES:
            raise ValueError(f"unsupported gesture type: {clean}")
        if clean in seen:
            continue
        out.append(clean)
        seen.add(clean)
    return tuple(out)


@dataclass(frozen=True)
class GestureTaxonomy:
    source_path: Path
    label_to_type: dict[str, str]
    patterns_by_type: dict[str, tuple[str, ...]]
    default_type: str = GESTURE_TYPE_STATIC

    def gesture_type_for_label(self, label: str) -> str:
        key = _normalize_label(label)
        if not key:
            return self.default_type
        explicit = self.label_to_type.get(key)
        if explicit:
            return explicit
        for gesture_type in SUPPORTED_GESTURE_TYPES:
            for pattern in self.patterns_by_type.get(gesture_type, ()):
                if fnmatch.fnmatchcase(key, pattern):
                    return gesture_type
        return self.default_type

    def labels_for_types(
        self,
        labels: Iterable[str],
        gesture_types: str | Iterable[str] | None,
    ) -> list[str]:
        selected = set(parse_gesture_type_scope(gesture_types))
        materialized = [str(label).strip() for label in labels if str(label).strip()]
        if not selected:
            return _dedupe_labels(materialized)
        return _dedupe_labels(
            label
            for label in materialized
            if self.gesture_type_for_label(label) in selected
        )


def _dedupe_labels(labels: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        clean = str(label).strip()
        key = _normalize_label(clean)
        if not clean or key in seen:
            continue
        out.append(clean)
        seen.add(key)
    return out


def load_gesture_taxonomy(path: Path | str | None = None) -> GestureTaxonomy:
    source = Path(path) if path is not None else DEFAULT_TAXONOMY_PATH
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("gesture taxonomy must be a JSON object")

    default_type = str(raw.get("default_type") or GESTURE_TYPE_STATIC).strip().lower()
    if default_type not in SUPPORTED_GESTURE_TYPES:
        raise ValueError(f"unsupported default gesture type: {default_type}")

    label_to_type: dict[str, str] = {}
    types_raw = raw.get("types") or {}
    if not isinstance(types_raw, dict):
        raise ValueError("gesture taxonomy field 'types' must be an object")
    for gesture_type, labels in types_raw.items():
        clean_type = str(gesture_type or "").strip().lower()
        if clean_type not in SUPPORTED_GESTURE_TYPES:
            raise ValueError(f"unsupported gesture type: {clean_type}")
        if not isinstance(labels, list):
            raise ValueError(f"taxonomy labels for {clean_type} must be a list")
        for label in labels:
            key = _normalize_label(str(label))
            if key:
                label_to_type[key] = clean_type

    patterns_by_type: dict[str, tuple[str, ...]] = {}
    patterns_raw = raw.get("patterns") or {}
    if not isinstance(patterns_raw, dict):
        raise ValueError("gesture taxonomy field 'patterns' must be an object")
    for gesture_type, patterns in patterns_raw.items():
        clean_type = str(gesture_type or "").strip().lower()
        if clean_type not in SUPPORTED_GESTURE_TYPES:
            raise ValueError(f"unsupported pattern gesture type: {clean_type}")
        if not isinstance(patterns, list):
            raise ValueError(f"taxonomy patterns for {clean_type} must be a list")
        patterns_by_type[clean_type] = tuple(
            _normalize_label(str(pattern)) for pattern in patterns if str(pattern).strip()
        )

    return GestureTaxonomy(
        source_path=source,
        label_to_type=label_to_type,
        patterns_by_type=patterns_by_type,
        default_type=default_type,
    )


def labels_for_gesture_types(
    labels: Iterable[str],
    gesture_types: str | Iterable[str] | None,
    *,
    taxonomy_path: Path | str | None = None,
) -> list[str]:
    taxonomy = load_gesture_taxonomy(taxonomy_path)
    return taxonomy.labels_for_types(labels, gesture_types)
