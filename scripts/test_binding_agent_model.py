#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-test the binding agent with Mistral or the local mock provider."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.app_config import load_project_dotenv  # noqa: E402
from app.services.binding_agent import build_agent_binding_draft  # noqa: E402


DEFAULT_GESTURES = ("palm", "swipe_down", "ctrlz", "cntrz")


def _gesture_items(values: list[str]) -> list[dict[str, str]]:
    labels: list[str] = []
    for value in values:
        labels.extend(part.strip() for part in value.split(",") if part.strip())
    if not labels:
        labels = list(DEFAULT_GESTURES)
    return [{"label": label} for label in labels]


def _requested_mistral(provider: str) -> bool:
    return provider.strip().lower().replace("_", "-") in {
        "mistral",
        "mistral-api",
        "llm",
    }


def _mistral_used(draft: dict[str, object]) -> bool:
    trace = draft.get("agentTrace")
    if not isinstance(trace, list):
        return False
    for item in trace:
        if not isinstance(item, dict):
            continue
        if item.get("agent") == "Mistral Agent" and item.get("status") == "ok":
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one binding-agent prompt through Mistral or mock."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="жест cntrz привяжи к открытию safari",
        help="Natural-language binding request.",
    )
    parser.add_argument(
        "--provider",
        default="mistral",
        help="Provider: mistral, mistral-api, llm, or mock.",
    )
    parser.add_argument(
        "--gesture",
        action="append",
        default=[],
        help="Known gesture label. Can be passed multiple times or comma-separated.",
    )
    parser.add_argument(
        "--current-gesture",
        default="",
        help="Gesture currently selected in the UI.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Return success even if Mistral falls back to the local mock parser.",
    )
    args = parser.parse_args(argv)

    load_project_dotenv()
    draft = build_agent_binding_draft(
        args.prompt,
        _gesture_items(args.gesture),
        current_gesture=args.current_gesture,
        provider=args.provider,
    )
    print(json.dumps(draft, ensure_ascii=False, indent=2))

    if _requested_mistral(args.provider) and not _mistral_used(draft):
        print(
            "Mistral model was not used. Check MISTRAL_API_KEY or pass "
            "--allow-fallback.",
            file=sys.stderr,
        )
        return 0 if args.allow_fallback else 3
    return 0 if draft.get("canApply") else 2


if __name__ == "__main__":
    raise SystemExit(main())
