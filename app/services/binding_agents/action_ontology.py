"""Natural-language action ontology for the binding MAS.

This module keeps broad user wording out of the low-level parser.  The agent can
map phrases like "поставить видео на паузу" or "resume playback" to the same
structured executor contract without scattering one-off string checks through
the orchestrator.
"""
from __future__ import annotations

import re
from typing import Any


ACTION_START_PATTERN = (
    r"(?:откр|запуст|включ|показ|покаж|уведом|подожд|нажм|сделай|"
    r"увелич|уменьш|заблок|скрин|сайт|постав|останов|приостанов|"
    r"продолж|возобнов|воспроизвед|повыс|пониз|прибав|убав|пауза|"
    r"pause|play|resume|next|prev|"
    r"previous|следующ|предыдущ)"
)

INLINE_ACTION_START_PATTERN = (
    r"(?:откр|запуст|включ|показ|покаж|подожд|нажм|сделай|"
    r"увелич|уменьш|заблок|скрин|постав|останов|приостанов|"
    r"продолж|возобнов|воспроизвед|повыс|пониз|прибав|убав|"
    r"пауза|pause|следующ|предыдущ)"
)

MEDIA_CONTEXT_MARKERS: tuple[str, ...] = (
    "видео",
    "ролик",
    "ютуб",
    "youtube",
    "плеер",
    "player",
    "медиа",
    "media",
    "музык",
    "music",
    "аудио",
    "audio",
    "трек",
    "track",
    "песн",
    "song",
    "воспроизвед",
    "playback",
)

WAIT_CONTEXT_MARKERS: tuple[str, ...] = (
    "подожд",
    "жди",
    "ждать",
    "wait",
    "задерж",
    "сек",
    "second",
    "минут",
    "minute",
    "таймер",
)


def normalize_action_text(value: str) -> str:
    return (value or "").strip().lower().replace("ё", "е")


def parse_action_intent(text: str) -> dict[str, Any] | None:
    """Return a structured action from the action ontology, if recognized."""
    return parse_media_action(text) or parse_volume_action(text)


def parse_volume_action(text: str) -> dict[str, Any] | None:
    lower = normalize_action_text(text)
    if not lower:
        return None
    has_volume_target = any(
        marker in lower
        for marker in (
            "громк",
            "звук",
            "звука",
            "volume",
            "audio volume",
            "sound",
        )
    )
    if not has_volume_target:
        return None
    if any(
        marker in lower
        for marker in (
            "увелич",
            "повы",
            "прибав",
            "выше",
            "громче",
            "volume up",
            "raise",
            "increase",
            "+",
        )
    ):
        return {"action": "volume_up", "platform": "macos"}
    if any(
        marker in lower
        for marker in (
            "уменьш",
            "пониз",
            "убав",
            "ниже",
            "тише",
            "volume down",
            "lower",
            "decrease",
            "-",
        )
    ):
        return {"action": "volume_down", "platform": "macos"}
    return None


def parse_media_action(text: str) -> dict[str, Any] | None:
    lower = normalize_action_text(text)
    if not lower:
        return None

    if _has_next_media_intent(lower):
        return _media_spec("next")
    if _has_previous_media_intent(lower):
        return _media_spec("prev")
    if _has_toggle_media_intent(lower):
        return _media_spec("play_pause")
    if _has_pause_media_intent(lower):
        return _media_spec("pause")
    if _has_play_media_intent(lower):
        return _media_spec("play")
    return None


def _media_spec(kind: str) -> dict[str, Any]:
    return {"action": "media_key", "platform": "macos", "kind": kind}


def _has_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _has_media_context(text: str) -> bool:
    return _has_marker(text, MEDIA_CONTEXT_MARKERS)


def _looks_like_wait_request(text: str) -> bool:
    return _has_marker(text, WAIT_CONTEXT_MARKERS) and bool(
        re.search(r"\d+(?:[,.]\d+)?\s*(?:сек|second|минут|minute)?", text)
    )


def _has_next_media_intent(text: str) -> bool:
    if not re.search(r"\b(?:next|nexttrack|следующ\w*)\b", text):
        return False
    return _has_media_context(text) or re.search(r"\b(?:track|song)\b", text) is not None


def _has_previous_media_intent(text: str) -> bool:
    if not re.search(r"\b(?:prev|previous|prevtrack|предыдущ\w*)\b", text):
        return False
    return _has_media_context(text) or re.search(r"\b(?:track|song)\b", text) is not None


def _has_toggle_media_intent(text: str) -> bool:
    toggle_patterns = (
        r"\bplay[_\s-]?pause\b",
        r"\bплей\s+пауза\b",
        r"\b(?:пауза|pause)\s+(?:или|/)\s+(?:продолж|play|resume)",
        r"\b(?:play|resume|продолж)\s+(?:или|/)\s+(?:пауза|pause)",
        r"\bстарт\s*/?\s*стоп\b",
    )
    return any(re.search(pattern, text) for pattern in toggle_patterns)


def _has_pause_media_intent(text: str) -> bool:
    explicit_pause = any(
        re.search(pattern, text)
        for pattern in (
            r"\bпостав\w*\s+(?:.+?\s+)?на\s+пауз",
            r"\bстав\w*\s+(?:.+?\s+)?на\s+пауз",
            r"\b(?:при)?останов\w+\s+(?:видео|ролик|музык|медиа|аудио|"
            r"воспроизвед)",
            r"\bpause\s+(?:video|media|music|audio|playback)\b",
            r"\bstop\s+(?:video|media|music|audio|playback)\b",
        )
    )
    if explicit_pause:
        return True
    if _looks_like_wait_request(text):
        return False
    pause_word = "пауза" in text or "pause" in text
    return pause_word and _has_media_context(text)


def _has_play_media_intent(text: str) -> bool:
    explicit_play = any(
        re.search(pattern, text)
        for pattern in (
            r"\b(?:продолж|возобнов)\w*\s+(?:видео|ролик|музык|медиа|"
            r"аудио|воспроизвед)",
            r"\b(?:воспроизвед|проигра)\w+\s+(?:видео|ролик|музык|медиа|аудио)",
            r"\b(?:play|resume)\s+(?:video|media|music|audio|playback)\b",
        )
    )
    if explicit_play:
        return True
    play_word = re.search(r"\bplay\b", text) is not None
    return play_word and _has_media_context(text)
