"""Multi-agent draft builder for gesture bindings.

The local parser pipeline stays deterministic by default. A Mistral provider can
be enabled through environment variables to test real model parsing while
keeping the same prompt -> agent trace -> draft binding contract.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import urllib.error
import urllib.request
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.services.binding_agents.action_ontology import (
    ACTION_START_PATTERN,
    INLINE_ACTION_START_PATTERN,
    parse_action_intent,
)
from app.services.binding_agents.contracts import (
    ActionCandidate,
    AgentStatus,
    AgentStep,
    BindingAgentContext,
    BindingAgentResult,
    ResultStatus,
    RiskLevel,
    TaskDomain,
    TaskEvidence,
    TaskFrame,
    TaskOperation,
)
from app.services.binding_agents.skills import (
    AgentSkill,
    skill_registry_cards as _skill_registry_cards,
    step_data_with_skills as _step_data_with_skills,
)
from app.services.binding_agents.skill_packs import skill_pack_cards as _skill_pack_cards


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MISTRAL_PROVIDER_ENV = "DPLM_BINDING_AGENT_PROVIDER"
MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"
MISTRAL_MODEL_ENV = "MISTRAL_MODEL"
MISTRAL_API_URL_ENV = "MISTRAL_API_URL"
MISTRAL_DEFAULT_MODEL = "mistral-small-latest"
MISTRAL_DEFAULT_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_BINDING_TEMPERATURE_ENV = "MISTRAL_BINDING_TEMPERATURE"
MISTRAL_ANSWER_TEMPERATURE_ENV = "MISTRAL_ANSWER_TEMPERATURE"
BINDING_AGENT_MLFLOW_ENV = "DPLM_BINDING_AGENT_MLFLOW"
BINDING_AGENT_MLFLOW_EXPERIMENT_ENV = "DPLM_BINDING_AGENT_MLFLOW_EXPERIMENT"
BINDING_AGENT_GENAI_TRACES_ENV = "DPLM_BINDING_AGENT_GENAI_TRACES"
BINDING_AGENT_GENAI_EVAL_ENV = "DPLM_BINDING_AGENT_GENAI_EVAL"
BINDING_AGENT_LOCAL_FIRST_ENV = "DPLM_BINDING_AGENT_LOCAL_FIRST"
BINDING_AGENT_REWRITE_ANSWERS_ENV = "DPLM_BINDING_AGENT_REWRITE_ANSWERS"

AGENT_ACTION_LABELS: dict[str, str] = {
    "open_app": "Открыть приложение",
    "open_path": "Открыть файл или папку",
    "open_url": "Открыть сайт",
    "key_combination": "Нажать сочетание клавиш",
    "press": "Нажать клавишу",
    "scroll": "Прокрутка",
    "wait": "Подождать",
    "notify": "Показать уведомление",
    "media_key": "Управлять медиа",
    "volume_up": "Увеличить громкость",
    "volume_down": "Уменьшить громкость",
    "mute_toggle": "Включить/выключить звук",
    "brightness_down": "Уменьшить яркость",
    "brightness_up": "Увеличить яркость",
    "lock_screen": "Заблокировать экран",
    "screenshot": "Сделать снимок экрана",
    "run_script": "Запустить Python-скрипт",
    "sequence": "Сценарий",
}

COMMON_APP_NAMES = (
    "Safari",
    "Telegram",
    "Preview",
    "Music",
    "Finder",
    "Chrome",
    "Visual Studio Code",
    "VS Code",
    "Terminal",
    "Notes",
    "Mail",
    "Calendar",
    "Reminders",
    "System Settings",
)

APP_ALIASES: dict[str, str] = {
    "сафари": "Safari",
    "safari": "Safari",
    "телеграм": "Telegram",
    "telegram": "Telegram",
    "джира": "Jira",
    "jira": "Jira",
    "заметки": "Notes",
    "notes": "Notes",
    "chat gpt": "ChatGPT",
    "chatgpt": "ChatGPT",
    "рамблер почта": "Rambler Mail",
    "рамблер почту": "Rambler Mail",
    "rambler mail": "Rambler Mail",
}

ABSTRACT_WORKFLOW_TARGET_MARKERS: tuple[str, ...] = (
    "мой рабочий день",
    "моего рабочего дня",
    "рабочий день",
    "рабочего дня",
    "рабочее место",
    "рабочего места",
    "рабочий процесс",
    "рабочего процесса",
    "рабочий старт",
    "рабочего старта",
    "мое утро",
    "моё утро",
    "моего утра",
    "мой день",
    "моего дня",
)

SITE_ALIASES: dict[str, str] = {
    "chat gpt": "https://chatgpt.com",
    "chatgpt": "https://chatgpt.com",
    "студент гуап": "https://new.guap.ru/targets/studs",
    "студента гуап": "https://new.guap.ru/targets/studs",
    "студенты гуап": "https://new.guap.ru/targets/studs",
    "студентов гуап": "https://new.guap.ru/targets/studs",
    "обучающимся гуап": "https://new.guap.ru/targets/studs",
    "обучающиеся гуап": "https://new.guap.ru/targets/studs",
    "почта рамблер": "https://mail.rambler.ru",
    "почту рамблер": "https://mail.rambler.ru",
    "рамблер почта": "https://mail.rambler.ru",
    "рамблер почту": "https://mail.rambler.ru",
    "rambler mail": "https://mail.rambler.ru",
}

KEY_ALIASES: dict[str, str] = {
    "cmd": "command",
    "command": "command",
    "⌘": "command",
    "ctrl": "ctrl",
    "control": "ctrl",
    "⌃": "ctrl",
    "option": "option",
    "alt": "option",
    "⌥": "option",
    "shift": "shift",
    "⇧": "shift",
    "fn": "fn",
    "space": "space",
    "пробел": "space",
    "enter": "enter",
    "return": "enter",
    "ввод": "enter",
    "esc": "escape",
    "escape": "escape",
    "tab": "tab",
    "delete": "delete",
    "backspace": "backspace",
    "pagedown": "pagedown",
    "page_down": "pagedown",
    "pageup": "pageup",
    "page_up": "pageup",
    "down": "down",
    "вниз": "down",
    "up": "up",
    "вверх": "up",
    "left": "left",
    "влево": "left",
    "right": "right",
    "вправо": "right",
    ",": ",",
    ".": ".",
    "+": "+",
    "-": "-",
}

SIMPLE_HOTKEYS: tuple[tuple[tuple[str, ...], list[str]], ...] = (
    (("отмена", "undo"), ["command", "z"]),
    (("повтор", "redo"), ["command", "shift", "z"]),
    (("копир", "copy"), ["command", "c"]),
    (("встав", "paste"), ["command", "v"]),
    (("поиск", "find"), ["command", "f"]),
)

GESTURE_WORD_ALIASES: dict[str, str] = {
    "gesture": "",
    "жест": "",
    "свайп": "swipe",
    "сваип": "swipe",
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
    "лево": "left",
    "left": "left",
    "вправо": "right",
    "право": "right",
    "right": "right",
    "ладонь": "palm",
    "пальма": "palm",
    "palm": "palm",
    "кулак": "fist",
    "fist": "fist",
    "ган": "gun",
    "пистолет": "gun",
    "gun": "gun",
    "щипок": "pinch",
    "pinch": "pinch",
    "ок": "ok",
    "okay": "ok",
}


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


def _clean_value(value: str) -> str:
    clean = (value or "").strip()
    clean = clean.strip(" \t\r\n\"'`«».,;:()[]{}")
    clean = re.sub(
        (
            r"^(?:приложени[еяю]|app|сайт|url|файл|папк[ауи]?"
            r"|путь|команд[ау])\s+"
        ),
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.split(
        r"\s+(?:потом|затем|после этого|далее|когда|если)\s+",
        clean,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    clean = re.sub(r"\s+жест(?:ом)?\s+\S+.*$", "", clean, flags=re.IGNORECASE)
    return clean.strip(" \t\r\n\"'`«».,;:()[]{}")


def _quoted_value(text: str) -> str:
    match = re.search(r"[«\"']([^»\"']+)[»\"']", text or "")
    return _clean_value(match.group(1)) if match else ""


def _explicit_gesture_label(prompt: str) -> str:
    match = re.search(
        r"(?:^|\s)(?:жест(?:ом|а)?|gesture)\s*[:=]?\s+([A-Za-zА-Яа-я0-9_.-]+)",
        prompt or "",
        re.IGNORECASE,
    )
    if not match:
        return ""
    label = _clean_value(match.group(1))
    if _norm(label) in {
        "привяжи",
        "привязать",
        "открой",
        "открыть",
        "к",
        "на",
        "для",
        "не",
    }:
        return ""
    return label


def _find_url(text: str) -> str:
    raw = text or ""
    match = re.search(r"(https?://[^\s,;]+|www\.[^\s,;]+)", raw, re.IGNORECASE)
    if match:
        url = match.group(1).rstrip(".,;)")
        return url if url.startswith(("http://", "https://")) else f"https://{url}"
    match = re.search(
        r"(?:сайт|url|адрес)\s+([a-z0-9.-]+\.[a-z]{2,}(?:/[^\s,;]*)?)",
        raw,
        re.IGNORECASE,
    )
    if match:
        return f"https://{match.group(1).rstrip('.,;)')}"
    site_match = re.search(
        r"(?:сайт|url|адрес)\s+(?P<value>[^,.;\n]+)",
        raw,
        re.IGNORECASE,
    )
    if site_match:
        value = _norm(_clean_value(site_match.group("value")))
        for alias, url in SITE_ALIASES.items():
            if alias in value:
                return url
    return ""


def _find_path(text: str, *, python_only: bool = False) -> str:
    quoted = _quoted_value(text)
    if quoted.startswith(("/", "~")) and (not python_only or quoted.endswith(".py")):
        return quoted
    suffix = r"\.py" if python_only else r"(?:\.[a-z0-9]{1,8})?"
    match = re.search(rf"((?:~|/)[^\s,;]+{suffix})", text or "", re.IGNORECASE)
    return match.group(1).rstrip(".,;)") if match else ""


def _normalize_key(token: str) -> str:
    clean = _norm(token).strip(" \t\r\n\"'`«»;:()[]{}")
    if not clean:
        return ""
    clean = clean.replace("page down", "pagedown").replace("page up", "pageup")
    if clean in KEY_ALIASES:
        return KEY_ALIASES[clean]
    if len(clean) == 1 and re.match(r"[a-z0-9,.\-+=/]", clean):
        return clean
    return ""


def _hotkey_from_text(text: str) -> list[str]:
    lower = _norm(text)
    plus_match = re.search(
        r"([a-zа-я0-9⌘⌥⇧⌃_,.\-]+(?:\s*\+\s*[a-zа-я0-9⌘⌥⇧⌃_,.\-]+)+)",
        text or "",
        re.IGNORECASE,
    )
    if plus_match:
        keys = [
            _normalize_key(part)
            for part in re.split(r"\s*\+\s*", plus_match.group(1))
        ]
        keys = [key for key in keys if key]
        if len(keys) >= 2:
            return keys

    for markers, keys in SIMPLE_HOTKEYS:
        if any(marker in lower for marker in markers):
            return list(keys)

    if not any(
        marker in lower
        for marker in ("cmd", "command", "ctrl", "control", "⌘", "⌃")
    ):
        return []
    tokens = re.findall(r"[a-zа-я0-9⌘⌥⇧⌃]+|[,.\-+=/]", text or "", re.IGNORECASE)
    keys = [_normalize_key(token) for token in tokens]
    keys = [key for key in keys if key]
    if (
        any(key in {"command", "ctrl", "option", "shift"} for key in keys)
        and len(keys) >= 2
    ):
        return keys
    return []


def _macos_navigation_action(text: str) -> dict[str, Any] | None:
    lower = _norm(text)
    navigation_marker = any(
        marker in lower
        for marker in (
            "перелист",
            "переключ",
            "перейти",
            "сменить",
            "switch",
            "move",
        )
    )
    target_marker = any(
        marker in lower
        for marker in (
            "экран",
            "рабочий стол",
            "desktop",
            "space",
            "spaces",
            "пространств",
        )
    )
    if not (navigation_marker and target_marker):
        return None

    direction = ""
    if any(
        marker in lower
        for marker in (
            "влево",
            "налево",
            "левый",
            "предыдущ",
            "swipe_left",
            "left",
        )
    ):
        direction = "left"
    elif any(
        marker in lower
        for marker in (
            "вправо",
            "направо",
            "правый",
            "следующ",
            "swipe_right",
            "right",
        )
    ):
        direction = "right"

    if direction:
        return {
            "action": "key_combination",
            "platform": "macos",
            "keys": ["ctrl", direction],
        }
    return None


def _number(text: str, default: float) -> float:
    match = re.search(r"(\d+(?:[,.]\d+)?)", text or "")
    if not match:
        return default
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return default


def _extract_app_name(text: str) -> str:
    quoted = _quoted_value(text)
    if quoted and not quoted.startswith(("/", "~")) and not _find_url(quoted):
        return quoted
    lower = _norm(text)
    for alias, app in APP_ALIASES.items():
        if alias in lower:
            return app
    for app in COMMON_APP_NAMES:
        if _norm(app) in lower:
            return app
    patterns = (
        (
            r"(?:откр\w*|запуст\w*|старт\w*|включ\w*)\s+"
            r"(?:приложени[еяю]\s+)?(?P<value>[^,.;\n]+)"
        ),
        (
            r"(?:к|для|на)\s+(?:приложени[еяю]\s+)?"
            r"(?P<value>[A-Za-z][\w .+\-]{1,48})"
        ),
    )
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            value = _clean_value(match.group("value"))
            if value and not value.startswith(("http://", "https://", "/", "~")):
                if _is_abstract_workflow_target(value):
                    return ""
                return value
    return ""


def _is_abstract_workflow_target(value: str) -> bool:
    lower = _norm(value).replace("ё", "е")
    if not lower:
        return False
    markers = tuple(marker.replace("ё", "е") for marker in ABSTRACT_WORKFLOW_TARGET_MARKERS)
    return any(marker in lower for marker in markers)


def _prompt_mentions_app(text: str, app: str) -> bool:
    lower = _norm(text).replace("ё", "е")
    app_norm = _norm(app).replace("ё", "е")
    if app_norm and app_norm in lower:
        return True
    for alias, mapped in APP_ALIASES.items():
        if mapped == app and alias.replace("ё", "е") in lower:
            return True
    if app == "Calendar" and "календар" in lower:
        return True
    return False


def _action_conflicts_with_abstract_workflow(
    text: str,
    action_spec: dict[str, Any],
) -> bool:
    if not action_spec or action_spec.get("action") != "open_app":
        return False
    app = str(action_spec.get("app") or "").strip()
    if _is_abstract_workflow_target(app):
        return True
    if not _is_abstract_workflow_target(text):
        return False
    return not _prompt_mentions_app(text, app)


def _parse_action(text: str) -> dict[str, Any] | None:
    lower = _norm(text)

    navigation_action = _macos_navigation_action(text)
    if navigation_action:
        return navigation_action

    url = _find_url(text)
    if url:
        return {"action": "open_url", "platform": "macos", "url": url}
    if any(marker in lower for marker in ("почт", "mail")):
        for alias, url in SITE_ALIASES.items():
            if alias in lower:
                return {"action": "open_url", "platform": "macos", "url": url}
    if any(marker in lower for marker in ("сайт", "website", "url", "адрес")):
        return None

    script_path = _find_path(text, python_only=True)
    if script_path and ("скрипт" in lower or "script" in lower or ".py" in lower):
        return {
            "action": "run_script",
            "platform": "macos",
            "script_path": script_path,
        }

    path = _find_path(text)
    if path and any(marker in lower for marker in ("файл", "папк", "путь", "откр")):
        return {"action": "open_path", "platform": "macos", "path": path}

    keys = _hotkey_from_text(text)
    if keys:
        return {"action": "key_combination", "platform": "macos", "keys": keys}

    ontology_action = parse_action_intent(text)
    if ontology_action:
        return ontology_action

    if any(marker in lower for marker in ("заблок", "lock screen", "lock_screen")):
        return {"action": "lock_screen", "platform": "macos"}
    if any(marker in lower for marker in ("скрин", "screenshot", "снимок экрана")):
        return {"action": "screenshot", "platform": "macos"}
    if "ярк" in lower and any(
        marker in lower for marker in ("увелич", "выше", "ярче", "+")
    ):
        return {"action": "brightness_up", "platform": "macos"}
    if "ярк" in lower and any(
        marker in lower for marker in ("уменьш", "ниже", "темнее", "-")
    ):
        return {"action": "brightness_down", "platform": "macos"}
    if "громк" in lower and any(
        marker in lower for marker in ("увелич", "выше", "громче", "+")
    ):
        return {"action": "volume_up", "platform": "macos"}
    if "громк" in lower and any(
        marker in lower for marker in ("уменьш", "ниже", "тише", "-")
    ):
        return {"action": "volume_down", "platform": "macos"}
    if any(
        marker in lower
        for marker in ("mute", "выключи звук", "без звука", "звук выкл")
    ):
        return {"action": "mute_toggle", "platform": "macos"}
    if any(marker in lower for marker in ("следующ", "next track", "next song")):
        return {"action": "media_key", "platform": "macos", "kind": "next"}
    if any(marker in lower for marker in ("предыдущ", "prev track", "previous")):
        return {"action": "media_key", "platform": "macos", "kind": "prev"}
    if any(
        marker in lower
        for marker in ("play_pause", "плей пауза", "play pause", "музык")
    ):
        return {"action": "media_key", "platform": "macos", "kind": "play_pause"}

    if any(marker in lower for marker in ("подожд", "wait", "пауза")):
        return {"action": "wait", "platform": "macos", "seconds": _number(text, 1.0)}

    if any(marker in lower for marker in ("уведом", "notify", "сообщени")):
        message = _quoted_value(text)
        if not message:
            match = re.search(
                r"(?:уведом\w*|notify|сообщени\w*)\s+(?P<value>[^,.;\n]+)",
                text or "",
                re.IGNORECASE,
            )
            message = _clean_value(match.group("value")) if match else "Готово"
        return {
            "action": "notify",
            "platform": "macos",
            "title": "GestureBind",
            "message": message or "Готово",
        }

    if any(marker in lower for marker in ("скрол", "прокрут")):
        amount = int(_number(text, 5))
        direction_down = any(marker in lower for marker in ("вниз", "down", "ниже"))
        clicks = -abs(amount) if direction_down else abs(amount)
        return {"action": "scroll", "platform": "macos", "clicks": clicks}

    if any(marker in lower for marker in ("page down", "pagedown", "страниц")) and any(
        marker in lower for marker in ("вниз", "down", "след")
    ):
        return {"action": "press", "platform": "macos", "key": "pagedown"}
    if "page up" in lower or "pageup" in lower:
        return {"action": "press", "platform": "macos", "key": "pageup"}
    if any(marker in lower for marker in ("нажми", "нажать", "press")):
        for key in (
            "enter",
            "escape",
            "tab",
            "delete",
            "backspace",
            "down",
            "up",
            "left",
            "right",
        ):
            if key in lower:
                return {"action": "press", "platform": "macos", "key": key}

    app = _extract_app_name(text)
    if app:
        return {"action": "open_app", "platform": "macos", "app": app}
    return None


def _split_sequence(text: str) -> list[str]:
    body = text or ""
    body = re.sub(
        (
            r"^\s*(?:привяжи|привязать|назначь|сохрани|сделай|создай|добавь)?\s*"
            r"(?:жест(?:ом|а)?|gesture)\s*[:=]?\s+"
            r"[A-Za-zА-Яа-я0-9_.-]+\s+(?:к|на|для)\s+"
        ),
        "",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        (
            r"^\s*(?:жест(?:ом|а)?|gesture)\s*[:=]?\s+"
            r"[A-Za-zА-Яа-я0-9_.-]+\s+"
            rf"(?={ACTION_START_PATTERN}\w*)"
        ),
        "",
        body,
        flags=re.IGNORECASE,
    )
    series_prefix = re.match(
        r"\s*(?:сер(?:и[яию]|ии)|последовательност[ьи])\s+команд\w*",
        body,
        re.IGNORECASE,
    )
    if series_prefix:
        body = body[series_prefix.end() :]
        body = re.sub(
            (
                r"^\s*(?:которая|который|которые)?\s*(?:будет|будут)?\s*"
                r"(?:называться|назваться|назови|назвать)\s+"
                r"[^-—:;,\n]+\s*[-:—]?\s*"
            ),
            "",
            body,
            flags=re.IGNORECASE,
        )
        body = re.sub(r"^\s*[-:—]\s*", "", body)
    scenario_marker = re.search(r"\bсценар\w*\s*:", body, re.IGNORECASE)
    if scenario_marker:
        body = body[scenario_marker.end() :]
    else:
        body = re.sub(
            (
                r"^\s*(?:сделай|создай|добавь|собери|подготовь)?\s*"
                r"сценар\w*(?:\s+под\s+названи(?:ем|е)\s+"
                r"[^:;,\n-]+)?\s*[-:—]?\s*"
            ),
            "",
            body,
            flags=re.IGNORECASE,
        )
    body = re.sub(
        (
            r"^\s*(?:сделай|создай|добавь|собери|подготовь)\s+"
            rf"(?:(?!{ACTION_START_PATTERN}\w*).){{1,80}}"
            r"[:—-]\s*"
            rf"(?={ACTION_START_PATTERN}\w*)"
        ),
        "",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        (
            r"\b(?:перв(?:ый|ым|ое)?(?:\s+шаг)?|1\s*[-.]?\s*шаг|"
            r"шаг\s*1|сначала)\b"
        ),
        ";",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        (
            r"\b(?:втор(?:ой|ым|ое)?(?:\s+шаг)?|2\s*[-.]?\s*шаг|"
            r"шаг\s*2)\b"
        ),
        ";",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        (
            r"\b(?:трет(?:ий|ьим|ье)?(?:\s+шаг)?|3\s*[-.]?\s*шаг|"
            r"шаг\s*3)\b"
        ),
        ";",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        (
            r"\b(?:четверт(?:ый|ым|ое)?(?:\s+шаг)?|4\s*[-.]?\s*шаг|"
            r"шаг\s*4)\b"
        ),
        ";",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        (
            r"\b(?:пят(?:ый|ым|ое)?(?:\s+шаг)?|5\s*[-.]?\s*шаг|"
            r"шаг\s*5)\b"
        ),
        ";",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        r"\s+(?:и\s+)?(?:потом|затем|после этого|далее)\s+",
        ";",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        rf"\s+и\s+(?={ACTION_START_PATTERN})",
        ";",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        rf"[,]\s*(?={ACTION_START_PATTERN}\w*)",
        ";",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        r"\s+(?=включ\w+\s+сайт\b)",
        ";",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        rf"\s+(?={INLINE_ACTION_START_PATTERN}\w*)",
        ";",
        body,
        flags=re.IGNORECASE,
    )
    return [
        _clean_value(part)
        for part in re.split(r"[;\n]+", body)
        if _clean_value(part)
    ]


def _extract_scenario_name(text: str) -> str:
    patterns = (
        (
            r"\bсценар\w*\s+(?:под\s+названи(?:ем|е)|с\s+названи(?:ем|е)|"
            r"назови)\s+[«\"']?(?P<value>.+?)[»\"']?\s*"
            r"(?=(?:[-—:;,]|\bперв(?:ый|ым|ое)?\b|\b1\s*[-.]?\s*шаг\b|"
            r"\bсначала\b|\bоткр|\bзапуст|\bвключ|\bнажм|\bподожд|\bпокаж|$))"
        ),
        (
            r"\b(?:сер(?:и[яию]|ии)|последовательност[ьи])\s+команд\w*"
            r"(?:\s+(?:которая|который|которые)\s+буд(?:ет|ут))?\s+"
            r"(?:называться|назваться|назови|назвать)\s+"
            r"[«\"']?(?P<value>.+?)[»\"']?\s*"
            r"(?=(?:[-—:;,]|\bперв(?:ый|ым|ое)?\b|\b1\s*[-.]?\s*шаг\b|"
            r"\bсначала\b|\bоткр|\bзапуст|\bвключ|\bнажм|\bподожд|\bпокаж|$))"
        ),
    )
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            return _clean_value(match.group("value"))
    return ""


def _action_title(spec: dict[str, Any]) -> str:
    action = str(spec.get("action") or "")
    if action == "open_app":
        return f"Открыть {spec.get('app') or 'приложение'}"
    if action == "open_url":
        return f"Открыть {spec.get('url') or 'сайт'}"
    if action == "open_path":
        return f"Открыть {spec.get('path') or 'путь'}"
    if action == "key_combination":
        return "Нажать " + " + ".join(str(k) for k in spec.get("keys", []))
    if action == "press":
        return f"Нажать {spec.get('key') or 'клавишу'}"
    if action == "notify":
        return "Уведомление"
    if action == "media_key":
        kind = str(spec.get("kind") or "play_pause")
        return {
            "play_pause": "Пауза/воспроизведение медиа",
            "play": "Запустить воспроизведение",
            "pause": "Поставить медиа на паузу",
            "next": "Следующий медиа-трек",
            "prev": "Предыдущий медиа-трек",
            "previous": "Предыдущий медиа-трек",
        }.get(kind, "Управлять медиа")
    if action == "sequence":
        name = str(spec.get("name") or "").strip()
        if name:
            return f"Сценарий «{name}» из {len(spec.get('steps') or [])} шагов"
        return f"Сценарий из {len(spec.get('steps') or [])} шагов"
    return AGENT_ACTION_LABELS.get(action, action or "Команда")


def _provider_name(provider: str | None) -> str:
    env_provider = os.getenv(MISTRAL_PROVIDER_ENV) or os.getenv("BINDING_AGENT_PROVIDER")
    if provider is not None:
        raw = provider
    elif env_provider:
        raw = env_provider
    elif os.getenv(MISTRAL_API_KEY_ENV):
        raw = "mistral"
    else:
        raw = "local"
    return _norm(str(raw)).replace("_", "-")


def _wants_mistral(provider: str | None) -> bool:
    return _provider_name(provider) in {"mistral", "mistral-api", "llm"}


def _local_first_enabled() -> bool:
    return _env_flag(BINDING_AGENT_LOCAL_FIRST_ENV, default=True)


def _answer_rewrite_enabled(provider: str | None) -> bool:
    return _wants_mistral(provider) and _env_flag(
        BINDING_AGENT_REWRITE_ANSWERS_ENV,
        default=False,
    )


def binding_agent_provider_label(provider: str | None = None) -> str:
    _load_project_dotenv()
    if _wants_mistral(provider):
        return "Mistral" if os.getenv(MISTRAL_API_KEY_ENV) else "Локальный fallback"
    return "Локальный агент"


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("empty model response")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model response is not a JSON object")
    return parsed


def _normalize_summary(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_missing(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_model_action_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    spec = dict(value)
    action = str(spec.get("action") or "").strip()
    if not action:
        return {}
    if action not in AGENT_ACTION_LABELS:
        return {}
    spec["action"] = action
    spec.setdefault("platform", "macos")
    if action == "key_combination":
        keys = spec.get("keys")
        if isinstance(keys, str):
            parsed_keys = _hotkey_from_text(keys)
            if not parsed_keys:
                parsed_keys = [
                    _normalize_key(part)
                    for part in re.split(r"\s*\+\s*|\s*,\s*", keys)
                ]
                parsed_keys = [key for key in parsed_keys if key]
            spec["keys"] = parsed_keys
    if action == "sequence":
        steps = spec.get("steps")
        if isinstance(steps, list):
            normalized_steps: list[dict[str, Any]] = []
            for item in steps:
                step = _normalize_model_action_spec(item)
                if not step:
                    continue
                step.pop("platform", None)
                if step.get("action"):
                    normalized_steps.append(step)
            spec["steps"] = normalized_steps
    return spec


def _unique_missing(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _norm(value)
        if key and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _load_project_dotenv() -> None:
    try:
        from app.services.app_config import load_project_dotenv
    except Exception:
        return
    load_project_dotenv()


def _env_flag(name: str, *, default: bool) -> bool:
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off", "disable", "disabled"}


def _mlflow_tracking_uri() -> str:
    tracking_uri = str(os.getenv("MLFLOW_TRACKING_URI") or "").strip()
    if tracking_uri:
        return tracking_uri
    return f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def _mlflow_experiment() -> str:
    return str(
        os.getenv(BINDING_AGENT_MLFLOW_EXPERIMENT_ENV)
        or os.getenv("GESTUREBIND_MLFLOW_EXPERIMENT")
        or os.getenv("GESTUREFLOW_MLFLOW_EXPERIMENT")
        or "GestureBind"
    ).strip()


def _short_text(value: str, limit: int = 250) -> str:
    clean = re.sub(r"\s+", " ", str(value or "")).strip()
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _metric_suffix(value: str) -> str:
    suffix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "").strip().lower())
    return suffix.strip("_.-") or "unknown"


def _history_items(history: list[dict[str, str]], limit: int = 8) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for item in history[-limit:]:
        role = str(item.get("role") or "").strip()
        text = str(item.get("text") or "").strip()
        if role and text:
            items.append({"role": role, "text": text})
    return items


def _history_text(history: list[dict[str, str]], limit: int = 8) -> str:
    labels = {"user": "Пользователь", "agent": "Агент", "assistant": "Агент"}
    lines: list[str] = []
    for item in _history_items(history, limit=limit):
        role = labels.get(item["role"], item["role"])
        lines.append(f"{role}: {_short_text(item['text'], 220)}")
    return "\n".join(lines)


def _variant_index(seed: str, count: int) -> int:
    if count <= 1:
        return 0
    return zlib.crc32(seed.encode("utf-8")) % count


def _pick_variant(seed: str, variants: tuple[str, ...]) -> str:
    return variants[_variant_index(seed, len(variants))]


def _is_capability_question(text: str) -> bool:
    lower = _norm(text)
    return any(
        marker in lower
        for marker in (
            "что ты умеешь",
            "что умеешь",
            "что можешь",
            "что можно",
            "помощь",
            "help",
            "capabilities",
            "какие команды",
            "какие жесты",
            "как ты работаешь",
        )
    )


def _is_agent_scope_question(text: str) -> bool:
    lower = _norm(text)
    return any(
        marker in lower
        for marker in (
            "почему ты оста",
            "зачем ты оста",
            "почему ты в рамках",
            "почему в рамках",
            "зачем в рамках",
            "почему не отвеча",
            "зачем не отвеча",
            "почему нельзя",
            "рамк",
            "контекст",
            "общие вопросы",
            "посторон",
            "отклон",
            "не отклон",
            "предыдущий ответ",
            "предыдущие инструкции",
            "правила агента",
        )
    )


def _is_question_like(text: str) -> bool:
    lower = _norm(text)
    if "?" in text:
        return True
    return any(
        lower.startswith(marker)
        for marker in (
            "что ",
            "как ",
            "почему ",
            "зачем ",
            "когда ",
            "где ",
            "какой ",
            "какая ",
            "какие ",
            "сколько ",
            "можно ли ",
            "правда ли ",
            "is ",
            "does ",
            "what ",
            "how ",
            "why ",
            "where ",
            "when ",
        )
    )


def _has_project_scope(text: str) -> bool:
    lower = _norm(text)
    return any(
        marker in lower
        for marker in (
            "dplm",
            "gesturebind",
            "проект",
            "прилож",
            "система",
            "агент",
            "жест",
            "gesture",
            "привяз",
            "команд",
            "сценар",
            "mlflow",
            "mistral",
            "llm",
            "модель",
            "распознаван",
            "датасет",
            "обуч",
        )
    )


def _looks_like_binding_request(context: BindingAgentContext) -> bool:
    lower = _norm(context.prompt)
    labels = _known_gesture_labels(context)
    if _match_gesture_label_in_text(context.prompt, labels):
        return True
    if _explicit_gesture_label(context.prompt):
        return True
    if _parse_action(context.prompt) is not None:
        return True
    return any(
        marker in lower
        for marker in (
            "привяж",
            "привяз",
            "сохрани",
            "создай",
            "добавь",
            "измени",
            "изменить",
            "поменяй",
            "замени",
            "обнови",
            "перепривяж",
            "переназнач",
            "назначь",
            "жест",
            "gesture",
            "сценар",
            "hotkey",
            "shortcut",
        )
    )


def _is_project_question(text: str) -> bool:
    return (
        _is_capability_question(text)
        or _is_agent_scope_question(text)
        or (_is_question_like(text) and _has_project_scope(text))
    )


def _is_out_of_scope_question(text: str) -> bool:
    lower = _norm(text)
    explicit_markers = (
        "погода",
        "новости",
        "курс валют",
        "биткоин",
        "рецепт",
        "анекдот",
        "столица",
        "кто президент",
        "фильм",
        "музыка",
        "спорт",
        "путешеств",
        "домашнее задание",
    )
    if any(marker in lower for marker in explicit_markers):
        return True
    return _is_question_like(text) and not _has_project_scope(text)


def _capability_answer_text(labels: list[str]) -> str:
    examples = [
        "`жест palm открыть Safari`",
        "`свайп вверх нажать command+z`",
        "`жест swipe_down сценарий: открыть Preview; подожди 1 секунду; покажи уведомление Готово`",
    ]
    gestures = ", ".join(labels[:6]) if labels else "пока список жестов не загружен"
    return (
        "**Что я умею**\n\n"
        "- Понимаю обычные фразы на русском и английском и собираю из них привязку.\n"
        "- Ищу жесты по синонимам: `свайп вверх` -> `swipe_up`, `ладонь` -> `palm`.\n"
        "- Уточняю недостающую часть, если есть жест без действия или действие без жеста.\n"
        "- Проверяю спорные команды: например, `command+z` это отмена, а не закрытие приложения.\n"
        "- Могу собрать простой сценарий из нескольких шагов.\n\n"
        f"**Доступные жесты:** {gestures}\n\n"
        "**Примеры запросов:**\n"
        + "\n".join(f"- {example}" for example in examples)
    )


def _history_mentions_scope_redirect(history: list[dict[str, str]]) -> bool:
    for item in _history_items(history):
        if item.get("role") not in {"agent", "assistant"}:
            continue
        lower = _norm(item.get("text") or "")
        if any(
            marker in lower
            for marker in (
                "остан",
                "контекст",
                "общие вопросы",
                "gesturebind",
                "рамк",
            )
        ):
            return True
    return False


def _scope_answer_text(text: str, history: list[dict[str, str]]) -> str:
    seed = _norm(text) + "|" + _history_text(history, limit=4)
    title = _pick_variant(
        seed + "|title",
        (
            "**Почему я держусь GestureBind**",
            "**Держу фокус на GestureBind**",
            "**Остаюсь в рабочем контуре GestureBind**",
        ),
    )
    prefix = ""
    if _history_mentions_scope_redirect(history):
        prefix = _pick_variant(
            seed + "|memory",
            (
                "Да, вижу предыдущий ответ и держу его в памяти. ",
                "Да, я помню прошлую реплику и продолжаю ту же линию. ",
                "Вижу контекст выше: я всё ещё аккуратно держу рамку GestureBind. ",
            ),
        )
    intro = _pick_variant(
        seed + "|intro",
        (
            "Ха-ха, понимаю желание получить ответ на всё сразу, но давай не будем уводить агента в соседние темы.",
            "Ха-ха, соблазн уйти в общий чат понятен, но здесь лучше держать руль на GestureBind.",
            "Ха-ха, можно было бы развернуться в обычный чат, но тогда агент начнёт мешать привязки с посторонними задачами.",
        ),
    )
    help_intro = _pick_variant(
        seed + "|help",
        (
            "Зато здесь я могу быть очень полезным:",
            "Внутри GestureBind я как раз полезен вот где:",
            "Лучше потрачу внимание на то, что реально помогает в приложении:",
        ),
    )
    example = _pick_variant(
        seed + "|example",
        (
            "`жест palm открыть Safari`",
            "`свайп вверх нажать command+z`",
            "`почему command+z не закрывает приложение?`",
        ),
    )
    return (
        f"{title}\n\n"
        f"{prefix}{intro} Я держусь рамок GestureBind, чтобы не смешивать "
        "настройку жестов, команд и сценариев с обычным чатом.\n\n"
        f"{help_intro}\n"
        "- разобрать фразу и собрать привязку жеста;\n"
        "- подсказать подходящую команду или hotkey;\n"
        "- объяснить, какие жесты доступны и чего не хватает для сохранения;\n"
        "- проверить MLflow/Mistral-пайплайн по агентам.\n\n"
        f"Например: {example}."
    )


def _project_answer_text(
    text: str,
    labels: list[str],
    history: list[dict[str, str]] | None = None,
) -> str:
    lower = _norm(text)
    if _is_capability_question(text):
        return _capability_answer_text(labels)
    if _is_agent_scope_question(text):
        return _scope_answer_text(text, list(history or []))
    if "mlflow" in lower:
        return (
            "**MLflow в этом проекте**\n\n"
            "- Я логирую multi-agent pipeline как run: входной текст, шаги агентов, "
            "черновик результата и метрики статусов.\n"
            "- Для GenAI-вкладки отдельно создаются traces и scorer assessments.\n"
            "- Если Quality пустая, значит не был выполнен `mlflow.genai.evaluate` "
            "или не подключены scorer'ы.\n\n"
            "Можно спросить: `почему привязка не сохранилась?` или "
            "`покажи, какие шаги прошёл агент`."
        )
    if "mistral" in lower or "llm" in lower or "модель" in lower:
        return (
            "**LLM-режим агента**\n\n"
            "- Без ключа работает локальный deterministic-пайплайн.\n"
            "- Если в `.env` есть `MISTRAL_API_KEY`, агент может использовать Mistral "
            "для разбора фразы.\n"
            "- После ответа модельный draft всё равно проходит локальные проверки: "
            "жест, действие, политика и reviewer.\n\n"
            "Так мы можем тестировать LLM, но не отдавать UI сырой ответ модели."
        )
    gestures = ", ".join(labels[:8]) if labels else "список жестов пока не загружен"
    return (
        "**GestureBind**\n\n"
        "- Основная задача: связать распознанный жест с командой macOS или сценарием.\n"
        "- Агент принимает обычную фразу, находит жест, действие и недостающие поля.\n"
        "- Ручная форма остаётся рядом, чтобы пользователь мог проверить и сохранить результат.\n"
        "- Наблюдаемость уходит в MLflow: intent, шаги агентов, draft и scorer-проверки.\n\n"
        f"**Жесты сейчас:** {gestures}"
    )


def _unsupported_answer_text(
    text: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    seed = _norm(text) + "|" + _history_text(list(history or []), limit=4)
    title = _pick_variant(
        seed + "|title",
        (
            "**Останемся в GestureBind**",
            "**Верну нас к GestureBind**",
            "**Держим фокус на GestureBind**",
        ),
    )
    intro = _pick_variant(
        seed + "|intro",
        (
            "Ха-ха, давай не будем отклоняться от темы.",
            "Ха-ха, понял вопрос, но мягко возвращаю нас в рабочую область.",
            "Ха-ха, это уже похоже на общий чат, а я здесь всё-таки про жесты и привязки.",
        ),
    )
    boundary = _pick_variant(
        seed + "|boundary",
        (
            "На общие вопросы вне GestureBind я лучше мягко сверну разговор, чтобы не смешивать настройку привязок с посторонними темами.",
            "Я не буду разворачивать постороннюю тему, чтобы не путать диалог агента с настройкой жестов, команд и сценариев.",
            "Так мы не потеряем контекст: агент остаётся помощником по GestureBind, а не универсальным собеседником.",
        ),
    )
    help_intro = _pick_variant(
        seed + "|help",
        (
            "Зато я могу помочь здесь:",
            "А вот внутри GestureBind я полезен:",
            "Лучше направим это в действие по приложению:",
        ),
    )
    example = _pick_variant(
        seed + "|example",
        (
            "`жест palm открыть Safari`",
            "`что умеет агент привязки?`",
            "`свайп вверх нажать command+z`",
        ),
    )
    return (
        f"{title}\n\n"
        f"{intro} {boundary}\n\n"
        f"{help_intro}\n"
        "- создать или изменить привязку жеста;\n"
        "- объяснить доступные жесты, команды и сценарии;\n"
        "- проверить, подходит ли hotkey под нужное действие;\n"
        "- подсказать, что видно в MLflow по агентскому пайплайну.\n\n"
        f"Пример: {example}."
    )


def _contains_secret_like(text: str) -> bool:
    raw = text or ""
    secret_patterns = (
        r"(?i)\b(?:api[_-]?key|token|secret|password|пароль|ключ)\s*[:=]\s*\S{6,}",
        r"\bsk-[A-Za-z0-9_-]{12,}\b",
        r"\b[A-Za-z0-9_-]{32,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b",
    )
    return any(re.search(pattern, raw) for pattern in secret_patterns)


def _contains_prompt_injection(text: str) -> bool:
    lower = _norm(text)
    return any(
        marker in lower
        for marker in (
            "ignore previous instructions",
            "ignore the previous instructions",
            "ignore all instructions",
            "system prompt",
            "developer message",
            "prompt injection",
            "jailbreak",
            "игнорируй инструкции",
            "игнорируй предыдущие инструкции",
            "не слушай инструкции",
            "не слушай предыдущие инструкции",
            "не следуй инструкциям",
            "забудь инструкции",
            "забудь предыдущие инструкции",
            "раскрой систем",
            "покажи систем",
            "обойди правила",
            "сломай правила",
        )
    )


def _guardrail_answer_text(reason: str) -> str:
    if reason == "secret_detected":
        return (
            "**Запрос остановлен guardrails**\n\n"
            "Похоже, в сообщении есть ключ, токен или пароль. Я не буду "
            "обрабатывать и логировать такие данные.\n\n"
            "Удалите секрет из текста и повторите запрос, например: "
            "`жест palm открыть Safari`."
        )
    if reason == "prompt_injection":
        return (
            "**Останемся в GestureBind**\n\n"
            "Ха-ха, понимаю ход, но правила агента я не переписываю из сообщения. "
            "Guardrails остановили часть запроса, где предлагается не слушать "
            "предыдущие инструкции.\n\n"
            "Зато я спокойно помогу в рамках GestureBind: привязки, жесты, "
            "команды, сценарии и наблюдаемость пайплайна."
        )
    if reason == "prompt_too_long":
        return (
            "**Запрос слишком длинный**\n\n"
            "Сократите задачу до одной привязки или одного вопроса по проекту. "
            "Так агент точнее определит интент и не потеряет важные детали."
        )
    return (
        "**Ответ остановлен guardrails**\n\n"
        "Я не могу безопасно выдать этот результат. Переформулируйте запрос "
        "в рамках GestureBind: жест, команда, сценарий или вопрос по проекту."
    )


def _is_validation_question(text: str) -> bool:
    lower = _norm(text)
    question_marker = any(
        marker in lower
        for marker in (
            "разве",
            "правильно ли",
            "верно ли",
            "подходит ли",
            "выполняет",
            "соответствует",
            "будет ли",
            "does ",
            "is ",
        )
    )
    command_marker = any(
        marker in lower
        for marker in (
            "command",
            "cmd",
            "ctrl",
            "control",
            "⌘",
            "команд",
            "сочет",
            "клав",
            "действ",
            "закр",
            "откр",
            "close",
            "quit",
        )
    )
    return question_marker and command_marker


def _keys_label(keys: list[str]) -> str:
    return "+".join(str(key) for key in keys if str(key))


def _validation_answer_text(text: str) -> str:
    lower = _norm(text)
    keys = _hotkey_from_text(text)
    close_intent = any(
        marker in lower for marker in ("закр", "закрыт", "close", "quit", "выход")
    )
    undo_intent = any(marker in lower for marker in ("отмен", "undo"))
    open_intent = any(marker in lower for marker in ("откр", "запуст", "open"))

    if keys:
        key_label = _keys_label(keys)
        if keys == ["command", "z"] and close_intent:
            return (
                "**Проверка команды**\n\n"
                f"Нет. `{key_label}` обычно отменяет последнее действие, "
                "а не закрывает приложение.\n\n"
                "- Для закрытия активного приложения на macOS обычно используют `command+q`.\n"
                "- Для закрытия текущего окна обычно используют `command+w`.\n"
                "- Если нужно закрывать именно Telegram, лучше уточнить сценарий: "
                "активировать Telegram и затем отправить `command+q`."
            )
        if keys == ["command", "q"] and close_intent:
            return (
                "**Проверка команды**\n\n"
                f"Да, `{key_label}` обычно завершает активное приложение на macOS. "
                "Важно: команда сработает для приложения, которое сейчас в фокусе."
            )
        if keys == ["command", "w"] and close_intent:
            return (
                "**Проверка команды**\n\n"
                f"Частично. `{key_label}` обычно закрывает текущее окно, "
                "но не обязательно завершает всё приложение."
            )
        if keys == ["command", "z"] and undo_intent:
            return (
                "**Проверка команды**\n\n"
                f"Да. `{key_label}` обычно выполняет отмену последнего действия."
            )
        return (
            "**Проверка команды**\n\n"
            f"Я вижу сочетание `{key_label}`, но не могу уверенно подтвердить "
            "ожидаемое действие из этой фразы. Напишите так: "
            "`разве command+q закрывает приложение?` или "
            "`разве command+z отменяет действие?`."
        )

    action_spec = _parse_action(text)
    if action_spec and open_intent:
        return (
            "**Проверка команды**\n\n"
            f"Да, это похоже на действие: {_action_title(action_spec)}. "
            "Если нужно, я могу сразу собрать привязку: "
            "`жест palm открыть Safari`."
        )
    return (
        "**Проверка команды**\n\n"
        "Я понял, что вы проверяете соответствие команды действию, "
        "но в вопросе не хватает самой команды или ожидаемого результата. "
        "Например: `разве command+z закрывает Telegram?`."
    )


def _known_gesture_labels(context: BindingAgentContext) -> list[str]:
    return [
        str(item.get("label") or "").strip()
        for item in context.gestures
        if str(item.get("label") or "").strip()
    ]


def _normalize_gesture_phrase(value: str) -> str:
    raw = _norm(value)
    raw = raw.replace("_", " ").replace("-", " ")
    tokens = re.findall(r"[a-zа-я0-9]+", raw, re.IGNORECASE)
    normalized: list[str] = []
    for token in tokens:
        mapped = GESTURE_WORD_ALIASES.get(token, token)
        if mapped:
            normalized.append(mapped)
    return "_".join(normalized)


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
        return _clean_value(match.group("value"))
    generic_bind_match = re.search(
        (
            r"(?:привяж\w*|привяз\w*|сохрани\w*|назнач\w*)\s+"
            r"(?P<value>.+?)(?=\s+(?:к|на|для)\s+)"
        ),
        raw,
        re.IGNORECASE,
    )
    if generic_bind_match:
        value = _clean_value(generic_bind_match.group("value"))
        if _normalize_gesture_phrase(value) not in {
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
    return _clean_value(raw)


def _gesture_label_lookup(labels: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for label in labels:
        normalized = _normalize_gesture_phrase(label)
        if normalized:
            lookup.setdefault(normalized, label)
        words = _normalize_gesture_phrase(label.replace("_", " "))
        if words:
            lookup.setdefault(words, label)
    return lookup


def _similar_gesture_labels(query: str, labels: list[str], limit: int = 3) -> list[str]:
    normalized = _normalize_gesture_phrase(query)
    if not normalized:
        return []
    lookup = _gesture_label_lookup(labels)
    exact = lookup.get(normalized)
    if exact:
        return [exact]
    matches = difflib.get_close_matches(
        normalized,
        list(lookup.keys()),
        n=limit,
        cutoff=0.62,
    )
    return [lookup[item] for item in matches]


def _match_gesture_label_in_text(text: str, labels: list[str]) -> str:
    query = _gesture_query_from_text(text)
    normalized_query = _normalize_gesture_phrase(query)
    lookup = _gesture_label_lookup(labels)
    if normalized_query in lookup:
        return lookup[normalized_query]
    normalized_full_text = _normalize_gesture_phrase(text)
    for normalized_label, label in lookup.items():
        if re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_label)}(?![a-z0-9])",
            normalized_full_text,
        ):
            return label

    text_norm = _norm(text)
    text_words = re.sub(r"[_-]+", " ", text_norm)
    explicit = _explicit_gesture_label(text)
    explicit_norm = _norm(explicit)
    if explicit_norm:
        for label in labels:
            if _norm(label) == explicit_norm:
                return label
        explicit_words = re.sub(r"[_-]+", " ", explicit_norm)
        for label in labels:
            if re.sub(r"[_-]+", " ", _norm(label)) == explicit_words:
                return label
        return explicit
    for label in sorted(labels, key=len, reverse=True):
        label_norm = _norm(label)
        if re.search(rf"(?<!\w){re.escape(label_norm)}(?!\w)", text_norm):
            return label
        label_words = re.sub(r"[_-]+", " ", label_norm)
        if label_words != label_norm and re.search(
            rf"(?<!\w){re.escape(label_words)}(?!\w)",
            text_words,
        ):
            return label
    return ""


class BindingAgentMlflowLogger:
    """Logs the binding-agent pipeline as an MLflow run when enabled."""

    def __init__(self, *, enabled: bool | None = None) -> None:
        self.enabled = enabled

    def log(
        self,
        context: BindingAgentContext,
        result: BindingAgentResult,
        *,
        provider: str,
        model: str = "",
    ) -> dict[str, Any]:
        _load_project_dotenv()
        if self.enabled is None:
            enabled = _env_flag(BINDING_AGENT_MLFLOW_ENV, default=True)
        else:
            enabled = self.enabled
        if not enabled:
            return {}
        experiment = _mlflow_experiment()
        if not experiment:
            return {}
        try:
            import mlflow
        except Exception as exc:
            print(f"[w] MLflow недоступен, agent tracking пропущен: {exc}", flush=True)
            return {}

        tracking_uri = _mlflow_tracking_uri()
        action = str(result.action_spec.get("action") or "no_action")
        run_name = (
            "binding-agent-"
            f"{_metric_suffix(result.gesture_label or 'no_gesture')}-"
            f"{_metric_suffix(action)}"
        )
        statuses = [str(step.status or "") for step in result.steps]
        reviewer_steps = [step for step in result.steps if step.agent == "Reviewer Agent"]
        intent_steps = [step for step in result.steps if step.agent == "Intent Agent"]
        intent_data = dict(intent_steps[-1].data) if intent_steps else {}
        reviewer_data = dict(reviewer_steps[-1].data) if reviewer_steps else {}
        reviewer_scores = {
            str(key): float(value)
            for key, value in dict(reviewer_data.get("scores") or {}).items()
        }
        reviewer_relevance = (
            float(reviewer_data.get("relevance") or 0.0)
            if reviewer_steps
            else 1.0
        )
        semantic_score = float(intent_data.get("semanticScore") or 0.0)
        status_counts = {
            "ok": statuses.count("ok"),
            "need_input": statuses.count("need_input"),
            "need_clarification": statuses.count("need_clarification"),
            "blocked": statuses.count("blocked"),
            "skipped": statuses.count("skipped"),
        }
        draft = result.to_legacy_draft()
        payload = {
            "prompt": context.prompt,
            "current_gesture": context.current_gesture,
            "conversation_history": _history_items(context.conversation_history),
            "known_gestures": [
                str(item.get("label") or "")
                for item in context.gestures
                if str(item.get("label") or "")
            ],
            "provider": provider,
            "model": model,
            "intent": result.intent,
            "intent_block": result.intent_block,
            "result": draft,
            "steps": [
                {
                    "index": index,
                    "agent": step.agent,
                    "status": step.status,
                    "message": step.message,
                    "data": _step_data_with_skills(step),
                }
                for index, step in enumerate(result.steps, start=1)
            ],
        }

        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)
            trace_meta = self._log_genai_trace(
                mlflow,
                context,
                result,
                payload=payload,
                provider=provider,
                model=model,
            )
            with mlflow.start_run(run_name=run_name, log_system_metrics=False) as run:
                run_id = str(getattr(getattr(run, "info", None), "run_id", "") or "")
                mlflow.set_tags(
                    {
                        "run_kind": "binding_agent_pipeline",
                        "source": "gesturebind_flet",
                        "provider": provider,
                        "model": model,
                        "result": "ok" if result.ok else "needs_review",
                    }
                )
                mlflow.log_params(
                    {
                        "provider": provider,
                        "model": model,
                        "prompt_preview": _short_text(context.prompt),
                        "current_gesture": context.current_gesture,
                        "gesture_label": result.gesture_label,
                        "command_name": _short_text(result.command_name, 180),
                        "mode": result.mode,
                        "action": action,
                        "intent": result.intent,
                        "intent_block": result.intent_block,
                        "intent_route_method": str(
                            intent_data.get("routeMethod") or ""
                        ),
                        "semantic_intent": str(
                            intent_data.get("semanticIntent") or ""
                        ),
                        "known_gestures_count": len(context.gestures),
                    }
                )
                mlflow.log_metrics(
                    {
                        "ok": 1.0 if result.ok else 0.0,
                        "can_apply": 1.0 if result.can_apply else 0.0,
                        "steps_total": float(len(result.steps)),
                        "missing_count": float(len(result.missing)),
                        "summary_lines": float(len(result.summary)),
                        "prompt_chars": float(len(context.prompt)),
                        "reviewer_relevance": reviewer_relevance,
                        "intent_semantic_score": semantic_score,
                        **{
                            f"reviewer_{name}": value
                            for name, value in reviewer_scores.items()
                        },
                        **{
                            f"steps_{status}": float(count)
                            for status, count in status_counts.items()
                        },
                    }
                )
                for index, step in enumerate(result.steps, start=1):
                    status = str(step.status or "")
                    mlflow.log_metrics(
                        {
                            "agent_step_ok": 1.0 if status == "ok" else 0.0,
                            "agent_step_blocked": (
                                1.0 if status == "blocked" else 0.0
                            ),
                            "agent_step_needs_input": (
                                1.0
                                if status in {"need_input", "need_clarification"}
                                else 0.0
                            ),
                        },
                        step=index,
                    )
                mlflow.log_dict(payload, "binding_agent_pipeline.json")
                mlflow.log_dict(payload["steps"], "agent_trace.json")
                mlflow.log_dict(draft, "binding_agent_draft.json")
            eval_meta = self._log_genai_evaluation(
                mlflow,
                context,
                result,
                provider=provider,
                model=model,
            )
            print(
                f"[✓] MLflow binding-agent run logged: run_id={run_id or 'unknown'} "
                f"uri={tracking_uri}",
                flush=True,
            )
            return {
                "mlflow_run_id": run_id,
                "mlflow_tracking_uri": tracking_uri,
                "mlflow_experiment": experiment,
                **trace_meta,
                **eval_meta,
            }
        except Exception as exc:
            print(f"[w] MLflow agent tracking failed: {exc}", flush=True)
            return {}

    def _log_genai_evaluation(
        self,
        mlflow: Any,
        context: BindingAgentContext,
        result: BindingAgentResult,
        *,
        provider: str,
        model: str,
    ) -> dict[str, Any]:
        if not _env_flag(BINDING_AGENT_GENAI_EVAL_ENV, default=False):
            return {}
        genai = getattr(mlflow, "genai", None)
        evaluate = getattr(genai, "evaluate", None)
        scorer = getattr(genai, "scorer", None)
        if not callable(evaluate) or not callable(scorer):
            return {}

        @scorer
        def binding_agent_has_gesture(
            outputs=None,
            expectations=None,
            **_kwargs,
        ) -> bool:
            if (expectations or {}).get("expected_status") == "answer":
                return True
            return bool(isinstance(outputs, dict) and outputs.get("gestureLabel"))

        @scorer
        def binding_agent_has_action(
            outputs=None,
            expectations=None,
            **_kwargs,
        ) -> bool:
            if (expectations or {}).get("expected_status") == "answer":
                return True
            return bool(isinstance(outputs, dict) and outputs.get("actionSpec"))

        @scorer
        def binding_agent_contract_ok(outputs=None, expectations=None, **_kwargs) -> bool:
            if not isinstance(outputs, dict):
                return False
            expected = (expectations or {}).get("expected_status")
            if expected == "answer":
                return bool(outputs.get("agentReply")) and not bool(
                    outputs.get("error")
                )
            if expected == "ready":
                return bool(outputs.get("ok") and outputs.get("actionSpec"))
            if expected == "needs_clarification":
                return bool(outputs.get("missing"))
            return not bool(outputs.get("error"))

        action = str(result.action_spec.get("action") or "")
        expected_status = (
            "answer"
            if result.mode == "answer"
            else
            "ready"
            if result.ok
            else "needs_clarification"
            if result.missing
            else "error"
        )
        output = result.to_legacy_draft()
        data = [
            {
                "inputs": {
                    "prompt": context.prompt,
                    "provider": provider,
                    "model": model,
                },
                "outputs": output,
                "expectations": {
                    "expected_status": expected_status,
                    "expected_action": action,
                    "expected_missing": list(result.missing),
                },
            }
        ]
        scorers = [
            binding_agent_has_gesture,
            binding_agent_has_action,
            binding_agent_contract_ok,
        ]

        def predict_fn(prompt: str = "", **_kwargs) -> dict[str, Any]:
            _ = prompt
            return output

        try:
            evaluate(
                data=data,
                predict_fn=predict_fn,
                scorers=scorers,
            )
        except TypeError:
            try:
                evaluate(data=data, scorers=scorers)
            except Exception as exc:
                print(f"[w] MLflow GenAI evaluation failed: {exc}", flush=True)
                return {}
        except Exception as exc:
            print(f"[w] MLflow GenAI evaluation failed: {exc}", flush=True)
            return {}
        return {"mlflow_genai_eval": True}

    def _log_genai_trace(
        self,
        mlflow: Any,
        context: BindingAgentContext,
        result: BindingAgentResult,
        *,
        payload: dict[str, Any],
        provider: str,
        model: str,
    ) -> dict[str, Any]:
        if not _env_flag(BINDING_AGENT_GENAI_TRACES_ENV, default=True):
            return {}
        if not callable(getattr(mlflow, "start_span", None)):
            return {}
        trace_meta: dict[str, Any] = {}
        root = self._start_span(
            mlflow,
            "binding_agent_pipeline",
            span_type="CHAIN",
            attributes={
                "provider": provider,
                "model": model,
                "gesture_label": result.gesture_label,
                "action": str(result.action_spec.get("action") or ""),
                "ok": result.ok,
            },
        )
        with root as root_span:
            self._update_current_trace(
                mlflow,
                context,
                result,
                provider=provider,
                model=model,
            )
            self._set_span_inputs(
                root_span,
                {
                    "prompt": context.prompt,
                    "current_gesture": context.current_gesture,
                    "conversation_history": payload.get("conversation_history", []),
                },
            )
            for index, step in enumerate(result.steps, start=1):
                child = self._start_span(
                    mlflow,
                    step.agent,
                    span_type="AGENT",
                    attributes={
                        "step_index": index,
                        "status": step.status,
                    },
                )
                with child as step_span:
                    self._set_span_inputs(
                        step_span,
                        {
                            "prompt": context.prompt,
                            "known_gestures_count": len(context.gestures),
                        },
                    )
                    self._set_span_outputs(
                        step_span,
                        {
                            "status": step.status,
                            "message": step.message,
                            "data": _step_data_with_skills(step),
                        },
                    )
            self._set_span_outputs(
                root_span,
                {
                    "ok": result.ok,
                    "can_apply": result.can_apply,
                    "missing": list(result.missing),
                    "gesture_label": result.gesture_label,
                    "command_name": result.command_name,
                    "mode": result.mode,
                    "action_spec": dict(result.action_spec),
                },
            )
            trace_id = self._span_trace_id(root_span)
            if trace_id:
                trace_meta["mlflow_trace_id"] = trace_id
        return trace_meta

    def _start_span(
        self,
        mlflow: Any,
        name: str,
        *,
        span_type: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if span_type:
            kwargs["span_type"] = span_type
        if attributes:
            kwargs["attributes"] = attributes
        try:
            return mlflow.start_span(name=name, **kwargs)
        except TypeError:
            return mlflow.start_span(name=name)

    def _update_current_trace(
        self,
        mlflow: Any,
        context: BindingAgentContext,
        result: BindingAgentResult,
        *,
        provider: str,
        model: str,
    ) -> None:
        updater = getattr(mlflow, "update_current_trace", None)
        if not callable(updater):
            return
        payload = {
            "tags": {
                "run_kind": "binding_agent_pipeline",
                "provider": provider,
                "model": model,
                "result": "ok" if result.ok else "needs_review",
            },
            "metadata": {
                "gesture_label": result.gesture_label,
                "command_name": result.command_name,
                "mode": result.mode,
                "action": str(result.action_spec.get("action") or ""),
                "prompt_preview": _short_text(context.prompt),
            },
        }
        try:
            updater(**payload)
        except TypeError:
            try:
                updater(payload)
            except TypeError:
                return

    def _set_span_inputs(self, span: Any, value: dict[str, Any]) -> None:
        setter = getattr(span, "set_inputs", None)
        if callable(setter):
            setter(value)
            return
        self._set_span_attribute(span, "inputs", value)

    def _set_span_outputs(self, span: Any, value: dict[str, Any]) -> None:
        setter = getattr(span, "set_outputs", None)
        if callable(setter):
            setter(value)
            return
        self._set_span_attribute(span, "outputs", value)

    def _set_span_attribute(self, span: Any, key: str, value: Any) -> None:
        setter = getattr(span, "set_attribute", None)
        if callable(setter):
            setter(key, value)

    def _span_trace_id(self, span: Any) -> str:
        for attr in ("trace_id", "request_id"):
            value = getattr(span, attr, "")
            if callable(value):
                value = value()
            if value:
                return str(value)
        getter = getattr(span, "get_trace_id", None)
        if callable(getter):
            value = getter()
            if value:
                return str(value)
        getter = getattr(span, "get_span_context", None)
        if callable(getter):
            context = getter()
            for attr in ("trace_id", "request_id"):
                value = getattr(context, attr, "")
                if value:
                    return str(value)
        return ""


from app.services.binding_agents import (
    ActionAgent,
    GestureAgent,
    GuardrailsAgent,
    IntentAgent,
    MemoryAgent,
    MistralBindingAgent,
    PolicyAgent,
    RelevanceReviewerAgent,
    ResearchAgent,
    ScenarioAgent,
    ValidationAgent,
)


class BindingAgentOrchestrator:
    """Coordinates local and model-backed agents for gesture binding drafts."""

    def __init__(
        self,
        *,
        mistral_agent: MistralBindingAgent | None = None,
        research_agent: ResearchAgent | None = None,
        mlflow_logger: BindingAgentMlflowLogger | None = None,
    ) -> None:
        self.intent_agent = IntentAgent()
        self.mistral_agent = mistral_agent or MistralBindingAgent()
        self.research_agent = research_agent or ResearchAgent()
        self.mlflow_logger = mlflow_logger or BindingAgentMlflowLogger()
        self.gesture_agent = GestureAgent()
        self.action_agent = ActionAgent()
        self.scenario_agent = ScenarioAgent()
        self.memory_agent = MemoryAgent()
        self.policy_agent = PolicyAgent()
        self.validation_agent = ValidationAgent()
        self.guardrails_agent = GuardrailsAgent()
        self.reviewer_agent = RelevanceReviewerAgent()

    def run(
        self,
        prompt: str,
        gestures: list[dict[str, Any]],
        *,
        current_gesture: str = "",
        conversation_history: list[dict[str, str]] | None = None,
        draft_state: dict[str, Any] | None = None,
        provider: str | None = None,
    ) -> BindingAgentResult:
        context = BindingAgentContext(
            prompt=(prompt or "").strip(),
            gestures=gestures,
            current_gesture=current_gesture,
            conversation_history=list(conversation_history or []),
            draft_state=dict(draft_state or {}),
        )
        steps: list[AgentStep] = []
        provider_name = _provider_name(provider)

        if not context.prompt:
            return self._finish_result(
                BindingAgentResult(
                    ok=False,
                    can_apply=False,
                    error="Напишите задачу для агента",
                    missing=[],
                    gesture_label="",
                    command_name="",
                    mode="single",
                    action_spec={},
                    summary=[],
                    response_text=(
                        "Локальный агент: напишите запрос, и я подготовлю привязку."
                    ),
                    steps=[
                        AgentStep(
                            "Orchestrator",
                            "need_input",
                            "Пустой запрос не отправлен агентам.",
                        )
                    ],
                ),
                context,
                provider=provider_name,
            )

        input_guardrail_step = self.guardrails_agent.run_input(context)
        steps.append(input_guardrail_step)
        if input_guardrail_step.status == "blocked":
            issues = list(input_guardrail_step.data.get("issues") or [])
            reason = str(issues[0] if issues else "input_blocked")
            result = BindingAgentResult(
                ok=True,
                can_apply=False,
                error="",
                missing=[],
                gesture_label="",
                command_name="",
                mode="answer",
                action_spec={},
                summary=[],
                response_text=_guardrail_answer_text(reason),
                steps=steps,
                intent="guardrail_block",
                intent_block="guardrails",
            )
            return self._finish_result(
                self._review_result(
                    context,
                    result,
                    AgentStep(
                        "Intent Agent",
                        "blocked",
                        "Маршрутизация остановлена guardrails.",
                        {
                            "intent": "guardrail_block",
                            "block": "guardrails",
                            "route": "blocked",
                        },
                    ),
                ),
                context,
                provider=provider_name,
            )

        task_frame = self.intent_agent.build_frame(context)
        context = replace(context, task_frame=task_frame)
        intent_step = self.intent_agent.run(context, frame=task_frame)
        steps.append(intent_step)
        intent = str(intent_step.data.get("intent") or "create_binding")
        block = str(intent_step.data.get("block") or "binding")

        if block == "project_question":
            labels = _known_gesture_labels(context)
            is_validation = intent == "validate_command"
            answer = (
                _validation_answer_text(context.prompt)
                if is_validation
                else _project_answer_text(
                    context.prompt,
                    labels,
                    context.conversation_history,
                )
            )
            answer_steps = list(steps)
            if _answer_rewrite_enabled(provider):
                rewrite_step, rewritten = self.mistral_agent.rewrite_answer(
                    context,
                    intent=intent,
                    block=block,
                    base_answer=answer,
                )
                answer_steps.append(rewrite_step)
                if rewritten:
                    answer = rewritten
            result = BindingAgentResult(
                ok=True,
                can_apply=False,
                error="",
                missing=[],
                gesture_label="",
                command_name="",
                mode="answer",
                action_spec={},
                summary=[],
                response_text=answer,
                steps=[
                    *answer_steps,
                    AgentStep(
                        "Conversation Agent",
                        "ok",
                        (
                            "Ответил на вопрос о соответствии команды действию."
                            if is_validation
                            else "Ответил на общий вопрос по проекту."
                        ),
                        {
                            "intent": intent,
                            "block": block,
                            "known_gestures_count": len(labels),
                        },
                    ),
                ],
            )
            return self._finish_result(
                self._review_result(context, result, intent_step),
                context,
                provider=provider_name,
            )

        if block == "unsupported_general":
            answer = _unsupported_answer_text(
                context.prompt,
                context.conversation_history,
            )
            answer_steps = list(steps)
            if _answer_rewrite_enabled(provider):
                rewrite_step, rewritten = self.mistral_agent.rewrite_answer(
                    context,
                    intent=intent,
                    block=block,
                    base_answer=answer,
                )
                answer_steps.append(rewrite_step)
                if rewritten:
                    answer = rewritten
            result = BindingAgentResult(
                ok=True,
                can_apply=False,
                error="",
                missing=[],
                gesture_label="",
                command_name="",
                mode="answer",
                action_spec={},
                summary=[],
                response_text=answer,
                steps=[
                    *answer_steps,
                    AgentStep(
                        "Conversation Agent",
                        "ok",
                        "Сформировал безопасный ответ вне области проекта.",
                        {"intent": intent, "block": block},
                    ),
                ],
            )
            return self._finish_result(
                self._review_result(context, result, intent_step),
                context,
                provider=provider_name,
            )

        wants_model = _wants_mistral(provider)
        local_first = _local_first_enabled()

        if wants_model and not local_first:
            model_result, steps = self._mistral_binding_result(
                context,
                intent=intent,
                block=block,
                base_steps=steps,
            )
            if model_result is not None:
                return self._finish_result(
                    self._review_result(context, model_result, intent_step),
                    context,
                    provider=provider_name,
                )

        local_result = self._run_local_binding_pipeline(
            context,
            intent=intent,
            steps=steps,
        )

        if wants_model and local_first and self._needs_mistral_fallback(local_result):
            reason = self._mistral_fallback_reason(local_result)
            fallback_steps = [
                *local_result.steps,
                AgentStep(
                    "Orchestrator",
                    "fallback",
                    "Локальный контракт неполный; пробую Mistral как fallback.",
                    {
                        "reason": reason,
                        "localFirst": True,
                        "localCanApply": local_result.can_apply,
                        "localMissing": list(local_result.missing),
                    },
                ),
            ]
            model_result, fallback_steps = self._mistral_binding_result(
                context,
                intent=intent,
                block=block,
                base_steps=fallback_steps,
            )
            if model_result is not None and self._model_candidate_improves(
                local_result,
                model_result,
            ):
                return self._finish_result(
                    self._review_result(context, model_result, intent_step),
                    context,
                    provider=provider_name,
                )
            local_result = replace(
                local_result,
                steps=[
                    *fallback_steps,
                    AgentStep(
                        "Orchestrator",
                        "skipped",
                        "Mistral fallback не улучшил локальный контракт.",
                        {"reason": reason},
                    ),
                ],
            )

        return self._finish_result(
            self._review_result(context, local_result, intent_step),
            context,
            provider=provider_name,
        )

    def _mistral_binding_result(
        self,
        context: BindingAgentContext,
        *,
        intent: str,
        block: str,
        base_steps: list[AgentStep],
    ) -> tuple[BindingAgentResult | None, list[AgentStep]]:
        steps = list(base_steps)
        model_step, model_draft = self.mistral_agent.run(
            context,
            intent=intent,
            block=block,
        )
        steps.append(model_step)
        if model_draft is None:
            return None, steps
        model_result = self._result_from_external_draft(
            context,
            model_draft,
            steps,
        )
        model_result = self._maybe_research_result(
            context,
            model_result,
            steps,
        )
        return model_result, list(model_result.steps)

    def _run_local_binding_pipeline(
        self,
        context: BindingAgentContext,
        *,
        intent: str,
        steps: list[AgentStep],
    ) -> BindingAgentResult:
        steps = list(steps)
        gesture_step = self.gesture_agent.run(context)
        steps.append(gesture_step)
        gesture = str(gesture_step.data.get("gesture") or "")

        memory_step = self.memory_agent.run(context, gesture)
        steps.append(memory_step)

        action_spec: dict[str, Any] = {}
        if intent == "build_sequence":
            scenario_step = self.scenario_agent.run(context, enabled=True)
            steps.append(scenario_step)
            action_spec = dict(scenario_step.data.get("action_spec") or {})
            if not action_spec:
                action_step = self.action_agent.run(context)
                steps.append(action_step)
                action_spec = dict(action_step.data.get("action_spec") or {})
        else:
            scenario_step = self.scenario_agent.run(context, enabled=False)
            steps.append(scenario_step)
            action_step = self.action_agent.run(context)
            steps.append(action_step)
            action_spec = dict(action_step.data.get("action_spec") or {})

        research: dict[str, Any] = {}
        if not action_spec:
            research_step = self.research_agent.run(context)
            self._append_research_step(steps, research_step)
            action_spec = dict(research_step.data.get("action_spec") or {})
            research = dict(research_step.data.get("research") or {})

        policy_step = self.policy_agent.run(gesture, action_spec)
        steps.append(policy_step)
        missing = list(policy_step.data.get("missing") or [])
        unresolved_steps = []
        if intent == "build_sequence":
            for step in reversed(steps):
                if step.agent != "Scenario Agent":
                    continue
                raw_unresolved = step.data.get("unresolved_steps")
                if isinstance(raw_unresolved, list):
                    unresolved_steps = [
                        item for item in raw_unresolved if isinstance(item, dict)
                    ]
                break
        if unresolved_steps:
            missing = _unique_missing([*missing, "шаги сценария"])

        validation_step = self.validation_agent.run(gesture, action_spec)
        steps.append(validation_step)
        validation_error = (
            validation_step.message
            if action_spec and validation_step.status == "blocked"
            else ""
        )

        if validation_error:
            error = validation_error
            command_name = ""
        elif not action_spec and "действие" not in missing:
            error = "Не удалось понять действие"
            command_name = ""
        elif not action_spec:
            error = ""
            command_name = ""
        else:
            error = ""
            command_name = str(validation_step.data.get("command_name") or "")
        ok = not error and not missing
        can_apply = bool(action_spec and not missing and not error)
        mode = "sequence" if action_spec.get("action") == "sequence" else "single"
        summary = []
        if action_spec:
            summary = [
                f"Жест: {gesture or 'не выбран'}",
                f"Команда: {command_name or _action_title(action_spec)}",
                f"Действие: {_action_title(action_spec)}",
            ]
        response = self._response_text(
            ok=ok,
            error=error,
            missing=missing,
            gesture=gesture,
            action_spec=action_spec,
            research=research,
        )
        result = BindingAgentResult(
            ok=ok,
            can_apply=can_apply,
            error=error,
            missing=missing,
            gesture_label=gesture,
            command_name=command_name,
            mode=mode,
            action_spec=action_spec,
            summary=summary,
            response_text=response,
            steps=steps,
            research=research,
        )
        return result

    def _needs_mistral_fallback(self, result: BindingAgentResult) -> bool:
        if result.can_apply:
            return False
        normalized_missing = {_norm(item) for item in result.missing}
        if "действие" in normalized_missing or "action" in normalized_missing:
            return True
        if result.error:
            return True
        return bool(result.gesture_label and not result.action_spec)

    def _mistral_fallback_reason(self, result: BindingAgentResult) -> str:
        if result.error:
            return "local_error"
        normalized_missing = {_norm(item) for item in result.missing}
        if "действие" in normalized_missing or "action" in normalized_missing:
            return "missing_action"
        if result.gesture_label and not result.action_spec:
            return "empty_action_spec"
        return "incomplete_contract"

    def _model_candidate_improves(
        self,
        local_result: BindingAgentResult,
        model_result: BindingAgentResult,
    ) -> bool:
        if model_result.can_apply and not local_result.can_apply:
            return True
        if model_result.ok and not local_result.ok:
            return True
        if model_result.action_spec and not local_result.action_spec:
            return len(model_result.missing) <= len(local_result.missing)
        return False

    def _review_result(
        self,
        context: BindingAgentContext,
        result: BindingAgentResult,
        intent_step: AgentStep,
    ) -> BindingAgentResult:
        intent = str(intent_step.data.get("intent") or result.intent or "")
        block = str(intent_step.data.get("block") or result.intent_block or "")
        result = replace(result, intent=intent, intent_block=block)
        output_guardrail_step = self.guardrails_agent.run_output(context, result)
        result = replace(result, steps=[*result.steps, output_guardrail_step])
        if output_guardrail_step.status == "blocked":
            result = replace(
                result,
                ok=True,
                can_apply=False,
                error="",
                missing=[],
                action_spec={},
                response_text=_guardrail_answer_text("output_blocked"),
            )
        reviewer_step = self.reviewer_agent.run(context, result, intent_step)
        reviewed = replace(result, steps=[*result.steps, reviewer_step])
        if reviewer_step.status != "blocked":
            return reviewed
        if block == "unsupported_general":
            return replace(
                reviewed,
                ok=True,
                can_apply=False,
                error="",
                missing=[],
                action_spec={},
                response_text=_unsupported_answer_text(
                    context.prompt,
                    context.conversation_history,
                ),
            )
        return reviewed

    def _finish_result(
        self,
        result: BindingAgentResult,
        context: BindingAgentContext,
        *,
        provider: str,
    ) -> BindingAgentResult:
        if result.task_frame is None and context.task_frame is not None:
            result = replace(result, task_frame=context.task_frame)
        telemetry = self.mlflow_logger.log(
            context,
            result,
            provider=provider,
            model=getattr(self.mistral_agent, "model", ""),
        )
        return replace(result, telemetry=telemetry) if telemetry else result

    def _maybe_research_result(
        self,
        context: BindingAgentContext,
        result: BindingAgentResult,
        steps: list[AgentStep],
    ) -> BindingAgentResult:
        needs_research = (
            not result.action_spec
            or "действие" in set(str(item) for item in result.missing)
            or bool(result.error)
        )
        if not needs_research:
            return result
        research_step = self.research_agent.run(context)
        self._append_research_step(steps, research_step)
        action_spec = dict(research_step.data.get("action_spec") or {})
        if not action_spec:
            return result
        research = dict(research_step.data.get("research") or {})
        return self._result_from_researched_action(
            context,
            steps,
            gesture=result.gesture_label,
            action_spec=action_spec,
            research=research,
        )

    def _append_research_step(
        self,
        steps: list[AgentStep],
        research_step: AgentStep,
    ) -> None:
        steps.append(research_step)
        pipeline = research_step.data.get("researchPipeline")
        if not isinstance(pipeline, list):
            return
        for item in pipeline:
            if not isinstance(item, dict):
                continue
            agent = str(item.get("agent") or "").strip()
            if not agent:
                continue
            data = item.get("data") if isinstance(item.get("data"), dict) else {}
            steps.append(
                AgentStep(
                    agent,
                    str(item.get("status") or ""),
                    str(item.get("message") or ""),
                    dict(data),
                )
            )

    def _result_from_researched_action(
        self,
        context: BindingAgentContext,
        steps: list[AgentStep],
        *,
        gesture: str,
        action_spec: dict[str, Any],
        research: dict[str, Any],
    ) -> BindingAgentResult:
        policy_step = self.policy_agent.run(gesture, action_spec)
        steps.append(policy_step)
        missing = list(policy_step.data.get("missing") or [])

        validation_step = self.validation_agent.run(gesture, action_spec)
        steps.append(validation_step)
        validation_error = (
            validation_step.message
            if action_spec and validation_step.status == "blocked"
            else ""
        )
        command_name = (
            ""
            if validation_error
            else str(validation_step.data.get("command_name") or "")
        )
        ok = not validation_error and not missing
        can_apply = bool(action_spec and not missing and not validation_error)
        summary = [
            f"Жест: {gesture or 'не выбран'}",
            f"Команда: {command_name or _action_title(action_spec)}",
            f"Действие: {_action_title(action_spec)}",
        ]
        response = self._response_text(
            ok=ok,
            error=validation_error,
            missing=missing,
            gesture=gesture,
            action_spec=action_spec,
            research=research,
        )
        return BindingAgentResult(
            ok=ok,
            can_apply=can_apply,
            error=validation_error,
            missing=missing,
            gesture_label=gesture,
            command_name=command_name,
            mode="sequence" if action_spec.get("action") == "sequence" else "single",
            action_spec=action_spec,
            summary=summary if action_spec else [],
            response_text=response,
            steps=steps,
            research=research,
        )

    def _result_from_external_draft(
        self,
        context: BindingAgentContext,
        draft: dict[str, Any],
        steps: list[AgentStep],
    ) -> BindingAgentResult:
        gesture = str(draft.get("gestureLabel") or "").strip()
        action_spec = dict(draft.get("actionSpec") or {})
        forced_missing: list[str] = []
        if _action_conflicts_with_abstract_workflow(context.prompt, action_spec):
            action_spec = {}
            draft = {**draft, "agentReply": ""}
            forced_missing.append("действие")
        memory_step = self.memory_agent.run(context, gesture)
        steps.append(memory_step)
        policy_step = self.policy_agent.run(gesture, action_spec)
        steps.append(policy_step)
        model_missing = _normalize_missing(draft.get("missing"))
        missing = _unique_missing(
            model_missing + forced_missing + list(policy_step.data.get("missing") or [])
        )
        if gesture:
            missing = [
                item for item in missing if _norm(item) not in {"жест", "gesture"}
            ]
        if action_spec:
            missing = [
                item
                for item in missing
                if _norm(item) not in {"действие", "команда", "action"}
            ]
        validation_step = self.validation_agent.run(gesture, action_spec)
        steps.append(validation_step)
        validation_error = (
            validation_step.message
            if action_spec and validation_step.status == "blocked"
            else ""
        )

        command_name = str(draft.get("commandName") or "").strip()
        if not command_name:
            command_name = str(validation_step.data.get("command_name") or "")
        if validation_error:
            error = validation_error
            command_name = ""
        else:
            error = (
                ""
                if action_spec or "действие" in missing
                else "Не удалось понять действие"
            )
        ok = not error and not missing
        mode = str(draft.get("mode") or "").strip().lower()
        if mode not in {"single", "sequence"}:
            mode = "sequence" if action_spec.get("action") == "sequence" else "single"
        summary = _normalize_summary(draft.get("summary"))
        if not summary and action_spec:
            summary = [
                f"Жест: {gesture or 'не выбран'}",
                f"Команда: {command_name or _action_title(action_spec)}",
                f"Действие: {_action_title(action_spec)}",
            ]
        can_apply = bool(action_spec and not missing and not error)
        response = str(draft.get("agentReply") or "").strip()
        if not response or self._external_reply_conflicts_with_contract(
            response,
            gesture=gesture,
            action_spec=action_spec,
            missing=missing,
            error=error,
        ):
            response = self._response_text(
                ok=ok,
                error=error,
                missing=missing,
                gesture=gesture,
                action_spec=action_spec,
            )
        return BindingAgentResult(
            ok=ok,
            can_apply=can_apply,
            error=error,
            missing=missing,
            gesture_label=gesture,
            command_name=command_name,
            mode=mode,
            action_spec=action_spec,
            summary=summary,
            response_text=response,
            steps=steps,
        )

    def _external_reply_conflicts_with_contract(
        self,
        response: str,
        *,
        gesture: str,
        action_spec: dict[str, Any],
        missing: list[str],
        error: str,
    ) -> bool:
        if missing or error:
            return False
        if not (gesture and action_spec):
            return False
        lower = _norm(response)
        clarify_markers = (
            "уточн",
            "укаж",
            "выбер",
            "не указан",
            "не выбран",
            "не хватает",
            "осталось выбрать",
            "нужно выбрать",
            "нужно указать",
            "нужно уточнить",
        )
        field_markers = (
            "жест",
            "gesture",
            "действ",
            "команд",
            "action",
            "привяз",
        )
        return any(marker in lower for marker in clarify_markers) and any(
            marker in lower for marker in field_markers
        )

    def _response_text(
        self,
        *,
        ok: bool,
        error: str,
        missing: list[str],
        gesture: str,
        action_spec: dict[str, Any],
        research: dict[str, Any] | None = None,
    ) -> str:
        if error:
            return f"Локальный агент: не смог разобрать запрос. {error}."
        if missing:
            if "шаги сценария" in missing:
                return (
                    "Я собрал только понятные части сценария, но часть шагов не "
                    "разобрал. Уточните непонятный шаг: например дайте точный URL, "
                    "название приложения или команду."
                )
            if "действие" in missing:
                return (
                    "Я понял жест. Уточните, какую команду к нему привязать: "
                    "например открыть Safari, нажать command+z, показать уведомление "
                    "или перечислить шаги сценария."
                )
            return (
                "Локальный агент: действие понял, но нужно уточнить: "
                + ", ".join(missing)
                + "."
            )
        if research:
            source = str(research.get("sourceTitle") or "проверенный источник")
            action = research.get("actionTitle") or _action_title(action_spec)
            if research.get("learned"):
                return (
                    f"Нашёл это в сохранённом skill: «{action}». "
                    "Можно заполнить форму или сохранить привязку."
                )
            return (
                f"Я нашёл проверенный рецепт: «{action}». Источник: {source}. "
                "Если одобришь через «Заполнить» или «Сохранить», я запомню это как skill."
            )
        if ok:
            action = AGENT_ACTION_LABELS.get(
                str(action_spec.get("action") or ""),
                str(action_spec.get("action") or "команда"),
            )
            return (
                f"Локальный агент: понял. Подготовил «{gesture}» → {action}. "
                "Можно заполнить форму или сохранить."
            )
        return "Локальный агент: запрос обработан, но привязка требует проверки."


def build_agent_binding_draft(
    prompt: str,
    gestures: list[dict[str, Any]],
    *,
    current_gesture: str = "",
    conversation_history: list[dict[str, str]] | None = None,
    draft_state: dict[str, Any] | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper returning the UI-facing draft dict."""
    return BindingAgentOrchestrator().run(
        prompt,
        gestures,
        current_gesture=current_gesture,
        conversation_history=conversation_history,
        draft_state=draft_state,
        provider=provider,
    ).to_legacy_draft()
