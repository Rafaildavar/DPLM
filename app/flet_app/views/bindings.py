"""
Экран «Привязки жестов к командам и сценариям» — главная фича диплома.

Реализует правила из docs/BINDING_RULES.md:
    R1 — 1:1 жест↔команда (с переспросом при перезаписи)
    R3 — фильтр жестов: только is_active=True и model_class_id IS NOT NULL
    R6 — предупреждение для опасных действий, не привязанных к двуручному жесту
    R7 — валидация action_spec
    R8 — уникальность имени команды

Поддерживает два пользовательских режима:
    - одиночная команда: выбор категории, действия и параметров;
    - сценарий: конструктор из нескольких понятных шагов без ручного JSON.

Базовый поток:
    1. Выбрать жест из словаря (БД).
    2. Выбрать тип запуска: одна команда или сценарий.
    3. Заполнить действие/шаги и имя.
    4. Сохранить — привязка попадает в БД, ``CommandExecutor`` синхронизируется.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

import flet as ft

from app.flet_app.controller import AppController
from app.flet_app.theme import (
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_MUTED,
    COLOR_ON_SURFACE,
    COLOR_SUCCESS,
    COLOR_SURFACE_HIGH,
    COLOR_WARNING,
    surface_card,
)
from app.services.binding_agent import (
    binding_agent_provider_label,
    build_agent_binding_draft as _multiagent_binding_draft,
)

SEQUENCE_STEP_ACTIONS: list[dict[str, str]] = [
    {
        "action": "open_path",
        "label": "Открыть файл или папку",
        "value_label": "Путь к файлу или папке",
        "hint": "/Users/remi/Documents/DPLM Demo/Задание.pdf",
    },
    {
        "action": "open_app",
        "label": "Открыть приложение",
        "value_label": "Название приложения",
        "hint": "Preview",
    },
    {
        "action": "open_url",
        "label": "Открыть сайт",
        "value_label": "Адрес сайта",
        "hint": "https://example.com",
    },
    {
        "action": "key_combination",
        "label": "Нажать сочетание клавиш",
        "value_label": "Клавиши через запятую",
        "hint": "command,+",
    },
    {
        "action": "press",
        "label": "Нажать клавишу",
        "value_label": "Имя клавиши",
        "hint": "pagedown",
    },
    {
        "action": "wait",
        "label": "Подождать",
        "value_label": "Пауза в секундах",
        "hint": "1.0",
    },
    {
        "action": "notify",
        "label": "Показать уведомление",
        "value_label": "Текст уведомления",
        "hint": "Рабочее место готово",
    },
    {
        "action": "media_key",
        "label": "Управлять музыкой",
        "value_label": "play_pause, next или prev",
        "hint": "play_pause",
    },
    {
        "action": "volume_up",
        "label": "Увеличить громкость",
        "value_label": "",
        "hint": "",
    },
    {
        "action": "volume_down",
        "label": "Уменьшить громкость",
        "value_label": "",
        "hint": "",
    },
    {
        "action": "mute_toggle",
        "label": "Включить/выключить звук",
        "value_label": "",
        "hint": "",
    },
    {
        "action": "brightness_down",
        "label": "Уменьшить яркость",
        "value_label": "",
        "hint": "",
    },
    {
        "action": "brightness_up",
        "label": "Увеличить яркость",
        "value_label": "",
        "hint": "",
    },
    {
        "action": "run_script",
        "label": "Запустить Python-скрипт",
        "value_label": "Абсолютный путь к .py скрипту",
        "hint": "/Users/remi/Developer/GUAP/DPLM/scripts/demo_prepare_pr.py",
    },
]

STEP_LABEL_BY_ACTION = {
    item["action"]: item["label"] for item in SEQUENCE_STEP_ACTIONS
}

AGENT_ACTION_LABELS: dict[str, str] = {
    "open_app": "Открыть приложение",
    "open_path": "Открыть файл или папку",
    "open_url": "Открыть сайт",
    "key_combination": "Нажать сочетание клавиш",
    "press": "Нажать клавишу",
    "scroll": "Прокрутка",
    "wait": "Подождать",
    "notify": "Показать уведомление",
    "media_key": "Управлять музыкой",
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

_COMMON_APP_NAMES = (
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

_KEY_ALIASES: dict[str, str] = {
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

_SIMPLE_HOTKEYS: tuple[tuple[tuple[str, ...], list[str]], ...] = (
    (("отмена", "undo"), ["command", "z"]),
    (("повтор", "redo"), ["command", "shift", "z"]),
    (("копир", "copy"), ["command", "c"]),
    (("встав", "paste"), ["command", "v"]),
    (("поиск", "find"), ["command", "f"]),
)


def _agent_norm(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


def _agent_clean_value(value: str) -> str:
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


def _agent_quoted_value(text: str) -> str:
    match = re.search(r"[«\"']([^»\"']+)[»\"']", text or "")
    return _agent_clean_value(match.group(1)) if match else ""


def _agent_explicit_gesture_label(prompt: str) -> str:
    match = re.search(
        r"(?:^|\s)(?:жест(?:ом|а)?|gesture)\s*[:=]?\s+([A-Za-zА-Яа-я0-9_.-]+)",
        prompt or "",
        re.IGNORECASE,
    )
    if not match:
        return ""
    label = _agent_clean_value(match.group(1))
    if _agent_norm(label) in {
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


def _agent_match_gesture_label(
    prompt: str,
    gestures: list[dict[str, Any]],
    current_gesture: str = "",
) -> str:
    labels = [
        str(item.get("label") or "").strip()
        for item in gestures
        if str(item.get("label") or "").strip()
    ]
    prompt_norm = _agent_norm(prompt)
    explicit = _agent_explicit_gesture_label(prompt)
    explicit_norm = _agent_norm(explicit)
    if explicit_norm:
        for label in labels:
            if _agent_norm(label) == explicit_norm:
                return label
        explicit_words = re.sub(r"[_-]+", " ", explicit_norm)
        for label in labels:
            if re.sub(r"[_-]+", " ", _agent_norm(label)) == explicit_words:
                return label
    prompt_words = re.sub(r"[_-]+", " ", prompt_norm)
    for label in sorted(labels, key=len, reverse=True):
        label_norm = _agent_norm(label)
        if re.search(rf"(?<!\w){re.escape(label_norm)}(?!\w)", prompt_norm):
            return label
        label_words = re.sub(r"[_-]+", " ", label_norm)
        if label_words != label_norm and re.search(
            rf"(?<!\w){re.escape(label_words)}(?!\w)",
            prompt_words,
        ):
            return label
    current = (current_gesture or "").strip()
    if current and current.lower() in {label.lower() for label in labels}:
        return next(label for label in labels if label.lower() == current.lower())
    if explicit:
        return explicit
    return ""


def _agent_find_url(text: str) -> str:
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
    return ""


def _agent_find_path(text: str, *, python_only: bool = False) -> str:
    quoted = _agent_quoted_value(text)
    if quoted.startswith(("/", "~")) and (not python_only or quoted.endswith(".py")):
        return quoted
    suffix = r"\.py" if python_only else r"(?:\.[a-z0-9]{1,8})?"
    match = re.search(
        rf"((?:~|/)[^\s,;]+{suffix})",
        text or "",
        re.IGNORECASE,
    )
    return match.group(1).rstrip(".,;)") if match else ""


def _agent_normalize_key(token: str) -> str:
    clean = _agent_norm(token).strip(" \t\r\n\"'`«»;:()[]{}")
    if not clean:
        return ""
    clean = clean.replace("page down", "pagedown").replace("page up", "pageup")
    if clean in _KEY_ALIASES:
        return _KEY_ALIASES[clean]
    if len(clean) == 1 and re.match(r"[a-z0-9,.\-+=/]", clean):
        return clean
    return ""


def _agent_hotkey_from_text(text: str) -> list[str]:
    lower = _agent_norm(text)
    plus_match = re.search(
        r"([a-zа-я0-9⌘⌥⇧⌃_,.\-]+(?:\s*\+\s*[a-zа-я0-9⌘⌥⇧⌃_,.\-]+)+)",
        text or "",
        re.IGNORECASE,
    )
    if plus_match:
        keys = [
            _agent_normalize_key(part)
            for part in re.split(r"\s*\+\s*", plus_match.group(1))
        ]
        keys = [key for key in keys if key]
        if len(keys) >= 2:
            return keys

    for markers, keys in _SIMPLE_HOTKEYS:
        if any(marker in lower for marker in markers):
            return list(keys)

    if not any(
        marker in lower
        for marker in ("cmd", "command", "ctrl", "control", "⌘", "⌃")
    ):
        return []
    tokens = re.findall(r"[a-zа-я0-9⌘⌥⇧⌃]+|[,.\-+=/]", text or "", re.IGNORECASE)
    keys = [_agent_normalize_key(token) for token in tokens]
    keys = [key for key in keys if key]
    if (
        any(key in {"command", "ctrl", "option", "shift"} for key in keys)
        and len(keys) >= 2
    ):
        return keys
    return []


def _agent_number(text: str, default: float) -> float:
    match = re.search(r"(\d+(?:[,.]\d+)?)", text or "")
    if not match:
        return default
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return default


def _agent_extract_app_name(text: str) -> str:
    quoted = _agent_quoted_value(text)
    if quoted and not quoted.startswith(("/", "~")) and not _agent_find_url(quoted):
        return quoted
    lower = _agent_norm(text)
    for app in _COMMON_APP_NAMES:
        if _agent_norm(app) in lower:
            return app
    patterns = (
        (
            r"(?:откр\w*|запуст\w*|старт\w*)\s+"
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
            value = _agent_clean_value(match.group("value"))
            if value and not value.startswith(("http://", "https://", "/", "~")):
                return value
    return ""


def _agent_parse_action(text: str) -> dict[str, Any] | None:
    lower = _agent_norm(text)

    url = _agent_find_url(text)
    if url:
        return {"action": "open_url", "platform": "macos", "url": url}

    script_path = _agent_find_path(text, python_only=True)
    if script_path and ("скрипт" in lower or "script" in lower or ".py" in lower):
        return {
            "action": "run_script",
            "platform": "macos",
            "script_path": script_path,
        }

    path = _agent_find_path(text)
    if path and any(marker in lower for marker in ("файл", "папк", "путь", "откр")):
        return {"action": "open_path", "platform": "macos", "path": path}

    keys = _agent_hotkey_from_text(text)
    if keys:
        return {"action": "key_combination", "platform": "macos", "keys": keys}

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
        return {
            "action": "wait",
            "platform": "macos",
            "seconds": _agent_number(text, 1.0),
        }

    if any(marker in lower for marker in ("уведом", "notify", "сообщени")):
        message = _agent_quoted_value(text)
        if not message:
            match = re.search(
                r"(?:уведом\w*|notify|сообщени\w*)\s+(?P<value>[^,.;\n]+)",
                text or "",
                re.IGNORECASE,
            )
            message = _agent_clean_value(match.group("value")) if match else "Готово"
        return {
            "action": "notify",
            "platform": "macos",
            "title": "DPLM",
            "message": message or "Готово",
        }

    if any(marker in lower for marker in ("скрол", "прокрут")):
        amount = int(_agent_number(text, 5))
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

    app = _agent_extract_app_name(text)
    if app:
        return {"action": "open_app", "platform": "macos", "app": app}
    return None


def _agent_split_sequence(text: str) -> list[str]:
    body = re.sub(r"^\s*сценар\w*\s*:?", "", text or "", flags=re.IGNORECASE)
    body = re.sub(
        r"\s+(?:и\s+)?(?:потом|затем|после этого|далее)\s+",
        ";",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        (
            r"\s+и\s+(?=откр|запуст|покаж|уведом|подожд|нажм"
            r"|сделай|увелич|уменьш|заблок|скрин)"
        ),
        ";",
        body,
        flags=re.IGNORECASE,
    )
    return [
        _agent_clean_value(part)
        for part in re.split(r"[;\n]+", body)
        if _agent_clean_value(part)
    ]


def _agent_action_title(spec: dict[str, Any]) -> str:
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
    if action == "sequence":
        return f"Сценарий из {len(spec.get('steps') or [])} шагов"
    return AGENT_ACTION_LABELS.get(action, action or "Команда")


def build_agent_binding_draft(
    prompt: str,
    gestures: list[dict[str, Any]],
    *,
    current_gesture: str = "",
    conversation_history: list[dict[str, str]] | None = None,
    draft_state: dict[str, Any] | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper around the multi-agent pipeline."""
    return _multiagent_binding_draft(
        prompt,
        gestures,
        current_gesture=current_gesture,
        conversation_history=conversation_history,
        draft_state=draft_state,
        provider=provider,
    )


class BindingsView:
    """Экран привязок: ComboBox жестов / категорий / действий + форма."""

    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller

        # Кэш данных
        self._gestures: list[dict] = []
        self._categories: list[dict] = []
        self._actions: list[dict] = []
        self._default_commands: list[dict] = []
        self._sequence_steps: list[dict] = []

        # Контролы
        # Flet 0.85: Dropdown использует on_select, а не on_change.
        self._gesture_dd = ft.Dropdown(
            label="Жест из словаря",
            hint_text="Выберите жест",
            border_color=COLOR_SURFACE_HIGH,
            on_select=self._on_gesture_changed,
            editable=False,
        )
        self._category_dd = ft.Dropdown(
            label="Категория команды",
            border_color=COLOR_SURFACE_HIGH,
            on_select=self._on_category_changed,
            editable=False,
        )
        self._default_command_dd = ft.Dropdown(
            label="Готовая команда",
            hint_text="Можно выбрать готовую команду или заполнить форму ниже",
            border_color=COLOR_SURFACE_HIGH,
            on_select=self._on_default_command_changed,
            editable=False,
        )
        self._mode_dd = ft.Dropdown(
            label="Что запускать жестом",
            value="single",
            border_color=COLOR_SURFACE_HIGH,
            options=[
                ft.DropdownOption(key="single", text="Одна команда"),
                ft.DropdownOption(key="sequence", text="Сценарий из нескольких шагов"),
            ],
            on_select=self._on_mode_changed,
            editable=False,
        )
        self._action_dd = ft.Dropdown(
            label="Действие",
            border_color=COLOR_SURFACE_HIGH,
            on_select=self._on_action_changed,
            editable=False,
        )
        self._step_action_dd = ft.Dropdown(
            label="Добавить шаг",
            value="open_path",
            border_color=COLOR_SURFACE_HIGH,
            options=[
                ft.DropdownOption(key=item["action"], text=item["label"])
                for item in SEQUENCE_STEP_ACTIONS
            ],
            on_select=self._on_step_action_changed,
            editable=False,
        )
        self._step_value_field = ft.TextField(
            label="Путь к файлу или папке",
            hint_text="/Users/remi/Documents/DPLM Demo/Задание.pdf",
            border_color=COLOR_SURFACE_HIGH,
        )
        self._step_title_field = ft.TextField(
            label="Заголовок уведомления",
            value="DPLM",
            border_color=COLOR_SURFACE_HIGH,
            visible=False,
        )
        self._add_step_btn = ft.FilledButton(
            content=ft.Text("Добавить шаг"),
            icon=ft.Icons.ADD,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
            ),
            on_click=self._on_add_step_click,
        )
        self._clear_steps_btn = ft.OutlinedButton(
            content=ft.Text("Очистить сценарий"),
            icon=ft.Icons.CLEAR_ALL,
            on_click=self._on_clear_steps_click,
        )
        self._steps_column = ft.Column(spacing=8)
        self._sequence_hint = ft.Text(
            "Шаги выполняются сверху вниз.",
            size=12,
            color=COLOR_MUTED,
        )
        self._single_mode_panel = ft.Column(spacing=12)
        self._sequence_panel = ft.Column(spacing=12, visible=False)
        self._params_field = ft.TextField(
            label="Параметры (JSON)",
            value="{}",
            multiline=True,
            min_lines=1,
            max_lines=4,
            border_color=COLOR_SURFACE_HIGH,
            text_style=ft.TextStyle(font_family="Menlo", size=13),
        )
        self._name_field = ft.TextField(
            label="Имя команды",
            hint_text="Например: Открыть Safari",
            border_color=COLOR_SURFACE_HIGH,
            on_change=lambda _e: self._refresh_overwrite_hint(),
        )

        self._field_hints = ft.Text("", size=12, color=COLOR_MUTED)

        self._warn_two_hands = ft.Container(
            content=ft.Text(
                "⚠ Это «опасное» действие. Рекомендуется привязать его к "
                "двуручному жесту (правило R6).",
                color="#FFB300",
                size=12,
            ),
            bgcolor="#3a2a14",
            border=ft.Border.all(1, "#FFB300"),
            border_radius=8,
            padding=10,
            visible=False,
        )
        self._overwrite_hint = ft.Container(
            content=ft.Text("", color=COLOR_ACCENT, size=12),
            bgcolor="#1a2a3a",
            border=ft.Border.all(1, COLOR_ACCENT),
            border_radius=8,
            padding=10,
            visible=False,
        )

        self._save_btn = ft.FilledButton(
            content=ft.Text("Сохранить привязку", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.SAVE,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=24, vertical=16),
            ),
            on_click=self._on_save_click,
        )

        self._test_btn = ft.OutlinedButton(
            content=ft.Text("Тестовый запуск"),
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_test_click,
        )

        self._refresh_btn = ft.IconButton(
            icon=ft.Icons.REFRESH,
            tooltip="Обновить список жестов из БД",
            on_click=lambda _e: self._refresh_all_lists(),
        )
        self._bindings_total_value = ft.Text(
            "0",
            size=20,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_SURFACE,
        )
        self._bindings_bound_value = ft.Text(
            "0",
            size=20,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_SURFACE,
        )
        self._bindings_free_value = ft.Text(
            "0",
            size=20,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_SURFACE,
        )
        self._bindings_focus_title = ft.Text(
            "Привязки загружаются",
            size=14,
            weight=ft.FontWeight.W_600,
            color=COLOR_ON_SURFACE,
        )
        self._bindings_focus_meta = ft.Text(
            "Список обновится после чтения БД",
            size=12,
            color=COLOR_MUTED,
        )
        self._bindings_chips_row = ft.Row(spacing=8, wrap=True)

        self._error_text = ft.Text("", color=COLOR_DANGER, size=13, visible=False)
        self._info_text = ft.Text("", color=COLOR_SUCCESS, size=13, visible=False)
        self._last_agent_draft: dict[str, Any] | None = None
        self._agent_dialog_messages: list[dict[str, str]] = []
        self._agent_request_id = 0
        self._agent_input = ft.TextField(
            hint_text="Спросите агента привязки",
            multiline=True,
            min_lines=1,
            max_lines=3,
            shift_enter=True,
            border=ft.InputBorder.NONE,
            bgcolor="#242426",
            color=COLOR_ON_SURFACE,
            text_size=16,
            hint_style=ft.TextStyle(color="#A7A7AA", size=16),
            content_padding=ft.Padding(0, 0, 0, 0),
            expand=True,
            on_submit=self._on_agent_parse_click,
        )
        self._agent_parse_btn = ft.IconButton(
            icon=ft.Icons.SEND,
            icon_color="#111315",
            icon_size=22,
            tooltip="Отправить агенту",
            bgcolor="#F4F4F3",
            size_constraints=ft.BoxConstraints(
                min_width=52,
                min_height=52,
                max_width=52,
                max_height=52,
            ),
            on_click=self._on_agent_parse_click,
        )
        self._agent_apply_btn = ft.OutlinedButton(
            content=ft.Text("Заполнить"),
            icon=ft.Icons.CHECK,
            disabled=True,
            on_click=self._on_agent_apply_click,
        )
        self._agent_save_btn = ft.FilledButton(
            content=ft.Text("Сохранить"),
            icon=ft.Icons.SAVE,
            disabled=True,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
            ),
            on_click=self._on_agent_save_click,
        )
        self._agent_result = ft.Column(
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            width=float("inf"),
        )
        self._agent_status = ft.Text("", size=12, color=COLOR_MUTED)
        self._style_form_controls()
        self._configure_mode_panels()
        self._render_sequence_steps()
        self._set_agent_empty_state()

    # ---- Жизненный цикл ---------------------------------------------------

    def _style_form_controls(self) -> None:
        controls = (
            self._gesture_dd,
            self._category_dd,
            self._default_command_dd,
            self._mode_dd,
            self._action_dd,
            self._step_action_dd,
            self._step_value_field,
            self._step_title_field,
            self._params_field,
            self._name_field,
        )
        for control in controls:
            control.filled = True
            control.fill_color = "#181B1F"
            control.border_radius = 8
            control.border_color = COLOR_SURFACE_HIGH
            control.focused_border_color = COLOR_ACCENT
            control.content_padding = ft.Padding(14, 12, 14, 12)

    def on_show(self) -> None:
        # Перечитываем словарь жестов из БД (R3) и категории при каждом показе.
        self._refresh_categories()
        self._refresh_default_commands()
        self._refresh_gestures()

    def on_hide(self) -> None:
        pass

    # ---- Данные -----------------------------------------------------------

    def _refresh_categories(self) -> None:
        self._categories = [
            item
            for item in self._controller.get_action_categories()
            if item.get("id") != "workflow"
        ]
        self._category_dd.options = [
            ft.DropdownOption(key=c["id"], text=c["label"]) for c in self._categories
        ]
        try:
            self._category_dd.update()
        except Exception:
            pass

    def _refresh_default_commands(self) -> None:
        self._default_commands = self._controller.list_commands()
        options: list[ft.DropdownOption] = []
        for command in self._default_commands:
            name = str(command.get("name") or "")
            action = str(command.get("action") or "")
            platform = str(command.get("platform") or "all")
            text = name
            detail = " · ".join(part for part in (action, platform) if part)
            if detail:
                text = f"{name} — {detail}"
            options.append(ft.DropdownOption(key=name, text=text))
        self._default_command_dd.options = options
        self._default_command_dd.hint_text = (
            "Готовые команды не найдены"
            if not options
            else "Можно выбрать готовую команду или заполнить форму ниже"
        )
        try:
            self._default_command_dd.update()
        except Exception:
            pass

    def _refresh_all_lists(self) -> None:
        self._refresh_default_commands()
        self._refresh_gestures()

    def _binding_metric_tile(
        self,
        label: str,
        value: ft.Text,
        icon: str,
        color: str,
    ) -> ft.Control:
        return ft.Container(
            padding=ft.Padding(12, 10, 12, 10),
            border_radius=8,
            bgcolor="#17191D",
            border=ft.Border.all(1, COLOR_SURFACE_HIGH),
            content=ft.Row(
                spacing=9,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Container(
                        width=32,
                        height=32,
                        border_radius=8,
                        bgcolor="#1B1D21",
                        alignment=ft.Alignment.CENTER,
                        content=ft.Icon(icon, color=color, size=18),
                    ),
                    ft.Column(
                        spacing=0,
                        controls=[
                            value,
                            ft.Text(label, size=11, color=COLOR_MUTED),
                        ],
                    ),
                ],
            ),
        )

    def _select_gesture_from_overview(self, label: str) -> None:
        self._gesture_dd.value = label
        self._on_gesture_changed(None)
        try:
            self._gesture_dd.update()
        except Exception:
            pass

    def _binding_chip(
        self,
        label: str,
        command: str,
        *,
        selected: bool = False,
    ) -> ft.Control:
        return ft.Container(
            padding=ft.Padding(10, 7, 10, 7),
            border_radius=8,
            bgcolor="#17262A" if selected else "#1B1D21",
            border=ft.Border.all(1, "#21434A" if selected else COLOR_SURFACE_HIGH),
            ink=True,
            on_click=lambda _e, value=label: self._select_gesture_from_overview(value),
            content=ft.Row(
                spacing=7,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Icon(
                        ft.Icons.LINK,
                        color=COLOR_ACCENT if selected else COLOR_MUTED,
                        size=14,
                    ),
                    ft.Text(label, color=COLOR_ON_SURFACE, size=12),
                    ft.Text("→", color=COLOR_MUTED, size=12),
                    ft.Text(
                        command,
                        color=COLOR_MUTED,
                        size=12,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                ],
            ),
        )

    def _refresh_bindings_overview(self) -> None:
        total = len(self._gestures)
        bound = [
            item
            for item in self._gestures
            if str(item.get("boundCommandName") or "").strip()
        ]
        free = max(total - len(bound), 0)
        self._bindings_total_value.value = str(total)
        self._bindings_bound_value.value = str(len(bound))
        self._bindings_free_value.value = str(free)

        selected = self._selected_gesture()
        if selected:
            label = str(selected.get("label") or "")
            command = str(selected.get("boundCommandName") or "").strip()
            if command:
                self._bindings_focus_title.value = f"{label} → {command}"
                self._bindings_focus_meta.value = "Выбранная активная привязка"
            else:
                self._bindings_focus_title.value = label
                self._bindings_focus_meta.value = "Жест свободен для новой команды"
        elif bound:
            self._bindings_focus_title.value = "Активные привязки готовы"
            self._bindings_focus_meta.value = (
                "Выберите жест ниже или нажмите чип для быстрого перехода"
            )
        elif total:
            self._bindings_focus_title.value = "Жесты есть, привязок пока нет"
            self._bindings_focus_meta.value = "Создайте первую команду через форму или агента"
        else:
            self._bindings_focus_title.value = "Жесты не найдены"
            self._bindings_focus_meta.value = "Добавьте жесты на экране «Жесты»"

        selected_label = str((selected or {}).get("label") or "")
        chips: list[ft.Control] = []
        for item in bound[:6]:
            label = str(item.get("label") or "")
            command = str(item.get("boundCommandName") or "")
            chips.append(
                self._binding_chip(
                    label,
                    command,
                    selected=bool(label and label == selected_label),
                )
            )
        if len(bound) > 6:
            chips.append(
                ft.Container(
                    padding=ft.Padding(10, 7, 10, 7),
                    border_radius=8,
                    bgcolor="#1B1D21",
                    border=ft.Border.all(1, COLOR_SURFACE_HIGH),
                    content=ft.Text(
                        f"+{len(bound) - 6}",
                        size=12,
                        color=COLOR_MUTED,
                    ),
                )
            )
        if not chips:
            chips = [
                ft.Text(
                    "Сохраненные привязки появятся здесь",
                    size=12,
                    color=COLOR_MUTED,
                )
            ]
        self._bindings_chips_row.controls = chips

        for control in (
            self._bindings_total_value,
            self._bindings_bound_value,
            self._bindings_free_value,
            self._bindings_focus_title,
            self._bindings_focus_meta,
            self._bindings_chips_row,
        ):
            try:
                control.update()
            except Exception:
                pass

    def _build_bindings_overview(self) -> ft.Control:
        return surface_card(
            ft.Column(
                spacing=12,
                controls=[
                    ft.Row(
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Column(
                                spacing=2,
                                expand=True,
                                controls=[
                                    self._bindings_focus_title,
                                    self._bindings_focus_meta,
                                ],
                            ),
                            self._binding_metric_tile(
                                "жестов",
                                self._bindings_total_value,
                                ft.Icons.BACK_HAND,
                                COLOR_ACCENT,
                            ),
                            self._binding_metric_tile(
                                "связано",
                                self._bindings_bound_value,
                                ft.Icons.LINK,
                                COLOR_SUCCESS,
                            ),
                            self._binding_metric_tile(
                                "свободно",
                                self._bindings_free_value,
                                ft.Icons.ADD_LINK,
                                COLOR_WARNING,
                            ),
                        ],
                    ),
                    ft.Divider(height=1, color=COLOR_SURFACE_HIGH),
                    self._bindings_chips_row,
                ],
            ),
            padding=14,
            radius=8,
        )

    def _refresh_gestures(self) -> None:
        self._gestures = self._controller.get_db_gestures()
        opts: list[ft.DropdownOption] = []
        for g in self._gestures:
            text = g["label"]
            bound = g.get("boundCommandName") or ""
            if bound:
                text = f"{text}  →  {bound}"
            opts.append(ft.DropdownOption(key=g["label"], text=text))
        self._gesture_dd.options = opts
        self._gesture_dd.hint_text = (
            "Нет жестов в БД — добавьте на экране «Жесты»"
            if not opts
            else "Выберите жест"
        )
        try:
            self._gesture_dd.update()
        except Exception:
            pass
        self._refresh_overwrite_hint()
        self._refresh_bindings_overview()

    def _refresh_actions(self) -> None:
        cat_id = self._category_dd.value or ""
        self._actions = self._controller.get_actions_for_category(cat_id)
        self._action_dd.options = [
            ft.DropdownOption(key=a["action"], text=a["action"]) for a in self._actions
        ]
        self._action_dd.value = None
        self._field_hints.value = ""
        try:
            self._action_dd.update()
            self._field_hints.update()
        except Exception:
            pass

    # ---- Обработчики ------------------------------------------------------

    def _selected_gesture(self) -> dict | None:
        val = self._gesture_dd.value
        if not val:
            return None
        for g in self._gestures:
            if g["label"] == val:
                return g
        return None

    def _selected_action(self) -> dict | None:
        val = self._action_dd.value
        if not val:
            return None
        for a in self._actions:
            if a["action"] == val:
                return a
        return None

    def _is_sequence_mode(self) -> bool:
        return (self._mode_dd.value or "single") == "sequence"

    def _on_gesture_changed(self, _e) -> None:
        self._refresh_overwrite_hint()
        self._refresh_warn_two_hands()
        self._refresh_bindings_overview()

    def _on_mode_changed(self, _e) -> None:
        self._refresh_mode_visibility()
        if self._is_sequence_mode() and not (self._name_field.value or "").strip():
            self._name_field.value = "Сценарий: подготовить рабочее место"
        self._refresh_warn_two_hands()
        try:
            self._name_field.update()
        except Exception:
            pass

    def _on_category_changed(self, _e) -> None:
        self._refresh_actions()
        self._refresh_warn_two_hands()

    def _on_default_command_changed(self, _e) -> None:
        name = (self._default_command_dd.value or "").strip()
        command = next(
            (item for item in self._default_commands if item.get("name") == name),
            None,
        )
        if command is None:
            return

        action = str(command.get("action") or "").strip()
        category_id = str(command.get("category") or "").strip()
        config = dict(command.get("config") or {})
        if action == "sequence":
            self._mode_dd.value = "sequence"
            steps = config.get("steps") or []
            self._sequence_steps = [
                dict(step) for step in steps if isinstance(step, dict)
            ]
            self._render_sequence_steps()
            self._name_field.value = name
            self._refresh_mode_visibility()
            try:
                self._mode_dd.update()
                self._name_field.update()
            except Exception:
                pass
            self._refresh_overwrite_hint()
            self._refresh_warn_two_hands()
            return

        self._mode_dd.value = "single"
        self._refresh_mode_visibility()
        if category_id:
            self._category_dd.value = category_id
            self._refresh_actions()
        if action:
            self._action_dd.value = action

        params = config
        params.pop("action", None)
        params.pop("platform", None)
        self._params_field.value = json.dumps(params, ensure_ascii=False)
        self._name_field.value = name

        selected = self._selected_action()
        hints = selected.get("fieldHints") if selected else {}
        self._field_hints.value = (
            "\n".join(f"• {k}: {v}" for k, v in hints.items())
            if hints
            else "Готовая команда: параметры уже подставлены"
        )
        try:
            self._category_dd.update()
            self._action_dd.update()
            self._mode_dd.update()
            self._params_field.update()
            self._name_field.update()
            self._field_hints.update()
        except Exception:
            pass
        self._refresh_overwrite_hint()
        self._refresh_warn_two_hands()

    def _on_step_action_changed(self, _e) -> None:
        self._refresh_step_fields()

    def _on_add_step_click(self, _e) -> None:
        self._show_message("", "")
        spec = self._build_sequence_step()
        if isinstance(spec, str):
            self._show_message(error=spec)
            return
        err = self._controller.validate_action_spec_json(
            json.dumps(spec, ensure_ascii=False)
        )
        if err:
            self._show_message(error=err)
            return
        self._sequence_steps.append(spec)
        self._step_value_field.value = ""
        self._render_sequence_steps()
        self._refresh_warn_two_hands()
        try:
            self._step_value_field.update()
        except Exception:
            pass

    def _on_clear_steps_click(self, _e) -> None:
        self._sequence_steps = []
        self._render_sequence_steps()
        self._refresh_warn_two_hands()

    def _on_action_changed(self, _e) -> None:
        action = self._selected_action()
        if action is None:
            return
        # Подставить пример параметров в поле, очищая action/platform.
        example = dict(action.get("example") or {})
        example.pop("action", None)
        example.pop("platform", None)
        self._params_field.value = json.dumps(example, ensure_ascii=False)
        if not (self._name_field.value or "").strip():
            self._name_field.value = f"User: {action['action']}"
        # Подсказки по полям.
        hints = action.get("fieldHints") or {}
        if hints:
            self._field_hints.value = "\n".join(f"• {k}: {v}" for k, v in hints.items())
        else:
            self._field_hints.value = "Без дополнительных параметров"
        try:
            self._params_field.update()
            self._name_field.update()
            self._field_hints.update()
        except Exception:
            pass
        self._refresh_warn_two_hands()

    def _refresh_warn_two_hands(self) -> None:
        g = self._selected_gesture()
        show = False
        if g is not None and not g.get("isTwoHands"):
            if self._is_sequence_mode():
                show = any(
                    self._controller.is_action_dangerous(
                        str(step.get("action") or "")
                    )
                    for step in self._sequence_steps
                )
            else:
                a = self._selected_action()
                if a is not None and self._controller.is_action_dangerous(a["action"]):
                    show = True
        if self._warn_two_hands.visible != show:
            self._warn_two_hands.visible = show
            try:
                self._warn_two_hands.update()
            except Exception:
                pass

    def _refresh_overwrite_hint(self) -> None:
        g = self._selected_gesture()
        if g is None:
            show = False
            msg = ""
        else:
            bound = (g.get("boundCommandName") or "").strip()
            current = (self._name_field.value or "").strip()
            if bound and bound != current:
                show = True
                msg = (
                    f"На этом жесте уже есть команда «{bound}». "
                    "При сохранении она будет заменена (правило R1)."
                )
            else:
                show = False
                msg = ""
        if msg and isinstance(self._overwrite_hint.content, ft.Text):
            self._overwrite_hint.content.value = msg
        self._overwrite_hint.visible = show
        try:
            self._overwrite_hint.update()
        except Exception:
            pass

    def _build_action_spec(self) -> dict | str:
        """Собрать action_spec из выбора + параметров. Возвращает dict или
        сообщение об ошибке (str)."""
        if self._is_sequence_mode():
            if not self._sequence_steps:
                return "Добавьте хотя бы один шаг сценария"
            return {
                "action": "sequence",
                "platform": "macos",
                "steps": [dict(step) for step in self._sequence_steps],
            }

        a = self._selected_action()
        if a is None:
            return "Выберите действие"
        spec: dict = {"action": a["action"], "platform": "macos"}
        raw = (self._params_field.value or "").strip()
        if raw:
            try:
                extra = json.loads(raw)
            except json.JSONDecodeError as e:
                return f"Параметры: некорректный JSON: {e}"
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if k not in ("action", "platform"):
                        spec[k] = v
        return spec

    def _build_sequence_step(self) -> dict | str:
        action = (self._step_action_dd.value or "").strip()
        if not action:
            return "Выберите тип шага"
        raw = (self._step_value_field.value or "").strip()

        if action == "open_path":
            if not raw:
                return "Укажите путь к файлу или папке"
            return {"action": action, "path": raw}
        if action == "open_app":
            if not raw:
                return "Укажите название приложения"
            return {"action": action, "app": raw}
        if action == "open_url":
            if not raw:
                return "Укажите адрес сайта"
            return {"action": action, "url": raw}
        if action == "key_combination":
            keys = [part.strip() for part in raw.split(",") if part.strip()]
            if not keys:
                return "Укажите клавиши через запятую, например command,+"
            return {"action": action, "keys": keys}
        if action == "press":
            if not raw:
                return "Укажите имя клавиши"
            return {"action": action, "key": raw}
        if action == "wait":
            if not raw:
                return "Укажите паузу в секундах"
            return {"action": action, "seconds": raw.replace(",", ".")}
        if action == "notify":
            if not raw:
                return "Укажите текст уведомления"
            title = (self._step_title_field.value or "DPLM").strip() or "DPLM"
            return {"action": action, "title": title, "message": raw}
        if action == "media_key":
            if not raw:
                return "Укажите play_pause, next или prev"
            return {"action": action, "kind": raw}
        if action == "run_script":
            if not raw:
                return "Укажите абсолютный путь к .py скрипту"
            return {"action": action, "script_path": raw}
        return {"action": action}

    def _refresh_mode_visibility(self) -> None:
        sequence = self._is_sequence_mode()
        self._single_mode_panel.visible = not sequence
        self._sequence_panel.visible = sequence
        try:
            self._single_mode_panel.update()
            self._sequence_panel.update()
        except Exception:
            pass

    def _refresh_step_fields(self) -> None:
        action = (self._step_action_dd.value or "open_path").strip()
        meta = next(
            (item for item in SEQUENCE_STEP_ACTIONS if item["action"] == action),
            SEQUENCE_STEP_ACTIONS[0],
        )
        value_label = meta.get("value_label") or ""
        self._step_value_field.label = value_label
        self._step_value_field.hint_text = meta.get("hint") or ""
        self._step_value_field.visible = bool(value_label)
        self._step_title_field.visible = action == "notify"
        try:
            self._step_value_field.update()
            self._step_title_field.update()
        except Exception:
            pass

    def _step_summary(self, step: dict, index: int) -> str:
        action = str(step.get("action") or "")
        label = STEP_LABEL_BY_ACTION.get(action, action)
        if action == "open_path":
            detail = step.get("path", "")
        elif action == "open_app":
            detail = step.get("app", "")
        elif action == "open_url":
            detail = step.get("url", "")
        elif action == "key_combination":
            detail = " + ".join(str(k) for k in step.get("keys", []))
        elif action == "press":
            detail = step.get("key", "")
        elif action == "wait":
            detail = f"{step.get('seconds', '')} сек."
        elif action == "notify":
            detail = step.get("message", "")
        elif action == "media_key":
            detail = step.get("kind", "")
        elif action == "run_script":
            detail = step.get("script_path", "")
        else:
            detail = ""
        suffix = f": {detail}" if detail else ""
        return f"{index}. {label}{suffix}"

    def _move_sequence_step(self, index: int, delta: int) -> None:
        target = index + delta
        if target < 0 or target >= len(self._sequence_steps):
            return
        self._sequence_steps[index], self._sequence_steps[target] = (
            self._sequence_steps[target],
            self._sequence_steps[index],
        )
        self._render_sequence_steps()

    def _remove_sequence_step(self, index: int) -> None:
        if 0 <= index < len(self._sequence_steps):
            self._sequence_steps.pop(index)
            self._render_sequence_steps()
            self._refresh_warn_two_hands()

    def _render_sequence_steps(self) -> None:
        if not self._sequence_steps:
            self._steps_column.controls = [
                ft.Container(
                    content=ft.Text(
                        "Пока нет шагов. Добавьте действие выше.",
                        color=COLOR_MUTED,
                        size=12,
                    ),
                    bgcolor=COLOR_SURFACE_HIGH,
                    border_radius=8,
                    padding=12,
                )
            ]
        else:
            controls: list[ft.Control] = []
            for index, step in enumerate(self._sequence_steps):
                controls.append(
                    ft.Container(
                        content=ft.Row(
                            controls=[
                                ft.Icon(
                                    ft.Icons.DRAG_INDICATOR,
                                    color=COLOR_MUTED,
                                    size=18,
                                ),
                                ft.Text(
                                    self._step_summary(step, index + 1),
                                    color=COLOR_ON_SURFACE,
                                    size=13,
                                    expand=True,
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.ARROW_UPWARD,
                                    tooltip="Поднять шаг",
                                    disabled=index == 0,
                                    on_click=lambda _e, i=index: self._move_sequence_step(i, -1),
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.ARROW_DOWNWARD,
                                    tooltip="Опустить шаг",
                                    disabled=index == len(self._sequence_steps) - 1,
                                    on_click=lambda _e, i=index: self._move_sequence_step(i, 1),
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.DELETE_OUTLINE,
                                    tooltip="Удалить шаг",
                                    icon_color=COLOR_DANGER,
                                    on_click=lambda _e, i=index: self._remove_sequence_step(i),
                                ),
                            ],
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        bgcolor=COLOR_SURFACE_HIGH,
                        border_radius=8,
                        padding=8,
                    )
                )
            self._steps_column.controls = controls
        try:
            self._steps_column.update()
        except Exception:
            pass

    def _configure_mode_panels(self) -> None:
        self._single_mode_panel.controls = [
            ft.Text(
                "Готовая команда",
                size=14,
                weight=ft.FontWeight.W_600,
                color=COLOR_ON_SURFACE,
            ),
            self._default_command_dd,
            ft.Text(
                "Категория и действие",
                size=14,
                weight=ft.FontWeight.W_600,
                color=COLOR_ON_SURFACE,
            ),
            self._category_dd,
            self._action_dd,
            self._field_hints,
            ft.Text(
                "Параметры",
                size=14,
                weight=ft.FontWeight.W_600,
                color=COLOR_ON_SURFACE,
            ),
            self._params_field,
        ]
        self._sequence_panel.controls = [
            self._sequence_hint,
            self._step_action_dd,
            self._step_value_field,
            self._step_title_field,
            ft.Row(
                spacing=10,
                controls=[self._add_step_btn, self._clear_steps_btn],
            ),
            self._steps_column,
        ]
        self._refresh_step_fields()

    def _show_message(self, error: str = "", info: str = "") -> None:
        self._error_text.value = error
        self._error_text.visible = bool(error)
        self._info_text.value = info
        self._info_text.visible = bool(info)
        try:
            self._error_text.update()
            self._info_text.update()
        except Exception:
            pass

    def _on_save_click(self, _e) -> None:
        self._show_message("", "")
        g = self._selected_gesture()
        if g is None:
            self._show_message(error="Выберите жест")
            return
        spec = self._build_action_spec()
        if isinstance(spec, str):
            self._show_message(error=spec)
            return
        name = (self._name_field.value or "").strip()
        bound_existing = (g.get("boundCommandName") or "").strip()
        # R8: уникальность имени — пропускаем проверку, если переписываем
        # привязку под существующим именем (overwrite candidate).
        if name and (not bound_existing or name != bound_existing):
            name_err = self._controller.validate_command_name(name)
            if name_err:
                self._show_message(error=name_err)
                return
        spec_err = self._controller.validate_action_spec_json(
            json.dumps(spec, ensure_ascii=False)
        )
        if spec_err:
            self._show_message(error=spec_err)
            return
        err = self._controller.save_binding(
            g["label"], name, json.dumps(spec, ensure_ascii=False)
        )
        if err:
            self._show_message(error=err)
            return
        self._show_message(info=f"Сохранено: «{g['label']}» → {name}")
        # Перечитать списки, чтобы статус «уже привязано» и команды обновились.
        self._refresh_all_lists()

    def _on_test_click(self, _e) -> None:
        """Запускает текущую команду (если она уже сохранена в БД) — удобно
        для отладки привязки сразу после её создания."""
        g = self._selected_gesture()
        if g is None:
            self._show_message(error="Выберите жест")
            return
        # Передаём заведомо высокую уверенность, чтобы обойти R4.
        ok = self._controller.execute_for_gesture(g["label"], confidence=1.0)
        if ok:
            self._show_message(info=f"Тест запущен: «{g['label']}»")
        else:
            self._show_message(
                error="Не удалось выполнить — нет привязки в БД или R5 cooldown"
            )

    # ---- Агент привязки ---------------------------------------------------

    def _set_agent_empty_state(self) -> None:
        self._last_agent_draft = None
        self._agent_dialog_messages = []
        self._agent_apply_btn.disabled = True
        self._agent_save_btn.disabled = True
        self._agent_status.value = "Ожидает ввод"
        self._agent_status.color = COLOR_MUTED
        self._agent_result.controls = [
            ft.Row(
                spacing=8,
                controls=[
                    ft.Icon(ft.Icons.LIGHTBULB_OUTLINE, color=COLOR_MUTED, size=18),
                    ft.Text(
                        "Напишите запрос или нажмите пример ниже",
                        color=COLOR_MUTED,
                        size=12,
                        expand=True,
                    ),
                ],
            )
        ]

    def _on_agent_parse_click(self, _e) -> None:
        prompt = (self._agent_input.value or "").strip()
        history = list(self._agent_dialog_messages)
        draft_state = dict(self._last_agent_draft or {})
        if prompt:
            self._agent_request_id += 1
            request_id = self._agent_request_id
            animate = self._page is not None
            self._start_agent_send_state(prompt, animate=animate)
            if animate:
                threading.Thread(
                    target=self._run_agent_request,
                    args=(request_id, prompt, history, draft_state),
                    daemon=True,
                ).start()
                return
        else:
            request_id = self._agent_request_id

        draft = build_agent_binding_draft(
            prompt,
            self._gestures,
            current_gesture=self._gesture_dd.value or "",
            conversation_history=history,
            draft_state=draft_state,
        )
        self._complete_agent_request(request_id, prompt, draft, animate=False)

    def _start_agent_send_state(self, prompt: str, *, animate: bool) -> None:
        self._agent_input.value = ""
        self._drop_agent_pending_messages()
        user_message = {"role": "user", "text": prompt}
        if animate:
            user_message["sending"] = "1"
        self._agent_dialog_messages.append(user_message)
        self._agent_dialog_messages.append(
            {
                "role": "agent",
                "text": "Агент печатает",
                "pending": "1",
                "typing": "1",
            }
        )
        self._agent_dialog_messages = self._agent_dialog_messages[-10:]
        self._agent_apply_btn.disabled = True
        self._agent_save_btn.disabled = True
        self._agent_status.value = "Отправляю..."
        self._agent_status.color = COLOR_ACCENT
        self._render_agent_dialog_only()
        try:
            self._agent_input.update()
        except Exception:
            pass
        if animate:
            request_id = self._agent_request_id
            threading.Thread(
                target=self._animate_agent_pending_state,
                args=(request_id,),
                daemon=True,
            ).start()

    def _run_agent_request(
        self,
        request_id: int,
        prompt: str,
        history: list[dict[str, str]],
        draft_state: dict[str, Any],
    ) -> None:
        try:
            draft = build_agent_binding_draft(
                prompt,
                self._gestures,
                current_gesture=self._gesture_dd.value or "",
                conversation_history=history,
                draft_state=draft_state,
            )
        except Exception as exc:
            draft = {
                "ok": False,
                "canApply": False,
                "error": f"Агент не смог обработать запрос: {exc}",
                "missing": [],
            }
        self._complete_agent_request(request_id, prompt, draft, animate=True)

    def _complete_agent_request(
        self,
        request_id: int,
        prompt: str,
        draft: dict[str, Any],
        *,
        animate: bool,
    ) -> None:
        if animate and request_id != self._agent_request_id:
            return
        if prompt:
            is_answer = str(draft.get("mode") or "") == "answer"
            self._mark_agent_user_messages_sent()
            if is_answer:
                if (
                    self._agent_dialog_messages
                    and self._agent_dialog_messages[-1].get("pending") == "1"
                ):
                    self._agent_dialog_messages.pop()
                self._agent_dialog_messages.append(
                    self._agent_response_message(self._agent_reply(draft))
                )
            else:
                if (
                    self._agent_dialog_messages
                    and self._agent_dialog_messages[-1].get("pending") == "1"
                ):
                    self._agent_dialog_messages[-1] = self._agent_response_message(
                        "" if animate else self._agent_reply(draft),
                        typing=animate,
                    )
                else:
                    self._agent_dialog_messages.append(
                        self._agent_response_message(
                            "" if animate else self._agent_reply(draft),
                            typing=animate,
                        )
                    )
            self._agent_dialog_messages = self._agent_dialog_messages[-10:]
        if animate:
            self._animate_agent_response(request_id, draft)
        else:
            self._set_agent_draft(draft)

    def _mark_agent_user_messages_sent(self) -> None:
        for message in self._agent_dialog_messages:
            if message.get("role") == "user":
                message.pop("sending", None)

    def _drop_agent_pending_messages(self) -> None:
        self._agent_dialog_messages = [
            message
            for message in self._agent_dialog_messages
            if message.get("pending") != "1"
        ]

    def _agent_response_message(self, text: str, *, typing: bool = False) -> dict[str, str]:
        message = {"role": "agent", "text": text}
        if typing:
            message["typing"] = "1"
        return message

    def _render_agent_dialog_only(self) -> None:
        self._agent_result.controls = [
            self._agent_message_bubble(message)
            for message in self._agent_dialog_messages
        ]
        self._safe_agent_update()

    def _animate_agent_pending_state(self, request_id: int) -> None:
        frames = (
            "Агент печатает",
            "Агент печатает.",
            "Агент печатает..",
            "Агент печатает...",
        )
        sent_marked = False
        index = 0
        while request_id == self._agent_request_id:
            pending = next(
                (
                    message
                    for message in reversed(self._agent_dialog_messages)
                    if message.get("pending") == "1"
                ),
                None,
            )
            if pending is None:
                return
            pending["text"] = frames[index % len(frames)]
            if not sent_marked and index >= 1:
                self._mark_agent_user_messages_sent()
                sent_marked = True
            self._agent_status.value = "Агент печатает..."
            self._agent_status.color = COLOR_ACCENT
            self._render_agent_dialog_only()
            index += 1
            time.sleep(0.28)

    def _typing_chunks(self, text: str) -> list[str]:
        if not text:
            return [""]
        chunk_size = max(12, min(28, len(text) // 18 or 12))
        chunks: list[str] = []
        cursor = 0
        while cursor < len(text):
            next_cursor = min(len(text), cursor + chunk_size)
            if next_cursor < len(text):
                space = text.rfind(" ", cursor + 1, next_cursor + 1)
                if space > cursor:
                    next_cursor = space + 1
            chunks.append(text[:next_cursor])
            cursor = next_cursor
        return chunks or [text]

    def _animate_agent_response(
        self,
        request_id: int,
        draft: dict[str, Any],
    ) -> None:
        if request_id != self._agent_request_id:
            return
        is_answer = str(draft.get("mode") or "") == "answer"
        final_reply = self._agent_reply(draft)
        self._agent_status.value = "Агент печатает..."
        self._agent_status.color = COLOR_ACCENT

        if is_answer:
            for partial in self._typing_chunks(final_reply):
                if request_id != self._agent_request_id:
                    return
                animated_draft = dict(draft)
                animated_draft["agentReply"] = partial or "_Агент печатает..._"
                self._set_agent_draft(animated_draft)
                time.sleep(0.025)
        else:
            for partial in self._typing_chunks(final_reply):
                if request_id != self._agent_request_id:
                    return
                if self._agent_dialog_messages:
                    self._agent_dialog_messages[-1] = {
                        "role": "agent",
                        "text": partial,
                        "typing": "1",
                    }
                self._render_agent_dialog_only()
                time.sleep(0.025)
            if self._agent_dialog_messages:
                self._agent_dialog_messages[-1] = {
                    "role": "agent",
                    "text": final_reply,
                }
        if request_id == self._agent_request_id:
            self._set_agent_draft(draft)

    def _agent_reply(self, draft: dict[str, Any]) -> str:
        reply = str(draft.get("agentReply") or "").strip()
        if reply:
            return reply
        error = str(draft.get("error") or "")
        if error:
            return f"Локальный агент: не смог разобрать запрос. {error}."
        if draft.get("missing"):
            if "действие" in list(draft.get("missing") or []):
                return (
                    "Я понял жест. Уточните, какую команду к нему привязать: "
                    "например `открыть Safari`, `нажать command+z` "
                    "или `показать уведомление`."
                )
            return (
                "Локальный агент: действие понял, но жест не указан. "
                "Выберите жест слева или нажмите плюс в поле ввода."
            )
        action_spec = dict(draft.get("actionSpec") or {})
        action = AGENT_ACTION_LABELS.get(
            str(action_spec.get("action") or ""),
            str(action_spec.get("action") or "команда"),
        )
        gesture = str(draft.get("gestureLabel") or "")
        return (
            f"Локальный агент: понял. Подготовил «{gesture}» → {action}. "
            "Можно заполнить форму или сохранить."
        )

    def _on_agent_add_selected_gesture_click(self, _e) -> None:
        gesture = (self._gesture_dd.value or "").strip()
        if not gesture:
            self._agent_status.value = "Сначала выберите жест"
            self._agent_status.color = COLOR_WARNING
            self._safe_agent_update()
            return
        current = (self._agent_input.value or "").strip()
        if gesture.lower() not in current.lower():
            self._agent_input.value = f"жест {gesture} {current}".strip()
        try:
            self._agent_input.update()
        except Exception:
            pass

    def _on_agent_apply_click(self, _e) -> None:
        draft = self._last_agent_draft
        if not draft or not draft.get("canApply"):
            self._set_agent_draft(
                {
                    "ok": False,
                    "canApply": False,
                    "error": "Сначала разберите задачу",
                    "missing": [],
                }
            )
            return
        self._apply_agent_draft(draft)
        self._show_message(info="Предложение агента перенесено в форму")
        self._agent_status.value = "Форма заполнена"
        self._agent_status.color = COLOR_SUCCESS
        self._safe_agent_update()

    def _on_agent_save_click(self, _e) -> None:
        draft = self._last_agent_draft
        if not draft or not draft.get("canApply"):
            self._set_agent_draft(
                {
                    "ok": False,
                    "canApply": False,
                    "error": "Сначала разберите задачу",
                    "missing": [],
                }
            )
            return
        if draft.get("missing"):
            self._set_agent_draft(draft)
            return
        self._apply_agent_draft(draft)
        self._on_save_click(_e)
        if not self._error_text.visible:
            self._agent_status.value = "Сохранено"
            self._agent_status.color = COLOR_SUCCESS
            self._safe_agent_update()

    def _set_agent_draft(self, draft: dict[str, Any]) -> None:
        self._last_agent_draft = draft
        mode = str(draft.get("mode") or "")
        is_answer = mode == "answer"
        can_apply = bool(draft.get("canApply"))
        missing = list(draft.get("missing") or [])
        error = str(draft.get("error") or "")

        self._agent_apply_btn.disabled = is_answer or not can_apply
        self._agent_save_btn.disabled = is_answer or (not can_apply) or bool(missing)
        if is_answer:
            self._agent_status.value = "Ответ"
            self._agent_status.color = COLOR_ACCENT
        elif error:
            self._agent_status.value = error
            self._agent_status.color = COLOR_DANGER
        elif missing:
            self._agent_status.value = "Нужно уточнение"
            self._agent_status.color = COLOR_WARNING
        else:
            self._agent_status.value = "Готово к привязке"
            self._agent_status.color = COLOR_SUCCESS

        if is_answer:
            controls: list[ft.Control] = []
            reply = str(draft.get("agentReply") or "").strip()
            if reply:
                controls.append(self._agent_answer_panel(reply))
        else:
            controls = [
                self._agent_message_bubble(message)
                for message in self._agent_dialog_messages
            ]
            if controls:
                controls.append(ft.Divider(height=1, color=COLOR_SURFACE_HIGH))

        if not is_answer and error:
            controls.append(
                self._agent_result_row(
                    ft.Icons.WARNING_AMBER,
                    error,
                    COLOR_DANGER,
                )
            )
        elif not is_answer:
            controls.extend(self._agent_structured_output(draft))
        self._agent_result.controls = controls or [
            self._agent_result_row(
                ft.Icons.INFO_OUTLINE,
                "Нет данных",
                COLOR_MUTED,
            )
        ]
        self._safe_agent_update()

    def _agent_trace_row(self, item: dict[str, Any]) -> ft.Control:
        status = str(item.get("status") or "")
        color = COLOR_SUCCESS
        icon = ft.Icons.CHECK_CIRCLE_OUTLINE
        if status in {"need_clarification", "need_input"}:
            color = COLOR_WARNING
            icon = ft.Icons.WARNING_AMBER
        elif status in {"blocked"}:
            color = COLOR_DANGER
            icon = ft.Icons.ERROR_OUTLINE
        elif status == "skipped":
            color = COLOR_MUTED
            icon = ft.Icons.RADIO_BUTTON_UNCHECKED
        return ft.Row(
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.START,
            controls=[
                ft.Icon(icon, color=color, size=16),
                ft.Column(
                    spacing=1,
                    expand=True,
                    controls=[
                        ft.Text(
                            str(item.get("agent") or "Agent"),
                            size=11,
                            color=COLOR_ON_SURFACE,
                            weight=ft.FontWeight.W_600,
                        ),
                        ft.Text(
                            str(item.get("message") or ""),
                            size=11,
                            color=COLOR_MUTED,
                        ),
                    ],
                ),
            ],
        )

    def _agent_action_description(self, spec: dict[str, Any]) -> str:
        action = str(spec.get("action") or "")
        if not action:
            return "Действие не распознано"
        if action == "open_app":
            return f"Открыть приложение {spec.get('app') or ''}".strip()
        if action == "open_url":
            return f"Открыть сайт {spec.get('url') or ''}".strip()
        if action == "open_path":
            return f"Открыть путь {spec.get('path') or ''}".strip()
        if action == "key_combination":
            return "Нажать " + " + ".join(str(k) for k in spec.get("keys", []))
        if action == "sequence":
            return f"Сценарий: {len(spec.get('steps') or [])} шагов"
        return AGENT_ACTION_LABELS.get(action, action)

    def _agent_output_row(
        self,
        label: str,
        value: str,
        icon: str,
        color: str = COLOR_MUTED,
    ) -> ft.Control:
        return ft.Row(
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.START,
            controls=[
                ft.Icon(icon, size=16, color=color),
                ft.Column(
                    spacing=1,
                    expand=True,
                    controls=[
                        ft.Text(label, size=10, color=COLOR_MUTED),
                        ft.Text(
                            value or "Не определено",
                            size=12,
                            color=COLOR_ON_SURFACE,
                        ),
                    ],
                ),
            ],
        )

    def _agent_markdown(self, value: str, *, selectable: bool = False) -> ft.Control:
        text_style = ft.TextStyle(size=12, color=COLOR_ON_SURFACE)
        muted_style = ft.TextStyle(size=12, color=COLOR_MUTED)
        return ft.Markdown(
            value=value or "",
            selectable=selectable,
            extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED,
            md_style_sheet=ft.MarkdownStyleSheet(
                p_text_style=text_style,
                strong_text_style=ft.TextStyle(
                    size=12,
                    color=COLOR_ON_SURFACE,
                    weight=ft.FontWeight.W_600,
                ),
                em_text_style=muted_style,
                code_text_style=ft.TextStyle(
                    size=12,
                    color=COLOR_ACCENT,
                    font_family="Menlo",
                ),
                list_bullet_text_style=text_style,
                table_head_text_style=text_style,
                table_body_text_style=text_style,
                blockquote_text_style=muted_style,
                block_spacing=4,
                list_indent=18,
            ),
            soft_line_break=True,
            shrink_wrap=True,
            fit_content=False,
            expand=False,
            width=float("inf"),
        )

    def _agent_answer_panel(self, value: str) -> ft.Control:
        return ft.Container(
            width=float("inf"),
            padding=12,
            border_radius=8,
            bgcolor="#17262A",
            border=ft.Border.all(1, "#21434A"),
            content=ft.Column(
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    ft.Row(
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Icon(
                                ft.Icons.CHAT_BUBBLE_OUTLINE,
                                size=16,
                                color=COLOR_ACCENT,
                            ),
                            ft.Text(
                                "Ответ агента",
                                size=12,
                                weight=ft.FontWeight.W_600,
                                color=COLOR_ON_SURFACE,
                            ),
                        ],
                    ),
                    self._agent_markdown(value, selectable=True),
                ],
            ),
        )

    def _agent_output_panel(
        self,
        title: str,
        icon: str,
        color: str,
        controls: list[ft.Control],
    ) -> ft.Control:
        return ft.Container(
            width=float("inf"),
            padding=12,
            border_radius=8,
            bgcolor="#17191D",
            border=ft.Border.all(1, COLOR_SURFACE_HIGH),
            content=ft.Column(
                spacing=10,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    ft.Row(
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Icon(icon, size=16, color=color),
                            ft.Text(
                                title,
                                size=12,
                                weight=ft.FontWeight.W_600,
                                color=COLOR_ON_SURFACE,
                            ),
                        ],
                    ),
                    *controls,
                ],
            ),
        )

    def _agent_structured_output(self, draft: dict[str, Any]) -> list[ft.Control]:
        spec = dict(draft.get("actionSpec") or {})
        missing = list(draft.get("missing") or [])
        gesture = str(draft.get("gestureLabel") or "").strip()
        command = str(draft.get("commandName") or "").strip()
        can_apply = bool(draft.get("canApply"))
        proposal_rows: list[ft.Control] = [
            self._agent_output_row(
                "Жест",
                gesture or "Нужно уточнить",
                ft.Icons.BACK_HAND,
                COLOR_ACCENT if gesture else COLOR_WARNING,
            )
        ]
        if spec or command:
            proposal_rows.extend(
                [
                    self._agent_output_row(
                        "Команда",
                        command or self._agent_action_description(spec),
                        ft.Icons.TERMINAL,
                        COLOR_SUCCESS if spec else COLOR_WARNING,
                    ),
                    self._agent_output_row(
                        "Действие",
                        self._agent_action_description(spec),
                        ft.Icons.ROUTE,
                        COLOR_MUTED,
                    ),
                ]
            )
        result: list[ft.Control] = [
            self._agent_output_panel(
                "Результат",
                ft.Icons.AUTO_AWESOME,
                COLOR_ACCENT,
                proposal_rows,
            )
        ]
        if missing:
            result.append(
                self._agent_output_panel(
                    "Нужно уточнить",
                    ft.Icons.WARNING_AMBER,
                    COLOR_WARNING,
                    [
                        self._agent_markdown(
                            "**Добавьте в запрос:** "
                            + ", ".join(str(x) for x in missing)
                            + "\n\nНапример: `жест sh3 привяжи к открытию Safari`."
                        ),
                    ],
                )
            )
        else:
            next_text = (
                "Можно заполнить форму или сразу сохранить привязку."
                if can_apply
                else "Уточните действие, и я подготовлю новую привязку."
            )
            result.append(
                self._agent_output_panel(
                    "Дальше",
                    ft.Icons.CHECK_CIRCLE_OUTLINE if can_apply else ft.Icons.INFO_OUTLINE,
                    COLOR_SUCCESS if can_apply else COLOR_MUTED,
                    [
                        self._agent_markdown(next_text),
                    ],
                )
            )
        return result

    def _agent_message_bubble(self, message: dict[str, str]) -> ft.Control:
        role = message.get("role") or "agent"
        is_user = role == "user"
        pending = (message.get("pending") or "") == "1"
        sending = is_user and (message.get("sending") or "") == "1"
        typing = (message.get("typing") or "") == "1"
        header_controls: list[ft.Control] = [
            ft.Text(
                "Вы" if is_user else "Агент",
                size=10,
                color=COLOR_MUTED,
            )
        ]
        if sending:
            header_controls.extend(
                [
                    ft.ProgressRing(
                        width=10,
                        height=10,
                        stroke_width=1.5,
                        color=COLOR_ACCENT,
                    ),
                    ft.Text("отправка", size=10, color=COLOR_MUTED),
                ]
            )
        elif typing and not is_user:
            header_controls.append(
                ft.Text("печатает", size=10, color=COLOR_ACCENT)
            )
        body_controls: list[ft.Control] = [
            ft.Row(
                spacing=5,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=header_controls,
            )
        ]
        if pending:
            body_controls.append(
                ft.Row(
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.ProgressRing(
                            width=14,
                            height=14,
                            stroke_width=2,
                            color=COLOR_ACCENT,
                        ),
                        ft.Container(
                            expand=True,
                            content=self._agent_markdown(
                                message.get("text") or "",
                                selectable=False,
                            ),
                        ),
                    ],
                )
            )
        else:
            body_controls.append(
                ft.Container(
                    animate_opacity=180,
                    content=self._agent_markdown(
                        message.get("text") or "",
                        selectable=False,
                    ),
                )
            )
        return ft.Row(
            alignment=(
                ft.MainAxisAlignment.END
                if is_user
                else ft.MainAxisAlignment.START
            ),
            controls=[
                ft.Container(
                    expand=True,
                    width=float("inf"),
                    margin=ft.Margin(80 if is_user else 0, 0, 0, 0),
                    padding=ft.Padding(12, 9, 12, 9),
                    border_radius=8,
                    bgcolor="#242426" if is_user else "#17262A",
                    border=ft.Border.all(
                        1,
                        "#333337" if is_user else "#21434A",
                    ),
                    animate_opacity=220,
                    animate_size=220,
                    content=ft.Column(
                        spacing=3,
                        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                        controls=body_controls,
                    ),
                )
            ],
        )

    def _agent_result_row(self, icon: str, text: str, color: str) -> ft.Control:
        return ft.Row(
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.START,
            controls=[
                ft.Icon(icon, color=color, size=18),
                ft.Text(text, size=12, color=color, expand=True),
            ],
        )

    def _safe_agent_update(self) -> None:
        for control in (
            self._agent_status,
            self._agent_result,
            self._agent_apply_btn,
            self._agent_save_btn,
        ):
            try:
                control.update()
            except Exception:
                pass

    def _category_id_for_action(self, action: str) -> str:
        for category in self._categories:
            category_id = str(category.get("id") or "")
            actions = self._controller.get_actions_for_category(category_id)
            if any(item.get("action") == action for item in actions):
                return category_id
        return ""

    def _apply_agent_draft(self, draft: dict[str, Any]) -> None:
        gesture = str(draft.get("gestureLabel") or "").strip()
        if gesture:
            self._gesture_dd.value = gesture

        spec = dict(draft.get("actionSpec") or {})
        action = str(spec.get("action") or "").strip()
        if action == "sequence":
            self._mode_dd.value = "sequence"
            self._sequence_steps = [
                dict(step)
                for step in spec.get("steps", [])
                if isinstance(step, dict)
            ]
            self._render_sequence_steps()
        else:
            self._mode_dd.value = "single"
            category_id = self._category_id_for_action(action)
            if category_id:
                self._category_dd.value = category_id
                self._refresh_actions()
            self._action_dd.value = action or None
            params = dict(spec)
            params.pop("action", None)
            params.pop("platform", None)
            self._params_field.value = json.dumps(params, ensure_ascii=False)
            selected = self._selected_action()
            hints = selected.get("fieldHints") if selected else {}
            self._field_hints.value = (
                "\n".join(f"• {k}: {v}" for k, v in hints.items())
                if hints
                else AGENT_ACTION_LABELS.get(action, "Предложение агента")
            )

        self._name_field.value = str(draft.get("commandName") or "").strip()
        self._refresh_mode_visibility()
        self._refresh_warn_two_hands()
        self._refresh_overwrite_hint()
        self._refresh_bindings_overview()
        for control in (
            self._gesture_dd,
            self._mode_dd,
            self._category_dd,
            self._action_dd,
            self._params_field,
            self._field_hints,
            self._name_field,
        ):
            try:
                control.update()
            except Exception:
                pass

    def _fill_agent_prompt(self, prompt: str) -> None:
        self._agent_input.value = prompt
        try:
            self._agent_input.update()
        except Exception:
            pass
        self._on_agent_parse_click(None)

    def _agent_example_chip(self, label: str, prompt: str) -> ft.Control:
        return ft.Container(
            padding=ft.Padding(10, 7, 10, 7),
            border_radius=8,
            bgcolor="#1B1D21",
            border=ft.Border.all(1, COLOR_SURFACE_HIGH),
            ink=True,
            on_click=lambda _e, value=prompt: self._fill_agent_prompt(value),
            content=ft.Row(
                spacing=6,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.AUTO_AWESOME, color=COLOR_ACCENT, size=14),
                    ft.Text(label, color=COLOR_MUTED, size=12),
                ],
            ),
        )

    def _build_agent_composer(self) -> ft.Control:
        return ft.Container(
            height=74,
            padding=ft.Padding(12, 8, 8, 8),
            bgcolor="#242426",
            border=ft.Border.all(1, "#313135"),
            border_radius=37,
            content=ft.Row(
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.IconButton(
                        icon=ft.Icons.ADD,
                        icon_color="#F4F4F3",
                        icon_size=28,
                        tooltip="Добавить выбранный жест",
                        size_constraints=ft.BoxConstraints(
                            min_width=42,
                            min_height=42,
                            max_width=42,
                            max_height=42,
                        ),
                        on_click=self._on_agent_add_selected_gesture_click,
                    ),
                    self._agent_input,
                    self._agent_parse_btn,
                ],
            ),
        )

    def _build_agent_panel(self) -> ft.Control:
        header = ft.Row(
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Container(
                    width=36,
                    height=36,
                    border_radius=8,
                    bgcolor="#18282C",
                    alignment=ft.Alignment.CENTER,
                    content=ft.Icon(
                        ft.Icons.PSYCHOLOGY,
                        color=COLOR_ACCENT,
                        size=21,
                    ),
                ),
                ft.Column(
                    spacing=1,
                    expand=True,
                    controls=[
                        ft.Text(
                            "Агент привязки",
                            size=17,
                            weight=ft.FontWeight.BOLD,
                            color=COLOR_ON_SURFACE,
                        ),
                        ft.Text(
                            "Команда из обычной фразы",
                            size=12,
                            color=COLOR_MUTED,
                        ),
                    ],
                ),
                ft.Container(
                    padding=ft.Padding(9, 5, 9, 5),
                    border_radius=8,
                    bgcolor="#1B1D21",
                    border=ft.Border.all(1, COLOR_SURFACE_HIGH),
                    content=ft.Text(
                        binding_agent_provider_label(),
                        size=11,
                        color=COLOR_MUTED,
                    ),
                ),
            ],
        )
        result_box = ft.Container(
            height=308,
            padding=14,
            bgcolor="#17191D",
            border=ft.Border.all(1, COLOR_SURFACE_HIGH),
            border_radius=8,
            content=ft.Column(
                spacing=10,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    ft.Row(
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Icon(
                                ft.Icons.RADIO_BUTTON_UNCHECKED,
                                color=COLOR_ACCENT,
                                size=14,
                            ),
                            ft.Text(
                                "Диалог агента",
                                color=COLOR_ON_SURFACE,
                                size=13,
                                weight=ft.FontWeight.W_600,
                            ),
                            ft.Container(expand=True),
                            self._agent_status,
                        ],
                    ),
                    ft.Divider(height=1, color=COLOR_SURFACE_HIGH),
                    ft.Container(
                        expand=True,
                        width=float("inf"),
                        content=ft.Column(
                            spacing=8,
                            scroll=ft.ScrollMode.AUTO,
                            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                            controls=[self._agent_result],
                        ),
                    ),
                ],
            ),
        )
        return ft.Column(
            spacing=14,
            controls=[
                header,
                result_box,
                ft.Row(
                    spacing=8,
                    wrap=True,
                    controls=[
                        self._agent_apply_btn,
                        self._agent_save_btn,
                    ],
                ),
                ft.Row(
                    spacing=8,
                    wrap=True,
                    controls=[
                        self._agent_example_chip(
                            "Safari",
                            "жест palm открывает Safari",
                        ),
                        self._agent_example_chip(
                            "Cmd+Z",
                            "сохрани ctrlz как command+z",
                        ),
                        self._agent_example_chip(
                            "Сценарий",
                            (
                                "жест swipe_down сценарий: открыть Preview; "
                                "потом подожди 1 секунду; "
                                "затем покажи уведомление Готово"
                            ),
                        ),
                    ],
                ),
                self._build_agent_composer(),
            ],
        )

    # ---- Сборка дерева ---------------------------------------------------

    def build(self) -> ft.Control:
        self._mode_dd.width = 300

        header = surface_card(
            ft.Row(
                spacing=14,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Container(
                        width=44,
                        height=44,
                        border_radius=8,
                        bgcolor="#18282C",
                        alignment=ft.Alignment.CENTER,
                        content=ft.Icon(
                            ft.Icons.LINK,
                            color=COLOR_ACCENT,
                            size=25,
                        ),
                    ),
                    ft.Column(
                        spacing=2,
                        expand=True,
                        controls=[
                            ft.Text(
                                "Привязки",
                                size=22,
                                weight=ft.FontWeight.BOLD,
                                color=COLOR_ON_SURFACE,
                            ),
                            ft.Text(
                                "Жесты, команды и сценарии",
                                color=COLOR_MUTED,
                                size=12,
                            ),
                        ],
                    ),
                    ft.Container(
                        padding=ft.Padding(10, 6, 10, 6),
                        border_radius=8,
                        bgcolor="#1B1D21",
                        border=ft.Border.all(1, COLOR_SURFACE_HIGH),
                        content=ft.Row(
                            spacing=6,
                            tight=True,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[
                                ft.Icon(
                                    ft.Icons.HUB,
                                    color=COLOR_ACCENT,
                                    size=15,
                                ),
                                ft.Text(
                                    "R1 / R6 / R7",
                                    color=COLOR_MUTED,
                                    size=12,
                                ),
                            ],
                        ),
                    ),
                    self._refresh_btn,
                ],
            ),
            padding=16,
            radius=8,
        )

        form_card = surface_card(
            ft.Column(
                spacing=14,
                controls=[
                    ft.Row(
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Container(
                                width=36,
                                height=36,
                                border_radius=8,
                                bgcolor="#251F15",
                                alignment=ft.Alignment.CENTER,
                                content=ft.Icon(
                                    ft.Icons.ROUTE,
                                    color=COLOR_WARNING,
                                    size=21,
                                ),
                            ),
                            ft.Column(
                                spacing=1,
                                expand=True,
                                controls=[
                                    ft.Text(
                                        "Конструктор",
                                        size=17,
                                        weight=ft.FontWeight.BOLD,
                                        color=COLOR_ON_SURFACE,
                                    ),
                                    ft.Text(
                                        "Ручная настройка привязки",
                                        size=12,
                                        color=COLOR_MUTED,
                                    ),
                                ],
                            ),
                            self._mode_dd,
                        ],
                    ),
                    ft.ResponsiveRow(
                        spacing=12,
                        run_spacing=12,
                        controls=[
                            ft.Container(
                                col={"xs": 12, "md": 6},
                                content=self._gesture_dd,
                            ),
                            ft.Container(
                                col={"xs": 12, "md": 6},
                                content=self._name_field,
                            ),
                        ],
                    ),
                    ft.Divider(color=COLOR_SURFACE_HIGH, thickness=1),
                    self._single_mode_panel,
                    self._sequence_panel,
                    self._warn_two_hands,
                    self._overwrite_hint,
                    ft.Row(
                        spacing=10,
                        wrap=True,
                        controls=[self._save_btn, self._test_btn],
                    ),
                    self._error_text,
                    self._info_text,
                ],
            ),
            padding=20,
            radius=8,
        )

        agent_card = surface_card(
            self._build_agent_panel(),
            padding=16,
            radius=8,
        )

        return ft.Column(
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            controls=[
                header,
                self._build_bindings_overview(),
                ft.ResponsiveRow(
                    spacing=14,
                    run_spacing=14,
                    controls=[
                        ft.Container(
                            col={"xs": 12, "lg": 7},
                            content=form_card,
                        ),
                        ft.Container(
                            col={"xs": 12, "lg": 5},
                            content=agent_card,
                        ),
                    ],
                ),
            ],
        )
