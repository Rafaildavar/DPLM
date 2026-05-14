# -*- coding: utf-8 -*-
"""
Пользовательские команды из БД без правки Python.

Архитектура
-----------
1. **Хранение** — поле ``Command.action_spec`` (TEXT с JSON). Интерфейс сохраняет туда описание
   действия; поля ``name``, ``platform``, ``gesture_id``, ``is_active`` задаются как сейчас.
2. **Выполнение** — ``gesture_command_bridge.execute_command_row`` сначала читает ``action_spec``
   и вызывает ``CommandExecutor.execute_config``. Иначе — ``script_path``, иначе — команда по имени
   в реестре.
3. **Синхронизация реестра** — ``sync_db_commands_to_executor`` регистрирует все активные команды
   с непустым ``action_spec`` в ``CommandExecutor``, чтобы ``executeCommand("Имя")`` и голосовой
   помощник работали по имени без отдельного запроса к строке жеста.

Рекомендации для UI
-------------------
- Форма «команда»: название (уникальное), платформа, чекбокс активна, привязка к жесту.
- Редактор действия: выпадающий список ``action`` + поля параметров (генерировать JSON по схеме).
- После сохранения в БД вызвать ``sync_db_commands_to_executor(session, get_executor())``.

Безопасность: не давать пользователю произвольный ``shell:`` через ``action_spec``; для shell
оставьте отдельный админ-режим или только ``script_path`` с предупреждением.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.models.database import Command

logger = logging.getLogger(__name__)

# Категории команд для группировки в UI (см. docs/BINDING_RULES.md, раздел 1).
CATEGORY_LAUNCH = "launch"
CATEGORY_SYSTEM = "system"
CATEGORY_NAVIGATION = "navigation"
CATEGORY_MEDIA = "media"
CATEGORY_HOTKEY = "hotkey"
CATEGORY_SCRIPT = "script"

CATEGORY_LABELS: Dict[str, str] = {
    CATEGORY_LAUNCH: "Запуск приложения",
    CATEGORY_SYSTEM: "Системное действие",
    CATEGORY_NAVIGATION: "Прокрутка / навигация",
    CATEGORY_MEDIA: "Мультимедиа",
    CATEGORY_HOTKEY: "Горячая клавиша",
    CATEGORY_SCRIPT: "Пользовательский скрипт",
}

# Соответствие action → категория (см. docs/BINDING_RULES.md, раздел 4).
ACTION_TO_CATEGORY: Dict[str, str] = {
    "open_app": CATEGORY_LAUNCH,
    "open_url": CATEGORY_LAUNCH,
    "volume_up": CATEGORY_SYSTEM,
    "volume_down": CATEGORY_SYSTEM,
    "mute_toggle": CATEGORY_SYSTEM,
    "brightness_up": CATEGORY_SYSTEM,
    "brightness_down": CATEGORY_SYSTEM,
    "lock_screen": CATEGORY_SYSTEM,
    "screenshot": CATEGORY_SYSTEM,
    "scroll": CATEGORY_NAVIGATION,
    "press": CATEGORY_NAVIGATION,
    "media_key": CATEGORY_MEDIA,
    "key_combination": CATEGORY_HOTKEY,
    "run_script": CATEGORY_SCRIPT,
}


def category_for_action(action: str) -> str:
    """Вернуть категорию команды для UI по полю ``action``. Пустая строка — неизвестно."""
    return ACTION_TO_CATEGORY.get((action or "").strip().lower(), "")


# Описание полей для генерации форм / подсказок в UI (не исполняется напрямую).
ACTION_SPEC_SCHEMA: Dict[str, Any] = {
    # --- Категория 1: запуск приложений ---
    "open_app": {
        "category": CATEGORY_LAUNCH,
        "fields": {"app": "str — имя приложения для macOS ``open -a`` (например Safari)"},
        "example": {"action": "open_app", "app": "Music", "platform": "macos"},
    },
    "open_url": {
        "category": CATEGORY_LAUNCH,
        "fields": {"url": "str — https://..."},
        "example": {"action": "open_url", "url": "https://example.com", "platform": "macos"},
    },
    # --- Категория 2: системные действия ---
    "volume_up": {
        "category": CATEGORY_SYSTEM,
        "example": {"action": "volume_up", "platform": "macos"},
    },
    "volume_down": {
        "category": CATEGORY_SYSTEM,
        "example": {"action": "volume_down", "platform": "macos"},
    },
    "mute_toggle": {
        "category": CATEGORY_SYSTEM,
        "example": {"action": "mute_toggle", "platform": "macos"},
    },
    "brightness_up": {
        "category": CATEGORY_SYSTEM,
        "example": {"action": "brightness_up", "platform": "macos"},
    },
    "brightness_down": {
        "category": CATEGORY_SYSTEM,
        "example": {"action": "brightness_down", "platform": "macos"},
    },
    "lock_screen": {
        "category": CATEGORY_SYSTEM,
        "example": {"action": "lock_screen", "platform": "macos"},
    },
    "screenshot": {
        "category": CATEGORY_SYSTEM,
        "example": {"action": "screenshot", "platform": "macos"},
    },
    # --- Категория 3: прокрутка / навигация ---
    "scroll": {
        "category": CATEGORY_NAVIGATION,
        "fields": {"clicks": "int — отрицательное вниз, положительное вверх (PyAutoGUI)"},
        "example": {"action": "scroll", "clicks": -5, "platform": "macos"},
    },
    "press": {
        "category": CATEGORY_NAVIGATION,
        "fields": {"key": "str — одна клавиша PyAutoGUI (pagedown, down, ...)"},
        "example": {"action": "press", "key": "pagedown", "platform": "macos"},
    },
    # --- Категория 4: мультимедиа ---
    "media_key": {
        "category": CATEGORY_MEDIA,
        "fields": {"kind": "str — play_pause | next | prev"},
        "example": {"action": "media_key", "kind": "play_pause", "platform": "macos"},
    },
    # --- Категория 5: горячие клавиши ---
    "key_combination": {
        "category": CATEGORY_HOTKEY,
        "fields": {"keys": "list[str] — последовательность для hotkey"},
        "example": {"action": "key_combination", "keys": ["command", "space"], "platform": "macos"},
    },
    # --- Категория 6: пользовательский скрипт ---
    "run_script": {
        "category": CATEGORY_SCRIPT,
        "fields": {
            "script_path": "str — абсолютный путь к существующему .py",
            "args": "list[str] — опционально",
        },
        "example": {"action": "run_script", "script_path": "/path/to/tool.py", "platform": "macos"},
    },
}


# Действия, для которых R6 рекомендует использовать двуручный жест.
DANGEROUS_ACTIONS: frozenset = frozenset({"lock_screen", "run_script"})


def is_dangerous_action(action: str) -> bool:
    """Проверка R6: «опасное» действие, для которого UI показывает предупреждение."""
    return (action or "").strip().lower() in DANGEROUS_ACTIONS


def validate_action_spec(spec: Dict[str, Any]) -> Optional[str]:
    """
    Валидация JSON-описания действия (см. R7 + структурные проверки).

    Returns:
        ``None`` если ок, иначе сообщение об ошибке (готовое для UI).
    """
    if not isinstance(spec, dict):
        return "action_spec должен быть объектом (dict)"

    action = (spec.get("action") or "").strip()
    if not action:
        return "Не задано поле «action»"
    if action not in ACTION_SPEC_SCHEMA:
        return f"Неизвестное действие «{action}»"

    if action == "run_script":
        path = (spec.get("script_path") or "").strip()
        if not path:
            return "run_script: укажите абсолютный путь к .py-файлу"
        if not path.startswith("/"):
            return "run_script: путь должен быть абсолютным (начинаться с «/»)"
        if not path.lower().endswith(".py"):
            return "run_script: поддерживаются только файлы с расширением .py"
        try:
            from pathlib import Path

            p = Path(path)
            if not p.is_file():
                return f"run_script: файл не найден: {path}"
            if p.stat().st_size == 0:
                return "run_script: файл пустой"
        except OSError as e:
            return f"run_script: не удалось прочитать файл: {e}"

    if action == "scroll":
        clicks = spec.get("clicks")
        if clicks is None:
            return "scroll: укажите поле «clicks» (целое число)"
        try:
            int(clicks)
        except (TypeError, ValueError):
            return "scroll: поле «clicks» должно быть целым числом"

    if action == "press":
        if not (spec.get("key") or "").strip():
            return "press: укажите поле «key» (имя клавиши PyAutoGUI)"

    if action == "key_combination":
        keys = spec.get("keys")
        if not isinstance(keys, list) or not keys:
            return "key_combination: поле «keys» должно быть непустым списком строк"
        if any(not isinstance(k, str) or not k.strip() for k in keys):
            return "key_combination: все элементы «keys» должны быть непустыми строками"

    if action == "media_key":
        kind = (spec.get("kind") or "").strip().lower()
        if kind not in {"play_pause", "play", "pause", "next", "prev", "previous"}:
            return "media_key: «kind» должен быть play_pause | next | prev"

    if action == "open_app":
        if not (spec.get("app") or "").strip():
            return "open_app: укажите имя приложения в поле «app»"

    if action == "open_url":
        url = (spec.get("url") or "").strip()
        if not url:
            return "open_url: укажите URL"
        if not (url.startswith("http://") or url.startswith("https://")):
            return "open_url: URL должен начинаться с http:// или https://"

    return None


def parse_command_action_spec(command: Command) -> Optional[Dict[str, Any]]:
    """Разобрать ``Command.action_spec`` в dict или ``None`` при ошибке/пустоте."""
    raw = (getattr(command, "action_spec", None) or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("action_spec JSON: %s", e)
        return None
    if not isinstance(data, dict):
        return None
    return data


def executor_config_from_row(command: Command, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Слить параметры строки БД с JSON (платформа из строки, если в JSON не задана явно)."""
    cfg: Dict[str, Any] = dict(spec)
    plat = str(cfg.get("platform") or "").strip()
    if not plat:
        cfg["platform"] = command.platform or "all"
    return cfg


def sync_db_commands_to_executor(session: Session, executor: Any) -> int:
    """
    Зарегистрировать в ``CommandExecutor`` все активные команды с валидным ``action_spec``.

    Вызывать после старта приложения и после каждого сохранения команд в UI.
    """
    from app.services.command_executor import CommandExecutor

    if not isinstance(executor, CommandExecutor):
        raise TypeError("executor должен быть CommandExecutor")

    n = 0
    for cmd in session.query(Command).filter(Command.is_active.is_(True)).all():
        spec = parse_command_action_spec(cmd)
        if not spec or not spec.get("action"):
            continue
        cfg = executor_config_from_row(cmd, spec)
        executor.register_command(cmd.name, cfg)
        n += 1
    return n
