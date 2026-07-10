# -*- coding: utf-8 -*-
"""
Связка «метка жеста (KNN / классификатор) → запись Command в БД → действие на ноутбуке».

Использование
-------------
1. В таблице ``gestures`` задаётся ``label`` (то же имя класса, что приходит из ``gestureDetected``).
2. В ``commands`` создаётся строка с ``gesture_id`` на этот жест.
3. Выполнение:
   - если ``script_path`` пустой — вызывается зарегистрированное имя в ``CommandExecutor`` по полю ``name``;
   - если ``script_path`` задан — см. раздел «Формат script_path» ниже.

Библиотеки (что уже в проекте)
-----------------------------
- **SQLAlchemy** — чтение связки жест↔команда из БД (``app.models.database``).
- **subprocess** (stdlib) — запуск программ, ``open`` на macOS.
- **pyautogui** (опционально) — горячие клавиши и медиа-клавиши; подключается внутри ``CommandExecutor``.

Поле action_spec (команды из интерфейса без кода)
-------------------------------------------------
JSON в ``Command.action_spec`` — приоритетнее ``script_path``. Схема и примеры:
``app.services.user_command_sync.ACTION_SPEC_SCHEMA``. После изменений в БД вызывайте
``sync_db_commands_to_executor``, чтобы команда была доступна по имени (голос, слоты Qt).

Формат script_path (произвольные действия)
------------------------------------------
- Пусто — только логическое имя ``Command.name`` (как в реестре ``CommandExecutor``), регистр не важен.
- Существующий файл ``.py`` — запуск интерпретатором текущего Python.
- Иной существующий исполняемый файл — ``Popen([path], ...)``.
- Строка начинается с ``http://`` или ``https://`` — на macOS: ``open <url>``.
- Префикс ``open:`` — на macOS остаток строки передаётся в ``open`` (пример: ``open:-a Safari`` → ``open -a Safari``).
- Префикс ``shell:`` — одна команда оболочки (**осторожно**: только доверенные строки).

Платформа
---------
Поле ``Command.platform``: ``all`` | ``macos`` | ``windows`` (и варианты вроде ``darwin`` нормализуются к ``macos``).
"""

from __future__ import annotations

import logging
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

from sqlalchemy.orm import Session

from app.models.database import Command, Gesture, GestureHistory
from app.services.binding_settings import BindingSettings, load_binding_settings
from app.services.gesture_labels import resolve_registered_gesture_label
from app.services.user_command_sync import (
    DANGEROUS_ACTIONS,
    executor_config_from_row,
    parse_command_action_spec,
)

if TYPE_CHECKING:
    from app.services.command_executor import CommandExecutor

logger = logging.getLogger(__name__)


# Причины отказа при проверке политики (стабильные коды для UI и логов).
REJECT_LOW_CONFIDENCE = "low_confidence"
REJECT_COOLDOWN = "cooldown"


@dataclass
class PolicyDecision:
    """Решение ``BindingPolicy.evaluate`` — стабильный объект для логов и UI."""

    allow: bool
    reason: str = ""
    confidence: float = 0.0
    threshold: float = 0.0
    cooldown_remaining_ms: int = 0


@dataclass
class BindingPolicy:
    """
    Реализация правил R4 (порог уверенности) и R5 (cooldown) перед запуском
    команды по жесту.

    Хранит последний момент срабатывания на каждую метку жеста в памяти.
    Потокобезопасно: `evaluate` берёт внутренний `Lock`.

    Использование:
        policy = BindingPolicy.from_session(session)
        decision = policy.evaluate("swipe_right", confidence=0.71)
        if decision.allow:
            execute_for_gesture_label(...)

    Настройки берутся из таблицы ``settings`` (см. ``binding_settings.py``).
    """

    settings: BindingSettings = field(default_factory=BindingSettings)
    _last_fired_ms: Dict[str, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    @classmethod
    def from_session(cls, session: Session) -> "BindingPolicy":
        return cls(settings=load_binding_settings(session))

    def reload(self, session: Session) -> None:
        """Перечитать настройки из БД (например, после изменения в UI настроек)."""
        with self._lock:
            self.settings = load_binding_settings(session)

    def reset_cooldowns(self) -> None:
        """Сбросить in-memory таймеры (для тестов и при перезапуске сессии)."""
        with self._lock:
            self._last_fired_ms.clear()

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)

    def evaluate(self, gesture_label: str, confidence: float) -> PolicyDecision:
        """
        Решить, можно ли исполнять команду для данного жеста сейчас.

        Не записывает срабатывание — только проверяет. Чтобы обновить таймер
        cooldown'а, вызовите :py:meth:`mark_fired` после успешного запуска.
        """
        label = (gesture_label or "").strip()
        threshold = float(self.settings.confidence_threshold)
        conf = float(confidence) if confidence is not None else 0.0

        if conf < threshold:
            return PolicyDecision(
                allow=False,
                reason=REJECT_LOW_CONFIDENCE,
                confidence=conf,
                threshold=threshold,
            )

        cooldown = int(self.settings.cooldown_ms)
        if cooldown <= 0 or not label:
            return PolicyDecision(
                allow=True,
                confidence=conf,
                threshold=threshold,
            )

        with self._lock:
            last = self._last_fired_ms.get(label)
        if last is None:
            return PolicyDecision(
                allow=True,
                confidence=conf,
                threshold=threshold,
            )

        elapsed = self._now_ms() - last
        if elapsed < cooldown:
            return PolicyDecision(
                allow=False,
                reason=REJECT_COOLDOWN,
                confidence=conf,
                threshold=threshold,
                cooldown_remaining_ms=cooldown - elapsed,
            )

        return PolicyDecision(
            allow=True,
            confidence=conf,
            threshold=threshold,
        )

    def mark_fired(self, gesture_label: str) -> None:
        """Зафиксировать момент успешного срабатывания жеста (для R5)."""
        label = (gesture_label or "").strip()
        if not label:
            return
        with self._lock:
            self._last_fired_ms[label] = self._now_ms()


def save_gesture_binding(
    session: Session,
    gesture_label: str,
    command_name: str,
    action_spec: Dict,
    *,
    platform: str = "macos",
    commit: bool = True,
) -> Tuple[Command, bool]:
    """
    Сохранить пару «жест → команда» в БД с применением R1 + R8.

    Семантика (см. ``docs/BINDING_RULES.md``):
      * R1 — на жесте может висеть только одна команда. Если уже есть
        и это **другая** команда — её ``gesture_id`` обнуляется (команда
        не удаляется, чтобы можно было быстро переназначить обратно).
      * R8 — имя ``commands.name`` уникально. Если команда с таким именем
        уже существует, обновляется именно она (включая ``action_spec`` и
        привязку к жесту), а не создаётся дубликат.

    Returns:
        Кортеж ``(command, created)``: SQLAlchemy-объект записи и флаг,
        был ли он создан этим вызовом.

    Raises:
        ValueError если жест с такой меткой не найден в БД
        или жест неактивен.
    """
    import json as _json

    label = (gesture_label or "").strip()
    name = (command_name or "").strip()
    if not label:
        raise ValueError("gesture_label обязателен")
    if not name:
        raise ValueError("command_name обязателен")

    gesture = session.query(Gesture).filter(Gesture.label == label).first()
    if gesture is None:
        raise ValueError(f"Жест «{label}» не найден в БД")
    if not bool(gesture.is_active):
        raise ValueError(
            f"Жест «{label}» неактивен и недоступен для привязки"
        )

    existing_for_gesture = (
        session.query(Command).filter(Command.gesture_id == gesture.id).first()
    )
    existing_by_name = session.query(Command).filter(Command.name == name).first()

    if (
        existing_for_gesture is not None
        and existing_for_gesture is not existing_by_name
    ):
        existing_for_gesture.gesture_id = None

    created = False
    if existing_by_name is not None:
        target = existing_by_name
        if not target.platform:
            target.platform = platform
        target.is_active = True
    else:
        target = Command(name=name, platform=platform, is_active=True)
        session.add(target)
        created = True

    target.action_spec = _json.dumps(action_spec, ensure_ascii=False)
    target.gesture_id = gesture.id

    if commit:
        session.commit()
    else:
        session.flush()

    return target, created


def is_dangerous_command(command: Command) -> bool:
    """
    R6: проверить, является ли команда «опасной» (lock_screen / run_script
    / явный shell в script_path).
    """
    spec = parse_command_action_spec(command)
    if spec:
        if _action_spec_is_dangerous(spec):
            return True
    sp = (command.script_path or "").strip().lower()
    if sp.startswith("shell:"):
        return True
    if sp.endswith(".py") and sp.startswith("/"):
        return True
    return False


def _action_spec_is_dangerous(spec: dict) -> bool:
    action = (spec.get("action") or "").strip().lower()
    if action in DANGEROUS_ACTIONS:
        return True
    if action == "sequence":
        steps = spec.get("steps") or []
        if isinstance(steps, list):
            return any(
                isinstance(step, dict) and _action_spec_is_dangerous(step)
                for step in steps
            )
    return False


def _normalize_platform(p: str) -> str:
    s = (p or "all").strip().lower()
    if s in ("darwin", "mac", "macos"):
        return "macos"
    if s in ("win32", "windows"):
        return "windows"
    return s


def _host_platform() -> str:
    s = platform.system().lower()
    if s == "darwin":
        return "macos"
    if s == "windows":
        return "windows"
    return s


def _platform_allowed(cmd_platform: str) -> bool:
    req = _normalize_platform(cmd_platform)
    if req in ("", "all"):
        return True
    return req == _host_platform()


def resolve_command_for_gesture(session: Session, gesture_label: str) -> Optional[Tuple[Gesture, Command]]:
    """
    Найти пару (жест, команда) по метке класса жеста.

    Связь: ``Command.gesture_id`` → ``Gesture.id`` (одна команда на жест).
    """
    label = (gesture_label or "").strip()
    if not label:
        return None
    gesture = session.query(Gesture).filter(Gesture.label == label).first()
    if gesture is None:
        gestures = session.query(Gesture).all()
        registered_label = resolve_registered_gesture_label(
            label,
            (item.label for item in gestures),
        )
        if registered_label is None:
            return None
        gesture = next(
            (item for item in gestures if item.label == registered_label),
            None,
        )
        if gesture is None:
            return None
    command = session.query(Command).filter(Command.gesture_id == gesture.id).first()
    if command is None:
        return None
    return gesture, command


def _run_script_path(script_path: str) -> bool:
    raw = (script_path or "").strip()
    if not raw:
        return False

    if raw.lower().startswith(("http://", "https://")):
        if _host_platform() == "macos":
            subprocess.Popen(["open", raw], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        if _host_platform() == "windows":
            subprocess.Popen(["cmd", "/c", "start", "", raw], shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        logger.warning("open URL: неподдерживаемая ОС")
        return False

    if raw.lower().startswith("open:"):
        rest = raw[5:].strip()
        if _host_platform() != "macos":
            logger.warning("префикс open: только для macOS")
            return False
        try:
            parts = shlex.split(rest)
        except ValueError as e:
            logger.error("open: не удалось разобрать аргументы: %s", e)
            return False
        subprocess.Popen(["open", *parts], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

    if raw.lower().startswith("shell:"):
        line = raw[6:].lstrip()
        subprocess.Popen(line, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

    path = Path(raw).expanduser()
    if path.is_file():
        if path.suffix.lower() == ".py":
            subprocess.Popen([sys.executable, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        try:
            subprocess.Popen([str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            logger.exception("не удалось запустить файл: %s", path)
            return False

    logger.warning("script_path не распознан и файл не найден: %s", raw)
    return False


def execute_command_row(
    command: Command,
    executor: Optional["CommandExecutor"] = None,
) -> bool:
    """
    Выполнить одну строку ``Command``: ``action_spec`` (JSON) → ``script_path`` → имя в ``executor``.
    """
    if not command.is_active:
        logger.info("команда неактивна: %s", command.name)
        return False
    if not _platform_allowed(command.platform):
        logger.warning("команда '%s' не для этой платформы (%s)", command.name, command.platform)
        return False

    if executor is None:
        from app.services.command_executor import get_executor

        executor = get_executor()

    spec = parse_command_action_spec(command)
    if spec and spec.get("action"):
        cfg = executor_config_from_row(command, spec)
        return bool(executor.execute_config(cfg))

    sp = (command.script_path or "").strip()
    if sp:
        ok = _run_script_path(sp)
        if ok:
            return True
        logger.info("fallback на CommandExecutor после неудачного script_path")

    return bool(executor.execute(command.name))


def execute_for_gesture_label(
    session: Session,
    gesture_label: str,
    executor: Optional["CommandExecutor"] = None,
    record_history: bool = False,
    *,
    confidence: Optional[float] = None,
    policy: Optional[BindingPolicy] = None,
) -> Tuple[bool, str]:
    """
    Полный цикл: метка жеста → БД → действие на ноутбуке.

    Если передан ``policy`` и ``confidence``, перед запуском команды
    применяются правила R4 (порог уверенности) и R5 (cooldown); при отказе
    функция возвращает ``(False, причина)``.

    Returns:
        (успех, человекочитаемое описание: имя команды или код причины отказа)
    """
    if policy is not None and confidence is not None:
        decision = policy.evaluate(gesture_label, confidence)
        if not decision.allow:
            logger.info(
                "policy reject: label=%r reason=%s conf=%.3f thr=%.3f cd=%d",
                gesture_label,
                decision.reason,
                decision.confidence,
                decision.threshold,
                decision.cooldown_remaining_ms,
            )
            return False, decision.reason

    resolved = resolve_command_for_gesture(session, gesture_label)
    if resolved is None:
        msg = "нет привязки жест→команда в БД"
        logger.info("%s: %r", msg, gesture_label)
        return False, msg

    gesture, command = resolved
    ok = execute_command_row(command, executor=executor)
    if ok and policy is not None:
        policy.mark_fired(gesture_label)
    if record_history and ok:
        try:
            session.add(
                GestureHistory(
                    gesture_id=gesture.id,
                    command_id=command.id,
                )
            )
            session.commit()
        except Exception:
            logger.exception("не удалось записать gesture_history")
            session.rollback()

    if ok:
        return True, command.name
    return False, command.name


class GestureCommandBridge:
    """
    Объектная обёртка над функциями модуля (удобно пробрасывать в ``AppController``).

    Содержит in-memory ``BindingPolicy``, реализующую R4/R5. Настройки
    автоматически подгружаются из БД при первом использовании; их можно
    обновить вызовом :py:meth:`reload_policy` после изменения настроек в UI.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        executor: Optional["CommandExecutor"] = None,
        policy: Optional[BindingPolicy] = None,
    ):
        self._session_factory = session_factory
        self._executor = executor
        self._policy = policy
        self._policy_lock = Lock()

    def _ensure_policy(self) -> BindingPolicy:
        with self._policy_lock:
            if self._policy is None:
                session = self._session_factory()
                try:
                    self._policy = BindingPolicy.from_session(session)
                finally:
                    session.close()
            return self._policy

    @property
    def policy(self) -> BindingPolicy:
        """Текущая ``BindingPolicy`` (создаётся при первом обращении)."""
        return self._ensure_policy()

    def reload_policy(self) -> BindingSettings:
        """Перечитать настройки R4/R5/R6 из БД (например, после правок в UI)."""
        policy = self._ensure_policy()
        session = self._session_factory()
        try:
            policy.reload(session)
        finally:
            session.close()
        return policy.settings

    def resolve(self, gesture_label: str) -> Optional[Tuple[Gesture, Command]]:
        session = self._session_factory()
        try:
            r = resolve_command_for_gesture(session, gesture_label)
            if r is None:
                return None
            g, c = r
            session.expunge(g)
            session.expunge(c)
            return g, c
        finally:
            session.close()

    def command_name_for_gesture(self, gesture_label: str) -> str:
        """Имя команды для UI; пустая строка если привязки нет."""
        r = self.resolve(gesture_label)
        if r is None:
            return ""
        return r[1].name

    def execute(
        self,
        gesture_label: str,
        *,
        record_history: bool = False,
        confidence: Optional[float] = None,
        apply_policy: bool = True,
    ) -> Tuple[bool, str]:
        """
        Запустить команду по метке жеста.

        Args:
            gesture_label: метка от классификатора (имя класса).
            record_history: писать ли в ``gesture_history`` при успехе.
            confidence: уверенность 0..1; нужна для R4 — если не задана,
                R4 не применяется.
            apply_policy: ``False`` для обхода R4/R5 (например, ручной запуск
                из «Доступных команд» в UI).
        """
        policy: Optional[BindingPolicy] = None
        conf: Optional[float] = None
        if apply_policy and confidence is not None:
            policy = self._ensure_policy()
            conf = float(confidence)

        session = self._session_factory()
        try:
            return execute_for_gesture_label(
                session,
                gesture_label,
                executor=self._executor,
                record_history=record_history,
                confidence=conf,
                policy=policy,
            )
        finally:
            session.close()
