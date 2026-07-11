# -*- coding: utf-8 -*-
"""
Настройки правил привязки жестов к командам.

Хранятся в таблице ``settings`` (key-value). Здесь — типизированная обёртка
для R4 (порог уверенности), R5 (cooldown), R6 (предупреждение про опасное
действие). См. ``docs/BINDING_RULES.md``, раздел 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from app.models.database import Settings

logger = logging.getLogger(__name__)


KEY_CONFIDENCE_THRESHOLD = "binding.confidence_threshold"
KEY_COOLDOWN_MS = "binding.cooldown_ms"
KEY_WARN_TWO_HANDS = "binding.warn_two_hands"

DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_COOLDOWN_MS = 1500
DEFAULT_WARN_TWO_HANDS = True


@dataclass(frozen=True)
class BindingSettings:
    """Снимок текущих настроек правил привязки."""

    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    cooldown_ms: int = DEFAULT_COOLDOWN_MS
    warn_two_hands: bool = DEFAULT_WARN_TWO_HANDS


def _read_value(session: Session, key: str) -> Optional[str]:
    row = session.query(Settings).filter(Settings.key == key).first()
    if row is None:
        return None
    return row.value


def _write_value(session: Session, key: str, value: str) -> None:
    row = session.query(Settings).filter(Settings.key == key).first()
    if row is None:
        session.add(Settings(key=key, value=value))
    else:
        row.value = value


def _to_float(raw: Optional[str], default: float) -> float:
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _to_int(raw: Optional[str], default: int) -> int:
    if raw is None:
        return default
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return default


def _to_bool(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def load_binding_settings(session: Session) -> BindingSettings:
    """Прочитать настройки привязки из БД (с дефолтами при отсутствии записей)."""
    threshold = _to_float(
        _read_value(session, KEY_CONFIDENCE_THRESHOLD), DEFAULT_CONFIDENCE_THRESHOLD
    )
    threshold = max(0.0, min(1.0, threshold))
    cooldown = max(0, _to_int(_read_value(session, KEY_COOLDOWN_MS), DEFAULT_COOLDOWN_MS))
    warn = _to_bool(_read_value(session, KEY_WARN_TWO_HANDS), DEFAULT_WARN_TWO_HANDS)
    return BindingSettings(
        confidence_threshold=threshold,
        cooldown_ms=cooldown,
        warn_two_hands=warn,
    )


def save_binding_settings(
    session: Session,
    *,
    confidence_threshold: Optional[float] = None,
    cooldown_ms: Optional[int] = None,
    warn_two_hands: Optional[bool] = None,
    commit: bool = True,
) -> BindingSettings:
    """
    Сохранить указанные значения. ``None`` означает «не менять».

    Возвращает финальный снимок (со значениями уже из БД, включая дефолты для
    тех ключей, что остались без явной записи).
    """
    if confidence_threshold is not None:
        v = max(0.0, min(1.0, float(confidence_threshold)))
        _write_value(session, KEY_CONFIDENCE_THRESHOLD, f"{v:.4f}")
    if cooldown_ms is not None:
        v_int = max(0, int(cooldown_ms))
        _write_value(session, KEY_COOLDOWN_MS, str(v_int))
    if warn_two_hands is not None:
        _write_value(session, KEY_WARN_TWO_HANDS, "true" if warn_two_hands else "false")
    if commit:
        session.commit()
    return load_binding_settings(session)
