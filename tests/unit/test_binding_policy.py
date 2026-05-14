# -*- coding: utf-8 -*-
"""
Тесты ``BindingPolicy``: R4 (порог уверенности), R5 (cooldown) и
``binding_settings.load_binding_settings`` (чтение из ``settings``).
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, Settings
from app.services.binding_settings import (
    BindingSettings,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_COOLDOWN_MS,
    KEY_CONFIDENCE_THRESHOLD,
    KEY_COOLDOWN_MS,
    KEY_WARN_TWO_HANDS,
    load_binding_settings,
    save_binding_settings,
)
from app.services.gesture_command_bridge import (
    REJECT_COOLDOWN,
    REJECT_LOW_CONFIDENCE,
    BindingPolicy,
)


@pytest.fixture
def memory_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()


def test_load_binding_settings_defaults(memory_session):
    s = load_binding_settings(memory_session)
    assert isinstance(s, BindingSettings)
    assert s.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
    assert s.cooldown_ms == DEFAULT_COOLDOWN_MS
    assert s.warn_two_hands is True


def test_save_binding_settings_round_trip(memory_session):
    save_binding_settings(
        memory_session,
        confidence_threshold=0.8,
        cooldown_ms=500,
        warn_two_hands=False,
    )
    s = load_binding_settings(memory_session)
    assert pytest.approx(s.confidence_threshold) == 0.8
    assert s.cooldown_ms == 500
    assert s.warn_two_hands is False

    rows = {row.key: row.value for row in memory_session.query(Settings).all()}
    assert KEY_CONFIDENCE_THRESHOLD in rows
    assert KEY_COOLDOWN_MS in rows
    assert KEY_WARN_TWO_HANDS in rows


def test_save_binding_settings_clamps_threshold(memory_session):
    save_binding_settings(memory_session, confidence_threshold=1.7)
    s = load_binding_settings(memory_session)
    assert s.confidence_threshold == 1.0

    save_binding_settings(memory_session, confidence_threshold=-0.1)
    s = load_binding_settings(memory_session)
    assert s.confidence_threshold == 0.0


def test_r4_low_confidence_rejected():
    policy = BindingPolicy(settings=BindingSettings(confidence_threshold=0.7, cooldown_ms=0))
    decision = policy.evaluate("swipe_right", confidence=0.5)
    assert decision.allow is False
    assert decision.reason == REJECT_LOW_CONFIDENCE
    assert decision.threshold == 0.7


def test_r4_above_threshold_allowed():
    policy = BindingPolicy(settings=BindingSettings(confidence_threshold=0.6, cooldown_ms=0))
    decision = policy.evaluate("swipe_right", confidence=0.61)
    assert decision.allow is True
    assert decision.reason == ""


def test_r5_cooldown_blocks_second_fire(monkeypatch):
    policy = BindingPolicy(settings=BindingSettings(confidence_threshold=0.0, cooldown_ms=1000))

    fake_time = {"v": 10_000}
    monkeypatch.setattr(BindingPolicy, "_now_ms", staticmethod(lambda: fake_time["v"]))

    d1 = policy.evaluate("palm", confidence=0.9)
    assert d1.allow is True
    policy.mark_fired("palm")

    fake_time["v"] += 200
    d2 = policy.evaluate("palm", confidence=0.95)
    assert d2.allow is False
    assert d2.reason == REJECT_COOLDOWN
    assert d2.cooldown_remaining_ms == 800


def test_r5_cooldown_lets_through_after_window(monkeypatch):
    policy = BindingPolicy(settings=BindingSettings(confidence_threshold=0.0, cooldown_ms=500))

    fake_time = {"v": 0}
    monkeypatch.setattr(BindingPolicy, "_now_ms", staticmethod(lambda: fake_time["v"]))

    assert policy.evaluate("palm", confidence=0.9).allow is True
    policy.mark_fired("palm")

    fake_time["v"] = 600
    assert policy.evaluate("palm", confidence=0.9).allow is True


def test_r5_cooldown_per_label(monkeypatch):
    """Cooldown отслеживается отдельно для каждой метки жеста."""
    policy = BindingPolicy(settings=BindingSettings(confidence_threshold=0.0, cooldown_ms=2000))

    fake_time = {"v": 0}
    monkeypatch.setattr(BindingPolicy, "_now_ms", staticmethod(lambda: fake_time["v"]))

    assert policy.evaluate("palm", confidence=0.9).allow is True
    policy.mark_fired("palm")

    assert policy.evaluate("zoom", confidence=0.9).allow is True


def test_policy_reset_cooldowns(monkeypatch):
    policy = BindingPolicy(settings=BindingSettings(confidence_threshold=0.0, cooldown_ms=2000))

    fake_time = {"v": 0}
    monkeypatch.setattr(BindingPolicy, "_now_ms", staticmethod(lambda: fake_time["v"]))

    policy.mark_fired("palm")
    fake_time["v"] = 100
    assert policy.evaluate("palm", confidence=0.9).allow is False

    policy.reset_cooldowns()
    assert policy.evaluate("palm", confidence=0.9).allow is True
