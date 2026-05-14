# -*- coding: utf-8 -*-
"""Тесты связки жест → команда (gesture_command_bridge)."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, Command, Gesture, GestureHistory
from app.services.gesture_command_bridge import (
    GestureCommandBridge,
    execute_command_row,
    execute_for_gesture_label,
    resolve_command_for_gesture,
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


def test_resolve_missing(memory_session):
    assert resolve_command_for_gesture(memory_session, "no_such") is None


def test_resolve_ok(memory_session):
    g = Gesture(label="wave_hello", samples_path=None, model_class_id=0)
    memory_session.add(g)
    memory_session.commit()
    c = Command(
        name="open browser",
        platform="all",
        script_path="",
        gesture_id=g.id,
    )
    memory_session.add(c)
    memory_session.commit()

    r = resolve_command_for_gesture(memory_session, "wave_hello")
    assert r is not None
    assert r[0].label == "wave_hello"
    assert r[1].name == "open browser"


def test_execute_uses_executor(memory_session, monkeypatch):
    g = Gesture(label="pinch", samples_path=None, model_class_id=1)
    memory_session.add(g)
    memory_session.commit()
    c = Command(name="volume up", platform="all", gesture_id=g.id)
    memory_session.add(c)
    memory_session.commit()

    calls = []

    class FakeEx:
        def execute(self, name: str, **kwargs):
            calls.append(name)
            return True

    ok, msg = execute_for_gesture_label(memory_session, "pinch", executor=FakeEx(), record_history=True)
    assert ok is True
    assert msg == "volume up"
    assert calls == ["volume up"]

    hist = memory_session.query(GestureHistory).all()
    assert len(hist) == 1
    assert hist[0].gesture_id == g.id
    assert hist[0].command_id == c.id


def test_execute_command_row_script_path_url_skipped_on_non_mac(memory_session, monkeypatch):
    """На не-mac не вызываем open; проверяем ветку отказа по script_path и fallback на executor."""
    g = Gesture(label="x", samples_path=None, model_class_id=0)
    memory_session.add(g)
    memory_session.commit()
    c = Command(
        name="volume up",
        platform="all",
        script_path="https://example.com",
        gesture_id=g.id,
    )
    memory_session.add(c)
    memory_session.commit()

    monkeypatch.setattr(
        "app.services.gesture_command_bridge._host_platform",
        lambda: "linux",
    )

    class FakeEx:
        def execute(self, name: str, **kwargs):
            return name == "volume up"

    assert execute_command_row(c, executor=FakeEx()) is True


def test_bridge_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s0 = Session()
    g = Gesture(label="g1", samples_path=None, model_class_id=0)
    s0.add(g)
    s0.commit()
    s0.add(Command(name="open browser", platform="all", gesture_id=g.id))
    s0.commit()
    s0.close()

    b = GestureCommandBridge(session_factory=Session)
    assert b.command_name_for_gesture("g1") == "open browser"
    r = b.resolve("g1")
    assert r is not None
    assert r[0].label == "g1"
