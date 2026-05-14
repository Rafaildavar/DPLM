# -*- coding: utf-8 -*-
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, Command, Gesture
from app.services.command_executor import CommandExecutor
from app.services.user_command_sync import (
    ACTION_SPEC_SCHEMA,
    executor_config_from_row,
    parse_command_action_spec,
    sync_db_commands_to_executor,
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


def test_parse_action_spec(memory_session):
    g = Gesture(label="g", samples_path=None, model_class_id=0)
    memory_session.add(g)
    memory_session.commit()
    c = Command(
        name="n1",
        platform="all",
        gesture_id=g.id,
        action_spec=json.dumps({"action": "scroll", "clicks": -2}),
    )
    memory_session.add(c)
    memory_session.commit()
    spec = parse_command_action_spec(c)
    assert spec == {"action": "scroll", "clicks": -2}


def test_executor_config_from_row_inherits_platform(memory_session):
    g = Gesture(label="g2", samples_path=None, model_class_id=0)
    memory_session.add(g)
    memory_session.commit()
    c = Command(name="n2", platform="macos", gesture_id=g.id)
    cfg = executor_config_from_row(c, {"action": "open_app", "app": "Safari"})
    assert cfg["platform"] == "macos"


def test_sync_db_commands_to_executor(memory_session):
    g = Gesture(label="g3", samples_path=None, model_class_id=0)
    memory_session.add(g)
    memory_session.commit()
    memory_session.add(
        Command(
            name="User Scroll Down",
            platform="all",
            gesture_id=g.id,
            is_active=True,
            action_spec=json.dumps({"action": "scroll", "clicks": -3}),
        )
    )
    memory_session.commit()

    ex = CommandExecutor()
    n = sync_db_commands_to_executor(memory_session, ex)
    assert n == 1
    assert "user scroll down" in ex.commands_registry


def test_action_spec_schema_has_known_actions():
    assert "open_app" in ACTION_SPEC_SCHEMA
    assert "scroll" in ACTION_SPEC_SCHEMA
