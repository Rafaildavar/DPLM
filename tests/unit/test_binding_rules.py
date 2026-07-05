# -*- coding: utf-8 -*-
"""
Тесты правил привязки жест→команда (см. ``docs/BINDING_RULES.md``):
R1, R3, R6, R7, R8 — содержательные проверки на стороне сервиса.
"""
import json
import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, Command, Gesture
from app.services.gesture_command_bridge import (
    is_dangerous_command,
    save_gesture_binding,
)
from app.services.user_command_sync import (
    ACTION_SPEC_SCHEMA,
    ACTION_TO_CATEGORY,
    CATEGORY_LABELS,
    DANGEROUS_ACTIONS,
    category_for_action,
    is_dangerous_action,
    validate_action_spec,
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


# ---------------------------------------------------------------------------
# Структурные проверки таксономии (раздел 1 BINDING_RULES.md)
# ---------------------------------------------------------------------------


def test_taxonomy_has_expected_categories():
    assert set(CATEGORY_LABELS.keys()) == {
        "launch",
        "system",
        "navigation",
        "media",
        "hotkey",
        "script",
        "workflow",
    }


def test_every_action_has_category():
    for action in ACTION_SPEC_SCHEMA:
        assert action in ACTION_TO_CATEGORY, action
        assert category_for_action(action) in CATEGORY_LABELS


def test_all_categories_have_at_least_one_action():
    counts = {cid: 0 for cid in CATEGORY_LABELS}
    for cid in ACTION_TO_CATEGORY.values():
        counts[cid] += 1
    for cid, n in counts.items():
        assert n >= 1, f"категория {cid} пуста"


# ---------------------------------------------------------------------------
# R1 — 1:1 жест ↔ команда
# ---------------------------------------------------------------------------


def test_r1_overwrite_existing_binding(memory_session):
    """Перепривязка того же жеста: старая команда теряет gesture_id, новая получает."""
    g = Gesture(label="palm", samples_path=None, model_class_id=0, is_active=True)
    memory_session.add(g)
    memory_session.commit()

    old = Command(
        name="Stop",
        platform="macos",
        gesture_id=g.id,
        action_spec=json.dumps({"action": "lock_screen"}),
    )
    memory_session.add(old)
    memory_session.commit()

    new = Command(
        name="OpenSafari",
        platform="macos",
        action_spec=json.dumps({"action": "open_app", "app": "Safari"}),
    )
    memory_session.add(new)
    memory_session.commit()

    old.gesture_id = None
    new.gesture_id = g.id
    memory_session.commit()

    bound = memory_session.query(Command).filter(Command.gesture_id == g.id).all()
    assert len(bound) == 1
    assert bound[0].name == "OpenSafari"


# ---------------------------------------------------------------------------
# R3 — фильтр жестов: только активные и обученные
# ---------------------------------------------------------------------------


def test_r3_filter_inactive_gestures(memory_session):
    memory_session.add(Gesture(label="ok", model_class_id=1, is_active=True))
    memory_session.add(Gesture(label="off", model_class_id=2, is_active=False))
    memory_session.add(Gesture(label="untrained", model_class_id=None, is_active=True))
    memory_session.commit()

    rows = (
        memory_session.query(Gesture)
        .filter(Gesture.is_active.is_(True))
        .filter(Gesture.model_class_id.isnot(None))
        .all()
    )
    assert {g.label for g in rows} == {"ok"}


# ---------------------------------------------------------------------------
# R6 — опасные действия
# ---------------------------------------------------------------------------


def test_r6_dangerous_action_warning():
    assert is_dangerous_action("lock_screen") is True
    assert is_dangerous_action("run_script") is True
    assert is_dangerous_action("scroll") is False
    assert is_dangerous_action("open_app") is False
    assert "lock_screen" in DANGEROUS_ACTIONS


def test_r6_is_dangerous_command_via_action_spec():
    cmd = Command(
        name="Lock",
        platform="macos",
        action_spec=json.dumps({"action": "lock_screen"}),
    )
    assert is_dangerous_command(cmd) is True

    cmd_safe = Command(
        name="Vol+",
        platform="macos",
        action_spec=json.dumps({"action": "volume_up"}),
    )
    assert is_dangerous_command(cmd_safe) is False


def test_r6_is_dangerous_command_inside_sequence():
    cmd = Command(
        name="Workflow",
        platform="macos",
        action_spec=json.dumps(
            {
                "action": "sequence",
                "steps": [
                    {"action": "open_app", "app": "Preview"},
                    {"action": "lock_screen"},
                ],
            }
        ),
    )
    assert is_dangerous_command(cmd) is True


def test_r6_is_dangerous_command_via_shell_prefix():
    cmd = Command(name="Shell", platform="macos", script_path="shell:rm -rf /tmp/x")
    assert is_dangerous_command(cmd) is True


# ---------------------------------------------------------------------------
# R7 — валидация action_spec
# ---------------------------------------------------------------------------


def test_r7_unknown_action_rejected():
    err = validate_action_spec({"action": "definitely_not_a_thing"})
    assert err is not None
    assert "Неизвестное действие" in err


def test_r7_open_app_requires_app_field():
    assert validate_action_spec({"action": "open_app"}) is not None
    assert validate_action_spec({"action": "open_app", "app": "Safari"}) is None


def test_r7_open_url_requires_http():
    assert validate_action_spec({"action": "open_url"}) is not None
    err = validate_action_spec({"action": "open_url", "url": "ftp://example.com"})
    assert err is not None
    assert "http" in err.lower()
    assert validate_action_spec({"action": "open_url", "url": "https://example.com"}) is None


def test_r7_open_path_requires_existing_absolute_path(tmp_path):
    target = tmp_path / "task.pdf"
    target.write_text("demo\n")

    assert validate_action_spec({"action": "open_path"}) is not None
    err = validate_action_spec({"action": "open_path", "path": "relative/file.txt"})
    assert err is not None
    assert "абсолютным" in err
    assert validate_action_spec({"action": "open_path", "path": str(target)}) is None


def test_r7_scroll_clicks_must_be_int():
    assert validate_action_spec({"action": "scroll"}) is not None
    err = validate_action_spec({"action": "scroll", "clicks": "abc"})
    assert err is not None
    assert validate_action_spec({"action": "scroll", "clicks": -3}) is None


def test_r7_press_requires_key():
    assert validate_action_spec({"action": "press"}) is not None
    assert validate_action_spec({"action": "press", "key": "pagedown"}) is None


def test_r7_key_combination_requires_non_empty_list():
    assert validate_action_spec({"action": "key_combination"}) is not None
    assert validate_action_spec({"action": "key_combination", "keys": []}) is not None
    assert validate_action_spec({"action": "key_combination", "keys": ["x", ""]}) is not None
    assert (
        validate_action_spec({"action": "key_combination", "keys": ["command", "space"]})
        is None
    )


def test_r7_media_key_kind_validated():
    assert validate_action_spec({"action": "media_key"}) is not None
    assert validate_action_spec({"action": "media_key", "kind": "what"}) is not None
    assert validate_action_spec({"action": "media_key", "kind": "play_pause"}) is None
    assert validate_action_spec({"action": "media_key", "kind": "next"}) is None


def test_r7_wait_and_notify_validated():
    assert validate_action_spec({"action": "wait"}) is not None
    assert validate_action_spec({"action": "wait", "seconds": "abc"}) is not None
    assert validate_action_spec({"action": "wait", "seconds": 0}) is not None
    assert validate_action_spec({"action": "wait", "seconds": 1.5}) is None

    assert validate_action_spec({"action": "notify"}) is not None
    assert validate_action_spec({"action": "notify", "message": "Готово"}) is None


def test_r7_sequence_validates_each_step(tmp_path):
    target = tmp_path / "lesson.txt"
    target.write_text("demo\n")

    spec = {
        "action": "sequence",
        "steps": [
            {"action": "open_path", "path": str(target)},
            {"action": "wait", "seconds": 0.5},
            {"action": "notify", "message": "Рабочее место готово"},
        ],
    }
    assert validate_action_spec(spec) is None

    assert validate_action_spec({"action": "sequence", "steps": []}) is not None
    err = validate_action_spec(
        {
            "action": "sequence",
            "steps": [{"action": "open_path", "path": "/no/such/file"}],
        }
    )
    assert err is not None
    assert "шаг 1" in err

    nested = {"action": "sequence", "steps": [{"action": "sequence", "steps": []}]}
    err = validate_action_spec(nested)
    assert err is not None
    assert "нельзя вкладывать" in err


def test_r7_run_script_requires_existing_py(tmp_path):
    err = validate_action_spec({"action": "run_script", "script_path": ""})
    assert err is not None

    err = validate_action_spec({"action": "run_script", "script_path": "relative/path.py"})
    assert err is not None
    assert "абсолютным" in err

    err = validate_action_spec(
        {"action": "run_script", "script_path": "/tmp/__no_such__.py"}
    )
    assert err is not None

    sh = tmp_path / "tool.sh"
    sh.write_text("#!/bin/sh\n")
    err = validate_action_spec({"action": "run_script", "script_path": str(sh)})
    assert err is not None
    assert ".py" in err

    py = tmp_path / "ok.py"
    py.write_text("print('hi')\n")
    assert (
        validate_action_spec({"action": "run_script", "script_path": str(py)}) is None
    )


def test_r7_system_actions_have_no_required_fields():
    for action in ("volume_up", "volume_down", "mute_toggle", "lock_screen", "screenshot"):
        assert validate_action_spec({"action": action}) is None


# ---------------------------------------------------------------------------
# R8 — уникальность имени команды
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# save_gesture_binding — регрессии R1/R8 на реальной SQLAlchemy-сессии
# ---------------------------------------------------------------------------


def test_save_binding_creates_new_command(memory_session):
    g = Gesture(label="palm", model_class_id=0, is_active=True)
    memory_session.add(g)
    memory_session.commit()

    cmd, created = save_gesture_binding(
        memory_session, "palm", "Open Safari",
        {"action": "open_app", "app": "Safari"},
    )
    assert created is True
    assert cmd.name == "Open Safari"
    assert cmd.gesture_id == g.id
    assert cmd.is_active is True
    assert cmd.platform == "macos"
    assert json.loads(cmd.action_spec)["app"] == "Safari"


def test_save_binding_unknown_gesture_raises(memory_session):
    with pytest.raises(ValueError, match="не найден"):
        save_gesture_binding(memory_session, "nope", "X", {"action": "open_app", "app": "Safari"})


def test_save_binding_rebind_creates_separate_commands(memory_session):
    """
    Регрессия: ранее saveBinding переименовывал старую команду вместо отвязки.
    После исправления старая команда сохраняет своё имя, теряет gesture_id;
    новая команда создаётся как отдельная запись.
    """
    g = Gesture(label="palm", model_class_id=0, is_active=True)
    memory_session.add(g)
    memory_session.commit()

    save_gesture_binding(memory_session, "palm", "Open Safari", {"action": "open_app", "app": "Safari"})
    save_gesture_binding(memory_session, "palm", "Lock", {"action": "lock_screen"})

    safari = memory_session.query(Command).filter(Command.name == "Open Safari").first()
    lock = memory_session.query(Command).filter(Command.name == "Lock").first()

    assert safari is not None, "старая команда не должна исчезать при перепривязке"
    assert lock is not None, "новая команда должна быть создана"
    assert safari.id != lock.id, "должны быть две разные записи"
    assert safari.gesture_id is None
    assert lock.gesture_id == g.id


def test_save_binding_reuses_command_by_name(memory_session):
    """
    R8: если команда с указанным именем уже существует, она и обновляется
    (включая action_spec), а не создаётся дубликат.
    """
    g1 = Gesture(label="palm", model_class_id=0, is_active=True)
    g2 = Gesture(label="zoom", model_class_id=1, is_active=True)
    memory_session.add_all([g1, g2])
    memory_session.commit()

    cmd1, created1 = save_gesture_binding(
        memory_session, "palm", "Open Safari",
        {"action": "open_app", "app": "Safari"},
    )
    assert created1 is True

    cmd2, created2 = save_gesture_binding(
        memory_session, "zoom", "Open Safari",
        {"action": "open_app", "app": "Music"},
    )
    assert created2 is False, "имя уже занято — должна обновиться существующая запись"
    assert cmd2.id == cmd1.id, "id не меняется"
    assert cmd2.gesture_id == g2.id, "должна перепривязаться к новому жесту"
    assert json.loads(cmd2.action_spec)["app"] == "Music"

    total = memory_session.query(Command).filter(Command.name == "Open Safari").count()
    assert total == 1, "в БД ровно одна запись с таким именем"


def test_save_binding_swap_two_commands(memory_session):
    """
    Сценарий «двойного свопа»: на палм висит А, на зум висит B.
    Перепривязываем палм на B (по имени) — должны отвязаться обе старые
    привязки, B встаёт на палм, А освобождается.
    """
    g_palm = Gesture(label="palm", model_class_id=0, is_active=True)
    g_zoom = Gesture(label="zoom", model_class_id=1, is_active=True)
    memory_session.add_all([g_palm, g_zoom])
    memory_session.commit()

    save_gesture_binding(memory_session, "palm", "A", {"action": "open_app", "app": "X"})
    save_gesture_binding(memory_session, "zoom", "B", {"action": "open_app", "app": "Y"})

    save_gesture_binding(memory_session, "palm", "B", {"action": "open_app", "app": "Y"})

    a = memory_session.query(Command).filter(Command.name == "A").first()
    b = memory_session.query(Command).filter(Command.name == "B").first()
    assert a.gesture_id is None
    assert b.gesture_id == g_palm.id


def test_r8_unique_name_constraint(memory_session):
    g1 = Gesture(label="a", model_class_id=0, is_active=True)
    g2 = Gesture(label="b", model_class_id=1, is_active=True)
    memory_session.add_all([g1, g2])
    memory_session.commit()

    memory_session.add(Command(name="Same", platform="macos", gesture_id=g1.id))
    memory_session.commit()

    memory_session.add(Command(name="Same", platform="macos", gesture_id=g2.id))
    with pytest.raises(Exception):
        memory_session.commit()
    memory_session.rollback()
