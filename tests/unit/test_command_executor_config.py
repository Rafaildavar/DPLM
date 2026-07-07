# -*- coding: utf-8 -*-
import pytest

import app.services.command_executor as ce


def _enable_fake_pyautogui(monkeypatch):
    monkeypatch.setattr(ce, "PYAUTOGUI_AVAILABLE", True)


def test_execute_config_scroll(monkeypatch):
    calls = []

    def fake_scroll(n):
        calls.append(n)

    _enable_fake_pyautogui(monkeypatch)
    monkeypatch.setattr(ce.pyautogui, "scroll", fake_scroll)
    ex = ce.CommandExecutor()
    ok = ex.execute_config({"action": "scroll", "clicks": -7, "platform": "all"})
    assert ok is True
    assert calls == [-7]


def test_execute_config_platform_mismatch(monkeypatch):
    ex = ce.CommandExecutor()
    monkeypatch.setattr(ce.CommandExecutor, "_host_platform_tag", lambda self: "macos")
    ok = ex.execute_config({"action": "scroll", "clicks": 1, "platform": "windows"})
    assert ok is False


def test_execute_config_open_path_uses_os_open(monkeypatch, tmp_path):
    ex = ce.CommandExecutor()
    monkeypatch.setattr(ex, "system", "darwin")
    target = tmp_path / "lesson.txt"
    target.write_text("demo\n")
    captured = {"args": None}

    class FakePopen:
        def __init__(self, args, **kwargs):
            captured["args"] = args

    monkeypatch.setattr(ce.subprocess, "Popen", FakePopen)
    ok = ex.execute_config({"action": "open_path", "path": str(target), "platform": "all"})
    assert ok is True
    assert captured["args"] == ["open", str(target)]


def test_execute_config_wait(monkeypatch):
    ex = ce.CommandExecutor()
    slept = []
    monkeypatch.setattr(ce.time, "sleep", lambda value: slept.append(value))

    ok = ex.execute_config({"action": "wait", "seconds": "1.5", "platform": "all"})
    assert ok is True
    assert slept == [1.5]


def test_execute_config_notify_macos(monkeypatch):
    ex = ce.CommandExecutor()
    monkeypatch.setattr(ex, "system", "darwin")
    captured = {"args": None}

    class FakePopen:
        def __init__(self, args, **kwargs):
            captured["args"] = args

    monkeypatch.setattr(ce.subprocess, "Popen", FakePopen)
    ok = ex.execute_config(
        {
            "action": "notify",
            "title": "GestureBind",
            "message": "Рабочее место готово",
            "platform": "all",
        }
    )
    assert ok is True
    assert captured["args"][0] == "osascript"
    assert "Рабочее место готово" in captured["args"][2]


def test_execute_config_sequence_runs_steps_in_order(monkeypatch):
    ex = ce.CommandExecutor()
    calls = []

    monkeypatch.setattr(
        ex,
        "_open_application",
        lambda app: calls.append(("open_app", app)) or True,
    )
    monkeypatch.setattr(
        ex,
        "_wait",
        lambda seconds: calls.append(("wait", seconds)) or True,
    )
    monkeypatch.setattr(
        ex,
        "_notify",
        lambda message, *, title="GestureBind": calls.append(("notify", title, message)) or True,
    )

    ok = ex.execute_config(
        {
            "action": "sequence",
            "platform": "all",
            "steps": [
                {"action": "open_app", "app": "Preview"},
                {"action": "wait", "seconds": 0.5},
                {"action": "notify", "title": "GestureBind", "message": "Готово"},
            ],
        }
    )
    assert ok is True
    assert calls == [
        ("open_app", "Preview"),
        ("wait", 0.5),
        ("notify", "GestureBind", "Готово"),
    ]


def test_execute_config_sequence_stops_on_failed_step(monkeypatch):
    ex = ce.CommandExecutor()
    calls = []
    monkeypatch.setattr(ex, "_open_application", lambda app: False)
    monkeypatch.setattr(
        ex,
        "_notify",
        lambda message, *, title="GestureBind": calls.append(("notify", message)) or True,
    )

    ok = ex.execute_config(
        {
            "action": "sequence",
            "platform": "all",
            "steps": [
                {"action": "open_app", "app": "NoSuchApp"},
                {"action": "notify", "message": "Не должно выполниться"},
            ],
        }
    )
    assert ok is False
    assert calls == []


# ---------------------------------------------------------------------------
# Новые macOS-действия (см. docs/BINDING_RULES.md, раздел 1.2 и 1.4)
# ---------------------------------------------------------------------------


def test_execute_config_mute_toggle(monkeypatch):
    pressed = []
    _enable_fake_pyautogui(monkeypatch)
    monkeypatch.setattr(ce.pyautogui, "press", lambda k: pressed.append(k))

    ex = ce.CommandExecutor()
    # mute на macOS идёт через AppleScript; тестируем pyautogui-ветку (Windows).
    monkeypatch.setattr(ex, "system", "windows")
    ok = ex.execute_config({"action": "mute_toggle", "platform": "all"})
    assert ok is True
    assert pressed == ["volumemute"]


def test_execute_config_brightness_up_uses_osascript(monkeypatch):
    ex = ce.CommandExecutor()
    monkeypatch.setattr(ce.CommandExecutor, "_host_platform_tag", lambda self: "macos")
    monkeypatch.setattr(ex, "system", "darwin")

    captured = {"args": None}

    class FakeProc:
        returncode = 0

    def fake_run(args, **kwargs):
        captured["args"] = args
        return FakeProc()

    monkeypatch.setattr(ce.subprocess, "run", fake_run)
    ok = ex.execute_config({"action": "brightness_up", "platform": "macos"})
    assert ok is True
    assert captured["args"][0] == "osascript"
    assert "key code 144" in " ".join(captured["args"])


def test_execute_config_brightness_down_uses_osascript(monkeypatch):
    ex = ce.CommandExecutor()
    monkeypatch.setattr(ce.CommandExecutor, "_host_platform_tag", lambda self: "macos")
    monkeypatch.setattr(ex, "system", "darwin")

    captured = {"args": None}

    def fake_run(args, **kwargs):
        captured["args"] = args
        class P:
            returncode = 0
        return P()

    monkeypatch.setattr(ce.subprocess, "run", fake_run)
    ok = ex.execute_config({"action": "brightness_down", "platform": "macos"})
    assert ok is True
    assert "key code 145" in " ".join(captured["args"])


def test_execute_config_lock_screen_pmset(monkeypatch):
    ex = ce.CommandExecutor()
    monkeypatch.setattr(ex, "system", "darwin")

    captured = {"args": None}

    class FakePopen:
        def __init__(self, args, **kwargs):
            captured["args"] = args

    monkeypatch.setattr(ce.subprocess, "Popen", FakePopen)
    ok = ex.execute_config({"action": "lock_screen", "platform": "macos"})
    assert ok is True
    assert captured["args"] == ["pmset", "displaysleepnow"]


def test_execute_config_screenshot(monkeypatch, tmp_path):
    ex = ce.CommandExecutor()
    monkeypatch.setattr(ex, "system", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Desktop").mkdir()

    captured = {"args": None}

    class FakePopen:
        def __init__(self, args, **kwargs):
            captured["args"] = args

    monkeypatch.setattr(ce.subprocess, "Popen", FakePopen)
    ok = ex.execute_config({"action": "screenshot", "platform": "macos"})
    assert ok is True
    assert captured["args"][0] == "screencapture"
    assert captured["args"][1] == "-x"
    assert "dplm_screenshot_" in captured["args"][2]


@pytest.mark.parametrize(
    "kind,expected_key",
    [
        ("play_pause", "playpause"),
        ("play", "playpause"),
        ("pause", "playpause"),
        ("next", "nexttrack"),
        ("prev", "prevtrack"),
        ("previous", "prevtrack"),
    ],
)
def test_execute_config_media_key(monkeypatch, kind, expected_key):
    pressed = []
    _enable_fake_pyautogui(monkeypatch)
    monkeypatch.setattr(ce.pyautogui, "press", lambda k: pressed.append(k))

    ex = ce.CommandExecutor()
    # На darwin медиаклавиши — osascript; на Linux проверяем pyautogui.
    monkeypatch.setattr(ex, "system", "linux")
    ok = ex.execute_config({"action": "media_key", "kind": kind, "platform": "all"})
    assert ok is True
    assert pressed == [expected_key]


def test_execute_config_media_key_unknown_kind(monkeypatch):
    pressed = []
    _enable_fake_pyautogui(monkeypatch)
    monkeypatch.setattr(ce.pyautogui, "press", lambda k: pressed.append(k))

    ex = ce.CommandExecutor()
    ok = ex.execute_config({"action": "media_key", "kind": "weird", "platform": "all"})
    assert ok is False
    assert pressed == []
