# -*- coding: utf-8 -*-
import json

from app.services.pointer_control import (
    PointerControlService,
    _pick_pointer_hand,
    parse_landmarks_json,
)


def test_parse_landmarks_json_extended_format():
    raw = json.dumps(
        [
            {
                "landmarks": [[0.5, 0.5] for _ in range(21)],
                "handedness": "Right",
            }
        ]
    )
    hands = parse_landmarks_json(raw)
    assert len(hands) == 1
    assert hands[0]["handedness"] == "Right"


def test_pick_pointer_hand_prefers_right():
    hands = [
        {"landmarks": [], "handedness": "Left"},
        {"landmarks": [], "handedness": "Right"},
    ]
    assert _pick_pointer_hand(hands)["handedness"] == "Right"


def test_pointer_moves_cursor(monkeypatch):
    moves = []

    class FakePyAutoGUI:
        FAILSAFE = False
        PAUSE = 0

        @staticmethod
        def size():
            return (1000, 800)

        @staticmethod
        def moveTo(x, y, *args, **kwargs):
            moves.append((x, y))

    import app.services.pointer_control as pc

    monkeypatch.setattr(pc, "pyautogui", FakePyAutoGUI)
    monkeypatch.setattr(pc, "PYAUTOGUI_AVAILABLE", True)
    monkeypatch.setattr(pc, "macos_accessibility_trusted", lambda: True)

    landmarks = [[0.5, 0.5] for _ in range(21)]
    landmarks[8] = [0.5, 0.5]
    payload = json.dumps([{"landmarks": landmarks, "handedness": "Right"}])

    svc = PointerControlService()
    result = svc.update(payload)
    assert result.ok
    assert result.moved
    assert moves


def test_bent_index_finger_clicks_once(monkeypatch):
    moves = []
    clicks = []

    class FakePyAutoGUI:
        FAILSAFE = False
        PAUSE = 0

        @staticmethod
        def size():
            return (1000, 800)

        @staticmethod
        def moveTo(x, y, *args, **kwargs):
            moves.append((x, y))

        @staticmethod
        def click(*args, **kwargs):
            clicks.append(True)

    import app.services.pointer_control as pc

    monkeypatch.setattr(pc, "pyautogui", FakePyAutoGUI)
    monkeypatch.setattr(pc, "PYAUTOGUI_AVAILABLE", True)
    monkeypatch.setattr(pc, "macos_accessibility_trusted", lambda: True)

    svc = PointerControlService(click_debounce_s=0.1)
    open_payload = _payload(_open_index_landmarks())
    folded_payload = _payload(_folded_index_landmarks())

    assert svc.update(open_payload).ok
    first = svc.update(folded_payload)
    held = svc.update(folded_payload)

    assert first.clicked
    assert not held.clicked
    assert clicks == [True]
    assert moves


def test_pointer_ignores_small_jitter(monkeypatch):
    moves = []

    class FakePyAutoGUI:
        FAILSAFE = False
        PAUSE = 0

        @staticmethod
        def size():
            return (1000, 800)

        @staticmethod
        def moveTo(x, y, *args, **kwargs):
            moves.append((x, y))

    import app.services.pointer_control as pc

    monkeypatch.setattr(pc, "pyautogui", FakePyAutoGUI)
    monkeypatch.setattr(pc, "PYAUTOGUI_AVAILABLE", True)
    monkeypatch.setattr(pc, "macos_accessibility_trusted", lambda: True)

    svc = PointerControlService(move_deadzone_px=8)
    first = svc.update(_payload(_open_index_landmarks(tip=(0.5, 0.18))))
    jitter = svc.update(_payload(_open_index_landmarks(tip=(0.501, 0.181))))

    assert first.moved
    assert not jitter.moved
    assert len(moves) == 1


def test_two_finger_swipe_left_switches_next_tab(monkeypatch):
    hotkeys = []
    scripts = []

    class FakePyAutoGUI:
        FAILSAFE = False
        PAUSE = 0

        @staticmethod
        def size():
            return (1000, 800)

        @staticmethod
        def moveTo(x, y, *args, **kwargs):
            pass

        @staticmethod
        def hotkey(*keys):
            hotkeys.append(keys)

    import app.services.pointer_control as pc

    monkeypatch.setattr(pc, "pyautogui", FakePyAutoGUI)
    monkeypatch.setattr(pc, "PYAUTOGUI_AVAILABLE", True)
    monkeypatch.setattr(pc, "macos_accessibility_trusted", lambda: True)
    monkeypatch.setattr(pc.sys, "platform", "darwin")
    monkeypatch.setattr(pc.subprocess, "run", _fake_subprocess_run(scripts))

    svc = PointerControlService(tab_swipe_cooldown_s=0.0)
    start = svc.update(_payload(_two_finger_landmarks(center=(0.62, 0.24))))
    swipe = svc.update(_payload(_two_finger_landmarks(center=(0.43, 0.24))))

    assert start.ok
    assert swipe.tab_switched == "left"
    assert hotkeys == []
    assert "key code 124" in scripts[0][2]
    assert "command down" in scripts[0][2]
    assert "option down" in scripts[0][2]


def test_two_finger_swipe_right_switches_previous_tab(monkeypatch):
    hotkeys = []
    scripts = []

    class FakePyAutoGUI:
        FAILSAFE = False
        PAUSE = 0

        @staticmethod
        def size():
            return (1000, 800)

        @staticmethod
        def moveTo(x, y, *args, **kwargs):
            pass

        @staticmethod
        def hotkey(*keys):
            hotkeys.append(keys)

    import app.services.pointer_control as pc

    monkeypatch.setattr(pc, "pyautogui", FakePyAutoGUI)
    monkeypatch.setattr(pc, "PYAUTOGUI_AVAILABLE", True)
    monkeypatch.setattr(pc, "macos_accessibility_trusted", lambda: True)
    monkeypatch.setattr(pc.sys, "platform", "darwin")
    monkeypatch.setattr(pc.subprocess, "run", _fake_subprocess_run(scripts))

    svc = PointerControlService(tab_swipe_cooldown_s=0.0)
    start = svc.update(_payload(_two_finger_landmarks(center=(0.38, 0.24))))
    swipe = svc.update(_payload(_two_finger_landmarks(center=(0.58, 0.24))))

    assert start.ok
    assert swipe.tab_switched == "right"
    assert hotkeys == []
    assert "key code 123" in scripts[0][2]
    assert "command down" in scripts[0][2]
    assert "option down" in scripts[0][2]


def test_open_palm_horizontal_motion_does_not_switch_tabs(monkeypatch):
    hotkeys = []

    class FakePyAutoGUI:
        FAILSAFE = False
        PAUSE = 0

        @staticmethod
        def size():
            return (1000, 800)

        @staticmethod
        def moveTo(x, y, *args, **kwargs):
            pass

        @staticmethod
        def hotkey(*keys):
            hotkeys.append(keys)

    import app.services.pointer_control as pc

    monkeypatch.setattr(pc, "pyautogui", FakePyAutoGUI)
    monkeypatch.setattr(pc, "PYAUTOGUI_AVAILABLE", True)
    monkeypatch.setattr(pc, "macos_accessibility_trusted", lambda: True)

    svc = PointerControlService(tab_swipe_cooldown_s=0.0)
    start = svc.update(_payload(_open_palm_landmarks(center=(0.62, 0.24))))
    motion = svc.update(_payload(_open_palm_landmarks(center=(0.43, 0.24))))

    assert start.ok
    assert not motion.tab_switched
    assert hotkeys == []


def _payload(landmarks):
    return json.dumps([{"landmarks": landmarks, "handedness": "Right"}])


def _fake_subprocess_run(scripts):
    def run(args, **_kwargs):
        scripts.append(args)

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    return run


def _blank_landmarks():
    return [[0.5, 0.5] for _ in range(21)]


def _open_index_landmarks(*, tip=(0.5, 0.18)):
    landmarks = _blank_landmarks()
    landmarks[0] = [0.5, 0.82]
    landmarks[5] = [0.5, 0.56]
    landmarks[6] = [0.5, 0.42]
    landmarks[7] = [0.5, 0.30]
    landmarks[8] = [tip[0], tip[1]]
    return landmarks


def _folded_index_landmarks():
    landmarks = _blank_landmarks()
    landmarks[0] = [0.5, 0.82]
    landmarks[5] = [0.5, 0.56]
    landmarks[6] = [0.5, 0.42]
    landmarks[7] = [0.55, 0.49]
    landmarks[8] = [0.51, 0.57]
    return landmarks


def _two_finger_landmarks(*, center=(0.5, 0.24)):
    landmarks = _blank_landmarks()
    index_x = center[0] - 0.035
    middle_x = center[0] + 0.035
    landmarks[0] = [center[0], 0.82]
    landmarks[5] = [index_x, 0.58]
    landmarks[6] = [index_x, 0.43]
    landmarks[7] = [index_x, 0.31]
    landmarks[8] = [index_x, center[1]]
    landmarks[9] = [middle_x, 0.58]
    landmarks[10] = [middle_x, 0.43]
    landmarks[11] = [middle_x, 0.31]
    landmarks[12] = [middle_x, center[1]]
    return landmarks


def _open_palm_landmarks(*, center=(0.5, 0.24)):
    landmarks = _two_finger_landmarks(center=center)
    ring_x = center[0] + 0.095
    pinky_x = center[0] + 0.15
    landmarks[13] = [ring_x, 0.58]
    landmarks[14] = [ring_x, 0.43]
    landmarks[15] = [ring_x, 0.31]
    landmarks[16] = [ring_x, center[1]]
    landmarks[17] = [pinky_x, 0.60]
    landmarks[18] = [pinky_x, 0.45]
    landmarks[19] = [pinky_x, 0.33]
    landmarks[20] = [pinky_x, center[1] + 0.02]
    return landmarks
