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

        @staticmethod
        def mouseDown(*args, **kwargs):
            raise AssertionError("quick click must not start drag")

        @staticmethod
        def mouseUp(*args, **kwargs):
            raise AssertionError("quick click must not release drag")

    import app.services.pointer_control as pc

    monkeypatch.setattr(pc, "pyautogui", FakePyAutoGUI)
    monkeypatch.setattr(pc, "PYAUTOGUI_AVAILABLE", True)
    monkeypatch.setattr(pc, "macos_accessibility_trusted", lambda: True)

    svc = PointerControlService(click_debounce_s=0.1)
    open_payload = _payload(_open_index_landmarks())
    folded_payload = _payload(_folded_index_landmarks())

    assert svc.update(open_payload).ok
    folded = svc.update(folded_payload)
    held = svc.update(folded_payload)
    released = svc.update(open_payload)

    assert not folded.clicked
    assert not held.clicked
    assert released.clicked
    assert clicks == [True]
    assert moves


def test_bent_index_finger_drag_selects_until_release(monkeypatch):
    events = []

    class FakePyAutoGUI:
        FAILSAFE = False
        PAUSE = 0

        @staticmethod
        def size():
            return (1000, 800)

        @staticmethod
        def moveTo(x, y, *args, **kwargs):
            events.append(("move", x, y))

        @staticmethod
        def click(*args, **kwargs):
            events.append(("click",))

        @staticmethod
        def mouseDown(*args, **kwargs):
            events.append(("down",))

        @staticmethod
        def mouseUp(*args, **kwargs):
            events.append(("up",))

    import app.services.pointer_control as pc

    monkeypatch.setattr(pc, "pyautogui", FakePyAutoGUI)
    monkeypatch.setattr(pc, "PYAUTOGUI_AVAILABLE", True)
    monkeypatch.setattr(pc, "macos_accessibility_trusted", lambda: True)

    svc = PointerControlService(drag_start_px=12)

    assert svc.update(_payload(_open_index_landmarks())).moved
    assert not svc.update(_payload(_folded_index_landmarks())).drag_started
    started = svc.update(_payload(_folded_index_landmarks(offset_x=0.08)))
    released = svc.update(_payload(_open_index_landmarks(tip=(0.58, 0.18))))

    assert started.drag_started
    assert started.dragging
    assert released.drag_ended
    assert ("down",) in events
    assert ("up",) in events
    assert ("click",) not in events


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


def _payload(landmarks):
    return json.dumps([{"landmarks": landmarks, "handedness": "Right"}])


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


def _folded_index_landmarks(*, offset_x=0.0):
    landmarks = _blank_landmarks()
    landmarks[0] = [0.5, 0.82]
    landmarks[5] = [0.5 + offset_x, 0.56]
    landmarks[6] = [0.5 + offset_x, 0.42]
    landmarks[7] = [0.55 + offset_x, 0.49]
    landmarks[8] = [0.51 + offset_x, 0.57]
    return landmarks
