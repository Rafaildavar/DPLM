# -*- coding: utf-8 -*-
import json

from app.services.pointer_control import (
    POINTER_STATE_CLICK_READY,
    POINTER_STATE_DISABLED,
    POINTER_STATE_LOST,
    POINTER_STATE_TRACKING,
    PointerControlService,
    _pick_pointer_hand,
    _screen_size,
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


def test_pick_pointer_hand_prefers_closest_visible_hand():
    far_right = _proximity_landmarks(center=(0.60, 0.50), scale=0.45)
    near_left = _proximity_landmarks(center=(0.40, 0.50), scale=1.10)
    hands = [
        {"landmarks": far_right, "handedness": "Right"},
        {"landmarks": near_left, "handedness": "Left"},
    ]

    assert _pick_pointer_hand(hands)["landmarks"] is near_left


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


def test_pointer_reports_missing_macos_accessibility(monkeypatch):
    moves = []

    class FakePyAutoGUI:
        @staticmethod
        def size():
            return (1000, 800)

        @staticmethod
        def moveTo(x, y, *args, **kwargs):
            moves.append((x, y))

    import app.services.pointer_control as pc

    monkeypatch.setattr(pc, "pyautogui", FakePyAutoGUI)
    monkeypatch.setattr(pc, "PYAUTOGUI_AVAILABLE", True)
    monkeypatch.setattr(pc.sys, "platform", "darwin")
    monkeypatch.setattr(pc, "macos_accessibility_trusted", lambda: False)

    landmarks = [[0.5, 0.5] for _ in range(21)]
    payload = json.dumps([{"landmarks": landmarks, "handedness": "Right"}])

    result = PointerControlService().update(payload)

    assert result.ok is False
    assert result.moved is False
    assert result.state == POINTER_STATE_DISABLED
    assert "Универсальный доступ" in result.error
    assert moves == []


def test_screen_size_falls_back_when_pyautogui_reports_zero(monkeypatch):
    class FakePyAutoGUI:
        @staticmethod
        def size():
            return (0, 0)

    import app.services.pointer_control as pc

    monkeypatch.setattr(pc, "pyautogui", FakePyAutoGUI)
    monkeypatch.setattr(pc, "PYAUTOGUI_AVAILABLE", True)
    monkeypatch.setattr(pc.sys, "platform", "linux")

    assert _screen_size() == (1440, 900)


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


def test_pointer_double_smoothing_dampens_large_jump(monkeypatch):
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

    svc = PointerControlService(
        smoothing=0.5,
        second_smoothing=0.4,
        edge_margin=0.0,
        move_deadzone_px=0.0,
        velocity_gate_px=0.0,
        pointer_jump_limit=1.0,
    )
    first = svc.update(_payload(_open_index_landmarks(tip=(0.20, 0.20))))
    second = svc.update(_payload(_open_index_landmarks(tip=(0.80, 0.20))))

    assert first.state == POINTER_STATE_TRACKING
    assert second.state == POINTER_STATE_TRACKING
    assert moves[0][0] == 200
    assert 200 < moves[1][0] < 800


def test_pointer_rejects_single_frame_tracking_spike(monkeypatch):
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

    svc = PointerControlService(
        edge_margin=0.0,
        move_deadzone_px=0.0,
        velocity_gate_px=0.0,
        pointer_jump_limit=0.12,
        pointer_jump_hold_frames=2,
    )
    first = svc.update(_payload(_open_index_landmarks(tip=(0.50, 0.20))))
    spike = svc.update(_payload(_open_index_landmarks(tip=(0.90, 0.20))))
    back = svc.update(_payload(_open_index_landmarks(tip=(0.50, 0.20))))

    assert first.moved
    assert not spike.moved
    assert not back.moved
    assert moves == [(500, 160)]


def test_pointer_reports_lost_and_click_ready_states(monkeypatch):
    class FakePyAutoGUI:
        FAILSAFE = False
        PAUSE = 0

        @staticmethod
        def size():
            return (1000, 800)

        @staticmethod
        def moveTo(*_args, **_kwargs):
            pass

        @staticmethod
        def click(*_args, **_kwargs):
            pass

    import app.services.pointer_control as pc

    monkeypatch.setattr(pc, "pyautogui", FakePyAutoGUI)
    monkeypatch.setattr(pc, "PYAUTOGUI_AVAILABLE", True)
    monkeypatch.setattr(pc, "macos_accessibility_trusted", lambda: True)

    svc = PointerControlService(click_debounce_s=0.1)

    assert svc.update("[]").state == POINTER_STATE_LOST
    svc.update(_payload(_open_index_landmarks()))
    folded = svc.update(_payload(_folded_index_landmarks()))

    assert folded.state == POINTER_STATE_CLICK_READY
    assert folded.clicked


def test_pointer_velocity_gate_suppresses_micro_motion(monkeypatch):
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

    svc = PointerControlService(
        edge_margin=0.0,
        move_deadzone_px=0.0,
        velocity_gate_px=3.0,
    )
    first = svc.update(_payload(_open_index_landmarks(tip=(0.50, 0.20))))
    micro = svc.update(_payload(_open_index_landmarks(tip=(0.502, 0.20))))

    assert first.moved
    assert not micro.moved
    assert len(moves) == 1


def test_two_finger_swipe_left_switches_previous_window_space(monkeypatch):
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
    svc.update(_payload(_two_finger_landmarks(center=(0.56, 0.24))))
    swipe = svc.update(_payload(_two_finger_landmarks(center=(0.43, 0.24))))

    assert start.ok
    assert swipe.tab_switched == "left"
    assert hotkeys == [("ctrl", "left")]
    assert scripts == []


def test_two_finger_swipe_right_switches_next_window_space(monkeypatch):
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
    svc.update(_payload(_two_finger_landmarks(center=(0.44, 0.24))))
    swipe = svc.update(_payload(_two_finger_landmarks(center=(0.58, 0.24))))

    assert start.ok
    assert swipe.tab_switched == "right"
    assert hotkeys == [("ctrl", "right")]
    assert scripts == []


def test_two_finger_swipe_survives_brief_pose_loss(monkeypatch):
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
    svc.update(_payload(_two_finger_landmarks(center=(0.58, 0.24))))
    lost_pose = svc.update(_payload(_open_index_landmarks(tip=(0.58, 0.24))))
    swipe = svc.update(_payload(_two_finger_landmarks(center=(0.49, 0.24))))

    assert start.ok
    assert lost_pose.ok
    assert swipe.tab_switched == "left"
    assert hotkeys == [("ctrl", "left")]
    assert scripts == []


def test_open_palm_horizontal_motion_does_not_switch_windows(monkeypatch):
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


def test_two_frame_two_finger_motion_does_not_switch_windows(monkeypatch):
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
    svc.update(_payload(_two_finger_landmarks(center=(0.62, 0.24))))
    motion = svc.update(_payload(_two_finger_landmarks(center=(0.43, 0.24))))

    assert not motion.tab_switched
    assert hotkeys == []


def test_two_finger_swipe_accepts_decisive_motion_after_small_reversal(monkeypatch):
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
    monkeypatch.setattr(pc.sys, "platform", "darwin")

    svc = PointerControlService(tab_swipe_cooldown_s=0.0)
    svc.update(_payload(_two_finger_landmarks(center=(0.50, 0.24))))
    svc.update(_payload(_two_finger_landmarks(center=(0.41, 0.24))))
    reversed_motion = svc.update(_payload(_two_finger_landmarks(center=(0.53, 0.24))))
    final = svc.update(_payload(_two_finger_landmarks(center=(0.32, 0.24))))

    assert not reversed_motion.tab_switched
    assert final.tab_switched == "left"
    assert hotkeys == [("ctrl", "left")]


def test_two_finger_swipe_requires_release_before_second_switch(monkeypatch):
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
    monkeypatch.setattr(pc.sys, "platform", "linux")

    svc = PointerControlService(tab_swipe_cooldown_s=0.0)
    svc.update(_payload(_two_finger_landmarks(center=(0.62, 0.24))))
    svc.update(_payload(_two_finger_landmarks(center=(0.56, 0.24))))
    first = svc.update(_payload(_two_finger_landmarks(center=(0.43, 0.24))))
    svc.update(_payload(_two_finger_landmarks(center=(0.62, 0.24))))
    held_again = svc.update(_payload(_two_finger_landmarks(center=(0.43, 0.24))))

    assert first.tab_switched == "left"
    assert not held_again.tab_switched
    assert hotkeys == [("alt", "shift", "tab")]


def test_two_finger_swipe_freezes_cursor_while_tracking(monkeypatch):
    moves = []
    hotkeys = []

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
        def hotkey(*keys):
            hotkeys.append(keys)

    import app.services.pointer_control as pc

    monkeypatch.setattr(pc, "pyautogui", FakePyAutoGUI)
    monkeypatch.setattr(pc, "PYAUTOGUI_AVAILABLE", True)
    monkeypatch.setattr(pc, "macos_accessibility_trusted", lambda: True)
    monkeypatch.setattr(pc.sys, "platform", "darwin")

    svc = PointerControlService(tab_swipe_cooldown_s=0.0)
    first = svc.update(_payload(_open_index_landmarks(tip=(0.50, 0.24))))
    armed = svc.update(_payload(_two_finger_landmarks(center=(0.62, 0.24))))
    tracking = svc.update(_payload(_two_finger_landmarks(center=(0.56, 0.24))))
    swipe = svc.update(_payload(_two_finger_landmarks(center=(0.43, 0.24))))

    assert first.moved
    assert armed.ok
    assert not tracking.moved
    assert swipe.tab_switched == "left"
    assert hotkeys == [("ctrl", "left")]
    assert len(moves) == 2
    assert moves[0] == moves[1]


def _payload(landmarks):
    return json.dumps([{"landmarks": landmarks, "handedness": "Right"}])


def _fake_subprocess_run(scripts):
    def run(args, **_kwargs):
        scripts.append(args)

        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

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


def _proximity_landmarks(*, center=(0.5, 0.5), scale=1.0):
    cx, cy = center
    offsets = {
        0: (0.00, 0.18),
        5: (-0.07, 0.05),
        6: (-0.08, -0.04),
        7: (-0.09, -0.12),
        8: (-0.10, -0.20),
        9: (0.00, 0.04),
        10: (0.00, -0.06),
        11: (0.00, -0.15),
        12: (0.00, -0.24),
        13: (0.07, 0.05),
        14: (0.08, -0.03),
        15: (0.09, -0.10),
        16: (0.10, -0.17),
        17: (0.13, 0.08),
        18: (0.14, 0.00),
        19: (0.15, -0.06),
        20: (0.16, -0.12),
    }
    landmarks = _blank_landmarks()
    for index, (dx, dy) in offsets.items():
        landmarks[index] = [cx + dx * scale, cy + dy * scale]
    return landmarks
