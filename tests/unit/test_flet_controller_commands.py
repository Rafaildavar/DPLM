# -*- coding: utf-8 -*-

import pytest

from app.flet_app.controller import AppController, GESTURE_CONFIRM_FRAMES, _Event


def _dispatch_controller():
    controller = AppController.__new__(AppController)
    controller._confidence = 0.0
    controller._landmarks_json = "[]"
    controller._last_label = ""
    controller._pending_label = ""
    controller._pending_frames = 0
    controller._pending_confidence_total = 0.0
    controller._gesture_mode = True
    controller._pointer_mode = False
    controller._show_landmark_overlay = True
    controller._auto_execute_on_gesture = True
    controller.confidence_changed = _Event()
    controller.landmarks_changed = _Event()
    controller.gesture_detected = _Event()
    controller.gesture_mode_changed = _Event()
    controller._update_pointer_from_landmarks = lambda _landmarks: None
    return controller


def test_list_commands_includes_action_category_and_config():
    controller = AppController.__new__(AppController)

    class FakeExecutor:
        commands_registry = {
            "open browser": {
                "action": "open_app",
                "app": "Safari",
                "platform": "macos",
            }
        }

    controller._command_executor = FakeExecutor()

    commands = controller.list_commands()

    assert commands == [
        {
            "name": "open browser",
            "description": "",
            "platform": "macos",
            "action": "open_app",
            "category": "launch",
            "config": {
                "action": "open_app",
                "app": "Safari",
                "platform": "macos",
            },
        }
    ]


def test_build_training_command_filters_to_active_db_labels(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)

    monkeypatch.setattr(controller, "_active_training_labels_from_db", lambda: ["new", "new2"])
    monkeypatch.setattr(controller, "_configured_classes_path", lambda: tmp_path / "classes.json")
    monkeypatch.setattr(controller, "_configured_feature_dim_path", lambda: tmp_path / "feature_dim.txt")

    cmd = controller._build_training_command(
        data_root=str(tmp_path / "gestures"),
        out_path=str(tmp_path / "knn.pkl"),
        neighbors=3,
        expect_dim=42,
    )

    assert "--lowercase-labels" in cmd
    assert cmd.count("--include-label") == 2
    assert cmd[cmd.index("--include-label") + 1] == "new"
    assert cmd[cmd.index("--include-label", cmd.index("--include-label") + 1) + 1] == "new2"
    assert "--expect-dim" in cmd
    assert cmd[cmd.index("--expect-dim") + 1] == "42"


def test_dispatch_waits_for_stable_gesture_before_execution():
    controller = _dispatch_controller()
    emitted = []
    executed = []
    recorded = []
    controller.gesture_detected.connect(emitted.append)
    controller.execute_for_gesture = lambda label, conf: executed.append((label, conf)) or True
    controller._record_recognition_event = lambda label, conf, ok: recorded.append((label, conf, ok))

    for _ in range(GESTURE_CONFIRM_FRAMES - 1):
        controller._dispatch_infer_result(
            {"label": "new", "confidence": 0.8, "landmarks_json": "[]"}
        )

    assert emitted == []
    assert executed == []
    assert recorded == []

    controller._dispatch_infer_result(
        {"label": "new", "confidence": 0.8, "landmarks_json": "[]"}
    )

    assert emitted == ["new"]
    assert executed == [("new", pytest.approx(0.8))]
    assert recorded == [("new", pytest.approx(0.8), True)]

    controller._dispatch_infer_result(
        {"label": "new", "confidence": 0.8, "landmarks_json": "[]"}
    )

    assert emitted == ["new"]
    assert executed == [("new", pytest.approx(0.8))]


def test_dispatch_resets_confirmation_when_label_changes():
    controller = _dispatch_controller()
    executed = []
    controller.execute_for_gesture = lambda label, conf: executed.append((label, conf)) or True
    controller._record_recognition_event = lambda *_args: None

    for _ in range(GESTURE_CONFIRM_FRAMES - 2):
        controller._dispatch_infer_result(
            {"label": "new", "confidence": 0.9, "landmarks_json": "[]"}
        )
    for _ in range(GESTURE_CONFIRM_FRAMES):
        controller._dispatch_infer_result(
            {"label": "new2", "confidence": 0.7, "landmarks_json": "[]"}
        )

    assert executed == [("new2", pytest.approx(0.7))]


def test_dispatch_no_label_rearms_same_gesture():
    controller = _dispatch_controller()
    emitted = []
    executed = []
    controller.gesture_detected.connect(emitted.append)
    controller.execute_for_gesture = lambda label, conf: executed.append((label, conf)) or True
    controller._record_recognition_event = lambda *_args: None

    for _ in range(GESTURE_CONFIRM_FRAMES):
        controller._dispatch_infer_result(
            {"label": "new", "confidence": 0.8, "landmarks_json": "[]"}
        )
    controller._dispatch_infer_result(
        {"label": "", "confidence": 0.0, "landmarks_json": "[]"}
    )
    for _ in range(GESTURE_CONFIRM_FRAMES):
        controller._dispatch_infer_result(
            {"label": "new", "confidence": 0.8, "landmarks_json": "[]"}
        )

    assert emitted == ["new", "", "new"]
    assert executed == [("new", pytest.approx(0.8)), ("new", pytest.approx(0.8))]


def test_dispatch_cursor_only_ignores_gesture_execution_but_keeps_landmarks():
    controller = _dispatch_controller()
    controller._gesture_mode = False
    pointer_payloads = []
    emitted = []
    executed = []
    controller._update_pointer_from_landmarks = pointer_payloads.append
    controller.gesture_detected.connect(emitted.append)
    controller.execute_for_gesture = lambda label, conf: executed.append((label, conf)) or True
    controller._record_recognition_event = lambda *_args: None

    for _ in range(GESTURE_CONFIRM_FRAMES + 2):
        controller._dispatch_infer_result(
            {"label": "new", "confidence": 0.9, "landmarks_json": "[[[]]]"}
        )

    assert pointer_payloads == ["[[[]]]"] * (GESTURE_CONFIRM_FRAMES + 2)
    assert emitted == []
    assert executed == []
    assert controller._confidence == 0.0
    assert controller._pending_frames == 0


def test_set_gesture_mode_clears_current_label_and_starts_cv_when_enabled():
    controller = _dispatch_controller()
    emitted = []
    mode_events = []
    starts = []
    statuses = []
    controller._embedded_active = True
    controller._last_label = "new"
    controller._pending_label = "new"
    controller._pending_frames = 3
    controller.gesture_detected.connect(emitted.append)
    controller.gesture_mode_changed.connect(mode_events.append)
    controller._ensure_embedded_recognition_for_live_controls = lambda: starts.append(True)
    controller._set_status = statuses.append

    controller.set_gesture_mode(False)

    assert controller.gesture_mode is False
    assert emitted == [""]
    assert mode_events == [False]
    assert controller._pending_frames == 0
    assert controller._confidence == 0.0

    controller.set_gesture_mode(True)

    assert controller.gesture_mode is True
    assert mode_events == [False, True]
    assert starts == [True]
    assert statuses[-1] == "Распознавание жестов включено"
