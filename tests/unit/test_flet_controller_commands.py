# -*- coding: utf-8 -*-

import json

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.flet_app.controller import AppController, GESTURE_CONFIRM_FRAMES, _Event
from app.models.database import Base, Command, Gesture, GestureHistory, GestureSample


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


def test_sync_dataset_to_db_imports_new_samples_before_training(monkeypatch, tmp_path):
    import app.flet_app.controller as controller_module

    data_root = tmp_path / "gestures"
    label_dir = data_root / "CTRLZ"
    label_dir.mkdir(parents=True)
    for idx in range(3):
        np.save(
            label_dir / f"sample_{idx:04d}.npy",
            np.zeros((30, 21, 2), dtype=np.float32),
        )

    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(["new", "new2"]))

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    controller = AppController.__new__(AppController)
    controller._db_initialized = True
    monkeypatch.setattr(controller, "_configured_data_dir", lambda: data_root)
    monkeypatch.setattr(controller, "_configured_classes_path", lambda: classes_path)
    monkeypatch.setattr(controller_module, "get_db_session", SessionLocal)

    summary = controller.sync_dataset_to_db()

    assert summary == {"created": 1, "updated": 0, "total": 1, "samples": 3}
    with SessionLocal() as session:
        gesture = session.query(Gesture).filter_by(label="ctrlz").one()
        assert gesture.model_class_id is None
        assert gesture.samples_path.endswith("CTRLZ")
        assert session.query(GestureSample).filter_by(gesture_id=gesture.id).count() == 3

    classes_path.write_text(json.dumps(["ctrlz"]))

    second = controller.sync_dataset_to_db()

    assert second == {"created": 0, "updated": 1, "total": 1, "samples": 3}
    with SessionLocal() as session:
        assert session.query(Gesture).count() == 1
        gesture = session.query(Gesture).filter_by(label="ctrlz").one()
        assert gesture.model_class_id == 0
        assert session.query(GestureSample).filter_by(gesture_id=gesture.id).count() == 3


def test_list_recorded_gestures_hides_empty_folders(monkeypatch, tmp_path):
    data_root = tmp_path / "gestures"
    empty_dir = data_root / "EMPTY"
    full_dir = data_root / "FULL"
    empty_dir.mkdir(parents=True)
    full_dir.mkdir(parents=True)
    (empty_dir / "samples_log.jsonl").write_text("{}\n")
    np.save(full_dir / "sample_0000.npy", np.zeros((30, 21, 2), dtype=np.float32))

    controller = AppController.__new__(AppController)
    monkeypatch.setattr(controller, "_configured_data_dir", lambda: data_root)

    assert controller.list_recorded_gestures() == [{"label": "FULL", "samples": 1}]


def test_delete_recorded_samples_removes_files_and_deactivates_gesture(monkeypatch, tmp_path):
    import app.flet_app.controller as controller_module

    data_root = tmp_path / "gestures"
    label_dir = data_root / "CTRLZ"
    label_dir.mkdir(parents=True)
    sample_paths = []
    for idx in range(2):
        sample_path = label_dir / f"sample_{idx:04d}.npy"
        np.save(sample_path, np.zeros((30, 21, 2), dtype=np.float32))
        sample_paths.append(sample_path)

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        gesture = Gesture(
            label="ctrlz",
            samples_path=str(label_dir),
            model_class_id=0,
            is_active=True,
        )
        session.add(gesture)
        session.flush()
        for idx, sample_path in enumerate(sample_paths):
            session.add(
                GestureSample(
                    gesture_id=gesture.id,
                    sample_index=idx,
                    features_path=str(sample_path),
                    frames=30,
                    hand_count=1,
                )
            )
        session.add(Command(name="Undo", platform="macos", gesture_id=gesture.id))
        session.commit()

    controller = AppController.__new__(AppController)
    controller._db_initialized = True
    monkeypatch.setattr(controller, "_configured_data_dir", lambda: data_root)
    monkeypatch.setattr(controller_module, "get_db_session", SessionLocal)

    summary = controller.delete_recorded_samples("CTRLZ")

    assert summary["ok"] is True
    assert summary["filesDeleted"] == 2
    assert summary["sampleRowsDeleted"] == 2
    assert summary["commandsUnbound"] == 1
    assert not label_dir.exists()
    with SessionLocal() as session:
        gesture = session.query(Gesture).filter_by(label="ctrlz").one()
        assert gesture.is_active is False
        assert gesture.model_class_id is None
        assert session.query(GestureSample).count() == 0
        assert session.query(Command).filter_by(name="Undo").one().gesture_id is None


def test_delete_db_command_removes_history_and_executor_entry(monkeypatch):
    import app.flet_app.controller as controller_module

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        gesture = Gesture(label="pinch", model_class_id=0, is_active=True)
        session.add(gesture)
        session.flush()
        command = Command(
            name="Pinch Open",
            platform="macos",
            gesture_id=gesture.id,
            action_spec=json.dumps({"action": "open_app", "app": "Safari"}),
        )
        session.add(command)
        session.flush()
        command_id = command.id
        session.add(GestureHistory(gesture_id=gesture.id, command_id=command.id))
        session.commit()

    class FakeExecutor:
        def __init__(self):
            self.commands_registry = {"pinch open": {"action": "open_app"}}

    controller = AppController.__new__(AppController)
    controller._db_initialized = True
    controller._command_executor = FakeExecutor()
    monkeypatch.setattr(controller_module, "get_db_session", SessionLocal)
    monkeypatch.setattr(controller_module, "sync_db_commands_to_executor", lambda *_args: 0)

    summary = controller.delete_db_command(command_id)

    assert summary["ok"] is True
    assert summary["deleted"] == 1
    assert summary["historyDeleted"] == 1
    assert "pinch open" not in controller._command_executor.commands_registry
    with SessionLocal() as session:
        assert session.query(Command).count() == 0
        assert session.query(GestureHistory).count() == 0


def test_toggle_recognition_stops_camera_when_recognizing_flag_is_stale():
    controller = AppController.__new__(AppController)
    controller._is_recognizing = False
    controller._embedded_active = False
    controller._is_camera_active = True
    controller.recognizing_changed = _Event()
    calls = []

    controller._is_recognition_pid_active = lambda: False
    controller.stop_embedded_recognition = lambda: calls.append("embedded")
    controller.stop_recognition = lambda: calls.append("background")

    def stop_camera():
        calls.append("camera")
        controller._is_camera_active = False

    controller.stop_camera = stop_camera
    controller.start_embedded_recognition = lambda: calls.append("start")

    controller.toggle_recognition()

    assert calls == ["embedded", "background", "camera"]
    assert controller._is_recognizing is False
    assert controller._is_camera_active is False


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
