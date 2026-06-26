# -*- coding: utf-8 -*-

import json
import threading

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.flet_app.controller import (
    AppController,
    DYNAMIC_MODEL_PROFILE_EXTRA_TREES,
    DYNAMIC_RECOGNITION_WINDOW,
    GESTURE_CONFIRM_FRAMES,
    _Event,
)
from app.models.database import Base, Command, Gesture, GestureHistory, GestureSample


def _dispatch_controller():
    controller = AppController.__new__(AppController)
    controller._confidence = 0.0
    controller._landmarks_json = "[]"
    controller._last_label = ""
    controller._pending_label = ""
    controller._pending_frames = 0
    controller._pending_confidence_total = 0.0
    controller._live_evaluation = None
    controller._last_live_evaluation_snapshot = None
    controller._live_evaluation_lock = threading.RLock()
    controller._gesture_mode = True
    controller._pointer_mode = False
    controller._show_landmark_overlay = True
    controller._auto_execute_on_gesture = True
    controller._status = "Idle"
    controller.confidence_changed = _Event()
    controller.landmarks_changed = _Event()
    controller.gesture_detected = _Event()
    controller.gesture_mode_changed = _Event()
    controller.status_changed = _Event()
    controller.dynamic_model_profile_changed = _Event()
    controller.live_evaluation_changed = _Event()
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


def test_build_training_command_can_write_dynamic_model_metadata(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)

    monkeypatch.setattr(controller, "_active_training_labels_from_db", lambda: ["swipe_right"])
    monkeypatch.setattr(controller, "_configured_classes_path", lambda: tmp_path / "classes.json")
    monkeypatch.setattr(controller, "_configured_feature_dim_path", lambda: tmp_path / "feature_dim.txt")

    cmd = controller._build_training_command(
        data_root=str(tmp_path / "gestures"),
        out_path=str(tmp_path / "dynamic_knn.pkl"),
        neighbors=5,
        feature_mode="dynamic_stats",
        model_type="extra_trees",
        classes_out_path=str(tmp_path / "dynamic_classes.json"),
        feature_dim_out_path=str(tmp_path / "dynamic_feature_dim.txt"),
        feature_mode_out_path=str(tmp_path / "dynamic_feature_mode.txt"),
    )

    assert cmd[cmd.index("--feature-mode") + 1] == "dynamic_stats"
    assert cmd[cmd.index("--model-type") + 1] == "extra_trees"
    assert cmd[cmd.index("--classes-out") + 1].endswith("dynamic_classes.json")
    assert cmd[cmd.index("--feature-dim-out") + 1].endswith("dynamic_feature_dim.txt")
    assert cmd[cmd.index("--feature-mode-out") + 1].endswith("dynamic_feature_mode.txt")


def test_embedded_model_paths_switch_to_dynamic(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)
    controller._recognition_model_mode = "static"
    model_dir = tmp_path / "models"

    monkeypatch.setattr(controller, "_configured_models_dir", lambda: model_dir)
    monkeypatch.setattr(controller, "_configured_model_path", lambda: model_dir / "knn.pkl")
    monkeypatch.setattr(controller, "_configured_classes_path", lambda: model_dir / "classes.json")
    monkeypatch.setattr(controller, "_configured_feature_dim_path", lambda: model_dir / "feature_dim.txt")

    static_paths = controller._embedded_model_paths()
    static_window = controller._embedded_recognition_window()
    controller._recognition_model_mode = "dynamic"
    dynamic_paths = controller._embedded_model_paths()
    dynamic_window = controller._embedded_recognition_window()

    assert [path.name for path in static_paths] == [
        "knn.pkl",
        "classes.json",
        "feature_dim.txt",
        "feature_mode.txt",
    ]
    assert [path.name for path in dynamic_paths] == [
        "dynamic_knn.pkl",
        "dynamic_classes.json",
        "dynamic_feature_dim.txt",
        "dynamic_feature_mode.txt",
    ]
    assert static_window == 30
    assert dynamic_window == DYNAMIC_RECOGNITION_WINDOW


def test_embedded_dynamic_model_path_uses_selected_profile(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)
    controller._recognition_model_mode = "dynamic"
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_EXTRA_TREES
    model_dir = tmp_path / "models"

    monkeypatch.setattr(controller, "_configured_models_dir", lambda: model_dir)

    dynamic_paths = controller._embedded_model_paths()

    assert [path.name for path in dynamic_paths] == [
        "dynamic_extra_trees.pkl",
        "dynamic_classes.json",
        "dynamic_feature_dim.txt",
        "dynamic_feature_mode.txt",
    ]


def test_set_dynamic_model_profile_restarts_embedded_infer():
    controller = _dispatch_controller()
    controller._recognition_model_mode = "dynamic"
    controller._dynamic_model_profile = "knn"
    controller._embedded_active = True
    closed = []
    emitted = []

    class FakeInfer:
        def close(self):
            closed.append(True)

    controller._embedded_infer = FakeInfer()
    controller.dynamic_model_profile_changed.connect(emitted.append)

    controller.set_dynamic_model_profile("extra_trees")

    assert controller.dynamic_model_profile == "extra_trees"
    assert controller._embedded_infer is None
    assert closed == [True]
    assert emitted == ["extra_trees"]
    assert "dynamic:extra_trees" in controller.status


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


def test_start_recording_uses_embedded_camera_session(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)
    controller._training_proc = None
    controller._sample_recording = None
    controller._sample_recording_detector = None
    controller._sample_recording_lock = threading.Lock()
    controller._sample_recording_detector_lock = threading.RLock()
    controller._is_camera_active = False
    controller._status = "Idle"
    controller.status_changed = _Event()
    calls = []
    lines = []

    monkeypatch.setattr(controller, "_configured_data_dir", lambda: tmp_path / "gestures")
    monkeypatch.setattr(controller, "stop_recognition", lambda: calls.append("background"))
    monkeypatch.setattr(controller, "stop_embedded_recognition", lambda: calls.append("embedded"))

    def fake_start_camera():
        calls.append("camera")
        controller._is_camera_active = True

    monkeypatch.setattr(controller, "start_camera", fake_start_camera)

    ok = controller.start_recording(
        label="Wave",
        num_samples=3,
        frames=7,
        two_hands=True,
        on_line=lines.append,
    )

    assert ok is True
    assert calls == ["background", "embedded", "camera"]
    assert controller._sample_recording is not None
    assert controller._sample_recording["label"] == "Wave"
    assert controller._sample_recording["target"] == 3
    assert controller._sample_recording["frames"] == 7
    assert controller._sample_recording["two_hands"] is True
    assert controller._sample_recording["include_global_motion"] is False
    assert controller._sample_recording["out_dir"] == tmp_path / "gestures" / "Wave"
    assert controller._status == "Запись жеста: Wave"
    assert any("Встроенная запись" in line for line in lines)


def test_cancel_sample_recording_closes_session_and_notifies_done():
    controller = AppController.__new__(AppController)
    controller._sample_recording_lock = threading.Lock()
    controller._sample_recording_detector_lock = threading.RLock()
    done_codes = []
    lines = []

    class FakeDetector:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    detector = FakeDetector()
    controller._sample_recording_detector = detector
    controller._sample_recording = {
        "label": "Wave",
        "on_line": lines.append,
        "on_done": done_codes.append,
    }
    controller._status = "Запись жеста: Wave"
    controller._is_camera_active = True
    controller.status_changed = _Event()

    assert controller.cancel_sample_recording() is True

    assert controller._sample_recording is None
    assert controller._sample_recording_detector is None
    assert detector.closed is True
    assert done_codes == [130]
    assert lines == ["[i] Запись сэмплов остановлена"]
    assert controller._status == "Camera: streaming"


def test_process_sample_recording_frame_saves_npy(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)
    controller._sample_recording_lock = threading.Lock()
    controller._sample_recording_detector_lock = threading.RLock()
    label_dir = tmp_path / "gestures" / "Wave"
    lines = []
    done_codes = []

    class FakeHand:
        landmarks = [(float(i) / 20.0, float(i % 5) / 5.0) for i in range(21)]

    class FakeDetector:
        def __init__(self):
            self.closed = False

        def detect_for_video_rgb(self, _rgb):
            return [FakeHand()]

        def close(self):
            self.closed = True

    detector = FakeDetector()
    controller._sample_recording_detector = detector
    controller._sample_recording = {
        "label": "Wave",
        "target": 1,
        "frames": 2,
        "two_hands": False,
        "saved": 0,
        "frames_buf": [],
        "out_dir": label_dir,
        "on_line": lines.append,
        "on_done": done_codes.append,
        "next_allowed_at": 0.0,
        "last_no_hand_log": 0.0,
    }
    controller._status = "Запись жеста: Wave"
    controller._is_camera_active = True
    controller.status_changed = _Event()
    monkeypatch.setattr(
        controller,
        "sync_dataset_to_db",
        lambda: {"created": 1, "updated": 0, "total": 1, "samples": 1},
    )

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    controller._process_sample_recording_frame(frame)
    controller._process_sample_recording_frame(frame)

    sample = label_dir / "sample_0000.npy"
    assert sample.exists()
    assert np.load(sample).shape == (2, 21, 2)
    assert controller._sample_recording is None
    assert controller._sample_recording_detector is None
    assert detector.closed is True
    assert done_codes == [0]
    assert any("Сохранено" in line for line in lines)


def test_process_dynamic_sample_recording_frame_saves_global_motion_features(
    monkeypatch,
    tmp_path,
):
    controller = AppController.__new__(AppController)
    controller._sample_recording_lock = threading.Lock()
    controller._sample_recording_detector_lock = threading.RLock()
    label_dir = tmp_path / "gestures" / "SwipeLeft"
    done_codes = []

    class FakeHand:
        landmarks = [(float(i) / 20.0, float(i % 5) / 5.0) for i in range(21)]

    class FakeDetector:
        def detect_for_video_rgb(self, _rgb):
            return [FakeHand()]

        def close(self):
            pass

    controller._sample_recording_detector = FakeDetector()
    controller._sample_recording = {
        "label": "SwipeLeft",
        "target": 1,
        "frames": 2,
        "two_hands": False,
        "include_global_motion": True,
        "saved": 0,
        "frames_buf": [],
        "out_dir": label_dir,
        "on_line": None,
        "on_done": done_codes.append,
        "next_allowed_at": 0.0,
        "last_no_hand_log": 0.0,
    }
    controller._status = "Запись жеста: SwipeLeft"
    controller._is_camera_active = True
    controller.status_changed = _Event()
    monkeypatch.setattr(
        controller,
        "sync_dataset_to_db",
        lambda: {"created": 1, "updated": 0, "total": 1, "samples": 1},
    )

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    controller._process_sample_recording_frame(frame)
    controller._process_sample_recording_frame(frame)

    sample = np.load(label_dir / "sample_0000.npy")
    assert sample.shape == (2, 44)
    assert np.allclose(sample[0, -2:], [0.0, 0.0])
    assert done_codes == [0]


def test_dynamic_sample_quality_report_accepts_expected_motion():
    controller = AppController.__new__(AppController)
    sample = np.zeros((36, 44), dtype=np.float32)
    movement = np.linspace(0.0, -0.8, sample.shape[0], dtype=np.float32)
    sample[:] = movement[:, None]

    report = controller._sample_quality_report(
        sample,
        label="swipe_up",
        include_global_motion=True,
    )

    assert report["ok"] is True
    assert report["dy"] < -0.05
    assert report["warnings"] == []


def test_dynamic_sample_quality_report_flags_wrong_direction():
    controller = AppController.__new__(AppController)
    sample = np.zeros((36, 44), dtype=np.float32)
    movement = np.linspace(0.0, 0.8, sample.shape[0], dtype=np.float32)
    sample[:] = movement[:, None]

    report = controller._sample_quality_report(
        sample,
        label="hand_left",
        include_global_motion=True,
    )

    assert report["ok"] is False
    assert "expected_left" in report["warnings"]
    assert report["dx"] > 0.05


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


def test_toggle_recognition_starts_cv_when_only_camera_is_active():
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

    assert calls == ["start"]
    assert controller._is_recognizing is False
    assert controller._is_camera_active is True


def test_start_embedded_recognition_keeps_recognizing_enabled_after_background_stop():
    controller = AppController.__new__(AppController)
    controller._embedded_active = False
    controller._embedded_infer = object()
    controller._is_recognizing = True
    controller._is_camera_active = True
    controller._status = "Idle"
    controller._last_label = "old"
    controller._landmarks_json = "[[]]"
    controller._confidence = 0.5
    controller._pending_label = "old"
    controller._pending_frames = 2
    controller._pending_confidence_total = 1.0
    controller.status_changed = _Event()
    controller.recognizing_changed = _Event()
    controller.confidence_changed = _Event()
    controller.landmarks_changed = _Event()

    def fake_stop_background():
        controller._set_recognizing(False)

    controller.stop_recognition = fake_stop_background
    controller.start_camera = lambda: None

    controller.start_embedded_recognition()

    assert controller._embedded_active is True
    assert controller._is_recognizing is True
    assert controller._embedded_infer is None
    assert controller._last_label == ""
    assert controller._pending_frames == 0


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


def test_dynamic_dispatch_confirms_gesture_faster():
    controller = _dispatch_controller()
    controller._recognition_model_mode = "dynamic"
    emitted = []
    executed = []
    recorded = []
    controller.gesture_detected.connect(emitted.append)
    controller.execute_for_gesture = lambda label, conf: executed.append((label, conf)) or True
    controller._record_recognition_event = lambda label, conf, ok: recorded.append((label, conf, ok))

    controller._dispatch_infer_result(
        {"label": "hand_left", "confidence": 0.9, "landmarks_json": "[]"}
    )

    assert emitted == []
    assert executed == []

    controller._dispatch_infer_result(
        {"label": "hand_left", "confidence": 0.8, "landmarks_json": "[]"}
    )

    assert emitted == ["hand_left"]
    assert executed == [("hand_left", pytest.approx(0.85))]
    assert recorded == [("hand_left", pytest.approx(0.85), True)]


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


def test_live_evaluation_counts_correct_wrong_and_missed(monkeypatch, tmp_path):
    controller = _dispatch_controller()
    controller._recognition_model_mode = "dynamic"
    controller._dynamic_model_profile = "knn"
    controller._ensure_embedded_recognition_for_live_controls = lambda: None
    monkeypatch.setattr(controller, "_configured_log_dir", lambda: tmp_path)

    assert controller.start_live_evaluation(
        "swipe_down",
        attempts=3,
        timeout_seconds=0.0,
        min_confidence=0.6,
    )
    controller._live_evaluation["attempt_started_at"] = 10.0
    controller._live_evaluation["next_ready_at"] = 10.0

    controller._consume_live_evaluation_prediction("swipe_down", 0.9, now=10.0)
    controller._consume_live_evaluation_prediction("swipe_left", 0.8, now=12.0)
    assert controller.mark_live_evaluation_missed() is True

    snapshot = controller.current_live_evaluation()
    assert snapshot["active"] is False
    assert snapshot["correct"] == 1
    assert snapshot["wrong"] == 1
    assert snapshot["missed"] == 1
    assert snapshot["total"] == 3
    assert snapshot["accuracy"] == pytest.approx(1 / 3)
    assert (tmp_path / "live_evaluation.jsonl").exists()
    rows = [
        json.loads(line)
        for line in (tmp_path / "live_evaluation.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["event_type"] == "run_completed"
    assert rows[-1]["recognition_model_mode"] == "dynamic"
    assert rows[-1]["dynamic_model_profile"] == "knn"


def test_live_evaluation_ignores_below_threshold_without_default_timeout(
    monkeypatch,
    tmp_path,
):
    controller = _dispatch_controller()
    controller._ensure_embedded_recognition_for_live_controls = lambda: None
    monkeypatch.setattr(controller, "_configured_log_dir", lambda: tmp_path)

    assert controller.start_live_evaluation(
        "swipe_down",
        attempts=1,
        timeout_seconds=0.0,
        min_confidence=0.8,
    )
    controller._live_evaluation["attempt_started_at"] = 20.0
    controller._live_evaluation["next_ready_at"] = 20.0

    controller._consume_live_evaluation_prediction("swipe_down", 0.7, now=20.0)
    snapshot = controller.current_live_evaluation()
    assert snapshot["active"] is True
    assert snapshot["total"] == 0
    assert snapshot["lastResult"] == "below_threshold"

    controller._update_live_evaluation_timeout(now=22.0)
    snapshot = controller.current_live_evaluation()
    assert snapshot["active"] is True
    assert snapshot["missed"] == 0

    assert controller.mark_live_evaluation_missed() is True
    snapshot = controller.current_live_evaluation()
    assert snapshot["active"] is False
    assert snapshot["missed"] == 1


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
