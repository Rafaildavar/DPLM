# -*- coding: utf-8 -*-

import csv
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.flet_app.controller import (
    AUTO_STATIC_GESTURE_CONFIRM_FRAMES,
    AppController,
    DYNAMIC_MODEL_PROFILE_PRODUCTION,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_ENSEMBLE,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_GRU_BACKBONE,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_LSTM_BACKBONE,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET,
    DYNAMIC_GESTURE_CONFIRM_FRAMES,
    DYNAMIC_RECOGNITION_LONG_WINDOW,
    DYNAMIC_RECOGNITION_WINDOW,
    GESTURE_CONFIRM_FRAMES,
    LIVE_EVAL_NO_COMMAND_LABEL,
    RECOGNITION_MODEL_AUTO,
    _Event,
)
from app.models.database import Base, Command, Gesture, GestureHistory, GestureSample
from app.services.app_config import AppConfig, ConfigStore
from app.services.live_gesture_state import LiveGestureState


def _dispatch_controller():
    controller = AppController.__new__(AppController)
    controller._confidence = 0.0
    controller._landmarks_json = "[]"
    controller._last_label = ""
    controller._pending_label = ""
    controller._pending_frames = 0
    controller._pending_confidence_total = 0.0
    controller._live_gesture_state = LiveGestureState()
    controller._last_live_gesture_state_payload = None
    controller._live_gesture_inspector_history = []
    controller._live_gesture_inspector_sequence = 0
    controller._last_execute_info = ""
    controller._dynamic_return_guard = {}
    controller._live_evaluation = None
    controller._last_live_evaluation_snapshot = None
    controller._live_evaluation_lock = threading.RLock()
    controller._gesture_mode = True
    controller._pointer_mode = False
    controller._pointer_control = None
    controller._show_landmark_overlay = True
    controller._recognition_model_mode = "static"
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_PRODUCTION
    controller._pointer_state_payload = {
        "enabled": False,
        "state": "idle",
        "moved": False,
        "clicked": False,
        "tabSwitched": "",
        "error": "",
    }
    controller._auto_execute_on_gesture = True
    controller._status = "Idle"
    controller.confidence_changed = _Event()
    controller.gesture_state_changed = _Event()
    controller.landmarks_changed = _Event()
    controller.gesture_detected = _Event()
    controller.gesture_mode_changed = _Event()
    controller.pointer_mode_changed = _Event()
    controller.pointer_state_changed = _Event()
    controller.status_changed = _Event()
    controller.dynamic_model_profile_changed = _Event()
    controller.recognition_model_mode_changed = _Event()
    controller.live_evaluation_changed = _Event()
    controller._update_pointer_from_landmarks = lambda _landmarks: None
    return controller


def test_runtime_performance_writes_aggregated_jsonl(monkeypatch, tmp_path):
    controller = _dispatch_controller()
    controller._recognition_model_mode = "auto"
    controller._target_fps = 30
    controller._runtime_inference_samples = []
    controller._runtime_perf_last_flush = 0.0
    monkeypatch.setattr(controller, "_configured_log_dir", lambda: tmp_path)

    controller._record_runtime_performance(
        {
            "total_inference_ms": 20.0,
            "detection_ms": 12.0,
            "shared_detection": True,
        }
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "runtime_performance.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["recognition_model_mode"] == "auto"
    assert rows[-1]["shared_detection_rate"] == 1.0
    assert rows[-1]["inference_ms_avg"] == 20.0
    assert rows[-1]["inference_fps_capacity"] == 50.0
    summary = json.loads((tmp_path / "live_usage_summary.json").read_text())
    assert summary["latest_runtime"]["inference_ms_avg"] == 20.0


def test_live_usage_event_writes_jsonl_and_summary(monkeypatch, tmp_path):
    controller = _dispatch_controller()
    controller._recognition_model_mode = RECOGNITION_MODEL_AUTO
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_LSTM_BACKBONE
    controller._static_rejection_method = "open_set_policy"
    monkeypatch.setattr(controller, "_configured_log_dir", lambda: tmp_path)

    controller._record_live_usage_event(
        "command_executed",
        "zoom",
        0.96,
        executed=True,
        route_metadata={
            "route": "dynamic",
            "selected_reason": "dynamic_accepted",
        },
        command_info="zoom: Open App",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "live_usage_events.jsonl").read_text().splitlines()
    ]
    summary = json.loads((tmp_path / "live_usage_summary.json").read_text())
    assert rows[-1]["label"] == "zoom"
    assert rows[-1]["event_type"] == "command_executed"
    assert rows[-1]["route"] == "dynamic"
    assert summary["total_events"] == 1
    assert summary["command_attempts"] == 1
    assert summary["executed_events"] == 1
    assert summary["command_success_rate"] == 1.0
    assert summary["label_counts"]["zoom"] == 1


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

    assert "--lowercase-labels" not in cmd
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
        out_path=str(tmp_path / "dynamic_sequence_mlp.pkl"),
        neighbors=5,
        feature_mode="dynamic_sequence",
        model_type="sequence_mlp",
        classes_out_path=str(tmp_path / "dynamic_sequence_mlp_classes.json"),
        feature_dim_out_path=str(tmp_path / "dynamic_sequence_mlp_feature_dim.txt"),
        feature_mode_out_path=str(tmp_path / "dynamic_sequence_mlp_feature_mode.txt"),
    )

    assert cmd[cmd.index("--feature-mode") + 1] == "dynamic_sequence"
    assert cmd[cmd.index("--model-type") + 1] == "sequence_mlp"
    assert cmd[cmd.index("--classes-out") + 1].endswith(
        "dynamic_sequence_mlp_classes.json"
    )
    assert cmd[cmd.index("--feature-dim-out") + 1].endswith(
        "dynamic_sequence_mlp_feature_dim.txt"
    )
    assert cmd[cmd.index("--feature-mode-out") + 1].endswith(
        "dynamic_sequence_mlp_feature_mode.txt"
    )


def test_build_training_command_appends_model_extra_args(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)

    monkeypatch.setattr(controller, "_active_training_labels_from_db", lambda: ["swipe_up"])
    monkeypatch.setattr(controller, "_configured_classes_path", lambda: tmp_path / "classes.json")
    monkeypatch.setattr(controller, "_configured_feature_dim_path", lambda: tmp_path / "feature_dim.txt")

    cmd = controller._build_training_command(
        data_root=str(tmp_path / "gestures"),
        out_path=str(tmp_path / "dynamic_sequence_gru_backbone.pkl"),
        feature_mode="dynamic_sequence",
        model_type="sequence_gru_backbone",
        extra_args=["--sequence-gru-optuna-trials", "3"],
    )

    assert cmd[cmd.index("--model-type") + 1] == "sequence_gru_backbone"
    assert cmd[-2:] == ["--sequence-gru-optuna-trials", "3"]


def test_build_training_command_filters_dynamic_scope(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)

    monkeypatch.setattr(
        controller,
        "_active_training_labels_from_db",
        lambda: ["palm", "swipe_down", "UP", "swipe_left", "hand_left"],
    )
    monkeypatch.setattr(controller, "_configured_classes_path", lambda: tmp_path / "classes.json")
    monkeypatch.setattr(controller, "_configured_feature_dim_path", lambda: tmp_path / "feature_dim.txt")

    cmd = controller._build_training_command(
        data_root=str(tmp_path / "gestures"),
        out_path=str(tmp_path / "dynamic_sequence_mlp.pkl"),
        feature_mode="dynamic_sequence",
        classes_out_path=str(tmp_path / "dynamic_sequence_mlp_classes.json"),
        feature_dim_out_path=str(tmp_path / "dynamic_sequence_mlp_feature_dim.txt"),
        feature_mode_out_path=str(tmp_path / "dynamic_sequence_mlp_feature_mode.txt"),
        training_scope="dynamic",
    )

    include_values = [
        cmd[index + 1]
        for index, item in enumerate(cmd)
        if item == "--include-label"
    ]
    assert include_values == ["swipe_down", "swipe_left"]


def test_build_dynamic_prototype_training_command_uses_external_negatives(
    monkeypatch,
    tmp_path,
):
    controller = AppController.__new__(AppController)
    model_dir = tmp_path / "models"
    variant_dir = tmp_path / "models" / "experiments" / "prototype_distance"

    monkeypatch.setattr(controller, "_configured_data_dir", lambda: tmp_path / "gestures")
    monkeypatch.setattr(controller, "_configured_models_dir", lambda: variant_dir)
    monkeypatch.setattr(
        controller,
        "_live_evaluation_mlflow_tracking_uri",
        lambda: "sqlite:///tmp_mlflow.db",
    )

    cmd = controller._build_dynamic_prototype_training_command(
        dynamic_model_out_path=str(model_dir / "dynamic_sequence_mlp.pkl"),
    )

    assert cmd[1:4] == ["-u", "-m", "scripts.dynamic_prototype_experiments"]
    assert cmd[cmd.index("--data-root") + 1] == str(tmp_path / "gestures")
    assert cmd[cmd.index("--external-negative-root") + 1].endswith("data/external")
    assert "--include-external-negatives" in cmd
    assert cmd[cmd.index("--methods") + 1] == "prototype_distance"
    assert cmd[cmd.index("--target-frames") + 1] == "36"
    assert cmd[cmd.index("--base-models-dir") + 1] == str(model_dir)
    assert "--write-production" in cmd
    assert cmd[cmd.index("--production-out") + 1] == str(
        model_dir / "dynamic_sequence_mlp_prototypes.json"
    )
    assert cmd[cmd.index("--mlflow-tracking-uri") + 1] == "sqlite:///tmp_mlflow.db"


def test_build_dynamic_prototype_training_command_passes_active_user_labels(
    monkeypatch,
    tmp_path,
):
    controller = AppController.__new__(AppController)
    controller._db_initialized = True
    model_dir = tmp_path / "models"

    monkeypatch.setattr(controller, "_configured_data_dir", lambda: tmp_path / "gestures")
    monkeypatch.setattr(
        controller,
        "_live_evaluation_mlflow_tracking_uri",
        lambda: "sqlite:///tmp_mlflow.db",
    )
    monkeypatch.setattr(
        controller,
        "_training_labels_for_scope",
        lambda scope: ["SwipeLeft", "random_motion"]
        if scope == "dynamic,negative"
        else [],
    )

    cmd = controller._build_dynamic_prototype_training_command(
        dynamic_model_out_path=str(model_dir / "dynamic_landmark_lstm_backbone.pkl"),
    )

    include_values = [
        cmd[index + 1]
        for index, item in enumerate(cmd)
        if item == "--include-label"
    ]
    assert include_values == ["SwipeLeft", "random_motion"]


def test_build_production_landmark_prototype_training_command_uses_long_window(
    monkeypatch,
    tmp_path,
):
    controller = AppController.__new__(AppController)
    model_dir = tmp_path / "models"

    monkeypatch.setattr(controller, "_configured_data_dir", lambda: tmp_path / "gestures")
    monkeypatch.setattr(
        controller,
        "_live_evaluation_mlflow_tracking_uri",
        lambda: "sqlite:///tmp_mlflow.db",
    )

    cmd = controller._build_dynamic_prototype_training_command(
        dynamic_model_out_path=str(model_dir / "dynamic_landmark_lstm_backbone.pkl"),
    )

    assert cmd[cmd.index("--target-frames") + 1] == str(
        DYNAMIC_RECOGNITION_LONG_WINDOW
    )
    assert cmd[cmd.index("--target-dim") + 1] == "65"
    assert cmd[cmd.index("--production-out") + 1] == str(
        model_dir / "dynamic_landmark_lstm_backbone_prototypes.json"
    )


def test_build_legacy_sequence_prototype_training_command_uses_production_output(
    monkeypatch,
    tmp_path,
):
    controller = AppController.__new__(AppController)
    model_dir = tmp_path / "models"

    monkeypatch.setattr(controller, "_configured_data_dir", lambda: tmp_path / "gestures")
    monkeypatch.setattr(
        controller,
        "_live_evaluation_mlflow_tracking_uri",
        lambda: "sqlite:///tmp_mlflow.db",
    )

    cmd = controller._build_dynamic_prototype_training_command(
        dynamic_model_out_path=str(model_dir / "dynamic_sequence_knn.pkl"),
    )

    assert cmd[cmd.index("--production-out") + 1] == str(
        model_dir / "dynamic_landmark_lstm_backbone_prototypes.json"
    )


def test_build_sequence_mlp_prototype_training_command_uses_own_sequence_output(
    monkeypatch,
    tmp_path,
):
    controller = AppController.__new__(AppController)
    model_dir = tmp_path / "models"

    monkeypatch.setattr(controller, "_configured_data_dir", lambda: tmp_path / "gestures")
    monkeypatch.setattr(
        controller,
        "_live_evaluation_mlflow_tracking_uri",
        lambda: "sqlite:///tmp_mlflow.db",
    )

    cmd = controller._build_dynamic_prototype_training_command(
        dynamic_model_out_path=str(model_dir / "dynamic_sequence_mlp.pkl"),
    )

    assert cmd[cmd.index("--production-out") + 1] == str(
        model_dir / "dynamic_sequence_mlp_prototypes.json"
    )


def test_build_sequence_rocket_prototype_training_command_uses_own_sequence_output(
    monkeypatch,
    tmp_path,
):
    controller = AppController.__new__(AppController)
    model_dir = tmp_path / "models"

    monkeypatch.setattr(controller, "_configured_data_dir", lambda: tmp_path / "gestures")
    monkeypatch.setattr(
        controller,
        "_live_evaluation_mlflow_tracking_uri",
        lambda: "sqlite:///tmp_mlflow.db",
    )

    cmd = controller._build_dynamic_prototype_training_command(
        dynamic_model_out_path=str(model_dir / "dynamic_sequence_rocket.pkl"),
    )

    assert cmd[cmd.index("--production-out") + 1] == str(
        model_dir / "dynamic_sequence_rocket_prototypes.json"
    )


def test_build_sequence_ensemble_prototype_training_command_uses_own_sequence_output(
    monkeypatch,
    tmp_path,
):
    controller = AppController.__new__(AppController)
    model_dir = tmp_path / "models"

    monkeypatch.setattr(controller, "_configured_data_dir", lambda: tmp_path / "gestures")
    monkeypatch.setattr(
        controller,
        "_live_evaluation_mlflow_tracking_uri",
        lambda: "sqlite:///tmp_mlflow.db",
    )

    cmd = controller._build_dynamic_prototype_training_command(
        dynamic_model_out_path=str(model_dir / "dynamic_sequence_ensemble.pkl"),
    )

    assert cmd[cmd.index("--production-out") + 1] == str(
        model_dir / "dynamic_sequence_ensemble_prototypes.json"
    )
    assert cmd[cmd.index("--target-frames") + 1] == "36"


def test_build_sequence_shapelet_72_prototype_training_command_uses_long_target(
    monkeypatch,
    tmp_path,
):
    controller = AppController.__new__(AppController)
    model_dir = tmp_path / "models"

    monkeypatch.setattr(controller, "_configured_data_dir", lambda: tmp_path / "gestures")
    monkeypatch.setattr(
        controller,
        "_live_evaluation_mlflow_tracking_uri",
        lambda: "sqlite:///tmp_mlflow.db",
    )

    cmd = controller._build_dynamic_prototype_training_command(
        dynamic_model_out_path=str(model_dir / "dynamic_sequence_shapelet_72.pkl"),
    )

    assert cmd[cmd.index("--production-out") + 1] == str(
        model_dir / "dynamic_sequence_shapelet_72_prototypes.json"
    )
    assert cmd[cmd.index("--target-frames") + 1] == "72"


def test_dynamic_prototype_training_only_for_dynamic_scope():
    controller = AppController.__new__(AppController)

    assert controller._should_train_dynamic_prototypes("dynamic") is True
    assert controller._should_train_dynamic_prototypes("static,negative") is False


def test_build_training_command_filters_static_scope_with_negative(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)

    monkeypatch.setattr(
        controller,
        "_active_training_labels_from_db",
        lambda: ["palm", "swipe_down", "no_gesture_static", "hand_left"],
    )
    monkeypatch.setattr(controller, "_configured_classes_path", lambda: tmp_path / "classes.json")
    monkeypatch.setattr(controller, "_configured_feature_dim_path", lambda: tmp_path / "feature_dim.txt")

    cmd = controller._build_training_command(
        data_root=str(tmp_path / "gestures"),
        out_path=str(tmp_path / "knn.pkl"),
        training_scope="static,quasi_static,negative",
    )

    include_values = [
        cmd[index + 1]
        for index, item in enumerate(cmd)
        if item == "--include-label"
    ]
    assert include_values == ["palm", "no_gesture_static", "hand_left"]


def test_infer_training_gesture_type_reads_dynamic_sample_metadata(tmp_path):
    controller = AppController.__new__(AppController)
    sample_path = tmp_path / "sample_0000.npy"
    np.save(sample_path, np.zeros((12, 65), dtype=np.float32))
    sample_path.with_suffix(".meta.json").write_text(
        json.dumps({"include_global_motion": True}),
        encoding="utf-8",
    )

    assert (
        controller._infer_training_gesture_type("SwipeLeft", [sample_path])
        == "dynamic"
    )


def test_build_negative_generation_command_uses_configured_paths(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)

    monkeypatch.setattr(controller, "_configured_data_dir", lambda: tmp_path / "gestures")
    monkeypatch.setattr(
        controller,
        "_configured_taxonomy_path",
        lambda: tmp_path / "gesture_taxonomy.json",
    )

    cmd = controller._build_negative_generation_command(
        samples_per_label=12,
        seed=99,
    )

    assert cmd[1:4] == ["-u", "-m", "scripts.generate_negative_samples"]
    assert cmd[cmd.index("--data-root") + 1] == str(tmp_path / "gestures")
    assert cmd[cmd.index("--taxonomy") + 1] == str(tmp_path / "gesture_taxonomy.json")
    assert cmd[cmd.index("--samples-per-label") + 1] == "12"
    assert cmd[cmd.index("--target-frames") + 1] == str(
        DYNAMIC_RECOGNITION_LONG_WINDOW
    )
    assert cmd[cmd.index("--seed") + 1] == "99"
    assert cmd[cmd.index("--manifest-out") + 1].endswith(
        "docs/experiments/negative_sampling_manifest.json"
    )


def test_dynamic_recording_marks_label_dynamic_in_taxonomy(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)
    taxonomy_path = tmp_path / "gesture_taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(
            {
                "default_type": "static",
                "types": {
                    "static": ["circle_clockwise", "palm"],
                    "quasi_static": [],
                    "dynamic": ["swipe_left"],
                    "negative": ["random_motion"],
                },
                "patterns": {
                    "dynamic": ["swipe_*"],
                    "negative": ["random_*"],
                },
            }
        ),
        encoding="utf-8",
    )
    lines = []
    monkeypatch.setattr(controller, "_configured_taxonomy_path", lambda: taxonomy_path)

    changed = controller._ensure_dynamic_label_in_taxonomy(
        "circle_clockwise",
        on_line=lines.append,
    )

    updated = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    assert changed is True
    assert "circle_clockwise" not in updated["types"]["static"]
    assert "circle_clockwise" in updated["types"]["dynamic"]
    assert getattr(controller, "_gesture_taxonomy_cache", "not-reset") is None
    assert any("помечен как dynamic" in line for line in lines)


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
        "dynamic_landmark_lstm_backbone.pkl",
        "dynamic_landmark_lstm_backbone_classes.json",
        "dynamic_landmark_lstm_backbone_feature_dim.txt",
        "dynamic_landmark_lstm_backbone_feature_mode.txt",
    ]
    assert static_window == 30
    assert dynamic_window == DYNAMIC_RECOGNITION_LONG_WINDOW


def test_embedded_dynamic_model_path_normalizes_legacy_profile(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)
    controller._recognition_model_mode = "dynamic"
    controller._dynamic_model_profile = "extra_trees"
    model_dir = tmp_path / "models"

    monkeypatch.setattr(controller, "_configured_models_dir", lambda: model_dir)

    dynamic_paths = controller._embedded_model_paths()

    assert controller.dynamic_model_profile == DYNAMIC_MODEL_PROFILE_PRODUCTION
    assert [path.name for path in dynamic_paths] == [
        "dynamic_landmark_lstm_backbone.pkl",
        "dynamic_landmark_lstm_backbone_classes.json",
        "dynamic_landmark_lstm_backbone_feature_dim.txt",
        "dynamic_landmark_lstm_backbone_feature_mode.txt",
    ]


def test_embedded_dynamic_model_paths_can_use_sequence_rocket(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)
    controller._recognition_model_mode = "dynamic"
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET
    model_dir = tmp_path / "models"

    monkeypatch.setattr(controller, "_configured_models_dir", lambda: model_dir)

    dynamic_paths = controller._embedded_model_paths()

    assert controller.dynamic_model_profile == DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET
    assert [path.name for path in dynamic_paths] == [
        "dynamic_sequence_rocket.pkl",
        "dynamic_sequence_rocket_classes.json",
        "dynamic_sequence_rocket_feature_dim.txt",
        "dynamic_sequence_rocket_feature_mode.txt",
    ]


def test_embedded_dynamic_model_paths_can_use_sequence_ensemble(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)
    controller._recognition_model_mode = "dynamic"
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_ENSEMBLE
    model_dir = tmp_path / "models"

    monkeypatch.setattr(controller, "_configured_models_dir", lambda: model_dir)

    dynamic_paths = controller._embedded_model_paths()

    assert controller.dynamic_model_profile == DYNAMIC_MODEL_PROFILE_SEQUENCE_ENSEMBLE
    assert [path.name for path in dynamic_paths] == [
        "dynamic_sequence_ensemble.pkl",
        "dynamic_sequence_ensemble_classes.json",
        "dynamic_sequence_ensemble_feature_dim.txt",
        "dynamic_sequence_ensemble_feature_mode.txt",
    ]


def test_embedded_dynamic_model_paths_can_use_sequence_gru_backbone(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)
    controller._recognition_model_mode = "dynamic"
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_GRU_BACKBONE
    model_dir = tmp_path / "models"

    monkeypatch.setattr(controller, "_configured_models_dir", lambda: model_dir)

    dynamic_paths = controller._embedded_model_paths()

    assert controller.dynamic_model_profile == DYNAMIC_MODEL_PROFILE_SEQUENCE_GRU_BACKBONE
    assert [path.name for path in dynamic_paths] == [
        "dynamic_sequence_gru_backbone.pkl",
        "dynamic_sequence_gru_backbone_classes.json",
        "dynamic_sequence_gru_backbone_feature_dim.txt",
        "dynamic_sequence_gru_backbone_feature_mode.txt",
    ]


def test_embedded_dynamic_model_paths_can_use_sequence_lstm_backbone(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)
    controller._recognition_model_mode = "dynamic"
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_LSTM_BACKBONE
    model_dir = tmp_path / "models"

    monkeypatch.setattr(controller, "_configured_models_dir", lambda: model_dir)

    dynamic_paths = controller._embedded_model_paths()

    assert controller.dynamic_model_profile == DYNAMIC_MODEL_PROFILE_SEQUENCE_LSTM_BACKBONE
    assert [path.name for path in dynamic_paths] == [
        "dynamic_sequence_lstm_backbone.pkl",
        "dynamic_sequence_lstm_backbone_classes.json",
        "dynamic_sequence_lstm_backbone_feature_dim.txt",
        "dynamic_sequence_lstm_backbone_feature_mode.txt",
    ]


def test_apply_model_variant_updates_config_and_resets_infer(monkeypatch, tmp_path):
    for key in (
        "DPLM_MODELS_DIR",
        "DPLM_MODEL_PATH",
        "DPLM_CLASSES_PATH",
        "DPLM_FEATURE_DIM_PATH",
    ):
        monkeypatch.delenv(key, raising=False)

    controller = _dispatch_controller()
    store = ConfigStore(tmp_path / "config.json")
    config = AppConfig()
    store.save(config)
    controller._config_store = store
    controller._config_file = config
    controller._config = config
    controller._apply_runtime_config = lambda: None
    controller._reset_db_bridge = lambda: None
    controller._embedded_active = False
    controller.model_variant_changed = _Event()
    emitted = []
    closed = []

    class FakeInfer:
        def close(self):
            closed.append(True)

    controller._embedded_infer = FakeInfer()
    controller.model_variant_changed.connect(emitted.append)
    monkeypatch.setenv("DPLM_MODELS_DIR", "models/old")

    ok, errors, _warnings = controller.apply_model_variant("ipn_external")

    assert ok is True
    assert errors == []
    saved = store.load(include_env=False)
    expected_dir = "models/experiments/external_negative/ipn_external"
    assert saved.paths.models_dir == expected_dir
    assert saved.paths.model_path == f"{expected_dir}/knn.pkl"
    assert saved.paths.classes_path == f"{expected_dir}/classes.json"
    assert saved.paths.feature_dim_path == f"{expected_dir}/feature_dim.txt"
    assert controller._embedded_infer is None
    assert closed == [True]
    assert emitted == ["ipn_external"]
    assert controller.model_variant == "ipn_external"
    assert "DPLM_MODELS_DIR" not in os.environ


def test_set_recognition_model_mode_accepts_auto_and_falls_back_to_auto():
    controller = _dispatch_controller()
    controller._recognition_model_mode = "static"
    controller._embedded_active = False
    controller._embedded_infer = None
    controller._is_recognizing = False
    controller._is_camera_active = False
    emitted = []
    controller.recognition_model_mode_changed.connect(emitted.append)

    controller.set_recognition_model_mode("auto")

    assert controller.recognition_model_mode == RECOGNITION_MODEL_AUTO
    assert emitted == [RECOGNITION_MODEL_AUTO]

    controller.set_recognition_model_mode("unknown")

    assert controller.recognition_model_mode == RECOGNITION_MODEL_AUTO
    assert emitted == [RECOGNITION_MODEL_AUTO]


def test_set_dynamic_model_profile_normalizes_legacy_profile():
    controller = _dispatch_controller()
    controller._recognition_model_mode = "dynamic"
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_PRODUCTION
    controller._embedded_active = True
    closed = []
    emitted = []

    class FakeInfer:
        def close(self):
            closed.append(True)

    infer = FakeInfer()
    controller._embedded_infer = infer
    controller.dynamic_model_profile_changed.connect(emitted.append)

    controller.set_dynamic_model_profile("extra_trees")

    assert controller.dynamic_model_profile == DYNAMIC_MODEL_PROFILE_PRODUCTION
    assert controller._embedded_infer is infer
    assert closed == []
    assert emitted == []


def test_set_dynamic_model_profile_accepts_sequence_rocket_and_resets_infer():
    controller = _dispatch_controller()
    controller._recognition_model_mode = "dynamic"
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_PRODUCTION
    controller._embedded_active = True
    closed = []
    emitted = []

    class FakeInfer:
        def close(self):
            closed.append(True)

    controller._embedded_infer = FakeInfer()
    controller.dynamic_model_profile_changed.connect(emitted.append)

    controller.set_dynamic_model_profile(DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET)

    assert controller.dynamic_model_profile == DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET
    assert controller._embedded_infer is None
    assert closed == [True]
    assert emitted == [DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET]


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
        gesture = session.query(Gesture).filter_by(label="CTRLZ").one()
        assert gesture.model_class_id is None
        assert gesture.samples_path.endswith("CTRLZ")
        samples = session.query(GestureSample).filter_by(gesture_id=gesture.id).all()
        assert len(samples) == 3
        assert {sample.source for sample in samples} == {"user"}
        assert gesture.description in {None, ""}

    classes_path.write_text(json.dumps(["ctrlz"]))

    second = controller.sync_dataset_to_db()

    assert second == {"created": 0, "updated": 1, "total": 1, "samples": 3}
    with SessionLocal() as session:
        assert session.query(Gesture).count() == 1
        gesture = session.query(Gesture).filter_by(label="ctrlz").one()
        assert gesture.model_class_id == 0
        assert session.query(GestureSample).filter_by(gesture_id=gesture.id).count() == 3


def test_get_db_gestures_shows_recorded_untrained_samples(monkeypatch, tmp_path):
    import app.flet_app.controller as controller_module

    data_root = tmp_path / "gestures"
    label_dir = data_root / "Circle"
    label_dir.mkdir(parents=True)
    sample_path = label_dir / "sample_0000.npy"
    np.save(sample_path, np.zeros((30, 21, 2), dtype=np.float32))

    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps([]), encoding="utf-8")
    dynamic_classes_path = tmp_path / "dynamic_classes.json"
    dynamic_classes_path.write_text(json.dumps([]), encoding="utf-8")

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        gesture = Gesture(
            label="circle",
            samples_path=str(label_dir),
            model_class_id=None,
            is_active=True,
        )
        session.add(gesture)
        session.flush()
        session.add(
            GestureSample(
                gesture_id=gesture.id,
                sample_index=0,
                features_path=str(sample_path),
                frames=30,
                hand_count=1,
            )
        )
        session.commit()

    controller = AppController.__new__(AppController)
    controller._db_initialized = True
    monkeypatch.setattr(controller, "_configured_classes_path", lambda: classes_path)
    monkeypatch.setattr(controller, "_dynamic_classes_path", lambda: dynamic_classes_path)
    monkeypatch.setattr(controller, "_configured_path", lambda value: Path(value))
    monkeypatch.setattr(controller_module, "get_db_session", SessionLocal)

    rows = controller.get_db_gestures()

    assert [row["label"] for row in rows] == ["circle"]
    assert rows[0]["sampleCount"] == 1


def test_get_db_gestures_hides_auto_imported_negative_dataset_classes(
    monkeypatch,
    tmp_path,
):
    import app.flet_app.controller as controller_module

    data_root = tmp_path / "gestures"
    label_dir = data_root / "wrong_axis_motion"
    label_dir.mkdir(parents=True)
    sample_path = label_dir / "sample_0000.npy"
    np.save(sample_path, np.zeros((30, 44), dtype=np.float32))

    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(["wrong_axis_motion"]), encoding="utf-8")
    dynamic_classes_path = tmp_path / "dynamic_classes.json"
    dynamic_classes_path.write_text(json.dumps(["wrong_axis_motion"]), encoding="utf-8")

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        gesture = Gesture(
            label="wrong_axis_motion",
            description=f"Auto-imported from {label_dir}",
            samples_path=str(label_dir),
            model_class_id=0,
            is_active=True,
        )
        session.add(gesture)
        session.flush()
        session.add(
            GestureSample(
                gesture_id=gesture.id,
                sample_index=0,
                features_path=str(sample_path),
                frames=30,
                hand_count=1,
                source="dataset",
            )
        )
        session.commit()

    controller = AppController.__new__(AppController)
    controller._db_initialized = True
    monkeypatch.setattr(controller, "_configured_classes_path", lambda: classes_path)
    monkeypatch.setattr(controller, "_dynamic_classes_path", lambda: dynamic_classes_path)
    monkeypatch.setattr(controller, "_configured_path", lambda value: Path(value))
    monkeypatch.setattr(controller_module, "get_db_session", SessionLocal)

    assert controller.get_db_gestures() == []


def test_get_db_gestures_shows_legacy_auto_imported_user_recordings(
    monkeypatch,
    tmp_path,
):
    import app.flet_app.controller as controller_module

    data_root = tmp_path / "gestures"
    label_dir = data_root / "CTRLZ"
    label_dir.mkdir(parents=True)
    sample_path = label_dir / "sample_0000.npy"
    np.save(sample_path, np.zeros((30, 21, 2), dtype=np.float32))

    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(["ctrlz"]), encoding="utf-8")
    dynamic_classes_path = tmp_path / "dynamic_classes.json"
    dynamic_classes_path.write_text(json.dumps([]), encoding="utf-8")

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        gesture = Gesture(
            label="ctrlz",
            description=f"Auto-imported from {label_dir}",
            samples_path=str(label_dir),
            model_class_id=0,
            is_active=True,
        )
        session.add(gesture)
        session.flush()
        session.add(
            GestureSample(
                gesture_id=gesture.id,
                sample_index=0,
                features_path=str(sample_path),
                frames=30,
                hand_count=1,
                source="dataset",
            )
        )
        session.commit()

    controller = AppController.__new__(AppController)
    controller._db_initialized = True
    monkeypatch.setattr(controller, "_configured_classes_path", lambda: classes_path)
    monkeypatch.setattr(controller, "_dynamic_classes_path", lambda: dynamic_classes_path)
    monkeypatch.setattr(controller, "_configured_path", lambda value: Path(value))
    monkeypatch.setattr(controller_module, "get_db_session", SessionLocal)

    rows = controller.get_db_gestures()

    assert [row["label"] for row in rows] == ["ctrlz"]
    assert rows[0]["sampleCount"] == 1
    assert rows[0]["samplePreviewPath"].endswith("sample_0000.npy")


def test_get_db_gestures_shows_legacy_auto_imported_dynamic_recordings(
    monkeypatch,
    tmp_path,
):
    import app.flet_app.controller as controller_module

    data_root = tmp_path / "gestures"
    label_dir = data_root / "swipe_up"
    label_dir.mkdir(parents=True)
    sample_path = label_dir / "sample_0000.npy"
    np.save(sample_path, np.zeros((30, 44), dtype=np.float32))

    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(["swipe_up"]), encoding="utf-8")
    dynamic_classes_path = tmp_path / "dynamic_classes.json"
    dynamic_classes_path.write_text(json.dumps(["swipe_up"]), encoding="utf-8")

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        gesture = Gesture(
            label="swipe_up",
            description=f"Auto-imported from {label_dir}",
            samples_path=str(label_dir),
            model_class_id=0,
            is_active=True,
        )
        session.add(gesture)
        session.flush()
        session.add(
            GestureSample(
                gesture_id=gesture.id,
                sample_index=0,
                features_path=str(sample_path),
                frames=30,
                hand_count=1,
                source="dataset",
            )
        )
        session.commit()

    controller = AppController.__new__(AppController)
    controller._db_initialized = True
    monkeypatch.setattr(controller, "_configured_classes_path", lambda: classes_path)
    monkeypatch.setattr(controller, "_dynamic_classes_path", lambda: dynamic_classes_path)
    monkeypatch.setattr(controller, "_configured_path", lambda value: Path(value))
    monkeypatch.setattr(controller_module, "get_db_session", SessionLocal)

    rows = controller.get_db_gestures()

    assert [row["label"] for row in rows] == ["swipe_up"]
    assert rows[0]["sampleCount"] == 1
    assert rows[0]["gestureType"] == "dynamic"
    assert rows[0]["samplePreviewPath"].endswith("sample_0000.npy")


def test_sync_dataset_to_db_keeps_camera_source_from_metadata(monkeypatch, tmp_path):
    import app.flet_app.controller as controller_module

    data_root = tmp_path / "gestures"
    label_dir = data_root / "Wave"
    label_dir.mkdir(parents=True)
    sample_path = label_dir / "sample_0000.npy"
    np.save(sample_path, np.zeros((30, 21, 2), dtype=np.float32))
    sample_path.with_suffix(".meta.json").write_text(
        json.dumps({"source": "camera", "kind": "real"}),
        encoding="utf-8",
    )

    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps([]), encoding="utf-8")
    dynamic_classes_path = tmp_path / "dynamic_classes.json"
    dynamic_classes_path.write_text(json.dumps([]), encoding="utf-8")

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    controller = AppController.__new__(AppController)
    controller._db_initialized = True
    monkeypatch.setattr(controller, "_configured_data_dir", lambda: data_root)
    monkeypatch.setattr(controller, "_configured_classes_path", lambda: classes_path)
    monkeypatch.setattr(controller, "_dynamic_classes_path", lambda: dynamic_classes_path)
    monkeypatch.setattr(controller, "_configured_path", lambda value: Path(value))
    monkeypatch.setattr(controller_module, "get_db_session", SessionLocal)

    controller.sync_dataset_to_db()

    with SessionLocal() as session:
        gesture = session.query(Gesture).filter_by(label="Wave").one()
        sample = session.query(GestureSample).filter_by(gesture_id=gesture.id).one()
        assert sample.source == "camera"
        assert gesture.description in {None, ""}

    rows = controller.get_db_gestures()

    assert [row["label"] for row in rows] == ["Wave"]
    assert rows[0]["sampleCount"] == 1
    assert rows[0]["samplePreviewPath"].endswith("sample_0000.npy")


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

    assert controller.list_recorded_gestures() == [
        {
            "label": "FULL",
            "samples": 1,
            "realSamples": 1,
            "augmentedSamples": 0,
            "userRecordedSamples": 1,
            "canDelete": True,
            "deleteReason": "",
            "systemClass": False,
        }
    ]


def test_list_recorded_gestures_marks_only_user_classes_deletable(monkeypatch, tmp_path):
    data_root = tmp_path / "gestures"
    user_dir = data_root / "USER_WAVE"
    system_dir = data_root / LIVE_EVAL_NO_COMMAND_LABEL
    imported_dir = data_root / "IMPORTED"
    user_dir.mkdir(parents=True)
    system_dir.mkdir(parents=True)
    imported_dir.mkdir(parents=True)

    user_sample = user_dir / "sample_0000.npy"
    system_sample = system_dir / "sample_0000.npy"
    imported_sample = imported_dir / "sample_0000.npy"
    np.save(user_sample, np.zeros((30, 21, 2), dtype=np.float32))
    np.save(system_sample, np.zeros((30, 21, 2), dtype=np.float32))
    np.save(imported_sample, np.zeros((30, 21, 2), dtype=np.float32))
    user_sample.with_suffix(".meta.json").write_text(
        json.dumps({"source": "camera", "kind": "real"}),
        encoding="utf-8",
    )
    system_sample.with_suffix(".meta.json").write_text(
        json.dumps({"source": "camera", "kind": "real"}),
        encoding="utf-8",
    )
    imported_sample.with_suffix(".meta.json").write_text(
        json.dumps({"source": "dataset", "kind": "real"}),
        encoding="utf-8",
    )

    controller = AppController.__new__(AppController)
    monkeypatch.setattr(controller, "_configured_data_dir", lambda: data_root)

    rows = {row["label"]: row for row in controller.list_recorded_gestures()}

    assert rows["USER_WAVE"]["canDelete"] is True
    assert rows["USER_WAVE"]["userRecordedSamples"] == 1
    assert rows[LIVE_EVAL_NO_COMMAND_LABEL]["canDelete"] is False
    assert rows[LIVE_EVAL_NO_COMMAND_LABEL]["systemClass"] is True
    assert rows["IMPORTED"]["canDelete"] is False
    assert rows["IMPORTED"]["userRecordedSamples"] == 0


def test_list_recognition_labels_ignores_stale_model_classes_by_default(
    monkeypatch,
    tmp_path,
):
    data_root = tmp_path / "gestures"
    label_dir = data_root / "FreshGesture"
    label_dir.mkdir(parents=True)
    np.save(label_dir / "sample_0000.npy", np.zeros((30, 21, 2), dtype=np.float32))
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(["old_static"]), encoding="utf-8")
    dynamic_classes_path = tmp_path / "dynamic_classes.json"
    dynamic_classes_path.write_text(json.dumps(["old_dynamic"]), encoding="utf-8")

    controller = AppController.__new__(AppController)
    monkeypatch.setattr(controller, "_configured_data_dir", lambda: data_root)
    monkeypatch.setattr(controller, "_configured_classes_path", lambda: classes_path)
    monkeypatch.setattr(controller, "_dynamic_classes_path", lambda: dynamic_classes_path)
    monkeypatch.setattr(controller, "_is_negative_label", lambda _label: False)

    labels = controller.list_recognition_labels()

    assert labels == [LIVE_EVAL_NO_COMMAND_LABEL, "FreshGesture"]
    assert "old_static" not in labels
    assert "old_dynamic" not in labels
    assert "old_static" in controller.list_recognition_labels(
        include_model_classes=True
    )


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
    assert controller._sample_recording["include_landmark_z"] is False
    assert controller._sample_recording["augment_count"] == 1
    assert controller._sample_recording["started_camera_for_recording"] is True
    assert controller._sample_recording["out_dir"] == tmp_path / "gestures" / "Wave"
    assert controller._status == "Запись жеста: Wave"
    assert any("Встроенная запись" in line for line in lines)


def test_completed_training_process_does_not_block_new_recording(monkeypatch, tmp_path):
    controller = AppController.__new__(AppController)

    class DoneProc:
        def poll(self):
            return 0

    controller._training_proc = DoneProc()
    controller._sample_recording = None
    controller._sample_recording_detector = None
    controller._sample_recording_lock = threading.Lock()
    controller._sample_recording_detector_lock = threading.RLock()
    controller._is_camera_active = False
    controller._status = "Idle"
    controller.status_changed = _Event()

    monkeypatch.setattr(controller, "_configured_data_dir", lambda: tmp_path / "gestures")
    monkeypatch.setattr(controller, "stop_recognition", lambda: None)
    monkeypatch.setattr(controller, "stop_embedded_recognition", lambda: None)

    def fake_start_camera():
        controller._is_camera_active = True

    monkeypatch.setattr(controller, "start_camera", fake_start_camera)

    ok = controller.start_recording(label="zoom", num_samples=1, frames=2)

    assert ok is True
    assert controller._training_proc is None
    assert controller._sample_recording is not None
    assert controller._sample_recording["label"] == "zoom"


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


def test_finish_sample_recording_stops_camera_owned_by_recording(monkeypatch):
    controller = AppController.__new__(AppController)
    controller._sample_recording_lock = threading.Lock()
    controller._sample_recording_detector_lock = threading.RLock()
    done_codes = []
    stopped = []
    controller._sample_recording_detector = None
    controller._sample_recording = {
        "label": "Wave",
        "on_line": None,
        "on_done": done_codes.append,
        "started_camera_for_recording": True,
    }
    controller._status = "Запись жеста: Wave"
    controller._is_camera_active = True
    controller.status_changed = _Event()
    monkeypatch.setattr(controller, "sync_dataset_to_db", lambda: {})

    def stop_camera():
        stopped.append(True)
        controller._is_camera_active = False
        controller._status = "Stopped"

    monkeypatch.setattr(controller, "stop_camera", stop_camera)

    controller._finish_sample_recording(0, "[✓] done")

    assert stopped == [True]
    assert done_codes == [0]
    assert controller._sample_recording is None


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


def test_process_static_sample_recording_frame_saves_xyz_features(
    monkeypatch,
    tmp_path,
):
    controller = AppController.__new__(AppController)
    controller._sample_recording_lock = threading.Lock()
    controller._sample_recording_detector_lock = threading.RLock()
    label_dir = tmp_path / "gestures" / "Gun"
    done_codes = []

    class FakeHand:
        landmarks = [(float(i) / 20.0, float(i % 5) / 5.0) for i in range(21)]
        landmarks_xyz = [
            (float(i) / 20.0, float(i % 5) / 5.0, float(i) / 100.0)
            for i in range(21)
        ]

    class FakeDetector:
        def detect_for_video_rgb(self, _rgb):
            return [FakeHand()]

        def close(self):
            pass

    controller._sample_recording_detector = FakeDetector()
    controller._sample_recording = {
        "label": "Gun",
        "target": 1,
        "frames": 2,
        "two_hands": False,
        "include_global_motion": False,
        "include_landmark_z": True,
        "saved": 0,
        "frames_buf": [],
        "out_dir": label_dir,
        "on_line": None,
        "on_done": done_codes.append,
        "next_allowed_at": 0.0,
        "last_no_hand_log": 0.0,
    }
    controller._status = "Запись жеста: Gun"
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
    metadata = json.loads(
        (label_dir / "sample_0000.meta.json").read_text(encoding="utf-8")
    )
    hand_points = np.asarray(FakeHand.landmarks, dtype=np.float32)
    hand_scale = np.max(np.linalg.norm(hand_points - hand_points[0], axis=1))
    expected_z = [float(i) / 100.0 / hand_scale for i in range(21)]
    assert sample.shape == (2, 21, 3)
    assert np.allclose(sample[0, :, 2], expected_z)
    assert metadata["raw_feature_dim"] == 63
    assert metadata["include_landmark_z"] is True
    assert metadata["include_global_motion"] is False
    assert metadata["sample_feature_format"] == "landmark_xyz"
    assert done_codes == [0]


def test_process_sample_recording_frame_saves_gislr_augmentations(
    monkeypatch,
    tmp_path,
):
    controller = AppController.__new__(AppController)
    controller._sample_recording_lock = threading.Lock()
    controller._sample_recording_detector_lock = threading.RLock()
    label_dir = tmp_path / "gestures" / "Wave"
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
        "label": "Wave",
        "target": 1,
        "frames": 2,
        "two_hands": False,
        "include_global_motion": False,
        "augment_count": 2,
        "saved": 0,
        "frames_buf": [],
        "out_dir": label_dir,
        "on_line": None,
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
        lambda: {"created": 1, "updated": 0, "total": 1, "samples": 3},
    )

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    controller._process_sample_recording_frame(frame)
    controller._process_sample_recording_frame(frame)

    assert (label_dir / "sample_0000.npy").exists()
    aug0 = label_dir / "aug_sample_0000_00.npy"
    aug1 = label_dir / "aug_sample_0000_01.npy"
    assert aug0.exists()
    assert aug1.exists()
    assert np.load(aug0).shape == (2, 21, 2)
    metadata = json.loads(aug0.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata["source"] == "augmented"
    assert metadata["source_sample"] == "sample_0000.npy"
    assert metadata["transform"] == "gislr_landmark_v1"
    assert metadata["transform_metadata"]["policy"] == "gislr_landmark_v1"
    assert done_codes == [0]


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
    metadata = json.loads(
        (label_dir / "sample_0000.meta.json").read_text(encoding="utf-8")
    )
    assert sample.shape == (2, 44)
    assert np.allclose(sample[0, -2:], [0.0, 0.0])
    assert metadata["raw_feature_dim"] == 44
    assert metadata["projected_hand_scale_median"] > 0.0
    assert done_codes == [0]


def test_process_dynamic_sample_recording_frame_saves_xyz_wrist_features(
    monkeypatch,
    tmp_path,
):
    controller = AppController.__new__(AppController)
    controller._sample_recording_lock = threading.Lock()
    controller._sample_recording_detector_lock = threading.RLock()
    label_dir = tmp_path / "gestures" / "SwipeRight"
    done_codes = []

    class FakeHand:
        landmarks = [(float(i) / 20.0, float(i % 5) / 5.0) for i in range(21)]
        landmarks_xyz = [
            (float(i) / 20.0, float(i % 5) / 5.0, float(i) / 100.0)
            for i in range(21)
        ]

    class FakeDetector:
        def detect_for_video_rgb(self, _rgb):
            return [FakeHand()]

        def close(self):
            pass

    controller._sample_recording_detector = FakeDetector()
    controller._sample_recording = {
        "label": "SwipeRight",
        "target": 1,
        "frames": 2,
        "two_hands": False,
        "include_global_motion": True,
        "include_landmark_z": True,
        "saved": 0,
        "frames_buf": [],
        "out_dir": label_dir,
        "on_line": None,
        "on_done": done_codes.append,
        "next_allowed_at": 0.0,
        "last_no_hand_log": 0.0,
    }
    controller._status = "Запись жеста: SwipeRight"
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
    metadata = json.loads(
        (label_dir / "sample_0000.meta.json").read_text(encoding="utf-8")
    )
    hand_points = np.asarray(FakeHand.landmarks, dtype=np.float32)
    hand_scale = np.max(np.linalg.norm(hand_points - hand_points[0], axis=1))
    expected_z = [float(i) / 100.0 / hand_scale for i in range(21)]
    assert sample.shape == (2, 65)
    assert np.allclose(sample[0, 2:63:3], expected_z)
    assert np.allclose(sample[0, -2:], [0.0, 0.0])
    assert metadata["raw_feature_dim"] == 65
    assert metadata["include_landmark_z"] is True
    assert metadata["sample_feature_format"] == "landmark_xyz_wrist_xy"
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
        sample_path.with_suffix(".meta.json").write_text(
            json.dumps({"projected_hand_scale_median": 0.2}),
            encoding="utf-8",
        )
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


def test_delete_recorded_samples_rejects_system_classes(monkeypatch, tmp_path):
    import app.flet_app.controller as controller_module

    data_root = tmp_path / "gestures"
    label_dir = data_root / LIVE_EVAL_NO_COMMAND_LABEL
    label_dir.mkdir(parents=True)
    sample_path = label_dir / "sample_0000.npy"
    np.save(sample_path, np.zeros((30, 21, 2), dtype=np.float32))
    sample_path.with_suffix(".meta.json").write_text(
        json.dumps({"source": "camera", "kind": "real"}),
        encoding="utf-8",
    )

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        gesture = Gesture(
            label=LIVE_EVAL_NO_COMMAND_LABEL,
            samples_path=str(label_dir),
            model_class_id=0,
            is_active=True,
        )
        session.add(gesture)
        session.flush()
        session.add(
            GestureSample(
                gesture_id=gesture.id,
                sample_index=0,
                features_path=str(sample_path),
                frames=30,
                hand_count=1,
                source="camera",
            )
        )
        session.commit()

    controller = AppController.__new__(AppController)
    controller._db_initialized = True
    monkeypatch.setattr(controller, "_configured_data_dir", lambda: data_root)
    monkeypatch.setattr(controller_module, "get_db_session", SessionLocal)

    summary = controller.delete_recorded_samples(LIVE_EVAL_NO_COMMAND_LABEL)

    assert summary["ok"] is False
    assert "записанные пользователем" in summary["error"]
    assert sample_path.exists()
    with SessionLocal() as session:
        gesture = session.query(Gesture).filter_by(label=LIVE_EVAL_NO_COMMAND_LABEL).one()
        assert gesture.is_active is True
        assert session.query(GestureSample).count() == 1


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
    states = []
    controller.gesture_detected.connect(emitted.append)
    controller.gesture_state_changed.connect(states.append)
    controller.execute_for_gesture = lambda label, conf: executed.append((label, conf)) or True
    controller._record_recognition_event = lambda label, conf, ok: recorded.append((label, conf, ok))

    for _ in range(AUTO_STATIC_GESTURE_CONFIRM_FRAMES - 1):
        controller._dispatch_infer_result(
            {"label": "new", "confidence": 0.8, "landmarks_json": "[]"}
        )

    assert emitted == []
    assert executed == []
    assert recorded == []
    assert states[-1]["phase"] == "pending"
    assert states[-1]["frames"] == AUTO_STATIC_GESTURE_CONFIRM_FRAMES - 1
    assert states[-1]["progress"] < 1.0

    controller._dispatch_infer_result(
        {"label": "new", "confidence": 0.8, "landmarks_json": "[]"}
    )

    assert emitted == ["new"]
    assert executed == [("new", pytest.approx(0.8))]
    assert recorded == [("new", pytest.approx(0.8), True)]
    assert states[-1]["phase"] == "confirmed"
    assert states[-1]["label"] == "new"
    assert states[-1]["progress"] == 1.0

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

    assert emitted == ["hand_left"]
    assert executed == [("hand_left", pytest.approx(0.9))]
    assert recorded == [("hand_left", pytest.approx(0.9), True)]


def test_auto_dispatch_confirms_taxonomy_dynamic_label_faster():
    controller = _dispatch_controller()
    controller._recognition_model_mode = RECOGNITION_MODEL_AUTO
    emitted = []
    executed = []
    recorded = []
    controller.gesture_detected.connect(emitted.append)
    controller.execute_for_gesture = lambda label, conf: executed.append((label, conf)) or True
    controller._record_recognition_event = lambda label, conf, ok: recorded.append((label, conf, ok))

    for _ in range(DYNAMIC_GESTURE_CONFIRM_FRAMES):
        controller._dispatch_infer_result(
            {"label": "swipe_up", "confidence": 0.9, "landmarks_json": "[]"}
        )

    assert emitted == ["swipe_up"]
    assert executed == [("swipe_up", pytest.approx(0.9))]
    assert recorded == [("swipe_up", pytest.approx(0.9), True)]


def test_dispatch_suppresses_opposite_return_motion_after_dynamic_event():
    controller = _dispatch_controller()
    controller._recognition_model_mode = RECOGNITION_MODEL_AUTO
    emitted = []
    executed = []
    recorded = []
    controller.gesture_detected.connect(emitted.append)
    controller.execute_for_gesture = lambda label, conf: executed.append((label, conf)) or True
    controller._record_recognition_event = lambda label, conf, ok: recorded.append((label, conf, ok))

    controller._dispatch_infer_result(
        {
            "label": "swipe_down",
            "confidence": 0.9,
            "landmarks_json": "[]",
            "router": {"route": "dynamic"},
        }
    )
    controller._dispatch_infer_result(
        {
            "label": "swipe_up",
            "confidence": 0.9,
            "landmarks_json": "[]",
            "router": {"route": "dynamic"},
        }
    )

    assert emitted == ["swipe_down"]
    assert executed == [("swipe_down", pytest.approx(0.9))]
    assert recorded == [("swipe_down", pytest.approx(0.9), True)]
    assert controller._status == "Suppressed return motion: swipe_up"


def test_dispatch_treats_negative_label_as_rejection_not_command():
    controller = _dispatch_controller()
    controller._recognition_model_mode = RECOGNITION_MODEL_AUTO
    emitted = []
    executed = []
    recorded = []
    states = []
    controller.gesture_detected.connect(emitted.append)
    controller.gesture_state_changed.connect(states.append)
    controller.execute_for_gesture = lambda label, conf: executed.append((label, conf)) or True
    controller._record_recognition_event = lambda label, conf, ok: recorded.append((label, conf, ok))

    controller._dispatch_infer_result(
        {
            "label": "random_motion",
            "confidence": 0.95,
            "landmarks_json": "[]",
            "router": {"route": "dynamic"},
        }
    )

    assert emitted == []
    assert executed == []
    assert recorded == [("random_motion", pytest.approx(0.95), False)]
    assert controller._status == "Rejected gesture evidence: random_motion"
    assert states[-1]["phase"] == "rejected"
    assert states[-1]["reason"] == "negative_label"


def test_dispatch_emits_cooldown_state_when_binding_policy_rejects():
    controller = _dispatch_controller()
    states = []
    controller.gesture_state_changed.connect(states.append)

    def reject_for_cooldown(_label, _conf):
        controller._last_execute_info = "cooldown"
        return False

    controller.execute_for_gesture = reject_for_cooldown
    controller._record_recognition_event = lambda *_args: None

    for _ in range(AUTO_STATIC_GESTURE_CONFIRM_FRAMES):
        controller._dispatch_infer_result(
            {"label": "new", "confidence": 0.8, "landmarks_json": "[]"}
        )

    assert states[-1]["phase"] == "cooldown"
    assert states[-1]["label"] == "new"
    assert states[-1]["reason"] == "cooldown"


def test_dispatch_holds_live_state_when_prediction_briefly_disappears(monkeypatch):
    from app.flet_app import controller as controller_module

    timers = []

    class FakeTimer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.cancelled = False
            self.started = False
            self.daemon = False
            timers.append(self)

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(controller_module.threading, "Timer", FakeTimer)

    controller = _dispatch_controller()
    controller._recognition_model_mode = "dynamic"
    states = []
    controller.gesture_state_changed.connect(states.append)
    controller.execute_for_gesture = lambda *_args: True
    controller._record_recognition_event = lambda *_args: None

    controller._dispatch_infer_result(
        {"label": "hand_left", "confidence": 0.9, "landmarks_json": "[]"}
    )

    assert states[-1]["phase"] == "confirmed"

    controller._dispatch_infer_result(
        {"label": "", "confidence": 0.0, "landmarks_json": "[]"}
    )

    assert states[-1]["phase"] == "confirmed"
    assert timers[-1].delay == pytest.approx(
        controller_module.LIVE_GESTURE_IDLE_HOLD_SECONDS
    )
    assert timers[-1].started is True

    timers[-1].callback()

    assert states[-1]["phase"] == "idle"


def test_dispatch_populates_live_recognition_inspector_history():
    controller = _dispatch_controller()
    controller._recognition_model_mode = "dynamic"
    states = []
    controller.gesture_state_changed.connect(states.append)
    controller.execute_for_gesture = lambda *_args: True
    controller._record_recognition_event = lambda *_args: None

    controller._dispatch_infer_result(
        {
            "label": "hand_left",
            "confidence": 0.9,
            "landmarks_json": "[]",
            "router": {
                "route": "dynamic",
                "selected_reason": "dynamic_accepted",
            },
        }
    )

    assert states[-1]["phase"] == "confirmed"
    assert states[-1]["mode"] == "dynamic"
    assert states[-1]["route"] == "dynamic"
    assert states[-1]["model"] == DYNAMIC_MODEL_PROFILE_PRODUCTION
    assert states[-1]["reason"] == "dynamic_accepted"
    assert states[-1]["sequence"] == 1

    history = controller.get_live_gesture_inspector_history()

    assert len(history) == 1
    assert history[0]["label"] == "hand_left"
    assert history[0]["model"] == DYNAMIC_MODEL_PROFILE_PRODUCTION
    assert history[0]["recordedAt"] > 0


def test_live_recognition_inspector_history_exports_jsonl_and_csv(monkeypatch, tmp_path):
    controller = _dispatch_controller()
    controller._live_gesture_inspector_history = [
        {
            "sequence": 2,
            "recordedAt": 2.0,
            "phase": "confirmed",
            "label": "swipe_down",
            "confidence": 0.9,
            "progress": 1.0,
            "frames": 1,
            "requiredFrames": 1,
            "mode": "auto",
            "route": "dynamic",
            "model": DYNAMIC_MODEL_PROFILE_PRODUCTION,
            "staticReject": "open_set_policy",
            "reason": "dynamic_accepted",
            "displayText": "swipe_down",
        },
        {
            "sequence": 1,
            "recordedAt": 1.0,
            "phase": "pending",
            "label": "swipe_down",
            "confidence": 0.7,
            "progress": 0.5,
            "frames": 1,
            "requiredFrames": 2,
            "mode": "auto",
            "route": "dynamic",
            "model": DYNAMIC_MODEL_PROFILE_PRODUCTION,
            "staticReject": "open_set_policy",
            "reason": "dynamic_pending",
            "displayText": "swipe_down 1/2",
        },
    ]
    monkeypatch.setattr(controller, "_configured_log_dir", lambda: tmp_path)

    jsonl_path = controller.export_live_gesture_inspector_history("jsonl")
    csv_path = controller.export_live_gesture_inspector_history("csv")

    json_rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    with csv_path.open(encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))

    assert [row["sequence"] for row in json_rows] == [1, 2]
    assert [int(row["sequence"]) for row in csv_rows] == [1, 2]
    assert csv_rows[-1]["model"] == DYNAMIC_MODEL_PROFILE_PRODUCTION

    controller.clear_live_gesture_inspector_history()

    assert controller.get_live_gesture_inspector_history() == []


def test_confirmed_dynamic_event_is_acknowledged_without_full_reset():
    controller = _dispatch_controller()
    controller._recognition_model_mode = RECOGNITION_MODEL_AUTO
    controller.execute_for_gesture = lambda *_args: True
    controller._record_recognition_event = lambda *_args: None

    class _Infer:
        acknowledge_calls = 0

        def acknowledge_dynamic_event(self):
            self.acknowledge_calls += 1

    controller._embedded_infer = _Infer()
    output = {
        "label": "swipe_left",
        "confidence": 0.9,
        "landmarks_json": "[]",
        "router": {"route": "dynamic"},
    }

    for _ in range(DYNAMIC_GESTURE_CONFIRM_FRAMES):
        controller._dispatch_infer_result(output)

    assert controller._embedded_infer.acknowledge_calls == 1


def test_auto_dynamic_route_confirms_user_label_immediately():
    controller = _dispatch_controller()
    controller._recognition_model_mode = RECOGNITION_MODEL_AUTO
    states = []
    executed = []
    controller.gesture_state_changed.connect(states.append)
    controller.execute_for_gesture = (
        lambda label, conf: executed.append((label, conf)) or True
    )
    controller._record_recognition_event = lambda *_args: None

    controller._dispatch_infer_result(
        {
            "label": "SwipeLeft",
            "confidence": 0.96,
            "landmarks_json": "[]",
            "router": {
                "route": "dynamic",
                "selected_reason": "dynamic_accepted",
            },
        }
    )

    assert states[-1]["phase"] == "confirmed"
    assert states[-1]["frames"] == 1
    assert states[-1]["requiredFrames"] == DYNAMIC_GESTURE_CONFIRM_FRAMES
    assert executed == [("SwipeLeft", pytest.approx(0.96))]


def test_auto_static_route_waits_for_deliberate_dwell():
    controller = _dispatch_controller()
    controller._recognition_model_mode = RECOGNITION_MODEL_AUTO
    states = []
    executed = []
    controller.gesture_state_changed.connect(states.append)
    controller.execute_for_gesture = (
        lambda label, conf: executed.append((label, conf)) or True
    )
    controller._record_recognition_event = lambda *_args: None

    output = {
        "label": "hand",
        "confidence": 0.72,
        "landmarks_json": "[]",
        "router": {
            "route": "static",
            "selected_reason": "static_fallback",
        },
    }
    for _ in range(AUTO_STATIC_GESTURE_CONFIRM_FRAMES - 1):
        controller._dispatch_infer_result(output)

    assert states[-1]["phase"] == "pending"
    assert states[-1]["frames"] == AUTO_STATIC_GESTURE_CONFIRM_FRAMES - 1
    assert states[-1]["requiredFrames"] == AUTO_STATIC_GESTURE_CONFIRM_FRAMES
    assert executed == []

    controller._dispatch_infer_result(output)

    assert states[-1]["phase"] == "confirmed"
    assert states[-1]["frames"] == AUTO_STATIC_GESTURE_CONFIRM_FRAMES
    assert states[-1]["requiredFrames"] == AUTO_STATIC_GESTURE_CONFIRM_FRAMES
    assert executed == [("hand", pytest.approx(0.72))]


def test_auto_static_label_requires_deliberate_dwell():
    controller = _dispatch_controller()
    controller._recognition_model_mode = RECOGNITION_MODEL_AUTO

    assert (
        controller._gesture_confirm_frames("gun")
        == AUTO_STATIC_GESTURE_CONFIRM_FRAMES
    )


def test_static_mode_uses_deliberate_dwell():
    controller = _dispatch_controller()
    controller._recognition_model_mode = "static"

    assert (
        controller._gesture_confirm_frames("hand")
        == AUTO_STATIC_GESTURE_CONFIRM_FRAMES
    )


def test_dispatch_resets_confirmation_when_label_changes():
    controller = _dispatch_controller()
    executed = []
    controller.execute_for_gesture = lambda label, conf: executed.append((label, conf)) or True
    controller._record_recognition_event = lambda *_args: None

    for _ in range(AUTO_STATIC_GESTURE_CONFIRM_FRAMES - 2):
        controller._dispatch_infer_result(
            {"label": "new", "confidence": 0.9, "landmarks_json": "[]"}
        )
    for _ in range(AUTO_STATIC_GESTURE_CONFIRM_FRAMES):
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

    for _ in range(AUTO_STATIC_GESTURE_CONFIRM_FRAMES):
        controller._dispatch_infer_result(
            {"label": "new", "confidence": 0.8, "landmarks_json": "[]"}
        )
    controller._dispatch_infer_result(
        {"label": "", "confidence": 0.0, "landmarks_json": "[]"}
    )
    for _ in range(AUTO_STATIC_GESTURE_CONFIRM_FRAMES):
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


def test_update_pointer_emits_pointer_state_payload():
    controller = _dispatch_controller()
    controller._pointer_mode = True
    emitted = []
    controller.pointer_state_changed.connect(emitted.append)

    class FakePointer:
        def update(self, landmarks_json):
            assert landmarks_json == "[[[]]]"

            class Result:
                ok = True
                moved = False
                clicked = True
                tab_switched = ""
                error = ""
                state = "click-ready"

            return Result()

    controller._pointer_control = FakePointer()

    AppController._update_pointer_from_landmarks(controller, "[[[]]]")

    assert emitted[-1] == {
        "enabled": True,
        "state": "click-ready",
        "moved": False,
        "clicked": True,
        "tabSwitched": "",
        "error": "",
    }


def test_live_evaluation_counts_correct_wrong_and_missed(monkeypatch, tmp_path):
    controller = _dispatch_controller()
    controller._recognition_model_mode = "dynamic"
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_PRODUCTION
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

    controller._consume_live_evaluation_prediction(
        "swipe_down",
        0.9,
        route_metadata={
            "route": "dynamic",
            "selected_reason": "dynamic_accepted",
            "static_label": "",
            "static_confidence": 0.0,
            "static_reject_reason": "no_label",
            "dynamic_label": "swipe_down",
            "dynamic_confidence": 0.9,
            "dynamic_type": "dynamic",
            "dynamic_end_reason": "hand_lost",
            "dynamic_motion_scale": 0.22,
        },
        now=10.0,
    )
    controller._consume_live_evaluation_prediction(
        "swipe_left",
        0.8,
        route_metadata={
            "route": "dynamic",
            "selected_reason": "dynamic_accepted",
            "static_label": "palm",
            "static_confidence": 0.7,
            "dynamic_label": "swipe_left",
            "dynamic_confidence": 0.8,
            "dynamic_type": "dynamic",
        },
        now=12.0,
    )
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
    assert rows[-1]["dynamic_model_profile"] == DYNAMIC_MODEL_PROFILE_PRODUCTION
    assert rows[-1]["route_counts"] == {"dynamic": 2, "none": 1}
    assert rows[0]["route"] == "dynamic"
    assert rows[0]["selected_reason"] == "dynamic_accepted"
    assert rows[0]["static_reject_reason"] == "no_label"
    assert rows[0]["dynamic_label"] == "swipe_down"
    assert rows[0]["dynamic_type"] == "dynamic"
    assert rows[0]["dynamic_end_reason"] == "hand_lost"
    assert rows[0]["dynamic_motion_scale"] == pytest.approx(0.22)


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


def test_negative_live_evaluation_counts_no_prediction_as_correct(monkeypatch, tmp_path):
    controller = _dispatch_controller()
    controller._ensure_embedded_recognition_for_live_controls = lambda: None
    controller._gesture_type_for_label = (
        lambda label: "negative" if label == "no_gesture_static" else "static"
    )
    monkeypatch.setattr(controller, "_configured_log_dir", lambda: tmp_path)

    assert controller.start_live_evaluation(
        "no_gesture_static",
        attempts=1,
        timeout_seconds=1.0,
        min_confidence=0.6,
    )
    controller._live_evaluation["attempt_started_at"] = 30.0
    controller._live_evaluation["next_ready_at"] = 30.0

    controller._update_live_evaluation_timeout(now=31.2)

    snapshot = controller.current_live_evaluation()
    assert snapshot["active"] is False
    assert snapshot["correct"] == 1
    assert snapshot["missed"] == 0


def test_no_command_live_evaluation_counts_negative_prediction_as_correct(
    monkeypatch,
    tmp_path,
):
    controller = _dispatch_controller()
    controller._ensure_embedded_recognition_for_live_controls = lambda: None
    monkeypatch.setattr(controller, "_configured_log_dir", lambda: tmp_path)

    assert controller.start_live_evaluation(
        LIVE_EVAL_NO_COMMAND_LABEL,
        attempts=2,
        timeout_seconds=0.0,
        min_confidence=0.6,
    )
    controller._live_evaluation["next_ready_at"] = 0.0
    controller._consume_live_evaluation_prediction(
        "random_motion",
        0.95,
        route_metadata={
            "route": "dynamic",
            "dynamic_decision_source": "negative_rejected",
        },
        now=40.0,
    )
    controller._consume_live_evaluation_prediction(
        "swipe_up",
        0.95,
        route_metadata={"route": "dynamic"},
        now=42.0,
    )

    snapshot = controller.current_live_evaluation()
    assert snapshot["active"] is False
    assert snapshot["correct"] == 1
    assert snapshot["wrong"] == 1
    assert snapshot["missed"] == 0


def test_live_evaluation_static_rejection_metrics_for_negative_expected():
    controller = _dispatch_controller()
    controller._gesture_type_for_label = (
        lambda label: "negative" if label == "no_gesture_static" else "static"
    )
    session = {
        "expected_label": "no_gesture_static",
        "target_attempts": 2,
        "total": 2,
        "correct": 1,
        "wrong": 1,
        "missed": 0,
        "attempts": [
            {
                "attempt": 1,
                "result": "wrong",
                "route": "static",
                "static_decision_source": "accepted",
                "static_rejection_method": "one_vs_rest_logreg",
            },
            {
                "attempt": 2,
                "result": "correct",
                "route": "none",
                "static_reject_reason": "negative_class",
                "static_decision_source": "negative_rejected",
                "static_rejection_method": "one_vs_rest_logreg",
            },
        ],
    }

    metrics = controller._live_evaluation_mlflow_metrics(session)

    assert metrics["live_static_accept_rate"] == pytest.approx(0.5)
    assert metrics["live_static_reject_rate"] == pytest.approx(0.5)
    assert metrics["live_static_false_positive_rate"] == pytest.approx(0.5)
    assert metrics["live_static_rejection_reason_negative_class_count"] == pytest.approx(1.0)
    assert metrics["live_static_decision_negative_rejected_count"] == pytest.approx(1.0)
    assert (
        metrics["live_static_rejection_method_one_vs_rest_logreg_count"]
        == pytest.approx(2.0)
    )


def test_dynamic_live_evaluation_counts_static_route_as_wrong(monkeypatch, tmp_path):
    controller = _dispatch_controller()
    controller._recognition_model_mode = RECOGNITION_MODEL_AUTO
    controller._ensure_embedded_recognition_for_live_controls = lambda: None
    monkeypatch.setattr(controller, "_configured_log_dir", lambda: tmp_path)

    assert controller.start_live_evaluation(
        "swipe_up",
        attempts=1,
        timeout_seconds=0.0,
        min_confidence=0.6,
    )
    controller._live_evaluation["next_ready_at"] = 0.0

    controller._consume_live_evaluation_prediction(
        "gun",
        1.0,
        route_metadata={"route": "static"},
        now=10.0,
    )

    snapshot = controller.current_live_evaluation()
    assert snapshot["active"] is False
    assert snapshot["total"] == 1
    assert snapshot["wrong"] == 1
    assert snapshot["lastResult"] == "wrong"


def test_live_evaluation_completion_logs_mlflow_metrics(monkeypatch, tmp_path):
    controller = _dispatch_controller()
    controller._recognition_model_mode = RECOGNITION_MODEL_AUTO
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_PRODUCTION
    controller._ensure_embedded_recognition_for_live_controls = lambda: None
    monkeypatch.setattr(controller, "_configured_log_dir", lambda: tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///test-live.db")

    calls = {
        "tracking_uri": "",
        "experiment": "",
        "run_name": "",
        "params": {},
        "metrics": {},
        "tags": {},
        "artifact_path": "",
        "artifact_payload": {},
        "artifact_bundle_path": "",
        "artifacts": [],
        "metric_history": [],
        "log_system_metrics": None,
    }

    class _Run:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeMlflow:
        @staticmethod
        def set_tracking_uri(value):
            calls["tracking_uri"] = value

        @staticmethod
        def set_experiment(value):
            calls["experiment"] = value

        @staticmethod
        def start_run(run_name="", log_system_metrics=None):
            calls["run_name"] = run_name
            calls["log_system_metrics"] = log_system_metrics
            return _Run()

        @staticmethod
        def set_tags(value):
            calls["tags"] = dict(value)

        @staticmethod
        def log_params(value):
            calls["params"] = dict(value)

        @staticmethod
        def log_metrics(value, step=None):
            if step is None:
                calls["metrics"] = dict(value)
            else:
                calls["metric_history"].append((step, dict(value)))

        @staticmethod
        def log_dict(payload, artifact_file):
            calls["artifact_payload"] = payload
            calls["artifact_path"] = artifact_file

        @staticmethod
        def log_artifacts(local_dir, artifact_path=None):
            root = Path(local_dir)
            calls["artifact_bundle_path"] = artifact_path
            calls["artifacts"] = sorted(
                str(path.relative_to(root))
                for path in root.rglob("*")
                if path.is_file()
            )

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow)

    assert controller.start_live_evaluation(
        "swipe_left",
        attempts=2,
        timeout_seconds=0.0,
        min_confidence=0.6,
    )
    controller._live_evaluation["started_at"] = 1.0
    controller._live_evaluation["attempt_started_at"] = 10.0
    controller._live_evaluation["next_ready_at"] = 10.0
    (tmp_path / "runtime_performance.jsonl").write_text(
        json.dumps(
            {
                "recorded_at": 2.0,
                "recognition_model_mode": "auto",
                "dynamic_model_profile": DYNAMIC_MODEL_PROFILE_PRODUCTION,
                "target_fps": 30,
                "samples": 10,
                "shared_detection_rate": 1.0,
                "inference_ms_avg": 20.0,
                "inference_ms_p95": 30.0,
                "detection_ms_avg": 12.0,
                "inference_fps_capacity": 50.0,
            }
        )
        + "\n"
    )

    controller._consume_live_evaluation_prediction(
        "swipe_left",
        0.9,
        route_metadata={
            "route": "dynamic",
            "dynamic_decision_source": "motion_first",
            "dynamic_end_reason": "hand_lost",
            "dynamic_motion_label": "swipe_left",
        },
        now=10.0,
    )
    controller._consume_live_evaluation_prediction(
        "swipe_down",
        0.8,
        route_metadata={
            "route": "dynamic",
            "dynamic_decision_source": "motion_first",
            "dynamic_end_reason": "velocity_drop",
            "dynamic_motion_label": "swipe_down",
        },
        now=12.0,
    )

    assert calls["tracking_uri"] == "sqlite:///test-live.db"
    assert calls["experiment"] == "GestureBind"
    assert (
        calls["run_name"]
        == "live-swipe_left-auto-dynamic_landmark_lstm_backbone-open_set_policy"
    )
    assert calls["log_system_metrics"] is True
    assert calls["tags"]["run_kind"] == "live_evaluation"
    assert calls["tags"]["artifact_bundle"] == "live_evaluation/index.html"
    assert calls["params"]["expected_label"] == "swipe_left"
    assert calls["params"]["expected_type"] == "dynamic"
    assert calls["params"]["static_rejection_method"] == "open_set_policy"
    assert calls["metrics"]["live_accuracy"] == pytest.approx(0.5)
    assert calls["metrics"]["live_recall"] == pytest.approx(0.5)
    assert calls["metrics"]["live_dynamic_recall"] == pytest.approx(0.5)
    assert calls["metrics"]["live_wrong_dynamic_direction_rate"] == pytest.approx(0.5)
    assert calls["metrics"]["live_static_hijack_rate"] == pytest.approx(0.0)
    assert calls["metrics"]["live_route_dynamic_count"] == pytest.approx(2.0)
    assert calls["metrics"]["live_decision_motion_first_count"] == pytest.approx(2.0)
    assert calls["metrics"]["live_end_reason_hand_lost_count"] == pytest.approx(1.0)
    assert calls["metrics"]["live_end_reason_velocity_drop_count"] == pytest.approx(1.0)
    assert calls["metrics"]["system_runtime_inference_ms_avg"] == pytest.approx(20.0)
    assert calls["metrics"]["system_runtime_fps_capacity_avg"] == pytest.approx(50.0)
    assert len(calls["metric_history"]) == 2
    assert calls["metric_history"][0][0] == 1
    assert calls["metric_history"][0][1]["attempt_is_correct"] == pytest.approx(1.0)
    assert calls["metric_history"][1][0] == 2
    assert calls["metric_history"][1][1]["attempt_wrong_direction"] == pytest.approx(1.0)
    assert calls["artifact_path"] == "live_evaluation_run.json"
    assert calls["artifact_payload"]["session"]["route_counts"] == {"dynamic": 2}
    assert calls["artifact_bundle_path"] == "live_evaluation"
    assert "index.html" in calls["artifacts"]
    assert "charts/quality.svg" in calls["artifacts"]
    assert "charts/attempt_timeline.svg" in calls["artifacts"]
    assert "metrics.csv" in calls["artifacts"]
    assert "attempts.csv" in calls["artifacts"]


def test_set_gesture_mode_clears_current_label_without_starting_cv():
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
    assert starts == []
    assert statuses[-1] == "Распознавание жестов включено (reject:open_set_policy)"


def test_set_pointer_mode_does_not_start_cv_while_idle():
    controller = _dispatch_controller()
    mode_events = []
    starts = []
    statuses = []
    controller._embedded_active = False
    controller._is_recognizing = False
    controller.pointer_mode_changed.connect(mode_events.append)
    controller._ensure_embedded_recognition_for_live_controls = lambda: starts.append(True)
    controller._set_status = statuses.append

    controller.set_pointer_mode(True)

    assert controller.pointer_mode is True
    assert mode_events == [True]
    assert starts == []
    assert statuses == []
