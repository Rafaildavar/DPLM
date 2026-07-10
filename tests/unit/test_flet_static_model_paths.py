import json
from pathlib import Path

from app.flet_app import controller as controller_module
from app.flet_app.controller import (
    AppController,
    DYNAMIC_MODEL_PROFILE_DYNAMIC_LANDMARK_LSTM_BACKBONE,
    DYNAMIC_MODEL_PROFILE_PRODUCTION,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_GRU_BACKBONE,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_LSTM_BACKBONE,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_MLP,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_MULTIROCKET,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_PHASE_HMM,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET_72,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_SPROCKET,
    MODEL_VARIANT_DIRS,
    MODEL_VARIANT_PRODUCTION,
)
from app.services.app_config import AppConfig


def _write_dynamic_profile_bundle(
    models_dir: Path,
    *,
    stem: str,
    model_name: str,
    classes: list[str],
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / model_name).write_text("model", encoding="utf-8")
    (models_dir / f"{stem}_classes.json").write_text(
        json.dumps(classes),
        encoding="utf-8",
    )
    (models_dir / f"{stem}_feature_dim.txt").write_text("1584", encoding="utf-8")
    (models_dir / f"{stem}_feature_mode.txt").write_text(
        "dynamic_sequence",
        encoding="utf-8",
    )


def test_list_dynamic_model_profiles_marks_profiles_with_deleted_labels_stale(
    monkeypatch,
    tmp_path,
):
    def resolve_under_tmp(value):
        path = Path(value)
        return path if path.is_absolute() else tmp_path / path

    monkeypatch.setattr(controller_module, "resolve_config_path", resolve_under_tmp)
    data_root = tmp_path / "data" / "gestures"
    (data_root / "swipeleft").mkdir(parents=True)
    (data_root / "swipeleft" / "sample_0001.npy").write_text("x", encoding="utf-8")
    (data_root / "no_gesture_static").mkdir(parents=True)
    (data_root / "no_gesture_static" / "sample_0001.npy").write_text(
        "x",
        encoding="utf-8",
    )
    models_dir = tmp_path / "models"
    _write_dynamic_profile_bundle(
        models_dir,
        stem="dynamic_landmark_lstm_backbone",
        model_name="dynamic_landmark_lstm_backbone.pkl",
        classes=["swipeleft", "no_gesture_static", "wrong_axis_motion"],
    )
    _write_dynamic_profile_bundle(
        models_dir,
        stem="dynamic_sequence_lstm_backbone",
        model_name="dynamic_sequence_lstm_backbone.pkl",
        classes=["swipe_up", "swipe_down", "wrong_axis_motion"],
    )

    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_PRODUCTION
    controller._config = AppConfig()
    controller._config.paths.data_dir = "data/gestures"
    controller._config.paths.models_dir = "models"

    rows = {
        row["key"]: row
        for row in controller.list_dynamic_model_profiles()
    }

    assert rows[DYNAMIC_MODEL_PROFILE_PRODUCTION]["quick_selectable"] is True
    assert rows[DYNAMIC_MODEL_PROFILE_PRODUCTION]["stale"] is False
    stale = rows[DYNAMIC_MODEL_PROFILE_SEQUENCE_LSTM_BACKBONE]
    assert stale["quick_selectable"] is False
    assert stale["stale"] is True
    assert stale["missing_dataset_labels"] == ["swipe_up", "swipe_down"]


def test_static_artifact_paths_fall_back_to_production_for_dynamic_only_variant(
    monkeypatch,
    tmp_path,
):
    def resolve_under_tmp(value):
        path = Path(value)
        return path if path.is_absolute() else tmp_path / path

    monkeypatch.setattr(controller_module, "resolve_config_path", resolve_under_tmp)
    production_dir = tmp_path / MODEL_VARIANT_DIRS[MODEL_VARIANT_PRODUCTION]
    production_dir.mkdir(parents=True)
    for name in (
        "knn.pkl",
        "classes.json",
        "feature_dim.txt",
        "feature_mode.txt",
        "static_rejection_verifiers.pkl",
    ):
        (production_dir / name).write_text("x", encoding="utf-8")

    variant_rel = "models/experiments/dynamic_prototype/prototype_distance"
    variant_dir = tmp_path / variant_rel
    variant_dir.mkdir(parents=True)
    (variant_dir / "dynamic_landmark_lstm_backbone.pkl").write_text(
        "x",
        encoding="utf-8",
    )

    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_PRODUCTION
    controller._config = AppConfig()
    controller._config.paths.models_dir = variant_rel
    controller._config.paths.model_path = f"{variant_rel}/knn.pkl"
    controller._config.paths.classes_path = f"{variant_rel}/classes.json"
    controller._config.paths.feature_dim_path = f"{variant_rel}/feature_dim.txt"

    assert controller._configured_models_dir() == variant_dir
    assert (
        controller._dynamic_model_path()
        == variant_dir / "dynamic_landmark_lstm_backbone.pkl"
    )
    assert controller._configured_model_path() == production_dir / "knn.pkl"
    assert controller._configured_classes_path() == production_dir / "classes.json"
    assert controller._configured_feature_dim_path() == production_dir / "feature_dim.txt"
    assert controller._configured_feature_mode_path() == production_dir / "feature_mode.txt"
    assert (
        controller._static_rejection_verifier_path()
        == production_dir / "static_rejection_verifiers.pkl"
    )


def test_apply_dynamic_only_model_variant_keeps_static_paths_in_production():
    controller = AppController.__new__(AppController)
    captured = {}
    emitted = []

    controller.get_app_config = lambda: {"paths": {}}
    controller.save_app_config = lambda raw: (captured.setdefault("raw", raw) and True, [], [])
    controller._reset_embedded_infer_after_model_change = lambda: None
    controller._set_status = lambda _status: None

    class Event:
        def emit(self, value):
            emitted.append(value)

    controller.model_variant_changed = Event()

    ok, errors, warnings = controller.apply_model_variant("prototype_distance")

    assert ok is True
    assert errors == []
    assert warnings == []
    paths = captured["raw"]["paths"]
    assert paths["models_dir"] == MODEL_VARIANT_DIRS["prototype_distance"]
    assert paths["model_path"] == "models/knn.pkl"
    assert paths["classes_path"] == "models/classes.json"
    assert paths["feature_dim_path"] == "models/feature_dim.txt"
    assert emitted == ["prototype_distance"]


def test_apply_static_landmark_cnn_model_variant_uses_cnn_static_artifact():
    controller = AppController.__new__(AppController)
    captured = {}
    emitted = []

    controller.get_app_config = lambda: {"paths": {}}
    controller.save_app_config = lambda raw: (captured.setdefault("raw", raw) and True, [], [])
    controller._reset_embedded_infer_after_model_change = lambda: None
    controller._set_status = lambda _status: None

    class Event:
        def emit(self, value):
            emitted.append(value)

    controller.model_variant_changed = Event()

    ok, errors, warnings = controller.apply_model_variant("static_landmark_cnn")

    assert ok is True
    assert errors == []
    assert warnings == []
    expected_dir = MODEL_VARIANT_DIRS["static_landmark_cnn"]
    paths = captured["raw"]["paths"]
    assert paths["models_dir"] == expected_dir
    assert paths["model_path"] == f"{expected_dir}/static_landmark_cnn.pkl"
    assert paths["classes_path"] == f"{expected_dir}/classes.json"
    assert paths["feature_dim_path"] == f"{expected_dir}/feature_dim.txt"
    assert emitted == ["static_landmark_cnn"]


def test_apply_static_landmark_image_extra_trees_variant_uses_regular_static_artifact():
    controller = AppController.__new__(AppController)
    captured = {}
    emitted = []

    controller.get_app_config = lambda: {"paths": {}}
    controller.save_app_config = lambda raw: (captured.setdefault("raw", raw) and True, [], [])
    controller._reset_embedded_infer_after_model_change = lambda: None
    controller._set_status = lambda _status: None

    class Event:
        def emit(self, value):
            emitted.append(value)

    controller.model_variant_changed = Event()

    ok, errors, warnings = controller.apply_model_variant(
        "static_landmark_image_extra_trees"
    )

    assert ok is True
    assert errors == []
    assert warnings == []
    expected_dir = MODEL_VARIANT_DIRS["static_landmark_image_extra_trees"]
    paths = captured["raw"]["paths"]
    assert paths["models_dir"] == expected_dir
    assert paths["model_path"] == f"{expected_dir}/knn.pkl"
    assert paths["classes_path"] == f"{expected_dir}/classes.json"
    assert paths["feature_dim_path"] == f"{expected_dir}/feature_dim.txt"
    assert emitted == ["static_landmark_image_extra_trees"]


def test_list_model_variants_reports_static_landmark_cnn_artifact(
    monkeypatch,
    tmp_path,
):
    def resolve_under_tmp(value):
        path = Path(value)
        return path if path.is_absolute() else tmp_path / path

    monkeypatch.setattr(controller_module, "resolve_config_path", resolve_under_tmp)
    cnn_dir = tmp_path / MODEL_VARIANT_DIRS["static_landmark_cnn"]
    cnn_dir.mkdir(parents=True)
    (cnn_dir / "static_landmark_cnn.pkl").write_text("x", encoding="utf-8")

    config = AppConfig()
    config.paths.models_dir = MODEL_VARIANT_DIRS["static_landmark_cnn"]

    class Store:
        def load(self, include_env=False):
            return config

    controller = AppController.__new__(AppController)
    controller._config_store = Store()

    variants = controller.list_model_variants()
    cnn = next(item for item in variants if item["key"] == "static_landmark_cnn")

    assert cnn["selected"] is True
    assert cnn["static_model_exists"] is True
    assert cnn["static_model_filename"] == "static_landmark_cnn.pkl"


def test_list_model_variants_reports_static_landmark_image_extra_trees_artifact(
    monkeypatch,
    tmp_path,
):
    def resolve_under_tmp(value):
        path = Path(value)
        return path if path.is_absolute() else tmp_path / path

    monkeypatch.setattr(controller_module, "resolve_config_path", resolve_under_tmp)
    variant_key = "static_landmark_image_extra_trees"
    variant_dir = tmp_path / MODEL_VARIANT_DIRS[variant_key]
    variant_dir.mkdir(parents=True)
    (variant_dir / "knn.pkl").write_text("x", encoding="utf-8")

    config = AppConfig()
    config.paths.models_dir = MODEL_VARIANT_DIRS[variant_key]

    class Store:
        def load(self, include_env=False):
            return config

    controller = AppController.__new__(AppController)
    controller._config_store = Store()

    variants = controller.list_model_variants()
    item = next(row for row in variants if row["key"] == variant_key)

    assert item["selected"] is True
    assert item["static_model_exists"] is True
    assert item["static_model_filename"] == "knn.pkl"


def test_legacy_dynamic_profile_falls_back_to_production_dynamic_paths(tmp_path):
    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = "sequence_knn"
    controller._config = AppConfig()
    controller._config.paths.models_dir = str(tmp_path / "models")

    assert controller.dynamic_model_profile == DYNAMIC_MODEL_PROFILE_PRODUCTION
    assert controller._dynamic_model_path().name == "dynamic_landmark_lstm_backbone.pkl"
    assert (
        controller._dynamic_classes_path().name
        == "dynamic_landmark_lstm_backbone_classes.json"
    )
    assert (
        controller._dynamic_feature_dim_path().name
        == "dynamic_landmark_lstm_backbone_feature_dim.txt"
    )
    assert (
        controller._dynamic_feature_mode_path().name
        == "dynamic_landmark_lstm_backbone_feature_mode.txt"
    )
    assert (
        controller._dynamic_prototypes_path().name
        == "dynamic_landmark_lstm_backbone_prototypes.json"
    )


def test_dynamic_landmark_lstm_profile_uses_own_metadata_files(tmp_path):
    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = (
        DYNAMIC_MODEL_PROFILE_DYNAMIC_LANDMARK_LSTM_BACKBONE
    )
    controller._config = AppConfig()
    controller._config.paths.models_dir = str(tmp_path / "models")

    assert controller._dynamic_model_path().name == "dynamic_landmark_lstm_backbone.pkl"
    assert (
        controller._dynamic_classes_path().name
        == "dynamic_landmark_lstm_backbone_classes.json"
    )
    assert (
        controller._dynamic_feature_dim_path().name
        == "dynamic_landmark_lstm_backbone_feature_dim.txt"
    )
    assert (
        controller._dynamic_feature_mode_path().name
        == "dynamic_landmark_lstm_backbone_feature_mode.txt"
    )
    assert (
        controller._dynamic_prototypes_path().name
        == "dynamic_landmark_lstm_backbone_prototypes.json"
    )


def test_sequence_mlp_dynamic_profile_uses_own_metadata_files(tmp_path):
    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_MLP
    controller._config = AppConfig()
    controller._config.paths.models_dir = str(tmp_path / "models")

    assert controller._dynamic_model_path().name == "dynamic_sequence_mlp.pkl"
    assert controller._dynamic_classes_path().name == "dynamic_sequence_mlp_classes.json"
    assert (
        controller._dynamic_feature_dim_path().name
        == "dynamic_sequence_mlp_feature_dim.txt"
    )
    assert (
        controller._dynamic_feature_mode_path().name
        == "dynamic_sequence_mlp_feature_mode.txt"
    )
    assert (
        controller._dynamic_prototypes_path().name
        == "dynamic_sequence_mlp_prototypes.json"
    )


def test_sequence_rocket_dynamic_profile_uses_own_metadata_files(tmp_path):
    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET
    controller._config = AppConfig()
    controller._config.paths.models_dir = str(tmp_path / "models")

    assert controller._dynamic_model_path().name == "dynamic_sequence_rocket.pkl"
    assert (
        controller._dynamic_classes_path().name
        == "dynamic_sequence_rocket_classes.json"
    )
    assert (
        controller._dynamic_feature_dim_path().name
        == "dynamic_sequence_rocket_feature_dim.txt"
    )
    assert (
        controller._dynamic_feature_mode_path().name
        == "dynamic_sequence_rocket_feature_mode.txt"
    )
    assert (
        controller._dynamic_prototypes_path().name
        == "dynamic_sequence_rocket_prototypes.json"
    )


def test_sequence_multirocket_dynamic_profile_uses_own_metadata_files(tmp_path):
    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_MULTIROCKET
    controller._config = AppConfig()
    controller._config.paths.models_dir = str(tmp_path / "models")

    assert controller._dynamic_model_path().name == "dynamic_sequence_multirocket.pkl"
    assert (
        controller._dynamic_classes_path().name
        == "dynamic_sequence_multirocket_classes.json"
    )
    assert (
        controller._dynamic_feature_dim_path().name
        == "dynamic_sequence_multirocket_feature_dim.txt"
    )
    assert (
        controller._dynamic_feature_mode_path().name
        == "dynamic_sequence_multirocket_feature_mode.txt"
    )
    assert (
        controller._dynamic_prototypes_path().name
        == "dynamic_sequence_multirocket_prototypes.json"
    )


def test_sequence_sprocket_dynamic_profile_uses_own_metadata_files(tmp_path):
    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_SPROCKET
    controller._config = AppConfig()
    controller._config.paths.models_dir = str(tmp_path / "models")

    assert controller._dynamic_model_path().name == "dynamic_sequence_sprocket.pkl"
    assert (
        controller._dynamic_classes_path().name
        == "dynamic_sequence_sprocket_classes.json"
    )
    assert (
        controller._dynamic_feature_dim_path().name
        == "dynamic_sequence_sprocket_feature_dim.txt"
    )
    assert (
        controller._dynamic_feature_mode_path().name
        == "dynamic_sequence_sprocket_feature_mode.txt"
    )
    assert (
        controller._dynamic_prototypes_path().name
        == "dynamic_sequence_sprocket_prototypes.json"
    )


def test_sequence_shapelet_dynamic_profile_uses_own_metadata_files(tmp_path):
    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET
    controller._config = AppConfig()
    controller._config.paths.models_dir = str(tmp_path / "models")

    assert controller._dynamic_model_path().name == "dynamic_sequence_shapelet.pkl"
    assert (
        controller._dynamic_classes_path().name
        == "dynamic_sequence_shapelet_classes.json"
    )
    assert (
        controller._dynamic_feature_dim_path().name
        == "dynamic_sequence_shapelet_feature_dim.txt"
    )
    assert (
        controller._dynamic_feature_mode_path().name
        == "dynamic_sequence_shapelet_feature_mode.txt"
    )
    assert (
        controller._dynamic_prototypes_path().name
        == "dynamic_sequence_shapelet_prototypes.json"
    )


def test_sequence_shapelet_72_dynamic_profile_uses_own_metadata_files(tmp_path):
    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_SHAPELET_72
    controller._config = AppConfig()
    controller._config.paths.models_dir = str(tmp_path / "models")

    assert controller._dynamic_model_path().name == "dynamic_sequence_shapelet_72.pkl"
    assert (
        controller._dynamic_classes_path().name
        == "dynamic_sequence_shapelet_72_classes.json"
    )
    assert (
        controller._dynamic_feature_dim_path().name
        == "dynamic_sequence_shapelet_72_feature_dim.txt"
    )
    assert (
        controller._dynamic_feature_mode_path().name
        == "dynamic_sequence_shapelet_72_feature_mode.txt"
    )
    assert (
        controller._dynamic_prototypes_path().name
        == "dynamic_sequence_shapelet_72_prototypes.json"
    )
    assert controller._dynamic_recognition_window() == 72


def test_sequence_phase_hmm_dynamic_profile_uses_own_metadata_files(tmp_path):
    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_PHASE_HMM
    controller._config = AppConfig()
    controller._config.paths.models_dir = str(tmp_path / "models")

    assert controller._dynamic_model_path().name == "dynamic_sequence_phase_hmm.pkl"
    assert (
        controller._dynamic_classes_path().name
        == "dynamic_sequence_phase_hmm_classes.json"
    )
    assert (
        controller._dynamic_feature_dim_path().name
        == "dynamic_sequence_phase_hmm_feature_dim.txt"
    )
    assert (
        controller._dynamic_feature_mode_path().name
        == "dynamic_sequence_phase_hmm_feature_mode.txt"
    )
    assert (
        controller._dynamic_prototypes_path().name
        == "dynamic_sequence_phase_hmm_prototypes.json"
    )


def test_sequence_gru_backbone_dynamic_profile_uses_own_metadata_files(tmp_path):
    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_GRU_BACKBONE
    controller._config = AppConfig()
    controller._config.paths.models_dir = str(tmp_path / "models")

    assert controller._dynamic_model_path().name == "dynamic_sequence_gru_backbone.pkl"
    assert (
        controller._dynamic_classes_path().name
        == "dynamic_sequence_gru_backbone_classes.json"
    )
    assert (
        controller._dynamic_feature_dim_path().name
        == "dynamic_sequence_gru_backbone_feature_dim.txt"
    )
    assert (
        controller._dynamic_feature_mode_path().name
        == "dynamic_sequence_gru_backbone_feature_mode.txt"
    )
    assert (
        controller._dynamic_prototypes_path().name
        == "dynamic_sequence_gru_backbone_prototypes.json"
    )


def test_sequence_lstm_backbone_dynamic_profile_uses_own_metadata_files(tmp_path):
    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_LSTM_BACKBONE
    controller._config = AppConfig()
    controller._config.paths.models_dir = str(tmp_path / "models")

    assert controller._dynamic_model_path().name == "dynamic_sequence_lstm_backbone.pkl"
    assert (
        controller._dynamic_classes_path().name
        == "dynamic_sequence_lstm_backbone_classes.json"
    )
    assert (
        controller._dynamic_feature_dim_path().name
        == "dynamic_sequence_lstm_backbone_feature_dim.txt"
    )
    assert (
        controller._dynamic_feature_mode_path().name
        == "dynamic_sequence_lstm_backbone_feature_mode.txt"
    )
    assert (
        controller._dynamic_prototypes_path().name
        == "dynamic_sequence_lstm_backbone_prototypes.json"
    )


def test_dynamic_profile_falls_back_to_production_classifier_for_experiment_variant(
    monkeypatch,
    tmp_path,
):
    def resolve_under_tmp(value):
        path = Path(value)
        return path if path.is_absolute() else tmp_path / path

    monkeypatch.setattr(controller_module, "resolve_config_path", resolve_under_tmp)
    production_dir = tmp_path / MODEL_VARIANT_DIRS[MODEL_VARIANT_PRODUCTION]
    production_dir.mkdir(parents=True)
    for name in (
        "dynamic_sequence_rocket.pkl",
        "dynamic_sequence_rocket_classes.json",
        "dynamic_sequence_rocket_feature_dim.txt",
        "dynamic_sequence_rocket_feature_mode.txt",
    ):
        (production_dir / name).write_text("x", encoding="utf-8")

    variant_rel = "models/experiments/dynamic_prototype/prototype_distance"
    variant_dir = tmp_path / variant_rel
    variant_dir.mkdir(parents=True)
    (variant_dir / "dynamic_sequence_rocket_prototypes.json").write_text(
        "{}", encoding="utf-8"
    )

    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET
    controller._config = AppConfig()
    controller._config.paths.models_dir = variant_rel

    assert controller._dynamic_model_path() == (
        production_dir / "dynamic_sequence_rocket.pkl"
    )
    assert controller._dynamic_classes_path() == (
        production_dir / "dynamic_sequence_rocket_classes.json"
    )
    assert controller._dynamic_feature_dim_path() == (
        production_dir / "dynamic_sequence_rocket_feature_dim.txt"
    )
    assert controller._dynamic_feature_mode_path() == (
        production_dir / "dynamic_sequence_rocket_feature_mode.txt"
    )
    assert controller._dynamic_prototypes_path() == (
        variant_dir / "dynamic_sequence_rocket_prototypes.json"
    )


def test_dynamic_profile_does_not_substitute_another_model_when_artifact_missing(
    monkeypatch,
    tmp_path,
):
    def resolve_under_tmp(value):
        path = Path(value)
        return path if path.is_absolute() else tmp_path / path

    monkeypatch.setattr(controller_module, "resolve_config_path", resolve_under_tmp)
    production_dir = tmp_path / MODEL_VARIANT_DIRS[MODEL_VARIANT_PRODUCTION]
    production_dir.mkdir(parents=True)
    for name in (
        "dynamic_sequence_mlp.pkl",
        "dynamic_sequence_knn.pkl",
    ):
        (production_dir / name).write_text("x", encoding="utf-8")

    variant_rel = "models/experiments/dynamic_prototype/prototype_distance"
    variant_dir = tmp_path / variant_rel
    variant_dir.mkdir(parents=True)

    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET
    controller._config = AppConfig()
    controller._config.paths.models_dir = variant_rel

    expected = variant_dir / "dynamic_sequence_rocket.pkl"
    assert not expected.exists()
    assert controller._dynamic_model_path() == expected
