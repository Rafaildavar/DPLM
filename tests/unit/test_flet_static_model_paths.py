from pathlib import Path

from app.flet_app import controller as controller_module
from app.flet_app.controller import (
    AppController,
    DYNAMIC_MODEL_PROFILE_PRODUCTION,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_MLP,
    DYNAMIC_MODEL_PROFILE_SEQUENCE_ROCKET,
    MODEL_VARIANT_DIRS,
    MODEL_VARIANT_PRODUCTION,
)
from app.services.app_config import AppConfig


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
    (variant_dir / "dynamic_sequence_mlp.pkl").write_text("x", encoding="utf-8")

    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = DYNAMIC_MODEL_PROFILE_PRODUCTION
    controller._config = AppConfig()
    controller._config.paths.models_dir = variant_rel
    controller._config.paths.model_path = f"{variant_rel}/knn.pkl"
    controller._config.paths.classes_path = f"{variant_rel}/classes.json"
    controller._config.paths.feature_dim_path = f"{variant_rel}/feature_dim.txt"

    assert controller._configured_models_dir() == variant_dir
    assert controller._dynamic_model_path() == variant_dir / "dynamic_sequence_mlp.pkl"
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


def test_legacy_dynamic_profile_falls_back_to_sequence_mlp_paths(tmp_path):
    controller = AppController.__new__(AppController)
    controller._dynamic_model_profile = "sequence_knn"
    controller._config = AppConfig()
    controller._config.paths.models_dir = str(tmp_path / "models")

    assert controller.dynamic_model_profile == DYNAMIC_MODEL_PROFILE_PRODUCTION
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
