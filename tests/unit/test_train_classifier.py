import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cv.gesture_features import (
    DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES,
    DYNAMIC_TRAJECTORY_FEATURE_DIM,
    FEATURE_DYNAMIC_LANDMARK_IMAGE,
    FEATURE_STATIC_CRAFT_FULL_STATS,
    FEATURE_STATIC_LANDMARK_IMAGE,
    feature_vector_size,
)
from cv.sequence_multirocket import RandomMultiRocketSequenceTransformer
from cv.sequence_phase_hmm import PhaseHMMSequenceClassifier
from cv.sequence_gru_backbone import TorchGRUBackboneClassifier, TorchLSTMBackboneClassifier
from cv.sequence_rocket import RandomConvolutionSequenceTransformer
from cv.sequence_shapelet import ShapeletSequenceTransformer
from cv.sequence_sprocket import SprocketSequenceTransformer
from cv.train_classifier import (
    _log_mlflow_run,
    balance_training_set,
    build_classifier,
    can_use_sequence_mlp_validation_split,
    default_rejection_metadata_path,
    load_dataset,
    model_uses_internal_class_balance,
    tune_extra_trees_hyperparameters,
)


def _write_sample(root: Path, label: str, index: int, value: float = 0.0) -> None:
    label_dir = root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        label_dir / f"sample_{index:04d}.npy",
        np.full((3, 21, 2), value, dtype=np.float32),
    )


def test_load_dataset_can_filter_and_normalize_labels(tmp_path):
    data_root = tmp_path / "gestures"
    _write_sample(data_root, "New", 0, value=1.0)
    _write_sample(data_root, "new2", 0, value=2.0)
    _write_sample(data_root, "zoom", 0, value=3.0)

    x, y, classes = load_dataset(
        data_root,
        expect_dim=42,
        include_labels=["new", "new2"],
        lowercase_labels=True,
    )

    assert x.shape == (2, 42)
    assert y.tolist() == [0, 1]
    assert classes == ["new", "new2"]


def test_load_dataset_supports_dynamic_feature_mode(tmp_path):
    data_root = tmp_path / "gestures"
    _write_sample(data_root, "swipe", 0, value=0.0)
    _write_sample(data_root, "circle", 0, value=1.0)

    x, y, classes = load_dataset(
        data_root,
        include_labels=["swipe", "circle"],
        feature_mode="dynamic_stats",
    )

    assert x.shape == (2, 42 * 6 + DYNAMIC_TRAJECTORY_FEATURE_DIM)
    assert y.tolist() == [0, 1]
    assert classes == ["circle", "swipe"]


def test_load_dataset_expect_dim_expands_dynamic_feature_size(tmp_path):
    data_root = tmp_path / "gestures"
    _write_sample(data_root, "swipe", 0, value=0.0)
    _write_sample(data_root, "circle", 0, value=1.0)

    x, _, _ = load_dataset(
        data_root,
        expect_dim=44,
        include_labels=["swipe", "circle"],
        feature_mode="dynamic_stats",
    )

    assert x.shape == (2, 44 * 6 + DYNAMIC_TRAJECTORY_FEATURE_DIM)


def test_load_dataset_supports_static_craft_full_feature_mode(tmp_path):
    data_root = tmp_path / "gestures"
    _write_sample(data_root, "palm", 0, value=0.0)
    _write_sample(data_root, "gun", 0, value=1.0)

    x, y, classes = load_dataset(
        data_root,
        expect_dim=63,
        include_labels=["palm", "gun"],
        feature_mode=FEATURE_STATIC_CRAFT_FULL_STATS,
    )

    assert x.shape == (2, feature_vector_size(FEATURE_STATIC_CRAFT_FULL_STATS, 63))
    assert y.tolist() == [0, 1]
    assert classes == ["gun", "palm"]


def test_load_dataset_supports_static_landmark_image_feature_mode(tmp_path):
    data_root = tmp_path / "gestures"
    _write_sample(data_root, "palm", 0, value=0.0)
    _write_sample(data_root, "gun", 0, value=1.0)

    x, y, classes = load_dataset(
        data_root,
        expect_dim=63,
        include_labels=["palm", "gun"],
        feature_mode=FEATURE_STATIC_LANDMARK_IMAGE,
    )

    assert x.shape == (2, feature_vector_size(FEATURE_STATIC_LANDMARK_IMAGE, 63))
    assert y.tolist() == [0, 1]
    assert classes == ["gun", "palm"]


def test_load_dataset_supports_dynamic_landmark_image_feature_mode(tmp_path):
    data_root = tmp_path / "gestures"
    _write_sample(data_root, "swipe", 0, value=0.0)
    _write_sample(data_root, "circle", 0, value=1.0)

    x, y, classes = load_dataset(
        data_root,
        expect_dim=65,
        include_labels=["swipe", "circle"],
        feature_mode=FEATURE_DYNAMIC_LANDMARK_IMAGE,
    )

    assert x.shape == (2, feature_vector_size(FEATURE_DYNAMIC_LANDMARK_IMAGE, 65))
    assert y.tolist() == [0, 1]
    assert classes == ["circle", "swipe"]


def test_build_classifier_supports_non_knn_models():
    clf = build_classifier("extra_trees", random_state=7)

    assert clf.__class__.__name__ == "ExtraTreesClassifier"


def test_build_classifier_supports_static_stacking_model():
    clf = build_classifier("static_stacking", random_state=7, static_stacking_cv_folds=2)

    assert clf.__class__.__name__ == "StackingClassifier"
    assert clf.cv == 2


def test_build_classifier_supports_static_landmark_cnn_model():
    clf = build_classifier(
        "static_landmark_cnn",
        random_state=7,
        static_cnn_max_epochs=3,
        static_cnn_batch_size=4,
    )

    assert clf.__class__.__name__ == "KerasStaticLandmarkCNNClassifier"
    assert clf.max_epochs == 3
    assert clf.batch_size == 4


def test_build_classifier_supports_dynamic_landmark_cnn_model():
    clf = build_classifier(
        "dynamic_landmark_cnn",
        random_state=7,
        static_cnn_max_epochs=3,
        static_cnn_batch_size=4,
    )

    assert clf.__class__.__name__ == "KerasStaticLandmarkCNNClassifier"
    assert clf.target_frames == DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES
    assert clf.feature_name == "dynamic_landmark_cnn"
    assert clf.max_epochs == 3
    assert clf.batch_size == 4


def test_build_classifier_allows_extra_trees_overrides():
    clf = build_classifier(
        "extra_trees",
        random_state=7,
        extra_trees_n_estimators=40,
        extra_trees_max_depth=6,
        extra_trees_min_samples_split=4,
        extra_trees_min_samples_leaf=2,
        extra_trees_max_features=0.5,
        extra_trees_criterion="entropy",
        extra_trees_bootstrap=True,
    )

    assert clf.__class__.__name__ == "ExtraTreesClassifier"
    assert clf.n_estimators == 40
    assert clf.max_depth == 6
    assert clf.min_samples_split == 4
    assert clf.min_samples_leaf == 2
    assert clf.max_features == 0.5
    assert clf.criterion == "entropy"
    assert clf.bootstrap is True


def test_extra_trees_optuna_tuning_returns_effective_params():
    pytest.importorskip("optuna")
    rng = np.random.default_rng(123)
    x = rng.normal(size=(18, 10)).astype(np.float32)
    x[6:12] += 0.6
    x[12:] -= 0.6
    y = np.asarray([0] * 6 + [1] * 6 + [2] * 6, dtype=np.int64)

    summary = tune_extra_trees_hyperparameters(
        x,
        y,
        ["a", "b", "c"],
        n_trials=2,
        cv_folds=3,
        timeout=30,
        random_state=7,
        class_balance="none",
        boost_labels=[],
    )

    assert summary["trials"] == 2
    assert summary["used_cv"] is True
    assert summary["cv_folds"] == 3
    assert 0.0 <= summary["best_score"] <= 1.0
    params = summary["best_params"]
    assert params["extra_trees_n_estimators"] >= 120
    assert params["extra_trees_min_samples_leaf"] >= 1


def test_build_classifier_supports_sequence_mlp_model():
    clf = build_classifier("sequence_mlp", random_state=7)

    assert clf.__class__.__name__ == "Pipeline"
    mlp = clf.steps[-1][1]
    assert mlp.__class__.__name__ == "MLPClassifier"
    assert mlp.alpha == 1e-3
    assert mlp.early_stopping is True
    assert mlp.validation_fraction == 0.20
    assert mlp.n_iter_no_change == 30


def test_build_classifier_supports_sequence_rocket_model():
    clf = build_classifier(
        "sequence_rocket",
        random_state=7,
        sequence_rocket_kernels=16,
        sequence_rocket_max_dilation=3,
        sequence_rocket_max_channels_per_kernel=4,
    )

    assert clf.__class__.__name__ == "Pipeline"
    assert clf.steps[0][1].__class__.__name__ == "RandomConvolutionSequenceTransformer"
    assert clf.steps[-1][1].__class__.__name__ == "LogisticRegression"


def test_build_classifier_supports_sequence_multirocket_model():
    clf = build_classifier(
        "sequence_multirocket",
        random_state=7,
        sequence_multirocket_kernels=16,
        sequence_multirocket_max_dilation=3,
        sequence_multirocket_max_channels_per_kernel=4,
    )

    assert clf.__class__.__name__ == "Pipeline"
    assert clf.steps[0][1].__class__.__name__ == "RandomMultiRocketSequenceTransformer"
    assert clf.steps[-1][1].__class__.__name__ == "LogisticRegression"


def test_build_classifier_supports_sequence_sprocket_model():
    clf = build_classifier(
        "sequence_sprocket",
        random_state=7,
        sequence_sprocket_kernels=12,
        sequence_sprocket_prototypes_per_class=2,
        sequence_sprocket_max_dilation=3,
        sequence_sprocket_max_channels_per_kernel=4,
    )

    assert clf.__class__.__name__ == "Pipeline"
    assert clf.steps[0][1].__class__.__name__ == "SprocketSequenceTransformer"
    assert clf.steps[-1][1].__class__.__name__ == "LogisticRegression"


def test_build_classifier_supports_sequence_shapelet_model():
    clf = build_classifier(
        "sequence_shapelet",
        random_state=7,
        sequence_shapelets_per_class=6,
        sequence_shapelet_max_channels=4,
    )

    assert clf.__class__.__name__ == "Pipeline"
    assert clf.steps[0][1].__class__.__name__ == "ShapeletSequenceTransformer"
    assert clf.steps[-1][1].__class__.__name__ == "LogisticRegression"


def test_build_classifier_supports_sequence_phase_hmm_model():
    clf = build_classifier(
        "sequence_phase_hmm",
        sequence_phase_hmm_states=4,
        sequence_phase_hmm_max_channels=5,
        sequence_phase_hmm_variance_regularization=0.3,
    )

    assert isinstance(clf, PhaseHMMSequenceClassifier)
    assert clf.n_states == 4
    assert clf.max_channels == 5
    assert clf.variance_regularization == 0.3


def test_build_classifier_supports_sequence_ensemble_model():
    clf = build_classifier(
        "sequence_ensemble",
        random_state=7,
        sequence_multirocket_kernels=8,
        sequence_sprocket_kernels=6,
        sequence_sprocket_prototypes_per_class=2,
        sequence_shapelets_per_class=3,
        sequence_phase_hmm_states=3,
        sequence_phase_hmm_max_channels=4,
    )

    assert clf.__class__.__name__ == "VotingClassifier"
    assert [name for name, _estimator in clf.estimators] == [
        "multirocket",
        "sprocket",
        "shapelet",
        "phase_hmm",
    ]
    assert clf.voting == "soft"


def test_build_classifier_supports_sequence_gru_backbone_model():
    clf = build_classifier(
        "sequence_gru_backbone",
        random_state=7,
        sequence_gru_backbone_dim=16,
        sequence_gru_hidden_dim=24,
        sequence_gru_layers=1,
        sequence_gru_max_epochs=3,
        sequence_gru_batch_size=4,
        sequence_gru_validation_fraction=0.0,
    )

    assert isinstance(clf, TorchGRUBackboneClassifier)
    assert clf.backbone_dim == 16
    assert clf.hidden_dim == 24
    assert clf.max_epochs == 3


def test_build_classifier_supports_sequence_lstm_backbone_model():
    clf = build_classifier(
        "sequence_lstm_backbone",
        random_state=7,
        sequence_lstm_backbone_dim=16,
        sequence_lstm_hidden_dim=24,
        sequence_lstm_layers=1,
        sequence_lstm_max_epochs=3,
        sequence_lstm_batch_size=4,
        sequence_lstm_validation_fraction=0.0,
    )

    assert isinstance(clf, TorchLSTMBackboneClassifier)
    assert clf.backbone_dim == 16
    assert clf.hidden_dim == 24
    assert clf.max_epochs == 3


def test_build_classifier_supports_dynamic_landmark_lstm_backbone_model():
    clf = build_classifier(
        "dynamic_landmark_lstm_backbone",
        random_state=7,
        sequence_lstm_backbone_dim=16,
        sequence_lstm_hidden_dim=24,
        sequence_lstm_layers=1,
        sequence_lstm_max_epochs=3,
        sequence_lstm_batch_size=4,
        sequence_lstm_validation_fraction=0.0,
    )

    assert isinstance(clf, TorchLSTMBackboneClassifier)
    assert clf.target_frames == DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES
    assert clf.feature_name == FEATURE_DYNAMIC_LANDMARK_IMAGE
    assert clf.sequence_model_name == "dynamic_landmark_lstm_backbone"
    assert clf.backbone_dim == 16
    assert clf.hidden_dim == 24
    assert clf.max_epochs == 3


def test_sequence_rocket_transformer_builds_deterministic_temporal_features():
    x = np.arange(4 * 36 * 4, dtype=np.float32).reshape(4, 36 * 4)
    transformer = RandomConvolutionSequenceTransformer(
        n_kernels=8,
        target_frames=36,
        max_channels_per_kernel=3,
        random_state=11,
    )

    features = transformer.fit_transform(x)
    repeated = transformer.transform(x)

    assert features.shape == (4, 16)
    assert np.allclose(features, repeated)
    assert transformer.n_features_in_ == 36 * 4
    assert transformer.n_channels_ == 4


def test_sequence_multirocket_transformer_builds_difference_features():
    x = np.arange(4 * 36 * 4, dtype=np.float32).reshape(4, 36 * 4)
    transformer = RandomMultiRocketSequenceTransformer(
        n_kernels=8,
        target_frames=36,
        max_channels_per_kernel=3,
        random_state=11,
    )

    features = transformer.fit_transform(x)
    repeated = transformer.transform(x)

    assert features.shape == (4, 40)
    assert np.allclose(features, repeated)
    assert any(kernel.use_difference for kernel in transformer.kernels_)


def test_sequence_shapelet_transformer_builds_match_features():
    x = np.arange(6 * 36 * 4, dtype=np.float32).reshape(6, 36 * 4)
    y = np.asarray([0, 0, 0, 1, 1, 1])
    transformer = ShapeletSequenceTransformer(
        shapelets_per_class=4,
        target_frames=36,
        max_channels_per_shapelet=3,
        random_state=11,
    )

    features = transformer.fit_transform(x, y)
    repeated = transformer.transform(x)

    assert features.shape == (6, 16)
    assert np.allclose(features, repeated)
    assert len(transformer.shapelets_) == 8


def test_sequence_rocket_pipeline_supports_predict_proba():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(12, 36 * 4)).astype(np.float32)
    x[6:] += 0.75
    y = np.asarray([0] * 6 + [1] * 6)
    clf = build_classifier(
        "sequence_rocket",
        random_state=3,
        sequence_rocket_kernels=12,
        sequence_rocket_max_channels_per_kernel=3,
    )

    clf.fit(x, y)
    proba = clf.predict_proba(x[:2])

    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_sequence_multirocket_pipeline_supports_predict_proba():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(12, 36 * 4)).astype(np.float32)
    x[6:] += 0.75
    y = np.asarray([0] * 6 + [1] * 6)
    clf = build_classifier(
        "sequence_multirocket",
        random_state=3,
        sequence_multirocket_kernels=12,
        sequence_multirocket_max_channels_per_kernel=3,
    )

    clf.fit(x, y)
    proba = clf.predict_proba(x[:2])

    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_sequence_sprocket_pipeline_supports_predict_proba():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(12, 36 * 4)).astype(np.float32)
    x[6:] += 0.75
    y = np.asarray([0] * 6 + [1] * 6)
    clf = build_classifier(
        "sequence_sprocket",
        random_state=3,
        sequence_sprocket_kernels=8,
        sequence_sprocket_prototypes_per_class=2,
        sequence_sprocket_max_channels_per_kernel=3,
    )

    clf.fit(x, y)
    transformer = clf.steps[0][1]
    proba = clf.predict_proba(x[:2])

    assert isinstance(transformer, SprocketSequenceTransformer)
    assert transformer.prototypes_.shape[0] == 4
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_sequence_shapelet_pipeline_supports_predict_proba():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(12, 36 * 4)).astype(np.float32)
    x[6:] += 0.75
    y = np.asarray([0] * 6 + [1] * 6)
    clf = build_classifier(
        "sequence_shapelet",
        random_state=3,
        sequence_shapelets_per_class=4,
        sequence_shapelet_max_channels=3,
    )

    clf.fit(x, y)
    transformer = clf.steps[0][1]
    proba = clf.predict_proba(x[:2])

    assert isinstance(transformer, ShapeletSequenceTransformer)
    assert len(transformer.shapelets_) == 8
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_sequence_phase_hmm_classifier_supports_predict_proba():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(12, 36 * 4)).astype(np.float32)
    x[6:] += 0.75
    y = np.asarray([0] * 6 + [1] * 6)
    clf = build_classifier(
        "sequence_phase_hmm",
        sequence_phase_hmm_states=4,
        sequence_phase_hmm_max_channels=3,
    )

    clf.fit(x, y)
    proba = clf.predict_proba(x[:2])

    assert isinstance(clf, PhaseHMMSequenceClassifier)
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_sequence_ensemble_pipeline_supports_predict_proba():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(12, 36 * 4)).astype(np.float32)
    x[6:] += 0.75
    y = np.asarray([0] * 6 + [1] * 6)
    clf = build_classifier(
        "sequence_ensemble",
        random_state=3,
        sequence_multirocket_kernels=6,
        sequence_multirocket_max_channels_per_kernel=3,
        sequence_sprocket_kernels=6,
        sequence_sprocket_prototypes_per_class=2,
        sequence_sprocket_max_channels_per_kernel=3,
        sequence_shapelets_per_class=2,
        sequence_shapelet_max_channels=3,
        sequence_phase_hmm_states=3,
        sequence_phase_hmm_max_channels=3,
    )

    clf.fit(x, y)
    proba = clf.predict_proba(x[:2])

    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_sequence_gru_backbone_classifier_supports_predict_proba():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(12, 36 * 4)).astype(np.float32)
    x[6:] += 0.75
    y = np.asarray([0] * 6 + [1] * 6)
    clf = build_classifier(
        "sequence_gru_backbone",
        random_state=3,
        sequence_gru_backbone_dim=12,
        sequence_gru_hidden_dim=16,
        sequence_gru_max_epochs=3,
        sequence_gru_batch_size=4,
        sequence_gru_validation_fraction=0.0,
    )

    clf.fit(x, y)
    proba = clf.predict_proba(x[:2])

    assert isinstance(clf, TorchGRUBackboneClassifier)
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_sequence_lstm_backbone_classifier_supports_predict_proba():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(12, 36 * 4)).astype(np.float32)
    x[6:] += 0.75
    y = np.asarray([0] * 6 + [1] * 6)
    clf = build_classifier(
        "sequence_lstm_backbone",
        random_state=3,
        sequence_lstm_backbone_dim=12,
        sequence_lstm_hidden_dim=16,
        sequence_lstm_max_epochs=3,
        sequence_lstm_batch_size=4,
        sequence_lstm_validation_fraction=0.0,
    )

    clf.fit(x, y)
    proba = clf.predict_proba(x[:2])

    assert isinstance(clf, TorchLSTMBackboneClassifier)
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_build_classifier_allows_sequence_mlp_validation_overrides():
    clf = build_classifier(
        "sequence_mlp",
        random_state=7,
        sequence_mlp_alpha=0.002,
        sequence_mlp_early_stopping=False,
        sequence_mlp_validation_fraction=0.25,
        sequence_mlp_n_iter_no_change=12,
    )

    mlp = clf.steps[-1][1]
    assert mlp.alpha == 0.002
    assert mlp.early_stopping is False
    assert mlp.validation_fraction == 0.25
    assert mlp.n_iter_no_change == 12


def test_sequence_mlp_validation_split_requires_enough_samples_per_class():
    assert can_use_sequence_mlp_validation_split(
        np.asarray([0, 0, 0, 1, 1, 1]),
        class_count=2,
        validation_fraction=0.33,
    )
    assert not can_use_sequence_mlp_validation_split(
        np.asarray([0, 1, 1, 1]),
        class_count=2,
        validation_fraction=0.25,
    )


def test_balance_training_set_oversamples_knn_and_boosts_weak_labels():
    X = np.arange(6, dtype=np.float32).reshape(6, 1)
    y = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int64)
    classes = ["other", "hend", "gun"]

    X_fit, y_fit, metadata = balance_training_set(
        X,
        y,
        classes,
        model_type="knn",
        strategy="auto",
        boost_labels=["hend", "gun"],
        boost_factor=1.5,
        random_state=7,
    )

    counts = np.bincount(y_fit, minlength=3)
    assert X_fit.shape[0] == y_fit.shape[0]
    assert metadata["effective"] == "oversample"
    assert counts.tolist() == [3, 5, 5]
    assert metadata["fit_class_counts"] == {"other": 3, "hend": 5, "gun": 5}


def test_balance_training_set_only_boosts_labels_for_weighted_models():
    X = np.arange(10, dtype=np.float32).reshape(10, 1)
    y = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 2], dtype=np.int64)
    classes = ["other", "hend", "gun"]

    _X_fit, y_fit, metadata = balance_training_set(
        X,
        y,
        classes,
        model_type="logreg",
        strategy="auto",
        boost_labels=["gun"],
        boost_factor=2.0,
        random_state=7,
    )

    assert model_uses_internal_class_balance("logreg")
    assert metadata["effective"] == "boost_labels"
    assert np.bincount(y_fit, minlength=3).tolist() == [5, 4, 2]


def test_default_rejection_metadata_path_keeps_dynamic_metadata_separate():
    assert (
        default_rejection_metadata_path(Path("models/knn.pkl"))
        == Path("models/gesture_rejection.json")
    )
    assert (
        default_rejection_metadata_path(Path("models/dynamic_knn.pkl"))
        == Path("models/dynamic_knn_rejection.json")
    )


def test_log_mlflow_run_records_training_metadata(monkeypatch, tmp_path):
    calls = {
        "tracking_uri": "",
        "experiment": "",
        "run_name": "",
        "params": {},
        "metrics": {},
        "artifacts": [],
    }

    class _Run:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return False

    class _FakeMlflow:
        def set_tracking_uri(self, value):
            calls["tracking_uri"] = value

        def set_experiment(self, value):
            calls["experiment"] = value

        def start_run(self, run_name=None):
            calls["run_name"] = run_name
            return _Run()

        def log_params(self, params):
            calls["params"] = dict(params)

        def log_metrics(self, metrics):
            calls["metrics"] = dict(metrics)

        def log_artifact(self, path):
            calls["artifacts"].append(Path(path).name)

    monkeypatch.setitem(sys.modules, "mlflow", _FakeMlflow())
    artifacts = []
    for name in (
        "model.pkl",
        "classes.json",
        "feature_dim.txt",
        "feature_mode.txt",
        "gesture_rejection.json",
    ):
        path = tmp_path / name
        path.write_text("x", encoding="utf-8")
        artifacts.append(path)

    args = SimpleNamespace(
        mlflow_experiment="GestureBind",
        mlflow_tracking_uri="sqlite:///mlflow.db",
        mlflow_run_name="dynamic-test",
        data_root="data/gestures",
        model_type="knn",
        feature_mode="dynamic_stats",
        neighbors=5,
        weights="distance",
        expect_dim=None,
        lowercase_labels=True,
        include_label=["swipe_up", "no_gesture_static"],
    )

    _log_mlflow_run(
        args=args,
        classes=["no_gesture_static", "swipe_up"],
        sample_count=40,
        feature_dim=271,
        train_accuracy=0.95,
        out_path=artifacts[0],
        classes_out=artifacts[1],
        feature_dim_out=artifacts[2],
        feature_mode_out=artifacts[3],
        rejection_out=artifacts[4],
        rejection_metadata={"negative_labels": ["no_gesture_static"]},
    )

    assert calls["tracking_uri"] == "sqlite:///mlflow.db"
    assert calls["experiment"] == "GestureBind"
    assert calls["run_name"] == "dynamic-test"
    assert calls["params"]["include_labels"] == "swipe_up,no_gesture_static"
    assert calls["params"]["sequence_mlp_early_stopping_effective"] is True
    assert calls["params"]["sequence_mlp_validation_fraction_effective"] == 0.20
    assert calls["params"]["sequence_mlp_alpha"] == 1e-3
    assert calls["metrics"]["train_accuracy"] == 0.95
    assert calls["artifacts"] == [
        "model.pkl",
        "classes.json",
        "feature_dim.txt",
        "feature_mode.txt",
        "gesture_rejection.json",
    ]
