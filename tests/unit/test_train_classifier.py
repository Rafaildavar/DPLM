import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cv.gesture_features import DYNAMIC_TRAJECTORY_FEATURE_DIM
from cv.train_classifier import (
    _log_mlflow_run,
    build_classifier,
    default_rejection_metadata_path,
    load_dataset,
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


def test_build_classifier_supports_non_knn_models():
    clf = build_classifier("extra_trees", random_state=7)

    assert clf.__class__.__name__ == "ExtraTreesClassifier"


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
        mlflow_experiment="GestureFlow",
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
    assert calls["experiment"] == "GestureFlow"
    assert calls["run_name"] == "dynamic-test"
    assert calls["params"]["include_labels"] == "swipe_up,no_gesture_static"
    assert calls["metrics"]["train_accuracy"] == 0.95
    assert calls["artifacts"] == [
        "model.pkl",
        "classes.json",
        "feature_dim.txt",
        "feature_mode.txt",
        "gesture_rejection.json",
    ]
