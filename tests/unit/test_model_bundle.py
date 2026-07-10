from pathlib import Path

import joblib
import pytest

from cv.model_bundle import publish_model_bundle_atomic


def test_publish_model_bundle_installs_model_and_sidecars(tmp_path):
    model_path = tmp_path / "models" / "classifier.pkl"
    classes_path = tmp_path / "models" / "classes.json"
    mode_path = tmp_path / "models" / "feature_mode.txt"

    publish_model_bundle_atomic(
        {"version": 2},
        model_path,
        {
            classes_path: '["SwipeLeft"]',
            mode_path: "dynamic_landmark_image",
        },
    )

    assert joblib.load(model_path) == {"version": 2}
    assert classes_path.read_text(encoding="utf-8") == '["SwipeLeft"]'
    assert mode_path.read_text(encoding="utf-8") == "dynamic_landmark_image"


def test_publish_model_bundle_restores_previous_version_on_failure(
    monkeypatch,
    tmp_path,
):
    import cv.model_bundle as model_bundle

    model_path = tmp_path / "classifier.pkl"
    classes_path = tmp_path / "classes.json"
    joblib.dump({"version": 1}, model_path)
    classes_path.write_text('["old"]', encoding="utf-8")
    original_replace = model_bundle.os.replace
    failed_once = False

    def fail_model_activation(source, destination):
        nonlocal failed_once
        if (
            not failed_once
            and Path(destination) == model_path
            and Path(source).suffix == ".tmp"
        ):
            failed_once = True
            raise OSError("simulated activation failure")
        return original_replace(source, destination)

    monkeypatch.setattr(model_bundle.os, "replace", fail_model_activation)

    with pytest.raises(OSError, match="activation failure"):
        publish_model_bundle_atomic(
            {"version": 2},
            model_path,
            {classes_path: '["new"]'},
        )

    assert joblib.load(model_path) == {"version": 1}
    assert classes_path.read_text(encoding="utf-8") == '["old"]'
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))
