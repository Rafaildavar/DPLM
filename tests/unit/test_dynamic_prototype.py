import numpy as np

from app.gesture_online_infer import GestureOnlineInfer
from cv.dynamic_prototype import (
    METHOD_PROTOTYPE_DISTANCE,
    METHOD_PROTOTYPE_DTW,
    DynamicSequenceRecord,
    fit_dynamic_prototype_model,
    predict_dynamic_prototype,
)
from scripts.dynamic_prototype_experiments import (
    _copy_base_dynamic_artifacts,
    _write_dynamic_prototype_artifact_bundle,
    filter_conflicting_external_negatives,
)


def _sequence(xs, ys=None):
    ys = ys if ys is not None else [0.5] * len(xs)
    frames = []
    for x, y in zip(xs, ys):
        frame = np.zeros(44, dtype=np.float32)
        frame[-2:] = [x, y]
        frames.append(frame)
    return np.stack(frames, axis=0)


def _left_sequence(frames=16):
    return _sequence(np.linspace(0.82, 0.25, frames))


def _right_sequence(frames=16):
    return _sequence(np.linspace(0.25, 0.82, frames))


def _up_sequence(frames=16):
    return _sequence([0.5] * frames, np.linspace(0.82, 0.25, frames))


class _ComplexSequenceClassifier:
    classes_ = np.asarray([0, 1, 2])

    def predict(self, _features):
        return np.asarray([1])

    def predict_proba(self, _features):
        return np.asarray([[0.14, 0.78, 0.08]])


def test_prototype_distance_accepts_positive_and_rejects_negative():
    train = [
        DynamicSequenceRecord("swipe_left", _left_sequence(14)),
        DynamicSequenceRecord("swipe_left", _left_sequence(22)),
        DynamicSequenceRecord("random_motion", _right_sequence(), is_negative=True),
    ]
    payload = fit_dynamic_prototype_model(
        train,
        method=METHOD_PROTOTYPE_DISTANCE,
        threshold_floor=0.001,
    )

    accepted = predict_dynamic_prototype(payload, _left_sequence(18))
    rejected = predict_dynamic_prototype(payload, _right_sequence(18))

    assert accepted["accepted"] is True
    assert accepted["label"] == "swipe_left"
    assert rejected["accepted"] is False
    assert rejected["reason"] in {"nearest_negative", "far_from_prototype"}


def test_prototype_dtw_tolerates_timing_variation():
    train = [
        DynamicSequenceRecord("swipe_up", _up_sequence(10)),
        DynamicSequenceRecord("swipe_up", _up_sequence(28)),
        DynamicSequenceRecord("random_motion", _right_sequence(), is_negative=True),
    ]
    payload = fit_dynamic_prototype_model(
        train,
        method=METHOD_PROTOTYPE_DTW,
        threshold_floor=0.001,
    )

    decision = predict_dynamic_prototype(payload, _up_sequence(18))

    assert decision["accepted"] is True
    assert decision["label"] == "swipe_up"
    assert decision["method"] == METHOD_PROTOTYPE_DTW


def test_online_dynamic_prediction_uses_prototype_as_reject_verifier():
    payload = fit_dynamic_prototype_model(
        [
            DynamicSequenceRecord("swipe_left", _left_sequence(14)),
            DynamicSequenceRecord("swipe_left", _left_sequence(22)),
            DynamicSequenceRecord("random_motion", _right_sequence(), is_negative=True),
        ],
        method=METHOD_PROTOTYPE_DISTANCE,
        threshold_floor=0.001,
    )
    infer = object.__new__(GestureOnlineInfer)
    infer._classes = ["swipe_left", "swipe_right", "random_motion"]
    infer._clf = None
    infer._dynamic_prototypes = payload
    infer._gesture_taxonomy = None

    label, confidence = infer._dynamic_prediction(
        np.empty((1, 0), dtype=np.float32),
        {
            "dx": 0.5,
            "dy": 0.0,
            "path_length": 0.5,
            "displacement": 0.5,
            "direction_cos": 1.0,
            "direction_sin": 0.0,
        },
        sequence=_right_sequence(18),
    )

    assert label == ""
    assert confidence == 0.0
    assert infer._last_dynamic_decision["source"] == "prototype_rejected"
    assert infer._last_dynamic_decision["prototype_reason"] == "nearest_negative"


def test_online_dynamic_motion_can_override_positive_prototype_reject():
    infer = object.__new__(GestureOnlineInfer)
    infer._classes = ["swipe_down", "swipe_left", "swipe_up", "random_motion"]
    infer._clf = None
    infer._dynamic_prototypes = {"positive_labels": ["swipe_left"]}
    infer._gesture_taxonomy = None
    infer._dynamic_prototype_decision = lambda _sequence: {
        "accepted": False,
        "reason": "far_from_prototype",
        "label": "",
        "confidence": 0.0,
    }

    label, confidence = infer._dynamic_prediction(
        np.empty((1, 0), dtype=np.float32),
        {
            "dx": -0.5,
            "dy": 0.0,
            "path_length": 0.5,
            "displacement": 0.5,
            "direction_cos": -1.0,
            "direction_sin": 0.0,
        },
        sequence=_left_sequence(10),
    )

    assert label == "swipe_left"
    assert confidence >= 0.68
    assert infer._last_dynamic_decision["source"] == "motion_over_prototype_reject"
    assert infer._last_dynamic_decision["prototype_reason"] == "far_from_prototype"


def test_online_dynamic_motion_does_not_override_nearest_negative_reject():
    infer = object.__new__(GestureOnlineInfer)
    infer._classes = ["swipe_down", "swipe_left", "swipe_up", "random_motion"]
    infer._clf = None
    infer._dynamic_prototypes = {"positive_labels": ["swipe_left"]}
    infer._gesture_taxonomy = None
    infer._dynamic_prototype_decision = lambda _sequence: {
        "accepted": False,
        "reason": "nearest_negative",
        "nearest_type": "negative",
        "label": "",
        "confidence": 0.0,
    }

    label, confidence = infer._dynamic_prediction(
        np.empty((1, 0), dtype=np.float32),
        {
            "dx": -0.5,
            "dy": 0.0,
            "path_length": 0.5,
            "displacement": 0.5,
            "direction_cos": -1.0,
            "direction_sin": 0.0,
        },
        sequence=_left_sequence(10),
    )

    assert label == ""
    assert confidence == 0.0
    assert infer._last_dynamic_decision["source"] == "prototype_rejected"
    assert infer._last_dynamic_decision["prototype_reason"] == "nearest_negative"


def test_online_dynamic_complex_model_can_override_swipe_motion():
    infer = object.__new__(GestureOnlineInfer)
    infer._classes = ["swipe_left", "circle_clockwise", "random_motion"]
    infer._clf = _ComplexSequenceClassifier()
    infer._dynamic_prototypes = {}
    infer._gesture_taxonomy = None

    label, confidence = infer._dynamic_prediction(
        np.zeros((1, 1584), dtype=np.float32),
        {
            "dx": -0.5,
            "dy": 0.0,
            "path_length": 0.65,
            "displacement": 0.5,
            "direction_cos": -1.0,
            "direction_sin": 0.0,
        },
        sequence=_left_sequence(18),
    )

    assert label == "circle_clockwise"
    assert confidence == 0.78
    assert infer._last_dynamic_decision["source"] == "complex_model_over_motion"
    assert infer._last_dynamic_decision["motion_label"] == "swipe_left"
    assert infer._last_dynamic_decision["complex_model_margin"] > 0.50


def test_online_dynamic_complex_prototype_does_not_conflict_with_swipe_motion():
    infer = object.__new__(GestureOnlineInfer)
    infer._classes = ["swipe_left", "circle_clockwise", "random_motion"]
    infer._clf = None
    infer._dynamic_prototypes = {"positive_labels": ["circle_clockwise"]}
    infer._gesture_taxonomy = None
    infer._dynamic_prototype_decision = lambda _sequence: {
        "accepted": True,
        "label": "circle_clockwise",
        "confidence": 0.77,
        "reason": "accepted",
    }

    label, confidence = infer._dynamic_prediction(
        np.empty((1, 0), dtype=np.float32),
        {
            "dx": -0.5,
            "dy": 0.0,
            "path_length": 0.65,
            "displacement": 0.5,
            "direction_cos": -1.0,
            "direction_sin": 0.0,
        },
        sequence=_left_sequence(18),
    )

    assert label == "circle_clockwise"
    assert confidence == 0.77
    assert infer._last_dynamic_decision["source"] == "complex_prototype_over_motion"
    assert infer._last_dynamic_decision["motion_label"] == "swipe_left"


def test_external_negative_conflict_filter_keeps_user_positive_priority(tmp_path):
    external_root = tmp_path / "external" / "ipn_hand"
    negative_dir = external_root / "negative_external_ipn_dynamic"
    negative_dir.mkdir(parents=True)
    conflict_path = negative_dir / "sample_conflict.npy"
    safe_path = negative_dir / "sample_safe.npy"
    np.save(conflict_path, _left_sequence())
    np.save(safe_path, _right_sequence())

    records = [
        DynamicSequenceRecord("swipe_left", _left_sequence(14), path="internal/a.npy"),
        DynamicSequenceRecord("swipe_left", _left_sequence(22), path="internal/b.npy"),
        DynamicSequenceRecord(
            "negative_external_ipn_dynamic",
            _left_sequence(18),
            path=str(conflict_path),
            is_negative=True,
        ),
        DynamicSequenceRecord(
            "negative_external_ipn_dynamic",
            _right_sequence(18),
            path=str(safe_path),
            is_negative=True,
        ),
    ]

    filtered, report = filter_conflicting_external_negatives(
        records,
        external_negative_root=external_root,
        threshold_floor=0.001,
        conflict_margin=1.2,
    )

    paths = {record.path for record in filtered}
    assert str(conflict_path) not in paths
    assert str(safe_path) in paths
    assert report["conflict_count"] == 1
    assert report["safe_external_negative_count"] == 1
    assert report["conflicting_negative_labels"] == {
        "negative_external_ipn_dynamic": 1
    }


def test_copy_base_dynamic_artifacts_skips_same_file(tmp_path):
    model_path = tmp_path / "dynamic_knn.pkl"
    model_path.write_bytes(b"model")

    _copy_base_dynamic_artifacts(tmp_path, tmp_path)

    assert model_path.read_bytes() == b"model"


def test_dynamic_prototype_artifact_bundle_contains_dashboard(tmp_path):
    report = {
        "method": "prototype_distance",
        "metrics": {
            "overall_success": 0.95,
            "positive_recall": 0.90,
            "negative_reject_rate": 1.0,
            "negative_false_positive_rate": 0.0,
            "positive_wrong_rate": 0.0,
            "positive_reject_rate": 0.1,
            "per_label": {
                "swipe_left": {
                    "total": 10,
                    "correct": 9,
                    "wrong": 0,
                    "rejected": 1,
                }
            },
        },
        "thresholds": {
            "swipe_left": {
                "threshold": 0.12,
                "positive_radius": 0.13,
                "positive_distance_mean": 0.04,
                "positive_distance_p95": 0.08,
                "impostor_distance_p10": 0.20,
            }
        },
        "negative_conflict_filter": {
            "external_negative_total": 12,
            "safe_external_negative_count": 11,
            "conflict_count": 1,
            "conflict_rate": 1 / 12,
        },
    }
    payload = {
        "positive_labels": ["swipe_left"],
        "negative_labels": ["random_motion"],
        "prototypes": [{"label": "swipe_left"}],
    }

    _write_dynamic_prototype_artifact_bundle(
        tmp_path / "dynamic_prototype",
        report=report,
        payload=payload,
    )

    assert (tmp_path / "dynamic_prototype" / "index.html").exists()
    assert (tmp_path / "dynamic_prototype" / "charts" / "quality.svg").exists()
    assert (tmp_path / "dynamic_prototype" / "charts" / "per_class.svg").exists()
    thresholds = (tmp_path / "dynamic_prototype" / "thresholds.csv").read_text(
        encoding="utf-8"
    )
    assert "swipe_left" in thresholds
