import json

import numpy as np

from app.gesture_online_infer import GestureOnlineInfer
from cv.dynamic_prototype import (
    METHOD_PROTOTYPE_DISTANCE,
    METHOD_PROTOTYPE_DTW,
    DynamicSequenceRecord,
    fit_dynamic_prototype_model,
    normalize_dynamic_sequence,
    predict_dynamic_prototype,
)
from scripts.dynamic_prototype_experiments import (
    _copy_base_dynamic_artifacts,
    _write_dynamic_prototype_artifact_bundle,
    discover_records,
    evaluate_model,
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


class _NearTopCompoundClassifier:
    classes_ = np.asarray([0, 1, 2])

    def predict(self, _features):
        return np.asarray([0])

    def predict_proba(self, _features):
        return np.asarray([[0.54, 0.50, 0.02]])


class _ConfidentSwipeClassifier:
    classes_ = np.asarray([0, 1, 2])

    def predict(self, _features):
        return np.asarray([0])

    def predict_proba(self, _features):
        return np.asarray([[0.94, 0.03, 0.03]])


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


def test_discover_records_uses_include_labels_as_user_dynamic_scope(tmp_path):
    data_root = tmp_path / "gestures"
    swipe_dir = data_root / "SwipeLeft"
    static_dir = data_root / "Palm"
    negative_dir = data_root / "random_motion"
    for directory in (swipe_dir, static_dir, negative_dir):
        directory.mkdir(parents=True)

    dynamic_sample = np.zeros((8, 65), dtype=np.float32)
    static_sample = np.zeros((8, 63), dtype=np.float32)
    np.save(swipe_dir / "sample_0000.npy", dynamic_sample)
    np.save(static_dir / "sample_0000.npy", static_sample)
    np.save(negative_dir / "sample_0000.npy", dynamic_sample)
    (swipe_dir / "sample_0000.meta.json").write_text(
        json.dumps({"include_global_motion": True}),
        encoding="utf-8",
    )

    records = discover_records(
        data_root=data_root,
        external_negative_root=None,
        target_dim=65,
        target_frames=6,
        include_labels=["SwipeLeft", "random_motion"],
    )

    by_label = {record.label: record for record in records}
    assert set(by_label) == {"SwipeLeft", "random_motion"}
    assert by_label["SwipeLeft"].is_negative is False
    assert by_label["random_motion"].is_negative is True


def test_normalize_dynamic_sequence_normalizes_xyz_z_channel():
    sequence = np.zeros((4, 65), dtype=np.float32)
    sequence[:, 8 * 3 + 1] = 0.5
    sequence[:, 2] = 0.2
    sequence[:, 8 * 3 + 2] = 0.7

    normalized = normalize_dynamic_sequence(sequence, target_dim=65, target_frames=6)

    assert np.allclose(normalized[:, 0 * 3 + 2], 0.0)
    assert np.allclose(normalized[:, 8 * 3 + 2], 1.0)


def test_prototype_labels_preserve_user_recorded_label_case():
    payload = fit_dynamic_prototype_model(
        [
            DynamicSequenceRecord("UpAndLeft", _left_sequence(14)),
            DynamicSequenceRecord("UpAndLeft", _left_sequence(22)),
            DynamicSequenceRecord("random_motion", _right_sequence(), is_negative=True),
        ],
        method=METHOD_PROTOTYPE_DISTANCE,
        threshold_floor=0.001,
    )

    decision = predict_dynamic_prototype(payload, _left_sequence(18))

    assert payload["positive_labels"] == ["UpAndLeft"]
    assert decision["accepted"] is True
    assert decision["label"] == "UpAndLeft"


def test_dynamic_prototype_evaluation_compares_labels_case_insensitively():
    payload = fit_dynamic_prototype_model(
        [
            DynamicSequenceRecord("UpAndLeft", _left_sequence(14)),
            DynamicSequenceRecord("UpAndLeft", _left_sequence(22)),
            DynamicSequenceRecord("random_motion", _right_sequence(), is_negative=True),
        ],
        method=METHOD_PROTOTYPE_DISTANCE,
        threshold_floor=0.001,
    )

    assert payload["positive_labels"] == ["UpAndLeft"]
    assert predict_dynamic_prototype(payload, _left_sequence(18))["label"] == "UpAndLeft"

    metrics, attempts = evaluate_model(
        payload,
        [DynamicSequenceRecord("UpAndLeft", _left_sequence(18))],
    )

    assert metrics["positive_recall"] == 1.0
    assert metrics["per_label"]["UpAndLeft"]["correct"] == 1
    assert attempts[0].expected == "UpAndLeft"
    assert attempts[0].predicted == "UpAndLeft"


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


def test_online_dynamic_model_can_override_far_positive_prototype_reject():
    infer = object.__new__(GestureOnlineInfer)
    infer._classes = ["swipe_left", "swipe_up", "random_motion"]
    infer._clf = _ConfidentSwipeClassifier()
    infer._dynamic_prototypes = {"positive_labels": ["swipe_left"]}
    infer._gesture_taxonomy = None
    infer._dynamic_prototype_decision = lambda _sequence: {
        "accepted": False,
        "reason": "far_from_prototype",
        "nearest_type": "positive",
        "nearest_label": "swipe_left",
        "label": "",
        "confidence": 0.0,
    }

    label, confidence = infer._dynamic_prediction(
        np.zeros((1, 1584), dtype=np.float32),
        {
            "dx": -0.20,
            "dy": 0.19,
            "path_length": 0.55,
            "displacement": 0.28,
            "direction_cos": -0.72,
            "direction_sin": 0.69,
        },
        sequence=_left_sequence(18),
    )

    assert label == "swipe_left"
    assert confidence == 0.94
    assert infer._last_dynamic_decision["source"] == "model_over_prototype_reject"
    assert infer._last_dynamic_decision["prototype_reason"] == "far_from_prototype"
    assert infer._last_dynamic_decision["model_override_margin"] > 0.80


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


def test_online_dynamic_compound_model_can_override_near_top_swipe_motion():
    infer = object.__new__(GestureOnlineInfer)
    infer._classes = ["swipe_up", "upandleft", "random_motion"]
    infer._clf = _NearTopCompoundClassifier()
    infer._dynamic_prototypes = {}
    infer._gesture_taxonomy = None

    label, confidence = infer._dynamic_prediction(
        np.zeros((1, 1584), dtype=np.float32),
        {
            "dx": -0.20,
            "dy": -0.32,
            "path_length": 0.48,
            "displacement": 0.38,
            "direction_cos": -0.52,
            "direction_sin": -0.85,
        },
        sequence=_sequence(
            np.linspace(0.65, 0.45, 18),
            np.linspace(0.75, 0.43, 18),
        ),
    )

    assert label == "upandleft"
    assert confidence == 0.50
    assert (
        infer._last_dynamic_decision["source"]
        == "complex_model_near_top_over_motion"
    )
    assert infer._last_dynamic_decision["motion_label"] == "swipe_up"


def test_online_dynamic_motion_first_is_suppressed_when_custom_labels_exist():
    infer = object.__new__(GestureOnlineInfer)
    infer._classes = ["swipe_up", "upandleft", "random_motion"]
    infer._clf = None
    infer._dynamic_prototypes = {}
    infer._gesture_taxonomy = None

    label, confidence = infer._dynamic_prediction(
        np.empty((1, 0), dtype=np.float32),
        {
            "dx": 0.0,
            "dy": -0.5,
            "path_length": 0.5,
            "displacement": 0.5,
            "direction_cos": 0.0,
            "direction_sin": -1.0,
        },
        sequence=_up_sequence(18),
    )

    assert label == ""
    assert confidence == 0.0
    assert (
        infer._last_dynamic_decision["source"]
        == "motion_fallback_suppressed_for_custom_labels"
    )
    assert infer._last_dynamic_decision["motion_label"] == "swipe_up"


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
