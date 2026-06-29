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
