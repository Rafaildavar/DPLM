import json

import joblib
import numpy as np

from scripts.train_static_rejection_verifiers import (
    DEFAULT_METHODS,
    train_static_rejection_verifiers,
)


def _write_samples(root, label: str, center: float) -> None:
    label_dir = root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        seq = np.full((6, 42), center + index * 0.01, dtype=np.float32)
        seq[:, 0] = center
        seq[:, 1] = center + index * 0.02
        np.save(label_dir / f"sample_{index:04d}.npy", seq)


def test_train_static_rejection_verifiers_writes_all_method_payloads(tmp_path):
    data_root = tmp_path / "gestures"
    _write_samples(data_root, "palm", 0.10)
    _write_samples(data_root, "gun", 0.40)
    _write_samples(data_root, "no_gesture_static", 0.80)
    _write_samples(data_root, "random_motion", 1.10)
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(
        json.dumps(["palm", "gun", "no_gesture_static", "random_motion"]),
        encoding="utf-8",
    )
    out_path = tmp_path / "static_rejection_verifiers.pkl"

    payload = train_static_rejection_verifiers(
        data_root=data_root,
        classes_path=classes_path,
        out_path=out_path,
        methods=DEFAULT_METHODS,
        random_state=7,
    )

    assert out_path.exists()
    loaded = joblib.load(out_path)
    assert loaded["sample_count"] == 16
    assert loaded["positive_labels"] == ["gun", "palm"]
    assert loaded["negative_labels"] == ["no_gesture_static", "random_motion"]
    assert sorted(loaded["methods"]) == sorted(DEFAULT_METHODS)
    assert payload["methods"]["one_vs_rest_logreg"]["status"] == "ok"
    assert payload["methods"]["metric_nca_centroid"]["status"] == "ok"
