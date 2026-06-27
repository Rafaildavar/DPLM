import json
from pathlib import Path

import numpy as np

from scripts.mlops_dashboard import build_dashboard


def _write_taxonomy(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "default_type": "static",
                "types": {
                    "static": ["palm"],
                    "dynamic": ["swipe_left"],
                    "negative": ["no_gesture_static"],
                },
                "patterns": {
                    "dynamic": ["swipe_*"],
                    "negative": ["no_gesture*"],
                },
            }
        ),
        encoding="utf-8",
    )


def _write_sample(root: Path, label: str) -> None:
    label_dir = root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    np.save(label_dir / "sample_0000.npy", np.zeros((36, 44), dtype=np.float32))


def test_mlops_dashboard_generates_html_and_summary(tmp_path):
    data_root = tmp_path / "data" / "gestures"
    models_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "dashboard"
    taxonomy_path = tmp_path / "gesture_taxonomy.json"
    models_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    _write_taxonomy(taxonomy_path)
    _write_sample(data_root, "swipe_left")
    _write_sample(data_root, "no_gesture_static")
    (models_dir / "dynamic_knn.pkl").write_bytes(b"model")
    (log_dir / "live_evaluation.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "attempt",
                        "expected": "swipe_left",
                        "predicted": "swipe_left",
                        "result": "correct",
                        "route": "dynamic",
                        "dynamic_decision_source": "motion_first",
                    }
                ),
                json.dumps(
                    {
                        "event_type": "attempt",
                        "expected": "no_gesture_static",
                        "predicted": "",
                        "result": "missed",
                        "route": "none",
                        "dynamic_decision_source": "negative_rejected",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (log_dir / "runtime_performance.jsonl").write_text(
        json.dumps({"inference_ms_p95": 22.5, "target_fps": 30}),
        encoding="utf-8",
    )

    report = build_dashboard(
        data_root=data_root,
        models_dir=models_dir,
        taxonomy_path=taxonomy_path,
        log_dir=log_dir,
        out_dir=out_dir,
    )

    assert report["dataset"]["total_samples"] == 2
    assert report["live_evaluation"]["attempts"] == 2
    assert report["live_evaluation"]["negative_rejections"] == 1
    assert (out_dir / "index.html").exists()
    assert (out_dir / "summary.json").exists()
    html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "GestureFlow MLOps Dashboard" in html
    assert "dynamic_knn.pkl" in html
