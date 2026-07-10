import json
import sqlite3
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


def _write_mlflow_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    with conn:
        conn.executescript(
            """
            create table experiments (
                experiment_id integer primary key,
                name varchar(256),
                artifact_location varchar(256),
                lifecycle_stage varchar(32),
                creation_time bigint,
                last_update_time bigint
            );
            create table runs (
                run_uuid varchar(32) primary key,
                name varchar(250),
                source_type varchar(20),
                source_name varchar(500),
                entry_point_name varchar(50),
                user_id varchar(256),
                status varchar(9),
                start_time bigint,
                end_time bigint,
                source_version varchar(50),
                lifecycle_stage varchar(20),
                artifact_uri varchar(200),
                experiment_id integer,
                deleted_time bigint
            );
            create table latest_metrics (
                key varchar(250),
                value float,
                timestamp bigint,
                step bigint,
                is_nan boolean,
                run_uuid varchar(32)
            );
            create table params (
                key varchar(250),
                value varchar(8000),
                run_uuid varchar(32)
            );
            create table tags (
                key varchar(250),
                value varchar(8000),
                run_uuid varchar(32)
            );
            """
        )
        conn.execute(
            "insert into experiments values (1, 'GestureBind', '', 'active', 0, 0)"
        )
        conn.executemany(
            """
            insert into runs values (
                ?, ?, '', '', '', 'tester', 'FINISHED', ?, ?, '', 'active', '', 1, null
            )
            """,
            [
                ("train_run", "extra_trees-static_craft_full_stats", 1000, 2000),
                ("live_run", "live-swipe_left-auto-sequence_mlp-open_set_policy", 3000, 4000),
            ],
        )
        conn.executemany(
            "insert into params values (?, ?, ?)",
            [
                ("model_type", "extra_trees", "train_run"),
                ("feature_mode", "static_craft_full_stats", "train_run"),
                ("dynamic_model_profile", "sequence_mlp", "live_run"),
                ("recognition_model_mode", "auto", "live_run"),
                ("static_rejection_method", "open_set_policy", "live_run"),
                ("expected_label", "swipe_left", "live_run"),
                ("expected_type", "dynamic", "live_run"),
            ],
        )
        conn.executemany(
            "insert into tags values (?, ?, ?)",
            [
                ("run_kind", "training", "train_run"),
                ("run_kind", "live_evaluation", "live_run"),
            ],
        )
        conn.executemany(
            "insert into latest_metrics values (?, ?, 0, 0, 0, ?)",
            [
                ("train_accuracy", 0.91, "train_run"),
                ("sample_count", 40, "train_run"),
                ("class_count", 5, "train_run"),
                ("feature_dim", 512, "train_run"),
                ("live_accuracy", 0.8, "live_run"),
                ("live_recall", 0.75, "live_run"),
                ("live_dynamic_recall", 0.75, "live_run"),
                ("live_negative_false_positive_rate", 0.0, "live_run"),
                ("live_static_false_positive_rate", 0.0, "live_run"),
                ("live_static_hijack_rate", 0.0, "live_run"),
                ("live_wrong_dynamic_direction_rate", 0.1, "live_run"),
                ("live_latency_avg_s", 0.12, "live_run"),
                ("live_total", 20, "live_run"),
            ],
        )
    conn.close()


def test_mlops_dashboard_generates_html_and_summary(tmp_path):
    data_root = tmp_path / "data" / "gestures"
    models_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "dashboard"
    taxonomy_path = tmp_path / "gesture_taxonomy.json"
    mlflow_db = tmp_path / "mlflow.db"
    models_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    _write_taxonomy(taxonomy_path)
    _write_sample(data_root, "swipe_left")
    _write_sample(data_root, "no_gesture_static")
    _write_mlflow_db(mlflow_db)
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
    (log_dir / "live_usage_summary.json").write_text(
        json.dumps(
            {
                "total_events": 2,
                "command_attempts": 1,
                "executed_events": 1,
                "rejected_events": 1,
                "suppressed_events": 0,
                "cooldown_events": 0,
                "command_success_rate": 1.0,
                "avg_confidence": 0.93,
                "label_counts": {"swipe_left": 2},
                "route_counts": {"dynamic": 2},
                "event_type_counts": {
                    "command_executed": 1,
                    "gesture_rejected": 1,
                },
                "latest_event": {
                    "event_type": "command_executed",
                    "label": "swipe_left",
                    "confidence": 0.96,
                    "route": "dynamic",
                    "executed": True,
                    "reason": "",
                    "command_info": "swipe_left: Open App",
                },
                "latest_runtime": {"inference_ms_avg": 18.0},
            }
        ),
        encoding="utf-8",
    )

    report = build_dashboard(
        data_root=data_root,
        models_dir=models_dir,
        taxonomy_path=taxonomy_path,
        log_dir=log_dir,
        out_dir=out_dir,
        mlflow_db=mlflow_db,
    )

    assert report["dataset"]["total_samples"] == 2
    assert report["live_evaluation"]["attempts"] == 2
    assert report["live_evaluation"]["negative_rejections"] == 1
    assert report["live_usage"]["total_events"] == 2
    assert report["live_usage"]["command_success_rate"] == 1.0
    assert report["mlflow"]["available"] is True
    assert report["mlflow"]["runs"] == 2
    assert report["mlflow"]["training_leaderboard"][0]["approach"] == (
        "extra_trees + static_craft_full_stats"
    )
    assert report["mlflow"]["profile_scoreboard"][0]["approach"] == (
        "sequence_mlp / open_set_policy"
    )
    assert (out_dir / "index.html").exists()
    assert (out_dir / "summary.json").exists()
    html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "GestureBind MLOps Dashboard" in html
    assert "MLflow Top Approaches" in html
    assert "Live Usage" in html
    assert "command_executed" in html
    assert "sequence_mlp / open_set_policy" in html
    assert "extra_trees + static_craft_full_stats" in html
    assert "dynamic_knn.pkl" in html
