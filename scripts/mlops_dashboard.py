"""Generate a local HTML dashboard for GestureBind ML observability."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services.gesture_taxonomy import load_gesture_taxonomy
from cv.gesture_dataset_files import gesture_sample_paths

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = Path.home() / ".dplm" / "logs"
DEFAULT_OUT_DIR = ROOT / "docs" / "mlops_dashboard"
DEFAULT_MLFLOW_DB = ROOT / "mlflow.db"

TRAINING_SCORE_KEYS = (
    "static_cv_macro_f1",
    "cv_macro_f1",
    "macro_f1",
    "f1_macro",
    "validation_macro_f1",
    "val_macro_f1",
    "test_macro_f1",
    "sequence_lstm_optuna_best_score",
    "sequence_gru_optuna_best_score",
    "extra_trees_optuna_best_score",
    "optuna_best_score",
    "best_score",
    "train_accuracy",
    "accuracy",
)

TIMELINE_SCORE_KEYS = (
    "live_accuracy",
    *TRAINING_SCORE_KEYS,
    "positive_recall",
    "negative_false_positive_rate",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def dataset_summary(data_root: Path, taxonomy_path: Path) -> dict[str, Any]:
    taxonomy = load_gesture_taxonomy(taxonomy_path)
    classes: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    total_samples = 0
    if data_root.exists():
        for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
            label = label_dir.name
            samples = len(gesture_sample_paths(label_dir))
            gesture_type = taxonomy.gesture_type_for_label(label)
            classes.append(
                {
                    "label": label,
                    "type": gesture_type,
                    "samples": samples,
                }
            )
            type_counts[gesture_type] += samples
            total_samples += samples
    return {
        "data_root": str(data_root),
        "total_classes": len(classes),
        "total_samples": total_samples,
        "type_counts": dict(sorted(type_counts.items())),
        "classes": classes,
    }


def model_artifacts(models_dir: Path) -> list[dict[str, Any]]:
    if not models_dir.exists():
        return []
    artifacts: list[dict[str, Any]] = []
    for path in sorted(models_dir.iterdir()):
        if path.suffix not in {".pkl", ".json", ".txt"}:
            continue
        if not path.is_file():
            continue
        payload = path.read_bytes()
        artifacts.append(
            {
                "name": path.name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()[:16],
                "modified_at": path.stat().st_mtime,
            }
        )
    return artifacts


def live_evaluation_summary(log_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(log_dir / "live_evaluation.jsonl")
    attempts = [row for row in rows if row.get("event_type") == "attempt"]
    total = len(attempts)
    correct = sum(1 for row in attempts if row.get("result") == "correct")
    wrong = sum(1 for row in attempts if row.get("result") == "wrong")
    missed = sum(1 for row in attempts if row.get("result") == "missed")
    route_counts = Counter(str(row.get("route") or "unknown") for row in attempts)
    source_counts = Counter(
        str(row.get("dynamic_decision_source") or "unknown") for row in attempts
    )
    static_method_counts = Counter(
        str(row.get("static_rejection_method") or "unknown") for row in attempts
    )
    static_reject_reason_counts = Counter(
        str(row.get("static_reject_reason") or "none") for row in attempts
    )
    static_decision_counts = Counter(
        str(row.get("static_decision_source") or "unknown") for row in attempts
    )
    negative_rejections = sum(
        1 for row in attempts if row.get("dynamic_decision_source") == "negative_rejected"
    )
    by_expected: dict[str, Counter[str]] = defaultdict(Counter)
    for row in attempts:
        expected = str(row.get("expected") or row.get("expected_label") or "unknown")
        result = str(row.get("result") or "unknown")
        by_expected[expected][result] += 1
    labels = []
    for expected, counter in sorted(by_expected.items()):
        label_total = sum(counter.values())
        labels.append(
            {
                "expected": expected,
                "attempts": label_total,
                "correct": counter.get("correct", 0),
                "wrong": counter.get("wrong", 0),
                "missed": counter.get("missed", 0),
                "accuracy": (
                    counter.get("correct", 0) / label_total if label_total else 0.0
                ),
            }
        )
    return {
        "source": str(log_dir / "live_evaluation.jsonl"),
        "attempts": total,
        "correct": correct,
        "wrong": wrong,
        "missed": missed,
        "accuracy": correct / total if total else 0.0,
        "route_counts": dict(sorted(route_counts.items())),
        "dynamic_decision_sources": dict(sorted(source_counts.items())),
        "static_rejection_methods": dict(sorted(static_method_counts.items())),
        "static_rejection_reasons": dict(sorted(static_reject_reason_counts.items())),
        "static_decision_sources": dict(sorted(static_decision_counts.items())),
        "negative_rejections": negative_rejections,
        "labels": labels,
    }


def runtime_summary(log_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(log_dir / "runtime_performance.jsonl")
    latest = rows[-1] if rows else {}
    return {
        "source": str(log_dir / "runtime_performance.jsonl"),
        "samples": len(rows),
        "latest": latest,
    }


def live_usage_summary(log_dir: Path) -> dict[str, Any]:
    summary_path = log_dir / "live_usage_summary.json"
    events_path = log_dir / "live_usage_events.jsonl"
    if summary_path.exists():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            payload = dict(loaded)
            payload.setdefault("source", str(summary_path))
            payload.setdefault("events_source", str(events_path))
            payload.setdefault("total_events", 0)
            payload.setdefault("command_attempts", 0)
            payload.setdefault("executed_events", 0)
            payload.setdefault("rejected_events", 0)
            payload.setdefault("suppressed_events", 0)
            payload.setdefault("cooldown_events", 0)
            payload.setdefault("command_success_rate", 0.0)
            payload.setdefault("avg_confidence", 0.0)
            payload.setdefault("label_counts", {})
            payload.setdefault("route_counts", {})
            payload.setdefault("event_type_counts", {})
            payload.setdefault("latest_event", {})
            payload.setdefault("latest_runtime", {})
            return payload

    rows = read_jsonl(events_path)
    executed = sum(1 for row in rows if bool(row.get("executed")))
    command_events = [
        row
        for row in rows
        if str(row.get("event_type") or "")
        in {"command_executed", "command_rejected", "command_cooldown"}
    ]
    confidence_values = [
        _safe_float(row.get("confidence"))
        for row in rows
        if _safe_float(row.get("confidence")) > 0.0
    ]
    event_types = Counter(str(row.get("event_type") or "unknown") for row in rows)
    return {
        "source": str(summary_path),
        "events_source": str(events_path),
        "total_events": len(rows),
        "command_attempts": len(command_events),
        "executed_events": executed,
        "rejected_events": sum(
            1
            for row in rows
            if str(row.get("event_type") or "")
            in {"gesture_rejected", "command_rejected"}
        ),
        "suppressed_events": event_types.get("gesture_suppressed", 0),
        "cooldown_events": event_types.get("command_cooldown", 0),
        "command_success_rate": (
            executed / len(command_events) if command_events else 0.0
        ),
        "avg_confidence": _avg(confidence_values),
        "label_counts": dict(
            sorted(Counter(str(row.get("label") or "unknown") for row in rows).items())
        ),
        "route_counts": dict(
            sorted(Counter(str(row.get("route") or "unknown") for row in rows).items())
        ),
        "event_type_counts": dict(sorted(event_types.items())),
        "latest_event": rows[-1] if rows else {},
        "latest_runtime": {},
    }


def mlflow_summary(mlflow_db: Path) -> dict[str, Any]:
    if not mlflow_db.exists():
        return {
            "available": False,
            "source": str(mlflow_db),
            "error": "mlflow.db not found",
            "runs": 0,
            "experiments": {},
            "run_kind_counts": {},
            "training_leaderboard": [],
            "live_leaderboard": [],
            "profile_scoreboard": [],
            "safety_scoreboard": [],
            "timeline": [],
        }

    try:
        conn = sqlite3.connect(str(mlflow_db))
        conn.row_factory = sqlite3.Row
        with conn:
            required = ("experiments", "runs", "latest_metrics", "params", "tags")
            missing = [
                table for table in required if not _mlflow_table_exists(conn, table)
            ]
            if missing:
                return {
                    "available": False,
                    "source": str(mlflow_db),
                    "error": "missing MLflow tables: " + ", ".join(missing),
                    "runs": 0,
                    "experiments": {},
                    "run_kind_counts": {},
                    "training_leaderboard": [],
                    "live_leaderboard": [],
                    "profile_scoreboard": [],
                    "safety_scoreboard": [],
                    "timeline": [],
                }
            runs = _mlflow_fetch_runs(conn)
    except sqlite3.Error as exc:
        return {
            "available": False,
            "source": str(mlflow_db),
            "error": str(exc),
            "runs": 0,
            "experiments": {},
            "run_kind_counts": {},
            "training_leaderboard": [],
            "live_leaderboard": [],
            "profile_scoreboard": [],
            "safety_scoreboard": [],
            "timeline": [],
        }
    finally:
        try:
            conn.close()
        except UnboundLocalError:
            pass

    run_kind_counts = Counter(_mlflow_run_kind(run) for run in runs)
    experiments = Counter(str(run.get("experiment_name") or "unknown") for run in runs)
    raw_training_rows = [
        row for row in (_mlflow_training_row(run) for run in runs) if row is not None
    ]
    live_rows = [row for row in (_mlflow_live_row(run) for run in runs) if row is not None]

    raw_training_rows.sort(
        key=lambda row: (
            _safe_float(row.get("rank_score")),
            _safe_float(row.get("sample_count")),
            row.get("started_at") or "",
        ),
        reverse=True,
    )
    live_rows.sort(key=_mlflow_live_sort_key, reverse=True)
    training_rows = _mlflow_training_scoreboard(raw_training_rows)
    live_label_rows = _mlflow_live_label_scoreboard(live_rows)

    return {
        "available": True,
        "source": str(mlflow_db),
        "runs": len(runs),
        "experiments": dict(sorted(experiments.items())),
        "run_kind_counts": dict(sorted(run_kind_counts.items())),
        "training_leaderboard": training_rows[:20],
        "live_leaderboard": live_label_rows[:30],
        "profile_scoreboard": _mlflow_profile_scoreboard(live_rows),
        "safety_scoreboard": _mlflow_safety_scoreboard(live_rows),
        "timeline": _mlflow_timeline(runs),
    }


def _mlflow_table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type='table' and name=? limit 1",
        (name,),
    ).fetchone()
    return row is not None


def _mlflow_fetch_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        select
            r.run_uuid,
            r.name as run_name,
            r.status,
            r.start_time,
            r.end_time,
            r.lifecycle_stage,
            r.artifact_uri,
            r.experiment_id,
            e.name as experiment_name
        from runs r
        left join experiments e on e.experiment_id = r.experiment_id
        where coalesce(r.lifecycle_stage, 'active') = 'active'
        order by r.start_time desc
        """
    ).fetchall()
    runs = [dict(row) | {"metrics": {}, "params": {}, "tags": {}} for row in rows]
    run_ids = [str(run["run_uuid"]) for run in runs]
    if not run_ids:
        return runs

    by_id = {str(run["run_uuid"]): run for run in runs}
    for table, target, cast in (
        ("latest_metrics", "metrics", _safe_float),
        ("params", "params", str),
        ("tags", "tags", str),
    ):
        for row in _mlflow_key_values(conn, table, run_ids):
            run = by_id.get(str(row["run_uuid"]))
            if run is None:
                continue
            run[target][str(row["key"])] = cast(row["value"])
    return runs


def _mlflow_key_values(
    conn: sqlite3.Connection,
    table: str,
    run_ids: list[str],
) -> list[sqlite3.Row]:
    rows: list[sqlite3.Row] = []
    for start in range(0, len(run_ids), 300):
        chunk = run_ids[start : start + 300]
        placeholders = ",".join("?" for _ in chunk)
        rows.extend(
            conn.execute(
                f"select run_uuid, key, value from {table} "
                f"where run_uuid in ({placeholders})",
                chunk,
            ).fetchall()
        )
    return rows


def _mlflow_run_kind(run: dict[str, Any]) -> str:
    tags = run.get("tags") or {}
    metrics = run.get("metrics") or {}
    params = run.get("params") or {}
    tagged = str(tags.get("run_kind") or "").strip()
    if tagged:
        return tagged
    if "live_accuracy" in metrics:
        return "live_evaluation"
    if params.get("model_type") or params.get("feature_mode"):
        return "training"
    run_name = str(run.get("run_name") or "")
    if run_name.startswith("dynamic-prototype-"):
        return "dynamic_prototype_experiment"
    if run_name.startswith("binding-agent-"):
        return "binding_agent_pipeline"
    return "uncategorized"


def _mlflow_training_row(run: dict[str, Any]) -> dict[str, Any] | None:
    metrics = run.get("metrics") or {}
    params = run.get("params") or {}
    kind = _mlflow_run_kind(run)
    is_training = (
        kind == "training"
        or "train_accuracy" in metrics
        or bool(params.get("model_type"))
        or bool(params.get("feature_mode"))
    )
    if not is_training:
        return None

    score_key, score = _first_metric(metrics, TRAINING_SCORE_KEYS)
    if score_key is None:
        score_key, score = "runs", 0.0
    model_type = str(params.get("model_type") or "model")
    feature_mode = str(params.get("feature_mode") or "default_features")
    approach = f"{model_type} + {feature_mode}"
    return {
        "run_name": str(run.get("run_name") or ""),
        "approach": approach,
        "model_type": model_type,
        "feature_mode": feature_mode,
        "score": score,
        "score_metric": score_key,
        "rank_score": _training_rank_score(score_key, score),
        "sample_count": _metric(metrics, "sample_count"),
        "class_count": _metric(metrics, "class_count"),
        "feature_dim": _metric(metrics, "feature_dim"),
        "status": str(run.get("status") or ""),
        "started_at": _mlflow_time(run.get("start_time")),
        "experiment": str(run.get("experiment_name") or ""),
    }


def _mlflow_live_row(run: dict[str, Any]) -> dict[str, Any] | None:
    metrics = run.get("metrics") or {}
    params = run.get("params") or {}
    if _mlflow_run_kind(run) != "live_evaluation" and "live_accuracy" not in metrics:
        return None

    profile = str(
        params.get("dynamic_model_profile")
        or params.get("recognition_model_mode")
        or "auto"
    )
    reject_method = str(params.get("static_rejection_method") or "none")
    mode = str(params.get("recognition_model_mode") or "auto")
    return {
        "run_name": str(run.get("run_name") or ""),
        "expected_label": str(params.get("expected_label") or ""),
        "expected_type": str(params.get("expected_type") or ""),
        "mode": mode,
        "profile": profile,
        "reject_method": reject_method,
        "approach": f"{profile} / {reject_method}",
        "accuracy": _metric(metrics, "live_accuracy"),
        "recall": _metric(metrics, "live_recall", "live_dynamic_recall"),
        "dynamic_recall": _metric(metrics, "live_dynamic_recall"),
        "negative_fp": _metric(metrics, "live_negative_false_positive_rate"),
        "static_fp": _metric(metrics, "live_static_false_positive_rate"),
        "static_hijack": _metric(metrics, "live_static_hijack_rate"),
        "wrong_direction": _metric(metrics, "live_wrong_dynamic_direction_rate"),
        "latency_avg_s": _latency_metric(metrics, "live_latency_avg_s"),
        "latency_p95_s": _latency_metric(metrics, "live_latency_p95_s"),
        "total": _metric(metrics, "live_total", "live_target_attempts"),
        "status": str(run.get("status") or ""),
        "started_at": _mlflow_time(run.get("start_time")),
        "experiment": str(run.get("experiment_name") or ""),
    }


def _mlflow_profile_scoreboard(live_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in live_rows:
        grouped[str(row.get("approach") or "unknown")].append(row)

    scoreboard: list[dict[str, Any]] = []
    for approach, rows in grouped.items():
        avg_accuracy = _avg(row.get("accuracy") for row in rows)
        avg_recall = _avg(row.get("recall") for row in rows)
        avg_negative_fp = _avg(row.get("negative_fp") for row in rows)
        avg_static_hijack = _avg(row.get("static_hijack") for row in rows)
        avg_wrong_direction = _avg(row.get("wrong_direction") for row in rows)
        avg_latency = _avg(row.get("latency_avg_s") for row in rows)
        presentation_score = (
            avg_accuracy
            - 0.35 * avg_negative_fp
            - 0.25 * max(avg_static_hijack, avg_wrong_direction)
        )
        scoreboard.append(
            {
                "approach": approach,
                "runs": len(rows),
                "avg_accuracy": avg_accuracy,
                "avg_recall": avg_recall,
                "avg_negative_fp": avg_negative_fp,
                "avg_static_hijack": avg_static_hijack,
                "avg_wrong_direction": avg_wrong_direction,
                "avg_latency_s": avg_latency,
                "presentation_score": max(0.0, presentation_score),
            }
        )
    scoreboard.sort(
        key=lambda row: (
            _safe_float(row.get("presentation_score")),
            _safe_float(row.get("avg_recall")),
            _safe_float(row.get("runs")),
        ),
        reverse=True,
    )
    return scoreboard[:12]


def _mlflow_safety_scoreboard(live_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in live_rows:
        grouped[str(row.get("reject_method") or "none")].append(row)

    scoreboard: list[dict[str, Any]] = []
    for method, rows in grouped.items():
        avg_negative_fp = _avg(row.get("negative_fp") for row in rows)
        avg_static_fp = _avg(row.get("static_fp") for row in rows)
        avg_static_hijack = _avg(row.get("static_hijack") for row in rows)
        avg_wrong_direction = _avg(row.get("wrong_direction") for row in rows)
        worst_fp = max(avg_negative_fp, avg_static_fp, avg_static_hijack)
        explicit_layer_bonus = 1.0 if method not in {"none", "unknown", ""} else 0.85
        scoreboard.append(
            {
                "reject_method": method,
                "runs": len(rows),
                "avg_negative_fp": avg_negative_fp,
                "avg_static_fp": avg_static_fp,
                "avg_static_hijack": avg_static_hijack,
                "avg_wrong_direction": avg_wrong_direction,
                "safety_score": max(0.0, 1.0 - worst_fp) * explicit_layer_bonus,
            }
        )
    scoreboard.sort(
        key=lambda row: (
            _safe_float(row.get("safety_score")),
            _safe_float(row.get("runs")),
        ),
        reverse=True,
    )
    return scoreboard[:10]


def _mlflow_training_scoreboard(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("approach") or "unknown")].append(row)

    scoreboard: list[dict[str, Any]] = []
    for approach, group in grouped.items():
        best = max(
            group,
            key=lambda row: (
                _safe_float(row.get("rank_score")),
                _safe_float(row.get("sample_count")),
                row.get("started_at") or "",
            ),
        )
        merged = dict(best)
        merged["approach"] = approach
        merged["runs"] = len(group)
        merged["sample_count"] = max(_safe_float(row.get("sample_count")) for row in group)
        merged["class_count"] = max(_safe_float(row.get("class_count")) for row in group)
        merged["feature_dim"] = max(_safe_float(row.get("feature_dim")) for row in group)
        scoreboard.append(merged)
    scoreboard.sort(
        key=lambda row: (
            _safe_float(row.get("rank_score")),
            _safe_float(row.get("sample_count")),
            row.get("started_at") or "",
        ),
        reverse=True,
    )
    return scoreboard


def _mlflow_live_label_scoreboard(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("expected_label") or "unknown"), str(row.get("approach") or "unknown"))
        grouped[key].append(row)

    scoreboard: list[dict[str, Any]] = []
    for (expected_label, approach), group in grouped.items():
        latest = max(group, key=lambda row: row.get("started_at") or "")
        scoreboard.append(
            {
                "expected_label": expected_label,
                "approach": approach,
                "accuracy": _avg(row.get("accuracy") for row in group),
                "recall": _avg(row.get("recall") for row in group),
                "negative_fp": _avg(row.get("negative_fp") for row in group),
                "wrong_direction": _avg(row.get("wrong_direction") for row in group),
                "latency_avg_s": _avg(row.get("latency_avg_s") for row in group),
                "total": sum(_safe_float(row.get("total")) for row in group),
                "runs": len(group),
                "started_at": latest.get("started_at"),
            }
        )
    scoreboard.sort(key=_mlflow_live_sort_key, reverse=True)
    return scoreboard


def _mlflow_timeline(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for run in sorted(
        runs,
        key=lambda item: int(item.get("start_time") or 0),
        reverse=True,
    )[:28]:
        metrics = run.get("metrics") or {}
        score_key, score = _first_metric(metrics, TIMELINE_SCORE_KEYS)
        timeline.append(
            {
                "started_at": _mlflow_time(run.get("start_time")),
                "kind": _mlflow_run_kind(run),
                "run_name": str(run.get("run_name") or ""),
                "score_metric": score_key or "",
                "score": score if score_key else None,
                "status": str(run.get("status") or ""),
            }
        )
    return timeline


def _mlflow_live_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    accuracy = _safe_float(row.get("accuracy"))
    recall = _safe_float(row.get("recall"))
    negative_fp = _safe_float(row.get("negative_fp"))
    static_hijack = _safe_float(row.get("static_hijack"))
    wrong_direction = _safe_float(row.get("wrong_direction"))
    latency = _safe_float(row.get("latency_avg_s"))
    total = _safe_float(row.get("total"))
    return (
        accuracy,
        recall,
        -negative_fp,
        -static_hijack,
        -wrong_direction,
        -latency,
        total,
    )


def _training_rank_score(score_key: str | None, score: float | None) -> float:
    numeric = _safe_float(score)
    if score_key == "train_accuracy":
        return numeric * 0.72
    if score_key in {"accuracy", "best_score"}:
        return numeric * 0.95
    return numeric


def _first_metric(
    metrics: dict[str, Any],
    keys: tuple[str, ...],
) -> tuple[str | None, float | None]:
    for key in keys:
        if key in metrics:
            return key, _safe_float(metrics[key])
    return None, None


def _metric(metrics: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in metrics:
            return _safe_float(metrics[key])
    return 0.0


def _latency_metric(metrics: dict[str, Any], *keys: str) -> float:
    value = _metric(metrics, *keys)
    return value if 0.0 < value <= 30.0 else 0.0


def _avg(values: Any) -> float:
    numeric = [_safe_float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mlflow_time(value: Any) -> str:
    try:
        timestamp = int(value) / 1000.0
    except (TypeError, ValueError):
        return ""
    if timestamp <= 0:
        return ""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def build_dashboard(
    *,
    data_root: Path = ROOT / "data" / "gestures",
    models_dir: Path = ROOT / "models",
    taxonomy_path: Path = ROOT / "configs" / "gesture_taxonomy.json",
    log_dir: Path = DEFAULT_LOG_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    mlflow_db: Path = DEFAULT_MLFLOW_DB,
    html_refresh_seconds: float = 0.0,
) -> dict[str, Any]:
    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dashboard_refresh_seconds": max(0.0, float(html_refresh_seconds or 0.0)),
        "dataset": dataset_summary(data_root, taxonomy_path),
        "models": model_artifacts(models_dir),
        "live_evaluation": live_evaluation_summary(log_dir),
        "live_usage": live_usage_summary(log_dir),
        "runtime": runtime_summary(log_dir),
        "mlflow": mlflow_summary(mlflow_db),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    return report


def render_html(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    live = report["live_evaluation"]
    live_usage = report.get("live_usage") or {}
    runtime = report["runtime"]
    latest_runtime = runtime.get("latest") or {}
    models = report["models"]
    mlflow = report.get("mlflow") or {}
    latest_usage_event = live_usage.get("latest_event") or {}
    refresh_seconds = int(max(0.0, _safe_float(report.get("dashboard_refresh_seconds"))))
    refresh_meta = (
        f'  <meta http-equiv="refresh" content="{refresh_seconds}">\n'
        if refresh_seconds > 0
        else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
{refresh_meta}  <title>GestureBind MLOps Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #10111d;
      --panel: #191b2b;
      --panel-2: #202336;
      --line: #30344d;
      --text: #f5f7fb;
      --muted: #aab4c3;
      --accent: #14c6d5;
      --accent-2: #f2b84b;
      --good: #4fd267;
      --bad: #ff5148;
      --warn: #ffd166;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 4px; font-size: 30px; }}
    h2 {{ margin: 0 0 8px; font-size: 18px; }}
    h3 {{ margin: 0 0 10px; font-size: 15px; }}
    p {{ margin: 8px 0 0; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-top: 16px;
    }}
    .hero {{
      background: #151827;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      margin-top: 18px;
    }}
    .stat-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 16px; }}
    .stat {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }}
    .metric {{ font-size: 28px; font-weight: 800; margin-top: 8px; }}
    .big-number {{ font-size: 30px; font-weight: 850; line-height: 1.1; margin: 8px 0 2px; }}
    .kicker {{ color: var(--accent-2); font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      color: var(--muted);
      font-size: 12px;
      margin: 2px 4px 2px 0;
      white-space: nowrap;
    }}
    .chart {{ display: grid; gap: 10px; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(160px, 260px) 1fr minmax(90px, auto); gap: 10px; align-items: center; }}
    .bar-label {{ overflow-wrap: anywhere; font-weight: 700; }}
    .bar-track {{ height: 12px; background: #0e101a; border: 1px solid var(--line); border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: var(--accent); border-radius: 999px; }}
    .bar-fill.warn {{ background: var(--accent-2); }}
    .bar-value {{ color: var(--muted); text-align: right; white-space: nowrap; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; }}
    td {{ vertical-align: top; overflow-wrap: anywhere; }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .warn {{ color: var(--warn); }}
    .mono {{ font-family: Menlo, Consolas, monospace; font-size: 12px; }}
    @media (max-width: 900px) {{
      .grid, .stat-row, .two-col {{ grid-template-columns: repeat(2, 1fr); }}
      .bar-row {{ grid-template-columns: 1fr; }}
      .bar-value {{ text-align: left; }}
    }}
    @media (max-width: 560px) {{
      main {{ padding: 16px; }}
      .grid, .stat-row, .two-col {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>GestureBind MLOps Dashboard</h1>
  <div class="muted">Generated {esc(report["generated_at"])}</div>

{mlflow_showcase(mlflow)}

  <section class="grid">
    {metric_card("Live accuracy", pct(live["accuracy"]), "correct / all attempts", live["accuracy"] >= 0.8)}
    {metric_card("Live attempts", str(live["attempts"]), "from live_evaluation.jsonl")}
    {metric_card("Negative rejects", str(live["negative_rejections"]), "blocked by rejection layer")}
    {metric_card("Dataset samples", str(dataset["total_samples"]), str(dataset["total_classes"]) + " classes")}
  </section>

  <section class="grid">
    {metric_card("Usage events", fmt_int(live_usage.get("total_events")), "rolling app telemetry")}
    {metric_card("Command success", pct(live_usage.get("command_success_rate")), fmt_int(live_usage.get("command_attempts")) + " command attempts", live_usage.get("command_success_rate", 0) >= 0.8 if live_usage.get("command_attempts") else None)}
    {metric_card("Avg confidence", pct(live_usage.get("avg_confidence")), "confirmed/rejected gestures")}
    {metric_card("Latest gesture", latest_usage_event.get("label") or "none", latest_usage_event.get("event_type") or "no events")}
  </section>

{mlflow_charts(mlflow)}

  <section class="panel">
    <h2>Runtime</h2>
    {runtime_table(latest_runtime)}
  </section>

  <section class="panel">
    <h2>Live Usage</h2>
    <p class="muted">
      Source: <span class="mono">{esc(live_usage.get("events_source") or live_usage.get("source") or "")}</span>
    </p>
    <div class="two-col">
      <div>
        <h3>Event Types</h3>
        {counter_table("Event", live_usage.get("event_type_counts") or {})}
      </div>
      <div>
        <h3>Recognized Labels</h3>
        {counter_table("Label", live_usage.get("label_counts") or {})}
      </div>
    </div>
    <div class="two-col">
      <div>
        <h3>Routes</h3>
        {counter_table("Route", live_usage.get("route_counts") or {})}
      </div>
      <div>
        <h3>Latest Event</h3>
        {event_table(latest_usage_event)}
      </div>
    </div>
  </section>

  <section class="panel">
    <h2>Live Evaluation By Label</h2>
    {label_table(live["labels"])}
  </section>

  <section class="panel">
    <h2>Routes And Decisions</h2>
    {counter_table("Route", live["route_counts"])}
    {counter_table("Dynamic decision source", live["dynamic_decision_sources"])}
    {counter_table("Static rejection method", live["static_rejection_methods"])}
    {counter_table("Static decision source", live["static_decision_sources"])}
    {counter_table("Static rejection reason", live["static_rejection_reasons"])}
  </section>

  <section class="panel">
    <h2>Dataset</h2>
    {counter_table("Gesture type", dataset["type_counts"])}
    {dataset_table(dataset["classes"])}
  </section>

  <section class="panel">
    <h2>Model Artifacts</h2>
    {model_table(models)}
  </section>
</main>
</body>
</html>
"""


def mlflow_showcase(mlflow: dict[str, Any]) -> str:
    if not mlflow.get("available"):
        return (
            '<section class="hero">'
            '<div class="kicker">MLflow Showcase</div>'
            "<h2>MLflow history is not available</h2>"
            f"<p class=\"muted\">{esc(mlflow.get('error') or 'No MLflow database found.')}</p>"
            "</section>"
        )

    best_profile = _first_item(mlflow.get("profile_scoreboard"))
    best_train = _first_item(mlflow.get("training_leaderboard"))
    best_safety = _first_item(mlflow.get("safety_scoreboard"))
    best_live = _first_item(mlflow.get("live_leaderboard"))
    run_kinds = mlflow.get("run_kind_counts") or {}
    kind_pills = "".join(
        f"<span class=\"pill\">{esc(key)}: {esc(value)}</span>"
        for key, value in sorted(run_kinds.items())
    )

    return f"""  <section class="hero">
    <div class="kicker">MLflow Showcase</div>
    <h2>Top approaches for presentation</h2>
    <p class="muted">
      Local MLflow history is summarized into leaderboard charts: training
      candidates, live A/B quality, rejection safety and recent experiment
      timeline.
    </p>
    <div class="stat-row">
      {stat_card("MLflow runs", fmt_int(mlflow.get("runs")), "tracked experiments")}
      {stat_card("Best live approach", pct(best_profile.get("presentation_score")), best_profile.get("approach") or "not enough live runs", best_profile.get("presentation_score", 0) >= 0.8)}
      {stat_card("Best training score", pct(best_train.get("score")), best_train.get("approach") or "no training runs", best_train.get("score", 0) >= 0.8)}
      {stat_card("Best safety layer", pct(best_safety.get("safety_score")), best_safety.get("reject_method") or "no rejection runs", best_safety.get("safety_score", 0) >= 0.9)}
    </div>
    <p class="muted">
      Best latest live run: {esc(best_live.get("expected_label") or "n/a")}
      via {esc(best_live.get("approach") or "n/a")};
      source: <span class="mono">{esc(mlflow.get("source"))}</span>
    </p>
    <div>{kind_pills}</div>
  </section>
"""


def mlflow_charts(mlflow: dict[str, Any]) -> str:
    if not mlflow.get("available"):
        return ""
    profile_rows = mlflow.get("profile_scoreboard") or []
    training_rows = mlflow.get("training_leaderboard") or []
    safety_rows = mlflow.get("safety_scoreboard") or []
    live_rows = mlflow.get("live_leaderboard") or []
    timeline = mlflow.get("timeline") or []
    return f"""  <section class="panel">
    <h2>MLflow Top Approaches</h2>
    <p class="muted">Presentation score = average live accuracy minus penalties for negative false positives, static hijacks and wrong dynamic direction.</p>
    <div class="two-col">
      {bar_chart("Live profiles", profile_rows, "approach", "presentation_score", max_items=8)}
      {bar_chart("Training candidates", training_rows, "approach", "rank_score", value_key_label="score_metric", max_items=8, fill="warn")}
    </div>
  </section>

  <section class="panel">
    <h2>Live A/B Scoreboard</h2>
    {mlflow_profile_table(profile_rows)}
    <h3>Best individual live runs</h3>
    {mlflow_live_table(live_rows)}
  </section>

  <section class="panel">
    <h2>Safety And Rejection</h2>
    <div class="two-col">
      {bar_chart("Safety by rejection method", safety_rows, "reject_method", "safety_score", max_items=8)}
      {bar_chart("Live run accuracy", live_rows, "expected_label", "accuracy", value_key_label="approach", max_items=8, fill="warn")}
    </div>
    {mlflow_safety_table(safety_rows)}
  </section>

  <section class="panel">
    <h2>Training Model Leaderboard</h2>
    <p class="muted">Rank score favors CV/Optuna/test metrics; train-only accuracy is discounted so overfit runs do not dominate the presentation.</p>
    {mlflow_training_table(training_rows)}
  </section>

  <section class="panel">
    <h2>Recent MLflow Timeline</h2>
    {mlflow_timeline_table(timeline)}
  </section>
"""


def stat_card(title: str, value: str, caption: str, ok: bool | None = None) -> str:
    cls = ""
    if ok is True:
        cls = " good"
    elif ok is False:
        cls = " bad"
    return (
        '<div class="stat">'
        f"<div class=\"muted\">{esc(title)}</div>"
        f"<div class=\"big-number{cls}\">{esc(value)}</div>"
        f"<div class=\"muted\">{esc(caption)}</div>"
        "</div>"
    )


def bar_chart(
    title: str,
    rows: list[dict[str, Any]],
    label_key: str,
    value_key: str,
    *,
    value_key_label: str | None = None,
    max_items: int = 8,
    fill: str = "",
) -> str:
    if not rows:
        return (
            '<div class="panel">'
            f"<h3>{esc(title)}</h3>"
            '<div class="muted">No MLflow data yet.</div>'
            "</div>"
        )
    visible = rows[:max_items]
    max_value = max(_safe_float(row.get(value_key)) for row in visible) or 1.0
    fill_cls = " warn" if fill == "warn" else ""
    bars = []
    for row in visible:
        value = _safe_float(row.get(value_key))
        width = max(2.0, min(100.0, value / max_value * 100.0))
        label = row.get(label_key) or row.get("run_name") or "unknown"
        caption = pct(value)
        if value_key_label:
            extra = row.get(value_key_label)
            if extra:
                caption = f"{caption} · {extra}"
        elif row.get("runs"):
            caption = f"{caption} · {fmt_int(row.get('runs'))} runs"
        bars.append(
            '<div class="bar-row">'
            f"<div class=\"bar-label\">{esc(label)}</div>"
            '<div class="bar-track">'
            f"<div class=\"bar-fill{fill_cls}\" style=\"width:{width:.1f}%\"></div>"
            "</div>"
            f"<div class=\"bar-value\">{esc(caption)}</div>"
            "</div>"
        )
    return (
        "<div>"
        f"<h3>{esc(title)}</h3>"
        f"<div class=\"chart\">{''.join(bars)}</div>"
        "</div>"
    )


def mlflow_profile_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="muted">No live MLflow runs yet.</div>'
    table_rows = [
        (
            row.get("approach"),
            fmt_int(row.get("runs")),
            pct(row.get("presentation_score")),
            pct(row.get("avg_accuracy")),
            pct(row.get("avg_recall")),
            pct(row.get("avg_negative_fp")),
            pct(row.get("avg_static_hijack")),
            seconds(row.get("avg_latency_s")),
        )
        for row in rows
    ]
    return simple_table(
        [
            "Approach",
            "Runs",
            "Score",
            "Accuracy",
            "Recall",
            "Neg FP",
            "Static hijack",
            "Latency",
        ],
        table_rows,
    )


def mlflow_training_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="muted">No training MLflow runs yet.</div>'
    table_rows = [
        (
            row.get("approach"),
            fmt_int(row.get("runs")),
            pct(row.get("score")),
            pct(row.get("rank_score")),
            row.get("score_metric"),
            fmt_int(row.get("sample_count")),
            fmt_int(row.get("class_count")),
            fmt_int(row.get("feature_dim")),
            row.get("started_at"),
        )
        for row in rows[:16]
    ]
    return simple_table(
        ["Approach", "Runs", "Score", "Rank", "Metric", "Samples", "Classes", "Dim", "Started"],
        table_rows,
    )


def mlflow_live_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="muted">No live MLflow runs yet.</div>'
    table_rows = [
        (
            row.get("expected_label"),
            row.get("approach"),
            fmt_int(row.get("runs")),
            pct(row.get("accuracy")),
            pct(row.get("recall")),
            pct(row.get("negative_fp")),
            pct(row.get("wrong_direction")),
            seconds(row.get("latency_avg_s")),
            fmt_int(row.get("total")),
            row.get("started_at"),
        )
        for row in rows[:16]
    ]
    return simple_table(
        [
            "Expected",
            "Approach",
            "Runs",
            "Accuracy",
            "Recall",
            "Neg FP",
            "Wrong dir",
            "Latency",
            "Total",
            "Started",
        ],
        table_rows,
    )


def mlflow_safety_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="muted">No rejection MLflow runs yet.</div>'
    table_rows = [
        (
            row.get("reject_method"),
            fmt_int(row.get("runs")),
            pct(row.get("safety_score")),
            pct(row.get("avg_negative_fp")),
            pct(row.get("avg_static_fp")),
            pct(row.get("avg_static_hijack")),
            pct(row.get("avg_wrong_direction")),
        )
        for row in rows
    ]
    return simple_table(
        [
            "Reject method",
            "Runs",
            "Safety",
            "Negative FP",
            "Static FP",
            "Static hijack",
            "Wrong dir",
        ],
        table_rows,
    )


def mlflow_timeline_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="muted">No MLflow runs yet.</div>'
    table_rows = [
        (
            row.get("started_at"),
            row.get("kind"),
            row.get("run_name"),
            row.get("status"),
            row.get("score_metric"),
            "" if row.get("score") is None else pct(row.get("score")),
        )
        for row in rows
    ]
    return simple_table(["Started", "Kind", "Run", "Status", "Metric", "Value"], table_rows)


def metric_card(title: str, value: str, caption: str, ok: bool | None = None) -> str:
    cls = ""
    if ok is True:
        cls = " good"
    elif ok is False:
        cls = " bad"
    return (
        '<div class="panel">'
        f"<div class=\"muted\">{esc(title)}</div>"
        f"<div class=\"metric{cls}\">{esc(value)}</div>"
        f"<div class=\"muted\">{esc(caption)}</div>"
        "</div>"
    )


def runtime_table(latest: dict[str, Any]) -> str:
    if not latest:
        return '<div class="muted">No runtime_performance.jsonl rows yet.</div>'
    rows = [
        ("mode", latest.get("recognition_model_mode", "")),
        ("dynamic profile", latest.get("dynamic_model_profile", "")),
        ("target FPS", latest.get("target_fps", "")),
        ("inference avg ms", latest.get("inference_ms_avg", "")),
        ("inference p95 ms", latest.get("inference_ms_p95", "")),
        ("FPS capacity", latest.get("inference_fps_capacity", "")),
        ("shared detection rate", latest.get("shared_detection_rate", "")),
    ]
    return simple_table(["Metric", "Value"], rows)


def event_table(event: dict[str, Any]) -> str:
    if not event:
        return '<div class="muted">No live usage events yet.</div>'
    rows = [
        ("event", event.get("event_type", "")),
        ("label", event.get("label", "")),
        ("confidence", pct(event.get("confidence"))),
        ("route", event.get("route", "")),
        ("executed", "yes" if event.get("executed") else "no"),
        ("reason", event.get("reason", "")),
        ("command", event.get("command_info", "")),
    ]
    return simple_table(["Field", "Value"], rows)


def label_table(labels: list[dict[str, Any]]) -> str:
    if not labels:
        return '<div class="muted">No live attempts yet.</div>'
    rows = [
        (
            item["expected"],
            item["attempts"],
            item["correct"],
            item["wrong"],
            item["missed"],
            pct(item["accuracy"]),
        )
        for item in labels
    ]
    return simple_table(["Expected", "Attempts", "Correct", "Wrong", "Missed", "Accuracy"], rows)


def counter_table(label: str, values: dict[str, Any]) -> str:
    if not values:
        return f'<div class="muted">No {esc(label.lower())} data.</div>'
    return simple_table([label, "Count"], sorted(values.items()))


def dataset_table(classes: list[dict[str, Any]]) -> str:
    if not classes:
        return '<div class="muted">No samples found.</div>'
    rows = [(item["label"], item["type"], item["samples"]) for item in classes]
    return simple_table(["Label", "Type", "Samples"], rows)


def model_table(models: list[dict[str, Any]]) -> str:
    if not models:
        return '<div class="muted">No model artifacts found.</div>'
    rows = [
        (
            item["name"],
            item["size_bytes"],
            item["sha256"],
            datetime.fromtimestamp(item["modified_at"]).strftime("%Y-%m-%d %H:%M:%S"),
        )
        for item in models
    ]
    return simple_table(["Name", "Bytes", "SHA256", "Modified"], rows)


def simple_table(headers: list[str], rows: list[tuple[Any, ...]]) -> str:
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def fmt_int(value: Any) -> str:
    try:
        return f"{int(float(value)):,}".replace(",", " ")
    except (TypeError, ValueError):
        return "0"


def seconds(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 0:
        return "0 ms"
    if numeric < 1:
        return f"{numeric * 1000:.0f} ms"
    return f"{numeric:.2f} s"


def _first_item(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return first
    return {}


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "gestures")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "configs" / "gesture_taxonomy.json",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mlflow-db", type=Path, default=DEFAULT_MLFLOW_DB)
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0.0,
        help="Regenerate the dashboard every N seconds; useful for live demos.",
    )
    return parser.parse_args()


def _build_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return build_dashboard(
        data_root=args.data_root,
        models_dir=args.models_dir,
        taxonomy_path=args.taxonomy,
        log_dir=args.log_dir,
        out_dir=args.out_dir,
        mlflow_db=args.mlflow_db,
        html_refresh_seconds=args.watch_seconds if args.watch_seconds > 0 else 0.0,
    )


def main() -> None:
    args = parse_args()
    interval = max(0.0, float(args.watch_seconds or 0.0))
    if interval > 0.0:
        interval = max(1.0, interval)
        while True:
            report = _build_from_args(args)
            print(
                "Dashboard refreshed at "
                f"{report['generated_at']} -> {args.out_dir / 'index.html'} "
                f"({report['live_evaluation']['attempts']} live attempts, "
                f"{report['live_usage']['total_events']} usage events)",
                flush=True,
            )
            time.sleep(interval)
    report = _build_from_args(args)
    print(
        "Dashboard written to "
        f"{args.out_dir / 'index.html'} "
        f"({report['live_evaluation']['attempts']} live attempts, "
        f"{report['live_usage']['total_events']} usage events)"
    )


if __name__ == "__main__":
    main()
