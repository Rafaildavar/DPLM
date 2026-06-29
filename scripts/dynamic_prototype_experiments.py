"""Train and compare prototype-based dynamic gesture verifiers."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from app.services.gesture_taxonomy import (
    GESTURE_TYPE_DYNAMIC,
    GESTURE_TYPE_NEGATIVE,
    load_gesture_taxonomy,
)
from cv.dynamic_prototype import (
    METHOD_PROTOTYPE_DISTANCE,
    METHOD_PROTOTYPE_DTW,
    SUPPORTED_DYNAMIC_PROTOTYPE_METHODS,
    DynamicSequenceRecord,
    fit_dynamic_prototype_model,
    is_negative_label,
    load_dynamic_sequence_file,
    predict_dynamic_prototype,
    save_dynamic_prototype_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "gestures"
DEFAULT_EXTERNAL_NEGATIVE_ROOT = (
    PROJECT_ROOT / "data" / "external" / "ipn_hand"
)
DEFAULT_REPORT_JSON = PROJECT_ROOT / "docs" / "experiments" / "dynamic_prototype_comparison.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "docs" / "experiments" / "dynamic_prototype_comparison.md"
DEFAULT_VARIANT_ROOT = PROJECT_ROOT / "models" / "experiments" / "dynamic_prototype"
DEFAULT_PRODUCTION_OUT = PROJECT_ROOT / "models" / "dynamic_prototypes.json"
DEFAULT_BASE_MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class EvaluationRow:
    expected: str
    predicted: str
    result: str
    reason: str
    distance: float
    threshold: float
    path: str


@dataclass(frozen=True)
class NegativeConflictRow:
    path: str
    negative_label: str
    nearest_positive_label: str
    distance: float
    threshold: float
    distance_ratio: float
    reason: str


def discover_records(
    *,
    data_root: Path,
    external_negative_root: Path | None,
    target_dim: int,
    target_frames: int,
) -> list[DynamicSequenceRecord]:
    taxonomy = load_gesture_taxonomy()
    records: list[DynamicSequenceRecord] = []
    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        label = label_dir.name
        gesture_type = taxonomy.gesture_type_for_label(label)
        if gesture_type not in {GESTURE_TYPE_DYNAMIC, GESTURE_TYPE_NEGATIVE}:
            continue
        negative = gesture_type == GESTURE_TYPE_NEGATIVE or is_negative_label(label)
        for sample_path in _sample_paths(label_dir):
            record = _record_from_path(
                sample_path,
                label=label,
                negative=negative,
                target_dim=target_dim,
                target_frames=target_frames,
            )
            if record is not None:
                records.append(record)

    if external_negative_root and external_negative_root.exists():
        for sample_path in sorted(external_negative_root.rglob("*.npy")):
            label = sample_path.parent.name
            record = _record_from_path(
                sample_path,
                label=label,
                negative=True,
                target_dim=target_dim,
                target_frames=target_frames,
            )
            if record is not None:
                records.append(record)
    return records


def filter_conflicting_external_negatives(
    records: Iterable[DynamicSequenceRecord],
    *,
    external_negative_root: Path | None,
    method: str = METHOD_PROTOTYPE_DISTANCE,
    target_dim: int = 44,
    target_frames: int = 36,
    max_prototypes_per_label: int = 8,
    threshold_multiplier: float = 1.25,
    threshold_floor: float = 0.015,
    conflict_margin: float = 1.20,
    max_report_rows: int = 100,
) -> tuple[list[DynamicSequenceRecord], dict[str, Any]]:
    items = list(records)
    root = _safe_resolve(external_negative_root) if external_negative_root else None
    positives = [record for record in items if not record.is_negative]
    external_negatives = [
        record
        for record in items
        if record.is_negative and root is not None and _is_under(record.path, root)
    ]
    base_report: dict[str, Any] = {
        "enabled": True,
        "method": method,
        "conflict_margin": float(conflict_margin),
        "external_negative_root": str(root) if root else "",
        "positive_count": len(positives),
        "external_negative_total": len(external_negatives),
        "safe_external_negative_count": len(external_negatives),
        "conflict_count": 0,
        "conflict_rate": 0.0,
        "nearest_positive_labels": {},
        "conflicting_negative_labels": {},
        "conflicts": [],
        "status": "ok",
    }
    if not root:
        return items, {**base_report, "status": "missing_external_root"}
    if not external_negatives:
        return items, {**base_report, "status": "no_external_negatives"}
    if not positives:
        return items, {**base_report, "status": "no_positive_gestures"}

    positive_model = fit_dynamic_prototype_model(
        positives,
        method=method,
        target_dim=target_dim,
        target_frames=target_frames,
        max_prototypes_per_label=max_prototypes_per_label,
        threshold_multiplier=threshold_multiplier,
        threshold_floor=threshold_floor,
    )

    conflict_paths: set[str] = set()
    conflicts: list[NegativeConflictRow] = []
    nearest_positive_labels: Counter[str] = Counter()
    conflicting_negative_labels: Counter[str] = Counter()
    for record in external_negatives:
        decision = predict_dynamic_prototype(positive_model, record.sequence)
        nearest_label = str(decision.get("nearest_label") or "")
        nearest_type = str(decision.get("nearest_type") or "")
        distance = _safe_float(decision.get("distance"), default=float("inf"))
        threshold = _safe_float(decision.get("threshold"), default=0.0)
        ratio = distance / threshold if threshold > 0 else float("inf")
        if nearest_label:
            nearest_positive_labels[nearest_label] += 1
        is_conflict = (
            nearest_type == "positive"
            and threshold > 0.0
            and distance <= threshold * float(conflict_margin)
        )
        if not is_conflict:
            continue
        conflict_paths.add(record.path)
        conflicting_negative_labels[record.label] += 1
        conflicts.append(
            NegativeConflictRow(
                path=record.path,
                negative_label=record.label,
                nearest_positive_label=nearest_label,
                distance=float(distance),
                threshold=float(threshold),
                distance_ratio=float(ratio),
                reason=str(decision.get("reason") or "near_positive"),
            )
        )

    filtered = [
        record
        for record in items
        if record.path not in conflict_paths
    ]
    conflict_count = len(conflict_paths)
    safe_count = max(0, len(external_negatives) - conflict_count)
    report = {
        **base_report,
        "safe_external_negative_count": safe_count,
        "conflict_count": conflict_count,
        "conflict_rate": conflict_count / len(external_negatives)
        if external_negatives
        else 0.0,
        "nearest_positive_labels": dict(sorted(nearest_positive_labels.items())),
        "conflicting_negative_labels": dict(sorted(conflicting_negative_labels.items())),
        "conflicts": [
            row.__dict__
            for row in sorted(
                conflicts,
                key=lambda item: (item.distance_ratio, item.negative_label, item.path),
            )[: max(0, int(max_report_rows))]
        ],
        "status": "ok",
    }
    return filtered, report


def split_records(
    records: Iterable[DynamicSequenceRecord],
    *,
    seed: int,
    test_fraction: float,
) -> tuple[list[DynamicSequenceRecord], list[DynamicSequenceRecord]]:
    grouped: dict[str, list[DynamicSequenceRecord]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)

    rng = random.Random(int(seed))
    train: list[DynamicSequenceRecord] = []
    test: list[DynamicSequenceRecord] = []
    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        if len(items) >= 4:
            test_count = max(1, int(round(len(items) * float(test_fraction))))
            test.extend(items[:test_count])
            train.extend(items[test_count:])
        else:
            train.extend(items)
            test.extend(items)
    return train, test


def evaluate_model(
    payload: dict[str, Any],
    rows: Iterable[DynamicSequenceRecord],
) -> tuple[dict[str, Any], list[EvaluationRow]]:
    attempts: list[EvaluationRow] = []
    positive_total = 0
    positive_correct = 0
    positive_rejected = 0
    positive_wrong = 0
    negative_total = 0
    negative_rejected = 0
    negative_false_positive = 0
    expected_sequence: list[str] = []
    predicted_sequence: list[str] = []
    per_label: dict[str, dict[str, int]] = {}

    for record in rows:
        decision = predict_dynamic_prototype(payload, record.sequence)
        predicted = str(decision.get("label") or "")
        reason = str(decision.get("reason") or "")
        distance = float(decision.get("distance") or 0.0)
        threshold = float(decision.get("threshold") or 0.0)
        if record.is_negative:
            negative_total += 1
            if predicted:
                negative_false_positive += 1
                result = "false_positive"
            else:
                negative_rejected += 1
                result = "rejected"
        else:
            positive_total += 1
            expected_sequence.append(record.label)
            if predicted:
                predicted_sequence.append(predicted)
            label_stats = per_label.setdefault(
                record.label,
                {"total": 0, "correct": 0, "wrong": 0, "rejected": 0},
            )
            label_stats["total"] += 1
            if predicted == record.label:
                positive_correct += 1
                label_stats["correct"] += 1
                result = "correct"
            elif not predicted:
                positive_rejected += 1
                label_stats["rejected"] += 1
                result = "rejected"
            else:
                positive_wrong += 1
                label_stats["wrong"] += 1
                result = "wrong"
        attempts.append(
            EvaluationRow(
                expected=record.label,
                predicted=predicted,
                result=result,
                reason=reason,
                distance=distance,
                threshold=threshold,
                path=record.path,
            )
        )

    sequence_distance = _levenshtein(expected_sequence, predicted_sequence)
    sequence_accuracy = (
        1.0 - sequence_distance / max(1, len(expected_sequence))
        if expected_sequence
        else 0.0
    )
    total = positive_total + negative_total
    success = positive_correct + negative_rejected
    metrics = {
        "total": total,
        "positive_total": positive_total,
        "negative_total": negative_total,
        "positive_recall": positive_correct / positive_total if positive_total else 0.0,
        "positive_reject_rate": positive_rejected / positive_total
        if positive_total
        else 0.0,
        "positive_wrong_rate": positive_wrong / positive_total
        if positive_total
        else 0.0,
        "negative_reject_rate": negative_rejected / negative_total
        if negative_total
        else 0.0,
        "negative_false_positive_rate": negative_false_positive / negative_total
        if negative_total
        else 0.0,
        "overall_success": success / total if total else 0.0,
        "sequence_edit_distance": sequence_distance,
        "sequence_accuracy": sequence_accuracy,
        "per_label": per_label,
    }
    return metrics, attempts


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    records = discover_records(
        data_root=args.data_root,
        external_negative_root=args.external_negative_root
        if args.include_external_negatives
        else None,
        target_dim=args.target_dim,
        target_frames=args.target_frames,
    )
    conflict_report = {
        "enabled": False,
        "status": "disabled",
        "conflict_count": 0,
        "conflict_rate": 0.0,
        "external_negative_total": 0,
        "safe_external_negative_count": 0,
    }
    if args.include_external_negatives and not args.disable_negative_conflict_filter:
        records, conflict_report = filter_conflicting_external_negatives(
            records,
            external_negative_root=args.external_negative_root,
            method=args.negative_conflict_method,
            target_dim=args.target_dim,
            target_frames=args.target_frames,
            max_prototypes_per_label=args.max_prototypes_per_label,
            threshold_multiplier=args.threshold_multiplier,
            threshold_floor=args.threshold_floor,
            conflict_margin=args.negative_conflict_margin,
            max_report_rows=args.negative_conflict_max_report_rows,
        )
    train, test = split_records(
        records,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )
    reports: list[dict[str, Any]] = []
    for method in args.methods:
        payload = fit_dynamic_prototype_model(
            train,
            method=method,
            target_dim=args.target_dim,
            target_frames=args.target_frames,
            max_prototypes_per_label=args.max_prototypes_per_label,
            threshold_multiplier=args.threshold_multiplier,
            threshold_floor=args.threshold_floor,
        )
        metrics, attempts = evaluate_model(payload, test)
        variant_dir = args.variant_root / method
        save_dynamic_prototype_model(payload, variant_dir / "dynamic_prototypes.json")
        _copy_base_dynamic_artifacts(args.base_models_dir, variant_dir)
        report = {
            "method": method,
            "metrics": metrics,
            "model_path": str(variant_dir / "dynamic_prototypes.json"),
            "attempts": [row.__dict__ for row in attempts],
            "thresholds": payload.get("thresholds", {}),
            "negative_conflict_filter": conflict_report,
        }
        reports.append(report)
        _log_mlflow(report, payload, args)

    best = sorted(
        reports,
        key=lambda item: (
            item["metrics"].get("overall_success", 0.0),
            item["metrics"].get("positive_recall", 0.0),
            item["metrics"].get("negative_reject_rate", 0.0),
            item["metrics"].get("sequence_accuracy", 0.0),
        ),
        reverse=True,
    )[0]
    if args.write_production:
        best_payload = json.loads(Path(best["model_path"]).read_text(encoding="utf-8"))
        save_dynamic_prototype_model(best_payload, args.production_out)

    summary = {
        "generated_at": time.time(),
        "data_root": str(args.data_root),
        "external_negative_root": str(args.external_negative_root),
        "include_external_negatives": bool(args.include_external_negatives),
        "train_count": len(train),
        "test_count": len(test),
        "records_by_label": _counts(records),
        "train_by_label": _counts(train),
        "test_by_label": _counts(test),
        "negative_conflict_filter": conflict_report,
        "best_method": best["method"],
        "reports": reports,
        "production_out": str(args.production_out) if args.write_production else "",
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def _sample_paths(label_dir: Path) -> list[Path]:
    paths = {
        *label_dir.glob("sample_*.npy"),
        *label_dir.glob("aug_sample_*.npy"),
    }
    return sorted(paths)


def _record_from_path(
    path: Path,
    *,
    label: str,
    negative: bool,
    target_dim: int,
    target_frames: int,
) -> DynamicSequenceRecord | None:
    try:
        sequence = load_dynamic_sequence_file(
            path,
            target_dim=target_dim,
            target_frames=target_frames,
        )
    except Exception:
        return None
    return DynamicSequenceRecord(
        label=str(label),
        sequence=sequence,
        path=str(path),
        is_negative=bool(negative),
    )


def _copy_base_dynamic_artifacts(base_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "dynamic_knn.pkl",
        "dynamic_svm.pkl",
        "dynamic_extra_trees.pkl",
        "dynamic_classes.json",
        "dynamic_feature_dim.txt",
        "dynamic_feature_mode.txt",
    ):
        source = base_dir / name
        if source.exists():
            shutil.copy2(source, target_dir / name)


def _counts(records: Iterable[DynamicSequenceRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1
    return dict(sorted(counts.items()))


def _levenshtein(left: list[str], right: list[str]) -> int:
    rows, cols = len(left), len(right)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        dp[i][0] = i
    for j in range(cols + 1):
        dp[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[rows][cols]


def _safe_resolve(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    try:
        return Path(path).resolve()
    except Exception:
        return Path(path)


def _is_under(path: str, root: Path) -> bool:
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(root)
        return True
    except Exception:
        return False


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Dynamic Prototype Comparison",
        "",
        f"- Best method: `{summary['best_method']}`",
        f"- Train samples: `{summary['train_count']}`",
        f"- Test samples: `{summary['test_count']}`",
        f"- External negatives: `{summary['include_external_negatives']}`",
        "",
        "## Negative Conflict Filter",
        "",
    ]
    conflict = summary.get("negative_conflict_filter") or {}
    if conflict.get("enabled"):
        lines.extend(
            [
                f"- Status: `{conflict.get('status', '')}`",
                f"- Method: `{conflict.get('method', '')}`",
                f"- Conflict margin: `{float(conflict.get('conflict_margin') or 0.0):.2f}`",
                f"- External negatives: `{int(conflict.get('external_negative_total') or 0)}`",
                f"- Safe external negatives: `{int(conflict.get('safe_external_negative_count') or 0)}`",
                f"- Conflicts removed: `{int(conflict.get('conflict_count') or 0)}`",
                f"- Conflict rate: `{float(conflict.get('conflict_rate') or 0.0):.4f}`",
                "",
            ]
        )
        if conflict.get("conflicting_negative_labels"):
            lines.extend(
                [
                    "| Negative label | Conflicts |",
                    "|---|---:|",
                    *[
                        f"| `{label}` | {count} |"
                        for label, count in sorted(
                            conflict["conflicting_negative_labels"].items()
                        )
                    ],
                    "",
                ]
            )
    else:
        lines.extend(["- Status: `disabled`", ""])
    lines.extend(
        [
        "## Method Metrics",
        "",
        "| Method | Overall | Positive recall | Negative reject | Negative FP | Sequence accuracy | Edit distance |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for report in summary["reports"]:
        metrics = report["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{report['method']}`",
                    f"{metrics['overall_success']:.4f}",
                    f"{metrics['positive_recall']:.4f}",
                    f"{metrics['negative_reject_rate']:.4f}",
                    f"{metrics['negative_false_positive_rate']:.4f}",
                    f"{metrics['sequence_accuracy']:.4f}",
                    str(metrics["sequence_edit_distance"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Per-Class Recall", ""])
    for report in summary["reports"]:
        lines.extend([f"### `{report['method']}`", ""])
        lines.append("| Label | Total | Correct | Wrong | Rejected |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, stats in sorted(report["metrics"].get("per_label", {}).items()):
            lines.append(
                f"| `{label}` | {stats['total']} | {stats['correct']} | "
                f"{stats['wrong']} | {stats['rejected']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- `prototype_distance` is a fast flattened-sequence verifier.",
            "- `prototype_dtw` is slower but more tolerant to timing variation.",
            "- A method is useful for live only if it keeps positive recall high "
            "while lowering negative false positives.",
        ]
    )
    return "\n".join(lines) + "\n"


def _log_mlflow(
    report: dict[str, Any],
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    try:
        import mlflow
    except Exception:
        return
    try:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment("GestureFlow")
        with mlflow.start_run(run_name=f"dynamic-prototype-{report['method']}"):
            mlflow.set_tags(
                {
                    "run_kind": "dynamic_prototype_experiment",
                    "method": report["method"],
                }
            )
            mlflow.log_params(
                {
                    "target_dim": args.target_dim,
                    "target_frames": args.target_frames,
                    "max_prototypes_per_label": args.max_prototypes_per_label,
                    "include_external_negatives": args.include_external_negatives,
                    "negative_conflict_filter_enabled": not args.disable_negative_conflict_filter,
                    "negative_conflict_method": args.negative_conflict_method,
                    "negative_conflict_margin": args.negative_conflict_margin,
                    "prototype_count": len(payload.get("prototypes", [])),
                }
            )
            for key, value in report["metrics"].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
            conflict = report.get("negative_conflict_filter") or {}
            for key in (
                "external_negative_total",
                "safe_external_negative_count",
                "conflict_count",
                "conflict_rate",
            ):
                value = conflict.get(key)
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"negative_{key}", float(value))
            mlflow.log_dict(report, "dynamic_prototype_report.json")
            model_path = Path(report["model_path"])
            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="model")
    except Exception as exc:
        print(f"[w] MLflow dynamic prototype logging failed: {exc}", flush=True)


def _parse_methods(raw: str) -> list[str]:
    methods = [item.strip().lower() for item in str(raw or "").split(",") if item.strip()]
    if not methods:
        return [METHOD_PROTOTYPE_DISTANCE, METHOD_PROTOTYPE_DTW]
    for method in methods:
        if method not in SUPPORTED_DYNAMIC_PROTOTYPE_METHODS:
            raise ValueError(f"unsupported method: {method}")
    return methods


def _parse_method(raw: str) -> str:
    method = str(raw or "").strip().lower()
    if method not in SUPPORTED_DYNAMIC_PROTOTYPE_METHODS:
        raise ValueError(f"unsupported method: {raw}")
    return method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--external-negative-root",
        type=Path,
        default=DEFAULT_EXTERNAL_NEGATIVE_ROOT,
    )
    parser.add_argument("--include-external-negatives", action="store_true")
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=_parse_methods("prototype_distance,prototype_dtw"),
    )
    parser.add_argument("--target-dim", type=int, default=44)
    parser.add_argument("--target-frames", type=int, default=36)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prototypes-per-label", type=int, default=8)
    parser.add_argument("--threshold-multiplier", type=float, default=1.25)
    parser.add_argument("--threshold-floor", type=float, default=0.015)
    parser.add_argument("--disable-negative-conflict-filter", action="store_true")
    parser.add_argument(
        "--negative-conflict-method",
        type=_parse_method,
        default=METHOD_PROTOTYPE_DISTANCE,
    )
    parser.add_argument("--negative-conflict-margin", type=float, default=1.20)
    parser.add_argument("--negative-conflict-max-report-rows", type=int, default=100)
    parser.add_argument("--variant-root", type=Path, default=DEFAULT_VARIANT_ROOT)
    parser.add_argument("--base-models-dir", type=Path, default=DEFAULT_BASE_MODELS_DIR)
    parser.add_argument("--write-production", action="store_true")
    parser.add_argument("--production-out", type=Path, default=DEFAULT_PRODUCTION_OUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(args)
    print(
        "[dynamic-prototype] "
        f"best={summary['best_method']} "
        f"train={summary['train_count']} test={summary['test_count']} "
        f"report={args.report_md}",
        flush=True,
    )


if __name__ == "__main__":
    main()
