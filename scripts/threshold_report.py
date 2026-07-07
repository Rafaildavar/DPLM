"""Build a confidence-threshold report for gesture classifiers.

Offline F1 is not enough for GestureBind: a wrong high-confidence prediction can
execute an OS command. This report evaluates the trade-off between accepted
prediction quality and rejected predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv.gesture_features import (  # noqa: E402
    FEATURE_STATIC_MEAN,
    GestureSequence,
    build_feature_matrix,
    class_counts,
    infer_target_dim,
    load_gesture_sequences,
)
from scripts.compare_models import build_estimators  # noqa: E402


@dataclass(frozen=True)
class DatasetInfo:
    data_root: str
    sample_count: int
    class_count: int
    class_counts: dict[str, int]
    target_dim: int
    cv_folds: int


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    coverage: float
    accepted_accuracy: float
    accepted_macro_f1: float
    rejected_count: int


@dataclass(frozen=True)
class CandidateThresholdReport:
    feature_mode: str
    model_name: str
    base_accuracy: float
    base_macro_f1: float
    recommended_threshold: ThresholdPoint
    curve: list[ThresholdPoint]


@dataclass(frozen=True)
class ThresholdReport:
    dataset: DatasetInfo
    candidates: list[CandidateThresholdReport]
    recommended_candidate: CandidateThresholdReport
    recommendation: str


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _select_records(
    records: Iterable[GestureSequence],
    min_samples_per_class: int,
) -> list[GestureSequence]:
    materialized = list(records)
    counts = class_counts(materialized)
    allowed = {label for label, count in counts.items() if count >= min_samples_per_class}
    return [record for record in materialized if record.label in allowed]


def _predict_confidence(estimator: Any, X: np.ndarray) -> np.ndarray | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    try:
        proba = estimator.predict_proba(X)
    except Exception:
        return None
    return np.max(proba, axis=1).astype(np.float32)


def _threshold_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    thresholds: list[float],
    class_count: int,
) -> list[ThresholdPoint]:
    points: list[ThresholdPoint] = []
    for threshold in thresholds:
        mask = confidence >= threshold
        accepted = int(np.count_nonzero(mask))
        rejected = int(len(confidence) - accepted)
        if accepted == 0:
            points.append(
                ThresholdPoint(
                    threshold=_round(threshold, 2),
                    coverage=0.0,
                    accepted_accuracy=0.0,
                    accepted_macro_f1=0.0,
                    rejected_count=rejected,
                )
            )
            continue

        accepted_true = y_true[mask]
        accepted_pred = y_pred[mask]
        points.append(
            ThresholdPoint(
                threshold=_round(threshold, 2),
                coverage=_round(accepted / len(confidence)),
                accepted_accuracy=_round(accuracy_score(accepted_true, accepted_pred)),
                accepted_macro_f1=_round(
                    f1_score(
                        accepted_true,
                        accepted_pred,
                        labels=list(range(class_count)),
                        average="macro",
                        zero_division=0,
                    )
                ),
                rejected_count=rejected,
            )
        )
    return points


def _choose_threshold(
    curve: list[ThresholdPoint],
    *,
    min_accuracy: float,
    min_coverage: float,
) -> ThresholdPoint:
    viable = [
        point
        for point in curve
        if point.coverage >= min_coverage and point.accepted_accuracy >= min_accuracy
    ]
    rows = viable or [point for point in curve if point.coverage > 0.0]
    if not rows:
        return curve[0]

    # Prefer accuracy first, then macro F1, then coverage. Lower threshold wins
    # ties to avoid over-rejecting in the live product.
    return sorted(
        rows,
        key=lambda point: (
            point.accepted_accuracy,
            point.accepted_macro_f1,
            point.coverage,
            -point.threshold,
        ),
        reverse=True,
    )[0]


def _cross_val_predictions(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    cv: StratifiedKFold,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []

    for train_idx, test_idx in cv.split(X, y):
        fitted = clone(estimator)
        fitted.fit(X[train_idx], y[train_idx])
        pred = fitted.predict(X[test_idx])
        confidence = _predict_confidence(fitted, X[test_idx])
        if confidence is None:
            confidence = np.ones(len(test_idx), dtype=np.float32)
        y_true_parts.append(y[test_idx])
        y_pred_parts.append(np.asarray(pred, dtype=np.int64))
        confidence_parts.append(confidence)

    return (
        np.concatenate(y_true_parts),
        np.concatenate(y_pred_parts),
        np.concatenate(confidence_parts),
    )


def evaluate_candidate_thresholds(
    records: list[GestureSequence],
    *,
    feature_mode: str,
    model_name: str,
    estimator: Any,
    cv: StratifiedKFold,
    target_dim: int,
    thresholds: list[float],
    min_accuracy: float,
    min_coverage: float,
) -> CandidateThresholdReport:
    X, y, labels = build_feature_matrix(records, mode=feature_mode, target_dim=target_dim)
    y_true, y_pred, confidence = _cross_val_predictions(X, y, estimator, cv)
    curve = _threshold_curve(
        y_true=y_true,
        y_pred=y_pred,
        confidence=confidence,
        thresholds=thresholds,
        class_count=len(labels),
    )
    return CandidateThresholdReport(
        feature_mode=feature_mode,
        model_name=model_name,
        base_accuracy=_round(accuracy_score(y_true, y_pred)),
        base_macro_f1=_round(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        recommended_threshold=_choose_threshold(
            curve,
            min_accuracy=min_accuracy,
            min_coverage=min_coverage,
        ),
        curve=curve,
    )


def parse_candidate(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"candidate must look like feature_mode:model_name, got {value!r}")
    feature_mode, model_name = value.split(":", 1)
    feature_mode = feature_mode.strip()
    model_name = model_name.strip()
    if not feature_mode or not model_name:
        raise ValueError(f"invalid candidate {value!r}")
    return feature_mode, model_name


def parse_candidates(value: str) -> list[tuple[str, str]]:
    return [parse_candidate(item.strip()) for item in value.split(",") if item.strip()]


def build_threshold_report(
    data_root: Path,
    *,
    candidates: list[tuple[str, str]],
    target_dim: int | None = None,
    min_samples_per_class: int = 5,
    max_folds: int = 5,
    random_state: int = 42,
    min_accuracy: float = 0.98,
    min_coverage: float = 0.60,
) -> ThresholdReport:
    records = _select_records(
        load_gesture_sequences(data_root),
        min_samples_per_class=min_samples_per_class,
    )
    if not records:
        raise RuntimeError("no classes with enough samples for threshold analysis")

    counts = class_counts(records)
    min_class_count = min(counts.values())
    folds = max(2, min(max_folds, min_class_count))
    actual_target_dim = int(target_dim or infer_target_dim(record.sequence for record in records))
    estimators = build_estimators(min_class_count=min_class_count, random_state=random_state)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    thresholds = [round(float(value), 2) for value in np.linspace(0.0, 1.0, 21)]

    reports: list[CandidateThresholdReport] = []
    for feature_mode, model_name in candidates:
        if model_name not in estimators:
            raise ValueError(f"unsupported model name: {model_name}")
        reports.append(
            evaluate_candidate_thresholds(
                records,
                feature_mode=feature_mode,
                model_name=model_name,
                estimator=estimators[model_name],
                cv=cv,
                target_dim=actual_target_dim,
                thresholds=thresholds,
                min_accuracy=min_accuracy,
                min_coverage=min_coverage,
            )
        )

    recommended = sorted(
        reports,
        key=lambda item: (
            item.recommended_threshold.accepted_accuracy,
            item.recommended_threshold.accepted_macro_f1,
            item.recommended_threshold.coverage,
            item.base_macro_f1,
            -item.recommended_threshold.threshold,
        ),
        reverse=True,
    )[0]
    point = recommended.recommended_threshold
    recommendation = (
        f"Use `{recommended.feature_mode}` + `{recommended.model_name}` with "
        f"confidence threshold {point.threshold:.2f}: accepted accuracy "
        f"{point.accepted_accuracy:.4f}, coverage {point.coverage:.4f}, "
        f"rejected predictions {point.rejected_count}."
    )

    return ThresholdReport(
        dataset=DatasetInfo(
            data_root=str(data_root),
            sample_count=len(records),
            class_count=len(counts),
            class_counts=counts,
            target_dim=actual_target_dim,
            cv_folds=folds,
        ),
        candidates=reports,
        recommended_candidate=recommended,
        recommendation=recommendation,
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _candidate_summary_table(report: ThresholdReport) -> str:
    rows: list[str] = []
    for candidate in report.candidates:
        point = candidate.recommended_threshold
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{candidate.feature_mode}`",
                    f"`{candidate.model_name}`",
                    _fmt(candidate.base_accuracy),
                    _fmt(candidate.base_macro_f1),
                    f"{point.threshold:.2f}",
                    _fmt(point.coverage),
                    _fmt(point.accepted_accuracy),
                    _fmt(point.accepted_macro_f1),
                    str(point.rejected_count),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _curve_table(candidate: CandidateThresholdReport) -> str:
    rows: list[str] = []
    for point in candidate.curve:
        rows.append(
            "| "
            + " | ".join(
                [
                    f"{point.threshold:.2f}",
                    _fmt(point.coverage),
                    _fmt(point.accepted_accuracy),
                    _fmt(point.accepted_macro_f1),
                    str(point.rejected_count),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def build_markdown(report: ThresholdReport) -> str:
    recommended = report.recommended_candidate
    count_rows = "\n".join(
        f"| `{label}` | {count} |" for label, count in report.dataset.class_counts.items()
    )
    return f"""# JMLC Threshold Report

## Краткий вывод

{report.recommendation}

## Датасет

| Класс | Семплы |
|---|---:|
{count_rows}

| Метрика | Значение |
|---|---:|
| Семплы | {report.dataset.sample_count} |
| Активные классы | {report.dataset.class_count} |
| Target dim | {report.dataset.target_dim} |
| CV folds | {report.dataset.cv_folds} |

## Candidate Summary

| Признаки | Модель | Base accuracy | Base macro F1 | Threshold | Coverage | Accepted accuracy | Accepted macro F1 | Rejected |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{_candidate_summary_table(report)}

## Recommended Curve

Кривая для `{recommended.feature_mode}` + `{recommended.model_name}`:

| Threshold | Coverage | Accepted accuracy | Accepted macro F1 | Rejected |
|---:|---:|---:|---:|---:|
{_curve_table(recommended)}

## Интерпретация

- `coverage` показывает долю предсказаний, которые останутся после порога.
- `accepted accuracy` показывает качество только среди принятых предсказаний.
- Высокий threshold может убрать ошибки, но сделать live-систему слишком
  молчаливой, поэтому этот отчет нужно подтвердить через `live_eval`.
"""


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def write_outputs(report: ThresholdReport, json_out: Path, markdown_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_out.write_text(build_markdown(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JMLC threshold report")
    parser.add_argument("--data-root", default="data/gestures", type=Path)
    parser.add_argument(
        "--candidates",
        default="static_mean:knn,static_mean:svm,static_stats:svm,hybrid_stats:knn",
        help="Comma-separated feature:model candidates",
    )
    parser.add_argument("--target-dim", default=0, type=int)
    parser.add_argument("--min-samples-per-class", default=5, type=int)
    parser.add_argument("--max-folds", default=5, type=int)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--min-accuracy", default=0.98, type=float)
    parser.add_argument("--min-coverage", default=0.60, type=float)
    parser.add_argument(
        "--json-out",
        default="docs/experiments/threshold_report.json",
        type=Path,
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/experiments/threshold_report.md",
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_threshold_report(
        data_root=args.data_root,
        candidates=parse_candidates(args.candidates),
        target_dim=args.target_dim or None,
        min_samples_per_class=args.min_samples_per_class,
        max_folds=args.max_folds,
        random_state=args.random_state,
        min_accuracy=args.min_accuracy,
        min_coverage=args.min_coverage,
    )
    write_outputs(report, json_out=args.json_out, markdown_out=args.markdown_out)
    print(f"[OK] Wrote {args.json_out} and {args.markdown_out}")


if __name__ == "__main__":
    main()

