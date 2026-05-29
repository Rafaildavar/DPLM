from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv.gesture_features import (  # noqa: E402
    FEATURE_DYNAMIC_STATS,
    FEATURE_STATIC_MEAN,
    SUPPORTED_FEATURE_MODES,
    GestureSequence,
    build_feature_matrix,
    class_counts,
    infer_target_dim,
    load_gesture_sequences,
)


@dataclass(frozen=True)
class DatasetInfo:
    data_root: str
    sample_count: int
    class_count: int
    class_counts: dict[str, int]
    raw_feature_dims: list[int]
    target_dim: int
    cv_folds: int


@dataclass(frozen=True)
class ThresholdSummary:
    threshold: float | None
    coverage: float | None
    accepted_accuracy: float | None
    rejected_count: int | None


@dataclass(frozen=True)
class ModelResult:
    feature_mode: str
    model_name: str
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    macro_false_positive_rate: float
    latency_ms_per_sample: float
    threshold: ThresholdSummary
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class ComparisonReport:
    dataset: DatasetInfo
    results: list[ModelResult]
    best_result: ModelResult
    notes: list[str]


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _macro_false_positive_rate(cm: np.ndarray) -> float:
    total = int(cm.sum())
    if total == 0:
        return 0.0
    rates: list[float] = []
    for idx in range(cm.shape[0]):
        fp = int(cm[:, idx].sum() - cm[idx, idx])
        fn = int(cm[idx, :].sum() - cm[idx, idx])
        tp = int(cm[idx, idx])
        tn = total - tp - fp - fn
        denom = fp + tn
        rates.append(0.0 if denom == 0 else fp / denom)
    return float(np.mean(rates))


def _predict_confidence(estimator: Any, X: np.ndarray) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X)
            return np.max(proba, axis=1).astype(np.float32)
        except Exception:
            return None
    return None


def recommend_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray | None,
) -> ThresholdSummary:
    if confidence is None or len(confidence) == 0:
        return ThresholdSummary(None, None, None, None)

    candidates = np.linspace(0.0, 1.0, 21)
    viable: list[tuple[float, float, float, int]] = []
    fallback: list[tuple[float, float, float, int]] = []
    for threshold in candidates:
        mask = confidence >= threshold
        accepted = int(np.count_nonzero(mask))
        rejected = int(len(confidence) - accepted)
        if accepted == 0:
            continue
        accepted_accuracy = float(np.mean(y_true[mask] == y_pred[mask]))
        coverage = float(accepted / len(confidence))
        row = (float(threshold), coverage, accepted_accuracy, rejected)
        fallback.append(row)
        if accepted_accuracy >= 0.90 and coverage >= 0.50:
            viable.append(row)

    rows = viable or fallback
    if not rows:
        return ThresholdSummary(None, None, None, None)
    best = sorted(rows, key=lambda item: (item[2], item[1], -item[0]), reverse=True)[0]
    return ThresholdSummary(
        threshold=_round(best[0], 2),
        coverage=_round(best[1], 4),
        accepted_accuracy=_round(best[2], 4),
        rejected_count=int(best[3]),
    )


def build_estimators(min_class_count: int, random_state: int) -> dict[str, Any]:
    knn_neighbors = max(1, min(3, int(min_class_count)))
    return {
        "knn": KNeighborsClassifier(
            n_neighbors=knn_neighbors,
            metric="euclidean",
            weights="distance",
        ),
        "svm": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=random_state,
            ),
        ),
        "rf": RandomForestClassifier(
            n_estimators=120,
            random_state=random_state,
            class_weight="balanced",
        ),
    }


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    model_name: str,
    feature_mode: str,
    cv: StratifiedKFold,
    labels: list[str],
) -> ModelResult:
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    latency_values: list[float] = []

    for train_idx, test_idx in cv.split(X, y):
        fitted = clone(estimator)
        fitted.fit(X[train_idx], y[train_idx])

        start = time.perf_counter()
        pred = fitted.predict(X[test_idx])
        elapsed = time.perf_counter() - start
        latency_values.append((elapsed / max(1, len(test_idx))) * 1000.0)

        conf = _predict_confidence(fitted, X[test_idx])
        if conf is not None:
            confidence_parts.append(conf)
        y_true_parts.append(y[test_idx])
        y_pred_parts.append(np.asarray(pred, dtype=np.int64))

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    confidence = np.concatenate(confidence_parts) if confidence_parts else None
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    return ModelResult(
        feature_mode=feature_mode,
        model_name=model_name,
        accuracy=_round(accuracy_score(y_true, y_pred)) or 0.0,
        macro_f1=_round(f1_score(y_true, y_pred, average="macro", zero_division=0)) or 0.0,
        macro_precision=_round(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        or 0.0,
        macro_recall=_round(recall_score(y_true, y_pred, average="macro", zero_division=0)) or 0.0,
        macro_false_positive_rate=_round(_macro_false_positive_rate(cm)) or 0.0,
        latency_ms_per_sample=_round(float(np.mean(latency_values)), digits=3) or 0.0,
        threshold=recommend_threshold(y_true, y_pred, confidence),
        confusion_matrix=cm.astype(int).tolist(),
    )


def _select_records(records: Iterable[GestureSequence], min_samples_per_class: int) -> list[GestureSequence]:
    counts = class_counts(list(records))
    allowed = {label for label, count in counts.items() if count >= min_samples_per_class}
    return [record for record in records if record.label in allowed]


def compare_models(
    data_root: Path,
    feature_modes: list[str] | None = None,
    model_names: list[str] | None = None,
    target_dim: int | None = None,
    min_samples_per_class: int = 2,
    max_folds: int = 5,
    random_state: int = 42,
) -> ComparisonReport:
    records = load_gesture_sequences(data_root)
    selected_records = _select_records(records, min_samples_per_class=min_samples_per_class)
    if not selected_records:
        raise RuntimeError("no classes with enough samples for comparison")

    counts = class_counts(selected_records)
    min_class_count = min(counts.values())
    if min_class_count < 2:
        raise RuntimeError("at least two samples per active class are required")

    folds = max(2, min(int(max_folds), int(min_class_count)))
    actual_target_dim = int(target_dim or infer_target_dim(record.sequence for record in selected_records))
    modes = feature_modes or [FEATURE_STATIC_MEAN, FEATURE_DYNAMIC_STATS]
    unknown_modes = [mode for mode in modes if mode not in SUPPORTED_FEATURE_MODES]
    if unknown_modes:
        raise ValueError(f"unsupported feature modes: {', '.join(unknown_modes)}")

    estimators = build_estimators(min_class_count=min_class_count, random_state=random_state)
    names = model_names or list(estimators)
    unknown_models = [name for name in names if name not in estimators]
    if unknown_models:
        raise ValueError(f"unsupported model names: {', '.join(unknown_models)}")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    results: list[ModelResult] = []
    labels_for_report: list[str] = []
    for mode in modes:
        X, y, labels = build_feature_matrix(selected_records, mode=mode, target_dim=actual_target_dim)
        labels_for_report = labels
        for name in names:
            results.append(
                evaluate_model(
                    X=X,
                    y=y,
                    estimator=estimators[name],
                    model_name=name,
                    feature_mode=mode,
                    cv=cv,
                    labels=labels,
                )
            )

    raw_dims = sorted({record.feature_dim for record in selected_records})
    notes = [
        f"Активные классы в сравнении: {', '.join(labels_for_report)}",
        f"CV folds: {folds}; минимальный размер класса: {min_class_count}",
    ]
    if len(raw_dims) > 1:
        notes.append(
            "Обнаружены разные исходные размерности признаков; для этого benchmark семплы "
            f"были дополнены или обрезаны до target_dim {actual_target_dim}."
        )
    if max(counts.values()) / min_class_count >= 3:
        notes.append("Дисбаланс классов высокий; macro-метрики важнее обычной accuracy.")

    dataset = DatasetInfo(
        data_root=str(data_root),
        sample_count=len(selected_records),
        class_count=len(counts),
        class_counts=counts,
        raw_feature_dims=raw_dims,
        target_dim=actual_target_dim,
        cv_folds=folds,
    )
    best = sorted(
        results,
        key=lambda item: (item.macro_f1, item.accuracy, -item.latency_ms_per_sample),
        reverse=True,
    )[0]
    return ComparisonReport(dataset=dataset, results=results, best_result=best, notes=notes)


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _format_latency(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _confusion_matrix_markdown(cm: list[list[int]], labels: list[str]) -> str:
    header = "| true \\ pred | " + " | ".join(f"`{label}`" for label in labels) + " |"
    divider = "|---|" + "|".join("---:" for _ in labels) + "|"
    rows = [header, divider]
    for label, row in zip(labels, cm):
        rows.append("| `" + label + "` | " + " | ".join(str(int(value)) for value in row) + " |")
    return "\n".join(rows)


def build_markdown_report(report: ComparisonReport) -> str:
    best = report.best_result
    labels = list(report.dataset.class_counts)
    rows = []
    for result in report.results:
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{result.feature_mode}`",
                    f"`{result.model_name}`",
                    _format_metric(result.accuracy),
                    _format_metric(result.macro_f1),
                    _format_metric(result.macro_precision),
                    _format_metric(result.macro_recall),
                    _format_metric(result.macro_false_positive_rate),
                    _format_latency(result.latency_ms_per_sample),
                    _format_metric(result.threshold.threshold),
                ]
            )
            + " |"
        )

    count_rows = "\n".join(
        f"| `{label}` | {count} |" for label, count in report.dataset.class_counts.items()
    )
    notes = "\n".join(f"- {note}" for note in report.notes)
    confusion = _confusion_matrix_markdown(best.confusion_matrix, labels)

    return f"""# JMLC Stage 2 - Model Comparison

## Краткий вывод

- Лучший результат: **`{best.feature_mode}` + `{best.model_name}`** с macro F1 `{best.macro_f1:.4f}` и accuracy `{best.accuracy:.4f}`.
- Датасет: {report.dataset.sample_count} семплов, {report.dataset.class_count} активных класса, целевая размерность признаков {report.dataset.target_dim}.
- Ценность для JMLC: это первый воспроизводимый слой доказательств, где признаки и модели сравниваются на одинаковых folds, а не описываются словами.

## Датасет

| Класс | Семплы |
|---|---:|
{count_rows}

| Метрика | Значение |
|---|---:|
| Исходные размерности | {", ".join(str(dim) for dim in report.dataset.raw_feature_dims)} |
| Целевая размерность | {report.dataset.target_dim} |
| CV folds | {report.dataset.cv_folds} |

## Результаты

| Признаки | Модель | Accuracy | Macro F1 | Macro precision | Macro recall | Macro FPR | Latency ms/sample | Рекомендованный threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Матрица ошибок лучшей модели

{confusion}

## Интерпретация threshold

Сводка по threshold для лучшей модели:

| Метрика | Значение |
|---|---:|
| Threshold | {_format_metric(best.threshold.threshold)} |
| Coverage | {_format_metric(best.threshold.coverage)} |
| Accuracy на принятых предсказаниях | {_format_metric(best.threshold.accepted_accuracy)} |
| Отклонено предсказаний | {best.threshold.rejected_count if best.threshold.rejected_count is not None else "n/a"} |

## Наблюдения

{notes}

## Оценка для JMLC

Что теперь выглядит сильнее:

1. У проекта есть воспроизводимый ML-benchmark, а не только ручной скрипт обучения.
2. Отчет сравнивает стратегии извлечения признаков и семейства моделей на одинаковой валидации.
3. Macro-метрики и матрица ошибок честно показывают маленький и несбалансированный датасет комиссии.

Риски, которые нужно закрывать дальше:

1. Датасет пока маленький и несбалансированный, особенно для `new2`.
2. One-hand и two-hand семплы в этом benchmark смешаны через padding; дальше нужно разделить сценарии или собрать совместимые классы.
3. Live false trigger rate пока не измерен; Stage 3 должен связать benchmark-threshold с журналом `recognition_logs`.
"""


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DPLM gesture recognition models")
    parser.add_argument("--data-root", default="data/gestures", help="Gesture dataset root")
    parser.add_argument(
        "--feature-modes",
        default="static_mean,dynamic_stats",
        help="Comma-separated feature modes",
    )
    parser.add_argument("--models", default="knn,svm,rf", help="Comma-separated model names")
    parser.add_argument("--target-dim", type=int, default=0, help="0 means infer max raw dimension")
    parser.add_argument("--min-samples-per-class", type=int, default=2)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--json-out", default="docs/experiments/model_comparison.json")
    parser.add_argument("--markdown-out", default="docs/experiments/model_comparison.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_models(
        data_root=Path(args.data_root),
        feature_modes=_parse_csv(args.feature_modes),
        model_names=_parse_csv(args.models),
        target_dim=args.target_dim or None,
        min_samples_per_class=args.min_samples_per_class,
        max_folds=args.max_folds,
        random_state=args.random_state,
    )

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown_path = Path(args.markdown_out)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"[OK] Wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
