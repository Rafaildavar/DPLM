"""Compare gesture feature modes and classifiers for GestureBind.

This script is deliberately CLI-first: it can run in CI and does not require a
camera or Flet. The report answers whether a universal model is enough, or
whether static-like and dynamic-like gestures should use different pipelines.
"""

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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv.gesture_dataset_files import gesture_sample_paths  # noqa: E402
from cv.gesture_features import (  # noqa: E402
    FEATURE_DYNAMIC_CRAFT_FULL_STATS,
    FEATURE_DYNAMIC_CRAFT_STATS,
    FEATURE_DYNAMIC_STATS,
    FEATURE_HYBRID_STATS,
    FEATURE_STATIC_CRAFT_FULL_STATS,
    FEATURE_STATIC_LANDMARK_IMAGE,
    FEATURE_STATIC_MEAN,
    FEATURE_STATIC_STATS,
    MOTION_DYNAMIC_LIKE,
    MOTION_STATIC_LIKE,
    SUPPORTED_FEATURE_MODES,
    ClassMotionProfile,
    GestureSequence,
    build_feature_matrix,
    class_counts,
    class_motion_profiles,
    infer_target_dim,
    load_gesture_sequences,
    sequence_to_matrix,
)
from cv.train_classifier import (  # noqa: E402
    balance_training_set,
    build_classifier,
    _gislr_augmented_sample_paths,
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
    motion_threshold: float
    motion_profiles: list[ClassMotionProfile]
    include_augmented: bool = False


@dataclass(frozen=True)
class ThresholdSummary:
    threshold: float | None
    coverage: float | None
    accepted_accuracy: float | None
    rejected_count: int | None


@dataclass(frozen=True)
class PerClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int
    motion_type: str


@dataclass(frozen=True)
class ModelResult:
    feature_mode: str
    model_name: str
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    macro_false_positive_rate: float
    static_like_macro_f1: float | None
    dynamic_like_macro_f1: float | None
    latency_ms_per_sample: float
    threshold: ThresholdSummary
    per_class: list[PerClassMetrics]
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class ComparisonReport:
    dataset: DatasetInfo
    results: list[ModelResult]
    best_overall: ModelResult
    best_static_like: ModelResult | None
    best_dynamic_like: ModelResult | None
    recommendation: str
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
    if not hasattr(estimator, "predict_proba"):
        return None
    try:
        proba = estimator.predict_proba(X)
    except Exception:
        return None
    return np.max(proba, axis=1).astype(np.float32)


def recommend_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray | None,
    *,
    min_accepted_accuracy: float = 0.95,
    min_coverage: float = 0.50,
) -> ThresholdSummary:
    if confidence is None or len(confidence) == 0:
        return ThresholdSummary(None, None, None, None)

    viable: list[tuple[float, float, float, int]] = []
    fallback: list[tuple[float, float, float, int]] = []
    for threshold in np.linspace(0.0, 1.0, 21):
        mask = confidence >= threshold
        accepted = int(np.count_nonzero(mask))
        if accepted == 0:
            continue
        rejected = int(len(confidence) - accepted)
        accepted_accuracy = float(np.mean(y_true[mask] == y_pred[mask]))
        coverage = float(accepted / len(confidence))
        row = (float(threshold), coverage, accepted_accuracy, rejected)
        fallback.append(row)
        if accepted_accuracy >= min_accepted_accuracy and coverage >= min_coverage:
            viable.append(row)

    rows = viable or fallback
    if not rows:
        return ThresholdSummary(None, None, None, None)

    # Prefer reliable accepted predictions, then wider coverage, then lower
    # threshold so live UX does not become overly silent.
    best = sorted(rows, key=lambda item: (item[2], item[1], -item[0]), reverse=True)[0]
    return ThresholdSummary(
        threshold=_round(best[0], 2),
        coverage=_round(best[1], 4),
        accepted_accuracy=_round(best[2], 4),
        rejected_count=int(best[3]),
    )


def _force_serial_jobs(estimator: Any) -> Any:
    if not hasattr(estimator, "get_params") or not hasattr(estimator, "set_params"):
        return estimator
    params = estimator.get_params(deep=True)
    serial_params = {
        name: 1
        for name in params
        if name == "n_jobs" or name.endswith("__n_jobs")
    }
    if serial_params:
        estimator.set_params(**serial_params)
    return estimator


def build_estimators(
    min_class_count: int,
    random_state: int,
    *,
    static_cnn_max_epochs: int = 30,
    static_cnn_patience: int = 6,
    static_cnn_batch_size: int = 16,
) -> dict[str, Any]:
    knn_neighbors = max(1, min(5, int(min_class_count) - 1))
    stacking_cv = max(2, min(3, max(2, int(min_class_count) - 1)))
    static_stacking = _force_serial_jobs(
        build_classifier(
            "static_stacking",
            random_state=random_state,
            static_stacking_cv_folds=stacking_cv,
        )
    )
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
                C=2.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=random_state,
            ),
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced",
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=250,
            random_state=random_state,
            class_weight="balanced",
        ),
        "static_stacking": static_stacking,
        "static_landmark_cnn": build_classifier(
            "static_landmark_cnn",
            random_state=random_state,
            static_cnn_max_epochs=max(1, int(static_cnn_max_epochs)),
            static_cnn_patience=max(1, int(static_cnn_patience)),
            static_cnn_batch_size=max(1, int(static_cnn_batch_size)),
        ),
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
    }


def _group_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    motion_type_by_label: dict[str, str],
    target_motion_type: str,
) -> float | None:
    selected = [
        idx
        for idx, label in enumerate(labels)
        if motion_type_by_label.get(label) == target_motion_type
    ]
    if not selected:
        return None
    return float(f1_score(y_true, y_pred, labels=selected, average="macro", zero_division=0))


def _per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    motion_type_by_label: dict[str, str],
) -> list[PerClassMetrics]:
    precision = precision_score(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        average=None,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        average=None,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        average=None,
        zero_division=0,
    )

    rows: list[PerClassMetrics] = []
    for idx, label in enumerate(labels):
        rows.append(
            PerClassMetrics(
                label=label,
                precision=_round(float(precision[idx])) or 0.0,
                recall=_round(float(recall[idx])) or 0.0,
                f1=_round(float(f1[idx])) or 0.0,
                support=int(np.count_nonzero(y_true == idx)),
                motion_type=motion_type_by_label.get(label, MOTION_STATIC_LIKE),
            )
        )
    return rows


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    model_name: str,
    feature_mode: str,
    cv: StratifiedKFold,
    labels: list[str],
    motion_type_by_label: dict[str, str],
    random_state: int = 42,
) -> ModelResult:
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    latency_values: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fitted = clone(estimator)
        X_fit, y_fit, _balance_metadata = balance_training_set(
            X[train_idx],
            y[train_idx],
            labels,
            model_type=model_name,
            strategy="auto",
            random_state=int(random_state) + int(fold_idx),
        )
        fitted.fit(X_fit, y_fit)

        start = time.perf_counter()
        pred = fitted.predict(X[test_idx])
        elapsed = time.perf_counter() - start
        latency_values.append((elapsed / max(1, len(test_idx))) * 1000.0)

        confidence = _predict_confidence(fitted, X[test_idx])
        if confidence is not None:
            confidence_parts.append(confidence)
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
        static_like_macro_f1=_round(
            _group_macro_f1(y_true, y_pred, labels, motion_type_by_label, MOTION_STATIC_LIKE)
        ),
        dynamic_like_macro_f1=_round(
            _group_macro_f1(y_true, y_pred, labels, motion_type_by_label, MOTION_DYNAMIC_LIKE)
        ),
        latency_ms_per_sample=_round(float(np.mean(latency_values)), digits=3) or 0.0,
        threshold=recommend_threshold(y_true, y_pred, confidence),
        per_class=_per_class_metrics(y_true, y_pred, labels, motion_type_by_label),
        confusion_matrix=cm.astype(int).tolist(),
    )


def _select_records(
    records: Iterable[GestureSequence],
    min_samples_per_class: int,
) -> list[GestureSequence]:
    materialized = list(records)
    counts = class_counts(materialized)
    allowed = {label for label, count in counts.items() if count >= min_samples_per_class}
    return [record for record in materialized if record.label in allowed]


def _load_records_for_comparison(
    data_root: Path,
    *,
    include_augmented: bool,
    include_labels: Iterable[str] | None,
    lowercase_labels: bool,
) -> list[GestureSequence]:
    raw_include = {
        str(label).strip()
        for label in (include_labels or [])
        if str(label).strip()
    }
    canonical_include = {
        label.lower() if lowercase_labels else label for label in raw_include
    }
    if not include_augmented and not raw_include and not lowercase_labels:
        return load_gesture_sequences(data_root)

    records: list[GestureSequence] = []
    if not data_root.exists():
        return records
    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        raw_label = label_dir.name
        label = raw_label.lower() if lowercase_labels else raw_label
        if canonical_include and label not in canonical_include and raw_label not in raw_include:
            continue
        sample_paths = gesture_sample_paths(label_dir)
        if include_augmented:
            sample_paths = [*sample_paths, *_gislr_augmented_sample_paths(label_dir)]
        for sample_path in sample_paths:
            try:
                sequence = sequence_to_matrix(np.load(sample_path))
            except Exception:
                continue
            records.append(GestureSequence(label=label, path=sample_path, sequence=sequence))
    return records


def _best_by_group(
    results: list[ModelResult],
    attr: str,
) -> ModelResult | None:
    eligible = [result for result in results if getattr(result, attr) is not None]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda item: (getattr(item, attr), item.macro_f1, -item.latency_ms_per_sample),
        reverse=True,
    )[0]


def _build_recommendation(
    best_overall: ModelResult,
    best_static_like: ModelResult | None,
    best_dynamic_like: ModelResult | None,
    profiles: list[ClassMotionProfile],
) -> str:
    dynamic_count = sum(
        1 for profile in profiles if profile.suggested_type == MOTION_DYNAMIC_LIKE
    )
    static_count = sum(1 for profile in profiles if profile.suggested_type == MOTION_STATIC_LIKE)

    if dynamic_count == 0:
        return (
            "Текущий датасет выглядит static-like по выбранному motion threshold. "
            "Пока рано доказывать отдельную dynamic-модель: нужно записать жесты "
            "с выраженным движением и повторить benchmark."
        )
    if static_count == 0:
        return (
            "Все активные классы выглядят dynamic-like. Можно начинать с единого "
            f"пайплайна `{best_overall.feature_mode}` + `{best_overall.model_name}`, "
            "а static-набор собрать отдельно."
        )
    if not best_static_like or not best_dynamic_like:
        return (
            f"Использовать единый пайплайн `{best_overall.feature_mode}` + "
            f"`{best_overall.model_name}` до расширения датасета."
        )

    same_pipeline = (
        best_static_like.feature_mode == best_dynamic_like.feature_mode
        and best_static_like.model_name == best_dynamic_like.model_name
    )
    if same_pipeline:
        return (
            f"На текущем датасете достаточно единого пайплайна "
            f"`{best_overall.feature_mode}` + `{best_overall.model_name}`: он не "
            "проигрывает отдельно static-like/dynamic-like группам."
        )

    static_score = best_static_like.static_like_macro_f1 or 0.0
    dynamic_score = best_dynamic_like.dynamic_like_macro_f1 or 0.0
    return (
        "Есть сигнал к разделению пайплайнов: для static-like лучше "
        f"`{best_static_like.feature_mode}` + `{best_static_like.model_name}` "
        f"(F1={static_score:.4f}), для dynamic-like лучше "
        f"`{best_dynamic_like.feature_mode}` + `{best_dynamic_like.model_name}` "
        f"(F1={dynamic_score:.4f}). Это нужно подтвердить после дозаписи данных."
    )


def compare_models(
    data_root: Path,
    feature_modes: list[str] | None = None,
    model_names: list[str] | None = None,
    target_dim: int | None = None,
    min_samples_per_class: int = 5,
    max_folds: int = 5,
    random_state: int = 42,
    motion_threshold: float = 0.015,
    include_augmented: bool = False,
    include_labels: Iterable[str] | None = None,
    lowercase_labels: bool = False,
    static_cnn_max_epochs: int = 30,
    static_cnn_patience: int = 6,
    static_cnn_batch_size: int = 16,
) -> ComparisonReport:
    records = _load_records_for_comparison(
        data_root,
        include_augmented=include_augmented,
        include_labels=include_labels,
        lowercase_labels=lowercase_labels,
    )
    selected_records = _select_records(
        records,
        min_samples_per_class=min_samples_per_class,
    )
    if not selected_records:
        raise RuntimeError("no classes with enough samples for comparison")

    counts = class_counts(selected_records)
    min_class_count = min(counts.values())
    if min_class_count < 2:
        raise RuntimeError("at least two samples per active class are required")

    folds = max(2, min(int(max_folds), int(min_class_count)))
    actual_target_dim = int(target_dim or infer_target_dim(record.sequence for record in selected_records))
    modes = feature_modes or [
        FEATURE_STATIC_MEAN,
        FEATURE_STATIC_STATS,
        FEATURE_STATIC_CRAFT_FULL_STATS,
        FEATURE_STATIC_LANDMARK_IMAGE,
        FEATURE_DYNAMIC_STATS,
        FEATURE_DYNAMIC_CRAFT_STATS,
        FEATURE_DYNAMIC_CRAFT_FULL_STATS,
        FEATURE_HYBRID_STATS,
    ]
    unknown_modes = [mode for mode in modes if mode not in SUPPORTED_FEATURE_MODES]
    if unknown_modes:
        raise ValueError(f"unsupported feature modes: {', '.join(unknown_modes)}")

    estimators = build_estimators(
        min_class_count=min_class_count,
        random_state=random_state,
        static_cnn_max_epochs=static_cnn_max_epochs,
        static_cnn_patience=static_cnn_patience,
        static_cnn_batch_size=static_cnn_batch_size,
    )
    names = model_names or ["knn", "svm", "rf", "extra_trees", "logreg"]
    unknown_models = [name for name in names if name not in estimators]
    if unknown_models:
        raise ValueError(f"unsupported model names: {', '.join(unknown_models)}")

    profiles = class_motion_profiles(
        selected_records,
        motion_threshold=motion_threshold,
    )
    motion_type_by_label = {profile.label: profile.suggested_type for profile in profiles}

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    results: list[ModelResult] = []
    skipped_pairs: list[str] = []
    labels_for_report: list[str] = []
    for mode in modes:
        X, y, labels = build_feature_matrix(
            selected_records,
            mode=mode,
            target_dim=actual_target_dim,
        )
        labels_for_report = labels
        for name in names:
            if name == "static_landmark_cnn" and mode != FEATURE_STATIC_LANDMARK_IMAGE:
                skipped_pairs.append(f"{mode}+{name}")
                continue
            results.append(
                evaluate_model(
                    X=X,
                    y=y,
                    estimator=estimators[name],
                    model_name=name,
                    feature_mode=mode,
                    cv=cv,
                    labels=labels,
                    motion_type_by_label=motion_type_by_label,
                    random_state=random_state,
                )
            )
    if not results:
        raise RuntimeError(
            "no compatible feature/model pairs were selected for comparison"
        )

    raw_dims = sorted({record.feature_dim for record in selected_records})
    notes = [
        f"Активные классы в сравнении: {', '.join(labels_for_report)}",
        f"CV folds: {folds}; минимальный размер класса: {min_class_count}",
        "Augmented samples: " + ("included" if include_augmented else "excluded"),
    ]
    if skipped_pairs:
        notes.append(
            "Пропущены несовместимые пары feature/model: "
            + ", ".join(f"`{pair}`" for pair in skipped_pairs)
        )
    if len(raw_dims) > 1:
        notes.append(
            "Обнаружены разные исходные размерности признаков; benchmark выравнивает "
            f"семплы до target_dim {actual_target_dim}."
        )
    if max(counts.values()) / min_class_count >= 2.5:
        notes.append("Дисбаланс классов высокий; macro-метрики важнее accuracy.")

    dataset = DatasetInfo(
        data_root=str(data_root),
        sample_count=len(selected_records),
        class_count=len(counts),
        class_counts=counts,
        raw_feature_dims=raw_dims,
        target_dim=actual_target_dim,
        cv_folds=folds,
        motion_threshold=float(motion_threshold),
        motion_profiles=profiles,
        include_augmented=bool(include_augmented),
    )
    best_overall = sorted(
        results,
        key=lambda item: (item.macro_f1, item.accuracy, -item.latency_ms_per_sample),
        reverse=True,
    )[0]
    best_static_like = _best_by_group(results, "static_like_macro_f1")
    best_dynamic_like = _best_by_group(results, "dynamic_like_macro_f1")
    recommendation = _build_recommendation(
        best_overall=best_overall,
        best_static_like=best_static_like,
        best_dynamic_like=best_dynamic_like,
        profiles=profiles,
    )
    return ComparisonReport(
        dataset=dataset,
        results=results,
        best_overall=best_overall,
        best_static_like=best_static_like,
        best_dynamic_like=best_dynamic_like,
        recommendation=recommendation,
        notes=notes,
    )


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


def _motion_table(profiles: list[ClassMotionProfile]) -> str:
    return "\n".join(
        "| "
        + " | ".join(
            [
                f"`{profile.label}`",
                str(profile.sample_count),
                _format_metric(profile.median_motion_energy),
                _format_metric(profile.median_displacement),
                f"`{profile.suggested_type}`",
            ]
        )
        + " |"
        for profile in profiles
    )


def _results_table(results: list[ModelResult]) -> str:
    rows: list[str] = []
    ranked = sorted(
        results,
        key=lambda item: (item.macro_f1, item.accuracy, -item.latency_ms_per_sample),
        reverse=True,
    )
    for result in ranked:
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{result.feature_mode}`",
                    f"`{result.model_name}`",
                    _format_metric(result.accuracy),
                    _format_metric(result.macro_f1),
                    _format_metric(result.static_like_macro_f1),
                    _format_metric(result.dynamic_like_macro_f1),
                    _format_metric(result.macro_false_positive_rate),
                    _format_latency(result.latency_ms_per_sample),
                    _format_metric(result.threshold.threshold),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _per_class_table(result: ModelResult) -> str:
    return "\n".join(
        "| "
        + " | ".join(
            [
                f"`{row.label}`",
                f"`{row.motion_type}`",
                str(row.support),
                _format_metric(row.precision),
                _format_metric(row.recall),
                _format_metric(row.f1),
            ]
        )
        + " |"
        for row in result.per_class
    )


def build_markdown_report(report: ComparisonReport) -> str:
    best = report.best_overall
    labels = list(report.dataset.class_counts)
    count_rows = "\n".join(
        f"| `{label}` | {count} |" for label, count in report.dataset.class_counts.items()
    )
    notes = "\n".join(f"- {note}" for note in report.notes)
    confusion = _confusion_matrix_markdown(best.confusion_matrix, labels)
    motion_rows = _motion_table(report.dataset.motion_profiles)
    result_rows = _results_table(report.results)
    per_class_rows = _per_class_table(best)

    return f"""# GestureBind Cross-Validation Model Comparison

## Краткий вывод

- Лучший общий результат: **`{best.feature_mode}` + `{best.model_name}`** с macro F1 `{best.macro_f1:.4f}` и accuracy `{best.accuracy:.4f}`.
- Датасет: {report.dataset.sample_count} семплов, {report.dataset.class_count} активных классов, target_dim {report.dataset.target_dim}.
- Рекомендация: {report.recommendation}

## Датасет

| Класс | Семплы |
|---|---:|
{count_rows}

| Метрика | Значение |
|---|---:|
| Исходные размерности | {", ".join(str(dim) for dim in report.dataset.raw_feature_dims)} |
| Целевая размерность | {report.dataset.target_dim} |
| CV folds | {report.dataset.cv_folds} |
| Motion threshold | {report.dataset.motion_threshold:.4f} |
| Augmented samples | {"included" if report.dataset.include_augmented else "excluded"} |

## Motion Profile

| Класс | Семплы | Median motion energy | Median displacement | Suggested type |
|---|---:|---:|---:|---|
{motion_rows}

## Результаты

| Признаки | Модель | Accuracy | Macro F1 | Static-like F1 | Dynamic-like F1 | Macro FPR | Latency ms/sample | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{result_rows}

## Лучшая модель: per-class метрики

| Класс | Motion type | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
{per_class_rows}

## Матрица ошибок лучшей модели

{confusion}

## Threshold для лучшей модели

| Метрика | Значение |
|---|---:|
| Threshold | {_format_metric(best.threshold.threshold)} |
| Coverage | {_format_metric(best.threshold.coverage)} |
| Accuracy на принятых предсказаниях | {_format_metric(best.threshold.accepted_accuracy)} |
| Отклонено предсказаний | {best.threshold.rejected_count if best.threshold.rejected_count is not None else "n/a"} |

## Наблюдения

{notes}

## Следующий эксперимент

1. Если dynamic-like классов мало или нет, записать 2-3 жеста с выраженным
   движением и повторить benchmark.
2. Если static-like и dynamic-like группы выбирают разные пайплайны, проверить
   split-routing: сначала классифицировать тип жеста, затем запускать отдельную
   модель.
3. Подтвердить recommended threshold через live-eval в приложении.
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
    parser = argparse.ArgumentParser(description="Compare GestureBind gesture recognition models")
    parser.add_argument("--data-root", default="data/gestures", type=Path)
    parser.add_argument(
        "--feature-modes",
        default=(
            "static_mean,static_stats,static_craft_full_stats,"
            "dynamic_stats,dynamic_craft_stats,dynamic_craft_full_stats,hybrid_stats"
        ),
        help="Comma-separated feature modes",
    )
    parser.add_argument(
        "--models",
        default="knn,svm,rf,extra_trees,logreg",
        help="Comma-separated model names",
    )
    parser.add_argument("--target-dim", type=int, default=0, help="0 means infer max raw dimension")
    parser.add_argument("--min-samples-per-class", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--motion-threshold", type=float, default=0.015)
    parser.add_argument(
        "--include-augmented",
        action="store_true",
        help="Include GISLR-marked aug_sample_* files with gislr_landmark_v1 metadata.",
    )
    parser.add_argument(
        "--include-label",
        action="append",
        default=[],
        help="Restrict comparison to a label. Can be passed multiple times.",
    )
    parser.add_argument(
        "--lowercase-labels",
        action="store_true",
        help="Normalize labels to lowercase before filtering and CV.",
    )
    parser.add_argument(
        "--static-cnn-max-epochs",
        type=int,
        default=30,
        help="Maximum epochs for static_landmark_cnn during CV.",
    )
    parser.add_argument(
        "--static-cnn-patience",
        type=int,
        default=6,
        help="Early-stopping patience for static_landmark_cnn during CV.",
    )
    parser.add_argument(
        "--static-cnn-batch-size",
        type=int,
        default=16,
        help="Mini-batch size for static_landmark_cnn during CV.",
    )
    parser.add_argument("--json-out", default="docs/experiments/model_comparison.json", type=Path)
    parser.add_argument("--markdown-out", default="docs/experiments/model_comparison.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_models(
        data_root=args.data_root,
        feature_modes=_parse_csv(args.feature_modes),
        model_names=_parse_csv(args.models),
        target_dim=args.target_dim or None,
        min_samples_per_class=args.min_samples_per_class,
        max_folds=args.max_folds,
        random_state=args.random_state,
        motion_threshold=args.motion_threshold,
        include_augmented=args.include_augmented,
        include_labels=args.include_label,
        lowercase_labels=args.lowercase_labels,
        static_cnn_max_epochs=args.static_cnn_max_epochs,
        static_cnn_patience=args.static_cnn_patience,
        static_cnn_batch_size=args.static_cnn_batch_size,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"[OK] Wrote {args.json_out} and {args.markdown_out}")


if __name__ == "__main__":
    main()
