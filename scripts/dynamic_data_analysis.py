"""Analyze dynamic gesture samples and data-processing risks for JMLC.

The report focuses on dynamic gestures such as ``swipe_up`` and
``swipe_down``. It connects three layers that matter for the contest:

* raw recording quality;
* motion descriptors that explain class separation;
* offline prediction quality versus live-evaluation logs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv.gesture_features import (  # noqa: E402
    FEATURE_DYNAMIC_STATS,
    GestureSequence,
    build_feature_matrix,
    class_counts,
    infer_target_dim,
    load_gesture_sequences,
    sequence_displacement,
    sequence_motion_energy,
    trajectory_features,
)

DESCRIPTOR_NAMES = [
    "motion_energy",
    "displacement",
    "dx",
    "dy",
    "abs_dx",
    "abs_dy",
    "path_length",
    "straightness",
    "speed",
    "direction_cos",
    "direction_sin",
    "vertical_ratio",
    "horizontal_ratio",
]


@dataclass(frozen=True)
class SampleDescriptor:
    label: str
    path: str
    frames: int
    raw_dim: int
    motion_energy: float
    displacement: float
    dx: float
    dy: float
    abs_dx: float
    abs_dy: float
    path_length: float
    straightness: float
    speed: float
    direction_cos: float
    direction_sin: float
    vertical_ratio: float
    horizontal_ratio: float
    expected_axis: str
    expected_delta: float | None
    axis_alignment: float | None
    direction_ok: bool | None
    quality_warnings: list[str]


@dataclass(frozen=True)
class ClassDescriptorSummary:
    label: str
    sample_count: int
    frames_median: float
    raw_dims: list[int]
    dx_median: float
    dy_median: float
    path_length_median: float
    straightness_median: float
    speed_median: float
    motion_energy_median: float
    direction_ok_rate: float | None
    warning_counts: dict[str, int]


@dataclass(frozen=True)
class FeatureRelevance:
    name: str
    mutual_info: float
    interpretation: str


@dataclass(frozen=True)
class DescriptorGap:
    name: str
    correct_mean: float | None
    wrong_mean: float | None
    effect_size: float | None
    interpretation: str


@dataclass(frozen=True)
class PredictionRecord:
    label: str
    path: str
    predicted: str
    confidence: float
    correct: bool


@dataclass(frozen=True)
class PredictionSummary:
    model: str
    feature_mode: str
    sample_count: int
    class_count: int
    labels: list[str]
    folds: int
    accuracy: float | None
    macro_f1: float | None
    confusion_matrix: list[list[int]]
    predictions: list[PredictionRecord]
    descriptor_gaps: list[DescriptorGap]


@dataclass(frozen=True)
class LiveSummary:
    source: str
    run_count: int
    attempt_count: int
    correct: int
    wrong: int
    missed: int
    accuracy: float | None
    wrong_labels: dict[str, int]
    by_expected: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class DynamicDataAnalysis:
    generated_at: str
    data_root: str
    include_prefix: str
    sample_count: int
    class_count: int
    class_counts: dict[str, int]
    raw_dims: list[int]
    target_dim: int | None
    samples: list[SampleDescriptor]
    classes: list[ClassDescriptorSummary]
    feature_relevance: list[FeatureRelevance]
    prediction: PredictionSummary | None
    live: LiveSummary | None
    findings: list[str]
    preprocessing_decisions: list[str]


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _expected_direction(label: str) -> tuple[str, float, str] | None:
    clean = label.lower()
    if "left" in clean:
        return ("x", -1.0, "left")
    if "right" in clean:
        return ("x", 1.0, "right")
    if "up" in clean:
        return ("y", -1.0, "up")
    if "down" in clean:
        return ("y", 1.0, "down")
    return None


def _descriptor_values(descriptor: SampleDescriptor) -> list[float]:
    return [float(getattr(descriptor, name)) for name in DESCRIPTOR_NAMES]


def describe_sample(
    record: GestureSequence,
    *,
    data_root: Path,
    direction_threshold: float,
    min_motion_energy: float,
    min_path_length: float,
    straightness_threshold: float,
    axis_alignment_threshold: float,
) -> SampleDescriptor:
    dx, dy, abs_dx, abs_dy, path_length, direction_cos, direction_sin = [
        float(value) for value in trajectory_features(record.sequence)
    ]
    global_displacement = float(np.hypot(dx, dy))
    straightness = global_displacement / path_length if path_length > 1e-9 else 0.0
    speed = path_length / max(1, int(record.frames) - 1)
    denominator = abs_dx + abs_dy + 1e-9
    vertical_ratio = abs_dy / denominator
    horizontal_ratio = abs_dx / denominator
    motion_energy = sequence_motion_energy(record.sequence)
    displacement = sequence_displacement(record.sequence)

    expected = _expected_direction(record.label)
    expected_axis = ""
    expected_delta: float | None = None
    axis_alignment: float | None = None
    direction_ok: bool | None = None
    warnings: list[str] = []
    if expected is not None:
        axis, sign, token = expected
        expected_axis = axis
        expected_delta = dx if axis == "x" else dy
        axis_alignment = horizontal_ratio if axis == "x" else vertical_ratio
        direction_ok = (expected_delta * sign) >= direction_threshold
        if not direction_ok:
            warnings.append(f"expected_{token}")
        if axis_alignment < axis_alignment_threshold:
            warnings.append("axis_drift")

    if motion_energy < min_motion_energy:
        warnings.append("low_pose_motion_energy")
    if path_length < min_path_length:
        warnings.append("low_global_path")
    if straightness < straightness_threshold:
        warnings.append("low_straightness")

    return SampleDescriptor(
        label=record.label,
        path=_relative(record.path, data_root),
        frames=record.frames,
        raw_dim=record.feature_dim,
        motion_energy=_round(motion_energy) or 0.0,
        displacement=_round(displacement) or 0.0,
        dx=_round(dx) or 0.0,
        dy=_round(dy) or 0.0,
        abs_dx=_round(abs_dx) or 0.0,
        abs_dy=_round(abs_dy) or 0.0,
        path_length=_round(path_length) or 0.0,
        straightness=_round(straightness) or 0.0,
        speed=_round(speed) or 0.0,
        direction_cos=_round(direction_cos) or 0.0,
        direction_sin=_round(direction_sin) or 0.0,
        vertical_ratio=_round(vertical_ratio) or 0.0,
        horizontal_ratio=_round(horizontal_ratio) or 0.0,
        expected_axis=expected_axis,
        expected_delta=_round(expected_delta),
        axis_alignment=_round(axis_alignment),
        direction_ok=direction_ok,
        quality_warnings=warnings,
    )


def summarize_classes(descriptors: Iterable[SampleDescriptor]) -> list[ClassDescriptorSummary]:
    grouped: dict[str, list[SampleDescriptor]] = {}
    for descriptor in descriptors:
        grouped.setdefault(descriptor.label, []).append(descriptor)

    summaries: list[ClassDescriptorSummary] = []
    for label in sorted(grouped):
        items = grouped[label]
        direction_values = [
            item.direction_ok for item in items if item.direction_ok is not None
        ]
        warnings = Counter(
            warning for item in items for warning in item.quality_warnings
        )
        summaries.append(
            ClassDescriptorSummary(
                label=label,
                sample_count=len(items),
                frames_median=_round(median(item.frames for item in items)) or 0.0,
                raw_dims=sorted({item.raw_dim for item in items}),
                dx_median=_round(median(item.dx for item in items)) or 0.0,
                dy_median=_round(median(item.dy for item in items)) or 0.0,
                path_length_median=_round(median(item.path_length for item in items)) or 0.0,
                straightness_median=_round(median(item.straightness for item in items)) or 0.0,
                speed_median=_round(median(item.speed for item in items)) or 0.0,
                motion_energy_median=_round(median(item.motion_energy for item in items)) or 0.0,
                direction_ok_rate=(
                    _round(sum(1 for value in direction_values if value) / len(direction_values))
                    if direction_values
                    else None
                ),
                warning_counts=dict(sorted(warnings.items())),
            )
        )
    return summaries


def _feature_interpretation(name: str) -> str:
    mapping = {
        "dy": "вертикальное смещение от старта к финишу; разделяет up/down",
        "dx": "горизонтальное смещение от старта к финишу; разделяет left/right",
        "direction_sin": "нормализованное вертикальное направление, устойчивое к амплитуде",
        "direction_cos": "нормализованное горизонтальное направление, устойчивое к амплитуде",
        "vertical_ratio": "доля вертикального движения относительно горизонтального",
        "horizontal_ratio": "доля горизонтального движения относительно вертикального",
        "straightness": "чистый ли это штрих или траектория с петлей/возвратом",
        "motion_energy": "среднее покадровое движение landmarks",
        "path_length": "суммарный путь global wrist",
        "speed": "path length, нормированный на число кадров",
        "abs_dx": "горизонтальная амплитуда",
        "abs_dy": "вертикальная амплитуда",
        "displacement": "start/end distance на уровне pose-признаков",
    }
    return mapping.get(name, "числовой motion descriptor")


def rank_feature_relevance(
    descriptors: list[SampleDescriptor],
    labels: list[str],
    *,
    random_state: int,
) -> list[FeatureRelevance]:
    if len(labels) < 2 or len(descriptors) < 3:
        return []
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    X = np.asarray([_descriptor_values(item) for item in descriptors], dtype=np.float32)
    y = np.asarray([label_to_idx[item.label] for item in descriptors], dtype=np.int64)
    try:
        scores = mutual_info_classif(X, y, random_state=random_state)
    except Exception:
        return []
    rows = [
        FeatureRelevance(
            name=name,
            mutual_info=_round(float(score)) or 0.0,
            interpretation=_feature_interpretation(name),
        )
        for name, score in zip(DESCRIPTOR_NAMES, scores)
    ]
    return sorted(rows, key=lambda item: item.mutual_info, reverse=True)


def _descriptor_gaps(
    descriptors: list[SampleDescriptor],
    predictions: list[PredictionRecord],
) -> list[DescriptorGap]:
    if not predictions or not any(not item.correct for item in predictions):
        return []
    by_path = {item.path: item for item in descriptors}
    correct = [by_path[item.path] for item in predictions if item.correct and item.path in by_path]
    wrong = [by_path[item.path] for item in predictions if not item.correct and item.path in by_path]
    if not correct or not wrong:
        return []

    rows: list[DescriptorGap] = []
    for name in DESCRIPTOR_NAMES:
        correct_values = np.asarray([float(getattr(item, name)) for item in correct], dtype=np.float32)
        wrong_values = np.asarray([float(getattr(item, name)) for item in wrong], dtype=np.float32)
        correct_mean = float(correct_values.mean())
        wrong_mean = float(wrong_values.mean())
        pooled = float(np.sqrt((correct_values.var() + wrong_values.var()) / 2.0))
        effect = None if pooled <= 1e-9 else (correct_mean - wrong_mean) / pooled
        rows.append(
            DescriptorGap(
                name=name,
                correct_mean=_round(correct_mean),
                wrong_mean=_round(wrong_mean),
                effect_size=_round(effect),
                interpretation=_feature_interpretation(name),
            )
        )
    return sorted(
        rows,
        key=lambda item: abs(item.effect_size or 0.0),
        reverse=True,
    )


def evaluate_prediction(
    records: list[GestureSequence],
    descriptors: list[SampleDescriptor],
    *,
    data_root: Path,
    min_samples_per_class: int,
    max_folds: int,
    random_state: int,
) -> PredictionSummary | None:
    counts = class_counts(records)
    allowed = {
        label for label, count in counts.items() if count >= int(min_samples_per_class)
    }
    selected = [record for record in records if record.label in allowed]
    if len(allowed) < 2 or not selected:
        return None

    min_count = min(counts[label] for label in allowed)
    folds = max(2, min(int(max_folds), int(min_count)))
    target_dim = infer_target_dim(record.sequence for record in selected)
    X, y, labels = build_feature_matrix(
        selected,
        mode=FEATURE_DYNAMIC_STATS,
        target_dim=target_dim,
    )

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    prediction_records: list[PredictionRecord] = []
    for train_idx, test_idx in cv.split(X, y):
        neighbors = max(1, min(5, len(train_idx)))
        estimator = KNeighborsClassifier(
            n_neighbors=neighbors,
            metric="euclidean",
            weights="distance",
        )
        estimator.fit(X[train_idx], y[train_idx])
        pred = estimator.predict(X[test_idx])
        proba = estimator.predict_proba(X[test_idx])
        confidence = np.max(proba, axis=1)
        y_true_parts.append(y[test_idx])
        y_pred_parts.append(np.asarray(pred, dtype=np.int64))
        for sample_idx, pred_idx, conf in zip(test_idx, pred, confidence):
            record = selected[int(sample_idx)]
            expected = labels[int(y[int(sample_idx)])]
            predicted = labels[int(pred_idx)]
            prediction_records.append(
                PredictionRecord(
                    label=expected,
                    path=_relative(record.path, data_root),
                    predicted=predicted,
                    confidence=_round(float(conf)) or 0.0,
                    correct=expected == predicted,
                )
            )

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    return PredictionSummary(
        model="knn(distance)",
        feature_mode=FEATURE_DYNAMIC_STATS,
        sample_count=len(selected),
        class_count=len(labels),
        labels=labels,
        folds=folds,
        accuracy=_round(accuracy_score(y_true, y_pred)),
        macro_f1=_round(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        confusion_matrix=cm.astype(int).tolist(),
        predictions=prediction_records,
        descriptor_gaps=_descriptor_gaps(descriptors, prediction_records),
    )


def load_live_summary(path: Path) -> LiveSummary | None:
    if not path.exists():
        return None
    runs: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event_type") in {"run_completed", "run_stopped"}:
            runs.append(row)
    if not runs:
        return None

    total = correct = wrong = missed = 0
    wrong_labels: Counter[str] = Counter()
    by_expected: dict[str, dict[str, Any]] = {}
    for run in runs:
        expected = str(run.get("expected_label") or run.get("expected") or "")
        attempts = run.get("attempts") if isinstance(run.get("attempts"), list) else []
        total += int(run.get("total") or len(attempts) or 0)
        correct += int(run.get("correct") or 0)
        wrong += int(run.get("wrong") or 0)
        missed += int(run.get("missed") or 0)
        bucket = by_expected.setdefault(
            expected,
            {"runs": 0, "attempts": 0, "correct": 0, "wrong": 0, "missed": 0},
        )
        bucket["runs"] += 1
        bucket["attempts"] += int(run.get("total") or len(attempts) or 0)
        bucket["correct"] += int(run.get("correct") or 0)
        bucket["wrong"] += int(run.get("wrong") or 0)
        bucket["missed"] += int(run.get("missed") or 0)
        for attempt in attempts:
            if attempt.get("result") == "wrong":
                wrong_labels[str(attempt.get("predicted") or "unknown")] += 1

    accuracy = None if total <= 0 else correct / total
    for bucket in by_expected.values():
        attempts = int(bucket["attempts"])
        bucket["accuracy"] = None if attempts <= 0 else round(bucket["correct"] / attempts, 4)
    return LiveSummary(
        source=str(path),
        run_count=len(runs),
        attempt_count=total,
        correct=correct,
        wrong=wrong,
        missed=missed,
        accuracy=_round(accuracy),
        wrong_labels=dict(sorted(wrong_labels.items())),
        by_expected=dict(sorted(by_expected.items())),
    )


def _build_findings(
    descriptors: list[SampleDescriptor],
    classes: list[ClassDescriptorSummary],
    feature_relevance: list[FeatureRelevance],
    prediction: PredictionSummary | None,
    live: LiveSummary | None,
) -> list[str]:
    findings: list[str] = []
    counts = {item.label: item.sample_count for item in classes}
    low_sample = [label for label, count in counts.items() if count < 20]
    if low_sample:
        findings.append(
            "Недостаточно dynamic-сэмплов: "
            + ", ".join(f"{label}={counts[label]}" for label in low_sample)
            + ". Эти классы нужно дозаписать перед сильными выводами о модели."
        )
    noisy = [
        item.label
        for item in classes
        if item.warning_counts.get("low_straightness", 0)
        or item.warning_counts.get("expected_up", 0)
        or item.warning_counts.get("expected_down", 0)
    ]
    if noisy:
        findings.append(
            "В сохраненных классах есть шум траекторий или конфликт направления: "
            + ", ".join(noisy)
            + ". Чистый протокол записи сейчас важнее очередной смены модели."
        )
    if feature_relevance:
        top = feature_relevance[0]
        findings.append(
            f"Самый информативный compact descriptor: `{top.name}` "
            f"(MI={top.mutual_info:.4f}): {top.interpretation}."
        )
    if prediction and prediction.accuracy is not None:
        findings.append(
            "Offline KNN на сохраненных dynamic-сэмплах дает accuracy "
            f"{prediction.accuracy:.4f} и macro F1 {prediction.macro_f1:.4f}; "
            "live-ошибки поэтому указывают на distribution shift между записью и показом."
        )
    if live and live.wrong_labels:
        wrong = ", ".join(f"{label}:{count}" for label, count in live.wrong_labels.items())
        findings.append(f"Live-ошибки концентрируются в predicted labels: {wrong}.")
    if not descriptors:
        findings.append("Dynamic-сэмплы не попали в анализ.")
    return findings


def _preprocessing_decisions() -> list[str]:
    return [
        "Держать отдельный dynamic dataset/model для жестов движения, пока live-метрики не докажут стабильность единой модели.",
        "Использовать канон записи на уровне класса: фиксированный старт, чистое направление, видимый финиш, сброс между сэмплами.",
        "На сохранении применять quality gates: знак ожидаемого направления, minimum global path, axis alignment, straightness и pose motion energy.",
        "Балансировать dynamic-классы минимум до 20 сэмплов на класс перед сравнением моделей; основной offline-показатель - macro F1.",
        "Высокий offline CV при слабом live считать distribution shift; каждую candidate-модель подтверждать live evaluation с expected_label.",
        "Production preprocessing выделяет active motion segment и resample до 36 кадров, чтобы медленные/быстрые жесты были сравнимы.",
    ]


def analyze_dynamic_data(
    data_root: Path,
    *,
    include_prefix: str = "swipe_",
    live_log: Path | None = None,
    direction_threshold: float = 0.05,
    min_motion_energy: float = 0.015,
    min_path_length: float = 0.25,
    straightness_threshold: float = 0.55,
    axis_alignment_threshold: float = 0.75,
    min_samples_per_class: int = 2,
    max_folds: int = 5,
    random_state: int = 42,
) -> DynamicDataAnalysis:
    records = [
        record
        for record in load_gesture_sequences(data_root)
        if not include_prefix or record.label.startswith(include_prefix)
    ]
    descriptors = [
        describe_sample(
            record,
            data_root=data_root,
            direction_threshold=direction_threshold,
            min_motion_energy=min_motion_energy,
            min_path_length=min_path_length,
            straightness_threshold=straightness_threshold,
            axis_alignment_threshold=axis_alignment_threshold,
        )
        for record in records
    ]
    counts = class_counts(records)
    labels = sorted(counts)
    raw_dims = sorted({record.feature_dim for record in records})
    target_dim = infer_target_dim(record.sequence for record in records) if records else None
    classes = summarize_classes(descriptors)
    feature_relevance = rank_feature_relevance(
        descriptors,
        labels,
        random_state=random_state,
    )
    prediction = evaluate_prediction(
        records,
        descriptors,
        data_root=data_root,
        min_samples_per_class=min_samples_per_class,
        max_folds=max_folds,
        random_state=random_state,
    )
    live = load_live_summary(live_log) if live_log is not None else None
    findings = _build_findings(
        descriptors=descriptors,
        classes=classes,
        feature_relevance=feature_relevance,
        prediction=prediction,
        live=live,
    )
    return DynamicDataAnalysis(
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        data_root=str(data_root),
        include_prefix=include_prefix,
        sample_count=len(records),
        class_count=len(labels),
        class_counts=counts,
        raw_dims=raw_dims,
        target_dim=target_dim,
        samples=descriptors,
        classes=classes,
        feature_relevance=feature_relevance,
        prediction=prediction,
        live=live,
        findings=findings,
        preprocessing_decisions=_preprocessing_decisions(),
    )


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def _warning_text(warnings: dict[str, int] | list[str]) -> str:
    if isinstance(warnings, dict):
        return ", ".join(f"{key}:{value}" for key, value in warnings.items()) or "-"
    return ", ".join(warnings) or "-"


def _confusion_matrix_markdown(summary: PredictionSummary) -> str:
    labels = summary.labels
    rows = [
        "| true \\ pred | " + " | ".join(f"`{label}`" for label in labels) + " |",
        "|---|" + "|".join("---:" for _ in labels) + "|",
    ]
    for label, row in zip(labels, summary.confusion_matrix):
        rows.append("| `" + label + "` | " + " | ".join(str(int(value)) for value in row) + " |")
    return "\n".join(rows)


def build_markdown_report(report: DynamicDataAnalysis) -> str:
    lines: list[str] = [
        "# Dynamic Gesture Data Analysis",
        "",
        f"Generated: `{report.generated_at}`",
        "",
        "## Краткий вывод",
        "",
    ]
    lines.extend(f"- {finding}" for finding in report.findings)
    lines.extend(
        [
            "",
            "## Объем данных",
            "",
            "| Метрика | Значение |",
            "|---|---:|",
            f"| Data root | `{report.data_root}` |",
            f"| Include prefix | `{report.include_prefix}` |",
            f"| Классы | {report.class_count} |",
            f"| Сэмплы | {report.sample_count} |",
            f"| Raw dimensions | {', '.join(str(dim) for dim in report.raw_dims) or 'n/a'} |",
            f"| Target dim | {_fmt(report.target_dim)} |",
            "",
            "| Класс | Сэмплы | Frames median | Raw dims | dx median | dy median | Path median | Straightness | Speed | Direction OK | Warnings |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in report.classes:
        lines.append(
            f"| `{item.label}` | {item.sample_count} | {_fmt(item.frames_median)} | "
            f"{', '.join(str(dim) for dim in item.raw_dims)} | {_fmt(item.dx_median)} | "
            f"{_fmt(item.dy_median)} | {_fmt(item.path_length_median)} | "
            f"{_fmt(item.straightness_median)} | {_fmt(item.speed_median)} | "
            f"{_fmt(item.direction_ok_rate)} | {_warning_text(item.warning_counts)} |"
        )

    lines.extend(
        [
            "",
            "## Ключевые признаки",
            "",
            "| Rank | Descriptor | Mutual information | Интерпретация |",
            "|---:|---|---:|---|",
        ]
    )
    for idx, item in enumerate(report.feature_relevance[:8], start=1):
        lines.append(
            f"| {idx} | `{item.name}` | {_fmt(item.mutual_info)} | {item.interpretation} |"
        )

    if report.prediction is not None:
        prediction = report.prediction
        wrong_predictions = [item for item in prediction.predictions if not item.correct]
        lines.extend(
            [
                "",
                "## Offline-валидация",
                "",
                "| Метрика | Значение |",
                "|---|---:|",
                f"| Model | `{prediction.model}` |",
                f"| Feature mode | `{prediction.feature_mode}` |",
                f"| CV folds | {prediction.folds} |",
                f"| Сэмплы | {prediction.sample_count} |",
                f"| Accuracy | {_fmt(prediction.accuracy)} |",
                f"| Macro F1 | {_fmt(prediction.macro_f1)} |",
                f"| Ошибочных сэмплов | {len(wrong_predictions)} |",
                "",
                "### Confusion Matrix",
                "",
                _confusion_matrix_markdown(prediction),
            ]
        )
        if wrong_predictions:
            lines.extend(
                [
                    "",
                    "### Ошибочные sample",
                    "",
                    "| Expected | Predicted | Confidence | Sample |",
                    "|---|---|---:|---|",
                ]
            )
            for item in wrong_predictions[:12]:
                lines.append(
                    f"| `{item.label}` | `{item.predicted}` | {_fmt(item.confidence)} | `{item.path}` |"
                )
        if prediction.descriptor_gaps:
            lines.extend(
                [
                    "",
                    "### Correct vs Wrong Descriptor Gap",
                    "",
                    "| Descriptor | Correct mean | Wrong mean | Effect size | Интерпретация |",
                    "|---|---:|---:|---:|---|",
                ]
            )
            for item in prediction.descriptor_gaps[:8]:
                lines.append(
                    f"| `{item.name}` | {_fmt(item.correct_mean)} | {_fmt(item.wrong_mean)} | "
                    f"{_fmt(item.effect_size)} | {item.interpretation} |"
                )
    if report.live is not None:
        live = report.live
        lines.extend(
            [
                "",
                "## Live-gap",
                "",
                "| Метрика | Значение |",
                "|---|---:|",
                f"| Source | `{live.source}` |",
                f"| Runs | {live.run_count} |",
                f"| Attempts | {live.attempt_count} |",
                f"| Correct | {live.correct} |",
                f"| Wrong | {live.wrong} |",
                f"| Missed | {live.missed} |",
                f"| Accuracy | {_fmt(live.accuracy)} |",
                f"| Wrong labels | {_warning_text(live.wrong_labels)} |",
                "",
                "| Expected | Runs | Attempts | Correct | Wrong | Missed | Accuracy |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for expected, item in live.by_expected.items():
            lines.append(
                f"| `{expected}` | {item['runs']} | {item['attempts']} | "
                f"{item['correct']} | {item['wrong']} | {item['missed']} | "
                f"{_fmt(item['accuracy'])} |"
            )

    problematic = [item for item in report.samples if item.quality_warnings]
    if problematic:
        lines.extend(
            [
                "",
                "## Проблемные сохраненные sample",
                "",
                "| Label | Sample | dx | dy | Path | Straightness | Energy | Warnings |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in sorted(
            problematic,
            key=lambda sample: (sample.label, len(sample.quality_warnings), sample.path),
            reverse=True,
        )[:24]:
            lines.append(
                f"| `{item.label}` | `{item.path}` | {_fmt(item.dx)} | {_fmt(item.dy)} | "
                f"{_fmt(item.path_length)} | {_fmt(item.straightness)} | "
                f"{_fmt(item.motion_energy)} | {_warning_text(item.quality_warnings)} |"
            )

    lines.extend(
        [
            "",
            "## Решения по предобработке",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report.preprocessing_decisions)
    lines.extend(
        [
            "",
            "## Интерпретация для JMLC",
            "",
            "- Анализ документирует понимание данных, критерии предобработки, протокол валидации и live/offline mismatch.",
            "- Active-segment preprocessing реализован; следующая измеримая проверка - live-evaluation event-based inference против прежнего sliding-window baseline.",
            "",
            "## Воспроизведение",
            "",
            "```bash",
            "python -m scripts.dynamic_data_analysis --out-md docs/experiments/dynamic_data_analysis.md --out-json docs/experiments/dynamic_data_analysis.json",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/gestures"))
    parser.add_argument("--include-prefix", default="swipe_")
    parser.add_argument(
        "--live-log",
        type=Path,
        default=Path.home() / ".dplm" / "logs" / "live_evaluation.jsonl",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/dynamic_data_analysis.md"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("docs/experiments/dynamic_data_analysis.json"),
    )
    parser.add_argument("--min-samples-per-class", type=int, default=2)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_dynamic_data(
        args.data_root,
        include_prefix=str(args.include_prefix),
        live_log=args.live_log,
        min_samples_per_class=int(args.min_samples_per_class),
        max_folds=int(args.max_folds),
        random_state=int(args.random_state),
    )
    markdown = build_markdown_report(report)
    print(markdown)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown, encoding="utf-8")
    args.out_json.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
