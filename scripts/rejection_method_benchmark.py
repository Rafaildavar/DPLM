"""Benchmark rejection methods for GestureBind open-set recognition.

The report answers a practical product question: which strategy rejects random or
partial gestures without breaking real gestures?
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
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import (
    KNeighborsClassifier,
    LocalOutlierFactor,
    NeighborhoodComponentsAnalysis,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import OneClassSVM, SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.gesture_taxonomy import (  # noqa: E402
    GESTURE_TYPE_DYNAMIC,
    GESTURE_TYPE_NEGATIVE,
    GESTURE_TYPE_QUASI_STATIC,
    GESTURE_TYPE_STATIC,
    GestureTaxonomy,
    load_gesture_taxonomy,
)
from cv.gesture_features import (  # noqa: E402
    FEATURE_DYNAMIC_STATS,
    FEATURE_STATIC_MEAN,
    build_feature_matrix,
    class_counts,
    infer_target_dim,
    load_gesture_sequences,
    GestureSequence,
    sequence_to_matrix,
)
from cv.gesture_dataset_files import augmented_sample_paths, gesture_sample_paths  # noqa: E402

DEFAULT_METHODS = (
    "negative_classes",
    "confidence_threshold",
    "open_set_policy",
    "one_vs_rest_logreg",
    "one_class_svm",
    "isolation_forest",
    "local_outlier_factor",
    "metric_nca_centroid",
    "mlp_negative_classes",
)


@dataclass(frozen=True)
class RejectionDatasetInfo:
    data_root: str
    scope: str
    feature_mode: str
    candidate_model: str
    target_dim: int
    sample_count: int
    class_count: int
    positive_labels: list[str]
    negative_labels: list[str]
    class_counts: dict[str, int]
    folds: int
    include_augmented: bool = False


@dataclass(frozen=True)
class NegativeLabelMetrics:
    label: str
    total: int
    rejected: int
    false_positive: int
    reject_rate: float
    false_positive_rate: float
    false_positive_predictions: dict[str, int]


@dataclass(frozen=True)
class RejectionMethodResult:
    method: str
    status: str
    overall_success: float
    positive_recall: float
    positive_reject_rate: float
    negative_reject_rate: float
    negative_false_positive_rate: float
    accepted_accuracy: float
    coverage: float
    total: int
    positive_total: int
    negative_total: int
    positive_correct: int
    positive_rejected: int
    negative_rejected: int
    negative_false_positive: int
    wrong_accept: int
    reject_reasons: dict[str, int]
    false_positive_predictions: dict[str, int]
    negative_label_metrics: list[NegativeLabelMetrics]
    error: str = ""


@dataclass(frozen=True)
class RejectionBenchmarkReport:
    generated_at: float
    dataset: RejectionDatasetInfo
    methods: list[RejectionMethodResult]
    best_method: str
    recommendation: str


@dataclass(frozen=True)
class Decision:
    label: str
    reason: str


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _scope_types(scope: str) -> tuple[str, ...]:
    clean = str(scope or "static").strip().lower()
    if clean == "static":
        return (GESTURE_TYPE_STATIC, GESTURE_TYPE_QUASI_STATIC, GESTURE_TYPE_NEGATIVE)
    if clean == "dynamic":
        return (GESTURE_TYPE_DYNAMIC, GESTURE_TYPE_NEGATIVE)
    if clean == "all":
        return (
            GESTURE_TYPE_STATIC,
            GESTURE_TYPE_QUASI_STATIC,
            GESTURE_TYPE_DYNAMIC,
            GESTURE_TYPE_NEGATIVE,
        )
    return tuple(item.strip() for item in clean.split(",") if item.strip())


def _is_negative_label(taxonomy: GestureTaxonomy, label: str) -> bool:
    return taxonomy.gesture_type_for_label(label) == GESTURE_TYPE_NEGATIVE


def _method_list(methods: Iterable[str] | str | None) -> list[str]:
    if methods is None:
        return list(DEFAULT_METHODS)
    if isinstance(methods, str):
        raw = methods.split(",")
    else:
        raw = list(methods)
    selected = [str(item).strip() for item in raw if str(item).strip()]
    return selected or list(DEFAULT_METHODS)


def _gislr_augmented_sample_paths(label_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for sample_path in augmented_sample_paths(label_dir):
        metadata_path = sample_path.with_suffix(".meta.json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        transform = str(metadata.get("transform") or "")
        transform_metadata = metadata.get("transform_metadata")
        policy = (
            str(transform_metadata.get("policy") or "")
            if isinstance(transform_metadata, dict)
            else ""
        )
        if transform == "gislr_landmark_v1" or policy == "gislr_landmark_v1":
            paths.append(sample_path)
    return paths


def _load_records_for_benchmark(
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


def _knn_neighbors(y_train: np.ndarray) -> int:
    counts = {label: int(np.count_nonzero(y_train == label)) for label in set(y_train)}
    min_count = min(counts.values()) if counts else 1
    return max(1, min(5, min_count))


def _fit_knn(X_train: np.ndarray, y_train: np.ndarray) -> KNeighborsClassifier:
    clf = KNeighborsClassifier(
        n_neighbors=_knn_neighbors(y_train),
        metric="euclidean",
        weights="distance",
    )
    clf.fit(X_train, y_train)
    return clf


def _fit_candidate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    candidate_model: str,
    random_state: int,
) -> Any:
    model = str(candidate_model or "knn").strip().lower()
    if model == "knn":
        return _fit_knn(X_train, y_train)
    if model == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=250,
            random_state=int(random_state),
            class_weight="balanced",
            n_jobs=1,
        )
        clf.fit(X_train, y_train)
        return clf
    if model == "rf":
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=int(random_state),
            class_weight="balanced",
            n_jobs=1,
        )
        clf.fit(X_train, y_train)
        return clf
    if model == "logreg":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=int(random_state),
            ),
        )
        clf.fit(X_train, y_train)
        return clf
    if model == "svm":
        clf = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=2.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=int(random_state),
            ),
        )
        clf.fit(X_train, y_train)
        return clf
    raise ValueError(f"unsupported candidate model: {candidate_model}")


def _ranked_probabilities(model: Any, sample: np.ndarray) -> list[tuple[float, str]]:
    if not hasattr(model, "predict_proba"):
        predicted = str(model.predict(sample.reshape(1, -1))[0])
        return [(1.0, predicted)]
    probabilities = np.asarray(model.predict_proba(sample.reshape(1, -1))[0], dtype=float)
    classes = [str(label) for label in getattr(model, "classes_", range(len(probabilities)))]
    ranked = [
        (float(probability), classes[index])
        for index, probability in enumerate(probabilities)
    ]
    return sorted(ranked, key=lambda item: item[0], reverse=True)


def _best_negative(
    ranked: list[tuple[float, str]],
    negative_labels: set[str],
) -> tuple[str, float]:
    best_label = ""
    best_confidence = 0.0
    for confidence, label in ranked:
        if label in negative_labels and confidence > best_confidence:
            best_label = label
            best_confidence = float(confidence)
    return best_label, best_confidence


def _fit_prototypes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    labels: Iterable[str],
) -> dict[str, dict[str, Any]]:
    prototypes: dict[str, dict[str, Any]] = {}
    feature_dim = int(X_train.shape[1]) if X_train.ndim == 2 else 1
    radius_floor = 0.015 * float(np.sqrt(max(1, feature_dim)))
    for label in labels:
        rows = X_train[y_train == label]
        if len(rows) == 0:
            continue
        centroid = rows.mean(axis=0)
        distances = np.linalg.norm(rows - centroid, axis=1)
        mean = float(np.mean(distances)) if len(distances) else 0.0
        std = float(np.std(distances)) if len(distances) else 0.0
        p95 = float(np.percentile(distances, 95)) if len(distances) else 0.0
        prototypes[str(label)] = {
            "centroid": centroid,
            "radius": max(p95, mean + 2.0 * std, radius_floor),
        }
    return prototypes


def _negative_classes_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    negative_labels: set[str],
    candidate_model: str,
    random_state: int,
    **_kwargs: Any,
) -> list[Decision]:
    clf = _fit_candidate_model(
        X_train,
        y_train,
        candidate_model=candidate_model,
        random_state=random_state,
    )
    decisions: list[Decision] = []
    for sample in X_test:
        label = str(clf.predict(sample.reshape(1, -1))[0])
        if label in negative_labels:
            decisions.append(Decision("", "negative_class"))
        else:
            decisions.append(Decision(label, "accepted"))
    return decisions


def _confidence_threshold_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    negative_labels: set[str],
    confidence_threshold: float,
    candidate_model: str,
    random_state: int,
    **_kwargs: Any,
) -> list[Decision]:
    clf = _fit_candidate_model(
        X_train,
        y_train,
        candidate_model=candidate_model,
        random_state=random_state,
    )
    decisions: list[Decision] = []
    for sample in X_test:
        ranked = _ranked_probabilities(clf, sample)
        confidence, label = ranked[0]
        if label in negative_labels:
            decisions.append(Decision("", "negative_class"))
        elif confidence < confidence_threshold:
            decisions.append(Decision("", "low_confidence"))
        else:
            decisions.append(Decision(label, "accepted"))
    return decisions


def _open_set_policy_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    positive_labels: set[str],
    negative_labels: set[str],
    negative_confidence_threshold: float,
    min_margin: float,
    distance_multiplier: float,
    candidate_model: str,
    random_state: int,
    **_kwargs: Any,
) -> list[Decision]:
    clf = _fit_candidate_model(
        X_train,
        y_train,
        candidate_model=candidate_model,
        random_state=random_state,
    )
    prototypes = _fit_prototypes(X_train, y_train, positive_labels)
    decisions: list[Decision] = []
    for sample in X_test:
        ranked = _ranked_probabilities(clf, sample)
        confidence, label = ranked[0]
        top2_confidence = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = float(confidence - top2_confidence)
        _negative_label, negative_confidence = _best_negative(ranked, negative_labels)
        if label in negative_labels or negative_confidence >= negative_confidence_threshold:
            decisions.append(Decision("", "negative_class"))
            continue
        if margin < min_margin:
            decisions.append(Decision("", "low_margin"))
            continue
        prototype = prototypes.get(label)
        if prototype is not None:
            distance = float(np.linalg.norm(sample - prototype["centroid"]))
            threshold = float(prototype["radius"]) * float(distance_multiplier)
            if threshold > 0.0 and distance > threshold:
                decisions.append(Decision("", "far_from_prototype"))
                continue
        decisions.append(Decision(label, "accepted"))
    return decisions


def _one_vs_rest_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    positive_labels: set[str],
    negative_labels: set[str],
    verifier_threshold: float,
    candidate_model: str,
    random_state: int,
    **_kwargs: Any,
) -> list[Decision]:
    fitted_candidate = _fit_candidate_model(
        X_train,
        y_train,
        candidate_model=candidate_model,
        random_state=random_state,
    )
    verifiers: dict[str, Any] = {}
    for label in sorted(positive_labels):
        binary_y = (y_train == label).astype(np.int64)
        if len(np.unique(binary_y)) < 2:
            continue
        verifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            ),
        )
        verifier.fit(X_train, binary_y)
        verifiers[label] = verifier

    decisions: list[Decision] = []
    for sample in X_test:
        candidate = str(fitted_candidate.predict(sample.reshape(1, -1))[0])
        if candidate in negative_labels:
            decisions.append(Decision("", "negative_class"))
            continue
        verifier = verifiers.get(candidate)
        if verifier is None:
            decisions.append(Decision("", "missing_verifier"))
            continue
        probability = float(verifier.predict_proba(sample.reshape(1, -1))[0, 1])
        if probability < verifier_threshold:
            decisions.append(Decision("", "verifier_rejected"))
        else:
            decisions.append(Decision(candidate, "accepted"))
    return decisions


def _outlier_verifier_predictions(
    detector_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    positive_labels: set[str],
    negative_labels: set[str],
    candidate_model: str,
    random_state: int,
    **_kwargs: Any,
) -> list[Decision]:
    fitted_candidate = _fit_candidate_model(
        X_train,
        y_train,
        candidate_model=candidate_model,
        random_state=random_state,
    )
    detectors: dict[str, tuple[StandardScaler, Any]] = {}
    for label in sorted(positive_labels):
        rows = X_train[y_train == label]
        if len(rows) < 3:
            continue
        scaler = StandardScaler()
        scaled_rows = scaler.fit_transform(rows)
        if detector_name == "one_class_svm":
            detector = OneClassSVM(nu=0.15, kernel="rbf", gamma="scale")
        elif detector_name == "isolation_forest":
            detector = IsolationForest(
                n_estimators=200,
                contamination=0.15,
                random_state=random_state,
            )
        elif detector_name == "local_outlier_factor":
            detector = LocalOutlierFactor(
                n_neighbors=max(2, min(20, len(rows) - 1)),
                novelty=True,
                contamination=0.15,
            )
        else:
            raise ValueError(f"unsupported outlier detector: {detector_name}")
        detector.fit(scaled_rows)
        detectors[label] = (scaler, detector)

    decisions: list[Decision] = []
    for sample in X_test:
        candidate = str(fitted_candidate.predict(sample.reshape(1, -1))[0])
        if candidate in negative_labels:
            decisions.append(Decision("", "negative_class"))
            continue
        bundle = detectors.get(candidate)
        if bundle is None:
            decisions.append(Decision("", "missing_outlier_model"))
            continue
        scaler, detector = bundle
        verdict = int(detector.predict(scaler.transform(sample.reshape(1, -1)))[0])
        if verdict < 0:
            decisions.append(Decision("", f"{detector_name}_outlier"))
        else:
            decisions.append(Decision(candidate, "accepted"))
    return decisions


def _metric_nca_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    positive_labels: set[str],
    distance_multiplier: float,
    random_state: int,
    **_kwargs: Any,
) -> list[Decision]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(y_train)
    n_components = max(1, min(8, X_train.shape[1], len(np.unique(encoded_y)) + 2))
    nca = NeighborhoodComponentsAnalysis(
        n_components=n_components,
        random_state=random_state,
        max_iter=250,
        tol=1e-4,
    )
    nca.fit(X_train_scaled, encoded_y)
    Z_train = nca.transform(X_train_scaled)
    prototypes = _fit_prototypes(Z_train, y_train, positive_labels)
    if not prototypes:
        raise RuntimeError("no positive prototypes for NCA metric method")

    decisions: list[Decision] = []
    Z_test = nca.transform(scaler.transform(X_test))
    for sample in Z_test:
        distances = []
        for label, prototype in prototypes.items():
            distance = float(np.linalg.norm(sample - prototype["centroid"]))
            distances.append((distance, label, prototype))
        distance, label, prototype = sorted(distances, key=lambda item: item[0])[0]
        threshold = float(prototype["radius"]) * float(distance_multiplier)
        if threshold > 0.0 and distance > threshold:
            decisions.append(Decision("", "metric_distance_rejected"))
        else:
            decisions.append(Decision(label, "accepted"))
    return decisions


def _mlp_negative_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    negative_labels: set[str],
    confidence_threshold: float,
    random_state: int,
    **_kwargs: Any,
) -> list[Decision]:
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            alpha=0.001,
            max_iter=600,
            random_state=random_state,
            early_stopping=True,
        ),
    )
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    model.fit(X_train, y_encoded)
    decisions: list[Decision] = []
    for sample in X_test:
        probabilities = np.asarray(
            model.predict_proba(sample.reshape(1, -1))[0],
            dtype=float,
        )
        classes = [
            str(label)
            for label in label_encoder.inverse_transform(
                np.asarray(model.classes_, dtype=np.int64)
            )
        ]
        ranked = sorted(
            [
                (float(probability), classes[index])
                for index, probability in enumerate(probabilities)
            ],
            key=lambda item: item[0],
            reverse=True,
        )
        confidence, label = ranked[0]
        if label in negative_labels:
            decisions.append(Decision("", "negative_class"))
        elif confidence < confidence_threshold:
            decisions.append(Decision("", "low_confidence"))
        else:
            decisions.append(Decision(label, "accepted"))
    return decisions


def _predict_method(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    **kwargs: Any,
) -> list[Decision]:
    if method == "negative_classes":
        return _negative_classes_predictions(X_train, y_train, X_test, **kwargs)
    if method == "confidence_threshold":
        return _confidence_threshold_predictions(X_train, y_train, X_test, **kwargs)
    if method == "open_set_policy":
        return _open_set_policy_predictions(X_train, y_train, X_test, **kwargs)
    if method == "one_vs_rest_logreg":
        return _one_vs_rest_predictions(X_train, y_train, X_test, **kwargs)
    if method in {"one_class_svm", "isolation_forest", "local_outlier_factor"}:
        return _outlier_verifier_predictions(
            method,
            X_train,
            y_train,
            X_test,
            **kwargs,
        )
    if method == "metric_nca_centroid":
        return _metric_nca_predictions(X_train, y_train, X_test, **kwargs)
    if method == "mlp_negative_classes":
        return _mlp_negative_predictions(X_train, y_train, X_test, **kwargs)
    raise ValueError(f"unknown rejection method: {method}")


def _summarize_predictions(
    method: str,
    y_true: list[str],
    decisions: list[Decision],
    *,
    positive_labels: set[str],
    negative_labels: set[str],
    error: str = "",
) -> RejectionMethodResult:
    total = len(y_true)
    positive_total = sum(1 for label in y_true if label in positive_labels)
    negative_total = sum(1 for label in y_true if label in negative_labels)
    positive_correct = 0
    positive_rejected = 0
    negative_rejected = 0
    negative_false_positive = 0
    wrong_accept = 0
    accepted = 0
    accepted_correct = 0
    reject_reasons: dict[str, int] = {}
    false_positive_predictions: dict[str, int] = {}
    negative_breakdown: dict[str, dict[str, Any]] = {
        label: {
            "total": 0,
            "rejected": 0,
            "false_positive": 0,
            "false_positive_predictions": {},
        }
        for label in sorted(negative_labels)
    }

    for true_label, decision in zip(y_true, decisions):
        predicted = str(decision.label or "")
        reason = str(decision.reason or "unknown")
        rejected = not predicted
        if rejected:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

        if true_label in negative_labels:
            bucket = negative_breakdown.setdefault(
                true_label,
                {
                    "total": 0,
                    "rejected": 0,
                    "false_positive": 0,
                    "false_positive_predictions": {},
                },
            )
            bucket["total"] = int(bucket["total"]) + 1
            if rejected:
                negative_rejected += 1
                bucket["rejected"] = int(bucket["rejected"]) + 1
            else:
                accepted += 1
                negative_false_positive += 1
                wrong_accept += 1
                bucket["false_positive"] = int(bucket["false_positive"]) + 1
                bucket_predictions = bucket["false_positive_predictions"]
                if isinstance(bucket_predictions, dict):
                    bucket_predictions[predicted] = int(bucket_predictions.get(predicted, 0)) + 1
                false_positive_predictions[predicted] = (
                    false_positive_predictions.get(predicted, 0) + 1
                )
            continue

        if rejected:
            positive_rejected += 1
            continue

        accepted += 1
        if predicted == true_label:
            positive_correct += 1
            accepted_correct += 1
        else:
            wrong_accept += 1

    overall_success = (
        (positive_correct + negative_rejected) / total if total else 0.0
    )
    negative_label_metrics = []
    for label, values in sorted(negative_breakdown.items()):
        label_total = int(values.get("total") or 0)
        label_rejected = int(values.get("rejected") or 0)
        label_false_positive = int(values.get("false_positive") or 0)
        label_predictions = values.get("false_positive_predictions") or {}
        negative_label_metrics.append(
            NegativeLabelMetrics(
                label=label,
                total=label_total,
                rejected=label_rejected,
                false_positive=label_false_positive,
                reject_rate=_round(label_rejected / label_total if label_total else 0.0),
                false_positive_rate=_round(
                    label_false_positive / label_total if label_total else 0.0
                ),
                false_positive_predictions=dict(sorted(label_predictions.items()))
                if isinstance(label_predictions, dict)
                else {},
            )
        )
    return RejectionMethodResult(
        method=method,
        status="failed" if error else "ok",
        overall_success=_round(overall_success),
        positive_recall=_round(positive_correct / positive_total if positive_total else 0.0),
        positive_reject_rate=_round(
            positive_rejected / positive_total if positive_total else 0.0
        ),
        negative_reject_rate=_round(
            negative_rejected / negative_total if negative_total else 0.0
        ),
        negative_false_positive_rate=_round(
            negative_false_positive / negative_total if negative_total else 0.0
        ),
        accepted_accuracy=_round(accepted_correct / accepted if accepted else 0.0),
        coverage=_round(accepted / total if total else 0.0),
        total=total,
        positive_total=positive_total,
        negative_total=negative_total,
        positive_correct=positive_correct,
        positive_rejected=positive_rejected,
        negative_rejected=negative_rejected,
        negative_false_positive=negative_false_positive,
        wrong_accept=wrong_accept,
        reject_reasons=dict(sorted(reject_reasons.items())),
        false_positive_predictions=dict(sorted(false_positive_predictions.items())),
        negative_label_metrics=negative_label_metrics,
        error=error,
    )


def benchmark_rejection_methods(
    *,
    data_root: Path,
    taxonomy_path: Path,
    scope: str = "static",
    feature_mode: str | None = None,
    candidate_model: str = "knn",
    target_dim: int | None = None,
    methods: Iterable[str] | str | None = None,
    include_labels: Iterable[str] | None = None,
    include_augmented: bool = False,
    lowercase_labels: bool = False,
    min_samples_per_class: int = 2,
    max_folds: int = 3,
    random_state: int = 42,
    confidence_threshold: float = 0.75,
    negative_confidence_threshold: float = 0.65,
    min_margin: float = 0.10,
    distance_multiplier: float = 2.50,
    verifier_threshold: float = 0.65,
) -> RejectionBenchmarkReport:
    taxonomy = load_gesture_taxonomy(taxonomy_path)
    selected_types = set(_scope_types(scope))
    actual_feature_mode = feature_mode or (
        FEATURE_DYNAMIC_STATS if str(scope).strip().lower() == "dynamic" else FEATURE_STATIC_MEAN
    )

    all_records = _load_records_for_benchmark(
        data_root,
        include_augmented=include_augmented,
        include_labels=include_labels,
        lowercase_labels=lowercase_labels,
    )
    raw_allowed_labels = [
        str(label).strip()
        for label in (include_labels or [])
        if str(label).strip()
    ]
    allowed_labels = (
        {
            label.lower() if lowercase_labels else label
            for label in raw_allowed_labels
        }
        if raw_allowed_labels
        else None
    )
    records = [
        record
        for record in all_records
        if taxonomy.gesture_type_for_label(record.label) in selected_types
        and (allowed_labels is None or record.label in allowed_labels)
    ]
    counts = class_counts(records)
    kept_labels = {
        label
        for label, count in counts.items()
        if int(count) >= int(min_samples_per_class)
    }
    records = [record for record in records if record.label in kept_labels]
    if not records:
        raise RuntimeError("no records selected for rejection benchmark")
    actual_target_dim = int(
        target_dim
        if target_dim is not None
        else infer_target_dim(record.sequence for record in records)
    )

    X, y_indices, labels = build_feature_matrix(
        records,
        mode=actual_feature_mode,
        target_dim=actual_target_dim,
    )
    y = np.asarray([labels[index] for index in y_indices], dtype=object)
    positive_labels = {
        label for label in labels if not _is_negative_label(taxonomy, label)
    }
    negative_labels = {
        label for label in labels if _is_negative_label(taxonomy, label)
    }
    if not positive_labels:
        raise RuntimeError("benchmark needs at least one positive label")
    if not negative_labels:
        raise RuntimeError("benchmark needs at least one negative label")

    filtered_counts = {label: int(np.count_nonzero(y == label)) for label in labels}
    min_count = min(filtered_counts.values())
    folds = max(2, min(int(max_folds), int(min_count)))
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=int(random_state),
    )
    selected_methods = _method_list(methods)
    method_true: dict[str, list[str]] = {method: [] for method in selected_methods}
    method_decisions: dict[str, list[Decision]] = {method: [] for method in selected_methods}
    method_errors: dict[str, str] = {}

    common_kwargs = {
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
        "candidate_model": str(candidate_model or "knn").strip().lower(),
        "confidence_threshold": float(confidence_threshold),
        "negative_confidence_threshold": float(negative_confidence_threshold),
        "min_margin": float(min_margin),
        "distance_multiplier": float(distance_multiplier),
        "verifier_threshold": float(verifier_threshold),
        "random_state": int(random_state),
    }

    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        y_test = [str(label) for label in y[test_idx]]
        for method in selected_methods:
            try:
                decisions = _predict_method(
                    method,
                    X_train,
                    y_train,
                    X_test,
                    **common_kwargs,
                )
            except Exception as exc:
                method_errors[method] = str(exc)
                decisions = [Decision("", "method_failed") for _ in y_test]
            method_true[method].extend(y_test)
            method_decisions[method].extend(decisions)

    results = [
        _summarize_predictions(
            method,
            method_true[method],
            method_decisions[method],
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            error=method_errors.get(method, ""),
        )
        for method in selected_methods
    ]
    ok_results = [result for result in results if result.status == "ok"]
    ranked = sorted(
        ok_results or results,
        key=lambda item: (
            item.overall_success,
            item.negative_reject_rate,
            item.positive_recall,
            -item.negative_false_positive_rate,
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    recommendation = (
        f"Use `{best.method}` first: overall_success={best.overall_success:.4f}, "
        f"positive_recall={best.positive_recall:.4f}, "
        f"negative_false_positive_rate={best.negative_false_positive_rate:.4f}."
        if best is not None
        else "No method produced a result."
    )
    dataset = RejectionDatasetInfo(
        data_root=str(data_root),
        scope=str(scope),
        feature_mode=actual_feature_mode,
        candidate_model=str(candidate_model or "knn").strip().lower(),
        target_dim=actual_target_dim,
        sample_count=int(len(y)),
        class_count=int(len(labels)),
        positive_labels=sorted(positive_labels),
        negative_labels=sorted(negative_labels),
        class_counts=dict(sorted(filtered_counts.items())),
        folds=folds,
        include_augmented=bool(include_augmented),
    )
    return RejectionBenchmarkReport(
        generated_at=time.time(),
        dataset=dataset,
        methods=results,
        best_method=best.method if best is not None else "",
        recommendation=recommendation,
    )


def build_markdown_report(report: RejectionBenchmarkReport) -> str:
    dataset = report.dataset
    lines = [
        "# Rejection Method Benchmark",
        "",
        f"Generated at: `{report.generated_at:.3f}`",
        "",
        "## Dataset",
        "",
        f"- data root: `{dataset.data_root}`",
        f"- scope: `{dataset.scope}`",
        f"- feature mode: `{dataset.feature_mode}`",
        f"- candidate model: `{dataset.candidate_model}`",
        f"- target dim: `{dataset.target_dim}`",
        f"- augmented samples: `{'included' if dataset.include_augmented else 'excluded'}`",
        f"- samples/classes/folds: `{dataset.sample_count}` / `{dataset.class_count}` / `{dataset.folds}`",
        f"- positive labels: `{', '.join(dataset.positive_labels)}`",
        f"- negative labels: `{', '.join(dataset.negative_labels)}`",
        "",
        "## Results",
        "",
        "| Method | Status | Overall | Pos recall | Pos reject | Neg reject | Neg FP | Accepted acc | Coverage | Wrong accepts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(
        report.methods,
        key=lambda item: (
            item.status != "ok",
            -item.overall_success,
            item.negative_false_positive_rate,
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{result.method}`",
                    result.status,
                    f"{result.overall_success:.4f}",
                    f"{result.positive_recall:.4f}",
                    f"{result.positive_reject_rate:.4f}",
                    f"{result.negative_reject_rate:.4f}",
                    f"{result.negative_false_positive_rate:.4f}",
                    f"{result.accepted_accuracy:.4f}",
                    f"{result.coverage:.4f}",
                    str(result.wrong_accept),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            report.recommendation,
            "",
            "## Method Notes",
            "",
            "- `negative_classes`: supervised multiclass model rejects when predicted class is negative.",
            "- `confidence_threshold`: rejects low-confidence positive predictions.",
            "- `open_set_policy`: current runtime-style policy: negative probability, top1/top2 margin, prototype distance.",
            "- `one_vs_rest_logreg`: binary verifier per positive class.",
            "- `one_class_svm`, `isolation_forest`, `local_outlier_factor`: one-class/outlier verifier per positive class.",
            "- `metric_nca_centroid`: metric-learning embedding + class prototype radius.",
            "- `mlp_negative_classes`: nonlinear supervised classifier with negative classes.",
            "",
            "## Best Method Negative Breakdown",
            "",
        ]
    )
    best_result = next(
        (result for result in report.methods if result.method == report.best_method),
        None,
    )
    if best_result is None:
        lines.append("No best method available.")
    else:
        lines.extend(
            [
                f"Best method: `{best_result.method}`",
                "",
                "| Negative label | Total | Rejected | False positive | Reject rate | FP rate | FP predictions |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in best_result.negative_label_metrics:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row.label}`",
                        str(row.total),
                        str(row.rejected),
                        str(row.false_positive),
                        f"{row.reject_rate:.4f}",
                        f"{row.false_positive_rate:.4f}",
                        f"`{json.dumps(row.false_positive_predictions, ensure_ascii=False)}`",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Reject Reasons",
            "",
        ]
    )
    for result in report.methods:
        lines.append(f"### `{result.method}`")
        if result.error:
            lines.append(f"- error: `{result.error}`")
        lines.append(f"- reject reasons: `{json.dumps(result.reject_reasons, ensure_ascii=False)}`")
        lines.append(
            "- false positive predictions: "
            f"`{json.dumps(result.false_positive_predictions, ensure_ascii=False)}`"
        )
        lines.append("")
    return "\n".join(lines)


def _write_report(report: RejectionBenchmarkReport, json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_out.write_text(build_markdown_report(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "gestures")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=PROJECT_ROOT / "configs" / "gesture_taxonomy.json",
    )
    parser.add_argument("--scope", default="static", choices=["static", "dynamic", "all"])
    parser.add_argument("--feature-mode", default="")
    parser.add_argument(
        "--candidate-model",
        default="knn",
        choices=["knn", "extra_trees", "rf", "logreg", "svm"],
    )
    parser.add_argument("--target-dim", type=int, default=None)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument(
        "--include-label",
        action="append",
        default=[],
        help="Restrict benchmark to a label. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-augmented",
        action="store_true",
        help="Include GISLR-marked aug_sample_* files with gislr_landmark_v1 metadata.",
    )
    parser.add_argument(
        "--lowercase-labels",
        action="store_true",
        help="Normalize labels to lowercase before filtering and taxonomy lookup.",
    )
    parser.add_argument("--min-samples-per-class", type=int, default=2)
    parser.add_argument("--max-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--negative-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--distance-multiplier", type=float, default=2.50)
    parser.add_argument("--verifier-threshold", type=float, default=0.65)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "experiments" / "rejection_method_benchmark.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "experiments" / "rejection_method_benchmark.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = benchmark_rejection_methods(
        data_root=args.data_root,
        taxonomy_path=args.taxonomy,
        scope=args.scope,
        feature_mode=str(args.feature_mode or "") or None,
        candidate_model=args.candidate_model,
        target_dim=args.target_dim,
        methods=args.methods,
        include_labels=args.include_label,
        include_augmented=args.include_augmented,
        lowercase_labels=args.lowercase_labels,
        min_samples_per_class=args.min_samples_per_class,
        max_folds=args.max_folds,
        random_state=args.random_state,
        confidence_threshold=args.confidence_threshold,
        negative_confidence_threshold=args.negative_confidence_threshold,
        min_margin=args.min_margin,
        distance_multiplier=args.distance_multiplier,
        verifier_threshold=args.verifier_threshold,
    )
    _write_report(report, args.json_out, args.md_out)
    print(build_markdown_report(report))
    print(f"[✓] JSON: {args.json_out}")
    print(f"[✓] Markdown: {args.md_out}")


if __name__ == "__main__":
    main()
