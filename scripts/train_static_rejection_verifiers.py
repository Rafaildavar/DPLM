"""Train live static rejection verifier methods.

The base static classifier predicts a gesture class. This script trains optional
second-stage verifiers so the Flet live UI can compare rejection strategies on a
real camera stream, not only in offline reports.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor, NeighborhoodComponentsAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv.train_classifier import (  # noqa: E402
    DEFAULT_REJECT_NEGATIVE_CONFIDENCE_THRESHOLD,
    DEFAULT_REJECT_DISTANCE_MULTIPLIER,
    FEATURE_STATIC_MEAN,
    is_negative_label,
    load_dataset,
)

DEFAULT_METHODS = (
    "one_vs_rest_logreg",
    "one_class_svm",
    "isolation_forest",
    "local_outlier_factor",
    "metric_nca_centroid",
    "mlp_negative_classes",
)


def _read_classes(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if str(item).strip()]


def _method_list(raw: Iterable[str] | str | None) -> list[str]:
    if raw is None:
        return list(DEFAULT_METHODS)
    if isinstance(raw, str):
        values = raw.split(",")
    else:
        values = list(raw)
    methods = [str(item).strip() for item in values if str(item).strip()]
    return methods or list(DEFAULT_METHODS)


def _fit_one_vs_rest(
    X: np.ndarray,
    y_labels: np.ndarray,
    positive_labels: list[str],
    *,
    random_state: int,
    threshold: float,
) -> dict[str, Any]:
    verifiers: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for label in positive_labels:
        binary_y = (y_labels == label).astype(int)
        if int(binary_y.sum()) < 2:
            skipped[label] = "needs at least 2 positive samples"
            continue
        if len(np.unique(binary_y)) < 2:
            skipped[label] = "needs positive and negative rows"
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=int(random_state),
            ),
        )
        model.fit(X, binary_y)
        verifiers[label] = model
    return {
        "status": "ok" if verifiers else "skipped",
        "threshold": float(threshold),
        "verifiers": verifiers,
        "skipped": skipped,
    }


def _fit_outlier_models(
    X: np.ndarray,
    y_labels: np.ndarray,
    positive_labels: list[str],
    *,
    method: str,
    random_state: int,
    contamination: float,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for label in positive_labels:
        rows = X[y_labels == label]
        if rows.shape[0] < 3:
            skipped[label] = "needs at least 3 positive samples"
            continue
        if method == "one_class_svm":
            model = make_pipeline(
                StandardScaler(),
                OneClassSVM(gamma="scale", nu=max(0.01, min(0.45, contamination))),
            )
        elif method == "isolation_forest":
            model = IsolationForest(
                contamination=max(0.01, min(0.45, contamination)),
                random_state=int(random_state),
            )
        elif method == "local_outlier_factor":
            neighbors = max(2, min(10, rows.shape[0] - 1))
            model = make_pipeline(
                StandardScaler(),
                LocalOutlierFactor(
                    n_neighbors=neighbors,
                    contamination=max(0.01, min(0.45, contamination)),
                    novelty=True,
                ),
            )
        else:
            skipped[label] = f"unsupported outlier method: {method}"
            continue
        model.fit(rows)
        models[label] = model
    return {
        "status": "ok" if models else "skipped",
        "models": models,
        "contamination": float(contamination),
        "skipped": skipped,
    }


def _fit_metric_centroids(
    X: np.ndarray,
    y_labels: np.ndarray,
    positive_labels: list[str],
    *,
    random_state: int,
    distance_multiplier: float,
) -> dict[str, Any]:
    mask = np.asarray([label in set(positive_labels) for label in y_labels], dtype=bool)
    X_pos = X[mask]
    y_pos = y_labels[mask]
    if X_pos.shape[0] < 4 or len(set(y_pos.tolist())) < 2:
        return {
            "status": "skipped",
            "reason": "needs at least 2 positive classes and 4 positive samples",
        }

    n_components = max(1, min(12, int(X_pos.shape[1]), int(X_pos.shape[0]) - 1))
    transformer = make_pipeline(
        StandardScaler(),
        NeighborhoodComponentsAnalysis(
            n_components=n_components,
            random_state=int(random_state),
            max_iter=500,
        ),
    )
    transformer.fit(X_pos, y_pos)
    embedded = np.asarray(transformer.transform(X_pos), dtype=np.float32)
    prototypes: dict[str, dict[str, Any]] = {}
    radius_floor = 0.015 * float(np.sqrt(max(1, embedded.shape[1])))
    for label in positive_labels:
        rows = embedded[y_pos == label]
        if rows.shape[0] == 0:
            continue
        centroid = rows.mean(axis=0)
        distances = np.linalg.norm(rows - centroid, axis=1)
        radius = max(
            float(np.percentile(distances, 95)) if distances.size else 0.0,
            float(np.mean(distances) + 2.0 * np.std(distances)) if distances.size else 0.0,
            radius_floor,
        )
        prototypes[label] = {
            "centroid": centroid,
            "radius": float(radius),
            "threshold": float(radius * distance_multiplier),
        }
    return {
        "status": "ok" if prototypes else "skipped",
        "transformer": transformer,
        "prototypes": prototypes,
        "distance_multiplier": float(distance_multiplier),
    }


def _fit_mlp_negative_classes(
    X: np.ndarray,
    y_labels: np.ndarray,
    *,
    random_state: int,
    negative_threshold: float,
) -> dict[str, Any]:
    if len(set(y_labels.tolist())) < 2:
        return {"status": "skipped", "reason": "needs at least 2 classes"}
    hidden = max(8, min(64, int(X.shape[1]) // 2))
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(hidden,),
            alpha=0.01,
            max_iter=1000,
            random_state=int(random_state),
        ),
    )
    model.fit(X, y_labels)
    return {
        "status": "ok",
        "model": model,
        "negative_threshold": float(negative_threshold),
    }


def train_static_rejection_verifiers(
    *,
    data_root: Path,
    classes_path: Path,
    out_path: Path,
    feature_mode: str = FEATURE_STATIC_MEAN,
    expect_dim: int = 42,
    methods: Iterable[str] | str | None = None,
    random_state: int = 42,
    one_vs_rest_threshold: float = 0.50,
    negative_threshold: float = DEFAULT_REJECT_NEGATIVE_CONFIDENCE_THRESHOLD,
    distance_multiplier: float = DEFAULT_REJECT_DISTANCE_MULTIPLIER,
    contamination: float = 0.10,
    lowercase_labels: bool = True,
) -> dict[str, Any]:
    selected_methods = _method_list(methods)
    include_labels = _read_classes(classes_path)
    X, y, classes = load_dataset(
        data_root,
        expect_dim=int(expect_dim),
        include_labels=include_labels,
        lowercase_labels=bool(lowercase_labels),
        feature_mode=str(feature_mode),
    )
    y_labels = np.asarray([classes[int(index)] for index in y], dtype=object)
    positive_labels = [label for label in classes if not is_negative_label(label)]
    negative_labels = [label for label in classes if is_negative_label(label)]

    method_payloads: dict[str, Any] = {}
    for method in selected_methods:
        if method == "one_vs_rest_logreg":
            method_payloads[method] = _fit_one_vs_rest(
                X,
                y_labels,
                positive_labels,
                random_state=random_state,
                threshold=one_vs_rest_threshold,
            )
        elif method in {
            "one_class_svm",
            "isolation_forest",
            "local_outlier_factor",
        }:
            method_payloads[method] = _fit_outlier_models(
                X,
                y_labels,
                positive_labels,
                method=method,
                random_state=random_state,
                contamination=contamination,
            )
        elif method == "metric_nca_centroid":
            method_payloads[method] = _fit_metric_centroids(
                X,
                y_labels,
                positive_labels,
                random_state=random_state,
                distance_multiplier=distance_multiplier,
            )
        elif method == "mlp_negative_classes":
            method_payloads[method] = _fit_mlp_negative_classes(
                X,
                y_labels,
                random_state=random_state,
                negative_threshold=negative_threshold,
            )
        else:
            method_payloads[method] = {
                "status": "skipped",
                "reason": f"unsupported method: {method}",
            }

    payload = {
        "schema_version": 1,
        "generated_by": "scripts.train_static_rejection_verifiers",
        "generated_at": time.time(),
        "data_root": str(data_root),
        "classes_path": str(classes_path),
        "feature_mode": str(feature_mode),
        "feature_dim": int(X.shape[1]),
        "expect_dim": int(expect_dim),
        "classes": [str(label) for label in classes],
        "positive_labels": [str(label) for label in positive_labels],
        "negative_labels": [str(label) for label in negative_labels],
        "sample_count": int(X.shape[0]),
        "methods": method_payloads,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_path)
    return payload


def _log_mlflow(
    payload: dict[str, Any],
    *,
    out_path: Path,
    experiment: str,
    tracking_uri: str,
    run_name: str,
) -> None:
    experiment = str(experiment or "").strip()
    if not experiment:
        return
    try:
        import mlflow
    except Exception as exc:
        print(f"[w] MLflow недоступен, verifier tracking пропущен: {exc}")
        return
    try:
        methods = payload.get("methods") if isinstance(payload, dict) else {}
        ok_methods = [
            method
            for method, raw in (methods or {}).items()
            if isinstance(raw, dict) and raw.get("status") == "ok"
        ]
        mlflow.set_tracking_uri(str(tracking_uri))
        mlflow.set_experiment(str(experiment))
        with mlflow.start_run(run_name=run_name or "static-rejection-verifiers"):
            mlflow.set_tags(
                {
                    "run_kind": "static_rejection_verifier_training",
                    "source": "scripts.train_static_rejection_verifiers",
                }
            )
            mlflow.log_params(
                {
                    "feature_mode": str(payload.get("feature_mode") or ""),
                    "expect_dim": int(payload.get("expect_dim") or 0),
                    "classes": ",".join(payload.get("classes") or []),
                    "positive_labels": ",".join(payload.get("positive_labels") or []),
                    "negative_labels": ",".join(payload.get("negative_labels") or []),
                    "methods": ",".join((methods or {}).keys()),
                    "ok_methods": ",".join(ok_methods),
                }
            )
            mlflow.log_metrics(
                {
                    "sample_count": float(payload.get("sample_count") or 0),
                    "class_count": float(len(payload.get("classes") or [])),
                    "positive_label_count": float(len(payload.get("positive_labels") or [])),
                    "negative_label_count": float(len(payload.get("negative_labels") or [])),
                    "method_count": float(len(methods or {})),
                    "ok_method_count": float(len(ok_methods)),
                }
            )
            if out_path.exists():
                mlflow.log_artifact(str(out_path))
        print(f"[✓] MLflow verifier run logged: experiment={experiment!r}, uri={tracking_uri}")
    except Exception as exc:
        print(f"[w] MLflow verifier logging failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train static rejection verifier layer")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "gestures")
    parser.add_argument("--classes-path", type=Path, default=PROJECT_ROOT / "models" / "classes.json")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "models" / "static_rejection_verifiers.pkl",
    )
    parser.add_argument("--feature-mode", default=FEATURE_STATIC_MEAN)
    parser.add_argument("--expect-dim", type=int, default=42)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--one-vs-rest-threshold", type=float, default=0.50)
    parser.add_argument("--negative-threshold", type=float, default=DEFAULT_REJECT_NEGATIVE_CONFIDENCE_THRESHOLD)
    parser.add_argument("--distance-multiplier", type=float, default=DEFAULT_REJECT_DISTANCE_MULTIPLIER)
    parser.add_argument("--contamination", type=float, default=0.10)
    parser.add_argument("--keep-label-case", action="store_true")
    parser.add_argument("--mlflow-experiment", default="GestureFlow")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow-run-name", default="static-rejection-verifiers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = train_static_rejection_verifiers(
        data_root=args.data_root,
        classes_path=args.classes_path,
        out_path=args.out,
        feature_mode=str(args.feature_mode),
        expect_dim=int(args.expect_dim),
        methods=str(args.methods),
        random_state=int(args.random_state),
        one_vs_rest_threshold=float(args.one_vs_rest_threshold),
        negative_threshold=float(args.negative_threshold),
        distance_multiplier=float(args.distance_multiplier),
        contamination=float(args.contamination),
        lowercase_labels=not bool(args.keep_label_case),
    )
    methods = payload.get("methods") or {}
    ok_methods = [
        method
        for method, raw in methods.items()
        if isinstance(raw, dict) and raw.get("status") == "ok"
    ]
    print(
        "[✓] Static rejection verifiers saved: "
        f"{args.out} ({len(ok_methods)}/{len(methods)} methods ready)"
    )
    for method, raw in methods.items():
        status = raw.get("status") if isinstance(raw, dict) else "unknown"
        print(f"  - {method}: {status}")
    _log_mlflow(
        payload,
        out_path=args.out,
        experiment=str(args.mlflow_experiment),
        tracking_uri=str(args.mlflow_tracking_uri),
        run_name=str(args.mlflow_run_name),
    )


if __name__ == "__main__":
    main()
