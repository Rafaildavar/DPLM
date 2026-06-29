import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cv.gesture_features import (
    FEATURE_STATIC_MEAN,
    SUPPORTED_FEATURE_MODES,
    build_feature_vector,
    feature_vector_size,
)
from cv.gesture_dataset_files import gesture_sample_paths

SUPPORTED_MODEL_TYPES = (
    "knn",
    "sequence_knn",
    "sequence_mlp",
    "svm",
    "extra_trees",
    "rf",
    "logreg",
)
DEFAULT_REJECT_NEGATIVE_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_REJECT_MIN_MARGIN = 0.10
DEFAULT_REJECT_DISTANCE_MULTIPLIER = 2.50
DEFAULT_PROTOTYPE_RADIUS_FLOOR_SCALE = 0.015


# --------------------------------------------------
# Тренер KNN: загрузка NPY семплов и обучение модели
# Комментарии на русском
# --------------------------------------------------


def load_dataset(
    data_root: Path,
    expect_dim: Optional[int] = None,
    include_labels: Optional[Iterable[str]] = None,
    lowercase_labels: bool = False,
    feature_mode: str = FEATURE_STATIC_MEAN,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Загружает реальные семплы из data_root/<label>/sample_*.npy
    Возвращает (X, y, classes), где:
      - X: (N, D) — признаки семплов в выбранном feature_mode
      - y: (N,) — индексы классов
      - classes: список имён классов по индексу
    """
    # Временно храним признаки переменной длины, затем выровняем по max/expect_dim
    feats_raw: List[np.ndarray] = []
    y_list: List[int] = []
    classes: List[str] = []
    class_indices: dict[str, int] = {}
    max_dim: int = 0
    raw_include = {str(label).strip() for label in (include_labels or []) if str(label).strip()}
    canonical_include = {
        label.lower() if lowercase_labels else label for label in raw_include
    }

    for label_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        raw_label = label_dir.name
        label = raw_label.lower() if lowercase_labels else raw_label
        if canonical_include and label not in canonical_include and raw_label not in raw_include:
            continue
        sample_files = gesture_sample_paths(label_dir)
        if not sample_files:
            print(f"[i] Пропуск: нет семплов в {label_dir}")
            continue

        if label not in class_indices:
            class_indices[label] = len(classes)
            classes.append(label)
        class_idx = class_indices[label]

        for sf in sample_files:
            arr = np.load(sf)  # ожидаем (T, D1, D2) или (T, D)

            # Приводим к (T, D)
            if arr.ndim == 3:
                T, D1, D2 = arr.shape
                arr = arr.reshape(T, D1 * D2)
            elif arr.ndim != 2:
                print(f"[!] Неожиданная форма {sf}: {arr.shape}, пропуск")
                continue

            feat = build_feature_vector(arr, mode=feature_mode, target_dim=expect_dim)
            # Сохраняем как есть, выровняем позже
            feats_raw.append(feat.astype(np.float32, copy=False))
            y_list.append(class_idx)
            if feat.shape[0] > max_dim:
                max_dim = int(feat.shape[0])

    if not feats_raw:
        raise RuntimeError("Датасет пуст — не найдено ни одного семпла")

    # ``expect_dim`` describes the raw per-frame dimension passed to
    # build_feature_vector. The final model feature dimension depends on the
    # selected feature mode, e.g. dynamic_stats expands 44 raw values to 271.
    target_dim = (
        feature_vector_size(feature_mode, expect_dim)
        if expect_dim is not None
        else max_dim
    )
    if expect_dim is not None and max_dim > target_dim:
        print(
            f"[w] Найдены признаки длиной {max_dim} > ожидаемой {target_dim}. "
            "Лишние компоненты будут обрезаны."
        )
    if len(set(f.shape[0] for f in feats_raw)) > 1:
        print(
            f"[i] Выравниваем разные длины признаков до {target_dim} "
            "(дополнение нулями/обрезка)"
        )

    # Выровнять признаки до target_dim: обрезать или дополнить нулями
    X_aligned: List[np.ndarray] = []
    for f in feats_raw:
        if f.shape[0] == target_dim:
            X_aligned.append(f)
        elif f.shape[0] > target_dim:
            X_aligned.append(f[:target_dim])
        else:
            pad = np.zeros(target_dim - f.shape[0], dtype=f.dtype)
            X_aligned.append(np.concatenate([f, pad], axis=0))

    X = np.stack(X_aligned, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, classes


def build_classifier(
    model_type: str,
    *,
    neighbors: int = 5,
    weights: str = "distance",
    random_state: int = 42,
):
    model = str(model_type or "knn").strip().lower()
    if model in {"knn", "sequence_knn"}:
        return KNeighborsClassifier(
            n_neighbors=max(1, int(neighbors)),
            metric="euclidean",
            weights=weights,
        )
    if model == "sequence_mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=800,
                n_iter_no_change=30,
                random_state=int(random_state),
            ),
        )
    if model == "svm":
        return make_pipeline(
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
    if model == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=250,
            random_state=int(random_state),
            class_weight="balanced",
        )
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=int(random_state),
            class_weight="balanced",
        )
    if model == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=int(random_state),
            ),
        )
    raise ValueError(f"unsupported model type: {model_type}")


def is_negative_label(label: str) -> bool:
    clean = str(label or "").strip().lower()
    return (
        clean.startswith("negative_")
        or clean.startswith("background_")
        or clean.startswith("no_gesture")
        or clean.startswith("random_")
        or clean.startswith("partial_")
        or clean.startswith("return_")
        or clean.startswith("wrong_axis_")
    )


def build_rejection_metadata(
    X: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    *,
    model_type: str,
    feature_mode: str,
    negative_confidence_threshold: float = DEFAULT_REJECT_NEGATIVE_CONFIDENCE_THRESHOLD,
    min_margin: float = DEFAULT_REJECT_MIN_MARGIN,
    distance_multiplier: float = DEFAULT_REJECT_DISTANCE_MULTIPLIER,
) -> dict:
    feature_dim = int(X.shape[1]) if X.ndim == 2 else 0
    radius_floor = float(DEFAULT_PROTOTYPE_RADIUS_FLOOR_SCALE * np.sqrt(max(1, feature_dim)))
    class_metadata: dict[str, dict[str, object]] = {}
    for class_index, label in enumerate(classes):
        rows = X[y == class_index]
        if rows.size == 0:
            continue
        centroid = rows.mean(axis=0)
        distances = np.linalg.norm(rows - centroid, axis=1)
        distance_mean = float(np.mean(distances)) if distances.size else 0.0
        distance_std = float(np.std(distances)) if distances.size else 0.0
        distance_p95 = float(np.percentile(distances, 95)) if distances.size else 0.0
        distance_max = float(np.max(distances)) if distances.size else 0.0
        radius = max(distance_p95, distance_mean + 2.0 * distance_std, radius_floor)
        class_metadata[str(label)] = {
            "sample_count": int(rows.shape[0]),
            "centroid": [float(value) for value in centroid.astype(float).tolist()],
            "distance_mean": distance_mean,
            "distance_std": distance_std,
            "distance_p95": distance_p95,
            "distance_max": distance_max,
            "prototype_radius": float(radius),
        }

    negative_labels = [str(label) for label in classes if is_negative_label(label)]
    return {
        "schema_version": 1,
        "generated_by": "cv.train_classifier",
        "generated_at": time.time(),
        "model_type": str(model_type),
        "feature_mode": str(feature_mode),
        "feature_dim": feature_dim,
        "negative_labels": negative_labels,
        "thresholds": {
            "negative_confidence": float(negative_confidence_threshold),
            "min_top1_top2_margin": float(min_margin),
            "distance_multiplier": float(distance_multiplier),
            "prototype_radius_floor": radius_floor,
        },
        "classes": class_metadata,
    }


def default_rejection_metadata_path(out_path: Path) -> Path:
    stem = out_path.stem
    if stem.startswith("dynamic_"):
        return out_path.with_name(f"{stem}_rejection.json")
    return out_path.parent / "gesture_rejection.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение KNN классификатора жестов")
    p.add_argument("--data-root", default="data/gestures", help="Корень датасета")
    p.add_argument("--out", default="models/knn.pkl", help="Путь для сохранения модели")
    p.add_argument("--classes-out", default=None, help="Путь для сохранения classes.json")
    p.add_argument("--feature-dim-out", default=None, help="Путь для сохранения feature_dim.txt")
    p.add_argument("--feature-mode-out", default=None, help="Путь для сохранения feature_mode.txt")
    p.add_argument(
        "--rejection-out",
        default=None,
        help="Путь для сохранения gesture_rejection.json",
    )
    p.add_argument(
        "--feature-mode",
        choices=SUPPORTED_FEATURE_MODES,
        default=FEATURE_STATIC_MEAN,
        help="Режим признаков: static_mean — текущий production baseline; dynamic_stats/hybrid_stats — JMLC-эксперименты",
    )
    p.add_argument("--neighbors", type=int, default=5, help="Число соседей KNN")
    p.add_argument(
        "--model-type",
        choices=SUPPORTED_MODEL_TYPES,
        default="knn",
        help="Тип классификатора: knn, sequence_knn, sequence_mlp, svm, extra_trees, rf или logreg",
    )
    p.add_argument("--random-state", type=int, default=42, help="Seed для моделей с рандомизацией")
    p.add_argument(
        "--weights",
        choices=["uniform", "distance"],
        default="distance",
        help="Вес соседей KNN: distance устойчивее для маленьких несбалансированных наборов",
    )
    p.add_argument("--expect-dim", type=int, default=None, help="Ожидаемая длина признака (например, 42 или 84)")
    p.add_argument(
        "--include-label",
        action="append",
        default=[],
        help="Обучать только указанный класс; можно передать несколько раз",
    )
    p.add_argument(
        "--lowercase-labels",
        action="store_true",
        help="Сохранять имена классов в нижнем регистре (New -> new)",
    )
    p.add_argument(
        "--mlflow-experiment",
        default="GestureFlow",
        help="MLflow experiment name; empty disables MLflow logging",
    )
    p.add_argument(
        "--mlflow-tracking-uri",
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI, e.g. sqlite:///mlflow.db",
    )
    p.add_argument(
        "--mlflow-run-name",
        default="",
        help="Optional MLflow run name",
    )
    p.add_argument(
        "--reject-negative-confidence-threshold",
        type=float,
        default=DEFAULT_REJECT_NEGATIVE_CONFIDENCE_THRESHOLD,
        help="Reject when any negative class probability is at least this value.",
    )
    p.add_argument(
        "--reject-min-margin",
        type=float,
        default=DEFAULT_REJECT_MIN_MARGIN,
        help="Reject when top1-top2 probability margin is below this value.",
    )
    p.add_argument(
        "--reject-distance-multiplier",
        type=float,
        default=DEFAULT_REJECT_DISTANCE_MULTIPLIER,
        help="Reject when distance to predicted prototype exceeds radius*multiplier.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X, y, classes = load_dataset(
        data_root,
        expect_dim=args.expect_dim,
        include_labels=args.include_label,
        lowercase_labels=bool(args.lowercase_labels),
        feature_mode=str(args.feature_mode),
    )
    print(
        f"[i] Загружено семплов: {len(X)}; классов: {len(classes)}; "
        f"режим признаков: {args.feature_mode}; размер признака: {X.shape[1]}; "
        f"модель: {args.model_type}"
    )

    clf = build_classifier(
        str(args.model_type),
        neighbors=int(args.neighbors),
        weights=str(args.weights),
        random_state=int(args.random_state),
    )
    clf.fit(X, y)
    train_accuracy = float(clf.score(X, y))

    joblib.dump(clf, out_path)
    print(f"[✓] Модель сохранена: {out_path}")

    # Сохраним классы и размерность признака для инференса
    classes_out = Path(args.classes_out) if args.classes_out else (out_path.parent / "classes.json")
    feature_dim_out = (
        Path(args.feature_dim_out)
        if args.feature_dim_out
        else (out_path.parent / "feature_dim.txt")
    )
    feature_mode_out = (
        Path(args.feature_mode_out)
        if args.feature_mode_out
        else (out_path.parent / "feature_mode.txt")
    )
    rejection_out = (
        Path(args.rejection_out)
        if args.rejection_out
        else default_rejection_metadata_path(out_path)
    )
    classes_out.parent.mkdir(parents=True, exist_ok=True)
    feature_dim_out.parent.mkdir(parents=True, exist_ok=True)
    feature_mode_out.parent.mkdir(parents=True, exist_ok=True)
    rejection_out.parent.mkdir(parents=True, exist_ok=True)
    rejection_metadata = build_rejection_metadata(
        X,
        y,
        classes,
        model_type=str(args.model_type),
        feature_mode=str(args.feature_mode),
        negative_confidence_threshold=float(args.reject_negative_confidence_threshold),
        min_margin=float(args.reject_min_margin),
        distance_multiplier=float(args.reject_distance_multiplier),
    )
    classes_out.write_text(json.dumps(classes, ensure_ascii=False, indent=2))
    feature_dim_out.write_text(str(X.shape[1]))
    feature_mode_out.write_text(str(args.feature_mode))
    rejection_out.write_text(
        json.dumps(rejection_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "[✓] Метаданные сохранены: "
        f"{classes_out}, {feature_dim_out}, {feature_mode_out}, {rejection_out}"
    )
    print(f"[i] Training accuracy: {train_accuracy:.4f}")

    _log_mlflow_run(
        args=args,
        classes=classes,
        sample_count=int(X.shape[0]),
        feature_dim=int(X.shape[1]),
        train_accuracy=train_accuracy,
        out_path=out_path,
        classes_out=classes_out,
        feature_dim_out=feature_dim_out,
        feature_mode_out=feature_mode_out,
        rejection_out=rejection_out,
        rejection_metadata=rejection_metadata,
    )


def _log_mlflow_run(
    *,
    args: argparse.Namespace,
    classes: list[str],
    sample_count: int,
    feature_dim: int,
    train_accuracy: float,
    out_path: Path,
    classes_out: Path,
    feature_dim_out: Path,
    feature_mode_out: Path,
    rejection_out: Path,
    rejection_metadata: dict,
) -> None:
    experiment = str(getattr(args, "mlflow_experiment", "") or "").strip()
    if not experiment:
        return
    try:
        import mlflow
    except Exception as exc:
        print(f"[w] MLflow недоступен, tracking пропущен: {exc}")
        return

    try:
        tracking_uri = str(
            getattr(args, "mlflow_tracking_uri", "") or "sqlite:///mlflow.db"
        )
        reject_negative_confidence_threshold = float(
            getattr(
                args,
                "reject_negative_confidence_threshold",
                DEFAULT_REJECT_NEGATIVE_CONFIDENCE_THRESHOLD,
            )
        )
        reject_min_margin = float(
            getattr(args, "reject_min_margin", DEFAULT_REJECT_MIN_MARGIN)
        )
        reject_distance_multiplier = float(
            getattr(
                args,
                "reject_distance_multiplier",
                DEFAULT_REJECT_DISTANCE_MULTIPLIER,
            )
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        run_name = str(getattr(args, "mlflow_run_name", "") or "").strip() or (
            f"{args.model_type}-{args.feature_mode}"
        )
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "data_root": str(args.data_root),
                    "model_type": str(args.model_type),
                    "feature_mode": str(args.feature_mode),
                    "neighbors": int(args.neighbors),
                    "weights": str(args.weights),
                    "expect_dim": (
                        int(args.expect_dim)
                        if args.expect_dim is not None
                        else ""
                    ),
                    "lowercase_labels": bool(args.lowercase_labels),
                    "include_labels": ",".join(args.include_label or []),
                    "classes": ",".join(classes),
                    "reject_negative_confidence_threshold": (
                        reject_negative_confidence_threshold
                    ),
                    "reject_min_margin": reject_min_margin,
                    "reject_distance_multiplier": reject_distance_multiplier,
                }
            )
            negative_labels = rejection_metadata.get("negative_labels") or []
            mlflow.log_metrics(
                {
                    "sample_count": float(sample_count),
                    "class_count": float(len(classes)),
                    "feature_dim": float(feature_dim),
                    "train_accuracy": float(train_accuracy),
                    "negative_class_count": float(len(negative_labels)),
                }
            )
            for artifact in (
                out_path,
                classes_out,
                feature_dim_out,
                feature_mode_out,
                rejection_out,
            ):
                if artifact.exists():
                    mlflow.log_artifact(str(artifact))
        print(f"[✓] MLflow run logged: experiment={experiment!r}, uri={tracking_uri}")
    except Exception as exc:
        print(f"[w] MLflow logging failed: {exc}")


if __name__ == "__main__":
    main()
