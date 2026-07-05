import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
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
from cv.sequence_multirocket import RandomMultiRocketSequenceTransformer
from cv.sequence_phase_hmm import PhaseHMMSequenceClassifier
from cv.sequence_rocket import RandomConvolutionSequenceTransformer
from cv.sequence_gru_backbone import (
    TorchGRUBackboneClassifier,
    TorchLSTMBackboneClassifier,
    tune_gru_backbone_hyperparameters,
    tune_lstm_backbone_hyperparameters,
)
from cv.sequence_shapelet import ShapeletSequenceTransformer
from cv.sequence_sprocket import SprocketSequenceTransformer

SUPPORTED_MODEL_TYPES = (
    "knn",
    "sequence_knn",
    "sequence_mlp",
    "sequence_rocket",
    "sequence_multirocket",
    "sequence_sprocket",
    "sequence_shapelet",
    "sequence_phase_hmm",
    "sequence_ensemble",
    "sequence_gru_backbone",
    "sequence_lstm_backbone",
    "svm",
    "extra_trees",
    "rf",
    "logreg",
)
DEFAULT_REJECT_NEGATIVE_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_REJECT_MIN_MARGIN = 0.10
DEFAULT_REJECT_DISTANCE_MULTIPLIER = 2.50
DEFAULT_PROTOTYPE_RADIUS_FLOOR_SCALE = 0.015
DEFAULT_SEQUENCE_MLP_ALPHA = 1e-3
DEFAULT_SEQUENCE_MLP_EARLY_STOPPING = True
DEFAULT_SEQUENCE_MLP_VALIDATION_FRACTION = 0.20
DEFAULT_SEQUENCE_MLP_N_ITER_NO_CHANGE = 30
DEFAULT_SEQUENCE_ROCKET_KERNELS = 256
DEFAULT_SEQUENCE_ROCKET_MAX_DILATION = 4
DEFAULT_SEQUENCE_ROCKET_MAX_CHANNELS_PER_KERNEL = 8
DEFAULT_SEQUENCE_MULTIROCKET_KERNELS = 320
DEFAULT_SEQUENCE_MULTIROCKET_MAX_DILATION = 6
DEFAULT_SEQUENCE_MULTIROCKET_MAX_CHANNELS_PER_KERNEL = 8
DEFAULT_SEQUENCE_SPROCKET_KERNELS = 192
DEFAULT_SEQUENCE_SPROCKET_PROTOTYPES_PER_CLASS = 3
DEFAULT_SEQUENCE_SPROCKET_MAX_DILATION = 6
DEFAULT_SEQUENCE_SPROCKET_MAX_CHANNELS_PER_KERNEL = 8
DEFAULT_SEQUENCE_SHAPELETS_PER_CLASS = 18
DEFAULT_SEQUENCE_SHAPELET_MAX_CHANNELS = 12
DEFAULT_SEQUENCE_PHASE_HMM_STATES = 6
DEFAULT_SEQUENCE_PHASE_HMM_MAX_CHANNELS = 44
DEFAULT_SEQUENCE_PHASE_HMM_VARIANCE_REGULARIZATION = 0.02
DEFAULT_SEQUENCE_ENSEMBLE_WEIGHTS = (0.35, 0.30, 0.25, 0.10)
DEFAULT_SEQUENCE_GRU_BACKBONE_DIM = 64
DEFAULT_SEQUENCE_GRU_HIDDEN_DIM = 96
DEFAULT_SEQUENCE_GRU_LAYERS = 1
DEFAULT_SEQUENCE_GRU_DROPOUT = 0.20
DEFAULT_SEQUENCE_GRU_LEARNING_RATE = 1e-3
DEFAULT_SEQUENCE_GRU_WEIGHT_DECAY = 1e-4
DEFAULT_SEQUENCE_GRU_MAX_EPOCHS = 160
DEFAULT_SEQUENCE_GRU_BATCH_SIZE = 16
DEFAULT_SEQUENCE_GRU_VALIDATION_FRACTION = 0.20
DEFAULT_SEQUENCE_GRU_PATIENCE = 24
DEFAULT_SEQUENCE_GRU_OPTUNA_TRIALS = 0
DEFAULT_SEQUENCE_GRU_OPTUNA_MAX_EPOCHS = 70
DEFAULT_SEQUENCE_LSTM_BACKBONE_DIM = 64
DEFAULT_SEQUENCE_LSTM_HIDDEN_DIM = 96
DEFAULT_SEQUENCE_LSTM_LAYERS = 1
DEFAULT_SEQUENCE_LSTM_DROPOUT = 0.25
DEFAULT_SEQUENCE_LSTM_LEARNING_RATE = 1e-3
DEFAULT_SEQUENCE_LSTM_WEIGHT_DECAY = 1e-4
DEFAULT_SEQUENCE_LSTM_MAX_EPOCHS = 180
DEFAULT_SEQUENCE_LSTM_BATCH_SIZE = 16
DEFAULT_SEQUENCE_LSTM_VALIDATION_FRACTION = 0.20
DEFAULT_SEQUENCE_LSTM_PATIENCE = 28
DEFAULT_SEQUENCE_LSTM_OPTUNA_TRIALS = 0
DEFAULT_SEQUENCE_LSTM_OPTUNA_MAX_EPOCHS = 70


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
    include_augmented: bool = False,
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
        sample_files = gesture_sample_paths(
            label_dir,
            include_augmented=bool(include_augmented),
        )
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
    sequence_mlp_alpha: float = DEFAULT_SEQUENCE_MLP_ALPHA,
    sequence_mlp_early_stopping: bool = DEFAULT_SEQUENCE_MLP_EARLY_STOPPING,
    sequence_mlp_validation_fraction: float = DEFAULT_SEQUENCE_MLP_VALIDATION_FRACTION,
    sequence_mlp_n_iter_no_change: int = DEFAULT_SEQUENCE_MLP_N_ITER_NO_CHANGE,
    sequence_rocket_kernels: int = DEFAULT_SEQUENCE_ROCKET_KERNELS,
    sequence_rocket_max_dilation: int = DEFAULT_SEQUENCE_ROCKET_MAX_DILATION,
    sequence_rocket_max_channels_per_kernel: int = (
        DEFAULT_SEQUENCE_ROCKET_MAX_CHANNELS_PER_KERNEL
    ),
    sequence_multirocket_kernels: int = DEFAULT_SEQUENCE_MULTIROCKET_KERNELS,
    sequence_multirocket_max_dilation: int = DEFAULT_SEQUENCE_MULTIROCKET_MAX_DILATION,
    sequence_multirocket_max_channels_per_kernel: int = (
        DEFAULT_SEQUENCE_MULTIROCKET_MAX_CHANNELS_PER_KERNEL
    ),
    sequence_sprocket_kernels: int = DEFAULT_SEQUENCE_SPROCKET_KERNELS,
    sequence_sprocket_prototypes_per_class: int = (
        DEFAULT_SEQUENCE_SPROCKET_PROTOTYPES_PER_CLASS
    ),
    sequence_sprocket_max_dilation: int = DEFAULT_SEQUENCE_SPROCKET_MAX_DILATION,
    sequence_sprocket_max_channels_per_kernel: int = (
        DEFAULT_SEQUENCE_SPROCKET_MAX_CHANNELS_PER_KERNEL
    ),
    sequence_shapelets_per_class: int = DEFAULT_SEQUENCE_SHAPELETS_PER_CLASS,
    sequence_shapelet_max_channels: int = DEFAULT_SEQUENCE_SHAPELET_MAX_CHANNELS,
    sequence_phase_hmm_states: int = DEFAULT_SEQUENCE_PHASE_HMM_STATES,
    sequence_phase_hmm_max_channels: int = DEFAULT_SEQUENCE_PHASE_HMM_MAX_CHANNELS,
    sequence_phase_hmm_variance_regularization: float = (
        DEFAULT_SEQUENCE_PHASE_HMM_VARIANCE_REGULARIZATION
    ),
    sequence_gru_backbone_dim: int = DEFAULT_SEQUENCE_GRU_BACKBONE_DIM,
    sequence_gru_hidden_dim: int = DEFAULT_SEQUENCE_GRU_HIDDEN_DIM,
    sequence_gru_layers: int = DEFAULT_SEQUENCE_GRU_LAYERS,
    sequence_gru_dropout: float = DEFAULT_SEQUENCE_GRU_DROPOUT,
    sequence_gru_bidirectional: bool = False,
    sequence_gru_learning_rate: float = DEFAULT_SEQUENCE_GRU_LEARNING_RATE,
    sequence_gru_weight_decay: float = DEFAULT_SEQUENCE_GRU_WEIGHT_DECAY,
    sequence_gru_max_epochs: int = DEFAULT_SEQUENCE_GRU_MAX_EPOCHS,
    sequence_gru_batch_size: int = DEFAULT_SEQUENCE_GRU_BATCH_SIZE,
    sequence_gru_validation_fraction: float = DEFAULT_SEQUENCE_GRU_VALIDATION_FRACTION,
    sequence_gru_patience: int = DEFAULT_SEQUENCE_GRU_PATIENCE,
    sequence_lstm_backbone_dim: int = DEFAULT_SEQUENCE_LSTM_BACKBONE_DIM,
    sequence_lstm_hidden_dim: int = DEFAULT_SEQUENCE_LSTM_HIDDEN_DIM,
    sequence_lstm_layers: int = DEFAULT_SEQUENCE_LSTM_LAYERS,
    sequence_lstm_dropout: float = DEFAULT_SEQUENCE_LSTM_DROPOUT,
    sequence_lstm_bidirectional: bool = False,
    sequence_lstm_learning_rate: float = DEFAULT_SEQUENCE_LSTM_LEARNING_RATE,
    sequence_lstm_weight_decay: float = DEFAULT_SEQUENCE_LSTM_WEIGHT_DECAY,
    sequence_lstm_max_epochs: int = DEFAULT_SEQUENCE_LSTM_MAX_EPOCHS,
    sequence_lstm_batch_size: int = DEFAULT_SEQUENCE_LSTM_BATCH_SIZE,
    sequence_lstm_validation_fraction: float = DEFAULT_SEQUENCE_LSTM_VALIDATION_FRACTION,
    sequence_lstm_patience: int = DEFAULT_SEQUENCE_LSTM_PATIENCE,
):
    model = str(model_type or "knn").strip().lower()
    if model in {"knn", "sequence_knn"}:
        return KNeighborsClassifier(
            n_neighbors=max(1, int(neighbors)),
            metric="euclidean",
            weights=weights,
        )
    if model == "sequence_mlp":
        validation_fraction = max(
            0.05,
            min(0.50, float(sequence_mlp_validation_fraction)),
        )
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=float(sequence_mlp_alpha),
                learning_rate_init=1e-3,
                max_iter=800,
                early_stopping=bool(sequence_mlp_early_stopping),
                validation_fraction=validation_fraction,
                n_iter_no_change=max(1, int(sequence_mlp_n_iter_no_change)),
                random_state=int(random_state),
            ),
        )
    if model == "sequence_multirocket":
        return make_pipeline(
            RandomMultiRocketSequenceTransformer(
                n_kernels=max(1, int(sequence_multirocket_kernels)),
                max_dilation=max(1, int(sequence_multirocket_max_dilation)),
                max_channels_per_kernel=max(
                    1,
                    int(sequence_multirocket_max_channels_per_kernel),
                ),
                random_state=int(random_state),
            ),
            StandardScaler(),
            LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                random_state=int(random_state),
            ),
        )
    if model == "sequence_sprocket":
        return make_pipeline(
            SprocketSequenceTransformer(
                n_kernels=max(1, int(sequence_sprocket_kernels)),
                prototypes_per_class=max(
                    1,
                    int(sequence_sprocket_prototypes_per_class),
                ),
                max_dilation=max(1, int(sequence_sprocket_max_dilation)),
                max_channels_per_kernel=max(
                    1,
                    int(sequence_sprocket_max_channels_per_kernel),
                ),
                random_state=int(random_state),
            ),
            StandardScaler(),
            LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                random_state=int(random_state),
            ),
        )
    if model == "sequence_shapelet":
        return make_pipeline(
            ShapeletSequenceTransformer(
                shapelets_per_class=max(1, int(sequence_shapelets_per_class)),
                max_channels_per_shapelet=max(
                    1,
                    int(sequence_shapelet_max_channels),
                ),
                random_state=int(random_state),
            ),
            StandardScaler(),
            LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                random_state=int(random_state),
            ),
        )
    if model == "sequence_phase_hmm":
        return PhaseHMMSequenceClassifier(
            n_states=max(2, int(sequence_phase_hmm_states)),
            max_channels=max(1, int(sequence_phase_hmm_max_channels)),
            variance_regularization=float(sequence_phase_hmm_variance_regularization),
            use_uniform_prior=True,
        )
    if model == "sequence_ensemble":
        return VotingClassifier(
            estimators=[
                (
                    "multirocket",
                    make_pipeline(
                        RandomMultiRocketSequenceTransformer(
                            n_kernels=max(1, int(sequence_multirocket_kernels)),
                            max_dilation=max(
                                1,
                                int(sequence_multirocket_max_dilation),
                            ),
                            max_channels_per_kernel=max(
                                1,
                                int(sequence_multirocket_max_channels_per_kernel),
                            ),
                            random_state=int(random_state),
                        ),
                        StandardScaler(),
                        LogisticRegression(
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=int(random_state),
                        ),
                    ),
                ),
                (
                    "sprocket",
                    make_pipeline(
                        SprocketSequenceTransformer(
                            n_kernels=max(1, int(sequence_sprocket_kernels)),
                            prototypes_per_class=max(
                                1,
                                int(sequence_sprocket_prototypes_per_class),
                            ),
                            max_dilation=max(1, int(sequence_sprocket_max_dilation)),
                            max_channels_per_kernel=max(
                                1,
                                int(sequence_sprocket_max_channels_per_kernel),
                            ),
                            random_state=int(random_state),
                        ),
                        StandardScaler(),
                        LogisticRegression(
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=int(random_state),
                        ),
                    ),
                ),
                (
                    "shapelet",
                    make_pipeline(
                        ShapeletSequenceTransformer(
                            shapelets_per_class=max(
                                1,
                                int(sequence_shapelets_per_class),
                            ),
                            max_channels_per_shapelet=max(
                                1,
                                int(sequence_shapelet_max_channels),
                            ),
                            random_state=int(random_state),
                        ),
                        StandardScaler(),
                        LogisticRegression(
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=int(random_state),
                        ),
                    ),
                ),
                (
                    "phase_hmm",
                    PhaseHMMSequenceClassifier(
                        n_states=max(2, int(sequence_phase_hmm_states)),
                        max_channels=max(1, int(sequence_phase_hmm_max_channels)),
                        variance_regularization=float(
                            sequence_phase_hmm_variance_regularization
                        ),
                        use_uniform_prior=True,
                    ),
                ),
            ],
            voting="soft",
            weights=list(DEFAULT_SEQUENCE_ENSEMBLE_WEIGHTS),
        )
    if model == "sequence_gru_backbone":
        return TorchGRUBackboneClassifier(
            backbone_dim=max(4, int(sequence_gru_backbone_dim)),
            hidden_dim=max(4, int(sequence_gru_hidden_dim)),
            num_layers=max(1, int(sequence_gru_layers)),
            dropout=max(0.0, float(sequence_gru_dropout)),
            use_bidirectional=bool(sequence_gru_bidirectional),
            learning_rate=max(1e-6, float(sequence_gru_learning_rate)),
            weight_decay=max(0.0, float(sequence_gru_weight_decay)),
            max_epochs=max(1, int(sequence_gru_max_epochs)),
            batch_size=max(1, int(sequence_gru_batch_size)),
            validation_fraction=max(
                0.0,
                min(0.50, float(sequence_gru_validation_fraction)),
            ),
            patience=max(1, int(sequence_gru_patience)),
            random_state=int(random_state),
        )
    if model == "sequence_lstm_backbone":
        return TorchLSTMBackboneClassifier(
            backbone_dim=max(4, int(sequence_lstm_backbone_dim)),
            hidden_dim=max(4, int(sequence_lstm_hidden_dim)),
            num_layers=max(1, int(sequence_lstm_layers)),
            dropout=max(0.0, float(sequence_lstm_dropout)),
            use_bidirectional=bool(sequence_lstm_bidirectional),
            learning_rate=max(1e-6, float(sequence_lstm_learning_rate)),
            weight_decay=max(0.0, float(sequence_lstm_weight_decay)),
            max_epochs=max(1, int(sequence_lstm_max_epochs)),
            batch_size=max(1, int(sequence_lstm_batch_size)),
            validation_fraction=max(
                0.0,
                min(0.50, float(sequence_lstm_validation_fraction)),
            ),
            patience=max(1, int(sequence_lstm_patience)),
            random_state=int(random_state),
        )
    if model == "sequence_rocket":
        return make_pipeline(
            RandomConvolutionSequenceTransformer(
                n_kernels=max(1, int(sequence_rocket_kernels)),
                max_dilation=max(1, int(sequence_rocket_max_dilation)),
                max_channels_per_kernel=max(
                    1,
                    int(sequence_rocket_max_channels_per_kernel),
                ),
                random_state=int(random_state),
            ),
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
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


def can_use_sequence_mlp_validation_split(
    y: np.ndarray,
    class_count: int,
    validation_fraction: float,
) -> bool:
    """Return whether sklearn can build a stratified validation split."""
    if class_count <= 1 or y.size <= class_count:
        return False
    class_sample_counts = np.bincount(y.astype(np.int64), minlength=class_count)
    if np.any(class_sample_counts < 2):
        return False
    validation_count = int(np.ceil(float(y.size) * float(validation_fraction)))
    train_count = int(y.size) - validation_count
    return validation_count >= class_count and train_count >= class_count


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
        help=(
            "Тип классификатора: knn, sequence_knn, sequence_mlp, "
            "sequence_rocket, sequence_multirocket, sequence_sprocket, "
            "sequence_shapelet, sequence_phase_hmm, sequence_ensemble, "
            "sequence_gru_backbone, sequence_lstm_backbone, "
            "svm, extra_trees, rf или logreg"
        ),
    )
    p.add_argument("--random-state", type=int, default=42, help="Seed для моделей с рандомизацией")
    p.add_argument(
        "--weights",
        choices=["uniform", "distance"],
        default="distance",
        help="Вес соседей KNN: distance устойчивее для маленьких несбалансированных наборов",
    )
    p.add_argument(
        "--sequence-mlp-alpha",
        type=float,
        default=DEFAULT_SEQUENCE_MLP_ALPHA,
        help="L2 regularization strength for sequence_mlp.",
    )
    p.add_argument(
        "--sequence-mlp-early-stopping",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SEQUENCE_MLP_EARLY_STOPPING,
        help="Use an internal validation split and stop sequence_mlp when validation score stops improving.",
    )
    p.add_argument(
        "--sequence-mlp-validation-fraction",
        type=float,
        default=DEFAULT_SEQUENCE_MLP_VALIDATION_FRACTION,
        help="Fraction of training samples held out internally for sequence_mlp early stopping.",
    )
    p.add_argument(
        "--sequence-mlp-n-iter-no-change",
        type=int,
        default=DEFAULT_SEQUENCE_MLP_N_ITER_NO_CHANGE,
        help="Early-stopping patience for sequence_mlp validation score.",
    )
    p.add_argument(
        "--sequence-rocket-kernels",
        type=int,
        default=DEFAULT_SEQUENCE_ROCKET_KERNELS,
        help="Number of random temporal convolution kernels for sequence_rocket.",
    )
    p.add_argument(
        "--sequence-rocket-max-dilation",
        type=int,
        default=DEFAULT_SEQUENCE_ROCKET_MAX_DILATION,
        help="Maximum dilation for sequence_rocket temporal kernels.",
    )
    p.add_argument(
        "--sequence-rocket-max-channels-per-kernel",
        type=int,
        default=DEFAULT_SEQUENCE_ROCKET_MAX_CHANNELS_PER_KERNEL,
        help="Maximum channel subset size per sequence_rocket kernel.",
    )
    p.add_argument(
        "--sequence-multirocket-kernels",
        type=int,
        default=DEFAULT_SEQUENCE_MULTIROCKET_KERNELS,
        help="Number of random temporal convolution kernels for sequence_multirocket.",
    )
    p.add_argument(
        "--sequence-multirocket-max-dilation",
        type=int,
        default=DEFAULT_SEQUENCE_MULTIROCKET_MAX_DILATION,
        help="Maximum dilation for sequence_multirocket temporal kernels.",
    )
    p.add_argument(
        "--sequence-multirocket-max-channels-per-kernel",
        type=int,
        default=DEFAULT_SEQUENCE_MULTIROCKET_MAX_CHANNELS_PER_KERNEL,
        help="Maximum channel subset size per sequence_multirocket kernel.",
    )
    p.add_argument(
        "--sequence-sprocket-kernels",
        type=int,
        default=DEFAULT_SEQUENCE_SPROCKET_KERNELS,
        help="Number of random temporal convolution kernels for sequence_sprocket.",
    )
    p.add_argument(
        "--sequence-sprocket-prototypes-per-class",
        type=int,
        default=DEFAULT_SEQUENCE_SPROCKET_PROTOTYPES_PER_CLASS,
        help="Number of representative prototypes per class for sequence_sprocket.",
    )
    p.add_argument(
        "--sequence-sprocket-max-dilation",
        type=int,
        default=DEFAULT_SEQUENCE_SPROCKET_MAX_DILATION,
        help="Maximum dilation for sequence_sprocket temporal kernels.",
    )
    p.add_argument(
        "--sequence-sprocket-max-channels-per-kernel",
        type=int,
        default=DEFAULT_SEQUENCE_SPROCKET_MAX_CHANNELS_PER_KERNEL,
        help="Maximum channel subset size per sequence_sprocket kernel.",
    )
    p.add_argument(
        "--sequence-shapelets-per-class",
        type=int,
        default=DEFAULT_SEQUENCE_SHAPELETS_PER_CLASS,
        help="Number of training-derived shapelets per class.",
    )
    p.add_argument(
        "--sequence-shapelet-max-channels",
        type=int,
        default=DEFAULT_SEQUENCE_SHAPELET_MAX_CHANNELS,
        help="Maximum channel subset size per shapelet.",
    )
    p.add_argument(
        "--sequence-phase-hmm-states",
        type=int,
        default=DEFAULT_SEQUENCE_PHASE_HMM_STATES,
        help="Number of ordered phases for sequence_phase_hmm.",
    )
    p.add_argument(
        "--sequence-phase-hmm-max-channels",
        type=int,
        default=DEFAULT_SEQUENCE_PHASE_HMM_MAX_CHANNELS,
        help="Maximum landmark channels used by sequence_phase_hmm.",
    )
    p.add_argument(
        "--sequence-phase-hmm-variance-regularization",
        type=float,
        default=DEFAULT_SEQUENCE_PHASE_HMM_VARIANCE_REGULARIZATION,
        help="Blend factor between per-class and global variance.",
    )
    p.add_argument(
        "--sequence-gru-backbone-dim",
        type=int,
        default=DEFAULT_SEQUENCE_GRU_BACKBONE_DIM,
        help="Per-frame MLP embedding size for sequence_gru_backbone.",
    )
    p.add_argument(
        "--sequence-gru-hidden-dim",
        type=int,
        default=DEFAULT_SEQUENCE_GRU_HIDDEN_DIM,
        help="GRU hidden state size for sequence_gru_backbone.",
    )
    p.add_argument(
        "--sequence-gru-layers",
        type=int,
        default=DEFAULT_SEQUENCE_GRU_LAYERS,
        help="Number of recurrent GRU layers.",
    )
    p.add_argument(
        "--sequence-gru-dropout",
        type=float,
        default=DEFAULT_SEQUENCE_GRU_DROPOUT,
        help="Dropout for sequence_gru_backbone.",
    )
    p.add_argument(
        "--sequence-gru-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a bidirectional GRU for offline/live classification.",
    )
    p.add_argument(
        "--sequence-gru-learning-rate",
        type=float,
        default=DEFAULT_SEQUENCE_GRU_LEARNING_RATE,
        help="AdamW learning rate for sequence_gru_backbone.",
    )
    p.add_argument(
        "--sequence-gru-weight-decay",
        type=float,
        default=DEFAULT_SEQUENCE_GRU_WEIGHT_DECAY,
        help="AdamW weight decay for sequence_gru_backbone.",
    )
    p.add_argument(
        "--sequence-gru-max-epochs",
        type=int,
        default=DEFAULT_SEQUENCE_GRU_MAX_EPOCHS,
        help="Maximum epochs for sequence_gru_backbone.",
    )
    p.add_argument(
        "--sequence-gru-batch-size",
        type=int,
        default=DEFAULT_SEQUENCE_GRU_BATCH_SIZE,
        help="Mini-batch size for sequence_gru_backbone.",
    )
    p.add_argument(
        "--sequence-gru-validation-fraction",
        type=float,
        default=DEFAULT_SEQUENCE_GRU_VALIDATION_FRACTION,
        help="Internal validation split for sequence_gru_backbone early stopping.",
    )
    p.add_argument(
        "--sequence-gru-patience",
        type=int,
        default=DEFAULT_SEQUENCE_GRU_PATIENCE,
        help="Early-stopping patience for sequence_gru_backbone.",
    )
    p.add_argument(
        "--sequence-gru-optuna-trials",
        type=int,
        default=DEFAULT_SEQUENCE_GRU_OPTUNA_TRIALS,
        help="Run Optuna tuning before final sequence_gru_backbone training.",
    )
    p.add_argument(
        "--sequence-gru-optuna-max-epochs",
        type=int,
        default=DEFAULT_SEQUENCE_GRU_OPTUNA_MAX_EPOCHS,
        help="Epoch budget per Optuna trial for sequence_gru_backbone.",
    )
    p.add_argument(
        "--sequence-gru-optuna-timeout",
        type=int,
        default=0,
        help="Optional Optuna timeout in seconds; 0 means no timeout.",
    )
    p.add_argument(
        "--sequence-lstm-backbone-dim",
        type=int,
        default=DEFAULT_SEQUENCE_LSTM_BACKBONE_DIM,
        help="Per-frame MLP embedding size for sequence_lstm_backbone.",
    )
    p.add_argument(
        "--sequence-lstm-hidden-dim",
        type=int,
        default=DEFAULT_SEQUENCE_LSTM_HIDDEN_DIM,
        help="LSTM hidden state size for sequence_lstm_backbone.",
    )
    p.add_argument(
        "--sequence-lstm-layers",
        type=int,
        default=DEFAULT_SEQUENCE_LSTM_LAYERS,
        help="Number of recurrent LSTM layers.",
    )
    p.add_argument(
        "--sequence-lstm-dropout",
        type=float,
        default=DEFAULT_SEQUENCE_LSTM_DROPOUT,
        help="Dropout for sequence_lstm_backbone.",
    )
    p.add_argument(
        "--sequence-lstm-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a bidirectional LSTM for offline/live classification.",
    )
    p.add_argument(
        "--sequence-lstm-learning-rate",
        type=float,
        default=DEFAULT_SEQUENCE_LSTM_LEARNING_RATE,
        help="AdamW learning rate for sequence_lstm_backbone.",
    )
    p.add_argument(
        "--sequence-lstm-weight-decay",
        type=float,
        default=DEFAULT_SEQUENCE_LSTM_WEIGHT_DECAY,
        help="AdamW weight decay for sequence_lstm_backbone.",
    )
    p.add_argument(
        "--sequence-lstm-max-epochs",
        type=int,
        default=DEFAULT_SEQUENCE_LSTM_MAX_EPOCHS,
        help="Maximum epochs for sequence_lstm_backbone.",
    )
    p.add_argument(
        "--sequence-lstm-batch-size",
        type=int,
        default=DEFAULT_SEQUENCE_LSTM_BATCH_SIZE,
        help="Mini-batch size for sequence_lstm_backbone.",
    )
    p.add_argument(
        "--sequence-lstm-validation-fraction",
        type=float,
        default=DEFAULT_SEQUENCE_LSTM_VALIDATION_FRACTION,
        help="Internal validation split for sequence_lstm_backbone early stopping.",
    )
    p.add_argument(
        "--sequence-lstm-patience",
        type=int,
        default=DEFAULT_SEQUENCE_LSTM_PATIENCE,
        help="Early-stopping patience for sequence_lstm_backbone.",
    )
    p.add_argument(
        "--sequence-lstm-optuna-trials",
        type=int,
        default=DEFAULT_SEQUENCE_LSTM_OPTUNA_TRIALS,
        help="Run Optuna tuning before final sequence_lstm_backbone training.",
    )
    p.add_argument(
        "--sequence-lstm-optuna-max-epochs",
        type=int,
        default=DEFAULT_SEQUENCE_LSTM_OPTUNA_MAX_EPOCHS,
        help="Epoch budget per Optuna trial for sequence_lstm_backbone.",
    )
    p.add_argument(
        "--sequence-lstm-optuna-timeout",
        type=int,
        default=0,
        help="Optional LSTM Optuna timeout in seconds; 0 means no timeout.",
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
        "--include-augmented",
        action="store_true",
        help="Включить aug_sample_* в обучение; по умолчанию используется только camera baseline.",
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
        include_augmented=bool(args.include_augmented),
    )
    print(
        f"[i] Загружено семплов: {len(X)}; классов: {len(classes)}; "
        f"режим признаков: {args.feature_mode}; размер признака: {X.shape[1]}; "
        f"модель: {args.model_type}"
    )

    sequence_mlp_validation_fraction = max(
        0.05,
        min(0.50, float(args.sequence_mlp_validation_fraction)),
    )
    sequence_mlp_early_stopping = bool(args.sequence_mlp_early_stopping)
    if str(args.model_type).strip().lower() == "sequence_mlp":
        if sequence_mlp_validation_fraction != float(args.sequence_mlp_validation_fraction):
            print(
                "[w] sequence_mlp validation_fraction скорректирован до "
                f"{sequence_mlp_validation_fraction:.2f}"
            )
        if sequence_mlp_early_stopping and not can_use_sequence_mlp_validation_split(
            y,
            class_count=len(classes),
            validation_fraction=sequence_mlp_validation_fraction,
        ):
            sequence_mlp_early_stopping = False
            print(
                "[w] sequence_mlp early_stopping отключен: слишком мало "
                "сэмплов для stratified validation split."
            )
    args.sequence_mlp_validation_fraction_effective = sequence_mlp_validation_fraction
    args.sequence_mlp_early_stopping_effective = sequence_mlp_early_stopping

    sequence_gru_params = {
        "sequence_gru_backbone_dim": int(args.sequence_gru_backbone_dim),
        "sequence_gru_hidden_dim": int(args.sequence_gru_hidden_dim),
        "sequence_gru_layers": int(args.sequence_gru_layers),
        "sequence_gru_dropout": float(args.sequence_gru_dropout),
        "sequence_gru_bidirectional": bool(args.sequence_gru_bidirectional),
        "sequence_gru_learning_rate": float(args.sequence_gru_learning_rate),
        "sequence_gru_weight_decay": float(args.sequence_gru_weight_decay),
        "sequence_gru_max_epochs": int(args.sequence_gru_max_epochs),
        "sequence_gru_batch_size": int(args.sequence_gru_batch_size),
        "sequence_gru_validation_fraction": float(args.sequence_gru_validation_fraction),
        "sequence_gru_patience": int(args.sequence_gru_patience),
    }
    args.sequence_gru_optuna_summary = None
    args.sequence_gru_optuna_out = ""
    if (
        str(args.model_type).strip().lower() == "sequence_gru_backbone"
        and int(args.sequence_gru_optuna_trials) > 0
    ):
        trials = max(1, int(args.sequence_gru_optuna_trials))
        print(f"[i] Optuna tuning для sequence_gru_backbone: trials={trials}")
        tuning = tune_gru_backbone_hyperparameters(
            X,
            y,
            target_frames=36,
            n_trials=trials,
            timeout=int(args.sequence_gru_optuna_timeout) or None,
            random_state=int(args.random_state),
            max_epochs=max(10, int(args.sequence_gru_optuna_max_epochs)),
        )
        best_params = dict(tuning.best_params)
        sequence_gru_params.update(
            {
                "sequence_gru_backbone_dim": int(
                    best_params.get(
                        "backbone_dim",
                        sequence_gru_params["sequence_gru_backbone_dim"],
                    )
                ),
                "sequence_gru_hidden_dim": int(
                    best_params.get(
                        "hidden_dim",
                        sequence_gru_params["sequence_gru_hidden_dim"],
                    )
                ),
                "sequence_gru_layers": int(
                    best_params.get(
                        "num_layers",
                        sequence_gru_params["sequence_gru_layers"],
                    )
                ),
                "sequence_gru_dropout": float(
                    best_params.get(
                        "dropout",
                        sequence_gru_params["sequence_gru_dropout"],
                    )
                ),
                "sequence_gru_bidirectional": bool(
                    best_params.get(
                        "use_bidirectional",
                        sequence_gru_params["sequence_gru_bidirectional"],
                    )
                ),
                "sequence_gru_learning_rate": float(
                    best_params.get(
                        "learning_rate",
                        sequence_gru_params["sequence_gru_learning_rate"],
                    )
                ),
                "sequence_gru_weight_decay": float(
                    best_params.get(
                        "weight_decay",
                        sequence_gru_params["sequence_gru_weight_decay"],
                    )
                ),
                "sequence_gru_batch_size": int(
                    best_params.get(
                        "batch_size",
                        sequence_gru_params["sequence_gru_batch_size"],
                    )
                ),
            }
        )
        args.sequence_gru_optuna_summary = {
            "best_score": float(tuning.best_score),
            "best_params": best_params,
            "trials": int(tuning.trials),
            "used_validation_split": bool(tuning.used_validation_split),
        }
        args.sequence_gru_optuna_out = str(
            out_path.with_name(f"{out_path.stem}_optuna.json")
        )
        Path(args.sequence_gru_optuna_out).write_text(
            json.dumps(args.sequence_gru_optuna_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "[i] Optuna best: "
            f"score={tuning.best_score:.4f}, params={json.dumps(best_params)}"
        )
    for key, value in sequence_gru_params.items():
        setattr(args, f"{key}_effective", value)

    sequence_lstm_params = {
        "sequence_lstm_backbone_dim": int(args.sequence_lstm_backbone_dim),
        "sequence_lstm_hidden_dim": int(args.sequence_lstm_hidden_dim),
        "sequence_lstm_layers": int(args.sequence_lstm_layers),
        "sequence_lstm_dropout": float(args.sequence_lstm_dropout),
        "sequence_lstm_bidirectional": bool(args.sequence_lstm_bidirectional),
        "sequence_lstm_learning_rate": float(args.sequence_lstm_learning_rate),
        "sequence_lstm_weight_decay": float(args.sequence_lstm_weight_decay),
        "sequence_lstm_max_epochs": int(args.sequence_lstm_max_epochs),
        "sequence_lstm_batch_size": int(args.sequence_lstm_batch_size),
        "sequence_lstm_validation_fraction": float(args.sequence_lstm_validation_fraction),
        "sequence_lstm_patience": int(args.sequence_lstm_patience),
    }
    args.sequence_lstm_optuna_summary = None
    args.sequence_lstm_optuna_out = ""
    if (
        str(args.model_type).strip().lower() == "sequence_lstm_backbone"
        and int(args.sequence_lstm_optuna_trials) > 0
    ):
        trials = max(1, int(args.sequence_lstm_optuna_trials))
        print(f"[i] Optuna tuning для sequence_lstm_backbone: trials={trials}")
        tuning = tune_lstm_backbone_hyperparameters(
            X,
            y,
            target_frames=36,
            n_trials=trials,
            timeout=int(args.sequence_lstm_optuna_timeout) or None,
            random_state=int(args.random_state),
            max_epochs=max(10, int(args.sequence_lstm_optuna_max_epochs)),
        )
        best_params = dict(tuning.best_params)
        sequence_lstm_params.update(
            {
                "sequence_lstm_backbone_dim": int(
                    best_params.get(
                        "backbone_dim",
                        sequence_lstm_params["sequence_lstm_backbone_dim"],
                    )
                ),
                "sequence_lstm_hidden_dim": int(
                    best_params.get(
                        "hidden_dim",
                        sequence_lstm_params["sequence_lstm_hidden_dim"],
                    )
                ),
                "sequence_lstm_layers": int(
                    best_params.get(
                        "num_layers",
                        sequence_lstm_params["sequence_lstm_layers"],
                    )
                ),
                "sequence_lstm_dropout": float(
                    best_params.get(
                        "dropout",
                        sequence_lstm_params["sequence_lstm_dropout"],
                    )
                ),
                "sequence_lstm_bidirectional": bool(
                    best_params.get(
                        "use_bidirectional",
                        sequence_lstm_params["sequence_lstm_bidirectional"],
                    )
                ),
                "sequence_lstm_learning_rate": float(
                    best_params.get(
                        "learning_rate",
                        sequence_lstm_params["sequence_lstm_learning_rate"],
                    )
                ),
                "sequence_lstm_weight_decay": float(
                    best_params.get(
                        "weight_decay",
                        sequence_lstm_params["sequence_lstm_weight_decay"],
                    )
                ),
                "sequence_lstm_batch_size": int(
                    best_params.get(
                        "batch_size",
                        sequence_lstm_params["sequence_lstm_batch_size"],
                    )
                ),
            }
        )
        args.sequence_lstm_optuna_summary = {
            "best_score": float(tuning.best_score),
            "best_params": best_params,
            "trials": int(tuning.trials),
            "used_validation_split": bool(tuning.used_validation_split),
        }
        args.sequence_lstm_optuna_out = str(
            out_path.with_name(f"{out_path.stem}_optuna.json")
        )
        Path(args.sequence_lstm_optuna_out).write_text(
            json.dumps(args.sequence_lstm_optuna_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "[i] Optuna best: "
            f"score={tuning.best_score:.4f}, params={json.dumps(best_params)}"
        )
    for key, value in sequence_lstm_params.items():
        setattr(args, f"{key}_effective", value)

    clf = build_classifier(
        str(args.model_type),
        neighbors=int(args.neighbors),
        weights=str(args.weights),
        random_state=int(args.random_state),
        sequence_mlp_alpha=float(args.sequence_mlp_alpha),
        sequence_mlp_early_stopping=sequence_mlp_early_stopping,
        sequence_mlp_validation_fraction=sequence_mlp_validation_fraction,
        sequence_mlp_n_iter_no_change=int(args.sequence_mlp_n_iter_no_change),
        sequence_rocket_kernels=int(args.sequence_rocket_kernels),
        sequence_rocket_max_dilation=int(args.sequence_rocket_max_dilation),
        sequence_rocket_max_channels_per_kernel=int(
            args.sequence_rocket_max_channels_per_kernel
        ),
        sequence_multirocket_kernels=int(args.sequence_multirocket_kernels),
        sequence_multirocket_max_dilation=int(args.sequence_multirocket_max_dilation),
        sequence_multirocket_max_channels_per_kernel=int(
            args.sequence_multirocket_max_channels_per_kernel
        ),
        sequence_sprocket_kernels=int(args.sequence_sprocket_kernels),
        sequence_sprocket_prototypes_per_class=int(
            args.sequence_sprocket_prototypes_per_class
        ),
        sequence_sprocket_max_dilation=int(args.sequence_sprocket_max_dilation),
        sequence_sprocket_max_channels_per_kernel=int(
            args.sequence_sprocket_max_channels_per_kernel
        ),
        sequence_shapelets_per_class=int(args.sequence_shapelets_per_class),
        sequence_shapelet_max_channels=int(args.sequence_shapelet_max_channels),
        sequence_phase_hmm_states=int(args.sequence_phase_hmm_states),
        sequence_phase_hmm_max_channels=int(args.sequence_phase_hmm_max_channels),
        sequence_phase_hmm_variance_regularization=float(
            args.sequence_phase_hmm_variance_regularization
        ),
        **sequence_gru_params,
        **sequence_lstm_params,
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
        sequence_mlp_alpha = float(
            getattr(args, "sequence_mlp_alpha", DEFAULT_SEQUENCE_MLP_ALPHA)
        )
        sequence_mlp_early_stopping = bool(
            getattr(
                args,
                "sequence_mlp_early_stopping",
                DEFAULT_SEQUENCE_MLP_EARLY_STOPPING,
            )
        )
        sequence_mlp_early_stopping_effective = bool(
            getattr(
                args,
                "sequence_mlp_early_stopping_effective",
                sequence_mlp_early_stopping,
            )
        )
        sequence_mlp_validation_fraction = float(
            getattr(
                args,
                "sequence_mlp_validation_fraction",
                DEFAULT_SEQUENCE_MLP_VALIDATION_FRACTION,
            )
        )
        sequence_mlp_validation_fraction_effective = float(
            getattr(
                args,
                "sequence_mlp_validation_fraction_effective",
                sequence_mlp_validation_fraction,
            )
        )
        sequence_mlp_n_iter_no_change = int(
            getattr(
                args,
                "sequence_mlp_n_iter_no_change",
                DEFAULT_SEQUENCE_MLP_N_ITER_NO_CHANGE,
            )
        )
        sequence_rocket_kernels = int(
            getattr(
                args,
                "sequence_rocket_kernels",
                DEFAULT_SEQUENCE_ROCKET_KERNELS,
            )
        )
        sequence_rocket_max_dilation = int(
            getattr(
                args,
                "sequence_rocket_max_dilation",
                DEFAULT_SEQUENCE_ROCKET_MAX_DILATION,
            )
        )
        sequence_rocket_max_channels_per_kernel = int(
            getattr(
                args,
                "sequence_rocket_max_channels_per_kernel",
                DEFAULT_SEQUENCE_ROCKET_MAX_CHANNELS_PER_KERNEL,
            )
        )
        sequence_multirocket_kernels = int(
            getattr(
                args,
                "sequence_multirocket_kernels",
                DEFAULT_SEQUENCE_MULTIROCKET_KERNELS,
            )
        )
        sequence_multirocket_max_dilation = int(
            getattr(
                args,
                "sequence_multirocket_max_dilation",
                DEFAULT_SEQUENCE_MULTIROCKET_MAX_DILATION,
            )
        )
        sequence_multirocket_max_channels_per_kernel = int(
            getattr(
                args,
                "sequence_multirocket_max_channels_per_kernel",
                DEFAULT_SEQUENCE_MULTIROCKET_MAX_CHANNELS_PER_KERNEL,
            )
        )
        sequence_sprocket_kernels = int(
            getattr(
                args,
                "sequence_sprocket_kernels",
                DEFAULT_SEQUENCE_SPROCKET_KERNELS,
            )
        )
        sequence_sprocket_prototypes_per_class = int(
            getattr(
                args,
                "sequence_sprocket_prototypes_per_class",
                DEFAULT_SEQUENCE_SPROCKET_PROTOTYPES_PER_CLASS,
            )
        )
        sequence_sprocket_max_dilation = int(
            getattr(
                args,
                "sequence_sprocket_max_dilation",
                DEFAULT_SEQUENCE_SPROCKET_MAX_DILATION,
            )
        )
        sequence_sprocket_max_channels_per_kernel = int(
            getattr(
                args,
                "sequence_sprocket_max_channels_per_kernel",
                DEFAULT_SEQUENCE_SPROCKET_MAX_CHANNELS_PER_KERNEL,
            )
        )
        sequence_shapelets_per_class = int(
            getattr(
                args,
                "sequence_shapelets_per_class",
                DEFAULT_SEQUENCE_SHAPELETS_PER_CLASS,
            )
        )
        sequence_shapelet_max_channels = int(
            getattr(
                args,
                "sequence_shapelet_max_channels",
                DEFAULT_SEQUENCE_SHAPELET_MAX_CHANNELS,
            )
        )
        sequence_phase_hmm_states = int(
            getattr(
                args,
                "sequence_phase_hmm_states",
                DEFAULT_SEQUENCE_PHASE_HMM_STATES,
            )
        )
        sequence_phase_hmm_max_channels = int(
            getattr(
                args,
                "sequence_phase_hmm_max_channels",
                DEFAULT_SEQUENCE_PHASE_HMM_MAX_CHANNELS,
            )
        )
        sequence_phase_hmm_variance_regularization = float(
            getattr(
                args,
                "sequence_phase_hmm_variance_regularization",
                DEFAULT_SEQUENCE_PHASE_HMM_VARIANCE_REGULARIZATION,
            )
        )
        sequence_gru_backbone_dim = int(
            getattr(args, "sequence_gru_backbone_dim", DEFAULT_SEQUENCE_GRU_BACKBONE_DIM)
        )
        sequence_gru_hidden_dim = int(
            getattr(args, "sequence_gru_hidden_dim", DEFAULT_SEQUENCE_GRU_HIDDEN_DIM)
        )
        sequence_gru_layers = int(
            getattr(args, "sequence_gru_layers", DEFAULT_SEQUENCE_GRU_LAYERS)
        )
        sequence_gru_dropout = float(
            getattr(args, "sequence_gru_dropout", DEFAULT_SEQUENCE_GRU_DROPOUT)
        )
        sequence_gru_bidirectional = bool(
            getattr(args, "sequence_gru_bidirectional", False)
        )
        sequence_gru_learning_rate = float(
            getattr(
                args,
                "sequence_gru_learning_rate",
                DEFAULT_SEQUENCE_GRU_LEARNING_RATE,
            )
        )
        sequence_gru_weight_decay = float(
            getattr(
                args,
                "sequence_gru_weight_decay",
                DEFAULT_SEQUENCE_GRU_WEIGHT_DECAY,
            )
        )
        sequence_gru_max_epochs = int(
            getattr(args, "sequence_gru_max_epochs", DEFAULT_SEQUENCE_GRU_MAX_EPOCHS)
        )
        sequence_gru_batch_size = int(
            getattr(args, "sequence_gru_batch_size", DEFAULT_SEQUENCE_GRU_BATCH_SIZE)
        )
        sequence_gru_validation_fraction = float(
            getattr(
                args,
                "sequence_gru_validation_fraction",
                DEFAULT_SEQUENCE_GRU_VALIDATION_FRACTION,
            )
        )
        sequence_gru_patience = int(
            getattr(args, "sequence_gru_patience", DEFAULT_SEQUENCE_GRU_PATIENCE)
        )
        sequence_gru_effective = {
            "sequence_gru_backbone_dim_effective": int(
                getattr(
                    args,
                    "sequence_gru_backbone_dim_effective",
                    sequence_gru_backbone_dim,
                )
            ),
            "sequence_gru_hidden_dim_effective": int(
                getattr(
                    args,
                    "sequence_gru_hidden_dim_effective",
                    sequence_gru_hidden_dim,
                )
            ),
            "sequence_gru_layers_effective": int(
                getattr(args, "sequence_gru_layers_effective", sequence_gru_layers)
            ),
            "sequence_gru_dropout_effective": float(
                getattr(args, "sequence_gru_dropout_effective", sequence_gru_dropout)
            ),
            "sequence_gru_bidirectional_effective": bool(
                getattr(
                    args,
                    "sequence_gru_bidirectional_effective",
                    sequence_gru_bidirectional,
                )
            ),
            "sequence_gru_learning_rate_effective": float(
                getattr(
                    args,
                    "sequence_gru_learning_rate_effective",
                    sequence_gru_learning_rate,
                )
            ),
            "sequence_gru_weight_decay_effective": float(
                getattr(
                    args,
                    "sequence_gru_weight_decay_effective",
                    sequence_gru_weight_decay,
                )
            ),
            "sequence_gru_batch_size_effective": int(
                getattr(
                    args,
                    "sequence_gru_batch_size_effective",
                    sequence_gru_batch_size,
                )
            ),
        }
        sequence_gru_optuna_trials = int(
            getattr(args, "sequence_gru_optuna_trials", DEFAULT_SEQUENCE_GRU_OPTUNA_TRIALS)
        )
        sequence_gru_optuna_summary = getattr(args, "sequence_gru_optuna_summary", None)
        sequence_lstm_backbone_dim = int(
            getattr(args, "sequence_lstm_backbone_dim", DEFAULT_SEQUENCE_LSTM_BACKBONE_DIM)
        )
        sequence_lstm_hidden_dim = int(
            getattr(args, "sequence_lstm_hidden_dim", DEFAULT_SEQUENCE_LSTM_HIDDEN_DIM)
        )
        sequence_lstm_layers = int(
            getattr(args, "sequence_lstm_layers", DEFAULT_SEQUENCE_LSTM_LAYERS)
        )
        sequence_lstm_dropout = float(
            getattr(args, "sequence_lstm_dropout", DEFAULT_SEQUENCE_LSTM_DROPOUT)
        )
        sequence_lstm_bidirectional = bool(
            getattr(args, "sequence_lstm_bidirectional", False)
        )
        sequence_lstm_learning_rate = float(
            getattr(
                args,
                "sequence_lstm_learning_rate",
                DEFAULT_SEQUENCE_LSTM_LEARNING_RATE,
            )
        )
        sequence_lstm_weight_decay = float(
            getattr(
                args,
                "sequence_lstm_weight_decay",
                DEFAULT_SEQUENCE_LSTM_WEIGHT_DECAY,
            )
        )
        sequence_lstm_max_epochs = int(
            getattr(args, "sequence_lstm_max_epochs", DEFAULT_SEQUENCE_LSTM_MAX_EPOCHS)
        )
        sequence_lstm_batch_size = int(
            getattr(args, "sequence_lstm_batch_size", DEFAULT_SEQUENCE_LSTM_BATCH_SIZE)
        )
        sequence_lstm_validation_fraction = float(
            getattr(
                args,
                "sequence_lstm_validation_fraction",
                DEFAULT_SEQUENCE_LSTM_VALIDATION_FRACTION,
            )
        )
        sequence_lstm_patience = int(
            getattr(args, "sequence_lstm_patience", DEFAULT_SEQUENCE_LSTM_PATIENCE)
        )
        sequence_lstm_effective = {
            "sequence_lstm_backbone_dim_effective": int(
                getattr(
                    args,
                    "sequence_lstm_backbone_dim_effective",
                    sequence_lstm_backbone_dim,
                )
            ),
            "sequence_lstm_hidden_dim_effective": int(
                getattr(
                    args,
                    "sequence_lstm_hidden_dim_effective",
                    sequence_lstm_hidden_dim,
                )
            ),
            "sequence_lstm_layers_effective": int(
                getattr(args, "sequence_lstm_layers_effective", sequence_lstm_layers)
            ),
            "sequence_lstm_dropout_effective": float(
                getattr(args, "sequence_lstm_dropout_effective", sequence_lstm_dropout)
            ),
            "sequence_lstm_bidirectional_effective": bool(
                getattr(
                    args,
                    "sequence_lstm_bidirectional_effective",
                    sequence_lstm_bidirectional,
                )
            ),
            "sequence_lstm_learning_rate_effective": float(
                getattr(
                    args,
                    "sequence_lstm_learning_rate_effective",
                    sequence_lstm_learning_rate,
                )
            ),
            "sequence_lstm_weight_decay_effective": float(
                getattr(
                    args,
                    "sequence_lstm_weight_decay_effective",
                    sequence_lstm_weight_decay,
                )
            ),
            "sequence_lstm_batch_size_effective": int(
                getattr(
                    args,
                    "sequence_lstm_batch_size_effective",
                    sequence_lstm_batch_size,
                )
            ),
        }
        sequence_lstm_optuna_trials = int(
            getattr(
                args,
                "sequence_lstm_optuna_trials",
                DEFAULT_SEQUENCE_LSTM_OPTUNA_TRIALS,
            )
        )
        sequence_lstm_optuna_summary = getattr(args, "sequence_lstm_optuna_summary", None)
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
                    "random_state": int(getattr(args, "random_state", 42)),
                    "sequence_mlp_alpha": sequence_mlp_alpha,
                    "sequence_mlp_early_stopping": sequence_mlp_early_stopping,
                    "sequence_mlp_early_stopping_effective": (
                        sequence_mlp_early_stopping_effective
                    ),
                    "sequence_mlp_validation_fraction": (
                        sequence_mlp_validation_fraction
                    ),
                    "sequence_mlp_validation_fraction_effective": (
                        sequence_mlp_validation_fraction_effective
                    ),
                    "sequence_mlp_n_iter_no_change": sequence_mlp_n_iter_no_change,
                    "sequence_rocket_kernels": sequence_rocket_kernels,
                    "sequence_rocket_max_dilation": sequence_rocket_max_dilation,
                    "sequence_rocket_max_channels_per_kernel": (
                        sequence_rocket_max_channels_per_kernel
                    ),
                    "sequence_multirocket_kernels": sequence_multirocket_kernels,
                    "sequence_multirocket_max_dilation": (
                        sequence_multirocket_max_dilation
                    ),
                    "sequence_multirocket_max_channels_per_kernel": (
                        sequence_multirocket_max_channels_per_kernel
                    ),
                    "sequence_sprocket_kernels": sequence_sprocket_kernels,
                    "sequence_sprocket_prototypes_per_class": (
                        sequence_sprocket_prototypes_per_class
                    ),
                    "sequence_sprocket_max_dilation": sequence_sprocket_max_dilation,
                    "sequence_sprocket_max_channels_per_kernel": (
                        sequence_sprocket_max_channels_per_kernel
                    ),
                    "sequence_shapelets_per_class": sequence_shapelets_per_class,
                    "sequence_shapelet_max_channels": sequence_shapelet_max_channels,
                    "sequence_phase_hmm_states": sequence_phase_hmm_states,
                    "sequence_phase_hmm_max_channels": sequence_phase_hmm_max_channels,
                    "sequence_phase_hmm_variance_regularization": (
                        sequence_phase_hmm_variance_regularization
                    ),
                    "sequence_gru_backbone_dim": sequence_gru_backbone_dim,
                    "sequence_gru_hidden_dim": sequence_gru_hidden_dim,
                    "sequence_gru_layers": sequence_gru_layers,
                    "sequence_gru_dropout": sequence_gru_dropout,
                    "sequence_gru_bidirectional": sequence_gru_bidirectional,
                    "sequence_gru_learning_rate": sequence_gru_learning_rate,
                    "sequence_gru_weight_decay": sequence_gru_weight_decay,
                    "sequence_gru_max_epochs": sequence_gru_max_epochs,
                    "sequence_gru_batch_size": sequence_gru_batch_size,
                    "sequence_gru_validation_fraction": (
                        sequence_gru_validation_fraction
                    ),
                    "sequence_gru_patience": sequence_gru_patience,
                    "sequence_gru_optuna_trials": sequence_gru_optuna_trials,
                    **sequence_gru_effective,
                    "sequence_lstm_backbone_dim": sequence_lstm_backbone_dim,
                    "sequence_lstm_hidden_dim": sequence_lstm_hidden_dim,
                    "sequence_lstm_layers": sequence_lstm_layers,
                    "sequence_lstm_dropout": sequence_lstm_dropout,
                    "sequence_lstm_bidirectional": sequence_lstm_bidirectional,
                    "sequence_lstm_learning_rate": sequence_lstm_learning_rate,
                    "sequence_lstm_weight_decay": sequence_lstm_weight_decay,
                    "sequence_lstm_max_epochs": sequence_lstm_max_epochs,
                    "sequence_lstm_batch_size": sequence_lstm_batch_size,
                    "sequence_lstm_validation_fraction": (
                        sequence_lstm_validation_fraction
                    ),
                    "sequence_lstm_patience": sequence_lstm_patience,
                    "sequence_lstm_optuna_trials": sequence_lstm_optuna_trials,
                    **sequence_lstm_effective,
                    "sequence_ensemble_members": (
                        "multirocket,sprocket,shapelet,phase_hmm"
                    ),
                    "sequence_ensemble_weights": ",".join(
                        f"{weight:.2f}" for weight in DEFAULT_SEQUENCE_ENSEMBLE_WEIGHTS
                    ),
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
            metrics = {
                "sample_count": float(sample_count),
                "class_count": float(len(classes)),
                "feature_dim": float(feature_dim),
                "train_accuracy": float(train_accuracy),
                "negative_class_count": float(len(negative_labels)),
            }
            if isinstance(sequence_gru_optuna_summary, dict):
                metrics["sequence_gru_optuna_best_score"] = float(
                    sequence_gru_optuna_summary.get("best_score", 0.0)
                )
                metrics["sequence_gru_optuna_trials_done"] = float(
                    sequence_gru_optuna_summary.get("trials", 0)
                )
            if isinstance(sequence_lstm_optuna_summary, dict):
                metrics["sequence_lstm_optuna_best_score"] = float(
                    sequence_lstm_optuna_summary.get("best_score", 0.0)
                )
                metrics["sequence_lstm_optuna_trials_done"] = float(
                    sequence_lstm_optuna_summary.get("trials", 0)
                )
            mlflow.log_metrics(metrics)
            for artifact in (
                out_path,
                classes_out,
                feature_dim_out,
                feature_mode_out,
                rejection_out,
            ):
                if artifact.exists():
                    mlflow.log_artifact(str(artifact))
            optuna_out_value = str(getattr(args, "sequence_gru_optuna_out", "") or "").strip()
            if optuna_out_value:
                optuna_out = Path(optuna_out_value)
                if optuna_out.exists():
                    mlflow.log_artifact(str(optuna_out))
            lstm_optuna_out_value = str(
                getattr(args, "sequence_lstm_optuna_out", "") or ""
            ).strip()
            if lstm_optuna_out_value:
                lstm_optuna_out = Path(lstm_optuna_out_value)
                if lstm_optuna_out.exists():
                    mlflow.log_artifact(str(lstm_optuna_out))
        print(f"[✓] MLflow run logged: experiment={experiment!r}, uri={tracking_uri}")
    except Exception as exc:
        print(f"[w] MLflow logging failed: {exc}")


if __name__ == "__main__":
    main()
