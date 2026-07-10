import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cv.gesture_features import (
    DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES,
    DYNAMIC_SEQUENCE_TARGET_FRAMES,
    FEATURE_STATIC_MEAN,
    SUPPORTED_FEATURE_MODES,
    build_feature_vector,
    feature_vector_size,
)
from cv.gesture_dataset_files import (
    augmented_sample_paths,
    gesture_sample_paths,
    sample_group_key,
)
from cv.gesture_validation import make_grouped_splitter, normalized_groups
from cv.model_bundle import publish_model_bundle_atomic
from cv.sequence_multirocket import RandomMultiRocketSequenceTransformer
from cv.sequence_phase_hmm import PhaseHMMSequenceClassifier
from cv.sequence_rocket import RandomConvolutionSequenceTransformer
from cv.sequence_gru_backbone import (
    TorchGRUBackboneClassifier,
    TorchLSTMBackboneClassifier,
    make_dynamic_landmark_lstm_backbone_classifier,
    tune_gru_backbone_hyperparameters,
    tune_lstm_backbone_hyperparameters,
)
from cv.sequence_shapelet import ShapeletSequenceTransformer
from cv.sequence_sprocket import SprocketSequenceTransformer
from cv.static_landmark_cnn import (
    KerasStaticLandmarkCNNClassifier,
    make_dynamic_landmark_cnn_classifier,
)

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
    "dynamic_landmark_lstm_backbone",
    "svm",
    "extra_trees",
    "static_stacking",
    "static_landmark_cnn",
    "dynamic_landmark_cnn",
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
DEFAULT_EXTRA_TREES_N_ESTIMATORS = 250
DEFAULT_EXTRA_TREES_MAX_DEPTH = None
DEFAULT_EXTRA_TREES_MIN_SAMPLES_SPLIT = 2
DEFAULT_EXTRA_TREES_MIN_SAMPLES_LEAF = 1
DEFAULT_EXTRA_TREES_MAX_FEATURES = "sqrt"
DEFAULT_EXTRA_TREES_CRITERION = "gini"
DEFAULT_EXTRA_TREES_BOOTSTRAP = False
DEFAULT_EXTRA_TREES_OPTUNA_TRIALS = 0
DEFAULT_EXTRA_TREES_OPTUNA_CV_FOLDS = 3
DEFAULT_STATIC_STACKING_CV_FOLDS = 3
DEFAULT_STATIC_CNN_MAX_EPOCHS = 120
DEFAULT_STATIC_CNN_BATCH_SIZE = 16
DEFAULT_STATIC_CNN_VALIDATION_FRACTION = 0.20
DEFAULT_STATIC_CNN_PATIENCE = 18
DEFAULT_STATIC_CNN_LEARNING_RATE = 1e-3
DEFAULT_STATIC_CNN_DROPOUT = 0.20
DEFAULT_STATIC_CNN_LABEL_SMOOTHING = 0.05
DEFAULT_CLASS_BALANCE = "auto"
DEFAULT_CLASS_BOOST_LABELS = ("hend", "gun")
DEFAULT_CLASS_BOOST_FACTOR = 1.5


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
    return_groups: bool = False,
) -> tuple:
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
    group_list: List[str] = []
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
        if include_augmented:
            sample_files = [
                *sample_files,
                *_gislr_augmented_sample_paths(label_dir),
            ]
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
            group_list.append(sample_group_key(sf))
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
    if return_groups:
        return X, y, classes, np.asarray(group_list, dtype=object)
    return X, y, classes


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
    extra_trees_n_estimators: int = DEFAULT_EXTRA_TREES_N_ESTIMATORS,
    extra_trees_max_depth: int | None = DEFAULT_EXTRA_TREES_MAX_DEPTH,
    extra_trees_min_samples_split: int = DEFAULT_EXTRA_TREES_MIN_SAMPLES_SPLIT,
    extra_trees_min_samples_leaf: int = DEFAULT_EXTRA_TREES_MIN_SAMPLES_LEAF,
    extra_trees_max_features: str | float | None = DEFAULT_EXTRA_TREES_MAX_FEATURES,
    extra_trees_criterion: str = DEFAULT_EXTRA_TREES_CRITERION,
    extra_trees_bootstrap: bool = DEFAULT_EXTRA_TREES_BOOTSTRAP,
    static_stacking_cv_folds: int = DEFAULT_STATIC_STACKING_CV_FOLDS,
    static_cnn_max_epochs: int = DEFAULT_STATIC_CNN_MAX_EPOCHS,
    static_cnn_batch_size: int = DEFAULT_STATIC_CNN_BATCH_SIZE,
    static_cnn_validation_fraction: float = DEFAULT_STATIC_CNN_VALIDATION_FRACTION,
    static_cnn_patience: int = DEFAULT_STATIC_CNN_PATIENCE,
    static_cnn_learning_rate: float = DEFAULT_STATIC_CNN_LEARNING_RATE,
    static_cnn_dropout: float = DEFAULT_STATIC_CNN_DROPOUT,
    static_cnn_label_smoothing: float = DEFAULT_STATIC_CNN_LABEL_SMOOTHING,
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
    if model == "dynamic_landmark_lstm_backbone":
        return make_dynamic_landmark_lstm_backbone_classifier(
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
        max_depth = (
            None
            if extra_trees_max_depth in {None, 0}
            else max(1, int(extra_trees_max_depth))
        )
        criterion = str(extra_trees_criterion or DEFAULT_EXTRA_TREES_CRITERION)
        if criterion not in {"gini", "entropy", "log_loss"}:
            criterion = DEFAULT_EXTRA_TREES_CRITERION
        return ExtraTreesClassifier(
            n_estimators=max(10, int(extra_trees_n_estimators)),
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=max(2, int(extra_trees_min_samples_split)),
            min_samples_leaf=max(1, int(extra_trees_min_samples_leaf)),
            max_features=extra_trees_max_features,
            bootstrap=bool(extra_trees_bootstrap),
            random_state=int(random_state),
            class_weight="balanced",
            n_jobs=-1,
        )
    if model == "static_stacking":
        cv_folds = max(2, int(static_stacking_cv_folds))
        return StackingClassifier(
            estimators=[
                (
                    "extra_trees",
                    ExtraTreesClassifier(
                        n_estimators=max(80, int(extra_trees_n_estimators)),
                        criterion=str(
                            extra_trees_criterion or DEFAULT_EXTRA_TREES_CRITERION
                        ),
                        max_depth=extra_trees_max_depth,
                        min_samples_split=max(
                            2,
                            int(extra_trees_min_samples_split),
                        ),
                        min_samples_leaf=max(1, int(extra_trees_min_samples_leaf)),
                        max_features=extra_trees_max_features,
                        bootstrap=bool(extra_trees_bootstrap),
                        random_state=int(random_state),
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=180,
                        random_state=int(random_state) + 1,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
                (
                    "svm",
                    make_pipeline(
                        StandardScaler(),
                        SVC(
                            kernel="rbf",
                            C=2.0,
                            gamma="scale",
                            class_weight="balanced",
                            probability=True,
                            random_state=int(random_state) + 2,
                        ),
                    ),
                ),
                (
                    "logreg",
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=int(random_state) + 3,
                        ),
                    ),
                ),
            ],
            final_estimator=LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=int(random_state) + 4,
            ),
            stack_method="predict_proba",
            cv=cv_folds,
            n_jobs=-1,
        )
    if model == "static_landmark_cnn":
        return KerasStaticLandmarkCNNClassifier(
            max_epochs=max(1, int(static_cnn_max_epochs)),
            batch_size=max(1, int(static_cnn_batch_size)),
            validation_fraction=max(
                0.0,
                min(0.50, float(static_cnn_validation_fraction)),
            ),
            patience=max(1, int(static_cnn_patience)),
            learning_rate=max(1e-6, float(static_cnn_learning_rate)),
            dropout=max(0.0, float(static_cnn_dropout)),
            label_smoothing=max(0.0, float(static_cnn_label_smoothing)),
            random_state=int(random_state),
        )
    if model == "dynamic_landmark_cnn":
        return make_dynamic_landmark_cnn_classifier(
            max_epochs=max(1, int(static_cnn_max_epochs)),
            batch_size=max(1, int(static_cnn_batch_size)),
            validation_fraction=max(
                0.0,
                min(0.50, float(static_cnn_validation_fraction)),
            ),
            patience=max(1, int(static_cnn_patience)),
            learning_rate=max(1e-6, float(static_cnn_learning_rate)),
            dropout=max(0.0, float(static_cnn_dropout)),
            label_smoothing=max(0.0, float(static_cnn_label_smoothing)),
            random_state=int(random_state),
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


def model_uses_internal_class_balance(model_type: str) -> bool:
    model = str(model_type or "knn").strip().lower()
    return model in {
        "sequence_rocket",
        "sequence_multirocket",
        "sequence_sprocket",
        "sequence_shapelet",
        "sequence_ensemble",
        "sequence_gru_backbone",
        "sequence_lstm_backbone",
        "dynamic_landmark_lstm_backbone",
        "svm",
        "extra_trees",
        "static_stacking",
        "static_landmark_cnn",
        "dynamic_landmark_cnn",
        "rf",
        "logreg",
    }


def _normalised_label_set(labels: Iterable[str]) -> set[str]:
    return {str(label).strip().lower() for label in labels if str(label).strip()}


def _resolve_class_balance_mode(
    requested: str,
    *,
    model_type: str,
    has_boosted_labels: bool,
) -> str:
    clean = str(requested or DEFAULT_CLASS_BALANCE).strip().lower()
    if clean == "none":
        return "none"
    if clean == "oversample":
        return "oversample"
    if clean != "auto":
        raise ValueError(f"unsupported class balance mode: {requested}")
    if not model_uses_internal_class_balance(model_type):
        return "oversample"
    return "boost_labels" if has_boosted_labels else "none"


def balance_training_set(
    X: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    *,
    model_type: str,
    strategy: str = DEFAULT_CLASS_BALANCE,
    boost_labels: Iterable[str] = DEFAULT_CLASS_BOOST_LABELS,
    boost_factor: float = DEFAULT_CLASS_BOOST_FACTOR,
    random_state: int = 42,
    return_indices: bool = False,
) -> tuple:
    """Return a fit set with optional GISLR-style weak-class amplification."""
    labels = [str(label) for label in classes]
    encoded = np.asarray(y, dtype=np.int64)
    class_count = len(labels)
    counts = (
        np.bincount(encoded, minlength=class_count).astype(np.int64)
        if class_count > 0
        else np.zeros(0, dtype=np.int64)
    )
    boost_set = _normalised_label_set(boost_labels)
    boosted_indices = {
        idx for idx, label in enumerate(labels) if label.strip().lower() in boost_set
    }
    present_boosted = {
        idx for idx in boosted_indices if idx < counts.shape[0] and counts[idx] > 0
    }
    mode = _resolve_class_balance_mode(
        strategy,
        model_type=model_type,
        has_boosted_labels=bool(present_boosted) and float(boost_factor) > 1.0,
    )
    metadata = {
        "requested": str(strategy or DEFAULT_CLASS_BALANCE),
        "effective": mode,
        "model_type": str(model_type),
        "boost_labels": sorted(boost_set),
        "boost_factor": float(boost_factor),
        "original_samples": int(encoded.shape[0]),
        "fit_samples": int(encoded.shape[0]),
        "class_counts": {
            label: int(counts[idx]) for idx, label in enumerate(labels)
        },
        "fit_class_counts": {
            label: int(counts[idx]) for idx, label in enumerate(labels)
        },
    }
    original_indices = np.arange(encoded.shape[0], dtype=np.int64)
    if mode == "none" or encoded.size == 0 or class_count <= 1:
        if return_indices:
            return X, encoded, metadata, original_indices
        return X, encoded, metadata

    targets = counts.copy()
    balance_target_count = int(counts.max()) if counts.size else 0
    if mode == "oversample" and counts.size:
        targets = np.where(counts > 0, balance_target_count, 0).astype(np.int64)

    factor = max(1.0, float(boost_factor))
    for class_idx in present_boosted:
        if mode == "oversample":
            boosted_target = int(np.ceil(float(max(1, balance_target_count)) * factor))
        else:
            boosted_target = int(np.ceil(float(counts[class_idx]) * factor))
        targets[class_idx] = max(int(targets[class_idx]), boosted_target)

    if np.array_equal(targets, counts):
        metadata["effective"] = "none"
        if return_indices:
            return X, encoded, metadata, original_indices
        return X, encoded, metadata

    rng = np.random.default_rng(int(random_state))
    indices = [int(idx) for idx in range(encoded.shape[0])]
    for class_idx, target_count in enumerate(targets):
        current_count = int(counts[class_idx]) if class_idx < counts.shape[0] else 0
        missing = int(target_count) - current_count
        if missing <= 0:
            continue
        class_indices = np.flatnonzero(encoded == class_idx)
        if class_indices.size <= 0:
            continue
        sampled = rng.choice(class_indices, size=missing, replace=True)
        indices.extend(int(idx) for idx in sampled)

    shuffled = np.asarray(indices, dtype=np.int64)
    rng.shuffle(shuffled)
    fit_y = encoded[shuffled]
    fit_counts = np.bincount(fit_y, minlength=class_count).astype(np.int64)
    metadata["fit_samples"] = int(fit_y.shape[0])
    metadata["fit_class_counts"] = {
        label: int(fit_counts[idx]) for idx, label in enumerate(labels)
    }
    if return_indices:
        return X[shuffled], fit_y, metadata, shuffled
    return X[shuffled], fit_y, metadata


def _extra_trees_default_params() -> dict[str, object]:
    return {
        "extra_trees_n_estimators": int(DEFAULT_EXTRA_TREES_N_ESTIMATORS),
        "extra_trees_max_depth": DEFAULT_EXTRA_TREES_MAX_DEPTH,
        "extra_trees_min_samples_split": int(DEFAULT_EXTRA_TREES_MIN_SAMPLES_SPLIT),
        "extra_trees_min_samples_leaf": int(DEFAULT_EXTRA_TREES_MIN_SAMPLES_LEAF),
        "extra_trees_max_features": DEFAULT_EXTRA_TREES_MAX_FEATURES,
        "extra_trees_criterion": DEFAULT_EXTRA_TREES_CRITERION,
        "extra_trees_bootstrap": bool(DEFAULT_EXTRA_TREES_BOOTSTRAP),
    }


def tune_extra_trees_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    *,
    n_trials: int,
    cv_folds: int = DEFAULT_EXTRA_TREES_OPTUNA_CV_FOLDS,
    timeout: int | None = None,
    random_state: int = 42,
    class_balance: str = DEFAULT_CLASS_BALANCE,
    boost_labels: Iterable[str] = DEFAULT_CLASS_BOOST_LABELS,
    boost_factor: float = DEFAULT_CLASS_BOOST_FACTOR,
    groups=None,
) -> dict[str, object]:
    """Tune ExtraTrees for static handcrafted features with macro-F1 CV."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - dependency is in requirements
        raise RuntimeError("Optuna is required for extra_trees tuning") from exc

    matrix = np.asarray(X, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    group_values = (
        normalized_groups(groups, labels.shape[0])
        if groups is not None
        else None
    )
    unique = np.unique(labels)
    default_params = _extra_trees_default_params()
    if matrix.ndim != 2 or labels.ndim != 1 or unique.size < 2:
        return {
            "best_score": 0.0,
            "best_params": default_params,
            "trials": 0,
            "cv_folds": 0,
            "used_cv": False,
            "grouped_cv": False,
            "group_count": int(len(set(group_values.tolist())))
            if group_values is not None
            else int(labels.shape[0]),
            "reason": "not_enough_classes",
        }

    counts = np.bincount(labels, minlength=len(classes))
    positive_counts = counts[counts > 0]
    min_class_count = int(positive_counts.min()) if positive_counts.size else 0
    grouped_cv = group_values is not None
    if grouped_cv:
        try:
            splitter, folds = make_grouped_splitter(
                labels,
                group_values,
                max_folds=cv_folds,
                random_state=random_state,
            )
            used_cv = True
        except ValueError:
            return {
                "best_score": 0.0,
                "best_params": default_params,
                "trials": 0,
                "cv_folds": 0,
                "used_cv": False,
                "grouped_cv": False,
                "group_count": int(len(set(group_values.tolist()))),
                "reason": "not_enough_source_groups",
            }
    else:
        folds = min(max(2, int(cv_folds)), min_class_count)
        used_cv = folds >= 2
        splitter = (
            StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=int(random_state),
            )
            if used_cv
            else None
        )

    def trial_params(trial) -> dict[str, object]:
        return {
            "extra_trees_n_estimators": int(
                trial.suggest_int("n_estimators", 120, 520, step=40)
            ),
            "extra_trees_max_depth": trial.suggest_categorical(
                "max_depth",
                [None, 6, 10, 14, 20, 30],
            ),
            "extra_trees_min_samples_split": int(
                trial.suggest_int("min_samples_split", 2, 8)
            ),
            "extra_trees_min_samples_leaf": int(
                trial.suggest_int("min_samples_leaf", 1, 5)
            ),
            "extra_trees_max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", 0.35, 0.50, 0.75, 1.0],
            ),
            "extra_trees_criterion": trial.suggest_categorical(
                "criterion",
                ["gini", "entropy"],
            ),
            "extra_trees_bootstrap": bool(
                trial.suggest_categorical("bootstrap", [False, True])
            ),
        }

    def fit_score(params: dict[str, object], train_idx, valid_idx) -> float:
        clf = build_classifier(
            "extra_trees",
            random_state=int(random_state),
            **params,
        )
        train_X = matrix[train_idx]
        train_y = labels[train_idx]
        fit_X, fit_y, _metadata = balance_training_set(
            train_X,
            train_y,
            classes,
            model_type="extra_trees",
            strategy=class_balance,
            boost_labels=boost_labels,
            boost_factor=boost_factor,
            random_state=int(random_state),
        )
        clf.fit(fit_X, fit_y)
        pred = clf.predict(matrix[valid_idx])
        return float(
            f1_score(
                labels[valid_idx],
                pred,
                average="macro",
                zero_division=0,
            )
        )

    def objective(trial) -> float:
        params = trial_params(trial)
        if splitter is None:
            indices = np.arange(labels.shape[0])
            return fit_score(params, indices, indices)
        split_rows = (
            splitter.split(matrix, labels, groups=group_values)
            if grouped_cv
            else splitter.split(matrix, labels)
        )
        scores = [
            fit_score(params, train_idx, valid_idx)
            for train_idx, valid_idx in split_rows
        ]
        return float(np.mean(scores)) if scores else 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(random_state))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=max(1, int(n_trials)),
        timeout=timeout if timeout and timeout > 0 else None,
        show_progress_bar=False,
    )
    raw_best = dict(study.best_trial.params)
    best_params = {
        "extra_trees_n_estimators": int(
            raw_best.get("n_estimators", DEFAULT_EXTRA_TREES_N_ESTIMATORS)
        ),
        "extra_trees_max_depth": raw_best.get(
            "max_depth",
            DEFAULT_EXTRA_TREES_MAX_DEPTH,
        ),
        "extra_trees_min_samples_split": int(
            raw_best.get(
                "min_samples_split",
                DEFAULT_EXTRA_TREES_MIN_SAMPLES_SPLIT,
            )
        ),
        "extra_trees_min_samples_leaf": int(
            raw_best.get(
                "min_samples_leaf",
                DEFAULT_EXTRA_TREES_MIN_SAMPLES_LEAF,
            )
        ),
        "extra_trees_max_features": raw_best.get(
            "max_features",
            DEFAULT_EXTRA_TREES_MAX_FEATURES,
        ),
        "extra_trees_criterion": str(
            raw_best.get("criterion", DEFAULT_EXTRA_TREES_CRITERION)
        ),
        "extra_trees_bootstrap": bool(
            raw_best.get("bootstrap", DEFAULT_EXTRA_TREES_BOOTSTRAP)
        ),
    }
    return {
        "best_score": float(study.best_value),
        "best_params": best_params,
        "trials": len(study.trials),
        "cv_folds": int(folds) if used_cv else 0,
        "used_cv": bool(used_cv),
        "grouped_cv": bool(grouped_cv and used_cv),
        "group_count": int(len(set(group_values.tolist())))
        if group_values is not None
        else int(labels.shape[0]),
        "reason": "",
    }


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
    p = argparse.ArgumentParser(description="Обучение классификатора жестов")
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
        help=(
            "Режим признаков: static_mean — production baseline; "
            "static_craft_full_stats — full hand geometry for static poses; "
            "static_landmark_image — GISLR-style time x hand points x xyz; "
            "dynamic_stats/hybrid_stats — dynamic baseline; "
            "dynamic_craft_stats — light GISLR-style geometry; "
            "dynamic_craft_full_stats — all 210 hand distances + 15 angles; "
            "dynamic_landmark_image — экспериментальный time x points x xyz benchmark"
        ),
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
            "dynamic_landmark_lstm_backbone, "
            "svm, extra_trees, static_stacking, static_landmark_cnn, "
            "dynamic_landmark_cnn, rf или logreg"
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
        "--extra-trees-optuna-trials",
        type=int,
        default=DEFAULT_EXTRA_TREES_OPTUNA_TRIALS,
        help="Run Optuna tuning before final extra_trees training.",
    )
    p.add_argument(
        "--extra-trees-optuna-cv-folds",
        type=int,
        default=DEFAULT_EXTRA_TREES_OPTUNA_CV_FOLDS,
        help="Stratified CV folds for extra_trees Optuna tuning.",
    )
    p.add_argument(
        "--extra-trees-optuna-timeout",
        type=int,
        default=0,
        help="Optional ExtraTrees Optuna timeout in seconds; 0 means no timeout.",
    )
    p.add_argument(
        "--static-stacking-cv-folds",
        type=int,
        default=DEFAULT_STATIC_STACKING_CV_FOLDS,
        help="Internal stratified CV folds for static_stacking.",
    )
    p.add_argument(
        "--static-cnn-max-epochs",
        type=int,
        default=DEFAULT_STATIC_CNN_MAX_EPOCHS,
        help="Maximum epochs for static_landmark_cnn.",
    )
    p.add_argument(
        "--static-cnn-batch-size",
        type=int,
        default=DEFAULT_STATIC_CNN_BATCH_SIZE,
        help="Mini-batch size for static_landmark_cnn.",
    )
    p.add_argument(
        "--static-cnn-validation-fraction",
        type=float,
        default=DEFAULT_STATIC_CNN_VALIDATION_FRACTION,
        help="Internal validation split for static_landmark_cnn early stopping.",
    )
    p.add_argument(
        "--static-cnn-patience",
        type=int,
        default=DEFAULT_STATIC_CNN_PATIENCE,
        help="Early-stopping patience for static_landmark_cnn.",
    )
    p.add_argument(
        "--static-cnn-learning-rate",
        type=float,
        default=DEFAULT_STATIC_CNN_LEARNING_RATE,
        help="Adam learning rate for static_landmark_cnn.",
    )
    p.add_argument(
        "--static-cnn-dropout",
        type=float,
        default=DEFAULT_STATIC_CNN_DROPOUT,
        help="Dropout for static_landmark_cnn.",
    )
    p.add_argument(
        "--static-cnn-label-smoothing",
        type=float,
        default=DEFAULT_STATIC_CNN_LABEL_SMOOTHING,
        help="Categorical label smoothing for static_landmark_cnn.",
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
        "--class-balance",
        choices=["auto", "none", "oversample"],
        default=DEFAULT_CLASS_BALANCE,
        help=(
            "Политика усиления классов: auto — class_weight/weighted loss там, "
            "где они есть, и oversampling для KNN/MLP/HMM; oversample — "
            "дублировать редкие классы для любой модели; none — выключить."
        ),
    )
    p.add_argument(
        "--boost-label",
        action="append",
        default=None,
        help=(
            "Дополнительно усилить конкретный класс через oversampling; можно "
            "передать несколько раз. По умолчанию: hend и gun."
        ),
    )
    p.add_argument(
        "--boost-factor",
        type=float,
        default=DEFAULT_CLASS_BOOST_FACTOR,
        help="Множитель усиления для --boost-label.",
    )
    p.add_argument(
        "--mlflow-experiment",
        default="GestureBind",
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

    X, y, classes, sample_groups = load_dataset(
        data_root,
        expect_dim=args.expect_dim,
        include_labels=args.include_label,
        lowercase_labels=bool(args.lowercase_labels),
        feature_mode=str(args.feature_mode),
        include_augmented=bool(args.include_augmented),
        return_groups=True,
    )
    args.sample_group_count = int(len(set(sample_groups.tolist())))
    print(
        f"[i] Загружено семплов: {len(X)}; классов: {len(classes)}; "
        f"режим признаков: {args.feature_mode}; размер признака: {X.shape[1]}; "
        f"модель: {args.model_type}"
    )
    model_type_clean = str(args.model_type).strip().lower()
    recurrent_lstm_target_frames = (
        DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES
        if model_type_clean == "dynamic_landmark_lstm_backbone"
        else DYNAMIC_SEQUENCE_TARGET_FRAMES
    )
    recurrent_lstm_feature_name = (
        "dynamic_landmark_image"
        if model_type_clean == "dynamic_landmark_lstm_backbone"
        else "dynamic_sequence"
    )

    sequence_mlp_validation_fraction = max(
        0.05,
        min(0.50, float(args.sequence_mlp_validation_fraction)),
    )
    sequence_mlp_early_stopping = bool(args.sequence_mlp_early_stopping)
    if model_type_clean == "sequence_mlp":
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

    boost_labels = (
        list(args.boost_label)
        if args.boost_label is not None
        else list(DEFAULT_CLASS_BOOST_LABELS)
    )
    extra_trees_params = _extra_trees_default_params()
    args.extra_trees_optuna_summary = None
    args.extra_trees_optuna_out = ""
    if (
        model_type_clean == "extra_trees"
        and int(args.extra_trees_optuna_trials) > 0
    ):
        trials = max(1, int(args.extra_trees_optuna_trials))
        print(f"[i] Optuna tuning для extra_trees: trials={trials}")
        try:
            tuning = tune_extra_trees_hyperparameters(
                X,
                y,
                classes,
                n_trials=trials,
                cv_folds=max(2, int(args.extra_trees_optuna_cv_folds)),
                timeout=int(args.extra_trees_optuna_timeout) or None,
                random_state=int(args.random_state),
                class_balance=str(args.class_balance),
                boost_labels=boost_labels,
                boost_factor=float(args.boost_factor),
                groups=sample_groups,
            )
        except RuntimeError as exc:
            print(
                "[!] Optuna tuning недоступен: "
                f"{exc}. Установи зависимости из requirements.txt."
            )
            raise SystemExit(2) from exc
        best_params = dict(tuning.get("best_params") or {})
        extra_trees_params.update(best_params)
        args.extra_trees_optuna_summary = tuning
        args.extra_trees_optuna_out = str(
            out_path.with_name(f"{out_path.stem}_optuna.json")
        )
        Path(args.extra_trees_optuna_out).write_text(
            json.dumps(args.extra_trees_optuna_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "[i] Optuna best: "
            f"score={float(tuning.get('best_score', 0.0)):.4f}, "
            f"cv_folds={int(tuning.get('cv_folds', 0))}, "
            f"params={json.dumps(best_params, ensure_ascii=False)}"
        )
    for key, value in extra_trees_params.items():
        setattr(args, f"{key}_effective", value)

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
        model_type_clean == "sequence_gru_backbone"
        and int(args.sequence_gru_optuna_trials) > 0
    ):
        trials = max(1, int(args.sequence_gru_optuna_trials))
        print(f"[i] Optuna tuning для sequence_gru_backbone: trials={trials}")
        tuning = tune_gru_backbone_hyperparameters(
            X,
            y,
            target_frames=DYNAMIC_SEQUENCE_TARGET_FRAMES,
            feature_name="dynamic_sequence",
            n_trials=trials,
            timeout=int(args.sequence_gru_optuna_timeout) or None,
            random_state=int(args.random_state),
            max_epochs=max(10, int(args.sequence_gru_optuna_max_epochs)),
            groups=sample_groups,
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
            "used_group_split": bool(tuning.used_group_split),
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
        model_type_clean
        in {"sequence_lstm_backbone", "dynamic_landmark_lstm_backbone"}
        and int(args.sequence_lstm_optuna_trials) > 0
    ):
        trials = max(1, int(args.sequence_lstm_optuna_trials))
        print(f"[i] Optuna tuning для {model_type_clean}: trials={trials}")
        tuning = tune_lstm_backbone_hyperparameters(
            X,
            y,
            target_frames=recurrent_lstm_target_frames,
            feature_name=recurrent_lstm_feature_name,
            n_trials=trials,
            timeout=int(args.sequence_lstm_optuna_timeout) or None,
            random_state=int(args.random_state),
            max_epochs=max(10, int(args.sequence_lstm_optuna_max_epochs)),
            groups=sample_groups,
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
            "used_group_split": bool(tuning.used_group_split),
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
        **extra_trees_params,
        static_stacking_cv_folds=int(args.static_stacking_cv_folds),
        static_cnn_max_epochs=int(args.static_cnn_max_epochs),
        static_cnn_batch_size=int(args.static_cnn_batch_size),
        static_cnn_validation_fraction=float(args.static_cnn_validation_fraction),
        static_cnn_patience=int(args.static_cnn_patience),
        static_cnn_learning_rate=float(args.static_cnn_learning_rate),
        static_cnn_dropout=float(args.static_cnn_dropout),
        static_cnn_label_smoothing=float(args.static_cnn_label_smoothing),
    )
    X_fit, y_fit, class_balance_metadata, fit_indices = balance_training_set(
        X,
        y,
        classes,
        model_type=str(args.model_type),
        strategy=str(args.class_balance),
        boost_labels=boost_labels,
        boost_factor=float(args.boost_factor),
        random_state=int(args.random_state),
        return_indices=True,
    )
    fit_groups = sample_groups[np.asarray(fit_indices, dtype=np.int64)]
    args.class_balance_effective = str(class_balance_metadata.get("effective", "none"))
    args.class_balance_original_samples = int(
        class_balance_metadata.get("original_samples", X.shape[0])
    )
    args.class_balance_fit_samples = int(
        class_balance_metadata.get("fit_samples", X_fit.shape[0])
    )
    args.class_balance_fit_class_counts = dict(
        class_balance_metadata.get("fit_class_counts", {})
    )
    args.boost_label_effective = boost_labels
    if X_fit.shape[0] != X.shape[0]:
        print(
            "[i] Балансировка классов: "
            f"{class_balance_metadata['effective']}, "
            f"{X.shape[0]} -> {X_fit.shape[0]} fit-семплов; "
            f"counts={class_balance_metadata['fit_class_counts']}"
        )
    else:
        print(
            "[i] Балансировка классов: "
            f"{class_balance_metadata['effective']} "
            f"(fit-семплов: {X_fit.shape[0]})"
        )

    if isinstance(clf, TorchGRUBackboneClassifier):
        clf.fit(X_fit, y_fit, groups=fit_groups)
    else:
        clf.fit(X_fit, y_fit)
    train_accuracy = float(clf.score(X, y))

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
    rejection_metadata["training_class_balance"] = class_balance_metadata
    rejection_metadata["training_group_count"] = int(args.sample_group_count)
    rejection_metadata["validation_grouped"] = bool(
        getattr(clf, "used_group_validation_", False)
    )
    rejection_metadata["validation_group_overlap"] = int(
        getattr(clf, "validation_group_overlap_", 0)
    )
    publish_model_bundle_atomic(
        clf,
        out_path,
        {
            classes_out: json.dumps(classes, ensure_ascii=False, indent=2),
            feature_dim_out: str(X.shape[1]),
            feature_mode_out: str(args.feature_mode),
            rejection_out: json.dumps(
                rejection_metadata,
                ensure_ascii=False,
                indent=2,
            ),
        },
    )
    print(f"[✓] Модель сохранена атомарно: {out_path}")
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
        class_balance = str(getattr(args, "class_balance", DEFAULT_CLASS_BALANCE))
        class_balance_effective = str(
            getattr(args, "class_balance_effective", "none")
        )
        class_balance_original_samples = int(
            getattr(args, "class_balance_original_samples", sample_count)
        )
        class_balance_fit_samples = int(
            getattr(args, "class_balance_fit_samples", sample_count)
        )
        boost_factor = float(getattr(args, "boost_factor", DEFAULT_CLASS_BOOST_FACTOR))
        boost_labels = getattr(args, "boost_label_effective", None)
        if boost_labels is None:
            boost_labels = getattr(args, "boost_label", None)
        if boost_labels is None:
            boost_labels = list(DEFAULT_CLASS_BOOST_LABELS)
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
        extra_trees_n_estimators = int(
            getattr(args, "extra_trees_n_estimators_effective", DEFAULT_EXTRA_TREES_N_ESTIMATORS)
        )
        extra_trees_max_depth = getattr(
            args,
            "extra_trees_max_depth_effective",
            DEFAULT_EXTRA_TREES_MAX_DEPTH,
        )
        extra_trees_min_samples_split = int(
            getattr(
                args,
                "extra_trees_min_samples_split_effective",
                DEFAULT_EXTRA_TREES_MIN_SAMPLES_SPLIT,
            )
        )
        extra_trees_min_samples_leaf = int(
            getattr(
                args,
                "extra_trees_min_samples_leaf_effective",
                DEFAULT_EXTRA_TREES_MIN_SAMPLES_LEAF,
            )
        )
        extra_trees_max_features = getattr(
            args,
            "extra_trees_max_features_effective",
            DEFAULT_EXTRA_TREES_MAX_FEATURES,
        )
        extra_trees_criterion = str(
            getattr(
                args,
                "extra_trees_criterion_effective",
                DEFAULT_EXTRA_TREES_CRITERION,
            )
        )
        extra_trees_bootstrap = bool(
            getattr(
                args,
                "extra_trees_bootstrap_effective",
                DEFAULT_EXTRA_TREES_BOOTSTRAP,
            )
        )
        extra_trees_optuna_trials = int(
            getattr(
                args,
                "extra_trees_optuna_trials",
                DEFAULT_EXTRA_TREES_OPTUNA_TRIALS,
            )
        )
        extra_trees_optuna_summary = getattr(args, "extra_trees_optuna_summary", None)
        static_stacking_cv_folds = int(
            getattr(args, "static_stacking_cv_folds", DEFAULT_STATIC_STACKING_CV_FOLDS)
        )
        static_cnn_max_epochs = int(
            getattr(args, "static_cnn_max_epochs", DEFAULT_STATIC_CNN_MAX_EPOCHS)
        )
        static_cnn_batch_size = int(
            getattr(args, "static_cnn_batch_size", DEFAULT_STATIC_CNN_BATCH_SIZE)
        )
        static_cnn_validation_fraction = float(
            getattr(
                args,
                "static_cnn_validation_fraction",
                DEFAULT_STATIC_CNN_VALIDATION_FRACTION,
            )
        )
        static_cnn_patience = int(
            getattr(args, "static_cnn_patience", DEFAULT_STATIC_CNN_PATIENCE)
        )
        static_cnn_learning_rate = float(
            getattr(
                args,
                "static_cnn_learning_rate",
                DEFAULT_STATIC_CNN_LEARNING_RATE,
            )
        )
        static_cnn_dropout = float(
            getattr(args, "static_cnn_dropout", DEFAULT_STATIC_CNN_DROPOUT)
        )
        static_cnn_label_smoothing = float(
            getattr(
                args,
                "static_cnn_label_smoothing",
                DEFAULT_STATIC_CNN_LABEL_SMOOTHING,
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
                    "class_balance": class_balance,
                    "class_balance_effective": class_balance_effective,
                    "class_balance_original_samples": class_balance_original_samples,
                    "class_balance_fit_samples": class_balance_fit_samples,
                    "boost_labels": ",".join(str(label) for label in boost_labels),
                    "boost_factor": boost_factor,
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
                    "extra_trees_n_estimators_effective": extra_trees_n_estimators,
                    "extra_trees_max_depth_effective": (
                        "" if extra_trees_max_depth is None else int(extra_trees_max_depth)
                    ),
                    "extra_trees_min_samples_split_effective": (
                        extra_trees_min_samples_split
                    ),
                    "extra_trees_min_samples_leaf_effective": (
                        extra_trees_min_samples_leaf
                    ),
                    "extra_trees_max_features_effective": str(
                        extra_trees_max_features
                    ),
                    "extra_trees_criterion_effective": extra_trees_criterion,
                    "extra_trees_bootstrap_effective": extra_trees_bootstrap,
                    "extra_trees_optuna_trials": extra_trees_optuna_trials,
                    "static_stacking_cv_folds": static_stacking_cv_folds,
                    "static_cnn_max_epochs": static_cnn_max_epochs,
                    "static_cnn_batch_size": static_cnn_batch_size,
                    "static_cnn_validation_fraction": static_cnn_validation_fraction,
                    "static_cnn_patience": static_cnn_patience,
                    "static_cnn_learning_rate": static_cnn_learning_rate,
                    "static_cnn_dropout": static_cnn_dropout,
                    "static_cnn_label_smoothing": static_cnn_label_smoothing,
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
                "fit_sample_count": float(class_balance_fit_samples),
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
            if isinstance(extra_trees_optuna_summary, dict):
                metrics["extra_trees_optuna_best_score"] = float(
                    extra_trees_optuna_summary.get("best_score", 0.0)
                )
                metrics["extra_trees_optuna_trials_done"] = float(
                    extra_trees_optuna_summary.get("trials", 0)
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
            extra_trees_optuna_out_value = str(
                getattr(args, "extra_trees_optuna_out", "") or ""
            ).strip()
            if extra_trees_optuna_out_value:
                extra_trees_optuna_out = Path(extra_trees_optuna_out_value)
                if extra_trees_optuna_out.exists():
                    mlflow.log_artifact(str(extra_trees_optuna_out))
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
