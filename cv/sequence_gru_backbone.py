"""PyTorch recurrent backbones for dynamic gesture sequences.

The classifier follows the same sklearn-style API as the other sequence
models in this project, so it can be saved with joblib and used by the
existing live inference path. It trains a small per-frame backbone plus a
GRU/LSTM recurrent layer over flattened temporal landmark features.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from cv.gesture_features import (
    DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES,
    DYNAMIC_SEQUENCE_TARGET_FRAMES,
)
from cv.gesture_validation import grouped_holdout_indices, normalized_groups


@dataclass(frozen=True)
class OptunaTuningSummary:
    best_score: float
    best_params: dict[str, Any]
    trials: int
    used_validation_split: bool
    used_group_split: bool = False


class TorchGRUBackboneClassifier(BaseEstimator, ClassifierMixin):
    """Small GRU classifier for multivariate landmark time-series."""

    sequence_model_name = "sequence_gru_backbone"
    estimator_name = "TorchGRUBackboneClassifier"
    log_prefix = "gru"

    def __init__(
        self,
        *,
        target_frames: int = DYNAMIC_SEQUENCE_TARGET_FRAMES,
        backbone_dim: int = 64,
        hidden_dim: int = 96,
        num_layers: int = 1,
        dropout: float = 0.20,
        use_bidirectional: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 160,
        batch_size: int = 16,
        validation_fraction: float = 0.20,
        patience: int = 24,
        random_state: int = 42,
        device: str = "cpu",
        verbose: bool = False,
        feature_name: str = "dynamic_sequence",
    ) -> None:
        self.target_frames = target_frames
        self.backbone_dim = backbone_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bidirectional = use_bidirectional
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self.feature_name = feature_name

    def fit(self, X, y, groups=None):  # noqa: D401 - sklearn API
        torch = _require_torch()
        matrix = self._validate_X(X)
        labels = np.asarray(y)
        if labels.shape[0] != matrix.shape[0]:
            raise ValueError("X and y have different sample counts")

        self.classes_, encoded = np.unique(labels, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError(f"{self.estimator_name} needs at least two classes")
        self.n_features_in_ = int(matrix.shape[1])
        self.n_channels_ = int(self.n_features_in_ // int(self.target_frames))

        sequences = matrix.reshape(matrix.shape[0], int(self.target_frames), self.n_channels_)
        X_train, X_val, y_train, y_val = self._split_train_validation(
            sequences,
            encoded,
            groups=groups,
        )
        X_train = self._fit_normalizer(X_train)
        if X_val.size:
            X_val = self._transform_normalizer(X_val)

        _seed_torch(torch, int(self.random_state))
        device = self._torch_device(torch)
        model = self._build_network(
            input_dim=int(self.n_channels_),
            class_count=int(self.classes_.size),
            backbone_dim=max(4, int(self.backbone_dim)),
            hidden_dim=max(4, int(self.hidden_dim)),
            num_layers=max(1, int(self.num_layers)),
            dropout=max(0.0, float(self.dropout)),
            use_bidirectional=bool(self.use_bidirectional),
        ).to(device)

        learning_rate = max(1e-6, float(self.learning_rate))
        weight_decay = max(0.0, float(self.weight_decay))
        optimizer_state: dict[int, tuple[Any, Any]] = {}
        optimizer_step = 0
        class_weights = self._class_weights(encoded, torch, device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        train_x = torch.tensor(X_train, dtype=torch.float32, device=device)
        train_y = torch.tensor(y_train, dtype=torch.long, device=device)
        val_x = torch.tensor(X_val, dtype=torch.float32, device=device) if X_val.size else None
        val_y = torch.tensor(y_val, dtype=torch.long, device=device) if y_val.size else None

        best_state = copy.deepcopy(model.state_dict())
        best_score = -math.inf
        best_loss = math.inf
        best_epoch = 0
        stale_epochs = 0
        history: list[dict[str, float]] = []

        for epoch in range(max(1, int(self.max_epochs))):
            model.train()
            losses = []
            for batch_indices in self._batch_indices(train_x.shape[0], epoch):
                batch_x = train_x[batch_indices]
                batch_y = train_y[batch_indices]
                model.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer_step += 1
                self._adamw_step(
                    torch,
                    model.parameters(),
                    state=optimizer_state,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    step=optimizer_step,
                )
                losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(losses)) if losses else 0.0
            eval_x = val_x if val_x is not None else train_x
            eval_y = val_y if val_y is not None else train_y
            eval_loss, eval_accuracy = self._evaluate_torch(
                torch,
                model,
                criterion,
                eval_x,
                eval_y,
            )
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "eval_accuracy": eval_accuracy,
                }
            )

            improved = (
                eval_accuracy > best_score + 1e-6
                or (
                    abs(eval_accuracy - best_score) <= 1e-6
                    and eval_loss < best_loss - 1e-6
                )
            )
            if improved:
                best_score = eval_accuracy
                best_loss = eval_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            if bool(self.verbose) and (epoch + 1) % 20 == 0:
                print(
                    f"[{self.log_prefix}] "
                    f"epoch={epoch + 1} train_loss={train_loss:.4f} "
                    f"eval_loss={eval_loss:.4f} eval_accuracy={eval_accuracy:.4f}"
                )
            if val_y is not None and stale_epochs >= max(1, int(self.patience)):
                break

        model.load_state_dict(best_state)
        self.model_state_dict_ = {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        }
        self.training_history_ = history
        self.best_epoch_ = int(best_epoch)
        self.best_eval_accuracy_ = float(best_score)
        self.best_eval_loss_ = float(best_loss)
        self.used_validation_split_ = bool(val_y is not None)
        return self

    def predict_proba(self, X):  # noqa: D401 - sklearn API
        torch = _require_torch()
        self._check_fitted()
        matrix = self._validate_X(X)
        if int(matrix.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                "feature dimension mismatch: "
                f"{matrix.shape[1]} != {self.n_features_in_}"
            )
        sequences = matrix.reshape(matrix.shape[0], int(self.target_frames), self.n_channels_)
        sequences = self._transform_normalizer(sequences)

        device = self._torch_device(torch)
        model = self._build_network(
            input_dim=int(self.n_channels_),
            class_count=int(self.classes_.size),
            backbone_dim=max(4, int(self.backbone_dim)),
            hidden_dim=max(4, int(self.hidden_dim)),
            num_layers=max(1, int(self.num_layers)),
            dropout=max(0.0, float(self.dropout)),
            use_bidirectional=bool(self.use_bidirectional),
        ).to(device)
        model.load_state_dict(self.model_state_dict_)
        model.eval()

        with torch.no_grad():
            tensor = torch.tensor(sequences, dtype=torch.float32, device=device)
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return probabilities.astype(np.float32, copy=False)

    def predict(self, X):  # noqa: D401 - sklearn API
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def _validate_X(self, X) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"expected 2D feature matrix, got shape={matrix.shape}")
        target_frames = int(self.target_frames)
        if target_frames <= 1:
            raise ValueError("target_frames must be greater than 1")
        if matrix.shape[1] <= 0 or matrix.shape[1] % target_frames != 0:
            raise ValueError(
                f"{self.sequence_model_name} expects flattened {self.feature_name} features "
                f"with dimension divisible by {target_frames}; got {matrix.shape[1]}"
            )
        if not np.isfinite(matrix).all():
            matrix = np.nan_to_num(matrix, copy=False)
        return matrix

    def _fit_normalizer(self, sequences: np.ndarray) -> np.ndarray:
        mean = sequences.mean(axis=(0, 1), keepdims=True)
        std = sequences.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        self.sequence_mean_ = mean.astype(np.float32, copy=False)
        self.sequence_std_ = std.astype(np.float32, copy=False)
        return self._transform_normalizer(sequences)

    def _transform_normalizer(self, sequences: np.ndarray) -> np.ndarray:
        return ((sequences - self.sequence_mean_) / self.sequence_std_).astype(
            np.float32,
            copy=False,
        )

    def _split_train_validation(
        self,
        sequences: np.ndarray,
        encoded: np.ndarray,
        *,
        groups=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fraction = max(0.0, min(0.50, float(self.validation_fraction)))
        self.used_group_validation_ = False
        self.validation_group_overlap_ = 0
        if groups is not None and fraction > 0.0:
            group_values = normalized_groups(groups, encoded.shape[0])
            split = grouped_holdout_indices(
                encoded,
                group_values,
                validation_fraction=fraction,
                random_state=int(self.random_state),
            )
            if split is not None:
                train_idx, validation_idx = split
                train_groups = set(group_values[train_idx].tolist())
                validation_groups = set(group_values[validation_idx].tolist())
                self.used_group_validation_ = True
                self.validation_group_overlap_ = len(
                    train_groups.intersection(validation_groups)
                )
                return (
                    sequences[train_idx],
                    sequences[validation_idx],
                    encoded[train_idx].astype(np.int64),
                    encoded[validation_idx].astype(np.int64),
                )
            empty_x = np.empty((0,) + sequences.shape[1:], dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.int64)
            return sequences, empty_x, encoded.astype(np.int64), empty_y
        if fraction <= 0.0 or not _can_stratified_split(encoded, fraction):
            empty_x = np.empty((0,) + sequences.shape[1:], dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.int64)
            return sequences, empty_x, encoded.astype(np.int64), empty_y
        return train_test_split(
            sequences,
            encoded.astype(np.int64),
            test_size=fraction,
            random_state=int(self.random_state),
            stratify=encoded,
        )

    def _batch_indices(self, sample_count: int, epoch: int) -> list[np.ndarray]:
        rng = np.random.default_rng(int(self.random_state) + int(epoch))
        indices = np.arange(sample_count, dtype=np.int64)
        rng.shuffle(indices)
        batch_size = max(1, int(self.batch_size))
        return [indices[start : start + batch_size] for start in range(0, sample_count, batch_size)]

    def _class_weights(self, encoded: np.ndarray, torch, device):
        counts = np.bincount(encoded.astype(np.int64), minlength=int(self.classes_.size))
        counts = np.maximum(counts, 1)
        weights = counts.sum() / (len(counts) * counts)
        return torch.tensor(weights, dtype=torch.float32, device=device)

    def _evaluate_torch(self, torch, model, criterion, x, y) -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            logits = model(x)
            loss = float(criterion(logits, y).detach().cpu().item())
            predicted = torch.argmax(logits, dim=1)
            accuracy = float((predicted == y).float().mean().detach().cpu().item())
        return loss, accuracy

    @staticmethod
    def _adamw_step(
        torch,
        parameters,
        *,
        state: dict[int, tuple[Any, Any]],
        learning_rate: float,
        weight_decay: float,
        step: int,
    ) -> None:
        # Avoid torch.optim construction: some CI PyTorch wheels import dynamo/triton
        # there and can segfault before training starts.
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        bias_correction1 = 1.0 - beta1**int(step)
        bias_correction2 = 1.0 - beta2**int(step)
        step_size = float(learning_rate) * math.sqrt(bias_correction2) / bias_correction1

        with torch.no_grad():
            for parameter in parameters:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if weight_decay > 0.0:
                    parameter.mul_(1.0 - float(learning_rate) * float(weight_decay))

                key = id(parameter)
                moments = state.get(key)
                if moments is None:
                    exp_avg = torch.zeros_like(parameter)
                    exp_avg_sq = torch.zeros_like(parameter)
                    state[key] = (exp_avg, exp_avg_sq)
                else:
                    exp_avg, exp_avg_sq = moments

                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                parameter.addcdiv_(
                    exp_avg,
                    exp_avg_sq.sqrt().add_(epsilon),
                    value=-step_size,
                )
                parameter.grad = None

    def _torch_device(self, torch):
        requested = str(self.device or "cpu").strip().lower()
        if requested == "mps" and getattr(torch.backends, "mps", None) is not None:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _check_fitted(self) -> None:
        required = (
            "classes_",
            "model_state_dict_",
            "sequence_mean_",
            "sequence_std_",
            "n_channels_",
            "n_features_in_",
        )
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                f"{self.estimator_name} is not fitted; missing "
                + ", ".join(missing)
            )

    def _build_network(
        self,
        *,
        input_dim: int,
        class_count: int,
        backbone_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_bidirectional: bool,
    ):
        return _GRUBackboneNet(
            input_dim=input_dim,
            class_count=class_count,
            backbone_dim=backbone_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_bidirectional=use_bidirectional,
        )


class TorchLSTMBackboneClassifier(TorchGRUBackboneClassifier):
    """Small LSTM classifier for multivariate landmark time-series."""

    sequence_model_name = "sequence_lstm_backbone"
    estimator_name = "TorchLSTMBackboneClassifier"
    log_prefix = "lstm"

    def _build_network(
        self,
        *,
        input_dim: int,
        class_count: int,
        backbone_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_bidirectional: bool,
    ):
        return _LSTMBackboneNet(
            input_dim=input_dim,
            class_count=class_count,
            backbone_dim=backbone_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_bidirectional=use_bidirectional,
        )


def _tune_recurrent_backbone_hyperparameters(
    classifier_cls,
    X: np.ndarray,
    y: np.ndarray,
    *,
    target_frames: int = DYNAMIC_SEQUENCE_TARGET_FRAMES,
    feature_name: str = "dynamic_sequence",
    n_trials: int = 8,
    timeout: int | None = None,
    random_state: int = 42,
    max_epochs: int = 70,
    groups=None,
) -> OptunaTuningSummary:
    """Run a compact Optuna search over recurrent hyperparameters."""
    optuna = _require_optuna()
    matrix = np.asarray(X, dtype=np.float32)
    labels = np.asarray(y)
    trials = max(1, int(n_trials))
    group_values = (
        normalized_groups(groups, labels.shape[0])
        if groups is not None
        else None
    )
    group_split = (
        grouped_holdout_indices(
            labels,
            group_values,
            validation_fraction=0.25,
            random_state=int(random_state),
        )
        if group_values is not None
        else None
    )
    used_group_split = group_split is not None
    used_validation = bool(used_group_split or _can_stratified_split(labels, 0.25))
    if group_split is not None:
        train_idx, validation_idx = group_split
        train_x, val_x = matrix[train_idx], matrix[validation_idx]
        train_y, val_y = labels[train_idx], labels[validation_idx]
    elif used_validation and group_values is None:
        train_x, val_x, train_y, val_y = train_test_split(
            matrix,
            labels,
            test_size=0.25,
            random_state=int(random_state),
            stratify=labels,
        )
    else:
        train_x, val_x, train_y, val_y = matrix, matrix, labels, labels

    def objective(trial) -> float:
        params = {
            "backbone_dim": trial.suggest_categorical("backbone_dim", [32, 48, 64, 96]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [48, 64, 96, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.05, 0.35),
            "use_bidirectional": trial.suggest_categorical(
                "use_bidirectional",
                [False, True],
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                5e-4,
                3e-3,
                log=True,
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                1e-5,
                2e-3,
                log=True,
            ),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 24]),
        }
        classifier = classifier_cls(
            target_frames=int(target_frames),
            feature_name=str(feature_name),
            max_epochs=max(10, int(max_epochs)),
            validation_fraction=0.0,
            patience=max(8, int(max_epochs) // 3),
            random_state=int(random_state) + int(trial.number),
            **params,
        )
        classifier.fit(train_x, train_y)
        return float(classifier.score(val_x, val_y))

    sampler = optuna.samplers.TPESampler(seed=int(random_state))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        objective,
        n_trials=trials,
        timeout=timeout if timeout and timeout > 0 else None,
        show_progress_bar=False,
    )
    return OptunaTuningSummary(
        best_score=float(study.best_value),
        best_params=dict(study.best_params),
        trials=len(study.trials),
        used_validation_split=bool(used_validation),
        used_group_split=bool(used_group_split),
    )


def tune_gru_backbone_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    *,
    target_frames: int = DYNAMIC_SEQUENCE_TARGET_FRAMES,
    feature_name: str = "dynamic_sequence",
    n_trials: int = 8,
    timeout: int | None = None,
    random_state: int = 42,
    max_epochs: int = 70,
    groups=None,
) -> OptunaTuningSummary:
    """Run a compact Optuna search over GRU hyperparameters."""
    return _tune_recurrent_backbone_hyperparameters(
        TorchGRUBackboneClassifier,
        X,
        y,
        target_frames=target_frames,
        feature_name=feature_name,
        n_trials=n_trials,
        timeout=timeout,
        random_state=random_state,
        max_epochs=max_epochs,
        groups=groups,
    )


def tune_lstm_backbone_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    *,
    target_frames: int = DYNAMIC_SEQUENCE_TARGET_FRAMES,
    feature_name: str = "dynamic_sequence",
    n_trials: int = 8,
    timeout: int | None = None,
    random_state: int = 42,
    max_epochs: int = 70,
    groups=None,
) -> OptunaTuningSummary:
    """Run a compact Optuna search over LSTM hyperparameters."""
    return _tune_recurrent_backbone_hyperparameters(
        TorchLSTMBackboneClassifier,
        X,
        y,
        target_frames=target_frames,
        feature_name=feature_name,
        n_trials=n_trials,
        timeout=timeout,
        random_state=random_state,
        max_epochs=max_epochs,
        groups=groups,
    )


def make_dynamic_landmark_lstm_backbone_classifier(
    **kwargs: Any,
) -> TorchLSTMBackboneClassifier:
    """Return the GISLR-style 72-frame landmark-image LSTM backbone."""
    classifier = TorchLSTMBackboneClassifier(
        target_frames=DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES,
        feature_name="dynamic_landmark_image",
        **kwargs,
    )
    classifier.sequence_model_name = "dynamic_landmark_lstm_backbone"
    classifier.estimator_name = "TorchDynamicLandmarkLSTMBackboneClassifier"
    classifier.log_prefix = "landmark_lstm"
    return classifier


class _GRUBackboneNet:
    def __new__(
        cls,
        *,
        input_dim: int,
        class_count: int,
        backbone_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_bidirectional: bool,
    ):
        torch = _require_torch()

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, backbone_dim),
                    torch.nn.LayerNorm(backbone_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                )
                self.gru = torch.nn.GRU(
                    input_size=backbone_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                    bidirectional=use_bidirectional,
                )
                directions = 2 if use_bidirectional else 1
                temporal_dim = hidden_dim * directions
                self.head = torch.nn.Sequential(
                    torch.nn.LayerNorm(temporal_dim * 3),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(temporal_dim * 3, class_count),
                )

            def forward(self, x):
                embedded = self.backbone(x)
                output, _hidden = self.gru(embedded)
                final = output[:, -1, :]
                mean = output.mean(dim=1)
                max_values = output.max(dim=1).values
                features = torch.cat([final, mean, max_values], dim=1)
                return self.head(features)

        return Net()


class _LSTMBackboneNet:
    def __new__(
        cls,
        *,
        input_dim: int,
        class_count: int,
        backbone_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_bidirectional: bool,
    ):
        torch = _require_torch()

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, backbone_dim),
                    torch.nn.LayerNorm(backbone_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                )
                self.lstm = torch.nn.LSTM(
                    input_size=backbone_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                    bidirectional=use_bidirectional,
                )
                directions = 2 if use_bidirectional else 1
                temporal_dim = hidden_dim * directions
                self.head = torch.nn.Sequential(
                    torch.nn.LayerNorm(temporal_dim * 3),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(temporal_dim * 3, class_count),
                )

            def forward(self, x):
                embedded = self.backbone(x)
                output, _hidden = self.lstm(embedded)
                final = output[:, -1, :]
                mean = output.mean(dim=1)
                max_values = output.max(dim=1).values
                features = torch.cat([final, mean, max_values], dim=1)
                return self.head(features)

        return Net()


def _can_stratified_split(y: np.ndarray, validation_fraction: float) -> bool:
    labels = np.asarray(y)
    if labels.size <= 2:
        return False
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size <= 1 or np.any(counts < 2):
        return False
    validation_count = int(math.ceil(labels.size * float(validation_fraction)))
    train_count = int(labels.size) - validation_count
    return validation_count >= classes.size and train_count >= classes.size


def _seed_torch(torch, seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    try:
        torch.set_num_threads(max(1, min(4, int(torch.get_num_threads()))))
    except Exception:
        pass


def _require_torch():
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise RuntimeError(
            "sequence_gru_backbone requires PyTorch. "
            "Install it with `.venv/bin/python -m pip install torch`."
        ) from exc
    return torch


def _require_optuna():
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - exercised only without optuna
        raise RuntimeError(
            "GRU Optuna tuning requires optuna. "
            "Install it with `.venv/bin/python -m pip install optuna`."
        ) from exc
    return optuna
