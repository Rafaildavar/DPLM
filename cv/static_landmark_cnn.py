"""TensorFlow CNN for GISLR-style landmark images.

The estimator consumes flattened landmark-image features
(``time x points x xyz``) and keeps only numpy weights after fitting, so it can
be saved with joblib like the sklearn models used by the app.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from cv.gesture_features import (
    DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES,
    STATIC_LANDMARK_IMAGE_TARGET_FRAMES,
)


class KerasStaticLandmarkCNNClassifier(ClassifierMixin, BaseEstimator):
    """Small depthwise CNN over hand landmark tensors."""

    def __init__(
        self,
        *,
        target_frames: int = STATIC_LANDMARK_IMAGE_TARGET_FRAMES,
        channels: int = 3,
        conv_filters: int = 48,
        dense_units: int = 64,
        dropout: float = 0.20,
        learning_rate: float = 1e-3,
        label_smoothing: float = 0.05,
        max_epochs: int = 120,
        batch_size: int = 16,
        validation_fraction: float = 0.20,
        patience: int = 18,
        random_state: int = 42,
        verbose: int = 0,
        feature_name: str = "static_landmark_cnn",
    ) -> None:
        self.target_frames = target_frames
        self.channels = channels
        self.conv_filters = conv_filters
        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.label_smoothing = label_smoothing
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.feature_name = feature_name

    def fit(self, X, y):  # noqa: D401 - sklearn API
        tf = _require_tensorflow()
        matrix = self._validate_X(X)
        labels = np.asarray(y)
        if labels.shape[0] != matrix.shape[0]:
            raise ValueError("X and y have different sample counts")
        if labels.size == 0:
            raise ValueError("empty training labels")

        self.classes_, encoded = np.unique(labels, return_inverse=True)
        self.n_features_in_ = int(matrix.shape[1])
        self.n_points_ = int(
            self.n_features_in_ // (int(self.target_frames) * int(self.channels))
        )
        images = matrix.reshape(
            matrix.shape[0],
            int(self.target_frames),
            int(self.n_points_),
            int(self.channels),
        )
        images = self._fit_normalizer(images)

        X_train, X_val, y_train, y_val = self._split_train_validation(images, encoded)
        y_train_one_hot = _one_hot(tf, y_train, int(self.classes_.size))
        y_val_one_hot = (
            _one_hot(tf, y_val, int(self.classes_.size)) if y_val.size else None
        )

        _seed_tensorflow(tf, int(self.random_state))
        model = self._build_model(tf)
        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=max(0.0, min(0.30, float(self.label_smoothing)))
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=max(1e-6, float(self.learning_rate))
            ),
            loss=loss,
            metrics=["accuracy"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if y_val.size else "loss",
                patience=max(1, int(self.patience)),
                restore_best_weights=True,
            )
        ]
        validation_data = (X_val, y_val_one_hot) if y_val.size else None
        history = model.fit(
            X_train,
            y_train_one_hot,
            sample_weight=self._sample_weights(y_train),
            validation_data=validation_data,
            epochs=max(1, int(self.max_epochs)),
            batch_size=max(1, min(int(self.batch_size), int(X_train.shape[0]))),
            callbacks=callbacks,
            verbose=int(self.verbose),
        )

        self.model_weights_ = [
            np.asarray(weight, dtype=np.float32) for weight in model.get_weights()
        ]
        self.training_history_ = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        self.used_validation_split_ = bool(y_val.size)
        return self

    def predict_proba(self, X):  # noqa: D401 - sklearn API
        tf = _require_tensorflow()
        check_is_fitted(
            self,
            [
                "classes_",
                "model_weights_",
                "image_mean_",
                "image_std_",
                "n_features_in_",
                "n_points_",
            ],
        )
        matrix = self._validate_X(X)
        if int(matrix.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                "feature dimension mismatch: "
                f"{matrix.shape[1]} != {self.n_features_in_}"
            )
        images = matrix.reshape(
            matrix.shape[0],
            int(self.target_frames),
            int(self.n_points_),
            int(self.channels),
        )
        images = self._transform_normalizer(images)

        model = self._build_model(tf)
        model.set_weights(self.model_weights_)
        probabilities = model.predict(images, verbose=0)
        return np.asarray(probabilities, dtype=np.float32)

    def predict(self, X):  # noqa: D401 - sklearn API
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def _validate_X(self, X) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"expected 2D feature matrix, got shape={matrix.shape}")
        target_frames = max(1, int(self.target_frames))
        channels = max(1, int(self.channels))
        divisor = target_frames * channels
        if matrix.shape[1] <= 0 or matrix.shape[1] % divisor != 0:
            raise ValueError(
                f"{getattr(self, 'feature_name', 'static_landmark_cnn')} expects "
                "flattened landmark-image "
                f"features divisible by {divisor}; got {matrix.shape[1]}"
            )
        if not np.isfinite(matrix).all():
            matrix = np.nan_to_num(matrix, copy=False)
        return matrix

    def _fit_normalizer(self, images: np.ndarray) -> np.ndarray:
        mean = images.mean(axis=(0, 1, 2), keepdims=True)
        std = images.std(axis=(0, 1, 2), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        self.image_mean_ = mean.astype(np.float32, copy=False)
        self.image_std_ = std.astype(np.float32, copy=False)
        return self._transform_normalizer(images)

    def _transform_normalizer(self, images: np.ndarray) -> np.ndarray:
        return ((images - self.image_mean_) / self.image_std_).astype(
            np.float32,
            copy=False,
        )

    def _split_train_validation(
        self,
        images: np.ndarray,
        encoded: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fraction = max(0.0, min(0.50, float(self.validation_fraction)))
        if fraction <= 0.0 or not _can_stratified_split(encoded, fraction):
            empty_x = np.empty((0,) + images.shape[1:], dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.int64)
            return images, empty_x, encoded.astype(np.int64), empty_y
        return train_test_split(
            images,
            encoded.astype(np.int64),
            test_size=fraction,
            random_state=int(self.random_state),
            stratify=encoded,
        )

    def _sample_weights(self, encoded: np.ndarray) -> np.ndarray:
        counts = np.bincount(encoded.astype(np.int64), minlength=int(self.classes_.size))
        counts = np.maximum(counts, 1)
        weights = counts.sum() / (len(counts) * counts)
        return weights[encoded.astype(np.int64)].astype(np.float32, copy=False)

    def _build_model(self, tf):
        layers = tf.keras.layers
        inputs = tf.keras.Input(
            shape=(int(self.target_frames), int(self.n_points_), int(self.channels))
        )
        x = layers.GaussianNoise(0.015)(inputs)
        x = layers.Conv2D(
            max(8, int(self.conv_filters)),
            kernel_size=(5, 3),
            padding="same",
            use_bias=False,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)
        x = layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,
        )(x)
        x = layers.Conv2D(
            max(16, int(self.conv_filters) * 2),
            kernel_size=1,
            padding="same",
            use_bias=False,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)
        x = layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,
        )(x)
        x = layers.Conv2D(
            max(24, int(self.conv_filters) * 2),
            kernel_size=1,
            padding="same",
            use_bias=False,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(max(0.0, min(0.60, float(self.dropout))))(x)
        x = layers.Dense(max(8, int(self.dense_units)), activation="swish")(x)
        x = layers.Dropout(max(0.0, min(0.60, float(self.dropout))))(x)
        outputs = layers.Dense(int(self.classes_.size), activation="softmax")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)


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


def _one_hot(tf, y: np.ndarray, class_count: int):
    return tf.keras.utils.to_categorical(
        np.asarray(y, dtype=np.int64),
        num_classes=max(1, int(class_count)),
    )


def _seed_tensorflow(tf, seed: int) -> None:
    np.random.seed(int(seed))
    try:
        tf.keras.utils.set_random_seed(int(seed))
    except Exception:
        tf.random.set_seed(int(seed))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "TensorFlow is required for landmark-image CNN models. "
            "Install tensorflow or choose a sklearn model."
        ) from exc
    return tf


def make_dynamic_landmark_cnn_classifier(**kwargs: Any) -> KerasStaticLandmarkCNNClassifier:
    """Return the GISLR-style dynamic landmark-image CNN benchmark."""
    return KerasStaticLandmarkCNNClassifier(
        target_frames=DYNAMIC_LANDMARK_IMAGE_TARGET_FRAMES,
        feature_name="dynamic_landmark_cnn",
        **kwargs,
    )
