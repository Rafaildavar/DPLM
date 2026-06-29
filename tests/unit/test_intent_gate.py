import numpy as np

from app.services.gesture_taxonomy import (
    GESTURE_TYPE_DYNAMIC,
    GESTURE_TYPE_NEGATIVE,
    GESTURE_TYPE_QUASI_STATIC,
    GESTURE_TYPE_STATIC,
)
from cv.intent_gate import (
    DEFAULT_INTENT_TARGET_DIM,
    INTENT_DYNAMIC,
    INTENT_NONE,
    INTENT_STATIC,
    build_intent_feature_vector,
    predict_intent_gate,
)
from scripts.train_intent_gate import (
    IntentRecord,
    build_intent_matrix,
    map_gesture_type_to_intent,
)


def test_intent_feature_vector_has_fixed_compact_size():
    static = np.zeros((5, 42), dtype=np.float32)
    dynamic = np.zeros((5, 44), dtype=np.float32)
    dynamic[:, -2] = [0.1, 0.2, 0.4, 0.7, 0.9]

    static_feature = build_intent_feature_vector(static, target_dim=DEFAULT_INTENT_TARGET_DIM)
    dynamic_feature = build_intent_feature_vector(dynamic, target_dim=DEFAULT_INTENT_TARGET_DIM)

    assert static_feature.shape == dynamic_feature.shape
    assert static_feature.shape[0] >= 20
    assert dynamic_feature[18] > static_feature[18]


def test_taxonomy_gesture_types_map_to_intents():
    assert map_gesture_type_to_intent(GESTURE_TYPE_STATIC) == INTENT_STATIC
    assert map_gesture_type_to_intent(GESTURE_TYPE_QUASI_STATIC) == INTENT_STATIC
    assert map_gesture_type_to_intent(GESTURE_TYPE_DYNAMIC) == INTENT_DYNAMIC
    assert map_gesture_type_to_intent(GESTURE_TYPE_NEGATIVE) == INTENT_NONE


def test_build_intent_matrix_uses_stable_label_order():
    records = [
        IntentRecord(
            label="palm",
            intent=INTENT_STATIC,
            path=__file__,
            sequence=np.zeros((4, 42), dtype=np.float32),
            source="test",
        ),
        IntentRecord(
            label="swipe_up",
            intent=INTENT_DYNAMIC,
            path=__file__,
            sequence=np.ones((4, 44), dtype=np.float32),
            source="test",
        ),
        IntentRecord(
            label="random_motion",
            intent=INTENT_NONE,
            path=__file__,
            sequence=np.full((4, 44), 2.0, dtype=np.float32),
            source="test",
        ),
    ]

    X, y, labels = build_intent_matrix(records, target_dim=44)

    assert X.shape[0] == 3
    assert labels == [INTENT_STATIC, INTENT_DYNAMIC, INTENT_NONE]
    assert y.tolist() == [0, 1, 2]


def test_predict_intent_gate_supports_predict_proba_model():
    class Model:
        classes_ = np.asarray([INTENT_STATIC, INTENT_DYNAMIC, INTENT_NONE])

        def predict_proba(self, _X):
            return np.asarray([[0.1, 0.8, 0.1]], dtype=float)

    decision = predict_intent_gate({"model": Model()}, np.zeros(25, dtype=np.float32))

    assert decision["label"] == INTENT_DYNAMIC
    assert decision["confidence"] == 0.8
    assert abs(decision["margin"] - 0.7) < 1e-9


def test_predict_intent_gate_decodes_numeric_sklearn_classes():
    class Model:
        classes_ = np.asarray([0, 1, 2])

        def predict_proba(self, _X):
            return np.asarray([[0.1, 0.8, 0.1]], dtype=float)

    decision = predict_intent_gate(
        {"model": Model(), "classes": [INTENT_STATIC, INTENT_DYNAMIC, INTENT_NONE]},
        np.zeros(25, dtype=np.float32),
    )

    assert decision["label"] == INTENT_DYNAMIC
    assert decision["probabilities"][INTENT_DYNAMIC] == 0.8
