import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from cv.gesture_features import build_frame_feature, hand_geometry_features, normalize_landmarks
from cv.train_classifier import build_model



def _sample_hand(offset: float = 0.0) -> np.ndarray:
    xs = np.linspace(0.1 + offset, 0.9 + offset, 21, dtype=np.float32)
    ys = np.linspace(0.2, 0.8, 21, dtype=np.float32)
    return np.stack([xs, ys], axis=1)


def test_normalize_landmarks_centers_wrist():
    pts = _sample_hand()
    norm = normalize_landmarks(pts)
    assert norm.shape == (21, 2)
    assert np.allclose(norm[0], np.zeros(2, dtype=np.float32), atol=1e-6)


def test_rotation_invariant_normalization_stabilizes_orientation():
    pts = _sample_hand()
    rotated = pts[:, ::-1].copy()
    rotated[:, 1] *= -1
    norm_a = normalize_landmarks(pts, rotation_invariant=True)
    norm_b = normalize_landmarks(rotated, rotation_invariant=True)
    assert np.allclose(norm_a, norm_b, atol=1e-4)


def test_hand_geometry_features_returns_expected_dimension():
    pts = normalize_landmarks(_sample_hand())
    geom = hand_geometry_features(pts)
    assert geom.shape == (10,)
    assert np.all(geom >= 0)


def test_build_frame_feature_two_hands_respects_left_right_order_and_presence_mask():
    left = _sample_hand(0.0)
    right = _sample_hand(0.2)
    feat = build_frame_feature([right, left], handedness_labels=["Right", "Left"], two_hands=True)
    assert feat.shape == (106,)
    assert np.allclose(feat[-2:], np.array([1.0, 1.0], dtype=np.float32))


def test_build_frame_feature_two_hands_pads_missing_hand_and_presence_mask():
    left = _sample_hand(0.0)
    feat = build_frame_feature([left], handedness_labels=["Left"], two_hands=True)
    assert feat.shape == (106,)
    assert feat[-2] == 1.0
    assert feat[-1] == 0.0


def test_train_pipeline_uses_scaler_and_distance_weighted_knn():
    model = build_model(neighbors=3)
    assert "scaler" in model.named_steps
    assert model.named_steps["knn"].weights == "distance"
    assert model.named_steps["knn"].n_neighbors == 3
