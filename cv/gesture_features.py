from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


LANDMARK_COUNT = 21
HAND_LABELS = ("Left", "Right")
FINGER_CHAINS = [
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
]


@dataclass
class HandFrame:
    label: str
    landmarks_xy: np.ndarray


def normalize_landmarks(landmarks_xy: Sequence[Tuple[float, float]], rotation_invariant: bool = True) -> np.ndarray:
    """Normalize 2D landmarks (21x2) by wrist-centered scaling and optional rotation."""
    pts = np.asarray(landmarks_xy, dtype=np.float32)
    if pts.shape != (LANDMARK_COUNT, 2):
        raise ValueError("Ожидалось 21 точка (x, y)")

    pts = pts.copy()
    wrist = pts[0].copy()
    pts -= wrist

    scale = float(np.max(np.linalg.norm(pts, axis=1)))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale

    if rotation_invariant:
        ref = pts[5]  # index_mcp
        angle = float(np.arctan2(ref[1], ref[0]))
        rot = np.array(
            [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]],
            dtype=np.float32,
        )
        pts = pts @ rot.T

    return pts


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return 0.0
    cos_v = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.arccos(cos_v))


def hand_geometry_features(norm_pts: np.ndarray) -> np.ndarray:
    """Five finger bend angles + five tip-to-wrist distances."""
    if norm_pts.shape != (LANDMARK_COUNT, 2):
        raise ValueError("Ожидалось 21x2 нормализованных точек")

    angles = [_joint_angle(norm_pts[i1], norm_pts[i2], norm_pts[i3]) for (_, i1, i2, i3) in FINGER_CHAINS]
    tip_to_wrist = [float(np.linalg.norm(norm_pts[tip] - norm_pts[0])) for (_, _, _, tip) in FINGER_CHAINS]
    return np.asarray(angles + tip_to_wrist, dtype=np.float32)


def build_frame_feature(
    hand_landmarks: Sequence[np.ndarray],
    handedness_labels: Optional[Sequence[str]] = None,
    two_hands: bool = False,
    include_presence_mask: bool = True,
) -> np.ndarray:
    """Build stable frame feature vector with optional Left/Right ordering and presence mask."""
    labels = list(handedness_labels or [""] * len(hand_landmarks))

    normalized: List[HandFrame] = []
    for idx, points in enumerate(hand_landmarks):
        label = labels[idx] if idx < len(labels) else ""
        norm_pts = normalize_landmarks(points)
        normalized.append(HandFrame(label=label, landmarks_xy=norm_pts))

    if not two_hands:
        if normalized:
            one = normalized[0].landmarks_xy
            geom = hand_geometry_features(one)
            base = np.concatenate([one.reshape(-1), geom], axis=0)
        else:
            base = np.zeros(LANDMARK_COUNT * 2 + 10, dtype=np.float32)
        if include_presence_mask:
            return np.concatenate([base, np.asarray([1.0 if normalized else 0.0], dtype=np.float32)], axis=0)
        return base

    by_label = {"Left": None, "Right": None}
    unknown = [h.landmarks_xy for h in normalized if h.label not in by_label]
    for h in normalized:
        if h.label in by_label and by_label[h.label] is None:
            by_label[h.label] = h.landmarks_xy

    for side in HAND_LABELS:
        if by_label[side] is None and unknown:
            by_label[side] = unknown.pop(0)

    present = []
    chunks = []
    for side in HAND_LABELS:
        pts = by_label[side]
        if pts is None:
            pts = np.zeros((LANDMARK_COUNT, 2), dtype=np.float32)
            geom = np.zeros(10, dtype=np.float32)
            present.append(0.0)
        else:
            geom = hand_geometry_features(pts)
            present.append(1.0)
        chunks.append(np.concatenate([pts.reshape(-1), geom], axis=0))

    feat = np.concatenate(chunks, axis=0)
    if include_presence_mask:
        feat = np.concatenate([feat, np.asarray(present, dtype=np.float32)], axis=0)
    return feat.astype(np.float32, copy=False)
