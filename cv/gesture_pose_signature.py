"""Small pose-signature helpers for rejecting near-miss gestures.

The KNN model works with landmark geometry, so it always returns the nearest
known class. These helpers add a cheap semantic guard: how many non-thumb
fingers are currently extended.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from cv.gesture_dataset_files import gesture_sample_paths


LANDMARKS_PER_HAND = 21
SIGNATURE_VERSION = 1

_FINGER_DEFS = (
    ("index", 5, 6, 7, 8),
    ("middle", 9, 10, 11, 12),
    ("ring", 13, 14, 15, 16),
    ("pinky", 17, 18, 19, 20),
)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _angle_degrees(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    bax = float(a[0]) - float(b[0])
    bay = float(a[1]) - float(b[1])
    bcx = float(c[0]) - float(b[0])
    bcy = float(c[1]) - float(b[1])
    norm_a = math.hypot(bax, bay)
    norm_c = math.hypot(bcx, bcy)
    if norm_a <= 1e-6 or norm_c <= 1e-6:
        return 0.0
    cosine = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (norm_a * norm_c)))
    return math.degrees(math.acos(cosine))


def _split_hands(frame: np.ndarray | Sequence[Sequence[float]], max_hands: int) -> list[np.ndarray]:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            return []
        points = arr.reshape(-1, 2)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        points = arr
    elif arr.ndim == 2 and arr.shape[1] != 2:
        if arr.size % 2 != 0:
            return []
        points = arr.reshape(-1, 2)
    else:
        return []

    hands: list[np.ndarray] = []
    for start in range(0, min(len(points), LANDMARKS_PER_HAND * max(1, max_hands)), LANDMARKS_PER_HAND):
        hand = points[start : start + LANDMARKS_PER_HAND]
        if hand.shape == (LANDMARKS_PER_HAND, 2):
            hands.append(hand)
    return hands


def _finger_extended(points: np.ndarray, mcp_idx: int, pip_idx: int, dip_idx: int, tip_idx: int) -> bool:
    mcp = points[mcp_idx]
    pip = points[pip_idx]
    dip = points[dip_idx]
    tip = points[tip_idx]

    finger_len = _distance(mcp, pip) + _distance(pip, dip) + _distance(dip, tip)
    if finger_len <= 1e-6:
        return False

    extension_ratio = _distance(mcp, tip) / finger_len
    pip_angle = _angle_degrees(mcp, pip, dip)
    dip_angle = _angle_degrees(pip, dip, tip)
    straight_enough = extension_ratio >= 0.58 and min(pip_angle, dip_angle) >= 132.0
    tip_ahead_of_knuckles = tip[1] <= pip[1] + 0.04 and tip[1] <= mcp[1] + 0.02
    return straight_enough or (extension_ratio >= 0.50 and tip_ahead_of_knuckles)


def _finger_raised(points: np.ndarray, mcp_idx: int, pip_idx: int, dip_idx: int, tip_idx: int) -> bool:
    mcp = points[mcp_idx]
    pip = points[pip_idx]
    tip = points[tip_idx]
    simple_raise = tip[1] < mcp[1] - 0.03 and tip[1] <= pip[1] + 0.08
    return simple_raise and (
        _finger_extended(points, mcp_idx, pip_idx, dip_idx, tip_idx)
        or _distance(mcp, tip) >= 0.10
    )


def hand_non_thumb_mask(landmarks: np.ndarray | Sequence[Sequence[float]]) -> int:
    """Return a 4-bit mask for index/middle/ring/pinky extended fingers."""
    hands = _split_hands(landmarks, max_hands=1)
    if not hands:
        return 0

    points = hands[0]
    mask = 0
    for bit, (_, mcp_idx, pip_idx, dip_idx, tip_idx) in enumerate(_FINGER_DEFS):
        if _finger_raised(points, mcp_idx, pip_idx, dip_idx, tip_idx):
            mask |= 1 << bit
    return mask


def frame_non_thumb_count(
    frame: np.ndarray | Sequence[Sequence[float]],
    *,
    max_hands: int = 1,
) -> int | None:
    hands = _split_hands(frame, max_hands=max_hands)
    if not hands:
        return None
    return sum(int(hand_non_thumb_mask(hand)).bit_count() for hand in hands)


def hands_non_thumb_count(
    hands_landmarks: Iterable[Sequence[Sequence[float]]],
    *,
    max_hands: int = 1,
) -> int | None:
    counts: list[int] = []
    for landmarks in list(hands_landmarks)[: max(1, max_hands)]:
        count = frame_non_thumb_count(landmarks, max_hands=1)
        if count is not None:
            counts.append(count)
    if not counts:
        return None
    return sum(counts)


def sample_non_thumb_count(sample: np.ndarray, *, max_hands: int = 1) -> tuple[int | None, float, int]:
    arr = np.asarray(sample, dtype=np.float32)
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 2):
        frames = [arr]
    elif arr.ndim >= 2:
        frames = [arr[i] for i in range(arr.shape[0])]
    else:
        frames = []

    counts = [
        count
        for count in (frame_non_thumb_count(frame, max_hands=max_hands) for frame in frames)
        if count is not None
    ]
    if not counts:
        return None, 0.0, 0

    counter = Counter(counts)
    count, support = counter.most_common(1)[0]
    return int(count), float(support / len(counts)), len(counts)


def class_signature_from_sample_paths(
    sample_paths: Iterable[Path],
    *,
    max_hands: int = 1,
) -> dict[str, int | float] | None:
    counts: list[int] = []
    total_frames = 0
    for path in sample_paths:
        try:
            sample = np.load(path)
        except Exception:
            continue
        count, _stability, frames = sample_non_thumb_count(sample, max_hands=max_hands)
        if count is None:
            continue
        counts.append(count)
        total_frames += frames

    if not counts:
        return None

    counter = Counter(counts)
    count, support = counter.most_common(1)[0]
    return {
        "non_thumb_count": int(count),
        "stability": float(support / len(counts)),
        "samples": int(len(counts)),
        "frames": int(total_frames),
    }


def build_signature_metadata(
    class_sample_paths: Mapping[str, Iterable[Path]],
    *,
    max_hands: int = 1,
) -> dict[str, object]:
    classes: dict[str, object] = {}
    for label, sample_paths in class_sample_paths.items():
        signature = class_signature_from_sample_paths(sample_paths, max_hands=max_hands)
        if signature is not None:
            classes[str(label)] = signature
    return {
        "version": SIGNATURE_VERSION,
        "type": "non_thumb_extended_count",
        "max_hands": int(max_hands),
        "classes": classes,
    }


def build_signature_metadata_from_data_root(
    data_root: Path,
    classes: Sequence[str],
    *,
    max_hands: int = 1,
) -> dict[str, object]:
    if not data_root.exists():
        return build_signature_metadata({}, max_hands=max_hands)

    label_dirs = {p.name: p for p in data_root.iterdir() if p.is_dir()}
    label_dirs_lower = {name.lower(): path for name, path in label_dirs.items()}
    class_sample_paths: dict[str, list[Path]] = {}
    for label in classes:
        label_dir = label_dirs.get(label) or label_dirs_lower.get(label.lower())
        if label_dir is None:
            continue
        class_sample_paths[label] = gesture_sample_paths(label_dir)
    return build_signature_metadata(class_sample_paths, max_hands=max_hands)


def load_signature_metadata(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}
