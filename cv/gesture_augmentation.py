"""GISLR-style landmark augmentations for recorded gesture samples.

The transforms in this module are adapted to GestureBind's hand-only samples.
They intentionally preserve gesture direction: no left/right flip is applied,
because a mirrored dynamic gesture can change the command semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

HAND_ROUTES: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
HAND_TREES: tuple[tuple[int, ...], ...] = tuple(
    tuple(route[start:])
    for route in HAND_ROUTES
    for start in range(len(route) - 1)
)


@dataclass(frozen=True)
class GISLRAugmentationConfig:
    affine_prob: float = 0.85
    affine_scale: tuple[float, float] = (0.86, 1.16)
    affine_shift: tuple[float, float] = (-0.045, 0.045)
    affine_degrees: tuple[float, float] = (-8.0, 8.0)
    time_warp_prob: float = 0.45
    time_warp_scale: tuple[float, float] = (0.86, 1.18)
    time_warp_shift: tuple[float, float] = (-0.35, 0.35)
    part_scale_prob: float = 0.45
    part_scale: tuple[float, float] = (0.90, 1.12)
    joint_rotate_prob: float = 0.65
    joint_degrees: tuple[float, float] = (-4.0, 4.0)
    joint_tree_prob: float = 0.35
    point_mask_prob: float = 0.20
    point_mask_fraction: tuple[float, float] = (0.02, 0.08)
    time_mask_prob: float = 0.18
    noise_prob: float = 0.55
    noise_sigma: float = 0.004
    global_motion_scale: tuple[float, float] = (0.88, 1.14)
    global_motion_shift: tuple[float, float] = (-0.018, 0.018)
    global_motion_noise_sigma: float = 0.002


DEFAULT_GISLR_AUGMENTATION_CONFIG = GISLRAugmentationConfig()


@dataclass(frozen=True)
class _HandBlock:
    pose_start: int
    pose_dim: int = 42
    coords_per_point: int = 2
    global_start: int | None = None

    @property
    def pose_end(self) -> int:
        return self.pose_start + self.pose_dim


def augment_gislr_landmark_sequence(
    sequence: np.ndarray,
    *,
    seed: int | None = None,
    include_global_motion: bool = False,
    config: GISLRAugmentationConfig = DEFAULT_GISLR_AUGMENTATION_CONFIG,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return one class-preserving augmented copy of a gesture sequence."""
    source = np.asarray(sequence, dtype=np.float32)
    if source.ndim not in {2, 3}:
        raise ValueError(f"expected 2D or 3D sequence, got shape={source.shape}")
    if source.shape[0] <= 0:
        raise ValueError("empty gesture sequence")

    rng = np.random.default_rng(seed)
    flat = _sequence_to_flat_matrix(source).copy()
    blocks = _hand_blocks_for_matrix(
        flat,
        original_ndim=source.ndim,
        include_global_motion=include_global_motion,
    )
    applied: list[str] = []

    if flat.shape[0] > 2 and rng.random() < config.time_warp_prob:
        flat = _time_warp_fixed(
            flat,
            rng=rng,
            scale_range=config.time_warp_scale,
            shift_range=config.time_warp_shift,
        )
        applied.append("time_warp")

    for block_index, block in enumerate(blocks):
        points = flat[:, block.pose_start : block.pose_end].reshape(
            flat.shape[0],
            21,
            block.coords_per_point,
        ).copy()
        if not _has_visible_hand(points):
            continue

        block_ops: list[str] = []
        if rng.random() < config.affine_prob:
            points = _random_affine_points(
                points,
                rng=rng,
                scale_range=config.affine_scale,
                shift_range=config.affine_shift,
                degree_range=config.affine_degrees,
            )
            block_ops.append("affine")

        if rng.random() < config.part_scale_prob:
            points = _scale_hand_part(points, rng=rng, scale_range=config.part_scale)
            block_ops.append("part_scale")

        if rng.random() < config.joint_rotate_prob:
            points = _random_hand_joint_rotate(
                points,
                rng=rng,
                degree_range=config.joint_degrees,
                joint_prob=config.joint_tree_prob,
            )
            block_ops.append("joint_rotate")

        if rng.random() < config.noise_prob:
            points = points + rng.normal(0.0, config.noise_sigma, size=points.shape).astype(np.float32)
            block_ops.append("noise")

        if rng.random() < config.point_mask_prob:
            points = _mask_random_points(
                points,
                rng=rng,
                fraction_range=config.point_mask_fraction,
            )
            block_ops.append("point_mask")

        if flat.shape[0] > 4 and rng.random() < config.time_mask_prob:
            points = _mask_short_time_span(points, rng=rng)
            block_ops.append("time_mask")

        flat[:, block.pose_start : block.pose_end] = points.reshape(
            flat.shape[0],
            block.pose_dim,
        )
        if block_ops:
            applied.append(f"hand{block_index}:" + "+".join(block_ops))

        if block.global_start is not None:
            wrist = flat[:, block.global_start : block.global_start + 2]
            if np.any(np.abs(wrist) > 1e-6):
                flat[:, block.global_start : block.global_start + 2] = _augment_global_wrist(
                    wrist,
                    rng=rng,
                    scale_range=config.global_motion_scale,
                    shift_range=config.global_motion_shift,
                    noise_sigma=config.global_motion_noise_sigma,
                )
                applied.append(f"hand{block_index}:global_motion")

    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    augmented = _restore_shape(flat, source)
    metadata = {
        "policy": "gislr_landmark_v1",
        "seed": int(seed) if seed is not None else None,
        "include_global_motion": bool(include_global_motion),
        "applied": applied,
    }
    return augmented, metadata


def _sequence_to_flat_matrix(sequence: np.ndarray) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])
    return arr.reshape(arr.shape[0], -1)


def _restore_shape(flat: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if reference.ndim == 3:
        return flat.reshape(reference.shape).astype(np.float32, copy=False)
    return flat.reshape(reference.shape).astype(np.float32, copy=False)


def _hand_blocks_for_matrix(
    flat: np.ndarray,
    *,
    original_ndim: int,
    include_global_motion: bool,
) -> list[_HandBlock]:
    dims = int(flat.shape[1])
    if original_ndim == 3 and dims % 42 == 0:
        return [_HandBlock(start) for start in range(0, dims, 42)]
    if include_global_motion and dims % 65 == 0:
        return [
            _HandBlock(start, pose_dim=63, coords_per_point=3, global_start=start + 63)
            for start in range(0, dims, 65)
        ]
    if include_global_motion and dims % 44 == 0:
        return [
            _HandBlock(start, global_start=start + 42)
            for start in range(0, dims, 44)
        ]
    if dims % 63 == 0:
        return [
            _HandBlock(start, pose_dim=63, coords_per_point=3)
            for start in range(0, dims, 63)
        ]
    if dims % 42 == 0:
        return [_HandBlock(start) for start in range(0, dims, 42)]
    return []


def _has_visible_hand(points: np.ndarray) -> bool:
    return bool(np.any(np.abs(points) > 1e-6))


def _time_warp_fixed(
    matrix: np.ndarray,
    *,
    rng: np.random.Generator,
    scale_range: tuple[float, float],
    shift_range: tuple[float, float],
) -> np.ndarray:
    frames = int(matrix.shape[0])
    if frames <= 2:
        return matrix
    old_positions = np.arange(frames, dtype=np.float32)
    normalized = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    scale = float(rng.uniform(*scale_range))
    shift = float(rng.uniform(*shift_range))
    warped = ((normalized - 0.5) * scale + 0.5) * float(frames - 1) + shift
    warped = np.clip(warped, 0.0, float(frames - 1))
    out = np.empty_like(matrix)
    for column in range(matrix.shape[1]):
        out[:, column] = np.interp(warped, old_positions, matrix[:, column])
    return out.astype(np.float32, copy=False)


def _random_affine_points(
    points: np.ndarray,
    *,
    rng: np.random.Generator,
    scale_range: tuple[float, float],
    shift_range: tuple[float, float],
    degree_range: tuple[float, float],
) -> np.ndarray:
    scale = float(rng.uniform(*scale_range))
    shift = rng.uniform(*shift_range, size=(1, 1, 2)).astype(np.float32)
    degree = float(rng.uniform(*degree_range))
    radian = degree / 180.0 * np.pi
    c = float(np.cos(radian))
    s = float(np.sin(radian))
    rotate = np.asarray([[c, -s], [s, c]], dtype=np.float32).T
    out = points.copy()
    out = out * scale
    out[..., :2] = out[..., :2] @ rotate
    out[..., :2] = out[..., :2] + shift
    return out.astype(np.float32, copy=False)


def _scale_hand_part(
    points: np.ndarray,
    *,
    rng: np.random.Generator,
    scale_range: tuple[float, float],
) -> np.ndarray:
    center = np.mean(points, axis=1, keepdims=True)
    scale = float(rng.uniform(*scale_range))
    return (center + (points - center) * scale).astype(np.float32, copy=False)


def _random_hand_joint_rotate(
    points: np.ndarray,
    *,
    rng: np.random.Generator,
    degree_range: tuple[float, float],
    joint_prob: float,
) -> np.ndarray:
    out = points.copy()
    for tree in HAND_TREES:
        if rng.random() >= joint_prob or len(tree) < 2:
            continue
        alpha = float(rng.uniform(*degree_range))
        center = out[:, tree[0:1], :2]
        out[:, tree[1:], :2] = _rotate_points(out[:, tree[1:], :2], center, alpha)
    return out.astype(np.float32, copy=False)


def _rotate_points(pos: np.ndarray, center: np.ndarray, alpha: float) -> np.ndarray:
    radian = alpha / 180.0 * np.pi
    c = float(np.cos(radian))
    s = float(np.sin(radian))
    rotation_matrix = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    translated = (pos - center).reshape(-1, 2)
    rotated = np.dot(rotation_matrix, translated.T).T.reshape(pos.shape)
    return rotated + center


def _mask_random_points(
    points: np.ndarray,
    *,
    rng: np.random.Generator,
    fraction_range: tuple[float, float],
) -> np.ndarray:
    out = points.copy()
    count = max(1, int(round(points.shape[1] * float(rng.uniform(*fraction_range)))))
    indices = rng.choice(points.shape[1], size=min(count, points.shape[1]), replace=False)
    out[:, indices, :] = 0.0
    return out


def _mask_short_time_span(points: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    out = points.copy()
    frames = int(points.shape[0])
    span = max(1, min(frames // 8, int(rng.integers(1, max(2, frames // 6 + 1)))))
    start = int(rng.integers(0, max(1, frames - span + 1)))
    if start > 0:
        out[start : start + span] = out[start - 1]
    elif start + span < frames:
        out[start : start + span] = out[start + span]
    else:
        out[start : start + span] = 0.0
    return out


def _augment_global_wrist(
    wrist: np.ndarray,
    *,
    rng: np.random.Generator,
    scale_range: tuple[float, float],
    shift_range: tuple[float, float],
    noise_sigma: float,
) -> np.ndarray:
    origin = wrist[0:1]
    delta = wrist - origin
    scale = float(rng.uniform(*scale_range))
    shift = rng.uniform(*shift_range, size=(1, 2)).astype(np.float32)
    noise = rng.normal(0.0, noise_sigma, size=wrist.shape).astype(np.float32)
    return (origin + shift + delta * scale + noise).astype(np.float32, copy=False)
