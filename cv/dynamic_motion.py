"""Motion segmentation and time normalization for dynamic gestures."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from cv.gesture_features import sequence_to_matrix, trajectory_features


@dataclass(frozen=True)
class MotionSegmentUpdate:
    phase: str
    frames: int
    completed_sequence: np.ndarray | None = None


def resample_sequence(sequence: np.ndarray, target_frames: int = 36) -> np.ndarray:
    """Linearly resample a variable-length sequence to a fixed frame count."""
    seq = sequence_to_matrix(sequence)
    target = max(2, int(target_frames))
    if seq.shape[0] == target:
        return seq.astype(np.float32, copy=True)
    old_positions = np.linspace(0.0, 1.0, seq.shape[0], dtype=np.float32)
    new_positions = np.linspace(0.0, 1.0, target, dtype=np.float32)
    columns = [
        np.interp(new_positions, old_positions, seq[:, index])
        for index in range(seq.shape[1])
    ]
    return np.stack(columns, axis=1).astype(np.float32, copy=False)


def canonical_dynamic_sequence(
    sequence: np.ndarray,
    *,
    target_frames: int = 36,
    min_step: float = 0.003,
    relative_step: float = 0.15,
    edge_context_frames: int = 2,
) -> np.ndarray:
    """Trim static edges and normalize gesture duration.

    Global wrist coordinates are preferred when present. The fallback inside
    ``trajectory_features`` keeps legacy 42-dimensional samples usable.
    """
    seq = sequence_to_matrix(sequence)
    if seq.shape[0] <= 2:
        return resample_sequence(seq, target_frames)

    points = _global_points(seq)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not np.any(np.isfinite(steps)) or float(np.max(steps)) <= 1e-8:
        return resample_sequence(seq, target_frames)

    threshold = max(float(min_step), float(np.max(steps)) * float(relative_step))
    moving_steps = np.flatnonzero(steps >= threshold)
    if moving_steps.size == 0:
        return resample_sequence(seq, target_frames)

    context = max(0, int(edge_context_frames))
    start = max(0, int(moving_steps[0]) - context)
    end = min(seq.shape[0], int(moving_steps[-1]) + 2 + context)
    if end - start < 2:
        return resample_sequence(seq, target_frames)
    return resample_sequence(seq[start:end], target_frames)


class DynamicMotionSegmenter:
    """Detect one deliberate motion and emit one normalized sequence."""

    def __init__(
        self,
        *,
        target_frames: int = 36,
        pre_roll_frames: int = 5,
        onset_path: float = 0.04,
        onset_displacement: float = 0.025,
        still_step: float = 0.006,
        end_still_frames: int = 5,
        min_active_frames: int = 8,
        max_active_frames: int = 60,
        cooldown_frames: int = 8,
    ) -> None:
        self.target_frames = max(2, int(target_frames))
        self.pre_roll_frames = max(3, int(pre_roll_frames))
        self.onset_path = max(0.0, float(onset_path))
        self.onset_displacement = max(0.0, float(onset_displacement))
        self.still_step = max(0.0, float(still_step))
        self.end_still_frames = max(1, int(end_still_frames))
        self.min_active_frames = max(3, int(min_active_frames))
        self.max_active_frames = max(self.min_active_frames, int(max_active_frames))
        self.cooldown_frames = max(0, int(cooldown_frames))
        self._pre_roll: Deque[np.ndarray] = deque(maxlen=self.pre_roll_frames)
        self._active: list[np.ndarray] = []
        self._still_frames = 0
        self._cooldown_remaining = 0

    @property
    def phase(self) -> str:
        if self._cooldown_remaining > 0:
            return "cooldown"
        if self._active:
            return "active"
        if len(self._pre_roll) < self.pre_roll_frames:
            return "warming_up"
        return "idle"

    def reset(self) -> None:
        self._pre_roll.clear()
        self._active.clear()
        self._still_frames = 0
        self._cooldown_remaining = 0

    def update(self, frame: np.ndarray) -> MotionSegmentUpdate:
        vector = np.asarray(frame, dtype=np.float32).reshape(-1)
        if vector.size <= 0 or not np.isfinite(vector).all():
            self.reset()
            return MotionSegmentUpdate("warming_up", 0)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                self._pre_roll.clear()
            return MotionSegmentUpdate("cooldown", 0)

        if not self._active:
            self._pre_roll.append(vector)
            if len(self._pre_roll) < self.pre_roll_frames:
                return MotionSegmentUpdate("warming_up", len(self._pre_roll))
            if not self._onset_detected():
                return MotionSegmentUpdate("idle", len(self._pre_roll))
            self._active = [item.copy() for item in self._pre_roll]
            self._still_frames = 0
            return MotionSegmentUpdate("active", len(self._active))

        previous = self._active[-1]
        self._active.append(vector)
        step_points = _global_points(np.stack([previous, vector], axis=0))
        step = float(np.linalg.norm(step_points[1] - step_points[0]))
        self._still_frames = self._still_frames + 1 if step <= self.still_step else 0
        enough_frames = len(self._active) >= self.min_active_frames
        ended = enough_frames and self._still_frames >= self.end_still_frames
        reached_limit = len(self._active) >= self.max_active_frames
        if not ended and not reached_limit:
            return MotionSegmentUpdate("active", len(self._active))

        sequence = np.stack(self._active, axis=0)
        if ended:
            keep = max(2, sequence.shape[0] - self._still_frames + 1)
            sequence = sequence[:keep]
        completed = canonical_dynamic_sequence(
            sequence,
            target_frames=self.target_frames,
        )
        self._active.clear()
        self._pre_roll.clear()
        self._still_frames = 0
        self._cooldown_remaining = self.cooldown_frames
        return MotionSegmentUpdate(
            "completed",
            int(sequence.shape[0]),
            completed_sequence=completed,
        )

    def _onset_detected(self) -> bool:
        sequence = np.stack(tuple(self._pre_roll), axis=0)
        motion = trajectory_features(sequence, target_dim=sequence.shape[1])
        displacement = float(np.hypot(float(motion[0]), float(motion[1])))
        return (
            float(motion[4]) >= self.onset_path
            and displacement >= self.onset_displacement
        )


def _global_points(sequence: np.ndarray) -> np.ndarray:
    seq = sequence_to_matrix(sequence)
    dims = int(seq.shape[1])
    if dims >= 44 and dims % 44 == 0:
        wrists: list[np.ndarray] = []
        for offset in range(0, dims, 44):
            wrist = seq[:, offset + 42 : offset + 44]
            if np.any(np.abs(wrist) > 1e-6):
                wrists.append(wrist)
        if wrists:
            return np.mean(np.stack(wrists, axis=0), axis=0).astype(
                np.float32,
                copy=False,
            )

    usable = (dims // 2) * 2
    if usable >= 2:
        points = seq[:, :usable].reshape(seq.shape[0], usable // 2, 2)
        return points.mean(axis=1).astype(np.float32, copy=False)
    return np.zeros((seq.shape[0], 2), dtype=np.float32)
