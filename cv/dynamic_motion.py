"""Motion segmentation and time normalization for dynamic gestures."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from cv.gesture_features import (
    global_trajectory_xy,
    global_wrist_slices,
    sequence_shape_change_energy,
    sequence_to_matrix,
    trajectory_features,
)

RUNTIME_SEGMENT_PRE_ROLL_FRAMES = 3
RUNTIME_SEGMENT_ONSET_PATH = 0.015
RUNTIME_SEGMENT_ONSET_DISPLACEMENT = 0.012
RUNTIME_SEGMENT_MIN_ACTIVE_FRAMES = 3
RUNTIME_SEGMENT_COMPLETION_GRACE_FRAMES = 5
RUNTIME_SEGMENT_CONTINUATION_WAIT_FRAMES = 24


@dataclass(frozen=True)
class MotionSegmentUpdate:
    phase: str
    frames: int
    completed_sequence: np.ndarray | None = None
    end_reason: str = ""
    completion_evidence: dict[str, float] | None = None
    raw_sequence: np.ndarray | None = None


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
    min_step: float = 0.0005,
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
    resampled = resample_sequence(seq[start:end], target_frames)
    return normalize_global_trajectory(resampled)


def normalize_global_trajectory(
    sequence: np.ndarray,
    *,
    reference_displacement: float = 0.5,
) -> np.ndarray:
    """Remove screen position and projected motion amplitude from wrist paths."""
    seq = sequence_to_matrix(sequence).copy()
    dims = int(seq.shape[1])
    wrist_slices = global_wrist_slices(dims)
    if not wrist_slices:
        return seq

    target = max(1e-6, float(reference_displacement))
    for start, end in wrist_slices:
        wrist = seq[:, start:end]
        if not np.any(np.abs(wrist) > 1e-6):
            continue
        delta = wrist[-1] - wrist[0]
        displacement = float(np.linalg.norm(delta))
        if displacement <= 1e-6:
            continue
        seq[:, start:end] = (wrist - wrist[0]) * (target / displacement)
    return seq.astype(np.float32, copy=False)


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
        end_still_frames: int = 2,
        release_frames: int = 2,
        release_step_ratio: float = 0.45,
        completion_grace_frames: int = 3,
        continuation_wait_frames: int = 24,
        resume_step_ratio: float = 0.62,
        min_active_frames: int = 8,
        max_active_frames: int = 60,
        cooldown_frames: int = 8,
        onset_path_scale_ratio: float = 0.15,
        onset_displacement_scale_ratio: float = 0.10,
        still_step_scale_ratio: float = 0.025,
    ) -> None:
        self.target_frames = max(2, int(target_frames))
        self.pre_roll_frames = max(3, int(pre_roll_frames))
        self.onset_path = max(0.0, float(onset_path))
        self.onset_displacement = max(0.0, float(onset_displacement))
        self.still_step = max(0.0, float(still_step))
        self.end_still_frames = max(1, int(end_still_frames))
        self.release_frames = max(1, int(release_frames))
        self.release_step_ratio = max(0.05, min(0.95, float(release_step_ratio)))
        self.completion_grace_frames = max(0, int(completion_grace_frames))
        self.continuation_wait_frames = max(1, int(continuation_wait_frames))
        self.resume_step_ratio = max(0.05, min(0.95, float(resume_step_ratio)))
        self.min_active_frames = max(3, int(min_active_frames))
        self.max_active_frames = max(self.min_active_frames, int(max_active_frames))
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.onset_path_scale_ratio = max(0.0, float(onset_path_scale_ratio))
        self.onset_displacement_scale_ratio = max(
            0.0,
            float(onset_displacement_scale_ratio),
        )
        self.still_step_scale_ratio = max(0.0, float(still_step_scale_ratio))
        self._pre_roll: Deque[np.ndarray] = deque(maxlen=self.pre_roll_frames)
        self._pre_roll_scales: Deque[float] = deque(maxlen=self.pre_roll_frames)
        self._active: list[np.ndarray] = []
        self._active_scales: list[float] = []
        self._still_frames = 0
        self._release_frames = 0
        self._peak_step = 0.0
        self._peak_shape_step = 0.0
        self._cooldown_remaining = 0
        self._pending_completion_reason = ""
        self._pending_completion_sequence: np.ndarray | None = None
        self._completion_grace_remaining = 0
        self._awaiting_continuation = False
        self._motion_confirmed = False
        self._continuation_wait_remaining = 0

    @property
    def phase(self) -> str:
        if self._cooldown_remaining > 0:
            return "cooldown"
        if self._awaiting_continuation:
            return "awaiting_continuation"
        if self._active:
            return "active"
        if len(self._pre_roll) < self.pre_roll_frames:
            return "warming_up"
        return "idle"

    @property
    def active_frames(self) -> int:
        return len(self._active)

    def reset(self) -> None:
        self._pre_roll.clear()
        self._pre_roll_scales.clear()
        self._active.clear()
        self._active_scales.clear()
        self._still_frames = 0
        self._release_frames = 0
        self._peak_step = 0.0
        self._peak_shape_step = 0.0
        self._cooldown_remaining = 0
        self._pending_completion_reason = ""
        self._pending_completion_sequence = None
        self._completion_grace_remaining = 0
        self._awaiting_continuation = False
        self._motion_confirmed = False
        self._continuation_wait_remaining = 0

    def reject_completed_candidate(
        self,
        sequence: np.ndarray,
        *,
        motion_scale: float = 0.0,
    ) -> None:
        """Preserve a rejected prefix and wait for deliberate continuation."""
        raw = sequence_to_matrix(sequence).astype(np.float32, copy=False)
        self.reset()
        self._active = [frame.copy() for frame in raw]
        scale = max(0.0, float(motion_scale))
        self._active_scales = [scale] * len(self._active)
        points = _global_points(raw)
        steps = (
            np.linalg.norm(np.diff(points, axis=0), axis=1)
            if points.shape[0] > 1
            else np.asarray([], dtype=np.float32)
        )
        self._peak_step = float(np.max(steps)) if steps.size else 0.0
        shape_steps = [
            sequence_shape_change_energy(raw[index - 1 : index + 1])
            for index in range(1, raw.shape[0])
        ]
        self._peak_shape_step = max(shape_steps, default=0.0)
        self._awaiting_continuation = True
        self._motion_confirmed = True
        self._continuation_wait_remaining = self.continuation_wait_frames

    def finish_due_to_hand_lost(self) -> MotionSegmentUpdate:
        """Complete an active swipe when the hand leaves the frame naturally."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                self._pre_roll.clear()
            return MotionSegmentUpdate("cooldown", 0, end_reason="hand_lost")
        if self._awaiting_continuation:
            self.reset()
            return MotionSegmentUpdate(
                "idle",
                0,
                end_reason="incomplete_rejected",
            )
        if not self._active:
            self.reset()
            return MotionSegmentUpdate("idle", 0, end_reason="hand_lost")
        if self._pending_completion_sequence is not None:
            return self._complete_active(
                end_reason=self._pending_completion_reason or "hand_lost",
                sequence=self._pending_completion_sequence,
            )

        sequence = np.stack(self._active, axis=0)
        enough_frames = len(self._active) >= self.min_active_frames
        if not enough_frames or not (
            self._motion_confirmed
            or self._motion_sufficient(sequence, self._active_scales)
        ):
            self.reset()
            return MotionSegmentUpdate("idle", 0, end_reason="hand_lost_rejected")
        return self._complete_active(end_reason="hand_lost")

    def update(
        self,
        frame: np.ndarray,
        *,
        motion_scale: float | None = None,
    ) -> MotionSegmentUpdate:
        vector = np.asarray(frame, dtype=np.float32).reshape(-1)
        if vector.size <= 0 or not np.isfinite(vector).all():
            self.reset()
            return MotionSegmentUpdate("warming_up", 0)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                self._pre_roll.clear()
            return MotionSegmentUpdate("cooldown", 0)

        if self._awaiting_continuation and self._active:
            previous = self._active[-1]
            current_scale = max(0.0, float(motion_scale or 0.0))
            step_points = _global_points(np.stack([previous, vector], axis=0))
            step = float(np.linalg.norm(step_points[1] - step_points[0]))
            shape_step = sequence_shape_change_energy(
                np.stack([previous, vector], axis=0)
            )
            still_threshold = self._scale_aware_threshold(
                current_scale,
                ratio=self.still_step_scale_ratio,
                fallback=self.still_step,
                minimum=0.002,
            )
            resume_threshold = max(
                still_threshold * 1.5,
                float(self._peak_step) * self.resume_step_ratio,
            )
            shape_resume_threshold = self._shape_resume_threshold(
                self._peak_shape_step
            )
            if (
                step <= resume_threshold
                and shape_step <= shape_resume_threshold
            ):
                self._continuation_wait_remaining -= 1
                if self._continuation_wait_remaining <= 0:
                    self.reset()
                    self._pre_roll.append(vector.copy())
                    self._pre_roll_scales.append(current_scale)
                    return MotionSegmentUpdate(
                        "idle",
                        0,
                        end_reason="completion_timeout",
                    )
                return MotionSegmentUpdate(
                    "awaiting_continuation",
                    len(self._active),
                )
            self._active.append(vector)
            self._active_scales.append(current_scale)
            self._peak_step = max(self._peak_step, step)
            self._peak_shape_step = max(self._peak_shape_step, shape_step)
            self._still_frames = 0
            self._release_frames = 0
            self._awaiting_continuation = False
            self._motion_confirmed = True
            self._continuation_wait_remaining = 0
            return MotionSegmentUpdate("active", len(self._active))

        if not self._active:
            self._pre_roll.append(vector)
            self._pre_roll_scales.append(max(0.0, float(motion_scale or 0.0)))
            if len(self._pre_roll) < self.pre_roll_frames:
                return MotionSegmentUpdate("warming_up", len(self._pre_roll))
            if not self._onset_detected():
                return MotionSegmentUpdate("idle", len(self._pre_roll))
            self._active = [item.copy() for item in self._pre_roll]
            self._active_scales = list(self._pre_roll_scales)
            self._still_frames = 0
            self._release_frames = 0
            self._peak_step = 0.0
            self._peak_shape_step = 0.0
            self._motion_confirmed = True
            return MotionSegmentUpdate("active", len(self._active))

        previous = self._active[-1]
        self._active.append(vector)
        current_scale = max(0.0, float(motion_scale or 0.0))
        self._active_scales.append(current_scale)
        step_points = _global_points(np.stack([previous, vector], axis=0))
        step = float(np.linalg.norm(step_points[1] - step_points[0]))
        shape_step = sequence_shape_change_energy(
            np.stack([previous, vector], axis=0)
        )
        self._peak_step = max(self._peak_step, step)
        self._peak_shape_step = max(self._peak_shape_step, shape_step)
        still_threshold = self._scale_aware_threshold(
            current_scale,
            ratio=self.still_step_scale_ratio,
            fallback=self.still_step,
            minimum=0.002,
        )
        shape_release_threshold = self._shape_release_threshold(
            self._peak_shape_step
        )
        is_still = (
            step <= still_threshold
            and shape_step <= shape_release_threshold
        )
        self._still_frames = self._still_frames + 1 if is_still else 0
        enough_frames = len(self._active) >= self.min_active_frames
        sequence = np.stack(self._active, axis=0)
        sufficient_motion = enough_frames and (
            self._motion_confirmed
            or self._motion_sufficient(sequence, self._active_scales)
        )
        release_threshold = max(
            still_threshold,
            float(self._peak_step) * self.release_step_ratio,
        )
        resume_threshold = max(
            still_threshold * 1.5,
            float(self._peak_step) * self.resume_step_ratio,
        )
        shape_resume_threshold = self._shape_resume_threshold(
            self._peak_shape_step
        )

        if self._pending_completion_sequence is not None:
            if (
                step > resume_threshold
                or shape_step > shape_resume_threshold
            ):
                self._pending_completion_reason = ""
                self._pending_completion_sequence = None
                self._completion_grace_remaining = 0
                self._still_frames = 0
                self._release_frames = 0
            else:
                self._completion_grace_remaining -= 1
                if self._completion_grace_remaining <= 0:
                    return self._complete_active(
                        end_reason=self._pending_completion_reason or "velocity_drop",
                        sequence=self._pending_completion_sequence,
                    )
                return MotionSegmentUpdate("active", len(self._active))

        if (
            sufficient_motion
            and step <= release_threshold
            and shape_step <= shape_release_threshold
        ):
            self._release_frames += 1
        else:
            self._release_frames = 0

        ended_by_still = sufficient_motion and self._still_frames >= self.end_still_frames
        ended_by_release = sufficient_motion and self._release_frames >= self.release_frames
        reached_limit = len(self._active) >= self.max_active_frames
        if not ended_by_still and not ended_by_release and not reached_limit:
            return MotionSegmentUpdate("active", len(self._active))

        if ended_by_still:
            end_reason = "still"
            keep = max(2, sequence.shape[0] - self._still_frames + 1)
            return self._defer_completion(
                end_reason=end_reason,
                sequence=sequence[:keep],
            )
        if ended_by_release:
            return self._defer_completion(
                end_reason="velocity_drop",
                sequence=sequence,
            )
        return self._complete_active(end_reason="max_frames", sequence=sequence)

    def _defer_completion(
        self,
        *,
        end_reason: str,
        sequence: np.ndarray,
    ) -> MotionSegmentUpdate:
        if self.completion_grace_frames <= 0:
            return self._complete_active(end_reason=end_reason, sequence=sequence)
        self._pending_completion_reason = str(end_reason or "velocity_drop")
        self._pending_completion_sequence = np.asarray(
            sequence,
            dtype=np.float32,
        ).copy()
        self._completion_grace_remaining = self.completion_grace_frames
        return MotionSegmentUpdate("active", len(self._active))

    def _complete_active(
        self,
        *,
        end_reason: str,
        sequence: np.ndarray | None = None,
    ) -> MotionSegmentUpdate:
        if sequence is None:
            sequence = np.stack(self._active, axis=0)
        raw_sequence = np.asarray(sequence, dtype=np.float32)
        from cv.dynamic_completion import build_dynamic_completion_evidence

        completion_evidence = build_dynamic_completion_evidence(
            raw_sequence,
            motion_scale=self._active_scales[: raw_sequence.shape[0]],
            target_frames=self.target_frames,
        )
        completed = canonical_dynamic_sequence(
            raw_sequence,
            target_frames=self.target_frames,
        )
        self._active.clear()
        self._active_scales.clear()
        self._pre_roll.clear()
        self._pre_roll_scales.clear()
        self._still_frames = 0
        self._release_frames = 0
        self._peak_step = 0.0
        self._peak_shape_step = 0.0
        self._pending_completion_reason = ""
        self._pending_completion_sequence = None
        self._completion_grace_remaining = 0
        self._awaiting_continuation = False
        self._motion_confirmed = False
        self._continuation_wait_remaining = 0
        self._cooldown_remaining = self.cooldown_frames
        return MotionSegmentUpdate(
            "completed",
            int(raw_sequence.shape[0]),
            completed_sequence=completed,
            end_reason=end_reason,
            completion_evidence=completion_evidence,
            raw_sequence=raw_sequence.copy(),
        )

    def _onset_detected(self) -> bool:
        sequence = np.stack(tuple(self._pre_roll), axis=0)
        return self._motion_sufficient(sequence, self._pre_roll_scales)

    def _motion_sufficient(
        self,
        sequence: np.ndarray,
        scales: list[float] | Deque[float],
    ) -> bool:
        motion = trajectory_features(sequence, target_dim=sequence.shape[1])
        displacement = float(np.hypot(float(motion[0]), float(motion[1])))
        valid_scales = [float(value) for value in scales if float(value) > 1e-6]
        scale = float(np.median(valid_scales)) if valid_scales else 0.0
        path_threshold = self._scale_aware_threshold(
            scale,
            ratio=self.onset_path_scale_ratio,
            fallback=self.onset_path,
            minimum=0.012,
        )
        displacement_threshold = self._scale_aware_threshold(
            scale,
            ratio=self.onset_displacement_scale_ratio,
            fallback=self.onset_displacement,
            minimum=0.008,
        )
        return (
            float(motion[4]) >= path_threshold
            and displacement >= displacement_threshold
        )

    @staticmethod
    def _scale_aware_threshold(
        scale: float,
        *,
        ratio: float,
        fallback: float,
        minimum: float,
    ) -> float:
        if scale <= 1e-6:
            return fallback
        return min(fallback, max(minimum, scale * ratio))

    @staticmethod
    def _shape_release_threshold(peak_shape_step: float) -> float:
        return max(0.004, min(0.020, float(peak_shape_step) * 0.45))

    @staticmethod
    def _shape_resume_threshold(peak_shape_step: float) -> float:
        return max(0.008, min(0.030, float(peak_shape_step) * 0.62))


def _global_points(sequence: np.ndarray) -> np.ndarray:
    return global_trajectory_xy(sequence)


def create_runtime_dynamic_segmenter(*, target_frames: int) -> DynamicMotionSegmenter:
    """Build the shared segmenter used by live inference and profile training."""
    return DynamicMotionSegmenter(
        target_frames=max(2, int(target_frames)),
        pre_roll_frames=RUNTIME_SEGMENT_PRE_ROLL_FRAMES,
        onset_path=RUNTIME_SEGMENT_ONSET_PATH,
        onset_displacement=RUNTIME_SEGMENT_ONSET_DISPLACEMENT,
        min_active_frames=RUNTIME_SEGMENT_MIN_ACTIVE_FRAMES,
        completion_grace_frames=RUNTIME_SEGMENT_COMPLETION_GRACE_FRAMES,
        continuation_wait_frames=RUNTIME_SEGMENT_CONTINUATION_WAIT_FRAMES,
        max_active_frames=max(60, int(target_frames)),
    )
