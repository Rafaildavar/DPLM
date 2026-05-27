"""State for capturing one timed gesture example from the camera stream."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecordingCycle:
    sequence_length: int
    countdown_seconds: float = 3.0
    frames: list[Any] = field(default_factory=list)
    recording: bool = False
    starts_at: float | None = None

    def __post_init__(self) -> None:
        self.sequence_length = max(1, int(self.sequence_length))
        self.countdown_seconds = max(0.0, float(self.countdown_seconds))

    @property
    def ready(self) -> bool:
        return len(self.frames) >= self.sequence_length

    @property
    def counting_down(self) -> bool:
        return self.starts_at is not None

    def start(self, now: float) -> None:
        self.frames.clear()
        self.recording = False
        self.starts_at = now + self.countdown_seconds

    def countdown_value(self, now: float) -> int | None:
        if self.starts_at is None:
            return None
        return max(1, int(math.ceil(self.starts_at - now)))

    def capture(self, frame: Any, now: float) -> tuple[bool, bool]:
        """Return `(started_now, became_ready)` after accepting a camera frame."""
        started_now = False
        if self.starts_at is not None:
            if now < self.starts_at:
                return False, False
            self.starts_at = None
            self.recording = True
            started_now = True

        if not self.recording or self.ready:
            return started_now, False

        self.frames.append(frame)
        if self.ready:
            self.recording = False
            return started_now, True
        return started_now, False

    def clear(self) -> None:
        self.frames.clear()
        self.recording = False
        self.starts_at = None
