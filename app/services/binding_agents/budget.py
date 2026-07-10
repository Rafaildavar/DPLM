"""Shared latency budget for one interactive MAS request."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass


PIPELINE_TIMEOUT_ENV = "DPLM_BINDING_AGENT_TIMEOUT_SECONDS"
DEFAULT_PIPELINE_TIMEOUT_SECONDS = 12.0


def configured_timeout() -> float:
    raw = str(os.getenv(PIPELINE_TIMEOUT_ENV) or "").strip().replace(",", ".")
    try:
        value = float(raw) if raw else DEFAULT_PIPELINE_TIMEOUT_SECONDS
    except ValueError:
        value = DEFAULT_PIPELINE_TIMEOUT_SECONDS
    return min(60.0, max(2.0, value))


@dataclass(frozen=True)
class PipelineBudget:
    started_at: float
    deadline: float

    @classmethod
    def start(cls, timeout: float | None = None) -> "PipelineBudget":
        started = time.perf_counter()
        duration = configured_timeout() if timeout is None else max(0.1, float(timeout))
        return cls(started_at=started, deadline=started + duration)

    @classmethod
    def from_deadline(
        cls,
        *,
        started_at: float,
        deadline: float,
    ) -> "PipelineBudget":
        now = time.perf_counter()
        return cls(started_at=started_at or now, deadline=deadline or now)

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline - time.perf_counter())

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000.0, 1)

    def timeout_for(self, cap: float, *, reserve: float = 0.15) -> float:
        return max(0.0, min(float(cap), self.remaining - max(0.0, reserve)))


__all__ = [
    "DEFAULT_PIPELINE_TIMEOUT_SECONDS",
    "PIPELINE_TIMEOUT_ENV",
    "PipelineBudget",
    "configured_timeout",
]
