"""Small state machine for live gesture confirmation feedback."""

from __future__ import annotations

from dataclasses import dataclass


PHASE_IDLE = "idle"
PHASE_DISABLED = "disabled"
PHASE_PENDING = "pending"
PHASE_CONFIRMED = "confirmed"
PHASE_REJECTED = "rejected"
PHASE_SUPPRESSED = "suppressed"
PHASE_COOLDOWN = "cooldown"


@dataclass(frozen=True)
class LiveGestureSnapshot:
    phase: str = PHASE_IDLE
    label: str = ""
    confidence: float = 0.0
    progress: float = 0.0
    frames: int = 0
    required_frames: int = 0
    reason: str = ""
    route: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "label": self.label,
            "confidence": self.confidence,
            "progress": self.progress,
            "frames": self.frames,
            "requiredFrames": self.required_frames,
            "reason": self.reason,
            "route": self.route,
            "displayText": self.display_text,
        }

    @property
    def display_text(self) -> str:
        if self.phase == PHASE_PENDING and self.label:
            return f"{self.label} {self.frames}/{max(1, self.required_frames)}"
        if self.phase == PHASE_REJECTED and self.label:
            return f"{self.label} rejected"
        if self.phase == PHASE_SUPPRESSED and self.label:
            return f"{self.label} suppressed"
        if self.phase == PHASE_COOLDOWN and self.label:
            return f"{self.label} cooldown"
        return self.label


class LiveGestureState:
    """Tracks candidate stability separately from command execution.

    The recognizer can emit noisy frame-by-frame candidates. This object keeps
    only the confirmation counter and the current UI-facing phase. It does not
    know anything about databases, commands, or classifiers.
    """

    def __init__(self) -> None:
        self._pending_label = ""
        self._pending_frames = 0
        self._pending_confidence_total = 0.0
        self._snapshot = LiveGestureSnapshot()

    @property
    def pending_label(self) -> str:
        return self._pending_label

    @property
    def pending_frames(self) -> int:
        return self._pending_frames

    @property
    def pending_confidence_total(self) -> float:
        return self._pending_confidence_total

    @property
    def snapshot(self) -> LiveGestureSnapshot:
        return self._snapshot

    def reset(self, *, phase: str = PHASE_IDLE, reason: str = "") -> LiveGestureSnapshot:
        self._pending_label = ""
        self._pending_frames = 0
        self._pending_confidence_total = 0.0
        self._snapshot = LiveGestureSnapshot(phase=phase, reason=reason)
        return self._snapshot

    def observe(
        self,
        label: str,
        confidence: float,
        *,
        required_frames: int,
        route: str = "",
        reason: str = "",
    ) -> tuple[bool, float, LiveGestureSnapshot]:
        clean = str(label or "").strip()
        if not clean:
            snapshot = self.reset(reason=reason)
            return False, 0.0, snapshot

        required = max(1, int(required_frames))
        conf = max(0.0, min(1.0, float(confidence or 0.0)))

        if clean != self._pending_label:
            self._pending_label = clean
            self._pending_frames = 1
            self._pending_confidence_total = conf
        else:
            self._pending_frames += 1
            self._pending_confidence_total += conf

        stable_confidence = self._pending_confidence_total / max(1, self._pending_frames)
        confirmed = self._pending_frames >= required
        phase = PHASE_CONFIRMED if confirmed else PHASE_PENDING
        self._snapshot = LiveGestureSnapshot(
            phase=phase,
            label=clean,
            confidence=stable_confidence,
            progress=min(1.0, self._pending_frames / float(required)),
            frames=self._pending_frames,
            required_frames=required,
            reason=reason,
            route=route,
        )
        return confirmed, stable_confidence, self._snapshot

    def mark_confirmed(
        self,
        label: str,
        confidence: float,
        *,
        route: str = "",
        reason: str = "",
    ) -> LiveGestureSnapshot:
        self._snapshot = LiveGestureSnapshot(
            phase=PHASE_CONFIRMED,
            label=str(label or "").strip(),
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            progress=1.0,
            frames=self._pending_frames,
            required_frames=max(1, self._pending_frames),
            reason=reason,
            route=route,
        )
        return self._snapshot

    def mark_rejected(
        self,
        label: str,
        confidence: float,
        *,
        reason: str,
        route: str = "",
    ) -> LiveGestureSnapshot:
        self._snapshot = LiveGestureSnapshot(
            phase=PHASE_REJECTED,
            label=str(label or "").strip(),
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            progress=1.0,
            frames=self._pending_frames,
            required_frames=max(1, self._pending_frames),
            reason=str(reason or ""),
            route=route,
        )
        return self._snapshot

    def mark_suppressed(
        self,
        label: str,
        confidence: float,
        *,
        reason: str,
        route: str = "",
    ) -> LiveGestureSnapshot:
        self._snapshot = LiveGestureSnapshot(
            phase=PHASE_SUPPRESSED,
            label=str(label or "").strip(),
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            progress=1.0,
            frames=self._pending_frames,
            required_frames=max(1, self._pending_frames),
            reason=str(reason or ""),
            route=route,
        )
        return self._snapshot

    def mark_cooldown(
        self,
        label: str,
        confidence: float,
        *,
        reason: str = "",
        route: str = "",
    ) -> LiveGestureSnapshot:
        self._snapshot = LiveGestureSnapshot(
            phase=PHASE_COOLDOWN,
            label=str(label or "").strip(),
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            progress=1.0,
            frames=self._pending_frames,
            required_frames=max(1, self._pending_frames),
            reason=str(reason or ""),
            route=route,
        )
        return self._snapshot
