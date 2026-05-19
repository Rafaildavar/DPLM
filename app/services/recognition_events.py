"""Recognition event persistence and listing helpers."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models.database import Command, Gesture, RecognitionLog


def record_recognition_event(
    session: Session,
    *,
    label: str,
    confidence: float,
    executed: bool,
    model_id: int | None = None,
    app_session_id: int | None = None,
    commit: bool = True,
) -> RecognitionLog:
    clean_label = (label or "").strip()
    gesture = (
        session.query(Gesture)
        .filter(Gesture.label == clean_label)
        .first()
        if clean_label
        else None
    )
    row = RecognitionLog(
        label=clean_label,
        confidence=float(confidence or 0.0),
        model_id=model_id,
        gesture_id=gesture.id if gesture is not None else None,
        session_id=app_session_id,
        executed=bool(executed),
    )
    session.add(row)
    if commit:
        session.commit()
    else:
        session.flush()
    return row


def list_recent_recognition_events(
    session: Session,
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit or 12), 100))
    rows = (
        session.query(RecognitionLog)
        .order_by(RecognitionLog.detected_at.desc(), RecognitionLog.id.desc())
        .limit(safe_limit)
        .all()
    )
    command_by_gesture_id = {
        int(command.gesture_id): command.name
        for command in session.query(Command).filter(Command.gesture_id.isnot(None)).all()
        if command.gesture_id is not None
    }
    return [_row_to_dict(row, command_by_gesture_id) for row in rows]


def _row_to_dict(row: RecognitionLog, command_by_gesture_id: dict[int, str]) -> dict[str, Any]:
    detected_at = getattr(row, "detected_at", None)
    if isinstance(detected_at, datetime):
        detected_text = detected_at.strftime("%H:%M:%S")
        detected_iso = detected_at.isoformat()
    else:
        detected_text = ""
        detected_iso = ""
    gesture_id = int(row.gesture_id) if row.gesture_id is not None else None
    return {
        "id": int(row.id or 0),
        "label": str(row.label or ""),
        "confidence": float(row.confidence or 0.0),
        "executed": bool(row.executed),
        "detectedAt": detected_text,
        "detectedAtIso": detected_iso,
        "gestureId": gesture_id,
        "commandName": command_by_gesture_id.get(gesture_id or -1, ""),
    }
