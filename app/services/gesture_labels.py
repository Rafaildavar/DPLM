"""Exact gesture-label matching without semantic aliases."""

from __future__ import annotations

from collections.abc import Iterable


def gesture_label_key(label: object) -> str:
    """Return a punctuation- and case-insensitive key for one user label."""
    return "".join(
        character
        for character in str(label or "").strip().casefold()
        if character.isalnum()
    )


def gesture_labels_match(left: object, right: object) -> bool:
    """Compare spelling variants of the same label, not semantic aliases."""
    left_key = gesture_label_key(left)
    return bool(left_key and left_key == gesture_label_key(right))


def resolve_registered_gesture_label(
    label: object,
    registered_labels: Iterable[object],
) -> str | None:
    """Return the exact stored label when its normalized match is unique."""
    clean = str(label or "").strip()
    if not clean:
        return None

    registered = [
        raw
        for value in registered_labels
        if (raw := str(value or "").strip())
    ]
    if clean in registered:
        return clean

    key = gesture_label_key(clean)
    matches = [raw for raw in registered if gesture_label_key(raw) == key]
    return matches[0] if len(matches) == 1 else None
