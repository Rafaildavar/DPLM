"""PII/credential minimization helpers for agent observability."""
from __future__ import annotations

import re
from typing import Any


REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
}
_SECRET_PATTERNS = (
    re.compile(
        r"(?i)(\b(?:api[_-]?key|token|secret|password|пароль|ключ)\s*[:=]\s*)\S{6,}"
    ),
    re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(
        r"\b[A-Za-z0-9_-]{32,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b"
    ),
)


def contains_secret(value: str) -> bool:
    raw = str(value or "")
    return any(pattern.search(raw) for pattern in _SECRET_PATTERNS)


def redact_text(value: str) -> str:
    clean = str(value or "")
    for index, pattern in enumerate(_SECRET_PATTERNS):
        if index < 2:
            clean = pattern.sub(lambda match: match.group(1) + REDACTED, clean)
        else:
            clean = pattern.sub(REDACTED, clean)
    return clean


def redact_payload(value: Any, *, key: str = "") -> Any:
    if key.strip().lower() in _SENSITIVE_KEYS:
        return REDACTED
    if isinstance(value, dict):
        return {
            str(item_key): redact_payload(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    if isinstance(value, tuple):
        return [redact_payload(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


__all__ = ["REDACTED", "contains_secret", "redact_payload", "redact_text"]
