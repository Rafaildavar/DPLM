"""Privacy-safe daily usage telemetry for the desktop application."""
from __future__ import annotations

import hashlib
import json
import platform
import threading
import time
import urllib.request
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


TELEMETRY_SCHEMA_VERSION = 1
DEFAULT_INTERVAL_SECONDS = 24 * 60 * 60
DEFAULT_RETRY_SECONDS = 60 * 60
DEFAULT_POLL_SECONDS = 60
MAX_JSONL_CHUNK_BYTES = 5 * 1024 * 1024
MAX_ERROR_LENGTH = 240

Transport = Callable[[str, dict[str, Any], dict[str, str], float], None]
Clock = Callable[[], float]


@dataclass(frozen=True)
class JsonlChunk:
    rows: list[dict[str, Any]]
    start_offset: int
    end_offset: int


def _read_jsonl_chunk(
    path: Path,
    offset: int,
    *,
    max_bytes: int = MAX_JSONL_CHUNK_BYTES,
) -> JsonlChunk:
    try:
        size = path.stat().st_size
    except OSError:
        return JsonlChunk([], 0, 0)

    start = int(offset or 0)
    if start < 0 or start > size:
        start = 0
    if start == size:
        return JsonlChunk([], start, start)

    try:
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(max(1, int(max_bytes)))
    except OSError:
        return JsonlChunk([], start, start)

    last_newline = data.rfind(b"\n")
    if last_newline < 0:
        return JsonlChunk([], start, start)
    complete = data[: last_newline + 1]
    rows: list[dict[str, Any]] = []
    for raw_line in complete.splitlines():
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return JsonlChunk(rows, start, start + len(complete))


def _bounded_name(value: Any, allowed: set[str]) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else "other"


def _counter(rows: list[dict[str, Any]], key: str, allowed: set[str]) -> dict[str, int]:
    values = Counter(_bounded_name(row.get(key), allowed) for row in rows)
    return dict(sorted(values.items()))


def _weighted_average(rows: list[dict[str, Any]], key: str) -> float:
    weighted_total = 0.0
    sample_total = 0
    for row in rows:
        try:
            samples = max(1, int(row.get("samples") or 1))
            value = float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
        weighted_total += value * samples
        sample_total += samples
    return weighted_total / sample_total if sample_total else 0.0


def _aggregate_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    event_types = {
        "gesture_confirmed",
        "gesture_rejected",
        "gesture_suppressed",
        "command_executed",
        "command_rejected",
        "command_cooldown",
        "recognition_feedback",
    }
    routes = {"static", "dynamic", "none"}
    model_modes = {"auto", "static", "dynamic"}
    filtered = [
        row
        for row in rows
        if str(row.get("event_type") or "").strip().lower() in event_types
    ]
    confidence_values: list[float] = []
    confidence_bins = Counter()
    feedback = Counter()
    for row in filtered:
        try:
            confidence = float(row.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence > 0.0:
            confidence_values.append(confidence)
            if confidence < 0.80:
                confidence_bins["below_0_80"] += 1
            elif confidence < 0.90:
                confidence_bins["from_0_80_to_0_90"] += 1
            elif confidence < 0.95:
                confidence_bins["from_0_90_to_0_95"] += 1
            else:
                confidence_bins["at_least_0_95"] += 1
        if str(row.get("event_type") or "") == "recognition_feedback":
            verdict = _bounded_name(row.get("feedback"), {"correct", "incorrect", "missed"})
            feedback[verdict] += 1

    command_attempts = sum(
        1
        for row in filtered
        if str(row.get("event_type") or "")
        in {"command_executed", "command_rejected", "command_cooldown"}
    )
    executed = sum(
        1 for row in filtered if str(row.get("event_type") or "") == "command_executed"
    )
    return {
        "event_count": len(filtered),
        "command_attempts": command_attempts,
        "executed_commands": executed,
        "command_success_rate": round(executed / command_attempts, 4)
        if command_attempts
        else 0.0,
        "average_confidence": round(
            sum(confidence_values) / len(confidence_values), 4
        )
        if confidence_values
        else 0.0,
        "event_type_counts": _counter(filtered, "event_type", event_types),
        "route_counts": _counter(filtered, "route", routes),
        "model_mode_counts": _counter(
            filtered, "recognition_model_mode", model_modes
        ),
        "confidence_bins": dict(sorted(confidence_bins.items())),
        "feedback_counts": dict(sorted(feedback.items())),
    }


def _aggregate_runtime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = 0
    p95_values: list[float] = []
    for row in rows:
        try:
            samples += max(0, int(row.get("samples") or 0))
            p95_values.append(float(row.get("inference_ms_p95") or 0.0))
        except (TypeError, ValueError):
            continue
    return {
        "window_count": len(rows),
        "sample_count": samples,
        "inference_ms_average": round(_weighted_average(rows, "inference_ms_avg"), 3),
        "inference_ms_p95_max": round(max(p95_values), 3) if p95_values else 0.0,
        "hand_detected_rate": round(
            _weighted_average(rows, "hand_detected_rate"), 4
        ),
    }


def _default_transport(
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> None:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = int(getattr(response, "status", 200) or 200)
        if status < 200 or status >= 300:
            raise RuntimeError(f"telemetry endpoint returned HTTP {status}")


def telemetry_endpoint_is_safe(endpoint: str) -> bool:
    parsed = urlparse(str(endpoint or "").strip())
    if parsed.scheme == "https" and bool(parsed.netloc):
        return True
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}


class DailyUsageTelemetry:
    """Uploads sanitized daily aggregates while retaining an offline cursor."""

    def __init__(
        self,
        *,
        log_dir: Path,
        endpoint: str,
        enabled: bool,
        project_key: str = "",
        app_version: str = "unknown",
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        retry_seconds: float = DEFAULT_RETRY_SECONDS,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        timeout_seconds: float = 10.0,
        transport: Transport | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.endpoint = str(endpoint or "").strip()
        self.enabled = bool(enabled)
        self.project_key = str(project_key or "").strip()
        self.app_version = str(app_version or "unknown").strip() or "unknown"
        self.interval_seconds = max(60.0, float(interval_seconds))
        self.retry_seconds = max(10.0, float(retry_seconds))
        self.poll_seconds = max(1.0, float(poll_seconds))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._transport = transport or _default_transport
        self._clock = clock or time.time
        self._state_path = self.log_dir / "telemetry_state.json"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._state = self._load_state()
        self._sync_consent_state()

    def _default_state(self) -> dict[str, Any]:
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "consent_active": False,
            "installation_id": "",
            "event_offset": 0,
            "runtime_offset": 0,
            "last_success_at": 0.0,
            "last_attempt_at": 0.0,
            "last_error": "",
            "last_report_id": "",
        }

    def _load_state(self) -> dict[str, Any]:
        state = self._default_state()
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if isinstance(raw, dict):
            for key in state:
                if key in raw:
                    state[key] = raw[key]
        return state

    def _save_state(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._state_path)
        try:
            self._state_path.chmod(0o600)
        except OSError:
            pass

    def _current_size(self, filename: str) -> int:
        try:
            return int((self.log_dir / filename).stat().st_size)
        except OSError:
            return 0

    def _sync_consent_state(self) -> None:
        active = bool(self._state.get("consent_active"))
        installation_id = str(self._state.get("installation_id") or "")
        try:
            identity_valid = bool(uuid.UUID(installation_id))
        except (ValueError, AttributeError):
            identity_valid = False
        if self.enabled and (not active or not identity_valid):
            self._state.update(
                {
                    "consent_active": True,
                    "installation_id": str(uuid.uuid4()),
                    "event_offset": self._current_size("live_usage_events.jsonl"),
                    "runtime_offset": self._current_size("runtime_performance.jsonl"),
                    "last_success_at": 0.0,
                    "last_attempt_at": 0.0,
                    "last_error": "",
                    "last_report_id": "",
                }
            )
            self._save_state()
        elif not self.enabled and active:
            self._state.update(
                {
                    "consent_active": False,
                    "installation_id": "",
                    "event_offset": self._current_size("live_usage_events.jsonl"),
                    "runtime_offset": self._current_size("runtime_performance.jsonl"),
                    "last_success_at": 0.0,
                    "last_attempt_at": 0.0,
                    "last_error": "",
                    "last_report_id": "",
                }
            )
            self._save_state()

    @property
    def configured(self) -> bool:
        return telemetry_endpoint_is_safe(self.endpoint)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "configured": self.configured,
                "interval_hours": round(self.interval_seconds / 3600.0, 2),
                "last_success_at": float(self._state.get("last_success_at") or 0.0),
                "last_attempt_at": float(self._state.get("last_attempt_at") or 0.0),
                "last_error": str(self._state.get("last_error") or ""),
            }

    def start(self) -> bool:
        if not self.enabled or not self.configured:
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="gesturebind-daily-telemetry",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.upload_if_due()
            except Exception as exc:
                self._record_error(exc)
            self._stop.wait(self.poll_seconds)

    def _record_error(self, exc: Exception) -> None:
        with self._lock:
            self._state["last_error"] = str(exc)[:MAX_ERROR_LENGTH]
            try:
                self._save_state()
            except OSError:
                pass

    def _is_due(self, now: float, *, force: bool) -> bool:
        if force:
            return True
        last_success = float(self._state.get("last_success_at") or 0.0)
        if last_success > 0.0 and now - last_success < self.interval_seconds:
            return False
        last_attempt = float(self._state.get("last_attempt_at") or 0.0)
        return last_attempt <= 0.0 or now - last_attempt >= self.retry_seconds

    def _report_id(
        self,
        installation_id: str,
        event_chunk: JsonlChunk,
        runtime_chunk: JsonlChunk,
        now: float,
    ) -> str:
        bucket = int(now // self.interval_seconds)
        source = (
            f"{installation_id}:{event_chunk.start_offset}:{event_chunk.end_offset}:"
            f"{runtime_chunk.start_offset}:{runtime_chunk.end_offset}:{bucket}"
        )
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]

    def _build_report(
        self,
        event_chunk: JsonlChunk,
        runtime_chunk: JsonlChunk,
        now: float,
    ) -> dict[str, Any]:
        since = float(self._state.get("last_success_at") or 0.0)

        def _new_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if since <= 0.0:
                return rows
            result: list[dict[str, Any]] = []
            for row in rows:
                try:
                    recorded_at = float(row.get("recorded_at") or 0.0)
                except (TypeError, ValueError):
                    continue
                if recorded_at > since:
                    result.append(row)
            return result

        events = _new_rows(event_chunk.rows)
        runtime = _new_rows(runtime_chunk.rows)
        installation_id = str(self._state.get("installation_id") or "")
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "report_id": self._report_id(
                installation_id, event_chunk, runtime_chunk, now
            ),
            "installation_id": installation_id,
            "generated_at": round(now, 3),
            "period_start": round(since, 3) if since > 0.0 else 0.0,
            "period_end": round(now, 3),
            "app": {
                "version": self.app_version,
                "platform": platform.system().lower() or "unknown",
                "architecture": _bounded_name(
                    platform.machine(), {"arm64", "aarch64", "x86_64", "amd64"}
                ),
            },
            "usage": _aggregate_usage(events),
            "runtime": _aggregate_runtime(runtime),
            "privacy": {
                "contains_camera_frames": False,
                "contains_landmarks": False,
                "contains_raw_labels": False,
                "contains_command_text": False,
            },
        }

    def upload_if_due(self, *, force: bool = False) -> bool:
        with self._lock:
            if not self.enabled or not self.configured:
                return False
            now = float(self._clock())
            if not self._is_due(now, force=force):
                return False
            self._state["last_attempt_at"] = now
            self._state["last_error"] = ""
            self._save_state()

            event_chunk = _read_jsonl_chunk(
                self.log_dir / "live_usage_events.jsonl",
                int(self._state.get("event_offset") or 0),
            )
            runtime_chunk = _read_jsonl_chunk(
                self.log_dir / "runtime_performance.jsonl",
                int(self._state.get("runtime_offset") or 0),
            )
            payload = self._build_report(event_chunk, runtime_chunk, now)
            headers = {"User-Agent": f"GestureBind/{self.app_version}"}
            if self.project_key:
                headers["X-GestureBind-Project-Key"] = self.project_key

            try:
                self._transport(
                    self.endpoint,
                    payload,
                    headers,
                    self.timeout_seconds,
                )
            except Exception as exc:
                self._state["last_error"] = str(exc)[:MAX_ERROR_LENGTH]
                self._save_state()
                return False

            self._state.update(
                {
                    "event_offset": event_chunk.end_offset,
                    "runtime_offset": runtime_chunk.end_offset,
                    "last_success_at": now,
                    "last_error": "",
                    "last_report_id": str(payload["report_id"]),
                }
            )
            self._save_state()
            return True
