"""Small self-hosted collector for GestureBind daily telemetry reports."""
from __future__ import annotations

import hmac
import json
import re
import sqlite3
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


MAX_REQUEST_BYTES = 256 * 1024
REPORT_ID_RE = re.compile(r"[a-f0-9]{32}")


class TelemetryValidationError(ValueError):
    pass


def _non_negative_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TelemetryValidationError(f"{field} must be an integer") from exc
    if result < 0:
        raise TelemetryValidationError(f"{field} must be non-negative")
    return result


def validate_daily_report(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TelemetryValidationError("payload must be a JSON object")
    try:
        schema_version = int(payload.get("schema_version") or 0)
    except (TypeError, ValueError) as exc:
        raise TelemetryValidationError("invalid schema_version") from exc
    if schema_version != 1:
        raise TelemetryValidationError("unsupported schema_version")

    report_id = str(payload.get("report_id") or "").strip().lower()
    if REPORT_ID_RE.fullmatch(report_id) is None:
        raise TelemetryValidationError("invalid report_id")
    installation_id = str(payload.get("installation_id") or "").strip()
    try:
        uuid.UUID(installation_id)
    except (ValueError, AttributeError) as exc:
        raise TelemetryValidationError("invalid installation_id") from exc

    usage = payload.get("usage")
    runtime = payload.get("runtime")
    privacy = payload.get("privacy")
    app = payload.get("app")
    if not isinstance(usage, dict) or not isinstance(runtime, dict):
        raise TelemetryValidationError("usage and runtime must be objects")
    if not isinstance(privacy, dict) or not isinstance(app, dict):
        raise TelemetryValidationError("privacy and app must be objects")
    forbidden_flags = (
        "contains_camera_frames",
        "contains_landmarks",
        "contains_raw_labels",
        "contains_command_text",
    )
    if any(bool(privacy.get(key)) for key in forbidden_flags):
        raise TelemetryValidationError("raw or identifying telemetry is not accepted")

    _non_negative_int(usage.get("event_count"), "usage.event_count")
    _non_negative_int(usage.get("command_attempts"), "usage.command_attempts")
    _non_negative_int(usage.get("executed_commands"), "usage.executed_commands")
    _non_negative_int(runtime.get("sample_count"), "runtime.sample_count")
    _non_negative_int(runtime.get("window_count"), "runtime.window_count")

    normalized = dict(payload)
    normalized["report_id"] = report_id
    normalized["installation_id"] = installation_id
    return normalized


class TelemetryStore:
    def __init__(self, path: Path | str, *, retention_days: int = 90) -> None:
        self.path = Path(path).expanduser()
        self.retention_days = min(3650, max(1, int(retention_days)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_reports (
                    report_id TEXT PRIMARY KEY,
                    installation_id TEXT NOT NULL,
                    received_at REAL NOT NULL,
                    generated_at REAL NOT NULL,
                    app_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_daily_reports_received_at
                ON daily_reports(received_at)
                """
            )

    def insert(self, payload: dict[str, Any], *, received_at: float | None = None) -> bool:
        report = validate_daily_report(payload)
        app = report.get("app") or {}
        timestamp = float(received_at if received_at is not None else time.time())
        serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM daily_reports WHERE received_at < ?",
                (timestamp - self.retention_days * 24 * 60 * 60,),
            )
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO daily_reports (
                    report_id,
                    installation_id,
                    received_at,
                    generated_at,
                    app_version,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report["report_id"],
                    report["installation_id"],
                    timestamp,
                    float(report.get("generated_at") or 0.0),
                    str(app.get("version") or "unknown")[:40],
                    serialized,
                ),
            )
            return int(cursor.rowcount or 0) > 0

    def summary(self, *, days: int = 30, now: float | None = None) -> dict[str, Any]:
        safe_days = min(365, max(1, int(days)))
        current = float(now if now is not None else time.time())
        since = current - safe_days * 24 * 60 * 60
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT installation_id, received_at, app_version, payload_json
                FROM daily_reports
                WHERE received_at >= ?
                ORDER BY received_at ASC
                """,
                (since,),
            ).fetchall()

        installs: set[str] = set()
        versions: dict[str, int] = {}
        event_count = 0
        command_attempts = 0
        executed_commands = 0
        feedback: dict[str, int] = {}
        event_types: dict[str, int] = {}
        routes: dict[str, int] = {}
        runtime_samples = 0
        weighted_inference = 0.0
        latest_received_at = 0.0
        for installation_id, received_at, app_version, payload_json in rows:
            installs.add(str(installation_id))
            versions[str(app_version)] = versions.get(str(app_version), 0) + 1
            latest_received_at = max(latest_received_at, float(received_at or 0.0))
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                continue
            usage = payload.get("usage") or {}
            runtime = payload.get("runtime") or {}
            event_count += max(0, int(usage.get("event_count") or 0))
            command_attempts += max(0, int(usage.get("command_attempts") or 0))
            executed_commands += max(0, int(usage.get("executed_commands") or 0))
            for key, value in (usage.get("feedback_counts") or {}).items():
                name = str(key)
                feedback[name] = feedback.get(name, 0) + max(0, int(value or 0))
            for key, value in (usage.get("event_type_counts") or {}).items():
                name = str(key)
                event_types[name] = event_types.get(name, 0) + max(0, int(value or 0))
            for key, value in (usage.get("route_counts") or {}).items():
                name = str(key)
                routes[name] = routes.get(name, 0) + max(0, int(value or 0))
            samples = max(0, int(runtime.get("sample_count") or 0))
            runtime_samples += samples
            weighted_inference += (
                float(runtime.get("inference_ms_average") or 0.0) * samples
            )

        correct = int(feedback.get("correct") or 0)
        incorrect = int(feedback.get("incorrect") or 0)
        evaluated = correct + incorrect
        return {
            "generated_at": current,
            "window_days": safe_days,
            "reports": len(rows),
            "active_installations": len(installs),
            "event_count": event_count,
            "command_attempts": command_attempts,
            "executed_commands": executed_commands,
            "command_success_rate": round(
                executed_commands / command_attempts, 4
            )
            if command_attempts
            else 0.0,
            "feedback_counts": dict(sorted(feedback.items())),
            "user_confirmed_precision": round(correct / evaluated, 4)
            if evaluated
            else 0.0,
            "incorrect_feedback_rate": round(incorrect / evaluated, 4)
            if evaluated
            else 0.0,
            "missed_feedback": int(feedback.get("missed") or 0),
            "event_type_counts": dict(sorted(event_types.items())),
            "route_counts": dict(sorted(routes.items())),
            "runtime_sample_count": runtime_samples,
            "inference_ms_average": round(
                weighted_inference / runtime_samples, 3
            )
            if runtime_samples
            else 0.0,
            "app_version_reports": dict(sorted(versions.items())),
            "latest_received_at": latest_received_at,
        }


def _authorized(value: str, expected: str) -> bool:
    if not expected:
        return True
    return hmac.compare_digest(str(value or ""), expected)


def make_collector_handler(
    store: TelemetryStore,
    *,
    project_key: str = "",
    admin_token: str = "",
) -> type[BaseHTTPRequestHandler]:
    class CollectorHandler(BaseHTTPRequestHandler):
        server_version = "GestureBindTelemetry/1"

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._json(HTTPStatus.OK, {"status": "ok"})
                return
            if parsed.path != "/v1/telemetry/summary":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            authorization = str(self.headers.get("Authorization") or "")
            token = authorization.removeprefix("Bearer ").strip()
            if not _authorized(token, admin_token):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            query = parse_qs(parsed.query)
            try:
                days = int((query.get("days") or ["30"])[0])
            except ValueError:
                days = 30
            self._json(HTTPStatus.OK, store.summary(days=days))

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/v1/telemetry/daily":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            if not _authorized(
                str(self.headers.get("X-GestureBind-Project-Key") or ""),
                project_key,
            ):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid_size"})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                inserted = store.insert(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, TelemetryValidationError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._json(
                HTTPStatus.ACCEPTED if inserted else HTTPStatus.OK,
                {"accepted": True, "duplicate": not inserted},
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

    return CollectorHandler


def serve_collector(
    *,
    host: str,
    port: int,
    database_path: Path | str,
    project_key: str = "",
    admin_token: str = "",
    retention_days: int = 90,
) -> None:
    store = TelemetryStore(database_path, retention_days=retention_days)
    handler = make_collector_handler(
        store,
        project_key=project_key,
        admin_token=admin_token,
    )
    server = ThreadingHTTPServer((host, int(port)), handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
