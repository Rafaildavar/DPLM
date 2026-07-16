from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.usage_telemetry import (
    DailyUsageTelemetry,
    telemetry_endpoint_is_safe,
)


class _Clock:
    def __init__(self, value: float = 1_700_000_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def test_daily_upload_sends_only_sanitized_aggregates(tmp_path):
    events_path = tmp_path / "live_usage_events.jsonl"
    runtime_path = tmp_path / "runtime_performance.jsonl"
    _append(
        events_path,
        {
            "recorded_at": 1_699_999_000.0,
            "event_type": "command_executed",
            "label": "private gesture name",
            "command_info": "Open private application",
            "confidence": 0.99,
            "route": "dynamic",
            "recognition_model_mode": "auto",
        },
    )
    captured: list[dict[str, Any]] = []
    clock = _Clock()
    client = DailyUsageTelemetry(
        log_dir=tmp_path,
        endpoint="https://telemetry.example.test/v1/telemetry/daily",
        enabled=True,
        project_key="public-key",
        app_version="0.8.1",
        clock=clock,
        transport=lambda _url, payload, headers, _timeout: captured.append(
            {"payload": payload, "headers": headers}
        ),
    )

    _append(
        events_path,
        {
            "recorded_at": clock.value + 1,
            "event_type": "command_executed",
            "label": "another private name",
            "command_info": "Sensitive command",
            "confidence": 0.96,
            "route": "dynamic",
            "recognition_model_mode": "auto",
        },
    )
    _append(
        runtime_path,
        {
            "recorded_at": clock.value + 1,
            "samples": 10,
            "inference_ms_avg": 14.0,
            "inference_ms_p95": 18.0,
            "hand_detected_rate": 0.9,
            "camera_frame_width": 1920,
        },
    )

    assert client.upload_if_due(force=True) is True
    assert len(captured) == 1
    payload = captured[0]["payload"]
    serialized = json.dumps(payload)
    assert payload["usage"]["event_count"] == 1
    assert payload["usage"]["executed_commands"] == 1
    assert payload["runtime"]["sample_count"] == 10
    assert payload["runtime"]["inference_ms_average"] == 14.0
    assert "private" not in serialized
    assert "Sensitive command" not in serialized
    assert "camera_frame_width" not in serialized
    assert payload["privacy"]["contains_camera_frames"] is False
    assert captured[0]["headers"]["X-GestureBind-Project-Key"] == "public-key"


def test_failed_upload_keeps_cursor_for_retry(tmp_path):
    clock = _Clock()

    def fail(*_args):
        raise OSError("offline")

    client = DailyUsageTelemetry(
        log_dir=tmp_path,
        endpoint="https://telemetry.example.test/v1/telemetry/daily",
        enabled=True,
        clock=clock,
        transport=fail,
    )
    _append(
        tmp_path / "live_usage_events.jsonl",
        {
            "recorded_at": clock.value + 1,
            "event_type": "gesture_confirmed",
            "confidence": 0.91,
            "route": "static",
            "recognition_model_mode": "auto",
        },
    )

    assert client.upload_if_due(force=True) is False
    failed_state = json.loads((tmp_path / "telemetry_state.json").read_text())
    assert failed_state["event_offset"] == 0
    assert failed_state["last_error"] == "offline"

    captured: list[dict[str, Any]] = []
    retry = DailyUsageTelemetry(
        log_dir=tmp_path,
        endpoint="https://telemetry.example.test/v1/telemetry/daily",
        enabled=True,
        clock=clock,
        transport=lambda _url, payload, _headers, _timeout: captured.append(payload),
    )
    assert retry.upload_if_due(force=True) is True
    assert captured[0]["usage"]["event_count"] == 1


def test_disabling_consent_rotates_identity_and_skips_old_events(tmp_path):
    clock = _Clock()
    enabled = DailyUsageTelemetry(
        log_dir=tmp_path,
        endpoint="https://telemetry.example.test/v1/telemetry/daily",
        enabled=True,
        clock=clock,
        transport=lambda *_args: None,
    )
    first_id = json.loads((tmp_path / "telemetry_state.json").read_text())[
        "installation_id"
    ]
    _append(
        tmp_path / "live_usage_events.jsonl",
        {"recorded_at": clock.value, "event_type": "gesture_confirmed"},
    )

    DailyUsageTelemetry(
        log_dir=tmp_path,
        endpoint="https://telemetry.example.test/v1/telemetry/daily",
        enabled=False,
        clock=clock,
    )
    disabled_state = json.loads((tmp_path / "telemetry_state.json").read_text())
    assert disabled_state["installation_id"] == ""

    reports: list[dict[str, Any]] = []
    reenabled = DailyUsageTelemetry(
        log_dir=tmp_path,
        endpoint="https://telemetry.example.test/v1/telemetry/daily",
        enabled=True,
        clock=clock,
        transport=lambda _url, payload, _headers, _timeout: reports.append(payload),
    )
    second_id = json.loads((tmp_path / "telemetry_state.json").read_text())[
        "installation_id"
    ]
    assert second_id and second_id != first_id
    assert reenabled.upload_if_due(force=True) is True
    assert reports[0]["usage"]["event_count"] == 0


def test_remote_endpoint_requires_https():
    assert telemetry_endpoint_is_safe("https://example.test/v1/telemetry/daily")
    assert telemetry_endpoint_is_safe("http://127.0.0.1:8787/v1/telemetry/daily")
    assert not telemetry_endpoint_is_safe("http://example.test/v1/telemetry/daily")


def test_automatic_upload_runs_at_most_once_per_day(tmp_path):
    clock = _Clock()
    reports: list[dict[str, Any]] = []
    client = DailyUsageTelemetry(
        log_dir=tmp_path,
        endpoint="https://telemetry.example.test/v1/telemetry/daily",
        enabled=True,
        interval_seconds=24 * 60 * 60,
        clock=clock,
        transport=lambda _url, payload, _headers, _timeout: reports.append(payload),
    )

    assert client.upload_if_due() is True
    clock.value += 23 * 60 * 60
    assert client.upload_if_due() is False
    clock.value += 60 * 60
    assert client.upload_if_due() is True
    assert len(reports) == 2
