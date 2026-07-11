from __future__ import annotations

import uuid

import pytest

from app.services.telemetry_collector import (
    TelemetryStore,
    TelemetryValidationError,
    validate_daily_report,
)


def _report() -> dict:
    return {
        "schema_version": 1,
        "report_id": "a" * 32,
        "installation_id": str(uuid.uuid4()),
        "generated_at": 1_700_000_000.0,
        "period_start": 0.0,
        "period_end": 1_700_000_000.0,
        "app": {"version": "0.8.1", "platform": "darwin", "architecture": "arm64"},
        "usage": {
            "event_count": 5,
            "command_attempts": 4,
            "executed_commands": 3,
            "feedback_counts": {"correct": 2, "incorrect": 1},
        },
        "runtime": {
            "window_count": 2,
            "sample_count": 10,
            "inference_ms_average": 15.0,
        },
        "privacy": {
            "contains_camera_frames": False,
            "contains_landmarks": False,
            "contains_raw_labels": False,
            "contains_command_text": False,
        },
    }


def test_collector_is_idempotent_and_builds_summary(tmp_path):
    store = TelemetryStore(tmp_path / "telemetry.sqlite")
    report = _report()

    assert store.insert(report, received_at=1_700_000_100.0) is True
    assert store.insert(report, received_at=1_700_000_101.0) is False

    summary = store.summary(days=30, now=1_700_000_200.0)
    assert summary["reports"] == 1
    assert summary["active_installations"] == 1
    assert summary["event_count"] == 5
    assert summary["command_success_rate"] == 0.75
    assert summary["feedback_counts"] == {"correct": 2, "incorrect": 1}
    assert summary["user_confirmed_precision"] == 0.6667
    assert summary["incorrect_feedback_rate"] == 0.3333
    assert summary["missed_feedback"] == 0
    assert summary["inference_ms_average"] == 15.0


def test_collector_rejects_raw_private_data_flags():
    report = _report()
    report["privacy"]["contains_raw_labels"] = True

    with pytest.raises(TelemetryValidationError):
        validate_daily_report(report)

    report = _report()
    report["schema_version"] = "broken"
    with pytest.raises(TelemetryValidationError):
        validate_daily_report(report)
