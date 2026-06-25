import json
from pathlib import Path

from scripts.live_evaluation_report import (
    build_markdown_report,
    load_attempts,
    summarize_attempts,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_live_evaluation_report_summarizes_attempt_rows(tmp_path):
    log_path = tmp_path / "live_evaluation.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "event_type": "attempt",
                "expected_label": "swipe_down",
                "expected": "swipe_down",
                "predicted": "swipe_down",
                "confidence": 0.91,
                "result": "correct",
            },
            {
                "event_type": "attempt",
                "expected_label": "swipe_down",
                "expected": "swipe_down",
                "predicted": "swipe_left",
                "confidence": 0.81,
                "result": "wrong",
            },
            {
                "event_type": "attempt",
                "expected_label": "swipe_left",
                "expected": "swipe_left",
                "predicted": "",
                "confidence": 0.0,
                "result": "missed",
            },
        ],
    )

    report = summarize_attempts(load_attempts(log_path), source=str(log_path))

    assert report.total_attempts == 3
    assert report.correct == 1
    assert report.wrong == 1
    assert report.missed == 1
    by_label = {row.expected: row for row in report.labels}
    assert by_label["swipe_down"].attempts == 2
    assert by_label["swipe_down"].wrong_labels == {"swipe_left": 1}
    assert by_label["swipe_left"].missed == 1

    markdown = build_markdown_report(report)
    assert "# Live Evaluation Metrics" in markdown
    assert "`swipe_down`" in markdown


def test_live_evaluation_report_falls_back_to_run_attempts(tmp_path):
    log_path = tmp_path / "live_evaluation.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "event_type": "run_completed",
                "expected_label": "swipe_up",
                "attempts": [
                    {
                        "expected": "swipe_up",
                        "predicted": "swipe_up",
                        "confidence": 0.99,
                        "result": "correct",
                    }
                ],
            }
        ],
    )

    report = summarize_attempts(load_attempts(log_path), source=str(log_path))

    assert report.total_attempts == 1
    assert report.correct == 1
    assert report.labels[0].expected == "swipe_up"
