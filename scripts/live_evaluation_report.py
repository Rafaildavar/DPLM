"""Build attempt-level metrics from Home live-evaluation logs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True)
class AttemptRecord:
    expected: str
    predicted: str
    result: str
    confidence: float
    elapsed_seconds: float | None = None


@dataclass(frozen=True)
class LabelMetrics:
    expected: str
    attempts: int
    correct: int
    wrong: int
    missed: int
    accuracy: float
    avg_confidence: float | None
    wrong_labels: dict[str, int]


@dataclass(frozen=True)
class RunMetrics:
    event_type: str
    expected: str
    recorded_at: float
    attempts: int
    correct: int
    wrong: int
    missed: int
    accuracy: float
    accepted_accuracy: float | None
    avg_confidence: float | None
    wrong_labels: dict[str, int]
    min_confidence: float | None
    timeout_seconds: float | None
    recognition_model_mode: str
    dynamic_model_profile: str


@dataclass(frozen=True)
class LiveEvaluationReport:
    source: str
    total_attempts: int
    correct: int
    wrong: int
    missed: int
    accuracy: float
    labels: list[LabelMetrics]
    runs: list[RunMetrics]


def _clean_label(value: Any) -> str:
    return str(value or "").strip()


def _parse_attempt_row(row: dict[str, Any]) -> AttemptRecord | None:
    result = _clean_label(row.get("result")).lower()
    if result not in {"correct", "wrong", "missed"}:
        return None

    expected = _clean_label(row.get("expected") or row.get("expected_label"))
    if not expected:
        return None

    try:
        confidence = float(row.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    elapsed_raw = row.get("elapsed_seconds")
    try:
        elapsed = None if elapsed_raw is None else float(elapsed_raw)
    except (TypeError, ValueError):
        elapsed = None

    return AttemptRecord(
        expected=expected,
        predicted=_clean_label(row.get("predicted")),
        result=result,
        confidence=confidence,
        elapsed_seconds=elapsed,
    )


def _parse_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _summarize_run(row: dict[str, Any]) -> RunMetrics | None:
    expected = _clean_label(row.get("expected") or row.get("expected_label"))
    if not expected:
        return None

    attempts_raw = row.get("attempts") or []
    attempts = [
        attempt
        for attempt in (
            _parse_attempt_row(item)
            for item in attempts_raw
            if isinstance(item, dict)
        )
        if attempt is not None
    ]
    if attempts:
        correct = sum(1 for item in attempts if item.result == "correct")
        wrong = sum(1 for item in attempts if item.result == "wrong")
        missed = sum(1 for item in attempts if item.result == "missed")
        attempts_count = len(attempts)
        confidences = [
            item.confidence
            for item in attempts
            if item.result in {"correct", "wrong"} and item.confidence > 0.0
        ]
        wrong_labels = Counter(
            item.predicted or "unknown" for item in attempts if item.result == "wrong"
        )
    else:
        attempts_count = int(
            row["total"]
            if row.get("total") is not None
            else row.get("target_attempts") or 0
        )
        correct = int(row.get("correct") or 0)
        wrong = int(row.get("wrong") or 0)
        missed = int(row.get("missed") or 0)
        confidences = []
        wrong_labels = Counter()

    accepted = correct + wrong
    return RunMetrics(
        event_type=_clean_label(row.get("event_type")),
        expected=expected,
        recorded_at=float(row.get("recorded_at") or 0.0),
        attempts=attempts_count,
        correct=correct,
        wrong=wrong,
        missed=missed,
        accuracy=correct / attempts_count if attempts_count else 0.0,
        accepted_accuracy=(correct / accepted if accepted else None),
        avg_confidence=mean(confidences) if confidences else None,
        wrong_labels=dict(sorted(wrong_labels.items())),
        min_confidence=_parse_float(row.get("min_confidence")),
        timeout_seconds=_parse_float(row.get("timeout_seconds")),
        recognition_model_mode=_clean_label(row.get("recognition_model_mode")),
        dynamic_model_profile=_clean_label(row.get("dynamic_model_profile")),
    )


def load_attempts(path: Path) -> list[AttemptRecord]:
    attempts: list[AttemptRecord] = []
    fallback_runs: list[dict[str, Any]] = []
    if not path.exists():
        return attempts

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        if row.get("event_type") == "attempt":
            attempt = _parse_attempt_row(row)
            if attempt is not None:
                attempts.append(attempt)
        elif isinstance(row.get("attempts"), list):
            fallback_runs.append(row)

    if attempts:
        return attempts

    for run in fallback_runs:
        for item in run.get("attempts") or []:
            if isinstance(item, dict):
                attempt = _parse_attempt_row(item)
                if attempt is not None:
                    attempts.append(attempt)
    return attempts


def load_runs(path: Path) -> list[RunMetrics]:
    runs: list[RunMetrics] = []
    if not path.exists():
        return runs

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event_type") not in {"run_completed", "run_stopped"}:
            continue
        run = _summarize_run(row)
        if run is not None:
            runs.append(run)
    return runs


def load_completed_runs(path: Path) -> list[RunMetrics]:
    return [run for run in load_runs(path) if run.event_type == "run_completed"]


def summarize_attempts(
    attempts: Iterable[AttemptRecord],
    *,
    source: str = "",
    runs: Iterable[RunMetrics] | None = None,
) -> LiveEvaluationReport:
    records = list(attempts)
    grouped: dict[str, list[AttemptRecord]] = defaultdict(list)
    for item in records:
        grouped[item.expected].append(item)

    labels: list[LabelMetrics] = []
    for expected in sorted(grouped):
        rows = grouped[expected]
        correct = sum(1 for row in rows if row.result == "correct")
        wrong = sum(1 for row in rows if row.result == "wrong")
        missed = sum(1 for row in rows if row.result == "missed")
        confidences = [
            row.confidence
            for row in rows
            if row.result in {"correct", "wrong"} and row.confidence > 0.0
        ]
        wrong_labels = Counter(
            row.predicted or "unknown"
            for row in rows
            if row.result == "wrong"
        )
        attempts_count = len(rows)
        labels.append(
            LabelMetrics(
                expected=expected,
                attempts=attempts_count,
                correct=correct,
                wrong=wrong,
                missed=missed,
                accuracy=correct / attempts_count if attempts_count else 0.0,
                avg_confidence=mean(confidences) if confidences else None,
                wrong_labels=dict(sorted(wrong_labels.items())),
            )
        )

    total = len(records)
    correct_total = sum(label.correct for label in labels)
    wrong_total = sum(label.wrong for label in labels)
    missed_total = sum(label.missed for label in labels)
    return LiveEvaluationReport(
        source=source,
        total_attempts=total,
        correct=correct_total,
        wrong=wrong_total,
        missed=missed_total,
        accuracy=correct_total / total if total else 0.0,
        labels=labels,
        runs=sorted(list(runs or []), key=lambda item: item.recorded_at),
    )


def build_markdown_report(report: LiveEvaluationReport) -> str:
    lines = [
        "# Live Evaluation Metrics",
        "",
        f"Source: `{report.source}`" if report.source else "Source: n/a",
        "",
        "## Summary",
        "",
        "| Attempts | Correct | Wrong | Missed | Accuracy |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {report.total_attempts} | {report.correct} | {report.wrong} | "
            f"{report.missed} | {report.accuracy:.3f} |"
        ),
        "",
        "## By Label",
        "",
        "| Expected | Attempts | Correct | Wrong | Missed | Accuracy | Avg confidence | Wrong labels |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label in report.labels:
        avg_conf = "n/a" if label.avg_confidence is None else f"{label.avg_confidence:.3f}"
        wrong_labels = (
            ", ".join(f"{name}:{count}" for name, count in label.wrong_labels.items())
            if label.wrong_labels
            else ""
        )
        lines.append(
            f"| `{label.expected}` | {label.attempts} | {label.correct} | "
            f"{label.wrong} | {label.missed} | {label.accuracy:.3f} | "
            f"{avg_conf} | {wrong_labels} |"
        )
    if report.runs:
        latest_by_label: dict[str, RunMetrics] = {}
        for run in report.runs:
            if run.event_type != "run_completed":
                continue
            latest_by_label[run.expected] = run

        if latest_by_label:
            lines.extend(
                [
                    "",
                    "## Latest Completed Run By Label",
                    "",
                    (
                        "| Expected | Model | Attempts | Correct | Wrong | Missed | "
                        "Accuracy | Accepted accuracy | Avg confidence | Wrong labels |"
                    ),
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for run in sorted(latest_by_label.values(), key=lambda item: item.expected):
                accepted_accuracy = (
                    "n/a"
                    if run.accepted_accuracy is None
                    else f"{run.accepted_accuracy:.3f}"
                )
                avg_conf = "n/a" if run.avg_confidence is None else f"{run.avg_confidence:.3f}"
                wrong_labels = (
                    ", ".join(f"{name}:{count}" for name, count in run.wrong_labels.items())
                    if run.wrong_labels
                    else ""
                )
                model = _format_run_model(run)
                lines.append(
                    f"| `{run.expected}` | {model} | {run.attempts} | {run.correct} | "
                    f"{run.wrong} | {run.missed} | {run.accuracy:.3f} | "
                    f"{accepted_accuracy} | {avg_conf} | {wrong_labels} |"
                )

        lines.extend(
            [
                "",
                "## Recent Runs",
                "",
                (
                    "| Event | Expected | Model | Attempts | Correct | Wrong | Missed | "
                    "Accuracy | Min conf | Timeout |"
                ),
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run in report.runs[-8:]:
            min_conf = "n/a" if run.min_confidence is None else f"{run.min_confidence:.2f}"
            timeout = "n/a" if run.timeout_seconds is None else f"{run.timeout_seconds:.1f}"
            event = run.event_type.removeprefix("run_") or "n/a"
            lines.append(
                f"| `{event}` | `{run.expected}` | {_format_run_model(run)} | "
                f"{run.attempts} | {run.correct} | {run.wrong} | "
                f"{run.missed} | {run.accuracy:.3f} | {min_conf} | {timeout} |"
            )
    lines.append("")
    return "\n".join(lines)


def _format_run_model(run: RunMetrics) -> str:
    mode = run.recognition_model_mode or "n/a"
    profile = run.dynamic_model_profile
    if mode == "dynamic" and profile:
        return f"`dynamic:{profile}`"
    return f"`{mode}`"


def parse_args() -> argparse.Namespace:
    default_input = Path.home() / ".dplm" / "logs" / "live_evaluation.jsonl"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(default_input), help="Path to live_evaluation.jsonl")
    parser.add_argument("--out-json", default="", help="Optional JSON report path")
    parser.add_argument("--out-md", default="", help="Optional Markdown report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    report = summarize_attempts(
        load_attempts(input_path),
        source=str(input_path),
        runs=load_runs(input_path),
    )
    markdown = build_markdown_report(report)
    print(markdown)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
