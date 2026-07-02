"""Build a compact JMLC report from Recognition Inspector exports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("logs/live_gesture_inspector.jsonl")
DEFAULT_OUTPUT = Path("docs/contest/live_inspector_report.md")


def _coerce_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _coerce_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean:
            continue
        item = json.loads(clean)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def read_inspector_export(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Inspector export not found: {source}")
    if source.suffix.lower() == ".csv":
        rows = _read_csv(source)
    else:
        rows = _read_jsonl(source)
    return sorted(rows, key=lambda row: _coerce_int(row.get("sequence")))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    phases = Counter(str(row.get("phase") or "unknown") for row in rows)
    routes = Counter(str(row.get("route") or "unknown") for row in rows)
    models = Counter(str(row.get("model") or "unknown") for row in rows)
    reasons = Counter(
        str(row.get("reason") or "none")
        for row in rows
        if str(row.get("reason") or "").strip()
    )
    confidences = [_coerce_float(row.get("confidence")) for row in rows]
    safety_total = sum(
        phases.get(name, 0)
        for name in ("rejected", "cooldown", "suppressed")
    )
    return {
        "total": len(rows),
        "phases": phases,
        "routes": routes,
        "models": models,
        "reasons": reasons,
        "safety_total": safety_total,
        "confirmed": phases.get("confirmed", 0),
        "avg_confidence": (
            sum(confidences) / len(confidences)
            if confidences
            else 0.0
        ),
    }


def _counter_table(title: str, counter: Counter[str]) -> list[str]:
    lines = [f"## {title}", "", "| Value | Count |", "| --- | ---: |"]
    if counter:
        for value, count in counter.most_common():
            lines.append(f"| `{value}` | {count} |")
    else:
        lines.append("| `none` | 0 |")
    lines.append("")
    return lines


def build_markdown_report(rows: list[dict[str, Any]]) -> str:
    summary = summarize_rows(rows)
    total = int(summary["total"])
    confirmed = int(summary["confirmed"])
    safety_total = int(summary["safety_total"])
    avg_conf = float(summary["avg_confidence"])

    lines = [
        "# JMLC Live Inspector Report",
        "",
        "Generated from `Recognition Inspector` export.",
        "",
        "## Summary",
        "",
        f"- Total decisions: {total}",
        f"- Confirmed decisions: {confirmed}",
        f"- Safety decisions: {safety_total}",
        f"- Average confidence: {avg_conf * 100:.1f}%",
        "",
    ]
    lines.extend(_counter_table("Phases", summary["phases"]))
    lines.extend(_counter_table("Routes", summary["routes"]))
    lines.extend(_counter_table("Models", summary["models"]))
    lines.extend(_counter_table("Reasons", summary["reasons"]))
    lines.extend(
        [
            "## Recent Decisions",
            "",
            "| # | Phase | Label | Route | Model | Confidence | Reason |",
            "| ---: | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in rows[-10:]:
        sequence = _coerce_int(row.get("sequence"))
        phase = str(row.get("phase") or "")
        label = str(row.get("label") or "")
        route = str(row.get("route") or "")
        model = str(row.get("model") or "")
        reason = str(row.get("reason") or "")
        confidence = _coerce_float(row.get("confidence"))
        lines.append(
            f"| {sequence} | `{phase}` | `{label}` | `{route}` | "
            f"`{model}` | {confidence * 100:.1f}% | `{reason or 'none'}` |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> Path:
    rows = read_inspector_export(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_markdown_report(rows), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build JMLC markdown report from live inspector export.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to live_gesture_inspector.jsonl or .csv",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to markdown report to write",
    )
    args = parser.parse_args(argv)
    path = write_report(args.input, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
