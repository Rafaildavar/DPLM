import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "jmlc_inspector_report.py"
    spec = importlib.util.spec_from_file_location("jmlc_inspector_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rows():
    return [
        {
            "sequence": 1,
            "phase": "pending",
            "label": "swipe_down",
            "route": "dynamic",
            "model": "sequence_mlp",
            "confidence": 0.7,
            "reason": "dynamic_pending",
        },
        {
            "sequence": 2,
            "phase": "confirmed",
            "label": "swipe_down",
            "route": "dynamic",
            "model": "sequence_mlp",
            "confidence": 0.9,
            "reason": "dynamic_accepted",
        },
        {
            "sequence": 3,
            "phase": "cooldown",
            "label": "swipe_down",
            "route": "dynamic",
            "model": "sequence_mlp",
            "confidence": 0.88,
            "reason": "cooldown",
        },
    ]


def test_jmlc_inspector_report_reads_jsonl_and_builds_markdown(tmp_path):
    module = _load_module()
    source = tmp_path / "live_gesture_inspector.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in reversed(_rows())),
        encoding="utf-8",
    )
    output = tmp_path / "report.md"

    result = module.write_report(source, output)
    text = result.read_text(encoding="utf-8")

    assert result == output
    assert "Total decisions: 3" in text
    assert "Safety decisions: 1" in text
    assert "| `cooldown` | 1 |" in text
    assert "| 3 | `cooldown` | `swipe_down`" in text


def test_jmlc_inspector_report_reads_csv(tmp_path):
    module = _load_module()
    source = tmp_path / "live_gesture_inspector.csv"
    with source.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sequence",
                "phase",
                "label",
                "route",
                "model",
                "confidence",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(reversed(_rows()))

    rows = module.read_inspector_export(source)
    summary = module.summarize_rows(rows)

    assert [int(row["sequence"]) for row in rows] == [1, 2, 3]
    assert summary["confirmed"] == 1
    assert summary["safety_total"] == 1
