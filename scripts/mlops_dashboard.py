"""Generate a local HTML dashboard for GestureFlow ML observability."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services.gesture_taxonomy import load_gesture_taxonomy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = Path.home() / ".dplm" / "logs"
DEFAULT_OUT_DIR = ROOT / "docs" / "mlops_dashboard"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def dataset_summary(data_root: Path, taxonomy_path: Path) -> dict[str, Any]:
    taxonomy = load_gesture_taxonomy(taxonomy_path)
    classes: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    total_samples = 0
    if data_root.exists():
        for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
            label = label_dir.name
            samples = len(list(label_dir.glob("sample_*.npy")))
            gesture_type = taxonomy.gesture_type_for_label(label)
            classes.append(
                {
                    "label": label,
                    "type": gesture_type,
                    "samples": samples,
                }
            )
            type_counts[gesture_type] += samples
            total_samples += samples
    return {
        "data_root": str(data_root),
        "total_classes": len(classes),
        "total_samples": total_samples,
        "type_counts": dict(sorted(type_counts.items())),
        "classes": classes,
    }


def model_artifacts(models_dir: Path) -> list[dict[str, Any]]:
    if not models_dir.exists():
        return []
    artifacts: list[dict[str, Any]] = []
    for path in sorted(models_dir.iterdir()):
        if path.suffix not in {".pkl", ".json", ".txt"}:
            continue
        if not path.is_file():
            continue
        payload = path.read_bytes()
        artifacts.append(
            {
                "name": path.name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()[:16],
                "modified_at": path.stat().st_mtime,
            }
        )
    return artifacts


def live_evaluation_summary(log_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(log_dir / "live_evaluation.jsonl")
    attempts = [row for row in rows if row.get("event_type") == "attempt"]
    total = len(attempts)
    correct = sum(1 for row in attempts if row.get("result") == "correct")
    wrong = sum(1 for row in attempts if row.get("result") == "wrong")
    missed = sum(1 for row in attempts if row.get("result") == "missed")
    route_counts = Counter(str(row.get("route") or "unknown") for row in attempts)
    source_counts = Counter(
        str(row.get("dynamic_decision_source") or "unknown") for row in attempts
    )
    static_method_counts = Counter(
        str(row.get("static_rejection_method") or "unknown") for row in attempts
    )
    static_reject_reason_counts = Counter(
        str(row.get("static_reject_reason") or "none") for row in attempts
    )
    static_decision_counts = Counter(
        str(row.get("static_decision_source") or "unknown") for row in attempts
    )
    negative_rejections = sum(
        1 for row in attempts if row.get("dynamic_decision_source") == "negative_rejected"
    )
    by_expected: dict[str, Counter[str]] = defaultdict(Counter)
    for row in attempts:
        expected = str(row.get("expected") or row.get("expected_label") or "unknown")
        result = str(row.get("result") or "unknown")
        by_expected[expected][result] += 1
    labels = []
    for expected, counter in sorted(by_expected.items()):
        label_total = sum(counter.values())
        labels.append(
            {
                "expected": expected,
                "attempts": label_total,
                "correct": counter.get("correct", 0),
                "wrong": counter.get("wrong", 0),
                "missed": counter.get("missed", 0),
                "accuracy": (
                    counter.get("correct", 0) / label_total if label_total else 0.0
                ),
            }
        )
    return {
        "source": str(log_dir / "live_evaluation.jsonl"),
        "attempts": total,
        "correct": correct,
        "wrong": wrong,
        "missed": missed,
        "accuracy": correct / total if total else 0.0,
        "route_counts": dict(sorted(route_counts.items())),
        "dynamic_decision_sources": dict(sorted(source_counts.items())),
        "static_rejection_methods": dict(sorted(static_method_counts.items())),
        "static_rejection_reasons": dict(sorted(static_reject_reason_counts.items())),
        "static_decision_sources": dict(sorted(static_decision_counts.items())),
        "negative_rejections": negative_rejections,
        "labels": labels,
    }


def runtime_summary(log_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(log_dir / "runtime_performance.jsonl")
    latest = rows[-1] if rows else {}
    return {
        "source": str(log_dir / "runtime_performance.jsonl"),
        "samples": len(rows),
        "latest": latest,
    }


def build_dashboard(
    *,
    data_root: Path = ROOT / "data" / "gestures",
    models_dir: Path = ROOT / "models",
    taxonomy_path: Path = ROOT / "configs" / "gesture_taxonomy.json",
    log_dir: Path = DEFAULT_LOG_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset": dataset_summary(data_root, taxonomy_path),
        "models": model_artifacts(models_dir),
        "live_evaluation": live_evaluation_summary(log_dir),
        "runtime": runtime_summary(log_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    return report


def render_html(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    live = report["live_evaluation"]
    runtime = report["runtime"]
    latest_runtime = runtime.get("latest") or {}
    models = report["models"]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GestureFlow MLOps Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #10111d;
      --panel: #191b2b;
      --line: #30344d;
      --text: #f5f7fb;
      --muted: #aab4c3;
      --accent: #14c6d5;
      --good: #4fd267;
      --bad: #ff5148;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 4px; font-size: 30px; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-top: 16px;
    }}
    .metric {{ font-size: 28px; font-weight: 800; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .mono {{ font-family: Menlo, Consolas, monospace; font-size: 12px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(2, 1fr); }} }}
    @media (max-width: 560px) {{ main {{ padding: 16px; }} .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>GestureFlow MLOps Dashboard</h1>
  <div class="muted">Generated {esc(report["generated_at"])}</div>

  <section class="grid">
    {metric_card("Live accuracy", pct(live["accuracy"]), "correct / all attempts", live["accuracy"] >= 0.8)}
    {metric_card("Live attempts", str(live["attempts"]), "from live_evaluation.jsonl")}
    {metric_card("Negative rejects", str(live["negative_rejections"]), "blocked by rejection layer")}
    {metric_card("Dataset samples", str(dataset["total_samples"]), str(dataset["total_classes"]) + " classes")}
  </section>

  <section class="panel">
    <h2>Runtime</h2>
    {runtime_table(latest_runtime)}
  </section>

  <section class="panel">
    <h2>Live Evaluation By Label</h2>
    {label_table(live["labels"])}
  </section>

  <section class="panel">
    <h2>Routes And Decisions</h2>
	    {counter_table("Route", live["route_counts"])}
	    {counter_table("Dynamic decision source", live["dynamic_decision_sources"])}
	    {counter_table("Static rejection method", live["static_rejection_methods"])}
	    {counter_table("Static decision source", live["static_decision_sources"])}
	    {counter_table("Static rejection reason", live["static_rejection_reasons"])}
	  </section>

  <section class="panel">
    <h2>Dataset</h2>
    {counter_table("Gesture type", dataset["type_counts"])}
    {dataset_table(dataset["classes"])}
  </section>

  <section class="panel">
    <h2>Model Artifacts</h2>
    {model_table(models)}
  </section>
</main>
</body>
</html>
"""


def metric_card(title: str, value: str, caption: str, ok: bool | None = None) -> str:
    cls = ""
    if ok is True:
        cls = " good"
    elif ok is False:
        cls = " bad"
    return (
        '<div class="panel">'
        f"<div class=\"muted\">{esc(title)}</div>"
        f"<div class=\"metric{cls}\">{esc(value)}</div>"
        f"<div class=\"muted\">{esc(caption)}</div>"
        "</div>"
    )


def runtime_table(latest: dict[str, Any]) -> str:
    if not latest:
        return '<div class="muted">No runtime_performance.jsonl rows yet.</div>'
    rows = [
        ("mode", latest.get("recognition_model_mode", "")),
        ("dynamic profile", latest.get("dynamic_model_profile", "")),
        ("target FPS", latest.get("target_fps", "")),
        ("inference avg ms", latest.get("inference_ms_avg", "")),
        ("inference p95 ms", latest.get("inference_ms_p95", "")),
        ("FPS capacity", latest.get("inference_fps_capacity", "")),
        ("shared detection rate", latest.get("shared_detection_rate", "")),
    ]
    return simple_table(["Metric", "Value"], rows)


def label_table(labels: list[dict[str, Any]]) -> str:
    if not labels:
        return '<div class="muted">No live attempts yet.</div>'
    rows = [
        (
            item["expected"],
            item["attempts"],
            item["correct"],
            item["wrong"],
            item["missed"],
            pct(item["accuracy"]),
        )
        for item in labels
    ]
    return simple_table(["Expected", "Attempts", "Correct", "Wrong", "Missed", "Accuracy"], rows)


def counter_table(label: str, values: dict[str, Any]) -> str:
    if not values:
        return f'<div class="muted">No {esc(label.lower())} data.</div>'
    return simple_table([label, "Count"], sorted(values.items()))


def dataset_table(classes: list[dict[str, Any]]) -> str:
    if not classes:
        return '<div class="muted">No samples found.</div>'
    rows = [(item["label"], item["type"], item["samples"]) for item in classes]
    return simple_table(["Label", "Type", "Samples"], rows)


def model_table(models: list[dict[str, Any]]) -> str:
    if not models:
        return '<div class="muted">No model artifacts found.</div>'
    rows = [
        (
            item["name"],
            item["size_bytes"],
            item["sha256"],
            datetime.fromtimestamp(item["modified_at"]).strftime("%Y-%m-%d %H:%M:%S"),
        )
        for item in models
    ]
    return simple_table(["Name", "Bytes", "SHA256", "Modified"], rows)


def simple_table(headers: list[str], rows: list[tuple[Any, ...]]) -> str:
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "gestures")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "configs" / "gesture_taxonomy.json",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_dashboard(
        data_root=args.data_root,
        models_dir=args.models_dir,
        taxonomy_path=args.taxonomy,
        log_dir=args.log_dir,
        out_dir=args.out_dir,
    )
    print(
        "Dashboard written to "
        f"{args.out_dir / 'index.html'} "
        f"({report['live_evaluation']['attempts']} live attempts)"
    )


if __name__ == "__main__":
    main()
