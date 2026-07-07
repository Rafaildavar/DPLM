"""Analyze whether external IPN dynamic samples help GestureBind rejection."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cv.dynamic_prototype import load_dynamic_prototype_model, predict_dynamic_prototype
from cv.gesture_features import trajectory_features
from scripts.convert_ipn_hand import _resolve_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISTANCE_MODEL = (
    PROJECT_ROOT
    / "models"
    / "experiments"
    / "dynamic_prototype"
    / "prototype_distance"
    / "dynamic_prototypes.json"
)
DEFAULT_DTW_MODEL = (
    PROJECT_ROOT
    / "models"
    / "experiments"
    / "dynamic_prototype"
    / "prototype_dtw"
    / "dynamic_prototypes.json"
)
DEFAULT_JSON_OUT = PROJECT_ROOT / "docs" / "experiments" / "ipn_external_negative_analysis.json"
DEFAULT_MD_OUT = PROJECT_ROOT / "docs" / "experiments" / "ipn_external_negative_analysis.md"


@dataclass(frozen=True)
class RootAnalysis:
    name: str
    root: str
    sample_count: int
    frame_count_mean: float
    detection_rate_mean: float
    original_labels: dict[str, int]
    motion_directions: dict[str, int]
    model_metrics: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class IpnExternalAnalysis:
    generated_at: float
    roots: list[RootAnalysis] = field(default_factory=list)
    recommendation: str = ""
    warnings: list[str] = field(default_factory=list)


def analyze_roots(
    roots: dict[str, Path],
    *,
    prototype_models: dict[str, Path],
    near_threshold_multiplier: float = 1.15,
) -> IpnExternalAnalysis:
    loaded_models = {
        name: load_dynamic_prototype_model(_resolve_path(path))
        for name, path in prototype_models.items()
        if _resolve_path(path).exists()
    }
    warnings: list[str] = []
    for name, path in prototype_models.items():
        if name not in loaded_models:
            warnings.append(f"prototype model not found: {path}")

    analyses: list[RootAnalysis] = []
    for name, root in roots.items():
        root_path = _resolve_path(root)
        samples = sorted(root_path.rglob("*.npy")) if root_path.exists() else []
        if not root_path.exists():
            warnings.append(f"external root not found: {root_path}")
        analyses.append(
            _analyze_root(
                name=name,
                root=root_path,
                samples=samples,
                models=loaded_models,
                near_threshold_multiplier=near_threshold_multiplier,
            )
        )

    recommendation = _recommend(analyses)
    return IpnExternalAnalysis(
        generated_at=time.time(),
        roots=analyses,
        recommendation=recommendation,
        warnings=warnings,
    )


def build_markdown_report(report: IpnExternalAnalysis) -> str:
    lines = [
        "# IPN External Negative Analysis",
        "",
        f"Generated at: `{report.generated_at:.3f}`",
        "",
        "## Recommendation",
        "",
        report.recommendation or "No recommendation.",
        "",
        "## Root Summary",
        "",
        "| Root | Samples | Mean frames | Mean detection | Motion directions |",
        "|---|---:|---:|---:|---|",
    ]
    for root in report.roots:
        lines.append(
            f"| `{root.name}` | {root.sample_count} | "
            f"{root.frame_count_mean:.2f} | {root.detection_rate_mean:.4f} | "
            f"`{json.dumps(root.motion_directions, ensure_ascii=False)}` |"
        )

    lines.extend(["", "## Model Safety", ""])
    for root in report.roots:
        lines.extend([f"### `{root.name}`", ""])
        lines.append(
            "| Model | Negative reject | False positive | Near positive | "
            "Accepted labels | Reject reasons |"
        )
        lines.append("|---|---:|---:|---:|---|---|")
        for model_name, metrics in sorted(root.model_metrics.items()):
            lines.append(
                f"| `{model_name}` | {metrics['negative_reject_rate']:.4f} | "
                f"{metrics['false_positive_rate']:.4f} | "
                f"{metrics['near_positive_rate']:.4f} | "
                f"`{json.dumps(metrics['accepted_labels'], ensure_ascii=False)}` | "
                f"`{json.dumps(metrics['reject_reasons'], ensure_ascii=False)}` |"
            )

    lines.extend(["", "## Original IPN Labels", ""])
    for root in report.roots:
        lines.extend([f"### `{root.name}`", "", "| Label | Samples |", "|---|---:|"])
        if root.original_labels:
            for label, count in sorted(root.original_labels.items()):
                lines.append(f"| `{label}` | {count} |")
        else:
            lines.append("| none | 0 |")
        lines.append("")

    if report.warnings:
        lines.extend(["## Warnings", ""])
        for warning in report.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Useful external negatives should have high negative reject rate and low false positive rate.",
            "- `near_positive_rate` flags samples that are rejected but lie close to swipe prototypes; these are good validation cases, but risky training negatives.",
            "- IPN throw-left/up/down/right classes should remain validation/reference data unless we intentionally model them as user commands.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(
    report: IpnExternalAnalysis,
    *,
    json_out: Path = DEFAULT_JSON_OUT,
    md_out: Path = DEFAULT_MD_OUT,
) -> None:
    json_out = _resolve_path(json_out)
    md_out = _resolve_path(md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_out.write_text(build_markdown_report(report), encoding="utf-8")


def _analyze_root(
    *,
    name: str,
    root: Path,
    samples: list[Path],
    models: dict[str, dict[str, Any]],
    near_threshold_multiplier: float,
) -> RootAnalysis:
    frame_counts: list[float] = []
    detection_rates: list[float] = []
    original_labels: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    model_state: dict[str, dict[str, Any]] = {
        model_name: {
            "total": 0,
            "rejected": 0,
            "false_positive": 0,
            "near_positive": 0,
            "accepted_labels": Counter(),
            "reject_reasons": Counter(),
            "nearest_labels": Counter(),
            "distance_ratio_values": [],
        }
        for model_name in models
    }

    for sample_path in samples:
        try:
            sequence = np.load(sample_path)
        except Exception:
            continue
        if sequence.ndim != 2 or sequence.shape[0] < 2:
            continue
        frame_counts.append(float(sequence.shape[0]))
        metadata = _read_metadata(sample_path)
        original_labels[str(metadata.get("original_label") or sample_path.parent.name)] += 1
        detection_rate = _float(metadata.get("detection_rate"))
        if detection_rate is not None:
            detection_rates.append(detection_rate)
        directions[_motion_direction(sequence)] += 1

        for model_name, payload in models.items():
            decision = predict_dynamic_prototype(payload, sequence)
            state = model_state[model_name]
            state["total"] += 1
            label = str(decision.get("label") or "")
            nearest_label = str(decision.get("nearest_label") or decision.get("label") or "")
            nearest_type = str(decision.get("nearest_type") or "")
            reason = str(decision.get("reason") or "")
            distance = _float(decision.get("distance"))
            threshold = _float(decision.get("threshold"))
            if label:
                state["false_positive"] += 1
                state["accepted_labels"][label] += 1
            else:
                state["rejected"] += 1
                state["reject_reasons"][reason or "unknown"] += 1
            if nearest_label:
                state["nearest_labels"][nearest_label] += 1
            if (
                nearest_type == "positive"
                and distance is not None
                and threshold is not None
                and threshold > 0.0
            ):
                ratio = float(distance / threshold)
                state["distance_ratio_values"].append(ratio)
                if ratio <= float(near_threshold_multiplier):
                    state["near_positive"] += 1

    model_metrics = {
        model_name: _finalize_model_metrics(state)
        for model_name, state in model_state.items()
    }
    return RootAnalysis(
        name=name,
        root=str(root),
        sample_count=len(frame_counts),
        frame_count_mean=_mean(frame_counts),
        detection_rate_mean=_mean(detection_rates),
        original_labels=dict(sorted(original_labels.items())),
        motion_directions=dict(sorted(directions.items())),
        model_metrics=model_metrics,
    )


def _finalize_model_metrics(state: dict[str, Any]) -> dict[str, Any]:
    total = int(state["total"])
    false_positive = int(state["false_positive"])
    rejected = int(state["rejected"])
    ratios = [float(value) for value in state["distance_ratio_values"]]
    return {
        "total": total,
        "negative_reject_rate": float(rejected / total) if total else 0.0,
        "false_positive_rate": float(false_positive / total) if total else 0.0,
        "false_positive_count": false_positive,
        "near_positive_rate": float(int(state["near_positive"]) / total) if total else 0.0,
        "near_positive_count": int(state["near_positive"]),
        "accepted_labels": dict(sorted(state["accepted_labels"].items())),
        "reject_reasons": dict(sorted(state["reject_reasons"].items())),
        "nearest_labels": dict(sorted(state["nearest_labels"].items())),
        "positive_distance_ratio_mean": _mean(ratios),
        "positive_distance_ratio_p95": _percentile(ratios, 95),
    }


def _recommend(analyses: Iterable[RootAnalysis]) -> str:
    parts: list[str] = []
    for root in analyses:
        distance = root.model_metrics.get("prototype_distance") or {}
        fp = float(distance.get("false_positive_rate") or 0.0)
        near = float(distance.get("near_positive_rate") or 0.0)
        if root.sample_count <= 0:
            parts.append(f"`{root.name}` is empty; do not use it yet.")
        elif fp == 0.0 and near <= 0.10:
            parts.append(
                f"`{root.name}` is safe as dynamic negative training/validation data "
                f"for `prototype_distance`."
            )
        elif fp == 0.0:
            parts.append(
                f"`{root.name}` has no false positives, but near-swipe rate is "
                f"{near:.2%}; prefer validation or limited training."
            )
        else:
            parts.append(
                f"`{root.name}` creates prototype false positives ({fp:.2%}); "
                "use it for analysis only until mapping/thresholds are refined."
            )
    return " ".join(parts)


def _read_metadata(sample_path: Path) -> dict[str, Any]:
    metadata_path = sample_path.with_suffix(".meta.json")
    if not metadata_path.exists():
        return {}
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _motion_direction(sequence: np.ndarray) -> str:
    try:
        dx, dy, abs_dx, abs_dy, path_length, *_ = [
            float(value)
            for value in trajectory_features(sequence, target_dim=int(sequence.shape[1]))
        ]
    except Exception:
        return "unknown"
    if path_length < 0.03:
        return "low_motion"
    if abs_dx >= abs_dy * 1.2:
        return "left" if dx < 0 else "right"
    if abs_dy >= abs_dx * 1.2:
        return "up" if dy < 0 else "down"
    return "diagonal_or_complex"


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(statistics.fmean(values)) if values else 0.0


def _percentile(values: Iterable[float], percentile: float) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    index = min(
        len(values) - 1,
        max(0, int(round((len(values) - 1) * float(percentile) / 100.0))),
    )
    return float(values[index])


def _parse_roots(raw: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).name or "external"
        out[name.strip()] = Path(path.strip())
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        default=(
            "existing=data/external/ipn_hand,"
            "tar_expanded=data/external/ipn_hand_tar"
        ),
    )
    parser.add_argument("--prototype-distance-model", type=Path, default=DEFAULT_DISTANCE_MODEL)
    parser.add_argument("--prototype-dtw-model", type=Path, default=DEFAULT_DTW_MODEL)
    parser.add_argument("--near-threshold-multiplier", type=float, default=1.15)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_roots(
        _parse_roots(args.roots),
        prototype_models={
            "prototype_distance": args.prototype_distance_model,
            "prototype_dtw": args.prototype_dtw_model,
        },
        near_threshold_multiplier=float(args.near_threshold_multiplier),
    )
    write_report(report, json_out=args.json_out, md_out=args.md_out)
    print(build_markdown_report(report))
    print(f"[✓] JSON: {_resolve_path(args.json_out)}")
    print(f"[✓] Markdown: {_resolve_path(args.md_out)}")


if __name__ == "__main__":
    main()
