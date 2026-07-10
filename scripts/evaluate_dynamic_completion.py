"""Benchmark the production completion gate on full and truncated gestures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from cv.dynamic_completion import (
    build_dynamic_completion_evidence,
    completion_progress_prefix,
    evaluate_dynamic_completion,
)
from cv.gesture_dataset_files import real_sample_paths
from cv.gesture_features import build_feature_vector, sequence_to_matrix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "dynamic_landmark_lstm_backbone.pkl"
DEFAULT_CLASSES = ROOT / "models" / "dynamic_landmark_lstm_backbone_classes.json"
DEFAULT_REJECTION = (
    ROOT / "models" / "dynamic_landmark_lstm_backbone_rejection.json"
)
DEFAULT_FEATURE_MODE = (
    ROOT / "models" / "dynamic_landmark_lstm_backbone_feature_mode.txt"
)
DEFAULT_FEATURE_DIM = (
    ROOT / "models" / "dynamic_landmark_lstm_backbone_feature_dim.txt"
)
DEFAULT_PROTOTYPES = (
    ROOT / "models" / "dynamic_landmark_lstm_backbone_prototypes.json"
)
DEFAULT_DATA_ROOT = ROOT / "data" / "gestures"
DEFAULT_REPORT_JSON = (
    ROOT / "docs" / "experiments" / "dynamic_completion_benchmark.json"
)
DEFAULT_REPORT_MD = (
    ROOT / "docs" / "experiments" / "dynamic_completion_benchmark.md"
)


def _model_label(raw_class: Any, classes: list[str]) -> str:
    if isinstance(raw_class, (int, np.integer)):
        index = int(raw_class)
        return classes[index] if 0 <= index < len(classes) else str(index)
    clean = str(raw_class or "").strip()
    try:
        index = int(clean)
    except ValueError:
        return clean
    return classes[index] if 0 <= index < len(classes) else clean


def _sample_scale(path: Path) -> float:
    try:
        payload = json.loads(path.with_suffix(".meta.json").read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return 0.0
    quality = payload.get("quality")
    if not isinstance(quality, dict):
        quality = {}
    return float(
        payload.get("projected_hand_scale_median")
        or quality.get("projected_hand_scale")
        or 0.0
    )


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _report_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def evaluate(
    *,
    data_root: Path,
    model_path: Path,
    classes_path: Path,
    rejection_path: Path,
    feature_mode_path: Path,
    target_dim: int,
    target_frames: int,
    confidence_threshold: float,
) -> dict[str, Any]:
    classes = [
        str(value)
        for value in json.loads(classes_path.read_text(encoding="utf-8"))
    ]
    rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
    profiles = rejection.get("dynamic_completion_profiles")
    if not isinstance(profiles, dict) or not profiles.get("classes"):
        raise RuntimeError("completion profiles are missing from rejection metadata")
    profile_classes = profiles.get("classes")
    if not isinstance(profile_classes, dict):
        raise RuntimeError("completion profile classes are invalid")

    feature_mode = feature_mode_path.read_text(encoding="utf-8").strip()
    model = joblib.load(model_path)
    estimator_classes = list(getattr(model, "classes_", range(len(classes))))
    prefix_fractions = [
        float(value) for value in profiles.get("prefix_fractions", [])
    ]
    fractions = [1.0, *sorted(prefix_fractions, reverse=True)]
    rows: list[dict[str, Any]] = []

    for label in profile_classes:
        label_dir = data_root / label
        for fraction in fractions:
            total = 0
            model_accepted = 0
            gate_accepted = 0
            final_accepted = 0
            scores: list[float] = []
            for sample_path in real_sample_paths(label_dir):
                sequence = sequence_to_matrix(np.load(sample_path)).astype(
                    np.float32,
                    copy=False,
                )
                if int(sequence.shape[1]) != int(target_dim):
                    continue
                scale = _sample_scale(sample_path)
                if fraction >= 1.0:
                    candidate = sequence
                else:
                    candidate = completion_progress_prefix(
                        sequence,
                        fraction,
                        motion_scale=scale,
                    )
                total += 1
                evidence = build_dynamic_completion_evidence(
                    candidate,
                    motion_scale=scale,
                    target_frames=target_frames,
                )
                gate = evaluate_dynamic_completion(profiles, label, evidence)
                feature = build_feature_vector(
                    candidate,
                    mode=feature_mode,
                    target_dim=target_dim,
                ).reshape(1, -1)
                probabilities = np.asarray(
                    model.predict_proba(feature)[0],
                    dtype=float,
                )
                column = int(np.argmax(probabilities))
                predicted = _model_label(estimator_classes[column], classes)
                confidence = float(probabilities[column])
                model_ok = (
                    predicted == label
                    and confidence >= float(confidence_threshold)
                )
                gate_ok = bool(gate.get("accepted"))
                model_accepted += int(model_ok)
                gate_accepted += int(gate_ok)
                final_accepted += int(model_ok and gate_ok)
                scores.append(float(gate.get("score") or 0.0))

            rows.append(
                {
                    "label": label,
                    "fraction": float(fraction),
                    "kind": "full" if fraction >= 1.0 else "prefix",
                    "total": total,
                    "model_accepted": model_accepted,
                    "model_accept_rate": _rate(model_accepted, total),
                    "gate_accepted": gate_accepted,
                    "gate_accept_rate": _rate(gate_accepted, total),
                    "final_accepted": final_accepted,
                    "final_accept_rate": _rate(final_accepted, total),
                    "completion_score_min": min(scores, default=0.0),
                    "completion_score_median": (
                        float(np.median(scores)) if scores else 0.0
                    ),
                }
            )

    full_rows = [row for row in rows if row["kind"] == "full"]
    prefix_rows = [row for row in rows if row["kind"] == "prefix"]

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = sum(int(row["total"]) for row in items)
        model_accepted = sum(int(row["model_accepted"]) for row in items)
        gate_accepted = sum(int(row["gate_accepted"]) for row in items)
        final_accepted = sum(int(row["final_accepted"]) for row in items)
        return {
            "total": total,
            "model_accepted": model_accepted,
            "model_accept_rate": _rate(model_accepted, total),
            "gate_accepted": gate_accepted,
            "gate_accept_rate": _rate(gate_accepted, total),
            "final_accepted": final_accepted,
            "final_accept_rate": _rate(final_accepted, total),
        }

    full = aggregate(full_rows)
    prefix = aggregate(prefix_rows)
    before = float(prefix["model_accept_rate"])
    after = float(prefix["final_accept_rate"])
    reduction = (before - after) / before if before > 0.0 else 0.0
    return {
        "schema_version": 2,
        "generated_at": time.time(),
        "model": _report_path(model_path),
        "classes": _report_path(classes_path),
        "rejection": _report_path(rejection_path),
        "feature_mode": feature_mode,
        "target_dim": int(target_dim),
        "target_frames": int(target_frames),
        "confidence_threshold": float(confidence_threshold),
        "profile_method": str(profiles.get("method") or ""),
        "profile_classes": sorted(str(label) for label in profile_classes),
        "full": full,
        "prefix": prefix,
        "prefix_false_accept_reduction": float(reduction),
        "rows": rows,
    }


def _replay_online_candidate(
    infer: Any,
    sequence: np.ndarray,
    *,
    motion_scale: float,
    confidence_threshold: float,
    hold_frames: int,
) -> dict[str, Any]:
    infer.reset_temporal_state()
    completion_rejections = 0
    accepted_label = ""
    accepted_confidence = 0.0
    accepted_frame = -1
    frames = [*sequence, *([sequence[-1]] * max(0, int(hold_frames)))]
    for frame_index, frame in enumerate(frames):
        feature = np.asarray(frame, dtype=np.float32)
        infer._ensure_intent_window().append(feature)
        output = infer._process_segmented_dynamic_frame(
            feature,
            "[]",
            motion_scale=float(motion_scale),
        )
        decision = output.get("dynamic_decision") or {}
        if str(decision.get("source") or "") == "completion_rejected":
            completion_rejections += 1
        label = str(output.get("label") or "").strip()
        confidence = float(output.get("confidence") or 0.0)
        if label and confidence >= float(confidence_threshold):
            accepted_label = label
            accepted_confidence = confidence
            accepted_frame = int(frame_index)
            break
    return {
        "accepted_label": accepted_label,
        "accepted_confidence": float(accepted_confidence),
        "accepted_frame": int(accepted_frame),
        "completion_rejections": int(completion_rejections),
    }


def evaluate_online_replay(
    *,
    data_root: Path,
    model_path: Path,
    classes_path: Path,
    rejection_path: Path,
    feature_dim_path: Path,
    feature_mode_path: Path,
    prototypes_path: Path,
    target_dim: int,
    confidence_threshold: float,
    hold_frames: int = 12,
) -> dict[str, Any]:
    """Replay recordings through the same state machine used by live mode."""
    from app.gesture_online_infer import GestureOnlineInfer

    rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
    profiles = rejection.get("dynamic_completion_profiles") or {}
    profile_classes = profiles.get("classes") or {}
    fractions = [float(value) for value in profiles.get("prefix_fractions", [])]
    infer = GestureOnlineInfer(
        model_path=model_path,
        classes_path=classes_path,
        feature_dim_path=feature_dim_path,
        feature_mode_path=feature_mode_path,
        gesture_rejection_path=rejection_path,
        dynamic_prototypes_path=prototypes_path,
        initialize_detector=False,
    )
    if infer.init_error or infer.model_error:
        raise RuntimeError(
            "online replay could not initialize: "
            f"{infer.init_error or infer.model_error}"
        )

    attempts: list[dict[str, Any]] = []
    for label in profile_classes:
        for sample_path in real_sample_paths(data_root / str(label)):
            sequence = sequence_to_matrix(np.load(sample_path)).astype(
                np.float32,
                copy=False,
            )
            if int(sequence.shape[1]) != int(target_dim):
                continue
            scale = _sample_scale(sample_path)
            candidates = [(1.0, sequence)]
            candidates.extend(
                (
                    fraction,
                    completion_progress_prefix(
                        sequence,
                        fraction,
                        motion_scale=scale,
                    ),
                )
                for fraction in fractions
            )
            for fraction, candidate in candidates:
                result = _replay_online_candidate(
                    infer,
                    candidate,
                    motion_scale=scale,
                    confidence_threshold=confidence_threshold,
                    hold_frames=hold_frames,
                )
                accepted_label = str(result["accepted_label"])
                attempts.append(
                    {
                        "label": str(label),
                        "sample": _report_path(sample_path),
                        "fraction": float(fraction),
                        "kind": "full" if fraction >= 1.0 else "prefix",
                        "accepted_label": accepted_label,
                        "accepted_confidence": float(
                            result["accepted_confidence"]
                        ),
                        "accepted_frame": int(result["accepted_frame"]),
                        "completion_rejections": int(
                            result["completion_rejections"]
                        ),
                        "correct": bool(
                            fraction >= 1.0 and accepted_label == str(label)
                        ),
                        "wrong": bool(
                            accepted_label and accepted_label != str(label)
                        ),
                    }
                )

    full_attempts = [row for row in attempts if row["kind"] == "full"]
    prefix_attempts = [row for row in attempts if row["kind"] == "prefix"]

    def summarize(rows: list[dict[str, Any]], *, prefix: bool) -> dict[str, Any]:
        accepted = sum(bool(row["accepted_label"]) for row in rows)
        correct = sum(bool(row["correct"]) for row in rows)
        wrong = sum(bool(row["wrong"]) for row in rows)
        completion_rejected = sum(
            int(row["completion_rejections"]) > 0 for row in rows
        )
        result = {
            "total": int(len(rows)),
            "accepted": int(accepted),
            "accept_rate": _rate(accepted, len(rows)),
            "no_command": int(len(rows) - accepted),
            "wrong_class": int(wrong),
            "completion_rejected_attempts": int(completion_rejected),
            "completion_rejected_attempt_rate": _rate(
                completion_rejected,
                len(rows),
            ),
        }
        if prefix:
            result["false_accept_rate"] = _rate(accepted, len(rows))
        else:
            result["correct"] = int(correct)
            result["correct_rate"] = _rate(correct, len(rows))
        return result

    return {
        "method": "gesture_online_infer_state_machine_replay",
        "confidence_threshold": float(confidence_threshold),
        "hold_frames": int(hold_frames),
        "full": summarize(full_attempts, prefix=False),
        "prefix": summarize(prefix_attempts, prefix=True),
        "attempts": attempts,
    }


def render_markdown(report: dict[str, Any]) -> str:
    full = report["full"]
    prefix = report["prefix"]
    lines = [
        "# Dynamic Completion Gate Benchmark",
        "",
        "This benchmark truncates every real positive recording by cumulative "
        "motion progress at the same fractions used to train the "
        "class-conditional completion profiles. Prefixes remain in raw "
        "scale-aware coordinates until the gate runs.",
        "",
        "## Summary",
        "",
        "| Set | Attempts | LSTM accepted | Gate accepted | Final accepted |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Full | `{full['total']}` | `{full['model_accepted']}` "
            f"(`{full['model_accept_rate']:.4f}`) | `{full['gate_accepted']}` "
            f"(`{full['gate_accept_rate']:.4f}`) | `{full['final_accepted']}` "
            f"(`{full['final_accept_rate']:.4f}`) |"
        ),
        (
            f"| Prefix | `{prefix['total']}` | `{prefix['model_accepted']}` "
            f"(`{prefix['model_accept_rate']:.4f}`) | `{prefix['gate_accepted']}` "
            f"(`{prefix['gate_accept_rate']:.4f}`) | `{prefix['final_accepted']}` "
            f"(`{prefix['final_accept_rate']:.4f}`) |"
        ),
        "",
        "Prefix false-accept reduction: "
        f"`{report['prefix_false_accept_reduction']:.4f}`.",
        "",
    ]
    online = report.get("online_replay")
    if isinstance(online, dict):
        online_full = online["full"]
        online_prefix = online["prefix"]
        lines.extend(
            [
                "## Online State-Machine Replay",
                "",
                "| Set | Attempts | Correct | Wrong class | No command |",
                "|---|---:|---:|---:|---:|",
                (
                    f"| Full | `{online_full['total']}` | "
                    f"`{online_full['correct']}` "
                    f"(`{online_full['correct_rate']:.4f}`) | "
                    f"`{online_full['wrong_class']}` | "
                    f"`{online_full['no_command']}` |"
                ),
                (
                    f"| Prefix | `{online_prefix['total']}` | n/a | "
                    f"`{online_prefix['wrong_class']}` | "
                    f"`{online_prefix['no_command']}` |"
                ),
                "",
                "Online prefix false-accept rate: "
                f"`{online_prefix['false_accept_rate']:.4f}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Per Class And Fraction",
            "",
            "| Class | Fraction | Attempts | LSTM accepted | Gate accepted | Final accepted |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["rows"]:
        lines.append(
            f"| `{row['label']}` | `{row['fraction']:.2f}` | "
            f"`{row['total']}` | `{row['model_accepted']}` | "
            f"`{row['gate_accepted']}` | `{row['final_accepted']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The report measures source recordings from the current personal "
            "dataset. Prefixes from one source never cross validation groups "
            "during profile calibration. It is an offline safety benchmark, "
            "not a replacement for the final live partial-motion matrix.",
            "",
        ]
    )
    return "\n".join(lines)


def _log_mlflow(report: dict[str, Any], tracking_uri: str, experiment: str) -> str:
    import mlflow

    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(str(experiment))
    with mlflow.start_run(run_name="dynamic-completion-release-benchmark") as run:
        mlflow.set_tags(
            {
                "run_kind": "completion_benchmark",
                "source": "scripts.evaluate_dynamic_completion",
                "release_candidate": "v0.8.0",
            }
        )
        mlflow.log_params(
            {
                "profile_method": report["profile_method"],
                "feature_mode": report["feature_mode"],
                "target_dim": report["target_dim"],
                "target_frames": report["target_frames"],
                "confidence_threshold": report["confidence_threshold"],
                "profile_class_count": len(report["profile_classes"]),
            }
        )
        mlflow.log_metrics(
            {
                "completion_full_accept_rate": report["full"][
                    "final_accept_rate"
                ],
                "completion_prefix_model_false_accept_rate": report["prefix"][
                    "model_accept_rate"
                ],
                "completion_prefix_final_false_accept_rate": report["prefix"][
                    "final_accept_rate"
                ],
                "completion_prefix_false_accept_reduction": report[
                    "prefix_false_accept_reduction"
                ],
                "completion_full_attempts": report["full"]["total"],
                "completion_prefix_attempts": report["prefix"]["total"],
            }
        )
        online = report.get("online_replay")
        if isinstance(online, dict):
            mlflow.log_metrics(
                {
                    "completion_online_full_correct_rate": online["full"][
                        "correct_rate"
                    ],
                    "completion_online_full_wrong_class": online["full"][
                        "wrong_class"
                    ],
                    "completion_online_prefix_false_accept_rate": online[
                        "prefix"
                    ]["false_accept_rate"],
                    "completion_online_prefix_attempts": online["prefix"][
                        "total"
                    ],
                }
            )
        mlflow.log_dict(report, "dynamic_completion_benchmark.json")
        return str(run.info.run_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--rejection", type=Path, default=DEFAULT_REJECTION)
    parser.add_argument("--feature-mode", type=Path, default=DEFAULT_FEATURE_MODE)
    parser.add_argument("--feature-dim", type=Path, default=DEFAULT_FEATURE_DIM)
    parser.add_argument("--prototypes", type=Path, default=DEFAULT_PROTOTYPES)
    parser.add_argument("--target-dim", type=int, default=65)
    parser.add_argument("--target-frames", type=int, default=72)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--skip-online-replay", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow-experiment", default="GestureBind-Completion")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(
        data_root=args.data_root,
        model_path=args.model,
        classes_path=args.classes,
        rejection_path=args.rejection,
        feature_mode_path=args.feature_mode,
        target_dim=int(args.target_dim),
        target_frames=int(args.target_frames),
        confidence_threshold=max(0.0, min(1.0, float(args.confidence_threshold))),
    )
    if not args.skip_online_replay:
        report["online_replay"] = evaluate_online_replay(
            data_root=args.data_root,
            model_path=args.model,
            classes_path=args.classes,
            rejection_path=args.rejection,
            feature_dim_path=args.feature_dim,
            feature_mode_path=args.feature_mode,
            prototypes_path=args.prototypes,
            target_dim=int(args.target_dim),
            confidence_threshold=max(
                0.0,
                min(1.0, float(args.confidence_threshold)),
            ),
        )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    if args.log_mlflow:
        run_id = _log_mlflow(
            report,
            str(args.mlflow_tracking_uri),
            str(args.mlflow_experiment),
        )
        print(f"[dynamic-completion] MLflow run={run_id}")
    print(
        "[dynamic-completion] "
        f"full={report['full']['final_accepted']}/{report['full']['total']} "
        f"prefix_before={report['prefix']['model_accepted']}/{report['prefix']['total']} "
        f"prefix_after={report['prefix']['final_accepted']}/{report['prefix']['total']}"
    )
    online = report.get("online_replay")
    if isinstance(online, dict):
        print(
            "[dynamic-completion] online "
            f"full={online['full']['correct']}/{online['full']['total']} "
            f"wrong={online['full']['wrong_class']} "
            f"prefix_false_accept={online['prefix']['accepted']}/"
            f"{online['prefix']['total']}"
        )
    print(f"[dynamic-completion] report={args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
