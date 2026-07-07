"""Train first-stage static/dynamic/none intent gate."""

from __future__ import annotations

import argparse
import html
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.services.gesture_taxonomy import (
    GESTURE_TYPE_DYNAMIC,
    GESTURE_TYPE_NEGATIVE,
    GESTURE_TYPE_QUASI_STATIC,
    GESTURE_TYPE_STATIC,
    GestureTaxonomy,
    load_gesture_taxonomy,
)
from cv.gesture_features import GestureSequence, load_gesture_sequences, sequence_to_matrix
from cv.intent_gate import (
    DEFAULT_INTENT_CONFIDENCE_THRESHOLD,
    DEFAULT_INTENT_TARGET_DIM,
    INTENT_DYNAMIC,
    INTENT_NONE,
    INTENT_STATIC,
    build_intent_feature_vector,
    save_intent_gate_model,
    write_intent_gate_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class IntentRecord:
    label: str
    intent: str
    path: Path
    sequence: np.ndarray
    source: str


def map_gesture_type_to_intent(gesture_type: str) -> str:
    clean = str(gesture_type or "").strip().lower()
    if clean in {GESTURE_TYPE_STATIC, GESTURE_TYPE_QUASI_STATIC}:
        return INTENT_STATIC
    if clean == GESTURE_TYPE_DYNAMIC:
        return INTENT_DYNAMIC
    if clean == GESTURE_TYPE_NEGATIVE:
        return INTENT_NONE
    return INTENT_NONE


def load_intent_records(
    data_root: Path,
    *,
    taxonomy: GestureTaxonomy,
    external_negative_root: Path | None = None,
    include_external_negatives: bool = False,
    max_external_negatives: int = 600,
) -> list[IntentRecord]:
    records: list[IntentRecord] = []
    for sample in load_gesture_sequences(data_root):
        gesture_type = taxonomy.gesture_type_for_label(sample.label)
        records.append(
            IntentRecord(
                label=sample.label,
                intent=map_gesture_type_to_intent(gesture_type),
                path=sample.path,
                sequence=sample.sequence,
                source="internal",
            )
        )

    if include_external_negatives and external_negative_root and external_negative_root.exists():
        external_paths = sorted(external_negative_root.rglob("*.npy"))[: max(0, int(max_external_negatives))]
        for path in external_paths:
            try:
                sequence = sequence_to_matrix(np.load(path))
            except Exception:
                continue
            records.append(
                IntentRecord(
                    label=path.parent.name,
                    intent=INTENT_NONE,
                    path=path,
                    sequence=sequence,
                    source="external_negative",
                )
            )
    return records


def build_intent_matrix(
    records: Iterable[IntentRecord],
    *,
    target_dim: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = [INTENT_STATIC, INTENT_DYNAMIC, INTENT_NONE]
    label_to_index = {label: index for index, label in enumerate(labels)}
    X: list[np.ndarray] = []
    y: list[int] = []
    for record in records:
        if record.intent not in label_to_index:
            continue
        try:
            X.append(build_intent_feature_vector(record.sequence, target_dim=target_dim))
            y.append(label_to_index[record.intent])
        except Exception:
            continue
    if not X:
        raise RuntimeError("intent gate dataset is empty")
    return np.stack(X, axis=0), np.asarray(y, dtype=np.int64), labels


def balance_training_data(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(random_state))
    counts = Counter(int(item) for item in y.tolist())
    if not counts:
        return X, y
    target_count = max(counts.values())
    indices: list[int] = []
    for class_index in sorted(counts):
        class_indices = np.flatnonzero(y == class_index)
        if class_indices.size == 0:
            continue
        indices.extend(class_indices.tolist())
        missing = target_count - int(class_indices.size)
        if missing > 0:
            indices.extend(rng.choice(class_indices, size=missing, replace=True).tolist())
    rng.shuffle(indices)
    return X[indices], y[indices]


def train_intent_gate(args: argparse.Namespace) -> dict:
    taxonomy = load_gesture_taxonomy(args.taxonomy)
    records = load_intent_records(
        Path(args.data_root),
        taxonomy=taxonomy,
        external_negative_root=Path(args.external_negative_root),
        include_external_negatives=bool(args.include_external_negatives),
        max_external_negatives=int(args.max_external_negatives),
    )
    X, y, labels = build_intent_matrix(records, target_dim=int(args.target_dim))
    counts = Counter(labels[int(item)] for item in y.tolist())
    stratify = y if min(counts.values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        stratify=stratify,
    )
    X_train_balanced, y_train_balanced = balance_training_data(
        X_train,
        y_train,
        random_state=int(args.random_state),
    )

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=700,
            n_iter_no_change=25,
            random_state=int(args.random_state),
        ),
    )
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    matrix = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))
    by_label = _classification_metrics(matrix, labels)

    payload = {
        "schema_version": 1,
        "model_type": "intent_gate_mlp",
        "model": model,
        "classes": labels,
        "target_dim": int(args.target_dim),
        "feature_dim": int(X.shape[1]),
        "confidence_threshold": float(args.confidence_threshold),
        "trained_at": time.time(),
    }
    out_path = Path(args.out)
    save_intent_gate_model(payload, out_path)
    metadata_out = Path(args.metadata_out)
    write_intent_gate_metadata(payload, metadata_out)

    summary = {
        "generated_at": time.time(),
        "data_root": str(args.data_root),
        "external_negative_root": str(args.external_negative_root),
        "include_external_negatives": bool(args.include_external_negatives),
        "sample_count": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "target_dim": int(args.target_dim),
        "labels": labels,
        "records_by_intent": dict(sorted(counts.items())),
        "records_by_source": dict(Counter(record.source for record in records)),
        "train_count": int(X_train.shape[0]),
        "test_count": int(X_test.shape[0]),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": matrix.astype(int).tolist(),
        "by_label": by_label,
        "model_path": str(out_path),
        "metadata_path": str(metadata_out),
    }
    _write_reports(summary, Path(args.report_json), Path(args.report_md))
    report_svg = Path(args.report_svg)
    _write_confusion_matrix_svg(summary, report_svg)
    _log_mlflow(
        args,
        summary,
        out_path,
        metadata_out,
        Path(args.report_json),
        Path(args.report_md),
        report_svg,
    )
    return summary


def _classification_metrics(matrix: np.ndarray, labels: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    total = float(matrix.sum())
    for index, label in enumerate(labels):
        tp = float(matrix[index, index])
        row_sum = float(matrix[index, :].sum())
        col_sum = float(matrix[:, index].sum())
        non_label_total = max(0.0, total - row_sum)
        false_positive = max(0.0, col_sum - tp)
        out[label] = {
            "precision": tp / col_sum if col_sum else 0.0,
            "recall": tp / row_sum if row_sum else 0.0,
            "false_positive_rate": false_positive / non_label_total if non_label_total else 0.0,
            "support": row_sum,
        }
    return out


def _write_reports(summary: dict, json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    labels = list(summary.get("labels") or [])
    matrix = summary.get("confusion_matrix") or []
    lines = [
        "# Intent Gate Training",
        "",
        f"- samples: `{summary.get('sample_count')}`",
        f"- feature dim: `{summary.get('feature_dim')}`",
        f"- accuracy: `{float(summary.get('accuracy') or 0.0):.4f}`",
        f"- macro F1: `{float(summary.get('macro_f1') or 0.0):.4f}`",
        f"- records by intent: `{json.dumps(summary.get('records_by_intent') or {}, ensure_ascii=False)}`",
        "",
        "## Confusion Matrix",
        "",
        "| expected \\ predicted | " + " | ".join(labels) + " |",
        "|---|" + "|".join(["---:"] * len(labels)) + "|",
    ]
    for index, label in enumerate(labels):
        row = matrix[index] if index < len(matrix) else []
        lines.append("| `" + str(label) + "` | " + " | ".join(str(int(v)) for v in row) + " |")
    lines.extend(["", "## Per-Class Metrics", ""])
    lines.append("| intent | precision | recall | false positive rate | support |")
    lines.append("|---|---:|---:|---:|---:|")
    by_label = summary.get("by_label") or {}
    for label in labels:
        metrics = by_label.get(label) or {}
        lines.append(
            "| `"
            + str(label)
            + "` | "
            + f"{float(metrics.get('precision') or 0.0):.4f}"
            + " | "
            + f"{float(metrics.get('recall') or 0.0):.4f}"
            + " | "
            + f"{float(metrics.get('false_positive_rate') or 0.0):.4f}"
            + " | "
            + f"{int(float(metrics.get('support') or 0.0))}"
            + " |"
        )
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_confusion_matrix_svg(summary: dict, svg_out: Path) -> None:
    labels = [str(label) for label in summary.get("labels") or []]
    matrix = summary.get("confusion_matrix") or []
    if not labels or not matrix:
        return

    cell = 92
    left = 150
    top = 92
    width = left + cell * len(labels) + 44
    height = top + cell * len(labels) + 78
    max_value = max([max([int(value) for value in row] or [0]) for row in matrix] or [1])
    max_value = max(1, int(max_value))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        '<text x="28" y="36" fill="#f8fafc" font-family="Inter, Arial, sans-serif" font-size="22" font-weight="700">Intent gate confusion matrix</text>',
        '<text x="28" y="62" fill="#94a3b8" font-family="Inter, Arial, sans-serif" font-size="13">expected rows, predicted columns</text>',
    ]
    for column, label in enumerate(labels):
        x = left + column * cell + cell / 2
        parts.append(
            f'<text x="{x}" y="{top - 18}" fill="#cbd5e1" font-family="Inter, Arial, sans-serif" font-size="13" text-anchor="middle">{html.escape(label)}</text>'
        )
    for row_index, label in enumerate(labels):
        y = top + row_index * cell + cell / 2 + 5
        parts.append(
            f'<text x="{left - 18}" y="{y}" fill="#cbd5e1" font-family="Inter, Arial, sans-serif" font-size="13" text-anchor="end">{html.escape(label)}</text>'
        )
        row = matrix[row_index] if row_index < len(matrix) else []
        for column_index in range(len(labels)):
            value = int(row[column_index]) if column_index < len(row) else 0
            strength = value / max_value
            fill = "#22c55e" if row_index == column_index else "#ef4444"
            opacity = 0.18 + strength * 0.72
            x = left + column_index * cell
            y0 = top + row_index * cell
            parts.append(
                f'<rect x="{x}" y="{y0}" width="{cell - 6}" height="{cell - 6}" rx="10" fill="{fill}" opacity="{opacity:.3f}"/>'
            )
            parts.append(
                f'<text x="{x + (cell - 6) / 2}" y="{y0 + cell / 2 + 5}" fill="#f8fafc" font-family="Inter, Arial, sans-serif" font-size="22" font-weight="700" text-anchor="middle">{value}</text>'
            )
    parts.append("</svg>")
    svg_out.parent.mkdir(parents=True, exist_ok=True)
    svg_out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _log_mlflow(
    args: argparse.Namespace,
    summary: dict,
    out_path: Path,
    metadata_out: Path,
    report_json: Path,
    report_md: Path,
    report_svg: Path,
) -> None:
    experiment = str(args.mlflow_experiment or "").strip()
    if not experiment:
        return
    try:
        import mlflow
    except Exception as exc:
        print(f"[w] MLflow intent gate logging skipped: {exc}", flush=True)
        return
    try:
        mlflow.set_tracking_uri(str(args.mlflow_tracking_uri))
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=str(args.mlflow_run_name or "intent-gate-mlp")):
            mlflow.set_tags({"run_kind": "intent_gate_training"})
            mlflow.log_params(
                {
                    "model_type": "intent_gate_mlp",
                    "data_root": str(args.data_root),
                    "target_dim": int(args.target_dim),
                    "include_external_negatives": bool(args.include_external_negatives),
                    "confidence_threshold": float(args.confidence_threshold),
                }
            )
            metrics = {
                "sample_count": float(summary.get("sample_count") or 0),
                "feature_dim": float(summary.get("feature_dim") or 0),
                "accuracy": float(summary.get("accuracy") or 0.0),
                "macro_f1": float(summary.get("macro_f1") or 0.0),
            }
            for label, values in (summary.get("by_label") or {}).items():
                safe_label = str(label).replace(" ", "_")
                metrics[f"{safe_label}_precision"] = float(values.get("precision") or 0.0)
                metrics[f"{safe_label}_recall"] = float(values.get("recall") or 0.0)
                metrics[f"{safe_label}_false_positive_rate"] = float(
                    values.get("false_positive_rate") or 0.0
                )
            mlflow.log_metrics(metrics)
            for artifact in (out_path, metadata_out, report_json, report_md, report_svg):
                if artifact.exists():
                    mlflow.log_artifact(str(artifact), artifact_path="intent_gate")
        print(
            f"[✓] MLflow intent gate run logged: experiment={experiment!r}",
            flush=True,
        )
    except Exception as exc:
        print(f"[w] MLflow intent gate logging failed: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GestureBind intent gate")
    parser.add_argument("--data-root", default="data/gestures")
    parser.add_argument("--taxonomy", default=str(PROJECT_ROOT / "configs" / "gesture_taxonomy.json"))
    parser.add_argument("--external-negative-root", default=str(PROJECT_ROOT / "data" / "external"))
    parser.add_argument("--include-external-negatives", action="store_true")
    parser.add_argument("--max-external-negatives", type=int, default=600)
    parser.add_argument("--target-dim", type=int, default=DEFAULT_INTENT_TARGET_DIM)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_INTENT_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--out", default=str(PROJECT_ROOT / "models" / "intent_gate_mlp.pkl"))
    parser.add_argument(
        "--metadata-out",
        default=str(PROJECT_ROOT / "models" / "intent_gate_metadata.json"),
    )
    parser.add_argument(
        "--report-json",
        default=str(PROJECT_ROOT / "docs" / "experiments" / "intent_gate_training.json"),
    )
    parser.add_argument(
        "--report-md",
        default=str(PROJECT_ROOT / "docs" / "experiments" / "intent_gate_training.md"),
    )
    parser.add_argument(
        "--report-svg",
        default=str(PROJECT_ROOT / "docs" / "experiments" / "intent_gate_confusion_matrix.svg"),
    )
    parser.add_argument("--mlflow-experiment", default="GestureBind")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow-run-name", default="intent-gate-mlp")
    return parser.parse_args()


def main() -> None:
    summary = train_intent_gate(parse_args())
    print(
        "[intent-gate] "
        f"samples={summary['sample_count']} accuracy={summary['accuracy']:.4f} "
        f"macro_f1={summary['macro_f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
