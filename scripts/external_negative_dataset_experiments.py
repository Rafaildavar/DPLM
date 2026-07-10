"""Run external public-dataset negative experiments for GestureBind.

The script keeps public datasets out of the production data folder. It expects
external samples already converted to the GestureBind landmark format:

    data/external/<source>/<original_label>/sample_*.npy

Each external source is mapped to a canonical negative label, materialized into
an experiment-only data root and compared against the internal baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.gesture_taxonomy import (  # noqa: E402
    GESTURE_TYPE_DYNAMIC,
    GESTURE_TYPE_NEGATIVE,
    GESTURE_TYPE_QUASI_STATIC,
    GESTURE_TYPE_STATIC,
    load_gesture_taxonomy,
)
from cv.gesture_dataset_files import gesture_sample_paths  # noqa: E402
from cv.gesture_features import FEATURE_DYNAMIC_STATS, FEATURE_STATIC_MEAN  # noqa: E402
from cv.train_classifier import (  # noqa: E402
    build_classifier,
    build_rejection_metadata,
    load_dataset,
)
from scripts.rejection_method_benchmark import (  # noqa: E402
    RejectionBenchmarkReport,
    benchmark_rejection_methods,
    build_markdown_report,
)
from scripts.train_static_rejection_verifiers import (  # noqa: E402
    train_static_rejection_verifiers,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "external_negative_datasets.json"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "gestures"
DEFAULT_WORK_ROOT = PROJECT_ROOT / "data" / "experiments" / "external_negative"
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "models" / "experiments" / "external_negative"
DEFAULT_DOC_JSON = (
    PROJECT_ROOT / "docs" / "experiments" / "external_negative_dataset_experiments.json"
)
DEFAULT_DOC_MD = (
    PROJECT_ROOT / "docs" / "experiments" / "external_negative_dataset_experiments.md"
)
DEFAULT_METHODS = (
    "negative_classes",
    "confidence_threshold",
    "open_set_policy",
    "one_vs_rest_logreg",
    "one_class_svm",
    "isolation_forest",
    "local_outlier_factor",
    "metric_nca_centroid",
    "mlp_negative_classes",
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: str
    domain: str
    target_label: str
    include_unmapped: bool = True
    include_labels: list[str] = field(default_factory=list)
    exclude_labels: list[str] = field(default_factory=list)
    max_samples_per_label: int = 80
    reference_url: str = ""


@dataclass(frozen=True)
class VariantSpec:
    name: str
    sources: list[str]


@dataclass(frozen=True)
class ImportedExternalSummary:
    source: str
    source_root: str
    target_label: str
    status: str
    labels: dict[str, int]
    imported_samples: int
    skipped_reason: str = ""


@dataclass(frozen=True)
class MaterializedVariantSummary:
    variant: str
    data_root: str
    model_root: str
    internal_samples: int
    external_samples: int
    class_counts: dict[str, int]
    sources: list[ImportedExternalSummary]
    status: str
    skipped_reason: str = ""


@dataclass(frozen=True)
class ScopeBenchmarkSummary:
    variant: str
    scope: str
    status: str
    best_method: str = ""
    recommendation: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class TrainedArtifactSummary:
    variant: str
    scope: str
    status: str
    model_root: str
    model_paths: dict[str, str] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)
    sample_count: int = 0
    feature_mode: str = ""
    feature_dim: int = 0
    error: str = ""


@dataclass(frozen=True)
class ExternalNegativeExperimentReport:
    generated_at: float
    config_path: str
    internal_data_root: str
    variants: list[MaterializedVariantSummary]
    benchmarks: list[ScopeBenchmarkSummary]
    artifacts: list[TrainedArtifactSummary]
    notes: list[str]


def _resolve_path(raw: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else base / path


def _slug(value: str) -> str:
    clean = str(value or "").strip().lower()
    clean = re.sub(r"[^a-z0-9а-яё_-]+", "_", clean, flags=re.IGNORECASE)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "unknown"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_specs(config_path: Path) -> tuple[dict[str, SourceSpec], list[VariantSpec]]:
    raw = _load_json(config_path)
    sources_raw = raw.get("sources") or {}
    variants_raw = raw.get("variants") or []
    if not isinstance(sources_raw, dict):
        raise ValueError("external config field 'sources' must be an object")
    if not isinstance(variants_raw, list):
        raise ValueError("external config field 'variants' must be a list")

    sources: dict[str, SourceSpec] = {}
    for name, item in sources_raw.items():
        if not isinstance(item, dict):
            continue
        source_name = str(name).strip()
        root = str(item.get("root") or "").strip()
        domain = str(item.get("domain") or "static").strip().lower()
        target_label = str(
            item.get("target_label") or f"negative_external_{_slug(source_name)}_{domain}"
        ).strip()
        if not source_name or not root:
            continue
        sources[source_name] = SourceSpec(
            name=source_name,
            root=root,
            domain=domain,
            target_label=target_label,
            include_unmapped=bool(item.get("include_unmapped", True)),
            include_labels=[str(label).strip() for label in item.get("include_labels") or []],
            exclude_labels=[str(label).strip() for label in item.get("exclude_labels") or []],
            max_samples_per_label=max(1, int(item.get("max_samples_per_label") or 80)),
            reference_url=str(item.get("reference_url") or "").strip(),
        )

    variants: list[VariantSpec] = []
    for item in variants_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        variants.append(
            VariantSpec(
                name=name,
                sources=[str(source).strip() for source in item.get("sources") or []],
            )
        )
    return sources, variants


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(source: Path, target: Path, *, symlink: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if symlink:
        try:
            os.symlink(source.resolve(), target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _sample_paths(label_dir: Path) -> list[Path]:
    return [path for path in gesture_sample_paths(label_dir) if path.exists()]


def _copy_internal_samples(
    *,
    internal_root: Path,
    variant_root: Path,
    symlink: bool,
) -> int:
    copied = 0
    if not internal_root.exists():
        return copied
    for label_dir in sorted(path for path in internal_root.iterdir() if path.is_dir()):
        target_dir = variant_root / label_dir.name
        for sample_path in _sample_paths(label_dir):
            _link_or_copy(sample_path, target_dir / sample_path.name, symlink=symlink)
            copied += 1
    return copied


def _selected_external_paths(
    paths: list[Path],
    *,
    max_samples: int,
    seed: int,
) -> list[Path]:
    if len(paths) <= max_samples:
        return sorted(paths)
    rng = np.random.default_rng(int(seed))
    indices = sorted(rng.choice(len(paths), size=max_samples, replace=False).tolist())
    return [paths[index] for index in indices]


def _import_external_source(
    *,
    source: SourceSpec,
    variant_root: Path,
    symlink: bool,
    seed: int,
) -> ImportedExternalSummary:
    source_root = _resolve_path(source.root)
    if not source_root.exists():
        return ImportedExternalSummary(
            source=source.name,
            source_root=str(source_root),
            target_label=source.target_label,
            status="missing",
            labels={},
            imported_samples=0,
            skipped_reason="source root does not exist",
        )

    include = {label.strip().lower() for label in source.include_labels if label.strip()}
    exclude = {label.strip().lower() for label in source.exclude_labels if label.strip()}
    counts: dict[str, int] = {}
    imported = 0
    target_dir = variant_root / source.target_label
    for label_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        source_label = label_dir.name
        key = source_label.strip().lower()
        if include and key not in include:
            continue
        if key in exclude:
            continue
        if not include and not source.include_unmapped:
            continue
        paths = _selected_external_paths(
            _sample_paths(label_dir),
            max_samples=source.max_samples_per_label,
            seed=seed + len(counts),
        )
        if not paths:
            continue
        counts[source_label] = len(paths)
        for index, sample_path in enumerate(paths):
            target_name = (
                f"sample_external_{_slug(source.name)}_"
                f"{_slug(source_label)}_{index:04d}.npy"
            )
            _link_or_copy(sample_path, target_dir / target_name, symlink=symlink)
            imported += 1

    status = "ok" if imported else "empty"
    reason = "" if imported else "no sample_*.npy files selected"
    return ImportedExternalSummary(
        source=source.name,
        source_root=str(source_root),
        target_label=source.target_label,
        status=status,
        labels=dict(sorted(counts.items())),
        imported_samples=imported,
        skipped_reason=reason,
    )


def _class_counts(data_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not data_root.exists():
        return counts
    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        count = len(_sample_paths(label_dir))
        if count:
            counts[label_dir.name] = count
    return dict(sorted(counts.items()))


def materialize_variant_dataset(
    *,
    variant: VariantSpec,
    sources: dict[str, SourceSpec],
    internal_root: Path,
    work_root: Path,
    models_root: Path,
    symlink: bool = True,
    seed: int = 42,
) -> MaterializedVariantSummary:
    variant_root = work_root / variant.name / "gestures"
    model_root = models_root / variant.name
    _reset_dir(variant_root)
    model_root.mkdir(parents=True, exist_ok=True)

    internal_count = _copy_internal_samples(
        internal_root=internal_root,
        variant_root=variant_root,
        symlink=symlink,
    )
    imported_sources: list[ImportedExternalSummary] = []
    for source_name in variant.sources:
        source = sources.get(source_name)
        if source is None:
            imported_sources.append(
                ImportedExternalSummary(
                    source=source_name,
                    source_root="",
                    target_label="",
                    status="missing",
                    labels={},
                    imported_samples=0,
                    skipped_reason="source is not defined in config",
                )
            )
            continue
        imported_sources.append(
            _import_external_source(
                source=source,
                variant_root=variant_root,
                symlink=symlink,
                seed=seed,
            )
        )

    external_count = sum(item.imported_samples for item in imported_sources)
    required_sources = bool(variant.sources)
    if required_sources and external_count == 0:
        status = "skipped"
        reason = "variant requires external sources, but none were imported"
    else:
        status = "ok"
        reason = ""
    return MaterializedVariantSummary(
        variant=variant.name,
        data_root=str(variant_root),
        model_root=str(model_root),
        internal_samples=internal_count,
        external_samples=external_count,
        class_counts=_class_counts(variant_root),
        sources=imported_sources,
        status=status,
        skipped_reason=reason,
    )


def _metric_payload(report: RejectionBenchmarkReport) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for result in report.methods:
        prefix = f"{result.method}_"
        metrics[prefix + "overall_success"] = float(result.overall_success)
        metrics[prefix + "positive_recall"] = float(result.positive_recall)
        metrics[prefix + "negative_reject_rate"] = float(result.negative_reject_rate)
        metrics[prefix + "negative_false_positive_rate"] = float(
            result.negative_false_positive_rate
        )
        metrics[prefix + "accepted_accuracy"] = float(result.accepted_accuracy)
        metrics[prefix + "coverage"] = float(result.coverage)
    return metrics


def _best_metrics(report: RejectionBenchmarkReport) -> dict[str, float]:
    for result in report.methods:
        if result.method == report.best_method:
            return {
                "best_overall_success": float(result.overall_success),
                "best_positive_recall": float(result.positive_recall),
                "best_negative_reject_rate": float(result.negative_reject_rate),
                "best_negative_false_positive_rate": float(
                    result.negative_false_positive_rate
                ),
                "best_accepted_accuracy": float(result.accepted_accuracy),
                "best_coverage": float(result.coverage),
            }
    return {}


def _log_benchmark_mlflow(
    *,
    variant: MaterializedVariantSummary,
    scope: str,
    report: RejectionBenchmarkReport,
    artifact_dir: Path,
    experiment: str,
    tracking_uri: str,
    sources: dict[str, SourceSpec],
) -> None:
    experiment = str(experiment or "").strip()
    if not experiment:
        return
    try:
        import mlflow
    except Exception as exc:
        print(f"[w] MLflow unavailable for external benchmark: {exc}")
        return
    try:
        mlflow.set_tracking_uri(str(tracking_uri))
        mlflow.set_experiment(experiment)
        source_names = [
            item.source for item in variant.sources if item.imported_samples > 0
        ]
        with mlflow.start_run(run_name=f"external-negative-{variant.variant}-{scope}"):
            mlflow.set_tags(
                {
                    "run_kind": "external_negative_dataset_benchmark",
                    "source": "scripts.external_negative_dataset_experiments",
                    "variant": variant.variant,
                    "scope": scope,
                }
            )
            mlflow.log_params(
                {
                    "variant": variant.variant,
                    "scope": scope,
                    "data_root": variant.data_root,
                    "external_sources": ",".join(source_names) or "none",
                    "external_reference_urls": ",".join(
                        sources[name].reference_url
                        for name in source_names
                        if name in sources and sources[name].reference_url
                    ),
                    "best_method": report.best_method,
                    "positive_labels": ",".join(report.dataset.positive_labels),
                    "negative_labels": ",".join(report.dataset.negative_labels),
                }
            )
            mlflow.log_metrics(
                {
                    "sample_count": float(report.dataset.sample_count),
                    "class_count": float(report.dataset.class_count),
                    "external_sample_count": float(variant.external_samples),
                    **_best_metrics(report),
                }
            )
            for key, value in _metric_payload(report).items():
                mlflow.log_metric(key, value)
            if artifact_dir.exists():
                mlflow.log_artifacts(str(artifact_dir), artifact_path="external_negative")
    except Exception as exc:
        print(f"[w] MLflow external benchmark logging failed: {exc}")


def _run_scope_benchmark(
    *,
    variant: MaterializedVariantSummary,
    taxonomy_path: Path,
    scope: str,
    methods: str,
    out_root: Path,
    mlflow_experiment: str,
    mlflow_tracking_uri: str,
    sources: dict[str, SourceSpec],
    external_label_domains: dict[str, str],
) -> ScopeBenchmarkSummary:
    data_root = Path(variant.data_root)
    feature_mode = FEATURE_DYNAMIC_STATS if scope == "dynamic" else FEATURE_STATIC_MEAN
    target_dim = 44 if scope == "dynamic" else 42
    artifact_dir = out_root / variant.variant / scope
    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        include_labels = _labels_for_scope(
            data_root,
            taxonomy_path,
            scope,
            external_label_domains=external_label_domains,
        )
        report = benchmark_rejection_methods(
            data_root=data_root,
            taxonomy_path=taxonomy_path,
            scope=scope,
            feature_mode=feature_mode,
            target_dim=target_dim,
            methods=methods,
            include_labels=include_labels,
            min_samples_per_class=2,
            max_folds=3,
        )
        (artifact_dir / "benchmark.json").write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (artifact_dir / "benchmark.md").write_text(
            build_markdown_report(report),
            encoding="utf-8",
        )
        _log_benchmark_mlflow(
            variant=variant,
            scope=scope,
            report=report,
            artifact_dir=artifact_dir,
            experiment=mlflow_experiment,
            tracking_uri=mlflow_tracking_uri,
            sources=sources,
        )
        metrics = _best_metrics(report)
        return ScopeBenchmarkSummary(
            variant=variant.variant,
            scope=scope,
            status="ok",
            best_method=report.best_method,
            recommendation=report.recommendation,
            metrics=metrics,
        )
    except Exception as exc:
        (artifact_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        return ScopeBenchmarkSummary(
            variant=variant.variant,
            scope=scope,
            status="failed",
            error=str(exc),
        )


def _labels_for_scope(
    data_root: Path,
    taxonomy_path: Path,
    scope: str,
    *,
    external_label_domains: dict[str, str] | None = None,
) -> list[str]:
    taxonomy = load_gesture_taxonomy(taxonomy_path)
    external_label_domains = external_label_domains or {}
    selected_types = (
        {GESTURE_TYPE_STATIC, GESTURE_TYPE_QUASI_STATIC, GESTURE_TYPE_NEGATIVE}
        if scope == "static"
        else {GESTURE_TYPE_DYNAMIC, GESTURE_TYPE_NEGATIVE}
    )
    labels: list[str] = []
    for label in _class_counts(data_root):
        external_domain = external_label_domains.get(label)
        if external_domain and external_domain != scope:
            continue
        if taxonomy.gesture_type_for_label(label) in selected_types:
            labels.append(label)
    return labels


def _external_label_domains(
    variant: MaterializedVariantSummary,
    sources: dict[str, SourceSpec],
) -> dict[str, str]:
    domains: dict[str, str] = {}
    for imported in variant.sources:
        if imported.imported_samples <= 0:
            continue
        source = sources.get(imported.source)
        if source is None:
            continue
        domains[imported.target_label] = source.domain
    return domains


def _write_training_files(
    *,
    X: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    model_type: str,
    out_path: Path,
    classes_out: Path,
    feature_dim_out: Path,
    feature_mode_out: Path,
    rejection_out: Path,
    feature_mode: str,
) -> float:
    clf = build_classifier(model_type)
    clf.fit(X, y)
    train_accuracy = float(clf.score(X, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)
    classes_out.write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    feature_dim_out.write_text(str(int(X.shape[1])), encoding="utf-8")
    feature_mode_out.write_text(str(feature_mode), encoding="utf-8")
    rejection = build_rejection_metadata(
        X,
        y,
        classes,
        model_type=model_type,
        feature_mode=feature_mode,
    )
    rejection_out.write_text(
        json.dumps(rejection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return train_accuracy


def _train_live_artifacts(
    *,
    variant: MaterializedVariantSummary,
    taxonomy_path: Path,
    external_label_domains: dict[str, str],
    dynamic_model_types: Iterable[str],
    train_static: bool,
    train_dynamic: bool,
) -> list[TrainedArtifactSummary]:
    if variant.status != "ok":
        return []
    data_root = Path(variant.data_root)
    model_root = Path(variant.model_root)
    artifacts: list[TrainedArtifactSummary] = []

    if train_static:
        try:
            include_labels = _labels_for_scope(
                data_root,
                taxonomy_path,
                "static",
                external_label_domains=external_label_domains,
            )
            X, y, classes = load_dataset(
                data_root,
                expect_dim=42,
                include_labels=include_labels,
                lowercase_labels=True,
                feature_mode=FEATURE_STATIC_MEAN,
            )
            model_path = model_root / "knn.pkl"
            classes_path = model_root / "classes.json"
            feature_dim_path = model_root / "feature_dim.txt"
            feature_mode_path = model_root / "feature_mode.txt"
            rejection_path = model_root / "gesture_rejection.json"
            _write_training_files(
                X=X,
                y=y,
                classes=classes,
                model_type="knn",
                out_path=model_path,
                classes_out=classes_path,
                feature_dim_out=feature_dim_path,
                feature_mode_out=feature_mode_path,
                rejection_out=rejection_path,
                feature_mode=FEATURE_STATIC_MEAN,
            )
            verifier_path = model_root / "static_rejection_verifiers.pkl"
            train_static_rejection_verifiers(
                data_root=data_root,
                classes_path=classes_path,
                out_path=verifier_path,
                feature_mode=FEATURE_STATIC_MEAN,
                expect_dim=42,
                lowercase_labels=True,
            )
            artifacts.append(
                TrainedArtifactSummary(
                    variant=variant.variant,
                    scope="static",
                    status="ok",
                    model_root=str(model_root),
                    model_paths={
                        "model": str(model_path),
                        "classes": str(classes_path),
                        "feature_dim": str(feature_dim_path),
                        "feature_mode": str(feature_mode_path),
                        "rejection": str(rejection_path),
                        "verifiers": str(verifier_path),
                    },
                    classes=classes,
                    sample_count=int(X.shape[0]),
                    feature_mode=FEATURE_STATIC_MEAN,
                    feature_dim=int(X.shape[1]),
                )
            )
        except Exception as exc:
            artifacts.append(
                TrainedArtifactSummary(
                    variant=variant.variant,
                    scope="static",
                    status="failed",
                    model_root=str(model_root),
                    error=str(exc),
                )
            )

    if train_dynamic:
        for model_type in dynamic_model_types:
            try:
                include_labels = _labels_for_scope(
                    data_root,
                    taxonomy_path,
                    "dynamic",
                    external_label_domains=external_label_domains,
                )
                X, y, classes = load_dataset(
                    data_root,
                    expect_dim=44,
                    include_labels=include_labels,
                    lowercase_labels=True,
                    feature_mode=FEATURE_DYNAMIC_STATS,
                )
                model_path = model_root / f"dynamic_{model_type}.pkl"
                classes_path = model_root / "dynamic_classes.json"
                feature_dim_path = model_root / "dynamic_feature_dim.txt"
                feature_mode_path = model_root / "dynamic_feature_mode.txt"
                rejection_path = model_root / f"dynamic_{model_type}_rejection.json"
                _write_training_files(
                    X=X,
                    y=y,
                    classes=classes,
                    model_type=model_type,
                    out_path=model_path,
                    classes_out=classes_path,
                    feature_dim_out=feature_dim_path,
                    feature_mode_out=feature_mode_path,
                    rejection_out=rejection_path,
                    feature_mode=FEATURE_DYNAMIC_STATS,
                )
                artifacts.append(
                    TrainedArtifactSummary(
                        variant=variant.variant,
                        scope=f"dynamic:{model_type}",
                        status="ok",
                        model_root=str(model_root),
                        model_paths={
                            "model": str(model_path),
                            "classes": str(classes_path),
                            "feature_dim": str(feature_dim_path),
                            "feature_mode": str(feature_mode_path),
                            "rejection": str(rejection_path),
                        },
                        classes=classes,
                        sample_count=int(X.shape[0]),
                        feature_mode=FEATURE_DYNAMIC_STATS,
                        feature_dim=int(X.shape[1]),
                    )
                )
            except Exception as exc:
                artifacts.append(
                    TrainedArtifactSummary(
                        variant=variant.variant,
                        scope=f"dynamic:{model_type}",
                        status="failed",
                        model_root=str(model_root),
                        error=str(exc),
                    )
                )
    return artifacts


def _log_artifacts_mlflow(
    *,
    artifacts: list[TrainedArtifactSummary],
    mlflow_experiment: str,
    mlflow_tracking_uri: str,
) -> None:
    experiment = str(mlflow_experiment or "").strip()
    if not experiment:
        return
    try:
        import mlflow
    except Exception as exc:
        print(f"[w] MLflow unavailable for external artifact run: {exc}")
        return
    try:
        mlflow.set_tracking_uri(str(mlflow_tracking_uri))
        mlflow.set_experiment(experiment)
        for artifact in artifacts:
            with mlflow.start_run(
                run_name=f"external-negative-artifacts-{artifact.variant}-{artifact.scope}"
            ):
                mlflow.set_tags(
                    {
                        "run_kind": "external_negative_live_artifact_training",
                        "source": "scripts.external_negative_dataset_experiments",
                        "variant": artifact.variant,
                        "scope": artifact.scope,
                    }
                )
                mlflow.log_params(
                    {
                        "variant": artifact.variant,
                        "scope": artifact.scope,
                        "status": artifact.status,
                        "model_root": artifact.model_root,
                        "classes": ",".join(artifact.classes),
                        "feature_mode": artifact.feature_mode,
                    }
                )
                mlflow.log_metrics(
                    {
                        "sample_count": float(artifact.sample_count),
                        "class_count": float(len(artifact.classes)),
                        "feature_dim": float(artifact.feature_dim),
                    }
                )
                if artifact.status == "ok":
                    for path in artifact.model_paths.values():
                        candidate = Path(path)
                        if candidate.exists():
                            mlflow.log_artifact(str(candidate), artifact_path="models")
                elif artifact.error:
                    mlflow.log_text(artifact.error, "error.txt")
    except Exception as exc:
        print(f"[w] MLflow external artifact logging failed: {exc}")


def run_external_negative_experiments(
    *,
    config_path: Path = DEFAULT_CONFIG,
    internal_data_root: Path = DEFAULT_DATA_ROOT,
    work_root: Path = DEFAULT_WORK_ROOT,
    models_root: Path = DEFAULT_MODELS_ROOT,
    taxonomy_path: Path = PROJECT_ROOT / "configs" / "gesture_taxonomy.json",
    scopes: Iterable[str] = ("static", "dynamic"),
    methods: str = ",".join(DEFAULT_METHODS),
    dynamic_model_types: Iterable[str] = ("knn",),
    symlink: bool = True,
    seed: int = 42,
    mlflow_experiment: str = "GestureBind",
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
    train_artifacts: bool = True,
) -> ExternalNegativeExperimentReport:
    sources, variant_specs = load_specs(config_path)
    variants: list[MaterializedVariantSummary] = []
    benchmarks: list[ScopeBenchmarkSummary] = []
    artifacts: list[TrainedArtifactSummary] = []
    notes: list[str] = []

    for variant_spec in variant_specs:
        variant = materialize_variant_dataset(
            variant=variant_spec,
            sources=sources,
            internal_root=internal_data_root,
            work_root=work_root,
            models_root=models_root,
            symlink=symlink,
            seed=seed,
        )
        variants.append(variant)
        if variant.status != "ok":
            notes.append(f"{variant.variant}: skipped - {variant.skipped_reason}")
            continue
        external_domains = _external_label_domains(variant, sources)
        for scope in scopes:
            clean_scope = str(scope).strip().lower()
            if clean_scope not in {"static", "dynamic"}:
                continue
            benchmarks.append(
                _run_scope_benchmark(
                    variant=variant,
                    taxonomy_path=taxonomy_path,
                    scope=clean_scope,
                    methods=methods,
                    out_root=work_root / "reports",
                    mlflow_experiment=mlflow_experiment,
                    mlflow_tracking_uri=mlflow_tracking_uri,
                    sources=sources,
                    external_label_domains=external_domains,
                )
            )
        if train_artifacts:
            artifacts.extend(
                _train_live_artifacts(
                    variant=variant,
                    taxonomy_path=taxonomy_path,
                    external_label_domains=external_domains,
                    dynamic_model_types=dynamic_model_types,
                    train_static=True,
                    train_dynamic=True,
                )
            )

    if artifacts:
        _log_artifacts_mlflow(
            artifacts=artifacts,
            mlflow_experiment=mlflow_experiment,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

    notes.extend(
        [
            "External datasets are treated as negative/rejection data, not as replacement for personalized GestureBind samples.",
            "Live validation still decides whether an external variant is useful for the user's camera and gesture style.",
        ]
    )
    return ExternalNegativeExperimentReport(
        generated_at=time.time(),
        config_path=str(config_path),
        internal_data_root=str(internal_data_root),
        variants=variants,
        benchmarks=benchmarks,
        artifacts=artifacts,
        notes=notes,
    )


def _format_metric(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def build_markdown_summary(report: ExternalNegativeExperimentReport) -> str:
    lines = [
        "# External Negative Dataset Experiments",
        "",
        f"Generated at: `{report.generated_at:.3f}`",
        "",
        "## Goal",
        "",
        "Проверить, улучшают ли публичные датасеты качество reject-layer без",
        "подмены персонального датасета GestureBind. Внешние данные используются",
        "как negative / out-of-distribution evidence.",
        "",
        "## Variants",
        "",
        "| Variant | Status | Internal samples | External samples | Sources | Model root |",
        "|---|---|---:|---:|---|---|",
    ]
    for variant in report.variants:
        source_names = ", ".join(
            f"{item.source}:{item.status}:{item.imported_samples}"
            for item in variant.sources
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{variant.variant}`",
                    variant.status,
                    str(variant.internal_samples),
                    str(variant.external_samples),
                    source_names or "none",
                    f"`{variant.model_root}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Offline Benchmarks",
            "",
            "| Variant | Scope | Status | Best method | Overall | Pos recall | Neg reject | Neg FP |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in report.benchmarks:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.variant}`",
                    item.scope,
                    item.status,
                    f"`{item.best_method}`" if item.best_method else "",
                    _format_metric(item.metrics.get("best_overall_success")),
                    _format_metric(item.metrics.get("best_positive_recall")),
                    _format_metric(item.metrics.get("best_negative_reject_rate")),
                    _format_metric(item.metrics.get("best_negative_false_positive_rate")),
                ]
            )
            + " |"
        )
        if item.error:
            lines.append(f"- `{item.variant}` / `{item.scope}` error: `{item.error}`")

    lines.extend(
        [
            "",
            "## Live Artifacts",
            "",
            "| Variant | Scope | Status | Samples | Classes | Feature mode | Model root |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for item in report.artifacts:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.variant}`",
                    f"`{item.scope}`",
                    item.status,
                    str(item.sample_count),
                    str(len(item.classes)),
                    item.feature_mode,
                    f"`{item.model_root}`",
                ]
            )
            + " |"
        )
        if item.error:
            lines.append(f"- `{item.variant}` / `{item.scope}` training error: `{item.error}`")

    lines.extend(
        [
            "",
            "## How To Live-Test A Variant",
            "",
            "Example for current `ipn_external` dynamic variant:",
            "",
            "```bash",
            "DPLM_MODELS_DIR=models/experiments/external_negative/ipn_external \\",
            "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m app.flet_app.main",
            "```",
            "",
            "Then use Live Evaluation exactly like the existing rejection protocol.",
            "",
            "## Notes",
            "",
        ]
    )
    for note in report.notes:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _write_report(
    report: ExternalNegativeExperimentReport,
    *,
    json_out: Path,
    md_out: Path,
) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_out.write_text(build_markdown_summary(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=PROJECT_ROOT / "configs" / "gesture_taxonomy.json",
    )
    parser.add_argument("--scopes", default="static,dynamic")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--dynamic-model-types", default="knn")
    parser.add_argument("--copy", action="store_true", help="Copy samples instead of symlinking")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_DOC_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_DOC_MD)
    parser.add_argument("--mlflow-experiment", default="GestureBind")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--skip-artifact-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_external_negative_experiments(
        config_path=args.config,
        internal_data_root=args.data_root,
        work_root=args.work_root,
        models_root=args.models_root,
        taxonomy_path=args.taxonomy,
        scopes=[item.strip() for item in str(args.scopes).split(",") if item.strip()],
        methods=str(args.methods),
        dynamic_model_types=[
            item.strip()
            for item in str(args.dynamic_model_types).split(",")
            if item.strip()
        ],
        symlink=not bool(args.copy),
        seed=int(args.seed),
        mlflow_experiment=str(args.mlflow_experiment),
        mlflow_tracking_uri=str(args.mlflow_tracking_uri),
        train_artifacts=not bool(args.skip_artifact_training),
    )
    _write_report(report, json_out=args.json_out, md_out=args.md_out)
    print(build_markdown_summary(report))
    print(f"[✓] JSON: {args.json_out}")
    print(f"[✓] Markdown: {args.md_out}")


if __name__ == "__main__":
    main()
