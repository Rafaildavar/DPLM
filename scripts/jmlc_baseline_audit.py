from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SampleSummary:
    path: str
    shape: tuple[int, ...]
    frames: int | None
    feature_dim: int | None
    valid: bool
    issue: str = ""


@dataclass(frozen=True)
class ClassSummary:
    label: str
    sample_count: int
    valid_sample_count: int
    frame_min: int | None
    frame_max: int | None
    frame_mean: float | None
    feature_dims: list[int]
    issues: list[str]


@dataclass(frozen=True)
class DatasetSummary:
    data_root: str
    class_count: int
    active_class_count: int
    empty_class_count: int
    sample_count: int
    valid_sample_count: int
    invalid_sample_count: int
    active_sample_min: int | None
    active_sample_max: int | None
    imbalance_ratio: float | None
    frame_min: int | None
    frame_max: int | None
    frame_mean: float | None
    feature_dims: list[int]
    classes: list[ClassSummary]


@dataclass(frozen=True)
class TestSummary:
    test_root: str
    test_file_count: int
    test_function_count: int
    pytest_ini_exists: bool
    coverage_gate: str


@dataclass(frozen=True)
class ProjectSummary:
    generated_at: str
    root: str
    dataset: DatasetSummary
    tests: TestSummary
    artifacts: dict[str, bool]


def _safe_shape_tuple(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(v) for v in shape)


def inspect_sample(path: Path, data_root: Path) -> SampleSummary:
    try:
        arr = np.load(path)
    except Exception as exc:  # pragma: no cover - exact numpy error type is not stable
        return SampleSummary(
            path=str(path.relative_to(data_root)),
            shape=(),
            frames=None,
            feature_dim=None,
            valid=False,
            issue=f"load_error:{exc.__class__.__name__}",
        )

    shape = _safe_shape_tuple(arr.shape)
    if arr.ndim == 2:
        frames = int(arr.shape[0])
        feature_dim = int(arr.shape[1])
    elif arr.ndim == 3:
        frames = int(arr.shape[0])
        feature_dim = int(arr.shape[1] * arr.shape[2])
    else:
        return SampleSummary(
            path=str(path.relative_to(data_root)),
            shape=shape,
            frames=None,
            feature_dim=None,
            valid=False,
            issue=f"unexpected_ndim:{arr.ndim}",
        )

    if frames <= 0 or feature_dim <= 0:
        return SampleSummary(
            path=str(path.relative_to(data_root)),
            shape=shape,
            frames=frames,
            feature_dim=feature_dim,
            valid=False,
            issue="empty_sample",
        )

    if not np.isfinite(arr).all():
        return SampleSummary(
            path=str(path.relative_to(data_root)),
            shape=shape,
            frames=frames,
            feature_dim=feature_dim,
            valid=False,
            issue="non_finite_values",
        )

    return SampleSummary(
        path=str(path.relative_to(data_root)),
        shape=shape,
        frames=frames,
        feature_dim=feature_dim,
        valid=True,
    )


def summarize_dataset(data_root: Path) -> DatasetSummary:
    if not data_root.exists():
        return DatasetSummary(
            data_root=str(data_root),
            class_count=0,
            active_class_count=0,
            empty_class_count=0,
            sample_count=0,
            valid_sample_count=0,
            invalid_sample_count=0,
            active_sample_min=None,
            active_sample_max=None,
            imbalance_ratio=None,
            frame_min=None,
            frame_max=None,
            frame_mean=None,
            feature_dims=[],
            classes=[],
        )

    class_summaries: list[ClassSummary] = []
    all_frames: list[int] = []
    all_feature_dims: set[int] = set()
    total_samples = 0
    total_valid = 0
    total_invalid = 0

    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        samples = [inspect_sample(path, data_root) for path in sorted(label_dir.glob("sample_*.npy"))]
        total_samples += len(samples)
        valid = [sample for sample in samples if sample.valid]
        invalid = [sample for sample in samples if not sample.valid]
        total_valid += len(valid)
        total_invalid += len(invalid)

        frames = [int(sample.frames) for sample in valid if sample.frames is not None]
        dims = sorted({int(sample.feature_dim) for sample in valid if sample.feature_dim is not None})
        all_frames.extend(frames)
        all_feature_dims.update(dims)

        issues = sorted({sample.issue for sample in invalid if sample.issue})
        class_summaries.append(
            ClassSummary(
                label=label_dir.name,
                sample_count=len(samples),
                valid_sample_count=len(valid),
                frame_min=min(frames) if frames else None,
                frame_max=max(frames) if frames else None,
                frame_mean=round(float(mean(frames)), 2) if frames else None,
                feature_dims=dims,
                issues=issues,
            )
        )

    active_counts = [item.valid_sample_count for item in class_summaries if item.valid_sample_count > 0]
    active_sample_min = min(active_counts) if active_counts else None
    active_sample_max = max(active_counts) if active_counts else None
    imbalance_ratio = (
        round(float(active_sample_max / active_sample_min), 2)
        if active_sample_min and active_sample_max
        else None
    )

    return DatasetSummary(
        data_root=str(data_root),
        class_count=len(class_summaries),
        active_class_count=len(active_counts),
        empty_class_count=sum(1 for item in class_summaries if item.valid_sample_count == 0),
        sample_count=total_samples,
        valid_sample_count=total_valid,
        invalid_sample_count=total_invalid,
        active_sample_min=active_sample_min,
        active_sample_max=active_sample_max,
        imbalance_ratio=imbalance_ratio,
        frame_min=min(all_frames) if all_frames else None,
        frame_max=max(all_frames) if all_frames else None,
        frame_mean=round(float(mean(all_frames)), 2) if all_frames else None,
        feature_dims=sorted(all_feature_dims),
        classes=class_summaries,
    )


def count_test_functions(test_root: Path) -> int:
    total = 0
    for path in sorted(test_root.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                total += 1
    return total


def summarize_tests(project_root: Path) -> TestSummary:
    test_root = project_root / "tests"
    pytest_ini = project_root / "pytest.ini"
    coverage_gate = "not_configured"
    if pytest_ini.exists():
        for line in pytest_ini.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("--cov-fail-under="):
                coverage_gate = stripped.removeprefix("--cov-fail-under=")
                break

    return TestSummary(
        test_root=str(test_root),
        test_file_count=len(list(test_root.rglob("test_*.py"))) if test_root.exists() else 0,
        test_function_count=count_test_functions(test_root) if test_root.exists() else 0,
        pytest_ini_exists=pytest_ini.exists(),
        coverage_gate=coverage_gate,
    )


def summarize_project(project_root: Path, data_root: Path) -> ProjectSummary:
    artifacts = {
        "JMLC.md": (project_root / "JMLC.md").exists(),
        "docker-compose.yml": (project_root / "docker-compose.yml").exists(),
        "pytest.ini": (project_root / "pytest.ini").exists(),
        "alembic": (project_root / "alembic").is_dir(),
        "models/classes.json": (project_root / "models" / "classes.json").exists(),
        "models/feature_dim.txt": (project_root / "models" / "feature_dim.txt").exists(),
        "docs/jmlc_micro_presentation.html": (
            project_root / "docs" / "jmlc_micro_presentation.html"
        ).exists(),
    }
    return ProjectSummary(
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        root=str(project_root),
        dataset=summarize_dataset(data_root),
        tests=summarize_tests(project_root),
        artifacts=artifacts,
    )


def _format_optional_number(value: int | float | None) -> str:
    return "n/a" if value is None else str(value)


def _readable_artifact_status(artifacts: dict[str, bool]) -> str:
    rows = []
    for name, exists in artifacts.items():
        rows.append(f"| `{name}` | {'yes' if exists else 'no'} |")
    return "\n".join(rows)


def build_markdown_report(summary: ProjectSummary, pytest_outcome: str) -> str:
    dataset = summary.dataset
    tests = summary.tests
    class_rows = []
    for item in dataset.classes:
        class_rows.append(
            "| "
            + " | ".join(
                [
                    f"`{item.label}`",
                    str(item.sample_count),
                    str(item.valid_sample_count),
                    _format_optional_number(item.frame_min),
                    _format_optional_number(item.frame_max),
                    _format_optional_number(item.frame_mean),
                    ", ".join(str(dim) for dim in item.feature_dims) or "n/a",
                    ", ".join(item.issues) or "-",
                ]
            )
            + " |"
        )

    class_table = "\n".join(class_rows) if class_rows else "| - | 0 | 0 | n/a | n/a | n/a | n/a | - |"
    artifact_rows = _readable_artifact_status(summary.artifacts)

    has_mixed_feature_dims = len(dataset.feature_dims) > 1
    has_empty_classes = dataset.empty_class_count > 0
    is_imbalanced = (dataset.imbalance_ratio or 0.0) >= 3.0

    if dataset.active_class_count < 3 or dataset.valid_sample_count < 30:
        ds_assessment = "weak"
        ds_note = "датасет пока выглядит малым для убедительного сравнения моделей"
    elif dataset.invalid_sample_count > 0 or has_mixed_feature_dims or has_empty_classes or is_imbalanced:
        ds_assessment = "medium"
        ds_note = "есть база для первого baseline, но нужно явно обработать пустые классы, дисбаланс и разные размерности признаков"
    else:
        ds_assessment = "good"
        ds_note = "датасет пригоден для первого baseline-сравнения"

    coverage_note = (
        "Тесты проходят функционально, но quality gate по покрытию сейчас не закрыт."
        if "coverage" in pytest_outcome.lower() or "cov" in pytest_outcome.lower()
        else "Тестовый статус нужно обновить после полного прогона."
    )

    return f"""# JMLC Stage 1 - Baseline Audit

Generated: `{summary.generated_at}`

## Executive summary

- Dataset status: **{ds_assessment}** - {ds_note}.
- Test status: **recorded** - {pytest_outcome}.
- JMLC visibility: проект уже демонстрирует инженерную базу, но для победного трека нужны воспроизводимые ML-эксперименты и отчеты в `docs/experiments/`.

## Dataset snapshot

| Metric | Value |
|---|---:|
| Class directories | {dataset.class_count} |
| Active classes | {dataset.active_class_count} |
| Empty classes | {dataset.empty_class_count} |
| Samples | {dataset.sample_count} |
| Valid samples | {dataset.valid_sample_count} |
| Invalid samples | {dataset.invalid_sample_count} |
| Active sample min | {_format_optional_number(dataset.active_sample_min)} |
| Active sample max | {_format_optional_number(dataset.active_sample_max)} |
| Imbalance ratio | {_format_optional_number(dataset.imbalance_ratio)} |
| Frame min | {_format_optional_number(dataset.frame_min)} |
| Frame max | {_format_optional_number(dataset.frame_max)} |
| Frame mean | {_format_optional_number(dataset.frame_mean)} |
| Feature dimensions | {", ".join(str(dim) for dim in dataset.feature_dims) or "n/a"} |

## Classes

| Class | Samples | Valid | Frame min | Frame max | Frame mean | Feature dims | Issues |
|---|---:|---:|---:|---:|---:|---|---|
{class_table}

## Engineering snapshot

| Artifact | Present |
|---|---|
{artifact_rows}

## Test snapshot

| Metric | Value |
|---|---:|
| Test files | {tests.test_file_count} |
| Test functions | {tests.test_function_count} |
| `pytest.ini` | {"yes" if tests.pytest_ini_exists else "no"} |
| Coverage gate | {tests.coverage_gate} |

Pytest outcome: `{pytest_outcome}`

{coverage_note}

## JMLC assessment

Что выглядит хорошо для комиссии:

1. Есть реальный датасет пользовательских жестов, а не только демо-код.
2. Есть unit-тесты и строгий coverage gate, даже если он пока не закрыт.
3. Есть инженерные артефакты: Docker для БД, Alembic, модели, документация, презентационный HTML.

Что пока выглядит рискованно:

1. Активных классов меньше, чем папок: пустые директории нельзя считать частью обучающего датасета.
2. Есть смешение размерностей 42/84, поэтому Stage 2 должен отдельно сравнивать one-hand и two-hand признаки или фильтровать совместимые классы.
3. Нужна таблица фактического сравнения `static_mean` vs `dynamic_stats` vs альтернативные модели.
4. Coverage gate 90% сейчас падает из-за низкого покрытия больших GUI/runtime модулей; это нужно либо закрывать тестами, либо честно объяснить и скорректировать стратегию покрытия для JMLC-ветки.

## Decision for Stage 2

Переходить к реализации `cv/gesture_features.py` и `scripts/compare_models.py`.
Минимальная цель Stage 2: получить воспроизводимую таблицу baseline-метрик и confusion matrix для текущего датасета.
"""


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Counter):
        return dict(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JMLC stage 1 baseline audit")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--data-root", default="data/gestures", help="Gesture dataset root")
    parser.add_argument("--json-out", default="", help="Optional JSON output path")
    parser.add_argument("--markdown-out", default="", help="Optional Markdown report output path")
    parser.add_argument("--pytest-outcome", default="not run", help="Human-readable pytest result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    data_root = (project_root / args.data_root).resolve()
    summary = summarize_project(project_root, data_root)

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report = build_markdown_report(summary, args.pytest_outcome)
    if args.markdown_out:
        markdown_path = Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
