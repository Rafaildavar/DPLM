"""Build a reproducible JMLC dataset profile for gesture recordings.

The profiler intentionally does not train a model. It answers the first Data
Science question for JMLC: what data do we actually have, what is its shape, and
which risks must be handled before comparing models.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SampleProfile:
    path: str
    shape: tuple[int, ...]
    frames: int | None
    feature_dim: int | None
    valid: bool
    issue: str = ""


@dataclass(frozen=True)
class ClassProfile:
    label: str
    sample_count: int
    valid_sample_count: int
    invalid_sample_count: int
    frame_min: int | None
    frame_max: int | None
    frame_mean: float | None
    feature_dims: list[int]
    issues: list[str]


@dataclass(frozen=True)
class DatasetProfile:
    generated_at: str
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
    classes: list[ClassProfile]
    risks: list[str]


def _shape_tuple(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(value) for value in shape)


def inspect_sample(path: Path, data_root: Path) -> SampleProfile:
    try:
        arr = np.load(path)
    except Exception as exc:  # pragma: no cover - numpy error classes vary
        return SampleProfile(
            path=str(path.relative_to(data_root)),
            shape=(),
            frames=None,
            feature_dim=None,
            valid=False,
            issue=f"load_error:{exc.__class__.__name__}",
        )

    shape = _shape_tuple(arr.shape)
    if arr.ndim == 2:
        frames = int(arr.shape[0])
        feature_dim = int(arr.shape[1])
    elif arr.ndim == 3:
        frames = int(arr.shape[0])
        feature_dim = int(arr.shape[1] * arr.shape[2])
    else:
        return SampleProfile(
            path=str(path.relative_to(data_root)),
            shape=shape,
            frames=None,
            feature_dim=None,
            valid=False,
            issue=f"unexpected_ndim:{arr.ndim}",
        )

    if frames <= 0 or feature_dim <= 0:
        return SampleProfile(
            path=str(path.relative_to(data_root)),
            shape=shape,
            frames=frames,
            feature_dim=feature_dim,
            valid=False,
            issue="empty_sample",
        )

    if not np.isfinite(arr).all():
        return SampleProfile(
            path=str(path.relative_to(data_root)),
            shape=shape,
            frames=frames,
            feature_dim=feature_dim,
            valid=False,
            issue="non_finite_values",
        )

    return SampleProfile(
        path=str(path.relative_to(data_root)),
        shape=shape,
        frames=frames,
        feature_dim=feature_dim,
        valid=True,
    )


def _class_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    return sorted(path for path in data_root.iterdir() if path.is_dir())


def build_profile(data_root: Path) -> DatasetProfile:
    class_profiles: list[ClassProfile] = []
    all_frames: list[int] = []
    all_dims: set[int] = set()
    total_samples = 0
    total_valid = 0
    total_invalid = 0

    for label_dir in _class_dirs(data_root):
        samples = [
            inspect_sample(path, data_root)
            for path in sorted(label_dir.glob("sample_*.npy"))
        ]
        valid = [sample for sample in samples if sample.valid]
        invalid = [sample for sample in samples if not sample.valid]
        frames = [int(sample.frames) for sample in valid if sample.frames is not None]
        dims = sorted({int(sample.feature_dim) for sample in valid if sample.feature_dim})
        issues = sorted({sample.issue for sample in invalid if sample.issue})

        total_samples += len(samples)
        total_valid += len(valid)
        total_invalid += len(invalid)
        all_frames.extend(frames)
        all_dims.update(dims)

        class_profiles.append(
            ClassProfile(
                label=label_dir.name,
                sample_count=len(samples),
                valid_sample_count=len(valid),
                invalid_sample_count=len(invalid),
                frame_min=min(frames) if frames else None,
                frame_max=max(frames) if frames else None,
                frame_mean=round(float(mean(frames)), 2) if frames else None,
                feature_dims=dims,
                issues=issues,
            )
        )

    active_counts = [
        item.valid_sample_count
        for item in class_profiles
        if item.valid_sample_count > 0
    ]
    active_min = min(active_counts) if active_counts else None
    active_max = max(active_counts) if active_counts else None
    imbalance_ratio = (
        round(float(active_max / active_min), 2)
        if active_min and active_max
        else None
    )

    risks = detect_risks(
        class_profiles=class_profiles,
        feature_dims=sorted(all_dims),
        imbalance_ratio=imbalance_ratio,
        invalid_sample_count=total_invalid,
    )

    return DatasetProfile(
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        data_root=str(data_root),
        class_count=len(class_profiles),
        active_class_count=len(active_counts),
        empty_class_count=sum(1 for item in class_profiles if item.valid_sample_count == 0),
        sample_count=total_samples,
        valid_sample_count=total_valid,
        invalid_sample_count=total_invalid,
        active_sample_min=active_min,
        active_sample_max=active_max,
        imbalance_ratio=imbalance_ratio,
        frame_min=min(all_frames) if all_frames else None,
        frame_max=max(all_frames) if all_frames else None,
        frame_mean=round(float(mean(all_frames)), 2) if all_frames else None,
        feature_dims=sorted(all_dims),
        classes=class_profiles,
        risks=risks,
    )


def detect_risks(
    *,
    class_profiles: list[ClassProfile],
    feature_dims: list[int],
    imbalance_ratio: float | None,
    invalid_sample_count: int,
) -> list[str]:
    risks: list[str] = []
    active = [item for item in class_profiles if item.valid_sample_count > 0]
    low_sample = [item.label for item in active if item.valid_sample_count < 25]
    empty = [item.label for item in class_profiles if item.valid_sample_count == 0]

    if len(active) < 5:
        risks.append("active_class_count_below_target")
    if low_sample:
        risks.append("low_samples_per_class:" + ",".join(low_sample))
    if empty:
        risks.append("empty_classes:" + ",".join(empty))
    if len(feature_dims) > 1:
        risks.append("mixed_feature_dimensions:" + ",".join(str(dim) for dim in feature_dims))
    if imbalance_ratio is not None and imbalance_ratio >= 2.5:
        risks.append(f"class_imbalance:{imbalance_ratio}")
    if invalid_sample_count:
        risks.append(f"invalid_samples:{invalid_sample_count}")
    return risks


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


def _fmt(value: int | float | None) -> str:
    return "n/a" if value is None else str(value)


def _class_table(profile: DatasetProfile) -> str:
    if not profile.classes:
        return "| - | 0 | 0 | 0 | n/a | n/a | n/a | n/a | - |"

    rows: list[str] = []
    for item in profile.classes:
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{item.label}`",
                    str(item.sample_count),
                    str(item.valid_sample_count),
                    str(item.invalid_sample_count),
                    _fmt(item.frame_min),
                    _fmt(item.frame_max),
                    _fmt(item.frame_mean),
                    ", ".join(str(dim) for dim in item.feature_dims) or "n/a",
                    ", ".join(item.issues) or "-",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def build_markdown(profile: DatasetProfile) -> str:
    risks = "\n".join(f"- `{risk}`" for risk in profile.risks) or "- нет явных рисков"
    class_table = _class_table(profile)

    return f"""# JMLC Dataset Profile

Generated: `{profile.generated_at}`

## Summary

| Metric | Value |
|---|---:|
| Class directories | {profile.class_count} |
| Active classes | {profile.active_class_count} |
| Empty classes | {profile.empty_class_count} |
| Samples | {profile.sample_count} |
| Valid samples | {profile.valid_sample_count} |
| Invalid samples | {profile.invalid_sample_count} |
| Active sample min | {_fmt(profile.active_sample_min)} |
| Active sample max | {_fmt(profile.active_sample_max)} |
| Imbalance ratio | {_fmt(profile.imbalance_ratio)} |
| Frame min | {_fmt(profile.frame_min)} |
| Frame max | {_fmt(profile.frame_max)} |
| Frame mean | {_fmt(profile.frame_mean)} |
| Feature dimensions | {", ".join(str(dim) for dim in profile.feature_dims) or "n/a"} |

## Classes

| Class | Samples | Valid | Invalid | Frame min | Frame max | Frame mean | Feature dims | Issues |
|---|---:|---:|---:|---:|---:|---:|---|---|
{class_table}

## Risks

{risks}

## JMLC Interpretation

- Основная offline-метрика должна быть `macro_f1`, потому что пользовательские
  классы могут быть несбалансированы.
- Если есть `mixed_feature_dimensions`, one-hand и two-hand сценарии нужно
  сравнивать отдельно или явно выравнивать признаки.
- Классы с малым числом семплов нельзя использовать как сильное доказательство
  качества модели без дополнительного сбора данных.
"""


def write_outputs(profile: DatasetProfile, json_out: Path, markdown_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(to_jsonable(profile), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_out.write_text(build_markdown(profile), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JMLC gesture dataset profile")
    parser.add_argument("--data-root", default="data/gestures", type=Path)
    parser.add_argument(
        "--json-out",
        default="docs/experiments/dataset_profile.json",
        type=Path,
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/experiments/dataset_profile.md",
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = build_profile(args.data_root)
    write_outputs(profile, json_out=args.json_out, markdown_out=args.markdown_out)
    print(f"[OK] Wrote {args.json_out} and {args.markdown_out}")


if __name__ == "__main__":
    main()

