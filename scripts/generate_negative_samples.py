"""Generate synthetic negative samples from existing gesture data."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from app.services.gesture_taxonomy import (
    GESTURE_TYPE_DYNAMIC,
    GESTURE_TYPE_NEGATIVE,
    GESTURE_TYPE_QUASI_STATIC,
    GESTURE_TYPE_STATIC,
    load_gesture_taxonomy,
)
from cv.dynamic_motion import resample_sequence
from cv.gesture_features import sequence_to_matrix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEGATIVE_LABELS = (
    "no_gesture_static",
    "random_motion",
    "partial_swipe",
    "return_motion",
    "wrong_axis_motion",
)
STATIC_SOURCE_TYPES = {GESTURE_TYPE_STATIC, GESTURE_TYPE_QUASI_STATIC}
SourceSample = tuple[str, Path, np.ndarray, str]


@dataclass(frozen=True)
class GeneratedNegativeSample:
    label: str
    path: str
    source_label: str
    source_path: str
    source_scope: str
    scenario: str
    frames: int
    raw_feature_dim: int


@dataclass(frozen=True)
class NegativeSamplingReport:
    generated_at: float
    data_root: str
    target_frames: int
    samples_per_label: int
    seed: int
    source_samples: int
    generated_samples: int
    labels: dict[str, int]
    samples: list[GeneratedNegativeSample]


def load_dynamic_source_samples(
    data_root: Path,
    *,
    taxonomy_path: Path,
) -> list[SourceSample]:
    taxonomy = load_gesture_taxonomy(taxonomy_path)
    samples: list[SourceSample] = []
    if not data_root.exists():
        return samples
    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        label = label_dir.name
        if taxonomy.gesture_type_for_label(label) != GESTURE_TYPE_DYNAMIC:
            continue
        for path in sorted(label_dir.glob("sample_*.npy")):
            try:
                sequence = _as_dynamic_frame_matrix(np.load(path))
            except Exception:
                continue
            samples.append((label, path, sequence, GESTURE_TYPE_DYNAMIC))
    return samples


def load_static_source_samples(
    data_root: Path,
    *,
    taxonomy_path: Path,
) -> list[SourceSample]:
    taxonomy = load_gesture_taxonomy(taxonomy_path)
    samples: list[SourceSample] = []
    if not data_root.exists():
        return samples
    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        label = label_dir.name
        gesture_type = taxonomy.gesture_type_for_label(label)
        if gesture_type not in STATIC_SOURCE_TYPES:
            continue
        for path in sorted(label_dir.glob("sample_*.npy")):
            try:
                sequence = _as_static_frame_matrix(np.load(path))
            except Exception:
                continue
            samples.append((label, path, sequence, gesture_type))
    return samples


def generate_negative_samples(
    *,
    data_root: Path = ROOT / "data" / "gestures",
    taxonomy_path: Path = ROOT / "configs" / "gesture_taxonomy.json",
    labels: Iterable[str] = DEFAULT_NEGATIVE_LABELS,
    samples_per_label: int = 20,
    target_frames: int = 36,
    seed: int = 42,
    clean_auto: bool = True,
    manifest_out: Path | None = ROOT / "docs" / "experiments" / "negative_sampling_manifest.json",
) -> NegativeSamplingReport:
    taxonomy = load_gesture_taxonomy(taxonomy_path)
    negative_labels = [
        str(label).strip()
        for label in labels
        if str(label).strip()
        and taxonomy.gesture_type_for_label(str(label).strip())
        == GESTURE_TYPE_NEGATIVE
    ]
    if not negative_labels:
        raise RuntimeError("no negative labels selected")

    dynamic_source_samples = load_dynamic_source_samples(
        data_root,
        taxonomy_path=taxonomy_path,
    )
    static_source_samples = load_static_source_samples(
        data_root,
        taxonomy_path=taxonomy_path,
    )
    source_samples = dynamic_source_samples + static_source_samples
    if not source_samples:
        raise RuntimeError("no source samples found for negative generation")

    rng = np.random.default_rng(int(seed))
    target = max(2, int(target_frames))
    per_label = max(1, int(samples_per_label))
    generated: list[GeneratedNegativeSample] = []
    counts: dict[str, int] = {}

    for label in negative_labels:
        out_dir = data_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        if clean_auto:
            for old_path in out_dir.glob("sample_auto_*.npy"):
                old_path.unlink(missing_ok=True)
                old_path.with_suffix(".meta.json").unlink(missing_ok=True)

        for index in range(per_label):
            scenario = _scenario_for_label(label)
            source_pool = _source_pool_for_scenario(
                scenario,
                dynamic_source_samples=dynamic_source_samples,
                static_source_samples=static_source_samples,
            )
            source_index = int(rng.integers(0, len(source_pool)))
            source_label, source_path, source_sequence, source_scope = source_pool[
                source_index
            ]
            generation_scenario = scenario
            if scenario == "static_hold" and source_scope in STATIC_SOURCE_TYPES:
                generation_scenario = _static_scenario_for_index(index)
                alternate = _alternate_static_source(
                    source_pool,
                    source_label=source_label,
                    index=source_index + index,
                )
                sequence = _static_negative_sequence(
                    source_sequence,
                    scenario=generation_scenario,
                    target_frames=target,
                    rng=rng,
                    alternate=alternate,
                )
            else:
                sequence = _negative_sequence(
                    source_sequence,
                    scenario=scenario,
                    target_frames=target,
                    rng=rng,
                )
            out_path = out_dir / f"sample_auto_{index:04d}.npy"
            np.save(out_path, sequence.astype(np.float32, copy=False))
            metadata = {
                "schema_version": 1,
                "generated_by": "scripts.generate_negative_samples",
                "label": label,
                "scenario": generation_scenario,
                "source_label": source_label,
                "source_path": _portable_path(source_path),
                "source_scope": source_scope,
                "frames": int(sequence.shape[0]),
                "raw_feature_dim": int(sequence.shape[1]),
                "seed": int(seed),
                "generated_at": time.time(),
            }
            out_path.with_suffix(".meta.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            generated.append(
                GeneratedNegativeSample(
                    label=label,
                    path=str(out_path),
                    source_label=source_label,
                    source_path=str(source_path),
                    source_scope=source_scope,
                    scenario=generation_scenario,
                    frames=int(sequence.shape[0]),
                    raw_feature_dim=int(sequence.shape[1]),
                )
            )
            counts[label] = counts.get(label, 0) + 1

    report = NegativeSamplingReport(
        generated_at=time.time(),
        data_root=str(data_root),
        target_frames=target,
        samples_per_label=per_label,
        seed=int(seed),
        source_samples=len(source_samples),
        generated_samples=len(generated),
        labels=dict(sorted(counts.items())),
        samples=generated,
    )
    if manifest_out is not None:
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(
            json.dumps(_report_to_dict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def _as_dynamic_frame_matrix(raw: np.ndarray) -> np.ndarray:
    seq = sequence_to_matrix(raw)
    if seq.shape[1] >= 44:
        return seq[:, :44].astype(np.float32, copy=False)
    pose = seq[:, :42] if seq.shape[1] >= 42 else _pad_columns(seq, 42)
    wrist = np.tile(np.asarray([[0.5, 0.5]], dtype=np.float32), (pose.shape[0], 1))
    return np.concatenate([pose[:, :42], wrist], axis=1).astype(np.float32, copy=False)


def _as_static_frame_matrix(raw: np.ndarray) -> np.ndarray:
    seq = sequence_to_matrix(raw)
    return _pad_columns(seq, 42).astype(np.float32, copy=False)


def _pad_columns(sequence: np.ndarray, width: int) -> np.ndarray:
    if sequence.shape[1] >= width:
        return sequence[:, :width]
    pad = np.zeros((sequence.shape[0], width - sequence.shape[1]), dtype=np.float32)
    return np.concatenate([sequence, pad], axis=1)


def _scenario_for_label(label: str) -> str:
    clean = str(label or "").strip().lower()
    if clean.startswith("no_gesture"):
        return "static_hold"
    if clean.startswith("random"):
        return "closed_random_walk"
    if clean.startswith("partial"):
        return "aborted_partial_motion"
    if clean.startswith("return"):
        return "out_and_back_return"
    if clean.startswith("wrong_axis"):
        return "ambiguous_diagonal"
    return "closed_random_walk"


def _source_pool_for_scenario(
    scenario: str,
    *,
    dynamic_source_samples: list[SourceSample],
    static_source_samples: list[SourceSample],
) -> list[SourceSample]:
    if scenario == "static_hold" and static_source_samples:
        return static_source_samples
    if dynamic_source_samples:
        return dynamic_source_samples
    return static_source_samples


def _static_scenario_for_index(index: int) -> str:
    scenarios = (
        "static_pose_jitter",
        "static_closed_pose",
        "static_pose_mixup",
        "static_partial_pose",
    )
    return scenarios[int(index) % len(scenarios)]


def _alternate_static_source(
    source_pool: list[SourceSample],
    *,
    source_label: str,
    index: int,
) -> np.ndarray | None:
    static_sources = [
        sequence
        for label, _path, sequence, scope in source_pool
        if scope in STATIC_SOURCE_TYPES and label != source_label
    ]
    if not static_sources:
        return None
    return static_sources[int(index) % len(static_sources)]


def _static_negative_sequence(
    source: np.ndarray,
    *,
    scenario: str,
    target_frames: int,
    rng: np.random.Generator,
    alternate: np.ndarray | None = None,
) -> np.ndarray:
    base = resample_sequence(_as_static_frame_matrix(source), target_frames=target_frames)
    if scenario == "static_pose_mixup" and alternate is not None:
        other = resample_sequence(
            _as_static_frame_matrix(alternate),
            target_frames=target_frames,
        )
        alpha = float(rng.uniform(0.35, 0.65))
        base = base * alpha + other * (1.0 - alpha)

    if scenario == "static_closed_pose":
        frames = np.stack([_fold_static_frame(frame, rng=rng) for frame in base], axis=0)
    elif scenario == "static_partial_pose":
        frames = np.stack([_partial_static_frame(frame, rng=rng) for frame in base], axis=0)
    else:
        frames = base.astype(np.float32, copy=True)

    drift = rng.normal(0.0, 0.004, size=(target_frames, 1)).astype(np.float32)
    jitter = rng.normal(0.0, 0.006, size=frames.shape).astype(np.float32)
    frames = frames + jitter + drift
    return np.clip(frames, 0.0, 1.0).astype(np.float32, copy=False)


def _fold_static_frame(frame: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(frame[:42], dtype=np.float32).reshape(21, 2).copy()
    palm = points[[0, 5, 9, 13, 17]].mean(axis=0)
    finger_indices = np.asarray(
        [6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20],
        dtype=np.int64,
    )
    strength = float(rng.uniform(0.35, 0.70))
    points[finger_indices] = (
        points[finger_indices] * (1.0 - strength)
        + palm[None, :] * strength
    )
    points[finger_indices] += rng.normal(
        0.0,
        0.015,
        size=(len(finger_indices), 2),
    ).astype(np.float32)
    return np.clip(points.reshape(-1), 0.0, 1.0).astype(np.float32, copy=False)


def _partial_static_frame(frame: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(frame[:42], dtype=np.float32).reshape(21, 2).copy()
    palm = points[[0, 5, 9, 13, 17]].mean(axis=0)
    fingers = (
        [6, 7, 8],
        [10, 11, 12],
        [14, 15, 16],
        [18, 19, 20],
    )
    keep_index = int(rng.integers(0, len(fingers)))
    for index, finger in enumerate(fingers):
        if index == keep_index:
            continue
        strength = float(rng.uniform(0.45, 0.85))
        finger_idx = np.asarray(finger, dtype=np.int64)
        points[finger_idx] = (
            points[finger_idx] * (1.0 - strength)
            + palm[None, :] * strength
        )
    return np.clip(points.reshape(-1), 0.0, 1.0).astype(np.float32, copy=False)


def _negative_sequence(
    source: np.ndarray,
    *,
    scenario: str,
    target_frames: int,
    rng: np.random.Generator,
) -> np.ndarray:
    source = resample_sequence(source, target_frames=target_frames)
    pose = _pose_template(source, target_frames=target_frames, rng=rng)
    start = _start_wrist(source, rng=rng)
    if scenario == "static_hold":
        wrist = _static_wrist(start, target_frames, rng=rng)
    elif scenario == "ambiguous_diagonal":
        wrist = _ambiguous_diagonal(start, target_frames, rng=rng)
    elif scenario == "out_and_back_return":
        wrist = _out_and_back(start, target_frames, rng=rng, amplitude=0.22)
    elif scenario == "aborted_partial_motion":
        wrist = _out_and_back(start, target_frames, rng=rng, amplitude=0.11)
    else:
        wrist = _closed_random_walk(start, target_frames, rng=rng)
    return np.concatenate([pose, wrist], axis=1).astype(np.float32, copy=False)


def _pose_template(
    source: np.ndarray,
    *,
    target_frames: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pose = source[:, :42]
    base = pose[int(rng.integers(0, max(1, pose.shape[0])))]
    jitter = rng.normal(0.0, 0.0025, size=(target_frames, 42)).astype(np.float32)
    return (np.tile(base, (target_frames, 1)) + jitter).astype(np.float32, copy=False)


def _start_wrist(source: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    if source.shape[1] >= 44 and np.any(np.abs(source[:, 42:44]) > 1e-6):
        wrist = source[int(rng.integers(0, max(1, source.shape[0]))), 42:44]
    else:
        wrist = np.asarray([0.5, 0.5], dtype=np.float32)
    return np.clip(wrist.astype(np.float32, copy=False), 0.15, 0.85)


def _static_wrist(
    start: np.ndarray,
    frames: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    jitter = rng.normal(0.0, 0.0015, size=(frames, 2)).astype(np.float32)
    return np.clip(np.tile(start, (frames, 1)) + jitter, 0.0, 1.0)


def _closed_random_walk(
    start: np.ndarray,
    frames: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, frames, dtype=np.float32)
    radius_x = float(rng.uniform(0.025, 0.055))
    radius_y = float(rng.uniform(0.020, 0.050))
    phase = float(rng.uniform(0.0, np.pi))
    loop = np.stack(
        [
            np.cos(t + phase) - np.cos(phase),
            np.sin(t + phase) - np.sin(phase),
        ],
        axis=1,
    )
    loop[:, 0] *= radius_x
    loop[:, 1] *= radius_y
    return np.clip(start + loop, 0.0, 1.0).astype(np.float32, copy=False)


def _ambiguous_diagonal(
    start: np.ndarray,
    frames: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    sign_x = -1.0 if rng.random() < 0.5 else 1.0
    sign_y = -1.0 if rng.random() < 0.5 else 1.0
    amplitude = float(rng.uniform(0.12, 0.22))
    end = start + np.asarray([sign_x * amplitude, sign_y * amplitude], dtype=np.float32)
    return _linear_path(start, np.clip(end, 0.05, 0.95), frames, rng=rng)


def _out_and_back(
    start: np.ndarray,
    frames: int,
    *,
    rng: np.random.Generator,
    amplitude: float,
) -> np.ndarray:
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
    mid = np.clip(start + direction * float(amplitude), 0.05, 0.95)
    first = _linear_path(start, mid, max(2, frames // 2), rng=rng)
    second = _linear_path(mid, start, frames - first.shape[0] + 1, rng=rng)
    return np.concatenate([first, second[1:]], axis=0)[:frames]


def _linear_path(
    start: np.ndarray,
    end: np.ndarray,
    frames: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(2, frames), dtype=np.float32)[:, None]
    path = start + (end - start) * t
    path += rng.normal(0.0, 0.002, size=path.shape).astype(np.float32)
    return np.clip(path, 0.0, 1.0).astype(np.float32, copy=False)


def _report_to_dict(report: NegativeSamplingReport) -> dict[str, object]:
    payload = asdict(report)
    payload["data_root"] = _portable_path(report.data_root)
    payload["samples"] = [
        {
            **asdict(item),
            "path": _portable_path(item.path),
            "source_path": _portable_path(item.source_path),
        }
        for item in report.samples
    ]
    return payload


def _portable_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "gestures")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "configs" / "gesture_taxonomy.json",
    )
    parser.add_argument("--samples-per-label", type=int, default=20)
    parser.add_argument("--target-frames", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Negative label to generate; can be passed multiple times.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=ROOT / "docs" / "experiments" / "negative_sampling_manifest.json",
    )
    parser.add_argument(
        "--keep-existing-auto",
        action="store_true",
        help="Do not remove existing sample_auto_*.npy files before generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = args.label or list(DEFAULT_NEGATIVE_LABELS)
    report = generate_negative_samples(
        data_root=args.data_root,
        taxonomy_path=args.taxonomy,
        labels=labels,
        samples_per_label=args.samples_per_label,
        target_frames=args.target_frames,
        seed=args.seed,
        clean_auto=not bool(args.keep_existing_auto),
        manifest_out=args.manifest_out,
    )
    print(
        "[✓] Generated "
        f"{report.generated_samples} negative samples from "
        f"{report.source_samples} source samples"
    )
    for label, count in sorted(report.labels.items()):
        print(f"  - {label}: {count}")


if __name__ == "__main__":
    main()
