"""Train and evaluate class-conditional dynamic gesture completion gates."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from cv.gesture_dataset_files import real_sample_paths
from cv.gesture_features import (
    global_trajectory_xy,
    global_wrist_slices,
    hand_feature_blocks,
    sequence_shape_change_energy,
    sequence_to_matrix,
)

COMPLETION_SCHEMA_VERSION = 2
COMPLETION_METHOD = "per_class_logistic_cross_class_prefix"
COMPLETION_PROFILE_METHOD = "logistic_cross_class_prefix_verifier"
COMPLETION_FEATURE_NAMES = (
    "relative_dx",
    "relative_dy",
    "relative_displacement",
    "relative_path",
    "relative_excursion",
    "relative_x_range",
    "relative_y_range",
    "relative_positive_x_path",
    "relative_negative_x_path",
    "relative_positive_y_path",
    "relative_negative_y_path",
    "straightness",
    "turning_energy",
    "shape_change_energy",
    "shape_endpoint_delta",
)
DEFAULT_PREFIX_FRACTIONS = (0.25, 0.40, 0.55, 0.70)
DEFAULT_MIN_SOURCE_SAMPLES = 6
DEFAULT_GROUPED_CV_FOLDS = 5
DEFAULT_MIN_COMPLETE_RECALL = 1.0
DEFAULT_MAX_CROSS_CLASS_SOURCES = 20


def supports_dynamic_completion(feature_mode: str) -> bool:
    clean = str(feature_mode or "").strip().lower()
    return clean.startswith("dynamic_") or clean == "hybrid_stats"


def completion_target_frames(feature_mode: str) -> int:
    clean = str(feature_mode or "").strip().lower()
    return 72 if clean in {"dynamic_landmark_image", "dynamic_sequence_72"} else 36


def is_negative_completion_label(label: str) -> bool:
    clean = str(label or "").strip().lower()
    return (
        clean.startswith("negative_")
        or clean.startswith("background_")
        or clean.startswith("no_gesture")
        or clean.startswith("random_")
        or clean.startswith("partial_")
        or clean.startswith("return_")
        or clean.startswith("wrong_axis_")
    )


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _motion_scale(value: float | Iterable[float] | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.number)):
        return max(0.0, _finite(value))
    values = [
        _finite(item)
        for item in value
        if _finite(item) > 1e-6
    ]
    return float(np.median(values)) if values else 0.0


def _shape_endpoint_delta(sequence: np.ndarray) -> float:
    values: list[float] = []
    for block in hand_feature_blocks(int(sequence.shape[1])):
        pose = sequence[:, block.start : block.pose_end]
        points = pose.reshape(
            sequence.shape[0],
            21,
            block.coords_per_point,
        )[:, :, :2]
        first = points[0]
        last = points[-1]
        first_distances = np.linalg.norm(
            first[:, None, :] - first[None, :, :],
            axis=2,
        )
        last_distances = np.linalg.norm(
            last[:, None, :] - last[None, :, :],
            axis=2,
        )
        upper = np.triu_indices(21, k=1)
        values.append(
            float(
                np.mean(
                    np.abs(
                        last_distances[upper] - first_distances[upper]
                    )
                )
            )
        )
    return max(values, default=0.0)


def _shape_step_signal(sequence: np.ndarray) -> np.ndarray:
    signals: list[np.ndarray] = []
    for block in hand_feature_blocks(int(sequence.shape[1])):
        pose = sequence[:, block.start : block.pose_end]
        points = pose.reshape(
            sequence.shape[0],
            21,
            block.coords_per_point,
        )[:, :, :2]
        upper = np.triu_indices(21, k=1)
        distances = np.linalg.norm(
            points[:, :, None, :] - points[:, None, :, :],
            axis=3,
        )[:, upper[0], upper[1]]
        signals.append(np.mean(np.abs(np.diff(distances, axis=0)), axis=1))
    if not signals:
        return np.zeros(max(0, sequence.shape[0] - 1), dtype=np.float32)
    return np.max(np.stack(signals, axis=0), axis=0).astype(
        np.float32,
        copy=False,
    )


def completion_progress_prefix(
    sequence: np.ndarray,
    fraction: float,
    *,
    motion_scale: float,
) -> np.ndarray:
    """Take a prefix by cumulative trajectory/shape progress, not frame count."""
    seq = sequence_to_matrix(sequence).astype(np.float32, copy=False)
    if seq.shape[0] <= 3:
        return seq.copy()
    target = max(0.10, min(0.90, float(fraction)))
    trajectory = global_trajectory_xy(seq)
    global_steps = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    scale = max(1e-6, float(motion_scale))
    progress_signal = global_steps / scale + _shape_step_signal(seq)
    finite = progress_signal[np.isfinite(progress_signal)]
    if finite.size == 0 or float(np.max(finite)) <= 1e-9:
        end = max(3, int(round(seq.shape[0] * target)))
        return seq[: min(seq.shape[0] - 1, end)].copy()

    noise_floor = max(1e-6, float(np.quantile(finite, 0.90)) * 0.05)
    progress_signal = np.where(progress_signal >= noise_floor, progress_signal, 0.0)
    cumulative = np.cumsum(progress_signal)
    total = float(cumulative[-1]) if cumulative.size else 0.0
    if total <= 1e-9:
        end = max(3, int(round(seq.shape[0] * target)))
        return seq[: min(seq.shape[0] - 1, end)].copy()
    step_index = int(np.searchsorted(cumulative, target * total, side="left"))
    end = max(3, min(seq.shape[0] - 1, step_index + 2))
    return seq[:end].copy()


def build_dynamic_completion_evidence(
    sequence: np.ndarray,
    *,
    motion_scale: float | Iterable[float] | None = None,
    target_frames: int = 72,
) -> dict[str, float]:
    """Extract completion evidence before global amplitude normalization."""
    seq = sequence_to_matrix(sequence).astype(np.float32, copy=False)
    trajectory = global_trajectory_xy(seq)
    steps = (
        np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        if trajectory.shape[0] > 1
        else np.asarray([], dtype=np.float32)
    )
    displacement = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
    path_length = float(steps.sum()) if steps.size else 0.0
    delta = trajectory[-1] - trajectory[0]
    step_vectors = (
        np.diff(trajectory, axis=0)
        if trajectory.shape[0] > 1
        else np.zeros((0, 2), dtype=np.float32)
    )
    ranges = np.ptp(trajectory, axis=0)
    excursion = float(
        np.linalg.norm(trajectory - trajectory[0], axis=1).max()
    )
    scale = _motion_scale(motion_scale)
    has_global_wrist = bool(global_wrist_slices(int(seq.shape[1])))

    # The classifier may normalize global amplitude later. Shape channels are
    # unaffected by that operation, so they can be measured on the canonical
    # time grid without losing completion information.
    from cv.dynamic_motion import canonical_dynamic_sequence

    canonical = canonical_dynamic_sequence(
        seq,
        target_frames=max(2, int(target_frames)),
    )
    safe_scale = scale if scale > 1e-6 else 0.0
    directions = (
        step_vectors / np.maximum(
            np.linalg.norm(step_vectors, axis=1, keepdims=True),
            1e-6,
        )
        if step_vectors.size
        else np.zeros((0, 2), dtype=np.float32)
    )
    turning_energy = (
        float(np.mean(1.0 - np.sum(directions[1:] * directions[:-1], axis=1)))
        if directions.shape[0] > 1
        else 0.0
    )

    def relative(value: float) -> float:
        return float(value / safe_scale) if safe_scale > 0.0 else 0.0

    return {
        "raw_frames": float(seq.shape[0]),
        "motion_scale": scale,
        "has_global_wrist": float(has_global_wrist),
        "raw_displacement": displacement,
        "raw_path": path_length,
        "raw_excursion": excursion,
        "relative_dx": relative(float(delta[0])),
        "relative_dy": relative(float(delta[1])),
        "relative_displacement": relative(displacement),
        "relative_path": relative(path_length),
        "relative_excursion": relative(excursion),
        "relative_x_range": relative(float(ranges[0])),
        "relative_y_range": relative(float(ranges[1])),
        "relative_positive_x_path": relative(
            float(np.clip(step_vectors[:, 0], 0.0, None).sum())
        ),
        "relative_negative_x_path": relative(
            float(np.clip(-step_vectors[:, 0], 0.0, None).sum())
        ),
        "relative_positive_y_path": relative(
            float(np.clip(step_vectors[:, 1], 0.0, None).sum())
        ),
        "relative_negative_y_path": relative(
            float(np.clip(-step_vectors[:, 1], 0.0, None).sum())
        ),
        "straightness": displacement / max(path_length, 1e-6),
        "turning_energy": turning_energy,
        "shape_change_energy": sequence_shape_change_energy(canonical),
        "shape_endpoint_delta": _shape_endpoint_delta(canonical),
    }


def extract_runtime_completion_update(
    sequence: np.ndarray,
    *,
    motion_scale: float,
    target_frames: int,
    hold_frames: int = 8,
) -> Any | None:
    """Replay a recording through the exact segmenter used by live inference."""
    from cv.dynamic_motion import create_runtime_dynamic_segmenter

    seq = sequence_to_matrix(sequence).astype(np.float32, copy=False)
    segmenter = create_runtime_dynamic_segmenter(target_frames=target_frames)
    for frame in seq:
        update = segmenter.update(frame, motion_scale=motion_scale)
        if update.completed_sequence is not None:
            return update
    for _ in range(max(0, int(hold_frames))):
        update = segmenter.update(seq[-1], motion_scale=motion_scale)
        if update.completed_sequence is not None:
            return update
    update = segmenter.finish_due_to_hand_lost()
    return update if update.completed_sequence is not None else None


def completion_feature_vector(
    evidence: dict[str, Any],
    feature_names: Sequence[str] = COMPLETION_FEATURE_NAMES,
) -> np.ndarray:
    return np.asarray(
        [_finite(evidence.get(name)) for name in feature_names],
        dtype=np.float64,
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_value = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(max(value, -700.0))
    return exp_value / (1.0 + exp_value)


def evaluate_completion_profile(
    profile: dict[str, Any] | None,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Return a fail-open assessment for one trained completion profile."""
    if not isinstance(profile, dict) or not bool(profile.get("enabled")):
        return {
            "enabled": False,
            "accepted": True,
            "score": 1.0,
            "threshold": 0.0,
            "reason": "profile_missing",
        }

    names = profile.get("feature_names")
    mean = np.asarray(profile.get("scaler_mean") or [], dtype=np.float64)
    scale = np.asarray(profile.get("scaler_scale") or [], dtype=np.float64)
    coefficients = np.asarray(
        profile.get("coefficients") or [],
        dtype=np.float64,
    )
    if not isinstance(names, list):
        names = list(COMPLETION_FEATURE_NAMES)
    vector = completion_feature_vector(evidence, names)
    if not (
        vector.shape == mean.shape == scale.shape == coefficients.shape
        and vector.size > 0
    ):
        return {
            "enabled": False,
            "accepted": True,
            "score": 1.0,
            "threshold": 0.0,
            "reason": "invalid_profile",
        }

    safe_scale = np.where(np.abs(scale) > 1e-9, scale, 1.0)
    normalized = (vector - mean) / safe_scale
    logit = float(np.dot(normalized, coefficients)) + _finite(
        profile.get("intercept")
    )
    score = _sigmoid(logit)
    threshold = max(0.0, min(1.0, _finite(profile.get("threshold"), 0.5)))
    accepted = score >= threshold
    return {
        "enabled": True,
        "accepted": bool(accepted),
        "score": float(score),
        "threshold": float(threshold),
        "reason": "completion_accepted" if accepted else "incomplete_gesture",
        "method": str(profile.get("method") or COMPLETION_PROFILE_METHOD),
    }


def evaluate_dynamic_completion(
    payload: dict[str, Any] | None,
    label: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    classes = payload.get("classes") if isinstance(payload, dict) else {}
    if not isinstance(classes, dict):
        classes = {}
    clean = str(label or "").strip()
    profile = classes.get(clean)
    if not isinstance(profile, dict):
        lowered = clean.lower()
        profile = next(
            (
                candidate
                for raw_label, candidate in classes.items()
                if str(raw_label).strip().lower() == lowered
                and isinstance(candidate, dict)
            ),
            None,
        )
    result = evaluate_completion_profile(profile, evidence)
    result["label"] = clean
    return result


def _sample_motion_scale(path: Path) -> float:
    metadata_path = path.with_suffix(".meta.json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0.0
    quality = metadata.get("quality")
    if not isinstance(quality, dict):
        quality = {}
    return max(
        0.0,
        _finite(
            metadata.get("projected_hand_scale_median")
            or quality.get("projected_hand_scale")
        ),
    )


def _label_directory(data_root: Path, label: str) -> Path | None:
    direct = data_root / str(label)
    if direct.is_dir():
        return direct
    lowered = str(label).strip().lower()
    return next(
        (
            path
            for path in data_root.iterdir()
            if path.is_dir() and path.name.lower() == lowered
        ),
        None,
    )


def _fit_completion_profile(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    fractions: np.ndarray,
    negative_kinds: np.ndarray,
    *,
    min_complete_recall: float,
    cv_folds: int,
    random_state: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    unique_groups = sorted(set(str(value) for value in groups.tolist()))
    folds = min(max(2, int(cv_folds)), len(unique_groups))
    splitter = GroupKFold(n_splits=folds)
    out_of_fold = np.zeros(targets.shape[0], dtype=np.float64)

    for train_indices, test_indices in splitter.split(features, targets, groups):
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=int(random_state),
            ),
        )
        pipeline.fit(features[train_indices], targets[train_indices])
        out_of_fold[test_indices] = pipeline.predict_proba(
            features[test_indices]
        )[:, 1]

    positive_scores = np.sort(out_of_fold[targets == 1])[::-1]
    required = max(
        1,
        min(
            positive_scores.size,
            int(math.ceil(float(min_complete_recall) * positive_scores.size)),
        ),
    )
    threshold = max(0.01, float(positive_scores[required - 1]) - 1e-9)
    accepted = out_of_fold >= threshold
    complete_recall = float(np.mean(accepted[targets == 1]))
    negative_mask = targets == 0
    prefix_mask = np.asarray(
        [str(value).endswith("prefix") for value in negative_kinds],
        dtype=bool,
    ) & negative_mask
    negative_false_accept_rate = float(np.mean(accepted[negative_mask]))
    partial_false_accept_rate = (
        float(np.mean(accepted[prefix_mask])) if np.any(prefix_mask) else 0.0
    )
    by_fraction = {
        f"{fraction:.2f}": float(
            np.mean(accepted[prefix_mask & np.isclose(fractions, fraction)])
        )
        for fraction in sorted(
            set(float(value) for value in fractions[prefix_mask].tolist())
        )
    }
    by_kind = {
        str(kind): float(np.mean(accepted[negative_kinds == kind]))
        for kind in sorted(set(str(value) for value in negative_kinds[negative_mask]))
    }

    final_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=int(random_state),
        ),
    )
    final_pipeline.fit(features, targets)
    scaler = final_pipeline.named_steps["standardscaler"]
    classifier = final_pipeline.named_steps["logisticregression"]

    return {
        "enabled": True,
        "method": COMPLETION_PROFILE_METHOD,
        "feature_names": list(COMPLETION_FEATURE_NAMES),
        "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
        "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
        "coefficients": [
            float(value) for value in classifier.coef_.reshape(-1).tolist()
        ],
        "intercept": float(classifier.intercept_.reshape(-1)[0]),
        "threshold": float(threshold),
        "validation": {
            "grouped_cv": True,
            "folds": int(folds),
            "source_groups": int(len(unique_groups)),
            "complete_recall": complete_recall,
            "negative_false_accept_rate": negative_false_accept_rate,
            "partial_false_accept_rate": partial_false_accept_rate,
            "partial_false_accept_rate_by_fraction": by_fraction,
            "negative_false_accept_rate_by_kind": by_kind,
            "full_score_min": float(np.min(out_of_fold[targets == 1])),
            "full_score_mean": float(np.mean(out_of_fold[targets == 1])),
            "partial_score_mean": float(np.mean(out_of_fold[targets == 0])),
        },
    }


def build_dynamic_completion_profiles(
    data_root: Path,
    classes: Sequence[str],
    *,
    target_dim: int,
    target_frames: int,
    prefix_fractions: Sequence[float] = DEFAULT_PREFIX_FRACTIONS,
    min_source_samples: int = DEFAULT_MIN_SOURCE_SAMPLES,
    cv_folds: int = DEFAULT_GROUPED_CV_FOLDS,
    min_complete_recall: float = DEFAULT_MIN_COMPLETE_RECALL,
    max_cross_class_sources: int = DEFAULT_MAX_CROSS_CLASS_SOURCES,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit completion profiles for arbitrary positive dynamic class names."""
    root = Path(data_root)
    fractions = tuple(
        sorted(
            {
                max(0.10, min(0.90, float(value)))
                for value in prefix_fractions
            }
        )
    )
    class_profiles: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    samples_by_label: dict[str, list[tuple[Path, np.ndarray, float]]] = {}

    for raw_label in classes:
        label = str(raw_label or "").strip()
        if not label:
            continue
        label_dir = _label_directory(root, label)
        if label_dir is None:
            if not is_negative_completion_label(label):
                skipped[label] = "missing_label_directory"
            continue
        label_samples: list[tuple[Path, np.ndarray, float]] = []
        for sample_path in real_sample_paths(label_dir):
            try:
                sequence = sequence_to_matrix(np.load(sample_path)).astype(
                    np.float32,
                    copy=False,
                )
            except Exception:
                continue
            if int(sequence.shape[1]) != int(target_dim):
                continue
            if not global_wrist_slices(int(sequence.shape[1])):
                continue
            scale = _sample_motion_scale(sample_path)
            if scale <= 1e-6:
                continue
            label_samples.append((sample_path, sequence, scale))
        samples_by_label[label] = label_samples

    cross_class_limit = max(2, int(max_cross_class_sources))
    for raw_label in classes:
        label = str(raw_label or "").strip()
        if not label or is_negative_completion_label(label):
            continue
        label_samples = samples_by_label.get(label, [])
        usable_sources = len(label_samples)
        if usable_sources < max(2, int(min_source_samples)):
            skipped[label] = f"not_enough_source_samples:{usable_sources}"
            continue

        rows: list[np.ndarray] = []
        targets: list[int] = []
        groups: list[str] = []
        sample_fractions: list[float] = []
        negative_kinds: list[str] = []
        negative_counts: dict[str, int] = {}

        def append_evidence(
            sequence: np.ndarray,
            *,
            scale: float,
            source_group: str,
            target: int,
            fraction: float,
            kind: str,
        ) -> None:
            evidence = build_dynamic_completion_evidence(
                sequence,
                motion_scale=scale,
                target_frames=target_frames,
            )
            rows.append(completion_feature_vector(evidence))
            targets.append(int(target))
            groups.append(source_group)
            sample_fractions.append(float(fraction))
            negative_kinds.append(str(kind))
            if not target:
                negative_counts[kind] = negative_counts.get(kind, 0) + 1

        for sample_path, sequence, scale in label_samples:
            source_group = str(sample_path)
            append_evidence(
                sequence,
                scale=scale,
                source_group=source_group,
                target=1,
                fraction=1.0,
                kind="complete_target",
            )
            for fraction in fractions:
                prefix = completion_progress_prefix(
                    sequence,
                    fraction,
                    motion_scale=scale,
                )
                append_evidence(
                    prefix,
                    scale=scale,
                    source_group=source_group,
                    target=0,
                    fraction=float(fraction),
                    kind="own_prefix",
                )

        for source_label, source_samples in samples_by_label.items():
            if source_label == label:
                continue
            is_negative = is_negative_completion_label(source_label)
            prefix_kind = "negative_prefix" if is_negative else "cross_class_prefix"
            full_kind = "negative_full" if is_negative else "cross_class_full"
            for sample_path, sequence, scale in source_samples[:cross_class_limit]:
                source_group = str(sample_path)
                append_evidence(
                    sequence,
                    scale=scale,
                    source_group=source_group,
                    target=0,
                    fraction=1.0,
                    kind=full_kind,
                )
                for fraction in fractions:
                    prefix = completion_progress_prefix(
                        sequence,
                        fraction,
                        motion_scale=scale,
                    )
                    append_evidence(
                        prefix,
                        scale=scale,
                        source_group=source_group,
                        target=0,
                        fraction=float(fraction),
                        kind=prefix_kind,
                    )

        if not negative_counts:
            skipped[label] = "no_prefix_or_cross_class_negatives"
            continue

        profile = _fit_completion_profile(
            np.asarray(rows, dtype=np.float64),
            np.asarray(targets, dtype=np.int64),
            np.asarray(groups, dtype=object),
            np.asarray(sample_fractions, dtype=np.float64),
            np.asarray(negative_kinds, dtype=object),
            min_complete_recall=max(0.80, min(1.0, float(min_complete_recall))),
            cv_folds=cv_folds,
            random_state=random_state,
        )
        profile["source_sample_count"] = int(usable_sources)
        profile["negative_sample_counts"] = {
            str(key): int(value) for key, value in sorted(negative_counts.items())
        }
        profile["segmented_prefix_count"] = int(
            sum(
                value
                for key, value in negative_counts.items()
                if str(key).endswith("prefix")
            )
        )
        profile["prefix_fractions"] = [float(value) for value in fractions]
        class_profiles[label] = profile

    return {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "method": COMPLETION_METHOD,
        "generated_at": time.time(),
        "target_dim": int(target_dim),
        "target_frames": int(target_frames),
        "prefix_fractions": [float(value) for value in fractions],
        "min_complete_recall": float(min_complete_recall),
        "max_cross_class_sources": int(cross_class_limit),
        "classes": class_profiles,
        "skipped_classes": skipped,
    }


def update_rejection_metadata(
    rejection_path: Path,
    profiles: dict[str, Any],
) -> None:
    path = Path(rejection_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("rejection metadata must be a JSON object")
    payload["dynamic_completion_profiles"] = profiles
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dynamic completion profiles without retraining weights."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/gestures"))
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--rejection", type=Path, required=True)
    parser.add_argument("--target-dim", type=int, required=True)
    parser.add_argument("--target-frames", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    classes = json.loads(args.classes.read_text(encoding="utf-8"))
    if not isinstance(classes, list):
        raise ValueError("classes file must contain a JSON list")
    profiles = build_dynamic_completion_profiles(
        args.data_root,
        [str(label) for label in classes],
        target_dim=int(args.target_dim),
        target_frames=int(args.target_frames),
    )
    update_rejection_metadata(args.rejection, profiles)
    validations = [
        profile.get("validation", {})
        for profile in profiles.get("classes", {}).values()
        if isinstance(profile, dict)
    ]
    complete_recall = float(
        np.mean([_finite(item.get("complete_recall")) for item in validations])
    ) if validations else 0.0
    partial_far = float(
        np.mean(
            [_finite(item.get("partial_false_accept_rate")) for item in validations]
        )
    ) if validations else 0.0
    print(
        "[dynamic-completion] "
        f"profiles={len(profiles.get('classes', {}))} "
        f"complete_recall={complete_recall:.4f} "
        f"partial_false_accept_rate={partial_far:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
