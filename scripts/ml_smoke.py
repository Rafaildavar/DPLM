"""CI smoke checks for GestureFlow ML artifacts.

The script intentionally avoids camera and GUI access. It validates that the
tracked model artifacts can be loaded, that metadata is consistent with feature
extraction, and that inference returns finite probabilities.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

from cv.gesture_features import (
    FEATURE_DYNAMIC_SEQUENCE,
    FEATURE_STATIC_MEAN,
    build_feature_vector,
    infer_raw_dim_from_feature_size,
)


@dataclass(frozen=True)
class SmokeProfile:
    name: str
    model: str
    classes: str
    feature_dim: str
    feature_mode: str
    rejection: str = ""
    prototypes: str = ""
    required: bool = True


@dataclass
class SmokeResult:
    name: str
    status: str
    model_path: str
    feature_mode: str
    feature_dim: int
    class_count: int
    predicted_label: str
    confidence: float
    inference_ms: float
    error: str = ""


DEFAULT_PROFILES: tuple[SmokeProfile, ...] = (
    SmokeProfile(
        name="static_default",
        model="knn.pkl",
        classes="classes.json",
        feature_dim="feature_dim.txt",
        feature_mode="feature_mode.txt",
        rejection="gesture_rejection.json",
    ),
    SmokeProfile(
        name="dynamic_sequence_mlp",
        model="dynamic_sequence_mlp.pkl",
        classes="dynamic_sequence_mlp_classes.json",
        feature_dim="dynamic_sequence_mlp_feature_dim.txt",
        feature_mode="dynamic_sequence_mlp_feature_mode.txt",
        rejection="dynamic_sequence_mlp_rejection.json",
        prototypes="dynamic_sequence_mlp_prototypes.json",
        required=False,
    ),
    SmokeProfile(
        name="dynamic_sequence_lstm_backbone",
        model="dynamic_sequence_lstm_backbone.pkl",
        classes="dynamic_sequence_lstm_backbone_classes.json",
        feature_dim="dynamic_sequence_lstm_backbone_feature_dim.txt",
        feature_mode="dynamic_sequence_lstm_backbone_feature_mode.txt",
        rejection="dynamic_sequence_lstm_backbone_rejection.json",
        prototypes="dynamic_sequence_lstm_backbone_prototypes.json",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model artifact smoke checks.")
    parser.add_argument("--models-dir", default="models", help="Directory with model artifacts.")
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Profile name to check. Defaults to static, MLP dynamic and LSTM dynamic.",
    )
    parser.add_argument(
        "--report-json",
        default="outputs/ci/ml_smoke_report.json",
        help="Where to write the machine-readable report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    selected = _select_profiles(args.profile)
    allow_optional_skips = not bool(args.profile)
    results = [
        _run_profile(
            models_dir,
            profile,
            allow_optional_skips=allow_optional_skips,
        )
        for profile in selected
    ]
    payload = {
        "status": (
            "ok"
            if all(item.status in {"ok", "skipped"} for item in results)
            else "failed"
        ),
        "models_dir": str(models_dir),
        "profiles": [asdict(item) for item in results],
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for item in results:
        if item.status == "ok":
            print(
                "[ml-smoke] "
                f"{item.name}: ok label={item.predicted_label!r} "
                f"conf={item.confidence:.3f} latency={item.inference_ms:.2f}ms"
            )
        elif item.status == "skipped":
            print(f"[ml-smoke] {item.name}: skipped {item.error}")
        else:
            print(f"[ml-smoke] {item.name}: failed {item.error}")
    print(f"[ml-smoke] report={report_path}")
    if payload["status"] != "ok":
        raise SystemExit(1)


def _select_profiles(names: Iterable[str]) -> list[SmokeProfile]:
    requested = [str(name).strip() for name in names if str(name).strip()]
    if not requested:
        return list(DEFAULT_PROFILES)
    by_name = {profile.name: profile for profile in DEFAULT_PROFILES}
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise SystemExit(f"Unknown smoke profile(s): {', '.join(missing)}")
    return [by_name[name] for name in requested]


def _run_profile(
    models_dir: Path,
    profile: SmokeProfile,
    *,
    allow_optional_skips: bool = False,
) -> SmokeResult:
    model_path = models_dir / profile.model
    try:
        paths = {
            "model": model_path,
            "classes": models_dir / profile.classes,
            "feature_dim": models_dir / profile.feature_dim,
            "feature_mode": models_dir / profile.feature_mode,
        }
        if profile.rejection:
            paths["rejection"] = models_dir / profile.rejection
        if profile.prototypes:
            paths["prototypes"] = models_dir / profile.prototypes
        for label, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"missing {label}: {path}")

        classes = json.loads(paths["classes"].read_text(encoding="utf-8"))
        if not isinstance(classes, list) or not classes:
            raise ValueError(f"classes metadata is empty or invalid: {paths['classes']}")
        feature_dim = int(paths["feature_dim"].read_text(encoding="utf-8").strip())
        feature_mode = paths["feature_mode"].read_text(encoding="utf-8").strip()
        raw_dim = infer_raw_dim_from_feature_size(feature_mode, feature_dim)
        sample = _synthetic_sequence(
            raw_dim=raw_dim,
            frames=72 if feature_mode == FEATURE_DYNAMIC_SEQUENCE else 16,
            dynamic=feature_mode != FEATURE_STATIC_MEAN,
        )
        feature = build_feature_vector(sample, mode=feature_mode, target_dim=raw_dim)
        if int(feature.shape[0]) != feature_dim:
            raise ValueError(
                f"feature size mismatch for {profile.name}: {feature.shape[0]} != {feature_dim}"
            )

        started_at = time.perf_counter()
        model = joblib.load(model_path)
        probabilities = model.predict_proba(feature.reshape(1, -1))[0]
        inference_ms = (time.perf_counter() - started_at) * 1000.0
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 1 or probabilities.size != len(classes):
            raise ValueError(
                f"probability shape mismatch: {probabilities.shape} vs classes={len(classes)}"
            )
        if not np.isfinite(probabilities).all():
            raise ValueError("model returned non-finite probabilities")
        probability_sum = float(probabilities.sum())
        if abs(probability_sum - 1.0) > 1e-3:
            raise ValueError(f"probabilities do not sum to 1.0: {probability_sum:.6f}")

        predicted_index = int(np.argmax(probabilities))
        return SmokeResult(
            name=profile.name,
            status="ok",
            model_path=str(model_path),
            feature_mode=feature_mode,
            feature_dim=feature_dim,
            class_count=len(classes),
            predicted_label=str(classes[predicted_index]),
            confidence=float(probabilities[predicted_index]),
            inference_ms=round(inference_ms, 3),
        )
    except Exception as exc:
        if (
            allow_optional_skips
            and not profile.required
            and _is_legacy_pickle_compatibility_error(exc)
        ):
            return SmokeResult(
                name=profile.name,
                status="skipped",
                model_path=str(model_path),
                feature_mode="",
                feature_dim=0,
                class_count=0,
                predicted_label="",
                confidence=0.0,
                inference_ms=0.0,
                error=(
                    "legacy optional artifact is incompatible with this "
                    f"NumPy/joblib runtime: {exc}"
                ),
            )
        return SmokeResult(
            name=profile.name,
            status="failed",
            model_path=str(model_path),
            feature_mode="",
            feature_dim=0,
            class_count=0,
            predicted_label="",
            confidence=0.0,
            inference_ms=0.0,
            error=str(exc),
        )


def _is_legacy_pickle_compatibility_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "BitGenerator" in message
        and "not a known BitGenerator module" in message
    )


def _synthetic_sequence(*, raw_dim: int, frames: int, dynamic: bool) -> np.ndarray:
    raw_dim = max(1, int(raw_dim))
    frames = max(2, int(frames))
    time_axis = np.linspace(0.0, 1.0, frames, dtype=np.float32).reshape(frames, 1)
    channels = np.linspace(-0.4, 0.4, raw_dim, dtype=np.float32).reshape(1, raw_dim)
    base = np.sin((channels + 1.3) * np.pi).astype(np.float32) * 0.05
    if dynamic:
        motion = time_axis * (0.10 + np.abs(channels) * 0.05)
        wave = np.sin(time_axis * np.pi * 2.0 + channels).astype(np.float32) * 0.02
        return (base + motion + wave).astype(np.float32, copy=False)
    return np.repeat(base, frames, axis=0).astype(np.float32, copy=False)


if __name__ == "__main__":
    main()
