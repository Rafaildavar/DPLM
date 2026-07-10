#!/usr/bin/env python3
"""Benchmark GestureBind latency for defense/demo metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {}
    return {
        "mean_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p90_ms": round(float(np.percentile(arr, 90)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
    }


def _write_or_print(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


def _ml_profile_paths(profile: str) -> dict[str, Path]:
    model_dir = ROOT / "models"
    if profile == "dynamic":
        stem = "dynamic_landmark_lstm_backbone"
        return {
            "model": model_dir / f"{stem}.pkl",
            "classes": model_dir / f"{stem}_classes.json",
            "feature_dim": model_dir / f"{stem}_feature_dim.txt",
            "feature_mode": model_dir / f"{stem}_feature_mode.txt",
        }
    return {
        "model": model_dir / "knn.pkl",
        "classes": model_dir / "classes.json",
        "feature_dim": model_dir / "feature_dim.txt",
        "feature_mode": model_dir / "feature_mode.txt",
    }


def _load_raw_sequences(data_root: Path, classes: list[str]) -> list[np.ndarray]:
    from app.services.gesture_labels import gesture_label_key
    from cv.gesture_dataset_files import real_sample_paths
    from cv.gesture_features import sequence_to_matrix

    class_keys = {gesture_label_key(label) for label in classes}
    sequences: list[np.ndarray] = []
    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if gesture_label_key(label_dir.name) not in class_keys:
            continue
        for path in real_sample_paths(label_dir):
            try:
                sequences.append(sequence_to_matrix(np.load(path)))
            except (OSError, ValueError):
                continue
    return sequences


def _benchmark_ml_profile(
    profile: str,
    *,
    iterations: int,
    warmup: int,
    data_root: Path,
) -> dict[str, Any]:
    import joblib

    from app.gesture_online_infer import configure_estimator_for_live_inference
    from cv.gesture_features import build_feature_vector, infer_raw_dim_from_feature_size

    paths = _ml_profile_paths(profile)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing {profile} model artifacts: {', '.join(missing)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = joblib.load(paths["model"])
    configured_estimators = configure_estimator_for_live_inference(clf)

    classes = json.loads(paths["classes"].read_text(encoding="utf-8"))
    feature_dim = int(paths["feature_dim"].read_text(encoding="utf-8").strip())
    feature_mode = paths["feature_mode"].read_text(encoding="utf-8").strip()
    raw_feature_dim = infer_raw_dim_from_feature_size(feature_mode, feature_dim)
    raw_sequences = _load_raw_sequences(data_root, classes)
    compatible_sequences: list[np.ndarray] = []
    for sequence in raw_sequences:
        try:
            feature = build_feature_vector(
                sequence,
                mode=feature_mode,
                target_dim=raw_feature_dim,
            )
        except ValueError:
            continue
        if int(feature.size) == feature_dim:
            compatible_sequences.append(sequence)
    if not compatible_sequences:
        raise RuntimeError(f"No compatible gesture samples found for {profile}")

    def predict(feature: np.ndarray) -> None:
        matrix = feature.astype(np.float32, copy=False).reshape(1, -1)
        clf.predict(matrix)
        if hasattr(clf, "predict_proba"):
            clf.predict_proba(matrix)

    for index in range(max(0, int(warmup))):
        sequence = compatible_sequences[index % len(compatible_sequences)]
        feature = build_feature_vector(
            sequence,
            mode=feature_mode,
            target_dim=raw_feature_dim,
        )
        predict(feature)

    feature_timings: list[float] = []
    prediction_timings: list[float] = []
    pipeline_timings: list[float] = []
    for idx in range(iterations):
        sequence = compatible_sequences[idx % len(compatible_sequences)]
        start = time.perf_counter_ns()
        feature = build_feature_vector(
            sequence,
            mode=feature_mode,
            target_dim=raw_feature_dim,
        )
        after_features = time.perf_counter_ns()
        predict(feature)
        end = time.perf_counter_ns()
        feature_timings.append((after_features - start) / 1_000_000)
        prediction_timings.append((end - after_features) / 1_000_000)
        pipeline_timings.append((end - start) / 1_000_000)

    pipeline_summary = _summary(pipeline_timings)
    mean_pipeline_ms = float(pipeline_summary.get("mean_ms", 0.0))

    return {
        "profile": profile,
        "model": str(paths["model"]),
        "model_type": type(clf).__name__,
        "feature_mode": feature_mode,
        "iterations": int(iterations),
        "warmup": int(warmup),
        "dataset_examples": len(compatible_sequences),
        "raw_feature_dim": int(raw_feature_dim),
        "feature_dim": feature_dim,
        "classes": classes,
        "runtime_estimators_set_to_one_worker": int(configured_estimators),
        "feature_extraction_latency": _summary(feature_timings),
        "model_prediction_latency": _summary(prediction_timings),
        "ml_pipeline_latency": pipeline_summary,
        "capacity_fps": (
            round(1000.0 / mean_pipeline_ms, 2) if mean_pipeline_ms > 0.0 else 0.0
        ),
    }


def benchmark_ml(
    iterations: int,
    *,
    warmup: int = 20,
    profile: str = "static",
    data_root: Path = ROOT / "data" / "gestures",
) -> dict[str, Any]:
    profiles = ["static", "dynamic"] if profile == "both" else [profile]
    return {
        "mode": "ml",
        "description": (
            "Feature extraction plus the same predict/predict_proba calls used live"
        ),
        "profiles": [
            _benchmark_ml_profile(
                item,
                iterations=max(1, int(iterations)),
                warmup=max(0, int(warmup)),
                data_root=data_root,
            )
            for item in profiles
        ],
    }


def _prepare_rgb(frame_bgr: np.ndarray, max_width: int) -> np.ndarray:
    import cv2

    frame_bgr = cv2.flip(frame_bgr, 1)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    if width > max_width:
        scale = float(max_width) / float(width)
        rgb = cv2.resize(
            rgb,
            (max_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        rgb = np.ascontiguousarray(rgb)
    return rgb


def _new_infer() -> Any:
    from app.gesture_online_infer import GestureOnlineInfer

    infer = GestureOnlineInfer(window=30)
    if not infer.ready:
        raise RuntimeError(f"GestureOnlineInfer is not ready: {infer.init_error}")
    return infer


def benchmark_image(image_path: Path, iterations: int, max_width: int) -> dict[str, Any]:
    import cv2

    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    infer = _new_infer()
    for _ in range(20):
        infer.process_frame_rgb(_prepare_rgb(frame_bgr, max_width))

    preprocess_ms: list[float] = []
    infer_ms: list[float] = []
    pipeline_ms: list[float] = []
    labels: list[str] = []
    confidences: list[float] = []
    hand_frames = 0

    for _ in range(iterations):
        start = time.perf_counter_ns()
        rgb = _prepare_rgb(frame_bgr, max_width)
        after_preprocess = time.perf_counter_ns()
        out = infer.process_frame_rgb(rgb)
        end = time.perf_counter_ns()

        landmarks_json = out.get("landmarks_json") or "[]"
        if landmarks_json != "[]":
            hand_frames += 1
        labels.append(out.get("label") or "")
        confidences.append(float(out.get("confidence") or 0.0))
        preprocess_ms.append((after_preprocess - start) / 1_000_000)
        infer_ms.append((end - after_preprocess) / 1_000_000)
        pipeline_ms.append((end - start) / 1_000_000)

    infer.close()
    return {
        "mode": "image",
        "description": "BGR2RGB + resize + MediaPipe + features + active classifier",
        "image": str(image_path),
        "iterations": int(iterations),
        "original_size": [int(frame_bgr.shape[1]), int(frame_bgr.shape[0])],
        "processed_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "preprocess_latency": _summary(preprocess_ms),
        "inference_latency": _summary(infer_ms),
        "pipeline_latency": _summary(pipeline_ms),
        "hand_detected_frames": hand_frames,
        "labels_seen": sorted({label for label in labels if label}),
        "avg_confidence": round(float(np.mean(confidences)), 3),
    }


def benchmark_camera(
    camera_index: int,
    frames: int,
    warmup: int,
    max_width: int,
    width: int,
    height: int,
    fps: int,
) -> dict[str, Any]:
    import cv2

    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else 0
    cap = cv2.VideoCapture(int(camera_index), backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(int(camera_index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))
    if not cap.isOpened():
        raise RuntimeError("Camera is not opened. Check macOS Camera permission.")

    infer = _new_infer()

    for _ in range(max(0, warmup)):
        ok, frame_bgr = cap.read()
        if ok:
            infer.process_frame_rgb(_prepare_rgb(frame_bgr, max_width))

    read_ms: list[float] = []
    preprocess_ms: list[float] = []
    infer_ms: list[float] = []
    pipeline_ms: list[float] = []
    with_read_ms: list[float] = []
    labels: list[str] = []
    confidences: list[float] = []
    hand_frames = 0
    processed_size = [0, 0]

    while len(pipeline_ms) < frames:
        start = time.perf_counter_ns()
        ok, frame_bgr = cap.read()
        after_read = time.perf_counter_ns()
        if not ok:
            continue

        rgb = _prepare_rgb(frame_bgr, max_width)
        after_preprocess = time.perf_counter_ns()
        out = infer.process_frame_rgb(rgb)
        end = time.perf_counter_ns()

        processed_size = [int(rgb.shape[1]), int(rgb.shape[0])]
        landmarks_json = out.get("landmarks_json") or "[]"
        if landmarks_json != "[]":
            hand_frames += 1
        labels.append(out.get("label") or "")
        confidences.append(float(out.get("confidence") or 0.0))

        read_ms.append((after_read - start) / 1_000_000)
        preprocess_ms.append((after_preprocess - after_read) / 1_000_000)
        infer_ms.append((end - after_preprocess) / 1_000_000)
        pipeline_ms.append((end - after_read) / 1_000_000)
        with_read_ms.append((end - start) / 1_000_000)

    cap.release()
    infer.close()

    return {
        "mode": "camera",
        "description": "OpenCV camera + BGR2RGB + resize + MediaPipe + active classifier",
        "frames": len(pipeline_ms),
        "camera_index": int(camera_index),
        "requested_camera": {"width": int(width), "height": int(height), "fps": int(fps)},
        "processed_size": processed_size,
        "read_latency": _summary(read_ms),
        "preprocess_latency": _summary(preprocess_ms),
        "inference_latency": _summary(infer_ms),
        "pipeline_latency": _summary(pipeline_ms),
        "with_camera_read_latency": _summary(with_read_ms),
        "hand_detected_frames": hand_frames,
        "labels_seen": sorted({label for label in labels if label}),
        "avg_confidence": round(float(np.mean(confidences)), 3),
        "note": (
            "pipeline_latency excludes cap.read blocking; with_camera_read_latency includes "
            "waiting for the next camera frame."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ml", "image", "camera"], default="ml")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument(
        "--profile",
        choices=["static", "dynamic", "both"],
        default="static",
        help="Production model profile for --mode ml.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "gestures",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "app" / "flet_app" / "runtime_cache" / "live_a.jpg",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "ml":
        payload = benchmark_ml(
            args.iterations,
            warmup=args.warmup,
            profile=args.profile,
            data_root=args.data_root,
        )
    elif args.mode == "image":
        payload = benchmark_image(args.image, args.iterations, args.max_width)
    else:
        payload = benchmark_camera(
            camera_index=args.camera_index,
            frames=args.frames,
            warmup=args.warmup,
            max_width=args.max_width,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    _write_or_print(payload, args.output)


if __name__ == "__main__":
    main()
