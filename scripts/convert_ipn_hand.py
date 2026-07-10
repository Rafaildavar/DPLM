"""Convert a small IPN Hand subset into GestureBind landmark samples.

The converter intentionally produces experiment data only:

    data/external/ipn_hand/<mapped_label>/sample_ipn_*.npy

It does not touch production models. IPN labels are mapped through
``configs/ipn_hand_mapping.json`` so public classes do not accidentally become
GestureBind commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv.hand_landmarker import HandLandmarkerVideo, normalize_landmarks  # noqa: E402

DEFAULT_MAPPING_PATH = PROJECT_ROOT / "configs" / "ipn_hand_mapping.json"
DEFAULT_IPN_ROOT = PROJECT_ROOT / "data" / "raw" / "ipn_hand"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "external" / "ipn_hand"
DEFAULT_JSON_OUT = PROJECT_ROOT / "docs" / "experiments" / "ipn_conversion_report.json"
DEFAULT_MD_OUT = PROJECT_ROOT / "docs" / "experiments" / "ipn_conversion_report.md"
DEFAULT_MLFLOW_URI = "sqlite:///mlflow.db"
ANNOTATION_CANDIDATE_NAMES = (
    "Annot_List.txt",
    "ipnall.json",
    "Annot_TrainList.txt",
    "Annot_TestList.txt",
)

LandmarkExtractor = Callable[[Sequence[Path]], tuple[np.ndarray, int, int]]


@dataclass(frozen=True)
class MappingEntry:
    include: bool
    target_label: str
    role: str
    notes: str = ""


@dataclass(frozen=True)
class IpnSegment:
    video_id: str
    original_label: str
    start_frame: int
    end_frame: int
    target_label: str
    role: str
    include: bool


@dataclass(frozen=True)
class ConvertedIpnSample:
    status: str
    path: str
    video_id: str
    original_label: str
    target_label: str
    role: str
    start_frame: int
    end_frame: int
    requested_frames: int
    detected_frames: int
    saved_frames: int
    detection_rate: float
    reason: str = ""


@dataclass(frozen=True)
class IpnConversionReport:
    generated_at: float
    status: str
    ipn_root: str
    frames_root: str
    annotation_path: str
    mapping_path: str
    out_root: str
    limit_per_target_label: int
    max_frames_per_segment: int
    min_detected_frame_ratio: float
    segments_found: int
    segments_included: int
    converted_samples: int
    skipped_samples: int
    detection_rate: float
    labels: dict[str, int]
    expectations: dict[str, str]
    samples: list[ConvertedIpnSample] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _resolve_path(path: Path | str, *, base: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def _slug(value: str) -> str:
    clean = str(value or "").strip().lower()
    clean = re.sub(r"[^a-z0-9а-яё_-]+", "_", clean, flags=re.IGNORECASE)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_ipn_annotation(ipn_root: Path = DEFAULT_IPN_ROOT) -> Path:
    """Find the most useful IPN annotation file in a downloaded bundle."""
    ipn_root = _resolve_path(ipn_root)
    preferred = ipn_root / "annotations" / "ipnall.json"
    if preferred.exists():
        return preferred
    for name in ANNOTATION_CANDIDATE_NAMES:
        matches = sorted(ipn_root.glob(f"**/{name}"))
        if matches:
            return matches[0]
    return preferred


def load_ipn_mapping(path: Path = DEFAULT_MAPPING_PATH) -> dict[str, MappingEntry]:
    raw = _read_json(path)
    default_raw = raw.get("default") or {}
    default = MappingEntry(
        include=bool(default_raw.get("include", False)),
        target_label=str(default_raw.get("target_label") or "negative_external_ipn_dynamic"),
        role=str(default_raw.get("role") or "unknown_dynamic"),
        notes=str(default_raw.get("notes") or ""),
    )
    mapping: dict[str, MappingEntry] = {"*": default}
    for raw_label, payload in (raw.get("labels") or {}).items():
        if not isinstance(payload, dict):
            continue
        label = str(raw_label).strip().upper()
        if not label:
            continue
        mapping[label] = MappingEntry(
            include=bool(payload.get("include", default.include)),
            target_label=str(payload.get("target_label") or default.target_label).strip(),
            role=str(payload.get("role") or default.role).strip(),
            notes=str(payload.get("notes") or ""),
        )
    return mapping


def _mapping_for_label(mapping: dict[str, MappingEntry], label: str) -> MappingEntry:
    return mapping.get(str(label).strip().upper()) or mapping.get("*") or MappingEntry(
        include=False,
        target_label="negative_external_ipn_dynamic",
        role="unknown_dynamic",
    )


def _video_id_from_key(key: str) -> str:
    clean = str(key or "").strip()
    if "^" in clean:
        clean = clean.split("^", 1)[0]
    clean = clean.lstrip("./")
    return Path(clean).name if "/" in clean else clean


def load_ipn_segments(
    annotation_path: Path,
    *,
    mapping: dict[str, MappingEntry],
) -> list[IpnSegment]:
    if annotation_path.suffix.lower() == ".json":
        return _load_json_segments(annotation_path, mapping=mapping)
    return _load_text_segments(annotation_path, mapping=mapping)


def _load_json_segments(
    annotation_path: Path,
    *,
    mapping: dict[str, MappingEntry],
) -> list[IpnSegment]:
    raw = _read_json(annotation_path)
    segments: list[IpnSegment] = []
    database = raw.get("database")
    if isinstance(database, dict):
        iterable = database.items()
    elif isinstance(raw.get("annotations"), list):
        iterable = ((str(i), item) for i, item in enumerate(raw["annotations"]))
    else:
        iterable = []

    for key, value in iterable:
        annotation = value.get("annotations") if isinstance(value, dict) else value
        if not isinstance(annotation, dict):
            continue
        label = str(annotation.get("label") or annotation.get("class") or "").strip()
        if not label:
            continue
        try:
            start = int(annotation.get("start_frame"))
            end = int(annotation.get("end_frame"))
        except (TypeError, ValueError):
            continue
        video_id = _video_id_from_key(str(key))
        if not video_id and isinstance(value, dict):
            video_id = _video_id_from_key(str(value.get("video") or value.get("video_id") or ""))
        if not video_id:
            continue
        segments.append(_segment(video_id, label, start, end, mapping))
    return segments


def _load_text_segments(
    annotation_path: Path,
    *,
    mapping: dict[str, MappingEntry],
) -> list[IpnSegment]:
    csv_segments = _load_csv_like_segments(annotation_path, mapping=mapping)
    if csv_segments:
        return csv_segments

    segments: list[IpnSegment] = []
    for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 4:
            continue
        video_id = _video_id_from_key(parts[0])
        label = parts[1]
        try:
            start = int(parts[2])
            end = int(parts[3])
        except ValueError:
            continue
        segments.append(_segment(video_id, label, start, end, mapping))
    return segments


def _load_csv_like_segments(
    annotation_path: Path,
    *,
    mapping: dict[str, MappingEntry],
) -> list[IpnSegment]:
    """Read official IPN annotation text exported as CSV.

    The downloadable annotation file uses a ``.txt`` extension but has columns:
    ``video,label,id,t_start,t_end,frames``.
    """
    try:
        with annotation_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return []
    segments: list[IpnSegment] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        video_id = _video_id_from_key(str(row.get("video") or row.get("Video") or ""))
        label = str(row.get("label") or row.get("Label") or "").strip()
        if not video_id or not label:
            continue
        try:
            start = int(row.get("t_start") or row.get("start_frame") or row.get("start"))
            end = int(row.get("t_end") or row.get("end_frame") or row.get("end"))
        except (TypeError, ValueError):
            continue
        segments.append(_segment(video_id, label, start, end, mapping))
    return segments


def _segment(
    video_id: str,
    label: str,
    start: int,
    end: int,
    mapping: dict[str, MappingEntry],
) -> IpnSegment:
    entry = _mapping_for_label(mapping, label)
    start_frame = max(1, int(start))
    end_frame = max(start_frame, int(end))
    return IpnSegment(
        video_id=str(video_id),
        original_label=str(label).strip().upper(),
        start_frame=start_frame,
        end_frame=end_frame,
        target_label=entry.target_label,
        role=entry.role,
        include=entry.include,
    )


def _candidate_frame_paths(frames_root: Path, video_id: str, frame_index: int) -> list[Path]:
    video_dir = frames_root / video_id
    return [
        video_dir / f"{video_id}_{frame_index:06d}.jpg",
        video_dir / f"{video_id}_{frame_index:05d}.jpg",
        video_dir / f"{frame_index:06d}.jpg",
        video_dir / f"{frame_index:05d}.jpg",
        video_dir / f"frame_{frame_index:06d}.jpg",
        video_dir / f"frame_{frame_index:05d}.jpg",
    ]


def frame_paths_for_segment(
    frames_root: Path,
    segment: IpnSegment,
    *,
    max_frames: int,
) -> list[Path]:
    start = int(segment.start_frame)
    end = int(segment.end_frame)
    all_indices = np.arange(start, end + 1, dtype=np.int64)
    limit = max(2, int(max_frames))
    if all_indices.size > limit:
        positions = np.linspace(0, all_indices.size - 1, limit)
        indices = [int(all_indices[int(round(pos))]) for pos in positions]
    else:
        indices = [int(item) for item in all_indices.tolist()]

    paths: list[Path] = []
    for frame_index in indices:
        candidates = _candidate_frame_paths(frames_root, segment.video_id, frame_index)
        for candidate in candidates:
            if candidate.exists():
                paths.append(candidate)
                break
    return paths


def _read_rgb_frame(path: Path) -> np.ndarray:
    import cv2

    frame_bgr = cv2.imread(str(path))
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError(f"cannot read frame: {path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _mediapipe_landmark_sequence(frame_paths: Sequence[Path]) -> tuple[np.ndarray, int, int]:
    rows: list[np.ndarray] = []
    requested = len(frame_paths)
    with HandLandmarkerVideo(num_hands=1) as detector:
        for frame_path in frame_paths:
            try:
                hands = detector.detect_for_video_rgb(_read_rgb_frame(Path(frame_path)))
            except Exception:
                hands = []
            if not hands:
                continue
            hand = max(hands, key=lambda item: float(item.score or 0.0))
            normalized = normalize_landmarks(hand.landmarks).reshape(-1)
            wrist = np.asarray(hand.landmarks[0], dtype=np.float32).reshape(2)
            rows.append(
                np.concatenate([normalized.astype(np.float32), wrist], axis=0).astype(
                    np.float32,
                    copy=False,
                )
            )
    if not rows:
        return np.zeros((0, 44), dtype=np.float32), requested, 0
    return np.stack(rows, axis=0).astype(np.float32, copy=False), requested, len(rows)


def _write_sample_metadata(path: Path, sample: ConvertedIpnSample) -> None:
    payload = {
        "schema_version": 1,
        "generated_by": "scripts.convert_ipn_hand",
        **asdict(sample),
    }
    path.with_suffix(".meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def convert_ipn_hand(
    *,
    ipn_root: Path = DEFAULT_IPN_ROOT,
    frames_root: Path | None = None,
    annotation_path: Path | None = None,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    out_root: Path = DEFAULT_OUT_ROOT,
    limit_per_target_label: int = 200,
    max_frames_per_segment: int = 60,
    min_detected_frame_ratio: float = 0.50,
    include_reference: bool = False,
    clean_auto: bool = True,
    extractor: LandmarkExtractor | None = None,
    skip_mediapipe_preflight: bool = False,
) -> IpnConversionReport:
    ipn_root = _resolve_path(ipn_root)
    frames_root = _resolve_path(frames_root or ipn_root / "frames")
    annotation_path = _resolve_path(annotation_path) if annotation_path else discover_ipn_annotation(ipn_root)
    mapping_path = _resolve_path(mapping_path)
    out_root = _resolve_path(out_root)
    warnings: list[str] = []
    expectations = {
        "production_model_unchanged": "met",
        "outputs_under_data_external": "not_checked",
        "mapped_to_negative_labels": "not_checked",
        "mlflow_conversion_report": "not_logged_yet",
    }

    if not mapping_path.exists():
        warnings.append(f"mapping file not found: {mapping_path}")
        return _empty_report(
            status="missing_mapping",
            ipn_root=ipn_root,
            frames_root=frames_root,
            annotation_path=annotation_path,
            mapping_path=mapping_path,
            out_root=out_root,
            limit_per_target_label=limit_per_target_label,
            max_frames_per_segment=max_frames_per_segment,
            min_detected_frame_ratio=min_detected_frame_ratio,
            expectations=expectations,
            warnings=warnings,
        )

    missing_inputs = []
    if not frames_root.exists():
        missing_inputs.append(f"frames root not found: {frames_root}")
    if not annotation_path.exists():
        missing_inputs.append(f"annotation file not found: {annotation_path}")
    if missing_inputs:
        warnings.extend(missing_inputs)
        return _empty_report(
            status="missing_input",
            ipn_root=ipn_root,
            frames_root=frames_root,
            annotation_path=annotation_path,
            mapping_path=mapping_path,
            out_root=out_root,
            limit_per_target_label=limit_per_target_label,
            max_frames_per_segment=max_frames_per_segment,
            min_detected_frame_ratio=min_detected_frame_ratio,
            expectations=expectations,
            warnings=warnings,
        )

    mapping = load_ipn_mapping(mapping_path)
    segments = load_ipn_segments(annotation_path, mapping=mapping)
    included = [
        segment
        for segment in segments
        if segment.include or (include_reference and segment.role == "validation_reference")
    ]
    if extractor is None and not skip_mediapipe_preflight:
        ok, detail = mediapipe_cli_available()
        if not ok:
            warnings.append(f"MediaPipe CLI preflight failed: {detail}")
            warnings.append("no IPN samples were converted")
            return IpnConversionReport(
                generated_at=time.time(),
                status="mediapipe_unavailable",
                ipn_root=str(ipn_root),
                frames_root=str(frames_root),
                annotation_path=str(annotation_path),
                mapping_path=str(mapping_path),
                out_root=str(out_root),
                limit_per_target_label=int(limit_per_target_label),
                max_frames_per_segment=int(max_frames_per_segment),
                min_detected_frame_ratio=float(min_detected_frame_ratio),
                segments_found=len(segments),
                segments_included=len(included),
                converted_samples=0,
                skipped_samples=0,
                detection_rate=0.0,
                labels={},
                expectations={
                    **expectations,
                    "outputs_under_data_external": "not_written",
                    "mapped_to_negative_labels": "not_checked",
                },
                samples=[],
                warnings=warnings,
            )
    if clean_auto and out_root.exists():
        for old_path in out_root.glob("**/sample_ipn_*.npy"):
            old_path.unlink(missing_ok=True)
            old_path.with_suffix(".meta.json").unlink(missing_ok=True)

    selected_counts: dict[str, int] = {}
    converted: list[ConvertedIpnSample] = []
    extractor_fn = extractor or _mediapipe_landmark_sequence
    for segment in included:
        current_count = selected_counts.get(segment.target_label, 0)
        if current_count >= max(1, int(limit_per_target_label)):
            continue
        frame_paths = frame_paths_for_segment(
            frames_root,
            segment,
            max_frames=max_frames_per_segment,
        )
        if not frame_paths:
            converted.append(
                _sample_result(
                    segment,
                    status="skipped",
                    path="",
                    requested=0,
                    detected=0,
                    saved=0,
                    reason="no frame files found for segment",
                )
            )
            continue
        try:
            sequence, requested, detected = extractor_fn(frame_paths)
        except Exception as exc:
            converted.append(
                _sample_result(
                    segment,
                    status="skipped",
                    path="",
                    requested=len(frame_paths),
                    detected=0,
                    saved=0,
                    reason=f"landmark extraction failed: {exc}",
                )
            )
            continue

        detection_rate = float(detected / requested) if requested else 0.0
        if sequence.shape[0] < 2 or detection_rate < float(min_detected_frame_ratio):
            converted.append(
                _sample_result(
                    segment,
                    status="skipped",
                    path="",
                    requested=requested,
                    detected=detected,
                    saved=int(sequence.shape[0]),
                    reason="insufficient detected hand frames",
                )
            )
            continue

        out_dir = out_root / segment.target_label
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_index = selected_counts.get(segment.target_label, 0)
        out_path = out_dir / (
            f"sample_ipn_{_slug(segment.original_label)}_"
            f"{_slug(segment.video_id)}_{sample_index:04d}.npy"
        )
        np.save(out_path, sequence.astype(np.float32, copy=False))
        sample = _sample_result(
            segment,
            status="converted",
            path=str(out_path),
            requested=requested,
            detected=detected,
            saved=int(sequence.shape[0]),
            reason="",
        )
        _write_sample_metadata(out_path, sample)
        converted.append(sample)
        selected_counts[segment.target_label] = sample_index + 1

    converted_count = sum(1 for item in converted if item.status == "converted")
    skipped_count = sum(1 for item in converted if item.status != "converted")
    requested_total = sum(item.requested_frames for item in converted)
    detected_total = sum(item.detected_frames for item in converted)
    detection_rate = float(detected_total / requested_total) if requested_total else 0.0
    labels = {
        label: sum(1 for item in converted if item.status == "converted" and item.target_label == label)
        for label in sorted({item.target_label for item in converted})
    }
    expectations.update(
        {
            "outputs_under_data_external": (
                "met" if converted_count == 0 or str(out_root).endswith("data/external/ipn_hand") else "check"
            ),
            "mapped_to_negative_labels": (
                "met"
                if all(
                    item.target_label.startswith("negative_external_")
                    for item in converted
                    if item.status == "converted"
                )
                else "check_reference_outputs"
            ),
        }
    )
    status = "ok" if converted_count else "empty"
    if converted_count == 0:
        warnings.append("no IPN samples were converted")
    return IpnConversionReport(
        generated_at=time.time(),
        status=status,
        ipn_root=str(ipn_root),
        frames_root=str(frames_root),
        annotation_path=str(annotation_path),
        mapping_path=str(mapping_path),
        out_root=str(out_root),
        limit_per_target_label=int(limit_per_target_label),
        max_frames_per_segment=int(max_frames_per_segment),
        min_detected_frame_ratio=float(min_detected_frame_ratio),
        segments_found=len(segments),
        segments_included=len(included),
        converted_samples=converted_count,
        skipped_samples=skipped_count,
        detection_rate=round(detection_rate, 4),
        labels=dict(sorted(labels.items())),
        expectations=expectations,
        samples=converted,
        warnings=warnings,
    )


def _empty_report(
    *,
    status: str,
    ipn_root: Path,
    frames_root: Path,
    annotation_path: Path,
    mapping_path: Path,
    out_root: Path,
    limit_per_target_label: int,
    max_frames_per_segment: int,
    min_detected_frame_ratio: float,
    expectations: dict[str, str],
    warnings: list[str],
) -> IpnConversionReport:
    return IpnConversionReport(
        generated_at=time.time(),
        status=status,
        ipn_root=str(ipn_root),
        frames_root=str(frames_root),
        annotation_path=str(annotation_path),
        mapping_path=str(mapping_path),
        out_root=str(out_root),
        limit_per_target_label=int(limit_per_target_label),
        max_frames_per_segment=int(max_frames_per_segment),
        min_detected_frame_ratio=float(min_detected_frame_ratio),
        segments_found=0,
        segments_included=0,
        converted_samples=0,
        skipped_samples=0,
        detection_rate=0.0,
        labels={},
        expectations=expectations,
        samples=[],
        warnings=warnings,
    )


def _sample_result(
    segment: IpnSegment,
    *,
    status: str,
    path: str,
    requested: int,
    detected: int,
    saved: int,
    reason: str,
) -> ConvertedIpnSample:
    return ConvertedIpnSample(
        status=status,
        path=path,
        video_id=segment.video_id,
        original_label=segment.original_label,
        target_label=segment.target_label,
        role=segment.role,
        start_frame=segment.start_frame,
        end_frame=segment.end_frame,
        requested_frames=int(requested),
        detected_frames=int(detected),
        saved_frames=int(saved),
        detection_rate=round(float(detected / requested), 4) if requested else 0.0,
        reason=reason,
    )


def build_markdown_report(report: IpnConversionReport) -> str:
    lines = [
        "# IPN Hand Conversion Report",
        "",
        f"Generated at: `{report.generated_at:.3f}`",
        "",
        "## Summary",
        "",
        f"- status: `{report.status}`",
        f"- IPN root: `{report.ipn_root}`",
        f"- frames root: `{report.frames_root}`",
        f"- annotation path: `{report.annotation_path}`",
        f"- output root: `{report.out_root}`",
        f"- segments found/included: `{report.segments_found}` / `{report.segments_included}`",
        f"- converted/skipped: `{report.converted_samples}` / `{report.skipped_samples}`",
        f"- detection rate: `{report.detection_rate:.4f}`",
        "",
        "## Expectation Check",
        "",
        "| Expectation | Result |",
        "|---|---|",
    ]
    for key, value in sorted(report.expectations.items()):
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Labels",
            "",
            "| Target label | Converted samples |",
            "|---|---:|",
        ]
    )
    if report.labels:
        for label, count in sorted(report.labels.items()):
            lines.append(f"| `{label}` | {count} |")
    else:
        lines.append("| none | 0 |")

    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in report.warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Sample Preview",
            "",
            "| Status | Target | IPN label | Video | Frames | Detection | Reason |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for sample in report.samples[:30]:
        lines.append(
            "| "
            + " | ".join(
                [
                    sample.status,
                    f"`{sample.target_label}`",
                    f"`{sample.original_label}`",
                    f"`{sample.video_id}`",
                    str(sample.saved_frames),
                    f"{sample.detection_rate:.4f}",
                    sample.reason.replace("|", "/"),
                ]
            )
            + " |"
        )
    if not report.samples:
        lines.append("| none | none | none | none | 0 | 0.0000 | no samples processed |")

    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "If status is `missing_input`, place IPN frames under",
            "`data/raw/ipn_hand/frames` and annotations as `Annot_List.txt`",
            "inside `data/raw/ipn_hand`, or pass `--frames-root` and",
            "`--annotation` explicitly.",
            "Then rerun:",
            "",
            "```bash",
            "PYTHON=.venv/bin/python make ipn-convert",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(
    report: IpnConversionReport,
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


def mediapipe_cli_available(timeout_seconds: int = 60) -> tuple[bool, str]:
    command = [
        sys.executable,
        "-c",
        (
            "import numpy as np; "
            "from cv.hand_landmarker import HandLandmarkerVideo; "
            "det=HandLandmarkerVideo(num_hands=1); "
            "det.detect_for_video_rgb(np.zeros((32,32,3), dtype=np.uint8)); "
            "det.close(); "
            "print('ok')"
        ),
    ]
    env = {
        **dict(os.environ),
        "MPLCONFIGDIR": "/private/tmp/gesturebind_mpl",
        "MEDIAPIPE_DISABLE_GPU": "1",
    }
    try:
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=int(timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_seconds}s"
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode == 0:
        return True, "ok"
    tail = "\n".join(output.strip().splitlines()[-8:])
    return False, f"exit={result.returncode}; {tail}"


def log_mlflow_conversion(
    report: IpnConversionReport,
    *,
    json_out: Path,
    md_out: Path,
    mapping_path: Path,
    experiment: str,
    tracking_uri: str,
) -> bool:
    experiment = str(experiment or "").strip()
    if not experiment:
        return False
    try:
        import mlflow
    except Exception as exc:
        print(f"[w] MLflow unavailable for IPN conversion: {exc}")
        return False
    try:
        mlflow.set_tracking_uri(str(tracking_uri))
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name="ipn-conversion"):
            mlflow.set_tags(
                {
                    "run_kind": "ipn_conversion",
                    "source": "scripts.convert_ipn_hand",
                    "status": report.status,
                }
            )
            mlflow.log_params(
                {
                    "ipn_root": report.ipn_root,
                    "frames_root": report.frames_root,
                    "annotation_path": report.annotation_path,
                    "mapping_path": report.mapping_path,
                    "out_root": report.out_root,
                    "limit_per_target_label": report.limit_per_target_label,
                    "max_frames_per_segment": report.max_frames_per_segment,
                    "min_detected_frame_ratio": report.min_detected_frame_ratio,
                }
            )
            mlflow.log_metrics(
                {
                    "segments_found": float(report.segments_found),
                    "segments_included": float(report.segments_included),
                    "converted_samples": float(report.converted_samples),
                    "skipped_samples": float(report.skipped_samples),
                    "detection_rate": float(report.detection_rate),
                    "target_label_count": float(len(report.labels)),
                    "warning_count": float(len(report.warnings)),
                }
            )
            for artifact in (json_out, md_out, mapping_path):
                candidate = _resolve_path(artifact)
                if candidate.exists():
                    mlflow.log_artifact(str(candidate), artifact_path="ipn_conversion")
        return True
    except Exception as exc:
        print(f"[w] MLflow IPN conversion logging failed: {exc}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipn-root", type=Path, default=DEFAULT_IPN_ROOT)
    parser.add_argument("--frames-root", type=Path, default=None)
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--limit-per-target-label", type=int, default=200)
    parser.add_argument("--max-frames-per-segment", type=int, default=60)
    parser.add_argument("--min-detected-frame-ratio", type=float, default=0.50)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--keep-existing-auto", action="store_true")
    parser.add_argument("--skip-mediapipe-preflight", action="store_true")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--mlflow-experiment", default="GestureBind")
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_MLFLOW_URI)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_ipn_hand(
        ipn_root=args.ipn_root,
        frames_root=args.frames_root,
        annotation_path=args.annotation,
        mapping_path=args.mapping,
        out_root=args.out_root,
        limit_per_target_label=int(args.limit_per_target_label),
        max_frames_per_segment=int(args.max_frames_per_segment),
        min_detected_frame_ratio=float(args.min_detected_frame_ratio),
        include_reference=bool(args.include_reference),
        clean_auto=not bool(args.keep_existing_auto),
        skip_mediapipe_preflight=bool(args.skip_mediapipe_preflight),
    )
    write_report(report, json_out=args.json_out, md_out=args.md_out)
    logged = log_mlflow_conversion(
        report,
        json_out=args.json_out,
        md_out=args.md_out,
        mapping_path=args.mapping,
        experiment=str(args.mlflow_experiment),
        tracking_uri=str(args.mlflow_tracking_uri),
    )
    report.expectations["mlflow_conversion_report"] = "met" if logged else "not_logged"
    write_report(report, json_out=args.json_out, md_out=args.md_out)
    print(build_markdown_report(report))
    print(f"[✓] JSON: {_resolve_path(args.json_out)}")
    print(f"[✓] Markdown: {_resolve_path(args.md_out)}")


if __name__ == "__main__":
    main()
