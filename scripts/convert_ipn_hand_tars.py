"""Convert IPN Hand frame tar archives into compact GestureBind samples.

The IPN frame archives are too large to extract in this workspace. This script
streams selected JPEG frames directly from ``frames*.tar`` and writes only the
small landmark sequences used by GestureBind experiments.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from cv.hand_landmarker import HandLandmarkerVideo, normalize_landmarks
from scripts.convert_ipn_hand import (
    DEFAULT_IPN_ROOT,
    DEFAULT_MAPPING_PATH,
    DEFAULT_MLFLOW_URI,
    IpnSegment,
    _resolve_path,
    discover_ipn_annotation,
    load_ipn_mapping,
    load_ipn_segments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAR_GLOB = str(PROJECT_ROOT / "data" / "frames*.tar")
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "external" / "ipn_hand_tar"
DEFAULT_JSON_OUT = PROJECT_ROOT / "docs" / "experiments" / "ipn_tar_conversion_report.json"
DEFAULT_MD_OUT = PROJECT_ROOT / "docs" / "experiments" / "ipn_tar_conversion_report.md"


@dataclass(frozen=True)
class TarIpnSample:
    status: str
    path: str
    video_id: str
    original_label: str
    target_label: str
    role: str
    start_frame: int
    end_frame: int
    requested_frames: int
    found_frames: int
    detected_frames: int
    saved_frames: int
    detection_rate: float
    archive: str = ""
    reason: str = ""


@dataclass(frozen=True)
class TarIpnConversionReport:
    generated_at: float
    status: str
    tar_glob: str
    tar_files: list[str]
    annotation_path: str
    mapping_path: str
    out_root: str
    limit_total: int
    max_per_original_label: int
    max_frames_per_segment: int
    min_detected_frame_ratio: float
    videos_in_archives: int
    segments_found: int
    candidate_segments: int
    selected_segments: int
    candidate_original_labels: dict[str, int]
    selected_original_labels: dict[str, int]
    converted_samples: int
    skipped_samples: int
    detection_rate: float
    labels: dict[str, int]
    original_labels: dict[str, int]
    samples: list[TarIpnSample] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def sample_frame_indices(start: int, end: int, *, max_frames: int) -> list[int]:
    start_frame = max(1, int(start))
    end_frame = max(start_frame, int(end))
    all_indices = np.arange(start_frame, end_frame + 1, dtype=np.int64)
    limit = max(2, int(max_frames))
    if all_indices.size <= limit:
        return [int(item) for item in all_indices.tolist()]
    positions = np.linspace(0, all_indices.size - 1, limit)
    return [int(all_indices[int(round(pos))]) for pos in positions]


def discover_tar_files(tar_glob: str) -> list[Path]:
    return sorted(Path(path) for path in glob.glob(str(tar_glob)) if Path(path).is_file())


def archive_video_map(tar_files: Sequence[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for tar_path in tar_files:
        with tarfile.open(tar_path) as archive:
            for member in archive:
                parts = member.name.split("/")
                if len(parts) >= 2 and parts[0] == "frames" and parts[1]:
                    out.setdefault(parts[1], tar_path)
    return out


def select_segments(
    segments: Iterable[IpnSegment],
    *,
    video_to_tar: dict[str, Path],
    limit_total: int,
    max_per_original_label: int,
    include_reference: bool,
) -> list[IpnSegment]:
    selected: list[IpnSegment] = []
    per_label: Counter[str] = Counter()
    candidates = [
        segment
        for segment in segments
        if segment.video_id in video_to_tar
        and (segment.include or (include_reference and segment.role == "validation_reference"))
    ]
    for segment in candidates:
        if len(selected) >= max(1, int(limit_total)):
            break
        if per_label[segment.original_label] >= max(1, int(max_per_original_label)):
            continue
        selected.append(segment)
        per_label[segment.original_label] += 1
    return selected


def build_member_index(
    selected: Sequence[IpnSegment],
    *,
    video_to_tar: dict[str, Path],
    max_frames_per_segment: int,
) -> dict[str, dict[int, tuple[Path, str]]]:
    wanted: dict[Path, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for segment in selected:
        tar_path = video_to_tar.get(segment.video_id)
        if tar_path is None:
            continue
        wanted[tar_path][segment.video_id].update(
            sample_frame_indices(
                segment.start_frame,
                segment.end_frame,
                max_frames=max_frames_per_segment,
            )
        )

    index: dict[str, dict[int, tuple[Path, str]]] = defaultdict(dict)
    for tar_path, videos in wanted.items():
        with tarfile.open(tar_path) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                parsed = _parse_member_frame(member.name)
                if parsed is None:
                    continue
                video_id, frame_index = parsed
                if video_id not in videos or frame_index not in videos[video_id]:
                    continue
                index[video_id][frame_index] = (tar_path, member.name)
    return index


def convert_ipn_tars(
    *,
    tar_glob: str = DEFAULT_TAR_GLOB,
    ipn_root: Path = DEFAULT_IPN_ROOT,
    annotation_path: Path | None = None,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    out_root: Path = DEFAULT_OUT_ROOT,
    limit_total: int = 240,
    max_per_original_label: int = 40,
    max_frames_per_segment: int = 48,
    min_detected_frame_ratio: float = 0.50,
    include_reference: bool = False,
    clean_auto: bool = True,
    skip_mediapipe_preflight: bool = False,
) -> TarIpnConversionReport:
    ipn_root = _resolve_path(ipn_root)
    annotation_path = (
        _resolve_path(annotation_path)
        if annotation_path
        else discover_ipn_annotation(ipn_root)
    )
    mapping_path = _resolve_path(mapping_path)
    out_root = _resolve_path(out_root)
    tar_files = discover_tar_files(tar_glob)
    warnings: list[str] = []
    if not tar_files:
        warnings.append(f"no tar files matched: {tar_glob}")
    if not annotation_path.exists():
        warnings.append(f"annotation file not found: {annotation_path}")
    if not mapping_path.exists():
        warnings.append(f"mapping file not found: {mapping_path}")
    if warnings:
        return _report(
            status="missing_input",
            tar_glob=tar_glob,
            tar_files=tar_files,
            annotation_path=annotation_path,
            mapping_path=mapping_path,
            out_root=out_root,
            limit_total=limit_total,
            max_per_original_label=max_per_original_label,
            max_frames_per_segment=max_frames_per_segment,
            min_detected_frame_ratio=min_detected_frame_ratio,
            videos_in_archives=0,
            segments_found=0,
            candidate_segments=0,
            selected_segments=0,
            candidate_original_labels={},
            selected_original_labels={},
            samples=[],
            warnings=warnings,
        )

    mapping = load_ipn_mapping(mapping_path)
    segments = load_ipn_segments(annotation_path, mapping=mapping)
    video_to_tar = archive_video_map(tar_files)
    candidates = [
        segment
        for segment in segments
        if segment.video_id in video_to_tar
        and (segment.include or (include_reference and segment.role == "validation_reference"))
    ]
    selected = select_segments(
        segments,
        video_to_tar=video_to_tar,
        limit_total=limit_total,
        max_per_original_label=max_per_original_label,
        include_reference=include_reference,
    )
    if not skip_mediapipe_preflight:
        ok, detail = mediapipe_cli_available()
        if not ok:
            warnings.append(f"MediaPipe CLI preflight failed: {detail}")
            return _report(
                status="mediapipe_unavailable",
                tar_glob=tar_glob,
                tar_files=tar_files,
                annotation_path=annotation_path,
                mapping_path=mapping_path,
                out_root=out_root,
                limit_total=limit_total,
                max_per_original_label=max_per_original_label,
                max_frames_per_segment=max_frames_per_segment,
                min_detected_frame_ratio=min_detected_frame_ratio,
                videos_in_archives=len(video_to_tar),
                segments_found=len(segments),
                candidate_segments=len(candidates),
                selected_segments=len(selected),
                candidate_original_labels=dict(sorted(Counter(item.original_label for item in candidates).items())),
                selected_original_labels=dict(sorted(Counter(item.original_label for item in selected).items())),
                samples=[],
                warnings=warnings,
            )
    member_index = build_member_index(
        selected,
        video_to_tar=video_to_tar,
        max_frames_per_segment=max_frames_per_segment,
    )

    if clean_auto and out_root.exists():
        for old_path in out_root.glob("**/sample_ipn_tar_*.npy"):
            old_path.unlink(missing_ok=True)
            old_path.with_suffix(".meta.json").unlink(missing_ok=True)

    samples: list[TarIpnSample] = []
    tar_cache: dict[Path, tarfile.TarFile] = {}
    try:
        with HandLandmarkerVideo(num_hands=1) as detector:
            converted_by_target: Counter[str] = Counter()
            for segment in selected:
                frame_indices = sample_frame_indices(
                    segment.start_frame,
                    segment.end_frame,
                    max_frames=max_frames_per_segment,
                )
                archive_name = str(video_to_tar.get(segment.video_id) or "")
                rows: list[np.ndarray] = []
                found_frames = 0
                for frame_index in frame_indices:
                    source = member_index.get(segment.video_id, {}).get(frame_index)
                    if source is None:
                        continue
                    found_frames += 1
                    tar_path, member_name = source
                    try:
                        frame_rgb = _read_tar_frame_rgb(
                            tar_path,
                            member_name,
                            tar_cache=tar_cache,
                        )
                        hands = detector.detect_for_video_rgb(frame_rgb)
                    except Exception:
                        hands = []
                    if not hands:
                        continue
                    hand = max(hands, key=lambda item: float(item.score or 0.0))
                    normalized = normalize_landmarks(hand.landmarks).reshape(-1)
                    wrist = np.asarray(hand.landmarks[0], dtype=np.float32).reshape(2)
                    rows.append(
                        np.concatenate(
                            [normalized.astype(np.float32), wrist],
                            axis=0,
                        ).astype(np.float32, copy=False)
                    )

                requested = len(frame_indices)
                detected = len(rows)
                detection_rate = float(detected / requested) if requested else 0.0
                if detected < 2 or detection_rate < float(min_detected_frame_ratio):
                    samples.append(
                        _tar_sample(
                            segment,
                            status="skipped",
                            path="",
                            requested=requested,
                            found=found_frames,
                            detected=detected,
                            saved=detected,
                            archive=archive_name,
                            reason="insufficient detected hand frames",
                        )
                    )
                    continue

                out_dir = out_root / segment.target_label
                out_dir.mkdir(parents=True, exist_ok=True)
                sample_index = converted_by_target[segment.target_label]
                out_path = out_dir / (
                    f"sample_ipn_tar_{_slug(segment.original_label)}_"
                    f"{_slug(segment.video_id)}_{sample_index:04d}.npy"
                )
                np.save(out_path, np.stack(rows, axis=0).astype(np.float32, copy=False))
                sample = _tar_sample(
                    segment,
                    status="converted",
                    path=str(out_path),
                    requested=requested,
                    found=found_frames,
                    detected=detected,
                    saved=detected,
                    archive=archive_name,
                    reason="",
                )
                _write_tar_sample_metadata(out_path, sample)
                samples.append(sample)
                converted_by_target[segment.target_label] += 1
    finally:
        for archive in tar_cache.values():
            archive.close()

    return _report(
        status="ok" if any(item.status == "converted" for item in samples) else "empty",
        tar_glob=tar_glob,
        tar_files=tar_files,
        annotation_path=annotation_path,
        mapping_path=mapping_path,
        out_root=out_root,
        limit_total=limit_total,
        max_per_original_label=max_per_original_label,
        max_frames_per_segment=max_frames_per_segment,
        min_detected_frame_ratio=min_detected_frame_ratio,
        videos_in_archives=len(video_to_tar),
        segments_found=len(segments),
        candidate_segments=len(candidates),
        selected_segments=len(selected),
        candidate_original_labels=dict(sorted(Counter(item.original_label for item in candidates).items())),
        selected_original_labels=dict(sorted(Counter(item.original_label for item in selected).items())),
        samples=samples,
        warnings=warnings,
    )


def build_markdown_report(report: TarIpnConversionReport) -> str:
    lines = [
        "# IPN Tar Conversion Report",
        "",
        f"Generated at: `{report.generated_at:.3f}`",
        "",
        "## Summary",
        "",
        f"- status: `{report.status}`",
        f"- tar files: `{len(report.tar_files)}`",
        f"- videos in archives: `{report.videos_in_archives}`",
        f"- annotation path: `{report.annotation_path}`",
        f"- output root: `{report.out_root}`",
        f"- segments found/candidate/selected: `{report.segments_found}` / `{report.candidate_segments}` / `{report.selected_segments}`",
        f"- converted/skipped: `{report.converted_samples}` / `{report.skipped_samples}`",
        f"- detection rate: `{report.detection_rate:.4f}`",
        "",
        "## Labels",
        "",
        "| Target label | Converted samples |",
        "|---|---:|",
    ]
    if report.labels:
        for label, count in sorted(report.labels.items()):
            lines.append(f"| `{label}` | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(["", "## Original IPN Labels", "", "| IPN label | Converted |", "|---|---:|"])
    if report.original_labels:
        for label, count in sorted(report.original_labels.items()):
            lines.append(f"| `{label}` | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(["", "## Candidate IPN Labels", "", "| IPN label | Candidate | Selected |", "|---|---:|---:|"])
    candidate_labels = set(report.candidate_original_labels) | set(report.selected_original_labels)
    if candidate_labels:
        for label in sorted(candidate_labels):
            lines.append(
                f"| `{label}` | {report.candidate_original_labels.get(label, 0)} | "
                f"{report.selected_original_labels.get(label, 0)} |"
            )
    else:
        lines.append("| none | 0 | 0 |")

    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in report.warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Sample Preview",
            "",
            "| Status | Target | IPN label | Video | Found | Detected | Detection | Reason |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for sample in report.samples[:40]:
        lines.append(
            "| "
            + " | ".join(
                [
                    sample.status,
                    f"`{sample.target_label}`",
                    f"`{sample.original_label}`",
                    f"`{sample.video_id}`",
                    str(sample.found_frames),
                    str(sample.detected_frames),
                    f"{sample.detection_rate:.4f}",
                    sample.reason.replace("|", "/"),
                ]
            )
            + " |"
        )
    if not report.samples:
        lines.append("| none | none | none | none | 0 | 0 | 0.0000 | no samples processed |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The script does not extract full archives to disk.",
            "- Only compact `44`-dimensional landmark sequences are saved.",
            "- IPN labels mapped as `validation_reference` are excluded by default to avoid teaching the model to reject swipe-like motions.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(
    report: TarIpnConversionReport,
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


def log_mlflow(report: TarIpnConversionReport, *, tracking_uri: str, experiment: str) -> bool:
    try:
        import mlflow
    except Exception:
        return False
    try:
        mlflow.set_tracking_uri(str(tracking_uri))
        mlflow.set_experiment(str(experiment or "GestureBind"))
        with mlflow.start_run(run_name="ipn-tar-conversion"):
            mlflow.set_tags(
                {
                    "run_kind": "ipn_tar_conversion",
                    "status": report.status,
                    "source": "scripts.convert_ipn_hand_tars",
                }
            )
            mlflow.log_params(
                {
                    "tar_file_count": len(report.tar_files),
                    "out_root": report.out_root,
                    "limit_total": report.limit_total,
                    "max_per_original_label": report.max_per_original_label,
                    "max_frames_per_segment": report.max_frames_per_segment,
                    "min_detected_frame_ratio": report.min_detected_frame_ratio,
                }
            )
            mlflow.log_metrics(
                {
                    "videos_in_archives": float(report.videos_in_archives),
                    "segments_found": float(report.segments_found),
                    "candidate_segments": float(report.candidate_segments),
                    "selected_segments": float(report.selected_segments),
                    "converted_samples": float(report.converted_samples),
                    "skipped_samples": float(report.skipped_samples),
                    "detection_rate": float(report.detection_rate),
                }
            )
        return True
    except Exception as exc:
        print(f"[w] MLflow IPN tar logging failed: {exc}", flush=True)
        return False


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


def _report(
    *,
    status: str,
    tar_glob: str,
    tar_files: Sequence[Path],
    annotation_path: Path,
    mapping_path: Path,
    out_root: Path,
    limit_total: int,
    max_per_original_label: int,
    max_frames_per_segment: int,
    min_detected_frame_ratio: float,
    videos_in_archives: int,
    segments_found: int,
    candidate_segments: int,
    selected_segments: int,
    candidate_original_labels: dict[str, int],
    selected_original_labels: dict[str, int],
    samples: Sequence[TarIpnSample],
    warnings: Sequence[str],
) -> TarIpnConversionReport:
    converted = [item for item in samples if item.status == "converted"]
    requested = sum(item.requested_frames for item in samples)
    detected = sum(item.detected_frames for item in samples)
    labels = Counter(item.target_label for item in converted)
    original_labels = Counter(item.original_label for item in converted)
    return TarIpnConversionReport(
        generated_at=time.time(),
        status=status,
        tar_glob=str(tar_glob),
        tar_files=[str(path) for path in tar_files],
        annotation_path=str(annotation_path),
        mapping_path=str(mapping_path),
        out_root=str(out_root),
        limit_total=int(limit_total),
        max_per_original_label=int(max_per_original_label),
        max_frames_per_segment=int(max_frames_per_segment),
        min_detected_frame_ratio=float(min_detected_frame_ratio),
        videos_in_archives=int(videos_in_archives),
        segments_found=int(segments_found),
        candidate_segments=int(candidate_segments),
        selected_segments=int(selected_segments),
        candidate_original_labels=dict(sorted(candidate_original_labels.items())),
        selected_original_labels=dict(sorted(selected_original_labels.items())),
        converted_samples=len(converted),
        skipped_samples=sum(1 for item in samples if item.status != "converted"),
        detection_rate=round(float(detected / requested), 4) if requested else 0.0,
        labels=dict(sorted(labels.items())),
        original_labels=dict(sorted(original_labels.items())),
        samples=list(samples),
        warnings=list(warnings),
    )


def _tar_sample(
    segment: IpnSegment,
    *,
    status: str,
    path: str,
    requested: int,
    found: int,
    detected: int,
    saved: int,
    archive: str,
    reason: str,
) -> TarIpnSample:
    return TarIpnSample(
        status=status,
        path=path,
        video_id=segment.video_id,
        original_label=segment.original_label,
        target_label=segment.target_label,
        role=segment.role,
        start_frame=segment.start_frame,
        end_frame=segment.end_frame,
        requested_frames=int(requested),
        found_frames=int(found),
        detected_frames=int(detected),
        saved_frames=int(saved),
        detection_rate=round(float(detected / requested), 4) if requested else 0.0,
        archive=archive,
        reason=reason,
    )


def _write_tar_sample_metadata(path: Path, sample: TarIpnSample) -> None:
    payload = {
        "schema_version": 1,
        "generated_by": "scripts.convert_ipn_hand_tars",
        **asdict(sample),
    }
    path.with_suffix(".meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parse_member_frame(member_name: str) -> tuple[str, int] | None:
    parts = member_name.split("/")
    if len(parts) < 3 or parts[0] != "frames":
        return None
    video_id = parts[1]
    filename = parts[-1]
    match = re.search(r"_(\d{5,6})\.jpe?g$", filename, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"(\d{5,6})\.jpe?g$", filename, flags=re.IGNORECASE)
    if match is None:
        return None
    return video_id, int(match.group(1))


def _read_tar_frame_rgb(
    tar_path: Path,
    member_name: str,
    *,
    tar_cache: dict[Path, tarfile.TarFile],
) -> np.ndarray:
    import cv2

    archive = tar_cache.get(tar_path)
    if archive is None:
        archive = tarfile.open(tar_path)
        tar_cache[tar_path] = archive
    handle = archive.extractfile(member_name)
    if handle is None:
        raise ValueError(f"missing tar member: {member_name}")
    raw = np.frombuffer(handle.read(), dtype=np.uint8)
    frame_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError(f"cannot decode tar frame: {member_name}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _slug(value: str) -> str:
    clean = str(value or "").strip().lower()
    clean = re.sub(r"[^a-z0-9а-яё_-]+", "_", clean, flags=re.IGNORECASE)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar-glob", default=DEFAULT_TAR_GLOB)
    parser.add_argument("--ipn-root", type=Path, default=DEFAULT_IPN_ROOT)
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--limit-total", type=int, default=240)
    parser.add_argument("--max-per-original-label", type=int, default=40)
    parser.add_argument("--max-frames-per-segment", type=int, default=48)
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
    report = convert_ipn_tars(
        tar_glob=str(args.tar_glob),
        ipn_root=args.ipn_root,
        annotation_path=args.annotation,
        mapping_path=args.mapping,
        out_root=args.out_root,
        limit_total=int(args.limit_total),
        max_per_original_label=int(args.max_per_original_label),
        max_frames_per_segment=int(args.max_frames_per_segment),
        min_detected_frame_ratio=float(args.min_detected_frame_ratio),
        include_reference=bool(args.include_reference),
        clean_auto=not bool(args.keep_existing_auto),
        skip_mediapipe_preflight=bool(args.skip_mediapipe_preflight),
    )
    write_report(report, json_out=args.json_out, md_out=args.md_out)
    log_mlflow(
        report,
        tracking_uri=str(args.mlflow_tracking_uri),
        experiment=str(args.mlflow_experiment),
    )
    print(build_markdown_report(report))
    print(f"[✓] JSON: {_resolve_path(args.json_out)}")
    print(f"[✓] Markdown: {_resolve_path(args.md_out)}")


if __name__ == "__main__":
    main()
