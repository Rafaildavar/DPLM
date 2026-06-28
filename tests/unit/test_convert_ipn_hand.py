import json
from pathlib import Path

import numpy as np

from scripts.convert_ipn_hand import (
    convert_ipn_hand,
    frame_paths_for_segment,
    load_ipn_mapping,
    load_ipn_segments,
)


def _write_mapping(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "default": {
                    "include": False,
                    "target_label": "negative_external_ipn_dynamic",
                    "role": "unknown",
                },
                "labels": {
                    "D0X": {
                        "include": True,
                        "target_label": "negative_external_ipn_dynamic",
                        "role": "non_gesture",
                    },
                    "G03": {
                        "include": False,
                        "target_label": "reference_ipn_throw_up",
                        "role": "validation_reference",
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _write_annotation(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "database": {
                    "VideoA^0001": {
                        "annotations": {
                            "label": "D0X",
                            "start_frame": 1,
                            "end_frame": 4,
                        }
                    },
                    "VideoA^0002": {
                        "annotations": {
                            "label": "G03",
                            "start_frame": 5,
                            "end_frame": 8,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def test_load_ipn_mapping_and_segments(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    annotation_path = tmp_path / "ipnall.json"
    _write_mapping(mapping_path)
    _write_annotation(annotation_path)

    mapping = load_ipn_mapping(mapping_path)
    segments = load_ipn_segments(annotation_path, mapping=mapping)

    assert mapping["D0X"].include is True
    assert segments[0].video_id == "VideoA"
    assert segments[0].target_label == "negative_external_ipn_dynamic"
    assert segments[0].include is True
    assert segments[1].target_label == "reference_ipn_throw_up"
    assert segments[1].include is False


def test_frame_paths_for_segment_uses_ipn_filename_pattern(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    annotation_path = tmp_path / "ipnall.json"
    frames_root = tmp_path / "frames"
    _write_mapping(mapping_path)
    _write_annotation(annotation_path)
    mapping = load_ipn_mapping(mapping_path)
    segment = load_ipn_segments(annotation_path, mapping=mapping)[0]
    video_dir = frames_root / "VideoA"
    video_dir.mkdir(parents=True)
    for index in range(1, 5):
        (video_dir / f"VideoA_{index:06d}.jpg").write_bytes(b"fake")

    paths = frame_paths_for_segment(frames_root, segment, max_frames=4)

    assert [path.name for path in paths] == [
        "VideoA_000001.jpg",
        "VideoA_000002.jpg",
        "VideoA_000003.jpg",
        "VideoA_000004.jpg",
    ]


def test_convert_ipn_hand_writes_negative_samples_with_fake_extractor(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    annotation_path = tmp_path / "ipnall.json"
    frames_root = tmp_path / "frames"
    out_root = tmp_path / "external" / "ipn_hand"
    _write_mapping(mapping_path)
    _write_annotation(annotation_path)
    video_dir = frames_root / "VideoA"
    video_dir.mkdir(parents=True)
    for index in range(1, 9):
        (video_dir / f"VideoA_{index:06d}.jpg").write_bytes(b"fake")

    def fake_extractor(paths):
        rows = np.zeros((len(paths), 44), dtype=np.float32)
        rows[:, 42] = np.linspace(0.2, 0.8, len(paths), dtype=np.float32)
        rows[:, 43] = 0.5
        return rows, len(paths), len(paths)

    report = convert_ipn_hand(
        ipn_root=tmp_path,
        frames_root=frames_root,
        annotation_path=annotation_path,
        mapping_path=mapping_path,
        out_root=out_root,
        limit_per_target_label=10,
        max_frames_per_segment=4,
        extractor=fake_extractor,
    )

    assert report.status == "ok"
    assert report.segments_found == 2
    assert report.segments_included == 1
    assert report.converted_samples == 1
    assert report.labels == {"negative_external_ipn_dynamic": 1}
    saved = list((out_root / "negative_external_ipn_dynamic").glob("sample_ipn_*.npy"))
    assert len(saved) == 1
    assert np.load(saved[0]).shape == (4, 44)
    metadata = json.loads(saved[0].with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata["generated_by"] == "scripts.convert_ipn_hand"
    assert metadata["original_label"] == "D0X"


def test_convert_ipn_hand_reports_missing_input(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    _write_mapping(mapping_path)

    report = convert_ipn_hand(
        ipn_root=tmp_path / "missing",
        mapping_path=mapping_path,
        out_root=tmp_path / "external",
    )

    assert report.status == "missing_input"
    assert report.converted_samples == 0
    assert any("frames root not found" in warning for warning in report.warnings)
