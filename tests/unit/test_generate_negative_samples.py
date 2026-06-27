import json

import numpy as np

from scripts.generate_negative_samples import generate_negative_samples


def _write_taxonomy(path):
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "default_type": "static",
                "types": {
                    "static": ["gun"],
                    "dynamic": ["swipe_left"],
                    "negative": ["no_gesture_static", "wrong_axis_motion"],
                },
                "patterns": {},
            }
        ),
        encoding="utf-8",
    )


def test_generate_negative_samples_creates_auto_files_without_touching_manual_data(tmp_path):
    data_root = tmp_path / "gestures"
    taxonomy_path = tmp_path / "gesture_taxonomy.json"
    manifest_out = tmp_path / "negative_sampling_manifest.json"
    _write_taxonomy(taxonomy_path)

    source_dir = data_root / "swipe_left"
    source_dir.mkdir(parents=True)
    source = np.zeros((36, 44), dtype=np.float32)
    source[:, 42] = np.linspace(0.78, 0.32, 36)
    source[:, 43] = 0.5
    np.save(source_dir / "sample_0000.npy", source)

    negative_dir = data_root / "no_gesture_static"
    negative_dir.mkdir(parents=True)
    manual_path = negative_dir / "sample_9999.npy"
    stale_auto_path = negative_dir / "sample_auto_9999.npy"
    np.save(manual_path, np.ones((36, 44), dtype=np.float32))
    np.save(stale_auto_path, np.ones((36, 44), dtype=np.float32))
    stale_auto_path.with_suffix(".meta.json").write_text("{}", encoding="utf-8")

    report = generate_negative_samples(
        data_root=data_root,
        taxonomy_path=taxonomy_path,
        labels=["no_gesture_static", "wrong_axis_motion"],
        samples_per_label=2,
        target_frames=36,
        seed=7,
        manifest_out=manifest_out,
    )

    assert report.source_samples == 1
    assert report.generated_samples == 4
    assert report.labels == {"no_gesture_static": 2, "wrong_axis_motion": 2}
    assert manual_path.exists()
    assert not stale_auto_path.exists()
    assert not stale_auto_path.with_suffix(".meta.json").exists()

    generated_path = negative_dir / "sample_auto_0000.npy"
    generated = np.load(generated_path)
    assert generated.shape == (36, 44)
    metadata = json.loads(generated_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata["generated_by"] == "scripts.generate_negative_samples"
    assert metadata["source_label"] == "swipe_left"

    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert manifest["generated_samples"] == 4
    assert manifest["samples"][0]["source_label"] == "swipe_left"
