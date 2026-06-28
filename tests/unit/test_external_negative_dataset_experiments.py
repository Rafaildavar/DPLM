import json

import numpy as np

from scripts.external_negative_dataset_experiments import (
    SourceSpec,
    VariantSpec,
    _labels_for_scope,
    load_specs,
    materialize_variant_dataset,
)


def test_load_specs_reads_sources_and_variants(tmp_path):
    config = tmp_path / "external.json"
    config.write_text(
        json.dumps(
            {
                "sources": {
                    "demo": {
                        "root": "data/external/demo",
                        "domain": "static",
                        "target_label": "negative_external_demo_static",
                    }
                },
                "variants": [{"name": "demo_variant", "sources": ["demo"]}],
            }
        ),
        encoding="utf-8",
    )

    sources, variants = load_specs(config)

    assert sources["demo"].target_label == "negative_external_demo_static"
    assert variants == [VariantSpec(name="demo_variant", sources=["demo"])]


def test_materialize_variant_maps_external_to_negative_label(tmp_path):
    internal_root = tmp_path / "internal"
    external_root = tmp_path / "external" / "hagrid" / "call"
    work_root = tmp_path / "work"
    models_root = tmp_path / "models"

    gun_dir = internal_root / "gun"
    gun_dir.mkdir(parents=True)
    np.save(gun_dir / "sample_0000.npy", np.ones((5, 42), dtype=np.float32))

    external_root.mkdir(parents=True)
    np.save(external_root / "sample_0000.npy", np.zeros((5, 42), dtype=np.float32))

    summary = materialize_variant_dataset(
        variant=VariantSpec(name="hagrid_external", sources=["hagrid"]),
        sources={
            "hagrid": SourceSpec(
                name="hagrid",
                root=str(tmp_path / "external" / "hagrid"),
                domain="static",
                target_label="negative_external_hagrid_static",
                max_samples_per_label=10,
            )
        },
        internal_root=internal_root,
        work_root=work_root,
        models_root=models_root,
        symlink=False,
    )

    assert summary.status == "ok"
    assert summary.internal_samples == 1
    assert summary.external_samples == 1
    assert summary.class_counts["gun"] == 1
    assert summary.class_counts["negative_external_hagrid_static"] == 1
    target_dir = work_root / "hagrid_external" / "gestures" / "negative_external_hagrid_static"
    assert len(list(target_dir.glob("sample_external_hagrid_call_*.npy"))) == 1


def test_materialize_variant_skips_when_required_external_is_missing(tmp_path):
    internal_root = tmp_path / "internal"
    gun_dir = internal_root / "gun"
    gun_dir.mkdir(parents=True)
    np.save(gun_dir / "sample_0000.npy", np.ones((5, 42), dtype=np.float32))

    summary = materialize_variant_dataset(
        variant=VariantSpec(name="missing_external", sources=["ipn"]),
        sources={
            "ipn": SourceSpec(
                name="ipn",
                root=str(tmp_path / "external" / "ipn"),
                domain="dynamic",
                target_label="negative_external_ipn_dynamic",
            )
        },
        internal_root=internal_root,
        work_root=tmp_path / "work",
        models_root=tmp_path / "models",
        symlink=False,
    )

    assert summary.status == "skipped"
    assert summary.external_samples == 0
    assert summary.sources[0].status == "missing"


def test_labels_for_scope_filters_external_negative_by_domain(tmp_path):
    data_root = tmp_path / "gestures"
    for label in ("gun", "swipe_up", "negative_external_ipn_dynamic"):
        label_dir = data_root / label
        label_dir.mkdir(parents=True)
        np.save(label_dir / "sample_0000.npy", np.ones((5, 44), dtype=np.float32))
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(
            {
                "version": 1,
                "default_type": "static",
                "types": {
                    "static": ["gun"],
                    "dynamic": ["swipe_up"],
                    "negative": [],
                },
                "patterns": {"negative": ["negative_*"]},
            }
        ),
        encoding="utf-8",
    )

    domains = {"negative_external_ipn_dynamic": "dynamic"}

    assert _labels_for_scope(
        data_root,
        taxonomy_path,
        "static",
        external_label_domains=domains,
    ) == ["gun"]
    assert _labels_for_scope(
        data_root,
        taxonomy_path,
        "dynamic",
        external_label_domains=domains,
    ) == ["negative_external_ipn_dynamic", "swipe_up"]
