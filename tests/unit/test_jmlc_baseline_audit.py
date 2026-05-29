from pathlib import Path

import numpy as np

from scripts.jmlc_baseline_audit import (
    build_markdown_report,
    count_test_functions,
    inspect_sample,
    summarize_dataset,
    summarize_project,
)


def test_inspect_sample_accepts_sequence_landmarks(tmp_path):
    data_root = tmp_path / "gestures"
    label_dir = data_root / "hello"
    label_dir.mkdir(parents=True)
    sample_path = label_dir / "sample_0000.npy"
    np.save(sample_path, np.zeros((30, 21, 2), dtype=np.float32))

    summary = inspect_sample(sample_path, data_root)

    assert summary.valid is True
    assert summary.frames == 30
    assert summary.feature_dim == 42
    assert summary.shape == (30, 21, 2)
    assert summary.path == "hello/sample_0000.npy"


def test_summarize_dataset_reports_class_balance_and_invalid_samples(tmp_path):
    data_root = tmp_path / "gestures"
    for label, sample_count in {"a": 2, "b": 1}.items():
        label_dir = data_root / label
        label_dir.mkdir(parents=True)
        for idx in range(sample_count):
            np.save(label_dir / f"sample_{idx:04d}.npy", np.zeros((20 + idx, 42), dtype=np.float32))

    bad_dir = data_root / "bad"
    bad_dir.mkdir()
    np.save(bad_dir / "sample_0000.npy", np.zeros((2, 3, 4, 5), dtype=np.float32))

    summary = summarize_dataset(data_root)

    assert summary.class_count == 3
    assert summary.active_class_count == 2
    assert summary.empty_class_count == 1
    assert summary.sample_count == 4
    assert summary.valid_sample_count == 3
    assert summary.invalid_sample_count == 1
    assert summary.active_sample_min == 1
    assert summary.active_sample_max == 2
    assert summary.imbalance_ratio == 2.0
    assert summary.frame_min == 20
    assert summary.frame_max == 21
    assert summary.feature_dims == [42]
    bad_class = next(item for item in summary.classes if item.label == "bad")
    assert bad_class.issues == ["unexpected_ndim:4"]


def test_count_test_functions_includes_methods(tmp_path):
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_sample.py").write_text(
        "\n".join(
            [
                "def test_top_level():",
                "    pass",
                "",
                "class TestThing:",
                "    def test_method(self):",
                "        pass",
                "",
                "def helper():",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )

    assert count_test_functions(test_root) == 2


def test_build_markdown_report_contains_jmlc_assessment(tmp_path):
    project_root = tmp_path
    data_root = project_root / "data" / "gestures"
    label_dir = data_root / "hello"
    label_dir.mkdir(parents=True)
    np.save(label_dir / "sample_0000.npy", np.zeros((30, 42), dtype=np.float32))
    (project_root / "tests").mkdir()
    (project_root / "pytest.ini").write_text("--cov-fail-under=90\n", encoding="utf-8")
    (project_root / "JMLC.md").write_text("# JMLC\n", encoding="utf-8")

    summary = summarize_project(project_root, data_root)
    report = build_markdown_report(
        summary,
        "71 passed, 1 skipped; coverage failure: total 26.40 < 90",
    )

    assert "# JMLC Stage 1 - Baseline Audit" in report
    assert "Dataset status" in report
    assert "Coverage gate | 90" in report
    assert "Active classes | 1" in report
    assert "`JMLC.md` | yes" in report
    assert "Decision for Stage 2" in report
