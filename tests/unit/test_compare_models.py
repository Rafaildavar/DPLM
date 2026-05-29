import numpy as np

from scripts.compare_models import build_markdown_report, compare_models


def _write_sample(data_root, label, idx, value, dims=42):
    label_dir = data_root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    sequence = np.full((6, dims), float(value), dtype=np.float32)
    sequence[:, 0] += np.linspace(0.0, 0.5, num=6, dtype=np.float32)
    np.save(label_dir / f"sample_{idx:04d}.npy", sequence)


def test_compare_models_runs_knn_for_two_feature_modes(tmp_path):
    data_root = tmp_path / "gestures"
    for idx in range(4):
        _write_sample(data_root, "alpha", idx, value=0.0)
        _write_sample(data_root, "beta", idx, value=5.0)
        _write_sample(data_root, "gamma", idx, value=10.0, dims=84)

    report = compare_models(
        data_root=data_root,
        feature_modes=["static_mean", "dynamic_stats"],
        model_names=["knn"],
        max_folds=2,
        random_state=7,
    )

    assert report.dataset.sample_count == 12
    assert report.dataset.class_count == 3
    assert report.dataset.raw_feature_dims == [42, 84]
    assert report.dataset.target_dim == 84
    assert len(report.results) == 2
    assert report.best_result.model_name == "knn"
    assert report.best_result.macro_f1 >= 0.9


def test_build_markdown_report_mentions_best_model(tmp_path):
    data_root = tmp_path / "gestures"
    for idx in range(3):
        _write_sample(data_root, "a", idx, value=0.0)
        _write_sample(data_root, "b", idx, value=8.0)

    report = compare_models(
        data_root=data_root,
        feature_modes=["static_mean"],
        model_names=["knn"],
        max_folds=3,
    )
    markdown = build_markdown_report(report)

    assert "# JMLC Stage 2 - Model Comparison" in markdown
    assert "Лучший результат" in markdown
    assert "Матрица ошибок" in markdown
    assert "`static_mean`" in markdown
