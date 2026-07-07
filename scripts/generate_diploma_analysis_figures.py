from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
MPLCONFIG = Path("/private/tmp/dplm_mplconfig")
MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_RESULTS = {
    "data_root": "data/gestures",
    "sample_count": 41,
    "feature_dim": 42,
    "classes": ["CTRLZ", "New"],
    "class_counts": {"CTRLZ": 21, "New": 20},
    "folds": 5,
    "neighbors": 5,
    "variants": {
        "baseline": {
            "accuracy": 0.926829268292683,
            "macro_precision": 0.9375,
            "macro_recall": 0.925,
            "macro_f1": 0.9261261261261261,
        },
        "train_standardize": {
            "accuracy": 0.926829268292683,
            "macro_precision": 0.9375,
            "macro_recall": 0.925,
            "macro_f1": 0.9261261261261261,
        },
        "sample_mean_center": {
            "accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
        },
        "sample_l2_normalize": {
            "accuracy": 0.9024390243902439,
            "macro_precision": 0.92,
            "macro_recall": 0.9,
            "macro_f1": 0.9009661835748792,
        },
    },
}

VARIANT_LABELS = {
    "baseline": "Базовый\npipeline",
    "train_standardize": "Стандартизация",
    "sample_mean_center": "Центрирование\nпримера",
    "sample_l2_normalize": "L2-\nнормализация",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_precision": "Macro precision",
    "macro_recall": "Macro recall",
    "macro_f1": "Macro F1",
}


def load_results() -> dict:
    candidates = [
        Path("/tmp/dplm_metrics_current.json"),
        ROOT / "docs" / "figures" / "gesture_metrics_current.json",
        ROOT / "docs" / "gesture_metrics_results_current.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    return DEFAULT_RESULTS


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial Unicode MS",
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfd",
            "axes.edgecolor": "#263238",
            "axes.grid": True,
            "grid.color": "#d8dee9",
            "grid.linewidth": 0.9,
            "grid.alpha": 0.9,
            "savefig.bbox": "tight",
        }
    )


def ordered_variants(results: dict) -> list[str]:
    preferred = ["baseline", "train_standardize", "sample_mean_center", "sample_l2_normalize"]
    available = list(results["variants"].keys())
    return [name for name in preferred if name in available] + [
        name for name in available if name not in preferred
    ]


def save_caption_note(fig: plt.Figure, results: dict) -> None:
    classes = ", ".join(results.get("classes", []))
    fig.text(
        0.02,
        0.02,
        f"Датасет: {results.get('sample_count')} примеров, классы: {classes}, "
        f"feature_dim={results.get('feature_dim')}, folds={results.get('folds')}, "
        f"KNN k={results.get('neighbors')}",
        fontsize=9,
        color="#54616f",
    )


def accuracy_bar(results: dict) -> None:
    variants = ordered_variants(results)
    labels = [VARIANT_LABELS.get(v, v) for v in variants]
    values = [results["variants"][v]["accuracy"] for v in variants]
    colors = ["#3b6fd8", "#2b9368", "#7a52c8", "#c77c00"]

    fig, ax = plt.subplots(figsize=(10.8, 6.0), dpi=300)
    bars = ax.bar(labels, values, color=colors[: len(values)], width=0.62, edgecolor="#263238", linewidth=0.8)

    ax.set_title("Сравнение вариантов обработки признаков по Accuracy", pad=18)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.88, 1.015)
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    best_idx = int(np.argmax(values))
    for idx, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.004,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold" if idx == best_idx else "normal",
            color="#1f2933",
        )
    ax.axhline(1.0, color="#8a95a5", linestyle="--", linewidth=1.0, alpha=0.75)
    save_caption_note(fig, results)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_5_3_metrics_accuracy.png")
    plt.close(fig)


def metrics_matrix(results: dict) -> None:
    variants = ordered_variants(results)
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    data = np.array(
        [[results["variants"][variant][metric] for metric in metrics] for variant in variants],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(11.2, 6.4), dpi=300)
    cmap = LinearSegmentedColormap.from_list(
        "gesture_metrics",
        ["#f6d8d8", "#f7e7b2", "#dcefd9", "#8bc9a7", "#2b9368"],
    )
    im = ax.imshow(data, cmap=cmap, vmin=0.88, vmax=1.0, aspect="auto")

    ax.set_title("Сравнение метрик классификации для вариантов обработки признаков", pad=18)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics])
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels([VARIANT_LABELS.get(v, v).replace("\n", " ") for v in variants])
    ax.grid(False)
    ax.tick_params(length=0)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if value > 0.965 else "#1f2933"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(variants), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Значение метрики", rotation=270, labelpad=18)
    save_caption_note(fig, results)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_5_3_metrics_matrix.png")
    plt.close(fig)


def main() -> None:
    setup_style()
    results = load_results()
    accuracy_bar(results)
    metrics_matrix(results)
    print(f"Generated analytical figures in {OUT}")


if __name__ == "__main__":
    main()
