from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
BI = OUT / "bi"
MPLCONFIG = Path("/private/tmp/dplm_mplconfig")
MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial Unicode MS",
            "figure.facecolor": "#0f172a",
            "axes.facecolor": "#111827",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#e5e7eb",
            "axes.titlecolor": "#f8fafc",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#f8fafc",
            "grid.color": "#334155",
            "grid.alpha": 0.65,
            "grid.linewidth": 0.8,
            "savefig.facecolor": "#0f172a",
            "savefig.bbox": "tight",
        }
    )


def panel(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=14, pad=12, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, axis="y")


def runtime_dashboard() -> None:
    target_fps = 30
    frame_budget_ms = 1000 / target_fps
    latency_labels = ["Базовая\nмодель", "Оптимизированная\nмодель", "Бюджет кадра\n30 FPS"]
    latency_values = [15, 8, frame_budget_ms]

    fig = plt.figure(figsize=(13, 7.4), dpi=240)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.18], hspace=0.38, wspace=0.28)
    fig.suptitle("Мониторинг производительности GestureBind", x=0.04, ha="left", fontsize=20, fontweight="bold")
    fig.text(
        0.04,
        0.91,
        "Аналитическая панель для главы тестирования: целевой FPS, задержка инференса и запас до бюджета кадра",
        color="#94a3b8",
        fontsize=10,
    )

    kpi_data = [
        ("Target FPS", "30", "Настройка распознавания"),
        ("Frame budget", f"{frame_budget_ms:.1f} мс", "Максимум на кадр при 30 FPS"),
        ("Optimized latency", "8 мс", "Среднее время инференса"),
    ]
    for i, (title, value, subtitle) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#111827")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.text(0.06, 0.72, title, fontsize=12, color="#cbd5e1", transform=ax.transAxes)
        ax.text(0.06, 0.36, value, fontsize=28, fontweight="bold", color="#38bdf8", transform=ax.transAxes)
        ax.text(0.06, 0.14, subtitle, fontsize=9, color="#94a3b8", transform=ax.transAxes)

    ax = fig.add_subplot(gs[1, :2])
    panel(ax, "Задержка инференса относительно бюджета кадра")
    colors = ["#f97316", "#22c55e", "#64748b"]
    bars = ax.bar(latency_labels, latency_values, color=colors, edgecolor="#e5e7eb", linewidth=0.8)
    ax.set_ylabel("мс")
    ax.set_ylim(0, 38)
    ax.axhline(frame_budget_ms, color="#94a3b8", linestyle="--", linewidth=1.2)
    for bar, value in zip(bars, latency_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}", ha="center", fontsize=11)

    ax2 = fig.add_subplot(gs[1, 2])
    panel(ax2, "Запас до 30 FPS")
    reserve = frame_budget_ms - 8
    ax2.bar(["Запас"], [reserve], color="#22c55e", edgecolor="#e5e7eb")
    ax2.set_ylim(0, frame_budget_ms)
    ax2.set_ylabel("мс")
    ax2.text(0, reserve + 1.0, f"{reserve:.1f} мс", ha="center", fontsize=12, fontweight="bold")
    ax2.text(0, 3, "меньше latency -> устойчивее UI", ha="center", fontsize=9, color="#94a3b8")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_5_8_runtime_monitoring.png")
    plt.close(fig)


def quality_dashboard() -> None:
    groups = [
        ("Правила привязок\nи безопасность", 32),
        ("Команды и\nсистемные действия", 24),
        ("БД и хранение\nданных", 19),
        ("CV, запись\nи инференс", 14),
        ("Flet-контроллер\nи UI", 14),
        ("Голосовой\nассистент", 12),
        ("Управление\nуказателем", 12),
        ("Конфигурация\nи диагностика", 10),
    ]
    labels = [g[0] for g in groups]
    values = [g[1] for g in groups]

    fig = plt.figure(figsize=(13, 7.4), dpi=240)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.18], hspace=0.38, wspace=0.3)
    fig.suptitle("Мониторинг качества и тестирования GestureBind", x=0.04, ha="left", fontsize=20, fontweight="bold")
    fig.text(
        0.04,
        0.91,
        "Сводка свежего прогона pytest: 137 passed in 26.01s",
        color="#94a3b8",
        fontsize=10,
    )

    kpi_data = [
        ("Passed", "137", "unit-тестов пройдено"),
        ("Failed", "0", "ошибок в выбранном наборе"),
        ("Duration", "26.01 s", "время выполнения"),
    ]
    for i, (title, value, subtitle) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#111827")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        color = "#22c55e" if title != "Failed" else "#38bdf8"
        ax.text(0.06, 0.72, title, fontsize=12, color="#cbd5e1", transform=ax.transAxes)
        ax.text(0.06, 0.36, value, fontsize=28, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.06, 0.14, subtitle, fontsize=9, color="#94a3b8", transform=ax.transAxes)

    ax = fig.add_subplot(gs[1, :])
    panel(ax, "Распределение пройденных тестов по функциональным группам")
    y = np.arange(len(labels))
    ax.barh(y, values, color="#38bdf8", edgecolor="#e5e7eb", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Количество пройденных тестов")
    ax.set_xlim(0, 36)
    for yi, value in zip(y, values):
        ax.text(value + 0.6, yi, str(value), va="center", fontsize=10)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_5_9_quality_monitoring.png")
    plt.close(fig)


def write_csvs() -> None:
    BI.mkdir(parents=True, exist_ok=True)
    (BI / "monitoring_runtime_metrics.csv").write_text(
        "\n".join(
            [
                "Metric,Value,Unit,Source",
                "Target FPS,30,fps,app_config",
                "Frame budget,33.333,ms,derived_from_target_fps",
                "Baseline inference latency,15,ms,optimization_report",
                "Optimized inference latency,8,ms,optimization_report",
                "Latency reserve at 30 FPS,25.333,ms,derived_from_optimized_latency",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (BI / "monitoring_quality_metrics.csv").write_text(
        "\n".join(
            [
                "Metric,Value,Unit,Source",
                "Passed tests,137,count,pytest",
                "Failed tests,0,count,pytest",
                "Skipped tests,0,count,pytest",
                "Duration,26.01,seconds,pytest",
                "Dataset samples,41,count,data/gestures",
                "Classes,2,count,models/classes.json",
                "Feature dimension,42,count,models/feature_dim.txt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    setup_style()
    write_csvs()
    runtime_dashboard()
    quality_dashboard()
    print(f"Generated monitoring figures in {OUT}")


if __name__ == "__main__":
    main()
