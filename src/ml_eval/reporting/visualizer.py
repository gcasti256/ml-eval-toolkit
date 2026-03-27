"""Chart generation for evaluation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_eval.evaluation.runner import RunResult


def plot_metric_distribution(
    result: RunResult,
    output_path: str | Path = "metric_distribution.png",
) -> Path:
    """Plot score distributions for each metric in an evaluation run.

    Args:
        result: The evaluation run result.
        output_path: Path to save the chart image.

    Returns:
        Path to the saved image.
    """
    output_path = Path(output_path)
    metrics = list(result.metric_results.keys())

    if not metrics:
        raise ValueError("No metrics to plot")

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)

    for i, metric in enumerate(metrics):
        ax = axes[0][i]
        scores = [r.score for r in result.metric_results[metric]]
        ax.hist(scores, bins=20, edgecolor="black", alpha=0.7, color="#6366f1")
        ax.set_title(f"{metric} Score Distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        avg = sum(scores) / len(scores) if scores else 0
        ax.axvline(avg, color="red", linestyle="--", label=f"Mean: {avg:.3f}")
        ax.legend()

    fig.suptitle(f"Evaluation: {result.name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_comparison(
    results: list[RunResult],
    output_path: str | Path = "comparison.png",
) -> Path:
    """Plot a grouped bar chart comparing average scores across configs.

    Args:
        results: List of evaluation run results to compare.
        output_path: Path to save the chart image.

    Returns:
        Path to the saved image.
    """
    output_path = Path(output_path)

    if not results:
        raise ValueError("No results to compare")

    all_metrics: list[str] = []
    for r in results:
        for m in r.aggregated:
            if m not in all_metrics:
                all_metrics.append(m)

    config_names = [r.name for r in results]
    data: dict[str, list[float]] = {m: [] for m in all_metrics}

    for metric in all_metrics:
        for r in results:
            data[metric].append(r.aggregated.get(metric, {}).get("avg", 0.0))

    fig, ax = plt.subplots(figsize=(max(8, len(all_metrics) * 2), 5))

    x_positions = list(range(len(all_metrics)))
    bar_width = 0.8 / len(results) if results else 0.8
    colors = ["#6366f1", "#ec4899", "#10b981", "#f59e0b", "#3b82f6"]

    for i, name in enumerate(config_names):
        offsets = [x + i * bar_width for x in x_positions]
        values = [data[m][i] for m in all_metrics]
        ax.bar(
            offsets,
            values,
            bar_width,
            label=name,
            color=colors[i % len(colors)],
            edgecolor="black",
            alpha=0.8,
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Average Score")
    ax.set_title("Model Comparison")
    center_offset = bar_width * (len(results) - 1) / 2
    ax.set_xticks([x + center_offset for x in x_positions])
    ax.set_xticklabels(all_metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
