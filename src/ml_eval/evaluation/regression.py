"""Prompt regression testing — detect score degradations against baselines."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from ml_eval.db import get_aggregated_scores, get_baseline


@dataclass
class RegressionResult:
    """Result of a regression check."""

    baseline_name: str
    baseline_run_id: str
    current_run_id: str
    regressions: list[dict[str, Any]] = field(default_factory=list)
    improvements: list[dict[str, Any]] = field(default_factory=list)
    has_regression: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline_name,
            "has_regression": self.has_regression,
            "regression_count": len(self.regressions),
            "improvement_count": len(self.improvements),
            "regressions": self.regressions,
            "improvements": self.improvements,
        }


def check_regression(
    conn: sqlite3.Connection,
    current_run_id: str,
    baseline_name: str,
    threshold: float = 0.05,
) -> RegressionResult:
    """Compare current evaluation run against a stored baseline.

    A regression is flagged when the current average score for any metric
    is worse than the baseline by more than `threshold`.

    Args:
        conn: SQLite connection.
        current_run_id: ID of the current evaluation run.
        baseline_name: Name of the baseline to compare against.
        threshold: Maximum allowable score decrease before flagging. Default 5%.

    Returns:
        RegressionResult with details on any regressions or improvements.

    Raises:
        ValueError: If the baseline is not found.
    """
    baseline = get_baseline(conn, baseline_name)
    if baseline is None:
        raise ValueError(f"Baseline '{baseline_name}' not found")

    baseline_run_id: str = baseline["run_id"]
    baseline_scores = get_aggregated_scores(conn, baseline_run_id)
    current_scores = get_aggregated_scores(conn, current_run_id)

    result = RegressionResult(
        baseline_name=baseline_name,
        baseline_run_id=baseline_run_id,
        current_run_id=current_run_id,
    )

    all_metrics = set(baseline_scores.keys()) | set(current_scores.keys())

    for metric in sorted(all_metrics):
        baseline_avg = baseline_scores.get(metric, {}).get("avg", 0.0)
        current_avg = current_scores.get(metric, {}).get("avg", 0.0)
        delta = current_avg - baseline_avg

        entry = {
            "metric": metric,
            "baseline_avg": round(baseline_avg, 4),
            "current_avg": round(current_avg, 4),
            "delta": round(delta, 4),
            "percent_change": round(delta / baseline_avg * 100, 2) if baseline_avg > 0 else 0.0,
        }

        if delta < -threshold:
            result.regressions.append(entry)
            result.has_regression = True
        elif delta > threshold:
            result.improvements.append(entry)

    return result
