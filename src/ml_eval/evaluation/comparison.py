"""Side-by-side model comparison."""

from __future__ import annotations

import copy
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from ml_eval.config import EvalConfig
from ml_eval.datasets.schema import DatasetSchema
from ml_eval.evaluation.runner import EvalRunner, RunResult


@dataclass
class ComparisonResult:
    results: list[RunResult] = field(default_factory=list)

    def summary_table(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for result in self.results:
            row: dict[str, Any] = {"config": result.name, "run_id": result.run_id}
            for metric, agg in result.aggregated.items():
                row[f"{metric}_avg"] = round(agg["avg"], 4)
                row[f"{metric}_min"] = round(agg["min"], 4)
                row[f"{metric}_max"] = round(agg["max"], 4)
            rows.append(row)
        return rows

    def best_config(self, metric: str) -> str | None:
        """Find the config with the highest average score for a given metric."""
        best_score = -1.0
        best_name: str | None = None
        for result in self.results:
            if metric in result.aggregated:
                avg = result.aggregated[metric]["avg"]
                if avg > best_score:
                    best_score = avg
                    best_name = result.name
        return best_name


def compare_configs(
    conn: sqlite3.Connection,
    dataset: DatasetSchema,
    configs: list[EvalConfig],
) -> ComparisonResult:
    """Run evaluation for each config and collect results for comparison."""
    comparison = ComparisonResult()

    for i, config in enumerate(configs):
        cfg = copy.copy(config)
        if not cfg.name:
            cfg.name = f"config_{i + 1}"
        runner = EvalRunner(conn, cfg)
        result = runner.run(dataset)
        comparison.results.append(result)

    return comparison
