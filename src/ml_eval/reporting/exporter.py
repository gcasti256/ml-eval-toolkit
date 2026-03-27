"""Export evaluation results to JSON and CSV formats."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from ml_eval.db import get_aggregated_scores, get_results_for_run, get_run


def export_json(conn: sqlite3.Connection, run_id: str, output_path: str | Path) -> Path:
    """Export evaluation results to a JSON file.

    Args:
        conn: SQLite connection.
        run_id: The evaluation run ID.
        output_path: Path for the output file.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    run = get_run(conn, run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found")

    results = get_results_for_run(conn, run_id)
    aggregated = get_aggregated_scores(conn, run_id)

    export_data: dict[str, Any] = {
        "run": run,
        "aggregated_scores": aggregated,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    return output_path


def export_csv(conn: sqlite3.Connection, run_id: str, output_path: str | Path) -> Path:
    """Export evaluation results to a CSV file.

    Args:
        conn: SQLite connection.
        run_id: The evaluation run ID.
        output_path: Path for the output file.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    results = get_results_for_run(conn, run_id)
    if not results:
        raise ValueError(f"No results found for run {run_id}")

    fieldnames = [
        "metric_name",
        "sample_index",
        "input_text",
        "expected_output",
        "actual_output",
        "score",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    return output_path
