"""Tests for reporting module (export and visualization)."""

import json
import sqlite3
from pathlib import Path

import pytest

from ml_eval.config import EvalConfig, MetricConfig
from ml_eval.datasets.schema import DatasetSchema, Sample
from ml_eval.db import init_db
from ml_eval.evaluation.runner import EvalRunner
from ml_eval.reporting.exporter import export_csv, export_json
from ml_eval.reporting.visualizer import plot_comparison, plot_metric_distribution


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


@pytest.fixture
def run_result(db: sqlite3.Connection) -> tuple[sqlite3.Connection, str]:
    """Run a quick evaluation and return (conn, run_id)."""
    dataset = DatasetSchema(
        samples=[
            Sample(input="Q1", expected_output="hello world", actual_output="hello world"),
            Sample(input="Q2", expected_output="foo bar", actual_output="foo bar"),
        ],
    )
    config = EvalConfig(
        dataset_path="test.json",
        metrics=[MetricConfig(name="bleu", params={"max_n": 2}), MetricConfig(name="rouge")],
        name="export_test",
    )
    runner = EvalRunner(db, config)
    result = runner.run(dataset)
    return db, result.run_id


class TestExportJSON:
    def test_export(self, run_result: tuple[sqlite3.Connection, str], tmp_path: Path) -> None:
        conn, run_id = run_result
        out = tmp_path / "results.json"
        path = export_json(conn, run_id, out)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "run" in data
        assert "aggregated_scores" in data
        assert "results" in data
        assert len(data["results"]) == 4  # 2 samples * 2 metrics

    def test_export_nonexistent_run(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            export_json(db, "nonexistent", tmp_path / "out.json")


class TestExportCSV:
    def test_export(self, run_result: tuple[sqlite3.Connection, str], tmp_path: Path) -> None:
        conn, run_id = run_result
        out = tmp_path / "results.csv"
        path = export_csv(conn, run_id, out)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 5  # header + 4 result rows
        assert "metric_name" in lines[0]
        assert "score" in lines[0]

    def test_export_nonexistent_run(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No results found"):
            export_csv(db, "nonexistent", tmp_path / "out.csv")


class TestPlotMetricDistribution:
    def test_generates_image(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        dataset = DatasetSchema(
            samples=[
                Sample(input="Q1", expected_output="hello world", actual_output="hello world"),
                Sample(input="Q2", expected_output="foo bar", actual_output="foo bar"),
            ],
        )
        runner = EvalRunner.from_metric_names(db, ["bleu", "rouge"], name="plot_test")
        result = runner.run(dataset)
        out = tmp_path / "dist.png"
        path = plot_metric_distribution(result, out)
        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotComparison:
    def test_generates_comparison_chart(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        dataset = DatasetSchema(
            samples=[
                Sample(input="Q1", expected_output="hello world", actual_output="hello world"),
            ],
        )
        results = []
        for name in ["config_a", "config_b"]:
            runner = EvalRunner.from_metric_names(db, ["rouge"], name=name)
            results.append(runner.run(dataset))

        out = tmp_path / "comparison.png"
        path = plot_comparison(results, out)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_empty_results_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No results"):
            plot_comparison([], tmp_path / "empty.png")
