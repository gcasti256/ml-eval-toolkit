"""Tests for prompt regression testing."""

import sqlite3

import pytest

from ml_eval.config import EvalConfig, MetricConfig
from ml_eval.datasets.schema import DatasetSchema, Sample
from ml_eval.db import save_baseline
from ml_eval.evaluation.regression import check_regression
from ml_eval.evaluation.runner import EvalRunner


def _run_eval(db: sqlite3.Connection, dataset: DatasetSchema, name: str = "run") -> str:
    config = EvalConfig(
        dataset_path="test.json",
        metrics=[MetricConfig(name="bleu", params={"max_n": 2})],
        name=name,
    )
    runner = EvalRunner(db, config)
    return runner.run(dataset).run_id


class TestCheckRegression:
    def test_no_regression_same_data(self, db: sqlite3.Connection) -> None:
        dataset = DatasetSchema(
            samples=[
                Sample(input="Q1", expected_output="hello world", actual_output="hello world"),
                Sample(input="Q2", expected_output="foo bar baz", actual_output="foo bar baz"),
            ],
        )
        baseline_id = _run_eval(db, dataset, "baseline")
        save_baseline(db, baseline_id, "v1")

        current_id = _run_eval(db, dataset, "current")
        reg = check_regression(db, current_id, "v1")

        assert not reg.has_regression
        assert len(reg.regressions) == 0
        assert reg.baseline_name == "v1"

    def test_regression_detected(self, db: sqlite3.Connection) -> None:
        perfect = DatasetSchema(
            samples=[Sample(input="Q", expected_output="hello world", actual_output="hello world")],
        )
        baseline_id = _run_eval(db, perfect, "perfect")
        save_baseline(db, baseline_id, "perfect_baseline")

        degraded = DatasetSchema(
            samples=[Sample(input="Q", expected_output="hello world", actual_output="completely different text here")],
        )
        current_id = _run_eval(db, degraded, "degraded")
        reg = check_regression(db, current_id, "perfect_baseline")

        assert reg.has_regression
        assert len(reg.regressions) == 1
        assert reg.regressions[0]["metric"] == "bleu"
        assert reg.regressions[0]["delta"] < 0
        assert reg.regressions[0]["baseline_avg"] > reg.regressions[0]["current_avg"]

    def test_custom_threshold(self, db: sqlite3.Connection) -> None:
        dataset = DatasetSchema(
            samples=[Sample(input="Q", expected_output="hello world", actual_output="hello world")],
        )
        baseline_id = _run_eval(db, dataset, "baseline")
        save_baseline(db, baseline_id, "thresh_baseline")
        current_id = _run_eval(db, dataset, "current")

        # Very tight threshold — still no regression with same data
        reg = check_regression(db, current_id, "thresh_baseline", threshold=0.001)
        assert not reg.has_regression

    def test_baseline_not_found(self, db: sqlite3.Connection) -> None:
        with pytest.raises(ValueError, match="not found"):
            check_regression(db, "some-run-id", "nonexistent")

    def test_summary_values(self, db: sqlite3.Connection) -> None:
        dataset = DatasetSchema(
            samples=[Sample(input="Q", expected_output="hello world", actual_output="hello world")],
        )
        baseline_id = _run_eval(db, dataset, "baseline")
        save_baseline(db, baseline_id, "sum_baseline")
        current_id = _run_eval(db, dataset, "current")

        reg = check_regression(db, current_id, "sum_baseline")
        summary = reg.summary()

        assert summary["has_regression"] is False
        assert summary["regression_count"] == 0
        assert summary["baseline"] == "sum_baseline"
