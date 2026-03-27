"""Tests for prompt regression testing."""

import sqlite3

import pytest

from ml_eval.config import EvalConfig, MetricConfig
from ml_eval.datasets.schema import DatasetSchema, Sample
from ml_eval.db import init_db, save_baseline
from ml_eval.evaluation.regression import check_regression
from ml_eval.evaluation.runner import EvalRunner


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


@pytest.fixture
def dataset() -> DatasetSchema:
    return DatasetSchema(
        samples=[
            Sample(input="Q1", expected_output="hello world", actual_output="hello world"),
            Sample(input="Q2", expected_output="foo bar baz", actual_output="foo bar baz"),
        ],
        name="regression_test",
    )


def _create_baseline(db: sqlite3.Connection, dataset: DatasetSchema) -> str:
    """Run evaluation and save as baseline."""
    config = EvalConfig(
        dataset_path="test.json",
        metrics=[MetricConfig(name="bleu", params={"max_n": 2})],
        name="baseline_run",
    )
    runner = EvalRunner(db, config)
    result = runner.run(dataset)
    save_baseline(db, result.run_id, "test_baseline")
    return result.run_id


class TestCheckRegression:
    def test_no_regression(self, db: sqlite3.Connection, dataset: DatasetSchema) -> None:
        _create_baseline(db, dataset)

        # Run again with same data — should have no regression
        config = EvalConfig(
            dataset_path="test.json",
            metrics=[MetricConfig(name="bleu", params={"max_n": 2})],
            name="current_run",
        )
        runner = EvalRunner(db, config)
        result = runner.run(dataset)

        reg = check_regression(db, result.run_id, "test_baseline")
        assert not reg.has_regression

    def test_regression_detected(self, db: sqlite3.Connection) -> None:
        # Create baseline with perfect scores
        perfect_ds = DatasetSchema(
            samples=[Sample(input="Q", expected_output="hello world", actual_output="hello world")],
        )
        _baseline_config = EvalConfig(
            dataset_path="test.json",
            metrics=[MetricConfig(name="bleu", params={"max_n": 2})],
            name="perfect_baseline",
        )
        runner = EvalRunner(db, _baseline_config)
        baseline_result = runner.run(perfect_ds)
        save_baseline(db, baseline_result.run_id, "perfect_baseline")

        # Run with degraded data
        degraded_ds = DatasetSchema(
            samples=[Sample(input="Q", expected_output="hello world", actual_output="completely different text here")],
        )
        config = EvalConfig(
            dataset_path="test.json",
            metrics=[MetricConfig(name="bleu", params={"max_n": 2})],
            name="degraded_run",
        )
        runner2 = EvalRunner(db, config)
        degraded_result = runner2.run(degraded_ds)

        reg = check_regression(db, degraded_result.run_id, "perfect_baseline")
        assert reg.has_regression
        assert len(reg.regressions) > 0

    def test_baseline_not_found(self, db: sqlite3.Connection) -> None:
        with pytest.raises(ValueError, match="not found"):
            check_regression(db, "some-run-id", "nonexistent")

    def test_summary(self, db: sqlite3.Connection, dataset: DatasetSchema) -> None:
        _create_baseline(db, dataset)
        config = EvalConfig(
            dataset_path="test.json",
            metrics=[MetricConfig(name="bleu", params={"max_n": 2})],
        )
        runner = EvalRunner(db, config)
        result = runner.run(dataset)
        reg = check_regression(db, result.run_id, "test_baseline")
        summary = reg.summary()
        assert "has_regression" in summary
        assert "regression_count" in summary
