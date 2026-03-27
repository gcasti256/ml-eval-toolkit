"""Tests for the evaluation runner."""

import sqlite3

import pytest

from ml_eval.config import EvalConfig, MetricConfig
from ml_eval.datasets.schema import DatasetSchema, Sample
from ml_eval.db import get_aggregated_scores, get_results_for_run, get_run, init_db
from ml_eval.evaluation.runner import EvalRunner


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


@pytest.fixture
def sample_dataset() -> DatasetSchema:
    return DatasetSchema(
        samples=[
            Sample(input="What is 2+2?", expected_output="The answer is 4", actual_output="The answer is 4"),
            Sample(input="Capital of France?", expected_output="Paris is the capital", actual_output="Paris is the capital of France"),
            Sample(input="Color of sky?", expected_output="The sky is blue", actual_output="The sky is blue"),
        ],
        name="test_dataset",
    )


class TestEvalRunner:
    def test_run_bleu(self, db: sqlite3.Connection, sample_dataset: DatasetSchema) -> None:
        config = EvalConfig(
            dataset_path="test.json",
            metrics=[MetricConfig(name="bleu", params={"max_n": 2})],
            name="bleu_test",
        )
        runner = EvalRunner(db, config)
        result = runner.run(sample_dataset)

        assert result.run_id
        assert "bleu" in result.aggregated
        assert result.aggregated["bleu"]["avg"] > 0

    def test_run_rouge(self, db: sqlite3.Connection, sample_dataset: DatasetSchema) -> None:
        config = EvalConfig(
            dataset_path="test.json",
            metrics=[MetricConfig(name="rouge")],
            name="rouge_test",
        )
        runner = EvalRunner(db, config)
        result = runner.run(sample_dataset)

        assert "rouge" in result.aggregated
        assert 0 <= result.aggregated["rouge"]["avg"] <= 1

    def test_run_multiple_metrics(self, db: sqlite3.Connection, sample_dataset: DatasetSchema) -> None:
        config = EvalConfig(
            dataset_path="test.json",
            metrics=[
                MetricConfig(name="bleu", params={"max_n": 2}),
                MetricConfig(name="rouge"),
            ],
            name="multi_test",
        )
        runner = EvalRunner(db, config)
        result = runner.run(sample_dataset)

        assert "bleu" in result.aggregated
        assert "rouge" in result.aggregated

    def test_results_stored_in_db(self, db: sqlite3.Connection, sample_dataset: DatasetSchema) -> None:
        config = EvalConfig(
            dataset_path="test.json",
            metrics=[MetricConfig(name="bleu", params={"max_n": 2})],
            name="db_test",
        )
        runner = EvalRunner(db, config)
        result = runner.run(sample_dataset)

        run = get_run(db, result.run_id)
        assert run is not None
        assert run["status"] == "completed"

        results = get_results_for_run(db, result.run_id)
        assert len(results) == 3  # 3 samples

    def test_aggregated_scores_from_db(self, db: sqlite3.Connection, sample_dataset: DatasetSchema) -> None:
        config = EvalConfig(
            dataset_path="test.json",
            metrics=[MetricConfig(name="bleu", params={"max_n": 2})],
        )
        runner = EvalRunner(db, config)
        result = runner.run(sample_dataset)

        agg = get_aggregated_scores(db, result.run_id)
        assert "bleu" in agg
        assert agg["bleu"]["count"] == 3

    def test_from_metric_names(self, db: sqlite3.Connection, sample_dataset: DatasetSchema) -> None:
        runner = EvalRunner.from_metric_names(db, ["bleu", "rouge"], name="factory_test")
        result = runner.run(sample_dataset)
        assert "bleu" in result.aggregated
        assert "rouge" in result.aggregated

    def test_summary(self, db: sqlite3.Connection, sample_dataset: DatasetSchema) -> None:
        config = EvalConfig(
            dataset_path="test.json",
            metrics=[MetricConfig(name="rouge")],
            name="summary_test",
        )
        runner = EvalRunner(db, config)
        result = runner.run(sample_dataset)
        summary = result.summary()
        assert "run_id" in summary
        assert "scores" in summary
        assert "rouge" in summary["scores"]

    def test_invalid_metric(self, db: sqlite3.Connection, sample_dataset: DatasetSchema) -> None:
        config = EvalConfig(
            dataset_path="test.json",
            metrics=[MetricConfig(name="nonexistent")],
        )
        runner = EvalRunner(db, config)
        with pytest.raises(ValueError, match="Unknown metric"):
            runner.run(sample_dataset)
