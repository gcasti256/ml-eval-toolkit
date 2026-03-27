"""Tests for the evaluation runner."""

import sqlite3

import pytest

from ml_eval.config import EvalConfig, MetricConfig
from ml_eval.datasets.schema import DatasetSchema, Sample
from ml_eval.db import get_aggregated_scores, get_results_for_run, get_run
from ml_eval.evaluation.runner import EvalRunner


@pytest.fixture
def sample_dataset() -> DatasetSchema:
    return DatasetSchema(
        samples=[
            Sample(input="What is 2+2?", expected_output="The answer is 4", actual_output="The answer is 4"),
            Sample(input="Capital of France?", expected_output="Paris is the capital of France", actual_output="Paris is the capital"),
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
        # 2/3 samples are perfect, 1 is close — average should be high
        assert result.aggregated["bleu"]["avg"] > 0.5
        assert result.aggregated["bleu"]["count"] == 3

    def test_run_rouge(self, db: sqlite3.Connection, sample_dataset: DatasetSchema) -> None:
        config = EvalConfig(
            dataset_path="test.json",
            metrics=[MetricConfig(name="rouge")],
            name="rouge_test",
        )
        runner = EvalRunner(db, config)
        result = runner.run(sample_dataset)

        assert "rouge" in result.aggregated
        assert result.aggregated["rouge"]["avg"] > 0.5

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
        assert result.aggregated["bleu"]["avg"] > 0
        assert result.aggregated["rouge"]["avg"] > 0

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
        assert run["name"] == "db_test"

        results = get_results_for_run(db, result.run_id)
        assert len(results) == 3
        assert all(r["metric_name"] == "bleu" for r in results)
        assert all(0 <= r["score"] <= 1 for r in results)

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
        assert agg["bleu"]["avg"] == pytest.approx(result.aggregated["bleu"]["avg"], abs=1e-4)

    def test_summary_has_correct_scores(self, db: sqlite3.Connection, sample_dataset: DatasetSchema) -> None:
        config = EvalConfig(
            dataset_path="test.json",
            metrics=[MetricConfig(name="rouge")],
            name="summary_test",
        )
        runner = EvalRunner(db, config)
        result = runner.run(sample_dataset)
        summary = result.summary()

        assert summary["run_id"] == result.run_id
        assert summary["name"] == "summary_test"
        assert "rouge" in summary["scores"]
        assert 0 < summary["scores"]["rouge"] <= 1

    def test_invalid_metric(self, db: sqlite3.Connection, sample_dataset: DatasetSchema) -> None:
        config = EvalConfig(
            dataset_path="test.json",
            metrics=[MetricConfig(name="nonexistent")],
        )
        runner = EvalRunner(db, config)
        with pytest.raises(ValueError, match="Unknown metric"):
            runner.run(sample_dataset)

    def test_failed_run_marked_in_db(self, db: sqlite3.Connection, sample_dataset: DatasetSchema) -> None:
        config = EvalConfig(
            dataset_path="test.json",
            metrics=[MetricConfig(name="nonexistent")],
            name="fail_test",
        )
        runner = EvalRunner(db, config)
        with pytest.raises(ValueError):
            runner.run(sample_dataset)

        # The run should be marked as failed in the DB
        rows = db.execute("SELECT status FROM eval_runs WHERE name = ?", ("fail_test",)).fetchall()
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
