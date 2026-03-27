"""Tests for model comparison evaluation."""

import sqlite3

import pytest

from ml_eval.config import EvalConfig, MetricConfig
from ml_eval.datasets.schema import DatasetSchema, Sample
from ml_eval.evaluation.comparison import compare_configs


@pytest.fixture
def dataset() -> DatasetSchema:
    return DatasetSchema(
        samples=[
            Sample(input="Q1", expected_output="hello world test", actual_output="hello world test"),
            Sample(input="Q2", expected_output="foo bar baz qux", actual_output="foo bar baz qux"),
        ],
        name="compare_test",
    )


class TestCompareConfigs:
    def test_compare_two_configs(self, db: sqlite3.Connection, dataset: DatasetSchema) -> None:
        configs = [
            EvalConfig(
                dataset_path="test.json",
                metrics=[MetricConfig(name="bleu", params={"max_n": 2})],
                name="config_a",
            ),
            EvalConfig(
                dataset_path="test.json",
                metrics=[MetricConfig(name="rouge")],
                name="config_b",
            ),
        ]
        result = compare_configs(db, dataset, configs)
        assert len(result.results) == 2
        assert result.results[0].name == "config_a"
        assert result.results[1].name == "config_b"
        assert "bleu" in result.results[0].aggregated
        assert "rouge" in result.results[1].aggregated

    def test_summary_table_values(self, db: sqlite3.Connection, dataset: DatasetSchema) -> None:
        configs = [
            EvalConfig(
                dataset_path="test.json",
                metrics=[MetricConfig(name="rouge")],
                name="rouge_run",
            ),
        ]
        result = compare_configs(db, dataset, configs)
        table = result.summary_table()
        assert len(table) == 1
        assert table[0]["config"] == "rouge_run"
        assert "rouge_avg" in table[0]
        assert 0 <= table[0]["rouge_avg"] <= 1

    def test_best_config_returns_none_for_missing_metric(self, db: sqlite3.Connection, dataset: DatasetSchema) -> None:
        configs = [
            EvalConfig(
                dataset_path="test.json",
                metrics=[MetricConfig(name="bleu", params={"max_n": 2})],
                name="only_bleu",
            ),
        ]
        result = compare_configs(db, dataset, configs)
        assert result.best_config("nonexistent") is None

    def test_auto_naming(self, db: sqlite3.Connection, dataset: DatasetSchema) -> None:
        configs = [
            EvalConfig(dataset_path="test.json", metrics=[MetricConfig(name="rouge")]),
        ]
        result = compare_configs(db, dataset, configs)
        assert result.results[0].name == "config_1"

    def test_does_not_mutate_input_configs(self, db: sqlite3.Connection, dataset: DatasetSchema) -> None:
        config = EvalConfig(dataset_path="test.json", metrics=[MetricConfig(name="rouge")])
        original_name = config.name
        compare_configs(db, dataset, [config])
        assert config.name == original_name
