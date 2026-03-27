"""Tests for model comparison evaluation."""

import sqlite3

import pytest

from ml_eval.config import EvalConfig, MetricConfig
from ml_eval.datasets.schema import DatasetSchema, Sample
from ml_eval.db import init_db
from ml_eval.evaluation.comparison import compare_configs


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
            Sample(input="Q1", expected_output="A1", actual_output="A1"),
            Sample(input="Q2", expected_output="A2 more text", actual_output="A2 more text"),
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
                metrics=[MetricConfig(name="bleu", params={"max_n": 2})],
                name="config_b",
            ),
        ]
        result = compare_configs(db, dataset, configs)
        assert len(result.results) == 2
        assert result.results[0].name == "config_a"

    def test_summary_table(self, db: sqlite3.Connection, dataset: DatasetSchema) -> None:
        configs = [
            EvalConfig(
                dataset_path="test.json",
                metrics=[MetricConfig(name="rouge")],
                name="rouge_a",
            ),
        ]
        result = compare_configs(db, dataset, configs)
        table = result.summary_table()
        assert len(table) == 1
        assert "config" in table[0]

    def test_best_config(self, db: sqlite3.Connection, dataset: DatasetSchema) -> None:
        configs = [
            EvalConfig(
                dataset_path="test.json",
                metrics=[MetricConfig(name="bleu", params={"max_n": 2})],
                name="alpha",
            ),
            EvalConfig(
                dataset_path="test.json",
                metrics=[MetricConfig(name="bleu", params={"max_n": 2})],
                name="beta",
            ),
        ]
        result = compare_configs(db, dataset, configs)
        best = result.best_config("bleu")
        assert best in ("alpha", "beta")

    def test_auto_naming(self, db: sqlite3.Connection, dataset: DatasetSchema) -> None:
        configs = [
            EvalConfig(dataset_path="test.json", metrics=[MetricConfig(name="rouge")]),
        ]
        result = compare_configs(db, dataset, configs)
        assert result.results[0].name == "config_1"
