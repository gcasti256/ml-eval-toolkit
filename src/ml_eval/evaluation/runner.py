"""Evaluation runner — orchestrates metric computation over datasets."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from ml_eval.config import EvalConfig, MetricConfig
from ml_eval.datasets.schema import DatasetSchema
from ml_eval.db import complete_run, fail_run, init_db, store_result, store_run
from ml_eval.metrics import get_metric
from ml_eval.metrics.base import BaseMetric, MetricResult


@dataclass
class RunResult:
    """Aggregated results from an evaluation run."""

    run_id: str
    name: str
    metric_results: dict[str, list[MetricResult]] = field(default_factory=dict)
    aggregated: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "scores": {
                metric: round(agg["avg"], 4) for metric, agg in self.aggregated.items()
            },
        }


class EvalRunner:
    """Orchestrates evaluation of metrics against datasets."""

    def __init__(self, conn: sqlite3.Connection, config: EvalConfig) -> None:
        self.conn = conn
        self.config = config
        init_db(conn)

    def run(self, dataset: DatasetSchema) -> RunResult:
        """Execute all configured metrics against the dataset."""
        config_dict = {
            "metrics": [{"name": m.name, "params": m.params} for m in self.config.metrics],
            "judge_model": self.config.judge_model,
            "embedding_model": self.config.embedding_model,
        }
        run_id = store_run(
            self.conn,
            name=self.config.name or "eval_run",
            description=self.config.description,
            dataset_path=self.config.dataset_path,
            config=config_dict,
        )

        result = RunResult(run_id=run_id, name=self.config.name or "eval_run")

        try:
            metrics = self._instantiate_metrics()
            for metric in metrics:
                metric_results = self._run_metric(metric, dataset, run_id)
                result.metric_results[metric.name] = metric_results
                result.aggregated[metric.name] = self._aggregate(metric_results)
            self.conn.commit()
            complete_run(self.conn, run_id)
        except Exception:
            fail_run(self.conn, run_id)
            raise

        return result

    def _instantiate_metrics(self) -> list[BaseMetric]:
        metrics: list[BaseMetric] = []
        for mc in self.config.metrics:
            kwargs = dict(mc.params)
            if mc.name == "llm_judge":
                kwargs.setdefault("model", self.config.judge_model)
            elif mc.name == "semantic":
                kwargs.setdefault("model_name", self.config.embedding_model)
            metrics.append(get_metric(mc.name, **kwargs))
        return metrics

    def _run_metric(
        self, metric: BaseMetric, dataset: DatasetSchema, run_id: str
    ) -> list[MetricResult]:
        references = [s.expected_output for s in dataset.samples]
        hypotheses = [s.actual_output or s.expected_output for s in dataset.samples]

        results = metric.compute_batch(references, hypotheses)

        for i, (sample, mr) in enumerate(zip(dataset.samples, results, strict=True)):
            store_result(
                self.conn,
                run_id=run_id,
                metric_name=metric.name,
                sample_index=i,
                input_text=sample.input,
                expected_output=sample.expected_output,
                actual_output=sample.actual_output or sample.expected_output,
                score=mr.score,
                details=mr.details,
            )

        return results

    @staticmethod
    def from_metric_names(
        conn: sqlite3.Connection,
        metric_names: list[str],
        dataset_path: str = "",
        name: str = "eval_run",
    ) -> EvalRunner:
        """Create an EvalRunner from a list of metric names."""
        config = EvalConfig(
            dataset_path=dataset_path,
            metrics=[MetricConfig(name=m) for m in metric_names],
            name=name,
        )
        return EvalRunner(conn, config)

    @staticmethod
    def _aggregate(results: list[MetricResult]) -> dict[str, float | int]:
        scores = [r.score for r in results]
        if not scores:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}
        return {
            "avg": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
        }
