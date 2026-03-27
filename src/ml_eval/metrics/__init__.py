"""Pluggable metrics for ML evaluation."""

from ml_eval.metrics.base import BaseMetric, MetricResult
from ml_eval.metrics.bleu import BLEUMetric
from ml_eval.metrics.llm_judge import LLMJudgeMetric
from ml_eval.metrics.rouge import ROUGEMetric
from ml_eval.metrics.rubric import RubricMetric
from ml_eval.metrics.semantic import SemanticSimilarityMetric

METRIC_REGISTRY: dict[str, type[BaseMetric]] = {
    "bleu": BLEUMetric,
    "rouge": ROUGEMetric,
    "semantic": SemanticSimilarityMetric,
    "rubric": RubricMetric,
    "llm_judge": LLMJudgeMetric,
}


def get_metric(name: str, **kwargs: object) -> BaseMetric:
    """Instantiate a metric by name."""
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name](**kwargs)


__all__ = [
    "METRIC_REGISTRY",
    "BLEUMetric",
    "BaseMetric",
    "LLMJudgeMetric",
    "MetricResult",
    "ROUGEMetric",
    "RubricMetric",
    "SemanticSimilarityMetric",
    "get_metric",
]
