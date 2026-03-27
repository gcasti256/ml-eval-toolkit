"""Base metric interface for all evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """Result from a single metric evaluation."""

    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")


class BaseMetric(ABC):
    """Base class for evaluation metrics. Score is always normalized to [0, 1]."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def compute(self, reference: str, hypothesis: str) -> MetricResult: ...

    def compute_batch(
        self, references: list[str], hypotheses: list[str]
    ) -> list[MetricResult]:
        """Compute metric for multiple pairs. Override for batch-optimized implementations."""
        if len(references) != len(hypotheses):
            raise ValueError(
                f"Length mismatch: {len(references)} references vs {len(hypotheses)} hypotheses"
            )
        return [self.compute(ref, hyp) for ref, hyp in zip(references, hypotheses, strict=True)]
