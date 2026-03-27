"""Custom rubric-based scoring metric.

Evaluates text against user-defined criteria with configurable scoring ranges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ml_eval.metrics.base import BaseMetric, MetricResult


@dataclass
class RubricCriterion:
    """A single criterion in a rubric."""

    name: str
    description: str
    weight: float = 1.0
    keywords: list[str] = field(default_factory=list)
    min_length: int = 0
    max_length: int = 0
    required_patterns: list[str] = field(default_factory=list)


def _evaluate_criterion(criterion: RubricCriterion, text: str) -> tuple[float, dict[str, Any]]:
    """Evaluate a single criterion against text. Returns (score, details)."""
    checks: list[tuple[str, bool, float]] = []
    text_lower = text.lower()

    # Keyword presence check
    if criterion.keywords:
        found = [kw for kw in criterion.keywords if kw.lower() in text_lower]
        kw_score = len(found) / len(criterion.keywords)
        checks.append(("keywords", kw_score > 0, kw_score))

    # Length checks
    word_count = len(text.split())
    if criterion.min_length > 0:
        meets_min = word_count >= criterion.min_length
        checks.append(("min_length", meets_min, 1.0 if meets_min else word_count / criterion.min_length))
    if criterion.max_length > 0:
        meets_max = word_count <= criterion.max_length
        checks.append(("max_length", meets_max, 1.0 if meets_max else criterion.max_length / word_count))

    # Regex pattern checks
    if criterion.required_patterns:
        pattern_results = []
        for pattern in criterion.required_patterns:
            match = bool(re.search(pattern, text, re.IGNORECASE))
            pattern_results.append(match)
        pattern_score = sum(pattern_results) / len(pattern_results)
        checks.append(("patterns", pattern_score > 0, pattern_score))

    if not checks:
        return 1.0, {"note": "no criteria to check"}

    total_score = sum(score for _, _, score in checks) / len(checks)
    details = {
        name: {"passed": passed, "score": round(score, 4)} for name, passed, score in checks
    }

    return min(1.0, max(0.0, total_score)), details


class RubricMetric(BaseMetric):
    """Evaluate text against a rubric of criteria.

    Args:
        criteria: List of RubricCriterion or dicts defining the rubric.
    """

    def __init__(self, criteria: list[dict[str, Any] | RubricCriterion] | None = None, **_: Any) -> None:
        self.criteria: list[RubricCriterion] = []
        for c in criteria or []:
            if isinstance(c, dict):
                self.criteria.append(RubricCriterion(**c))
            else:
                self.criteria.append(c)

    @property
    def name(self) -> str:
        return "rubric"

    def compute(self, reference: str, hypothesis: str) -> MetricResult:
        if not self.criteria:
            return MetricResult(score=1.0, details={"note": "no criteria defined"})

        criterion_scores: list[float] = []
        details: dict[str, Any] = {}
        total_weight = sum(c.weight for c in self.criteria)

        for criterion in self.criteria:
            score, criterion_details = _evaluate_criterion(criterion, hypothesis)
            weighted = score * (criterion.weight / total_weight) if total_weight > 0 else score
            criterion_scores.append(weighted)
            details[criterion.name] = {
                "raw_score": round(score, 4),
                "weighted_score": round(weighted, 4),
                "weight": criterion.weight,
                **criterion_details,
            }

        final_score = sum(criterion_scores)
        final_score = max(0.0, min(1.0, final_score))
        return MetricResult(score=final_score, details=details)
