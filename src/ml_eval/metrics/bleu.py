"""BLEU score with configurable n-gram order and smoothing."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from ml_eval.metrics.base import BaseMetric, MetricResult


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return text.lower().split()


def _get_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    """Extract n-grams from a token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(
    reference_tokens: list[str], hypothesis_tokens: list[str], n: int
) -> tuple[int, int]:
    """Compute modified n-gram precision (clipped counts).

    Returns (clipped_count, total_hypothesis_ngrams).
    """
    ref_ngrams = _get_ngrams(reference_tokens, n)
    hyp_ngrams = _get_ngrams(hypothesis_tokens, n)

    clipped_count = 0
    for ngram, count in hyp_ngrams.items():
        clipped_count += min(count, ref_ngrams.get(ngram, 0))

    total = max(sum(hyp_ngrams.values()), 1)
    return clipped_count, total


def _brevity_penalty(reference_length: int, hypothesis_length: int) -> float:
    """Compute the brevity penalty."""
    if hypothesis_length == 0:
        return 0.0
    if hypothesis_length >= reference_length:
        return 1.0
    return math.exp(1 - reference_length / hypothesis_length)


class BLEUMetric(BaseMetric):
    """BLEU score with configurable n-gram order and smoothing.

    Args:
        max_n: Maximum n-gram order (default 4 for BLEU-4).
        weights: Weights for each n-gram level. Defaults to uniform.
        smoothing: Smoothing method — "none", "add_epsilon", or "floor".
        epsilon: Epsilon value for add_epsilon smoothing (default 0.1).
    """

    def __init__(
        self,
        max_n: int = 4,
        weights: list[float] | None = None,
        smoothing: str = "add_epsilon",
        epsilon: float = 0.1,
        **_: Any,
    ) -> None:
        self.max_n = max_n
        self.weights = weights or [1.0 / max_n] * max_n
        self.smoothing = smoothing
        self.epsilon = epsilon

        if len(self.weights) != self.max_n:
            raise ValueError(f"weights length ({len(self.weights)}) must match max_n ({self.max_n})")

    @property
    def name(self) -> str:
        return "bleu"

    def compute(self, reference: str, hypothesis: str) -> MetricResult:
        ref_tokens = _tokenize(reference)
        hyp_tokens = _tokenize(hypothesis)

        if not hyp_tokens:
            return MetricResult(score=0.0, details={"brevity_penalty": 0.0, "precisions": []})

        bp = _brevity_penalty(len(ref_tokens), len(hyp_tokens))
        precisions: list[float] = []
        log_avg = 0.0

        for n in range(1, self.max_n + 1):
            clipped, total = _modified_precision(ref_tokens, hyp_tokens, n)
            p = clipped / total if total > 0 else 0.0

            if self.smoothing == "add_epsilon" and p == 0.0:
                p = self.epsilon / total if total > 0 else 0.0
            elif self.smoothing == "floor" and p == 0.0:
                p = 1e-7

            precisions.append(p)

            if p > 0:
                log_avg += self.weights[n - 1] * math.log(p)
            else:
                return MetricResult(
                    score=0.0,
                    details={
                        "brevity_penalty": bp,
                        "precisions": precisions,
                        "ref_length": len(ref_tokens),
                        "hyp_length": len(hyp_tokens),
                    },
                )

        score = bp * math.exp(log_avg)
        score = max(0.0, min(1.0, score))

        return MetricResult(
            score=score,
            details={
                "brevity_penalty": bp,
                "precisions": precisions,
                "ref_length": len(ref_tokens),
                "hyp_length": len(hyp_tokens),
            },
        )
