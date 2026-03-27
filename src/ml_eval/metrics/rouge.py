"""ROUGE score implementation from scratch.

Implements ROUGE-1, ROUGE-2, and ROUGE-L with precision, recall, and F1 scores.
"""

from __future__ import annotations

from typing import Any

from ml_eval.metrics.base import BaseMetric, MetricResult


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing."""
    return text.lower().split()


def _ngram_overlap(
    reference_tokens: list[str], hypothesis_tokens: list[str], n: int
) -> tuple[int, int, int]:
    """Count n-gram overlap between reference and hypothesis.

    Returns (overlap_count, reference_ngram_count, hypothesis_ngram_count).
    """
    ref_ngrams: dict[tuple[str, ...], int] = {}
    for i in range(len(reference_tokens) - n + 1):
        ng = tuple(reference_tokens[i : i + n])
        ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

    hyp_ngrams: dict[tuple[str, ...], int] = {}
    for i in range(len(hypothesis_tokens) - n + 1):
        ng = tuple(hypothesis_tokens[i : i + n])
        hyp_ngrams[ng] = hyp_ngrams.get(ng, 0) + 1

    overlap = 0
    for ng, count in hyp_ngrams.items():
        overlap += min(count, ref_ngrams.get(ng, 0))

    ref_count = sum(ref_ngrams.values())
    hyp_count = sum(hyp_ngrams.values())
    return overlap, ref_count, hyp_count


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute the length of the longest common subsequence."""
    m, n = len(x), len(y)
    # Space-optimized LCS using two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _f1(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class ROUGEMetric(BaseMetric):
    """ROUGE metric with support for ROUGE-1, ROUGE-2, and ROUGE-L.

    Args:
        variant: Which ROUGE variant to use as the primary score.
            One of "rouge-1", "rouge-2", "rouge-l". Default "rouge-l".
        score_type: Which score to use as the primary. One of "f1", "precision", "recall".
    """

    def __init__(
        self,
        variant: str = "rouge-l",
        score_type: str = "f1",
        **_: Any,
    ) -> None:
        valid_variants = {"rouge-1", "rouge-2", "rouge-l"}
        if variant not in valid_variants:
            raise ValueError(f"variant must be one of {valid_variants}")
        if score_type not in {"f1", "precision", "recall"}:
            raise ValueError("score_type must be one of: f1, precision, recall")
        self.variant = variant
        self.score_type = score_type

    @property
    def name(self) -> str:
        return "rouge"

    def _compute_rouge_n(
        self, ref_tokens: list[str], hyp_tokens: list[str], n: int
    ) -> dict[str, float]:
        """Compute ROUGE-N precision, recall, and F1."""
        overlap, ref_count, hyp_count = _ngram_overlap(ref_tokens, hyp_tokens, n)
        precision = overlap / hyp_count if hyp_count > 0 else 0.0
        recall = overlap / ref_count if ref_count > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }

    def _compute_rouge_l(
        self, ref_tokens: list[str], hyp_tokens: list[str]
    ) -> dict[str, float]:
        """Compute ROUGE-L precision, recall, and F1."""
        lcs = _lcs_length(ref_tokens, hyp_tokens)
        precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0
        recall = lcs / len(ref_tokens) if ref_tokens else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }

    def compute(self, reference: str, hypothesis: str) -> MetricResult:
        ref_tokens = _tokenize(reference)
        hyp_tokens = _tokenize(hypothesis)

        results = {
            "rouge-1": self._compute_rouge_n(ref_tokens, hyp_tokens, 1),
            "rouge-2": self._compute_rouge_n(ref_tokens, hyp_tokens, 2),
            "rouge-l": self._compute_rouge_l(ref_tokens, hyp_tokens),
        }

        primary = results[self.variant][self.score_type]
        primary = max(0.0, min(1.0, primary))

        return MetricResult(score=primary, details=results)
