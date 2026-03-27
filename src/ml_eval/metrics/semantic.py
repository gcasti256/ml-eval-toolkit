"""Semantic similarity metric using sentence-transformers."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ml_eval.metrics.base import BaseMetric, MetricResult


def _cosine_similarity(a: Any, b: Any) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class SemanticSimilarityMetric(BaseMetric):
    """Semantic similarity via sentence embeddings and cosine distance.

    Uses sentence-transformers to encode text and compute cosine similarity.
    The model is loaded lazily on first use.

    Args:
        model_name: Name of the sentence-transformers model. Default from env or all-MiniLM-L6-v2.
    """

    def __init__(self, model_name: str = "", **_: Any) -> None:
        self.model_name = model_name or os.environ.get(
            "ML_EVAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self._model: Any = None

    @property
    def name(self) -> str:
        return "semantic"

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def compute(self, reference: str, hypothesis: str) -> MetricResult:
        model = self._get_model()
        embeddings = model.encode([reference, hypothesis])
        sim = _cosine_similarity(embeddings[0], embeddings[1])
        # Cosine sim can be negative; clamp to [0, 1]
        score = max(0.0, min(1.0, sim))
        return MetricResult(
            score=score,
            details={"model": self.model_name, "raw_cosine_similarity": sim},
        )

    def compute_batch(
        self, references: list[str], hypotheses: list[str]
    ) -> list[MetricResult]:
        """Batch-optimized: encode all texts in one pass."""
        if len(references) != len(hypotheses):
            raise ValueError("Length mismatch between references and hypotheses")

        model = self._get_model()
        all_texts = references + hypotheses
        embeddings = model.encode(all_texts)
        n = len(references)
        ref_embeddings = embeddings[:n]
        hyp_embeddings = embeddings[n:]

        results = []
        for ref_emb, hyp_emb in zip(ref_embeddings, hyp_embeddings, strict=True):
            sim = _cosine_similarity(ref_emb, hyp_emb)
            score = max(0.0, min(1.0, sim))
            results.append(
                MetricResult(
                    score=score,
                    details={"model": self.model_name, "raw_cosine_similarity": sim},
                )
            )
        return results
