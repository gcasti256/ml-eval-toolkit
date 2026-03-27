"""Tests for semantic similarity metric."""

import numpy as np
import pytest

from ml_eval.metrics.semantic import SemanticSimilarityMetric, _cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        result = _cosine_similarity(a, a)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = _cosine_similarity(a, b)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        result = _cosine_similarity(a, b)
        assert result == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        result = _cosine_similarity(a, b)
        assert result == 0.0

    def test_both_zero_vectors(self) -> None:
        a = np.array([0.0, 0.0])
        result = _cosine_similarity(a, a)
        assert result == 0.0

    def test_high_dimensional(self) -> None:
        rng = np.random.RandomState(42)
        a = rng.randn(384)
        result = _cosine_similarity(a, a)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_known_angle(self) -> None:
        # 45-degree angle: cos(45) ≈ 0.7071
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 1.0])
        result = _cosine_similarity(a, b)
        assert result == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-6)

    def test_accepts_lists(self) -> None:
        result = _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert result == pytest.approx(1.0, abs=1e-6)


class TestSemanticSimilarityMetric:
    def test_name(self) -> None:
        metric = SemanticSimilarityMetric()
        assert metric.name == "semantic"

    def test_default_model(self) -> None:
        metric = SemanticSimilarityMetric()
        assert metric.model_name == "all-MiniLM-L6-v2"

    def test_custom_model(self) -> None:
        metric = SemanticSimilarityMetric(model_name="all-mpnet-base-v2")
        assert metric.model_name == "all-mpnet-base-v2"

    def test_lazy_model_loading(self) -> None:
        metric = SemanticSimilarityMetric()
        assert metric._model is None
