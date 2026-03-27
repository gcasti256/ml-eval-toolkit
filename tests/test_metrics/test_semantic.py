"""Tests for semantic similarity metric."""

import numpy as np
import pytest

from ml_eval.metrics.semantic import _cosine_similarity


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

    def test_high_dimensional(self) -> None:
        rng = np.random.RandomState(42)
        a = rng.randn(384)
        result = _cosine_similarity(a, a)
        assert result == pytest.approx(1.0, abs=1e-5)
