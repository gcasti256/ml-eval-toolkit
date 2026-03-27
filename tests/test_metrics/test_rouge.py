"""Tests for ROUGE metric implementation."""

import pytest

from ml_eval.metrics.rouge import ROUGEMetric, _f1, _lcs_length, _ngram_overlap


class TestLCSLength:
    def test_identical(self) -> None:
        assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_no_common(self) -> None:
        assert _lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_partial(self) -> None:
        assert _lcs_length(["a", "b", "c", "d"], ["a", "c", "d"]) == 3

    def test_empty(self) -> None:
        assert _lcs_length([], ["a", "b"]) == 0


class TestNgramOverlap:
    def test_perfect_overlap(self) -> None:
        tokens = ["the", "cat", "sat"]
        overlap, ref_count, _hyp_count = _ngram_overlap(tokens, tokens, 1)
        assert overlap == 3
        assert ref_count == 3

    def test_no_overlap(self) -> None:
        overlap, _, _ = _ngram_overlap(["a", "b"], ["c", "d"], 1)
        assert overlap == 0

    def test_bigram_overlap(self) -> None:
        ref = ["the", "cat", "sat", "on"]
        hyp = ["the", "cat", "ran", "on"]
        overlap, _ref_count, _hyp_count = _ngram_overlap(ref, hyp, 2)
        assert overlap == 1  # ("the", "cat")


class TestF1:
    def test_perfect(self) -> None:
        assert _f1(1.0, 1.0) == pytest.approx(1.0)

    def test_zero(self) -> None:
        assert _f1(0.0, 0.0) == 0.0

    def test_balanced(self) -> None:
        assert _f1(0.5, 0.5) == pytest.approx(0.5)


class TestROUGEMetric:
    def test_perfect_rouge_1(self) -> None:
        metric = ROUGEMetric(variant="rouge-1")
        result = metric.compute("the cat sat on the mat", "the cat sat on the mat")
        assert result.score == pytest.approx(1.0)

    def test_perfect_rouge_l(self) -> None:
        metric = ROUGEMetric(variant="rouge-l")
        result = metric.compute("the cat sat on the mat", "the cat sat on the mat")
        assert result.score == pytest.approx(1.0)

    def test_partial_rouge_1(self) -> None:
        metric = ROUGEMetric(variant="rouge-1")
        result = metric.compute("the cat sat on the mat", "the dog sat on the mat")
        assert 0.5 < result.score < 1.0

    def test_no_overlap_rouge_2(self) -> None:
        metric = ROUGEMetric(variant="rouge-2")
        result = metric.compute("hello world", "foo bar baz")
        assert result.score == 0.0

    def test_details_structure(self) -> None:
        metric = ROUGEMetric()
        result = metric.compute("hello world", "hello world")
        assert "rouge-1" in result.details
        assert "rouge-2" in result.details
        assert "rouge-l" in result.details
        assert "precision" in result.details["rouge-1"]
        assert "recall" in result.details["rouge-1"]
        assert "f1" in result.details["rouge-1"]

    def test_invalid_variant(self) -> None:
        with pytest.raises(ValueError, match="variant must be"):
            ROUGEMetric(variant="rouge-3")

    def test_recall_mode(self) -> None:
        metric = ROUGEMetric(variant="rouge-1", score_type="recall")
        result = metric.compute("hello world foo", "hello world")
        # Recall should be < 1 since hypothesis is shorter
        assert result.score < 1.0

    def test_empty_hypothesis(self) -> None:
        metric = ROUGEMetric(variant="rouge-l")
        result = metric.compute("hello world", "")
        assert result.score == 0.0
