"""Tests for ROUGE metric."""

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
        assert _lcs_length(["a"], []) == 0


class TestNgramOverlap:
    def test_perfect_overlap(self) -> None:
        tokens = ["the", "cat", "sat"]
        overlap, ref_count, hyp_count = _ngram_overlap(tokens, tokens, 1)
        assert overlap == 3
        assert ref_count == 3
        assert hyp_count == 3

    def test_no_overlap(self) -> None:
        overlap, _, _ = _ngram_overlap(["a", "b"], ["c", "d"], 1)
        assert overlap == 0

    def test_bigram_partial(self) -> None:
        ref = ["the", "cat", "sat", "on"]
        hyp = ["the", "cat", "ran", "on"]
        overlap, ref_count, hyp_count = _ngram_overlap(ref, hyp, 2)
        assert overlap == 1  # ("the", "cat")
        assert ref_count == 3
        assert hyp_count == 3


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
        # "the" appears twice in ref; "dog" replaces "cat" = 5/6 overlap
        result = metric.compute("the cat sat on the mat", "the dog sat on the mat")
        assert result.score == pytest.approx(5 / 6, abs=0.01)

    def test_no_overlap_rouge_2(self) -> None:
        metric = ROUGEMetric(variant="rouge-2")
        result = metric.compute("hello world", "foo bar baz")
        assert result.score == 0.0

    def test_details_structure(self) -> None:
        metric = ROUGEMetric()
        result = metric.compute("hello world", "hello world")
        for variant in ("rouge-1", "rouge-2", "rouge-l"):
            assert variant in result.details
            assert set(result.details[variant].keys()) == {"precision", "recall", "f1"}

    def test_invalid_variant(self) -> None:
        with pytest.raises(ValueError, match="variant must be"):
            ROUGEMetric(variant="rouge-3")

    def test_invalid_score_type(self) -> None:
        with pytest.raises(ValueError, match="score_type must be"):
            ROUGEMetric(score_type="accuracy")

    def test_recall_mode(self) -> None:
        metric = ROUGEMetric(variant="rouge-1", score_type="recall")
        result = metric.compute("hello world foo", "hello world")
        assert result.score == pytest.approx(2 / 3, abs=0.01)

    def test_precision_mode(self) -> None:
        metric = ROUGEMetric(variant="rouge-1", score_type="precision")
        result = metric.compute("hello world", "hello world extra")
        assert result.score == pytest.approx(2 / 3, abs=0.01)

    def test_empty_hypothesis(self) -> None:
        metric = ROUGEMetric(variant="rouge-l")
        result = metric.compute("hello world", "")
        assert result.score == 0.0

    def test_name(self) -> None:
        assert ROUGEMetric().name == "rouge"
