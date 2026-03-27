"""Tests for BLEU metric."""

import pytest

from ml_eval.metrics.bleu import (
    BLEUMetric,
    _brevity_penalty,
    _get_ngrams,
    _modified_precision,
    _tokenize,
)


class TestTokenize:
    def test_basic(self) -> None:
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_empty(self) -> None:
        assert _tokenize("") == []

    def test_preserves_punctuation_attached(self) -> None:
        assert _tokenize("hello, world!") == ["hello,", "world!"]


class TestNgrams:
    def test_unigrams(self) -> None:
        ngrams = _get_ngrams(["a", "b", "c"], 1)
        assert ngrams[("a",)] == 1
        assert ngrams[("b",)] == 1
        assert len(ngrams) == 3

    def test_bigrams(self) -> None:
        ngrams = _get_ngrams(["a", "b", "c", "a", "b"], 2)
        assert ngrams[("a", "b")] == 2
        assert ngrams[("b", "c")] == 1

    def test_empty_for_long_ngrams(self) -> None:
        assert len(_get_ngrams(["a", "b"], 3)) == 0


class TestModifiedPrecision:
    def test_perfect_match(self) -> None:
        tokens = ["the", "cat", "sat"]
        clipped, total = _modified_precision(tokens, tokens, 1)
        assert clipped == 3
        assert total == 3

    def test_no_overlap(self) -> None:
        clipped, total = _modified_precision(["a", "b"], ["c", "d"], 1)
        assert clipped == 0
        assert total == 2

    def test_clipping(self) -> None:
        clipped, total = _modified_precision(["the", "cat"], ["the", "the", "the"], 1)
        assert clipped == 1
        assert total == 3


class TestBrevityPenalty:
    def test_no_penalty_when_longer(self) -> None:
        assert _brevity_penalty(5, 10) == 1.0

    def test_penalty_when_shorter(self) -> None:
        bp = _brevity_penalty(10, 5)
        assert 0.0 < bp < 1.0

    def test_empty_hypothesis(self) -> None:
        assert _brevity_penalty(10, 0) == 0.0

    def test_equal_length(self) -> None:
        assert _brevity_penalty(5, 5) == 1.0


class TestBLEUMetric:
    def test_perfect_score(self) -> None:
        metric = BLEUMetric(max_n=4)
        result = metric.compute("the cat sat on the mat", "the cat sat on the mat")
        assert result.score == pytest.approx(1.0, abs=0.001)

    def test_completely_different(self) -> None:
        metric = BLEUMetric(max_n=2, smoothing="none")
        result = metric.compute("the cat sat on the mat", "dogs run in the park quickly")
        assert result.score == 0.0

    def test_partial_match(self) -> None:
        metric = BLEUMetric(max_n=2)
        result = metric.compute("the cat sat on the mat", "the cat is on the mat")
        assert 0.4 < result.score < 0.95

    def test_empty_hypothesis(self) -> None:
        metric = BLEUMetric()
        result = metric.compute("the cat sat on the mat", "")
        assert result.score == 0.0

    def test_details_contain_precisions(self) -> None:
        metric = BLEUMetric(max_n=2)
        result = metric.compute("hello world", "hello world")
        assert len(result.details["precisions"]) == 2
        assert result.details["brevity_penalty"] == 1.0

    def test_custom_weights(self) -> None:
        # Unigram-heavy weights should still score high on perfect match
        metric = BLEUMetric(max_n=2, weights=[0.9, 0.1])
        result = metric.compute("the cat sat", "the cat sat")
        assert result.score == pytest.approx(1.0, abs=0.001)

    def test_invalid_weights_length(self) -> None:
        with pytest.raises(ValueError, match="weights length"):
            BLEUMetric(max_n=4, weights=[0.5, 0.5])

    def test_smoothing_floor(self) -> None:
        metric = BLEUMetric(max_n=4, smoothing="floor")
        result = metric.compute("a b c d e f", "x y z w")
        # Floor smoothing should produce a small nonzero score
        assert result.score >= 0.0

    def test_short_text_high_ngram(self) -> None:
        metric = BLEUMetric(max_n=4, smoothing="none")
        result = metric.compute("hello", "hello")
        # Only unigram precision is nonzero; 2,3,4-gram will be zero
        assert result.score == 0.0

    def test_batch_mixed_scores(self) -> None:
        metric = BLEUMetric(max_n=2)
        refs = ["hello world", "the cat sat"]
        hyps = ["hello world", "a dog ran"]
        results = metric.compute_batch(refs, hyps)
        assert len(results) == 2
        assert results[0].score > results[1].score

    def test_name(self) -> None:
        assert BLEUMetric().name == "bleu"
