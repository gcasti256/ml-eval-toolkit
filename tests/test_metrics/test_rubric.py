"""Tests for rubric-based scoring metric."""

import pytest

from ml_eval.metrics.rubric import RubricCriterion, RubricMetric, _evaluate_criterion


class TestEvaluateCriterion:
    def test_keyword_match_all(self) -> None:
        criterion = RubricCriterion(name="keywords", description="test", keywords=["python", "code"])
        score, details = _evaluate_criterion(criterion, "This is python code for testing")
        assert score == pytest.approx(1.0)
        assert details["keywords"]["score"] == pytest.approx(1.0)

    def test_keyword_partial(self) -> None:
        criterion = RubricCriterion(name="keywords", description="test", keywords=["python", "java", "rust"])
        score, details = _evaluate_criterion(criterion, "I love python programming")
        assert score == pytest.approx(1 / 3, abs=0.01)
        assert details["keywords"]["score"] == pytest.approx(1 / 3, abs=0.01)

    def test_keyword_none_found(self) -> None:
        criterion = RubricCriterion(name="keywords", description="test", keywords=["java", "rust"])
        score, _ = _evaluate_criterion(criterion, "I love python programming")
        assert score == 0.0

    def test_keyword_case_insensitive(self) -> None:
        criterion = RubricCriterion(name="keywords", description="test", keywords=["PYTHON"])
        score, _ = _evaluate_criterion(criterion, "python is great")
        assert score == pytest.approx(1.0)

    def test_min_length_passes(self) -> None:
        criterion = RubricCriterion(name="length", description="test", min_length=5)
        score, _ = _evaluate_criterion(criterion, "one two three four five")
        assert score == pytest.approx(1.0)

    def test_min_length_partial(self) -> None:
        criterion = RubricCriterion(name="length", description="test", min_length=10)
        score, _ = _evaluate_criterion(criterion, "only three words")
        # 3 words / 10 min_length = 0.3
        assert score == pytest.approx(0.3)

    def test_max_length_passes(self) -> None:
        criterion = RubricCriterion(name="length", description="test", max_length=10)
        score, _ = _evaluate_criterion(criterion, "short text")
        assert score == pytest.approx(1.0)

    def test_max_length_exceeds(self) -> None:
        criterion = RubricCriterion(name="length", description="test", max_length=2)
        score, details = _evaluate_criterion(criterion, "one two three four")
        assert score < 1.0
        assert details["max_length"]["passed"] is False

    def test_pattern_match_all(self) -> None:
        criterion = RubricCriterion(
            name="pattern", description="test", required_patterns=[r"\d+", r"[A-Z]"]
        )
        score, _ = _evaluate_criterion(criterion, "Has 123 and Capital")
        assert score == pytest.approx(1.0)

    def test_pattern_partial(self) -> None:
        criterion = RubricCriterion(
            name="pattern", description="test", required_patterns=[r"\d+", r"@"]
        )
        score, _ = _evaluate_criterion(criterion, "has 123 but no at-sign")
        assert score == pytest.approx(0.5)

    def test_no_criteria(self) -> None:
        criterion = RubricCriterion(name="empty", description="test")
        score, details = _evaluate_criterion(criterion, "anything")
        assert score == pytest.approx(1.0)
        assert details == {"note": "no criteria to check"}

    def test_combined_criteria(self) -> None:
        criterion = RubricCriterion(
            name="combo", description="test", keywords=["hello"], min_length=2
        )
        score, details = _evaluate_criterion(criterion, "hello world")
        # keywords: 1.0, min_length: 1.0 -> avg = 1.0
        assert score == pytest.approx(1.0)
        assert "keywords" in details
        assert "min_length" in details


class TestRubricMetric:
    def test_no_criteria(self) -> None:
        metric = RubricMetric(criteria=[])
        result = metric.compute("ref", "hyp")
        assert result.score == pytest.approx(1.0)

    def test_single_criterion_match(self) -> None:
        metric = RubricMetric(
            criteria=[{"name": "keywords", "description": "test", "keywords": ["hello"]}]
        )
        result = metric.compute("", "hello world")
        assert result.score == pytest.approx(1.0)

    def test_single_criterion_miss(self) -> None:
        metric = RubricMetric(
            criteria=[{"name": "keywords", "description": "test", "keywords": ["missing"]}]
        )
        result = metric.compute("", "hello world")
        assert result.score == pytest.approx(0.0)

    def test_weighted_criteria_exact(self) -> None:
        metric = RubricMetric(
            criteria=[
                {"name": "c1", "description": "test", "keywords": ["present"], "weight": 3.0},
                {"name": "c2", "description": "test", "keywords": ["missing"], "weight": 1.0},
            ]
        )
        result = metric.compute("", "the word present is here")
        # c1: raw=1.0, weighted=1.0*(3/4)=0.75; c2: raw=0.0, weighted=0.0*(1/4)=0.0
        assert result.score == pytest.approx(0.75)

    def test_equal_weights(self) -> None:
        metric = RubricMetric(
            criteria=[
                {"name": "c1", "description": "test", "keywords": ["yes"]},
                {"name": "c2", "description": "test", "keywords": ["no"]},
            ]
        )
        result = metric.compute("", "yes is present")
        # c1: 1.0 * (1/2) = 0.5; c2: 0.0 * (1/2) = 0.0
        assert result.score == pytest.approx(0.5)

    def test_details_per_criterion(self) -> None:
        metric = RubricMetric(
            criteria=[
                {"name": "alpha", "description": "test", "keywords": ["x"]},
                {"name": "beta", "description": "test", "min_length": 3},
            ]
        )
        result = metric.compute("", "x y z")
        assert "alpha" in result.details
        assert "beta" in result.details
        assert "raw_score" in result.details["alpha"]
        assert "weighted_score" in result.details["alpha"]
        assert "weight" in result.details["alpha"]

    def test_from_rubric_criterion_objects(self) -> None:
        criteria = [RubricCriterion(name="test", description="test", keywords=["hello"])]
        metric = RubricMetric(criteria=criteria)
        result = metric.compute("", "hello")
        assert result.score == pytest.approx(1.0)

    def test_name(self) -> None:
        assert RubricMetric().name == "rubric"

    def test_score_clamped_to_unit_range(self) -> None:
        metric = RubricMetric(
            criteria=[{"name": "c1", "description": "test", "keywords": ["a"]}]
        )
        result = metric.compute("", "a b c")
        assert 0.0 <= result.score <= 1.0
