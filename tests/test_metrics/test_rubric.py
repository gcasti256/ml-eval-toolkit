"""Tests for rubric-based scoring metric."""


from ml_eval.metrics.rubric import RubricCriterion, RubricMetric, _evaluate_criterion


class TestEvaluateCriterion:
    def test_keyword_match(self) -> None:
        criterion = RubricCriterion(name="keywords", description="test", keywords=["python", "code"])
        score, _details = _evaluate_criterion(criterion, "This is python code for testing")
        assert score == 1.0  # both keywords present

    def test_keyword_partial(self) -> None:
        criterion = RubricCriterion(name="keywords", description="test", keywords=["python", "java", "rust"])
        score, _details = _evaluate_criterion(criterion, "I love python programming")
        assert 0.0 < score < 1.0

    def test_min_length(self) -> None:
        criterion = RubricCriterion(name="length", description="test", min_length=5)
        score, _ = _evaluate_criterion(criterion, "one two three four five")
        assert score == 1.0

    def test_min_length_fails(self) -> None:
        criterion = RubricCriterion(name="length", description="test", min_length=10)
        score, _ = _evaluate_criterion(criterion, "only three words")
        assert score < 1.0

    def test_pattern_match(self) -> None:
        criterion = RubricCriterion(
            name="pattern", description="test", required_patterns=[r"\d+", r"[A-Z]"]
        )
        score, _ = _evaluate_criterion(criterion, "Has 123 and Capital")
        assert score == 1.0

    def test_no_criteria(self) -> None:
        criterion = RubricCriterion(name="empty", description="test")
        score, _details = _evaluate_criterion(criterion, "anything")
        assert score == 1.0


class TestRubricMetric:
    def test_no_criteria(self) -> None:
        metric = RubricMetric(criteria=[])
        result = metric.compute("ref", "hyp")
        assert result.score == 1.0

    def test_single_criterion(self) -> None:
        metric = RubricMetric(
            criteria=[{"name": "keywords", "description": "test", "keywords": ["hello"]}]
        )
        result = metric.compute("", "hello world")
        assert result.score > 0.5

    def test_weighted_criteria(self) -> None:
        metric = RubricMetric(
            criteria=[
                {"name": "c1", "description": "test", "keywords": ["present"], "weight": 3.0},
                {"name": "c2", "description": "test", "keywords": ["missing"], "weight": 1.0},
            ]
        )
        result = metric.compute("", "the word present is here")
        # c1 matches (weight 3), c2 doesn't (weight 1)
        assert result.score > 0.5

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

    def test_from_rubric_criterion_objects(self) -> None:
        criteria = [RubricCriterion(name="test", description="test", keywords=["hello"])]
        metric = RubricMetric(criteria=criteria)
        result = metric.compute("", "hello")
        assert result.score > 0.0
