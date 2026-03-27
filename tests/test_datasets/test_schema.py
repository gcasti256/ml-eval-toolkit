"""Tests for dataset schema validation."""

import pytest
from pydantic import ValidationError

from ml_eval.datasets.schema import DatasetSchema, Sample


class TestSample:
    def test_valid_sample(self) -> None:
        s = Sample(input="question", expected_output="answer")
        assert s.input == "question"
        assert s.actual_output == ""
        assert s.metadata == {}

    def test_with_actual_output(self) -> None:
        s = Sample(input="q", expected_output="a", actual_output="b")
        assert s.actual_output == "b"

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Sample(input="", expected_output="answer")

    def test_whitespace_only_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Sample(input="   ", expected_output="answer")

    def test_metadata(self) -> None:
        s = Sample(input="q", expected_output="a", metadata={"tag": "math"})
        assert s.metadata["tag"] == "math"


class TestDatasetSchema:
    def test_valid_dataset(self) -> None:
        ds = DatasetSchema(
            samples=[Sample(input="q1", expected_output="a1")],
            name="test",
        )
        assert len(ds) == 1
        assert ds[0].input == "q1"

    def test_empty_samples_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DatasetSchema(samples=[])

    def test_name_default(self) -> None:
        ds = DatasetSchema(samples=[Sample(input="q", expected_output="a")])
        assert ds.name == ""
