"""Tests for dataset loading."""

import json
from pathlib import Path

import pytest

from ml_eval.datasets.loader import load_dataset


@pytest.fixture
def json_dataset(tmp_path: Path) -> Path:
    data = [
        {"input": "What is 2+2?", "expected_output": "4"},
        {"input": "What is 3+3?", "expected_output": "6"},
    ]
    path = tmp_path / "test.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def jsonl_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "test.jsonl"
    lines = [
        json.dumps({"input": "Q1", "expected_output": "A1"}),
        json.dumps({"input": "Q2", "expected_output": "A2"}),
        json.dumps({"input": "Q3", "expected_output": "A3"}),
    ]
    path.write_text("\n".join(lines))
    return path


@pytest.fixture
def csv_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "test.csv"
    path.write_text("input,expected_output\nQ1,A1\nQ2,A2\n")
    return path


class TestLoadDataset:
    def test_load_json_list(self, json_dataset: Path) -> None:
        ds = load_dataset(json_dataset)
        assert len(ds) == 2
        assert ds[0].input == "What is 2+2?"
        assert ds[0].expected_output == "4"

    def test_load_json_with_samples_key(self, tmp_path: Path) -> None:
        data = {
            "name": "test_ds",
            "samples": [{"input": "Q", "expected_output": "A"}],
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        ds = load_dataset(path)
        assert ds.name == "test_ds"
        assert len(ds) == 1

    def test_load_jsonl(self, jsonl_dataset: Path) -> None:
        ds = load_dataset(jsonl_dataset)
        assert len(ds) == 3

    def test_load_csv(self, csv_dataset: Path) -> None:
        ds = load_dataset(csv_dataset)
        assert len(ds) == 2
        assert ds[0].input == "Q1"

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path.json")

    def test_unsupported_format(self, tmp_path: Path) -> None:
        path = tmp_path / "test.xml"
        path.write_text("<data/>")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset(path)

    def test_metadata_extraction(self, tmp_path: Path) -> None:
        data = [{"input": "Q", "expected_output": "A", "category": "math", "difficulty": "easy"}]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        ds = load_dataset(path)
        assert ds[0].metadata["category"] == "math"
        assert ds[0].metadata["difficulty"] == "easy"

    def test_invalid_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"input": "Q", "expected_output": "A"}\nnot json\n')
        with pytest.raises(ValueError, match="Invalid JSON on line 2"):
            load_dataset(path)
