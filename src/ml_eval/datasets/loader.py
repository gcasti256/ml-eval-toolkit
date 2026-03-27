"""Dataset loading from CSV, JSON, and JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml_eval.datasets.schema import DatasetSchema, Sample


def load_dataset(path: str | Path) -> DatasetSchema:
    """Load a dataset from a file.

    Supports CSV, JSON, and JSONL formats. All formats must contain
    at minimum 'input' and 'expected_output' columns/fields.

    Args:
        path: Path to the dataset file.

    Returns:
        Validated DatasetSchema instance.

    Raises:
        ValueError: If file format is unsupported or schema validation fails.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _load_csv(path)
    elif suffix == ".json":
        return _load_json(path)
    elif suffix == ".jsonl":
        return _load_jsonl(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv, .json, or .jsonl")


def _load_csv(path: Path) -> DatasetSchema:
    """Load dataset from CSV file."""
    df = pd.read_csv(path)
    _validate_columns(df.columns.tolist(), path)
    samples = _dataframe_to_samples(df)
    return DatasetSchema(samples=samples, name=path.stem)


def _load_json(path: Path) -> DatasetSchema:
    """Load dataset from JSON file.

    Accepts either:
    - A list of objects: [{"input": "...", "expected_output": "..."}, ...]
    - An object with "samples" key: {"samples": [...], "name": "..."}
    """
    with open(path) as f:
        data: Any = json.load(f)

    if isinstance(data, list):
        samples = [_dict_to_sample(d) for d in data]
        return DatasetSchema(samples=samples, name=path.stem)

    if isinstance(data, dict):
        if "samples" in data:
            samples = [_dict_to_sample(d) for d in data["samples"]]
            return DatasetSchema(
                samples=samples,
                name=data.get("name", path.stem),
                description=data.get("description", ""),
                version=data.get("version", "1.0"),
            )
        # Single object with required fields — treat as one-sample dataset
        if "input" in data and "expected_output" in data:
            return DatasetSchema(samples=[_dict_to_sample(data)], name=path.stem)

    raise ValueError(
        f"Invalid JSON structure in {path}. "
        "Expected a list of samples or an object with 'samples' key."
    )


def _load_jsonl(path: Path) -> DatasetSchema:
    """Load dataset from JSONL file (one JSON object per line)."""
    samples: list[Sample] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e
            samples.append(_dict_to_sample(data))

    if not samples:
        raise ValueError(f"No samples found in {path}")
    return DatasetSchema(samples=samples, name=path.stem)


def _validate_columns(columns: list[str], path: Path) -> None:
    """Validate that required columns exist."""
    required = {"input", "expected_output"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def _dataframe_to_samples(df: pd.DataFrame) -> list[Sample]:
    """Convert DataFrame rows to Sample objects."""
    samples: list[Sample] = []
    for _, row in df.iterrows():
        d = row.to_dict()
        samples.append(_dict_to_sample(d))
    return samples


def _dict_to_sample(d: dict[str, Any]) -> Sample:
    """Convert a dict to a Sample, extracting metadata from extra fields."""
    known_fields = {"input", "expected_output", "actual_output"}
    metadata = {k: v for k, v in d.items() if k not in known_fields}
    return Sample(
        input=str(d.get("input", "")),
        expected_output=str(d.get("expected_output", "")),
        actual_output=str(d.get("actual_output", "")),
        metadata=metadata,
    )
