"""Evaluation configuration and YAML loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MetricConfig:
    """Configuration for a single metric."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    dataset_path: str
    metrics: list[MetricConfig] = field(default_factory=list)
    name: str = ""
    description: str = ""
    output_format: str = "json"
    judge_model: str = ""
    embedding_model: str = ""

    def __post_init__(self) -> None:
        if not self.judge_model:
            self.judge_model = os.environ.get("ML_EVAL_JUDGE_MODEL", "gpt-4o-mini")
        if not self.embedding_model:
            self.embedding_model = os.environ.get("ML_EVAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")



def load_config(path: str | Path) -> EvalConfig:
    """Load evaluation config from a YAML file."""
    path = Path(path)
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    metrics = [
        MetricConfig(name=m["name"], params=m.get("params", {}))
        for m in raw.get("metrics", [])
    ]

    return EvalConfig(
        dataset_path=raw.get("dataset_path", ""),
        metrics=metrics,
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        output_format=raw.get("output_format", "json"),
        judge_model=raw.get("judge_model", ""),
        embedding_model=raw.get("embedding_model", ""),
    )


