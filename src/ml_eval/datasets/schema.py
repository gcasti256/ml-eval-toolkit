"""Dataset schema validation with Pydantic."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator


class Sample(BaseModel):
    """A single evaluation sample."""

    input: str
    expected_output: str
    actual_output: str = ""
    metadata: dict[str, Any] = {}

    @field_validator("input", "expected_output")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty or whitespace-only")
        return v


class DatasetSchema(BaseModel):
    """Schema for a complete evaluation dataset."""

    samples: list[Sample]
    name: str = ""
    description: str = ""
    version: str = "1.0"

    @field_validator("samples")
    @classmethod
    def must_have_samples(cls, v: list[Sample]) -> list[Sample]:
        if not v:
            raise ValueError("Dataset must contain at least one sample")
        return v

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]
