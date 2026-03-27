"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MetricConfigRequest(BaseModel):
    name: str
    params: dict[str, Any] = {}


class EvalRequest(BaseModel):
    dataset_path: str
    metrics: list[MetricConfigRequest]
    name: str = "api_eval"
    description: str = ""


class CompareRequest(BaseModel):
    dataset_path: str
    configs: list[EvalRequest]


class EvalRunResponse(BaseModel):
    run_id: str
    name: str
    status: str
    scores: dict[str, float]


class RunDetailResponse(BaseModel):
    run_id: str
    name: str
    description: str
    dataset_path: str
    status: str
    created_at: str
    config: dict[str, Any]
    aggregated_scores: dict[str, dict[str, float]]
    results: list[dict[str, Any]]


class CompareResponse(BaseModel):
    comparison: list[dict[str, Any]]
    best_per_metric: dict[str, str | None]


class RunListResponse(BaseModel):
    runs: list[dict[str, Any]]
    count: int


class HealthResponse(BaseModel):
    status: str
    version: str
