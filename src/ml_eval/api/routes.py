"""API route definitions."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator

from fastapi import APIRouter, Depends, HTTPException

from ml_eval import __version__
from ml_eval.api.schemas import (
    CompareRequest,
    CompareResponse,
    EvalRequest,
    EvalRunResponse,
    HealthResponse,
    RunDetailResponse,
    RunListResponse,
)
from ml_eval.config import EvalConfig, MetricConfig
from ml_eval.datasets.loader import load_dataset
from ml_eval.db import (
    get_aggregated_scores,
    get_connection,
    get_results_for_run,
    get_run,
    init_db,
    list_runs,
)
from ml_eval.evaluation.comparison import compare_configs
from ml_eval.evaluation.runner import EvalRunner

router = APIRouter()


def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Dependency: yield a database connection and close it after the request."""
    conn = get_connection()
    init_db(conn)
    try:
        yield conn
    finally:
        conn.close()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version=__version__)


@router.post("/evaluate", response_model=EvalRunResponse)
def run_evaluation(
    request: EvalRequest, conn: sqlite3.Connection = Depends(get_db)
) -> EvalRunResponse:
    """Run an evaluation with the given configuration."""
    try:
        dataset = load_dataset(request.dataset_path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    config = EvalConfig(
        dataset_path=request.dataset_path,
        metrics=[MetricConfig(name=m.name, params=m.params) for m in request.metrics],
        name=request.name,
        description=request.description,
    )

    try:
        runner = EvalRunner(conn, config)
        result = runner.run(dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    summary = result.summary()
    return EvalRunResponse(
        run_id=summary["run_id"],
        name=summary["name"],
        status="completed",
        scores=summary["scores"],
    )


@router.get("/results", response_model=RunListResponse)
def get_results(
    limit: int = 50, conn: sqlite3.Connection = Depends(get_db)
) -> RunListResponse:
    """List all evaluation runs."""
    runs = list_runs(conn, limit=limit)
    return RunListResponse(runs=runs, count=len(runs))


@router.get("/results/{run_id}", response_model=RunDetailResponse)
def get_result_detail(
    run_id: str, conn: sqlite3.Connection = Depends(get_db)
) -> RunDetailResponse:
    """Get detailed results for a specific run."""
    run = get_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    results = get_results_for_run(conn, run_id)
    aggregated = get_aggregated_scores(conn, run_id)

    return RunDetailResponse(
        run_id=run["id"],
        name=run["name"],
        description=run.get("description", ""),
        dataset_path=run["dataset_path"],
        status=run["status"],
        created_at=run["created_at"],
        config=run.get("config_json", {}),
        aggregated_scores=aggregated,
        results=results,
    )


@router.post("/compare", response_model=CompareResponse)
def run_comparison(
    request: CompareRequest, conn: sqlite3.Connection = Depends(get_db)
) -> CompareResponse:
    """Run a side-by-side comparison of multiple configs."""
    try:
        dataset = load_dataset(request.dataset_path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    configs: list[EvalConfig] = []
    for c in request.configs:
        configs.append(
            EvalConfig(
                dataset_path=request.dataset_path,
                metrics=[MetricConfig(name=m.name, params=m.params) for m in c.metrics],
                name=c.name,
                description=c.description,
            )
        )

    try:
        comparison = compare_configs(conn, dataset, configs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    table = comparison.summary_table()
    all_metrics: set[str] = set()
    for r in comparison.results:
        all_metrics.update(r.aggregated.keys())

    best: dict[str, str | None] = {}
    for m in all_metrics:
        best[m] = comparison.best_config(m)

    return CompareResponse(comparison=table, best_per_metric=best)
