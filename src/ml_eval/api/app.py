"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from ml_eval import __version__
from ml_eval.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="ML Eval Toolkit",
        description="ML model evaluation and benchmarking API",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
