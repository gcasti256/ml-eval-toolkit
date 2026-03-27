"""Tests for API routes."""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml_eval.api.app import create_app
from ml_eval.api.routes import get_db
from ml_eval.db import init_db


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


@pytest.fixture
def client(db: sqlite3.Connection) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    data = [
        {"input": "Q1", "expected_output": "answer one", "actual_output": "answer one"},
        {"input": "Q2", "expected_output": "answer two", "actual_output": "answer two"},
    ]
    path = tmp_path / "test.json"
    path.write_text(json.dumps(data))
    return path


class TestHealthEndpoint:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestEvaluateEndpoint:
    def test_evaluate(self, client: TestClient, sample_dataset: Path) -> None:
        response = client.post(
            "/api/v1/evaluate",
            json={
                "dataset_path": str(sample_dataset),
                "metrics": [{"name": "bleu", "params": {"max_n": 2}}],
                "name": "api_test",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "bleu" in data["scores"]

    def test_evaluate_missing_dataset(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/evaluate",
            json={
                "dataset_path": "/nonexistent.json",
                "metrics": [{"name": "bleu"}],
            },
        )
        assert response.status_code == 400

    def test_evaluate_invalid_metric(self, client: TestClient, sample_dataset: Path) -> None:
        response = client.post(
            "/api/v1/evaluate",
            json={
                "dataset_path": str(sample_dataset),
                "metrics": [{"name": "nonexistent"}],
            },
        )
        assert response.status_code == 400


class TestResultsEndpoint:
    def test_list_results_empty(self, client: TestClient) -> None:
        response = client.get("/api/v1/results")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0

    def test_list_results_after_eval(self, client: TestClient, sample_dataset: Path) -> None:
        # Run an evaluation first
        client.post(
            "/api/v1/evaluate",
            json={
                "dataset_path": str(sample_dataset),
                "metrics": [{"name": "rouge"}],
            },
        )
        response = client.get("/api/v1/results")
        data = response.json()
        assert data["count"] == 1

    def test_get_result_detail(self, client: TestClient, sample_dataset: Path) -> None:
        eval_response = client.post(
            "/api/v1/evaluate",
            json={
                "dataset_path": str(sample_dataset),
                "metrics": [{"name": "bleu", "params": {"max_n": 2}}],
                "name": "detail_test",
            },
        )
        run_id = eval_response.json()["run_id"]

        response = client.get(f"/api/v1/results/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "detail_test"
        assert len(data["results"]) == 2

    def test_get_nonexistent_result(self, client: TestClient) -> None:
        response = client.get("/api/v1/results/nonexistent-id")
        assert response.status_code == 404
