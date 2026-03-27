"""Tests for API routes."""

import json
import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml_eval.api.app import create_app
from ml_eval.api.routes import get_db


@pytest.fixture
def client(db: sqlite3.Connection) -> TestClient:
    app = create_app()

    def override_db() -> Generator[sqlite3.Connection, None, None]:
        yield db

    app.dependency_overrides[get_db] = override_db
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
        assert 0 <= data["scores"]["bleu"] <= 1

    def test_evaluate_multiple_metrics(self, client: TestClient, sample_dataset: Path) -> None:
        response = client.post(
            "/api/v1/evaluate",
            json={
                "dataset_path": str(sample_dataset),
                "metrics": [
                    {"name": "bleu", "params": {"max_n": 2}},
                    {"name": "rouge"},
                ],
                "name": "multi_metric_test",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "bleu" in data["scores"]
        assert "rouge" in data["scores"]

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
        assert response.json()["count"] == 0

    def test_list_results_after_eval(self, client: TestClient, sample_dataset: Path) -> None:
        client.post(
            "/api/v1/evaluate",
            json={"dataset_path": str(sample_dataset), "metrics": [{"name": "rouge"}]},
        )
        response = client.get("/api/v1/results")
        assert response.json()["count"] == 1

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
        assert data["status"] == "completed"
        assert len(data["results"]) == 2
        assert "aggregated_scores" in data

    def test_get_nonexistent_result(self, client: TestClient) -> None:
        response = client.get("/api/v1/results/nonexistent-id")
        assert response.status_code == 404


class TestCompareEndpoint:
    def test_compare(self, client: TestClient, sample_dataset: Path) -> None:
        response = client.post(
            "/api/v1/compare",
            json={
                "dataset_path": str(sample_dataset),
                "configs": [
                    {
                        "dataset_path": str(sample_dataset),
                        "metrics": [{"name": "rouge"}],
                        "name": "config_a",
                    },
                    {
                        "dataset_path": str(sample_dataset),
                        "metrics": [{"name": "rouge"}],
                        "name": "config_b",
                    },
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["comparison"]) == 2
        assert "best_per_metric" in data

    def test_compare_missing_dataset(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/compare",
            json={
                "dataset_path": "/nonexistent.json",
                "configs": [
                    {"dataset_path": "/x.json", "metrics": [{"name": "bleu"}], "name": "a"},
                ],
            },
        )
        assert response.status_code == 400
