"""SQLite storage for evaluation results."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any


def _get_db_path() -> str:
    return os.environ.get("ML_EVAL_DB_PATH", "./ml_eval_results.db")


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled."""
    path = db_path or _get_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            dataset_path TEXT NOT NULL,
            config_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT DEFAULT 'pending'
        );

        CREATE TABLE IF NOT EXISTS eval_results (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES eval_runs(id),
            metric_name TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            input_text TEXT NOT NULL,
            expected_output TEXT NOT NULL,
            actual_output TEXT DEFAULT '',
            score REAL NOT NULL,
            details_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS baselines (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES eval_runs(id),
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_results_run_id ON eval_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_results_metric ON eval_results(metric_name);
    """)
    conn.commit()


def store_run(
    conn: sqlite3.Connection,
    name: str,
    description: str,
    dataset_path: str,
    config: dict[str, Any],
) -> str:
    """Store an evaluation run record. Returns the run ID."""
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO eval_runs (id, name, description, dataset_path, config_json, created_at, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, name, description, dataset_path, json.dumps(config), now, "running"),
    )
    conn.commit()
    return run_id


def store_result(
    conn: sqlite3.Connection,
    run_id: str,
    metric_name: str,
    sample_index: int,
    input_text: str,
    expected_output: str,
    actual_output: str,
    score: float,
    details: dict[str, Any] | None = None,
) -> str:
    """Store a single evaluation result. Returns the result ID."""
    result_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO eval_results (id, run_id, metric_name, sample_index, input_text, expected_output, actual_output, score, details_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            result_id,
            run_id,
            metric_name,
            sample_index,
            input_text,
            expected_output,
            actual_output,
            score,
            json.dumps(details or {}),
            now,
        ),
    )
    conn.commit()
    return result_id


def complete_run(conn: sqlite3.Connection, run_id: str) -> None:
    """Mark a run as completed."""
    conn.execute("UPDATE eval_runs SET status = 'completed' WHERE id = ?", (run_id,))
    conn.commit()


def fail_run(conn: sqlite3.Connection, run_id: str) -> None:
    """Mark a run as failed."""
    conn.execute("UPDATE eval_runs SET status = 'failed' WHERE id = ?", (run_id,))
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Get a single run by ID."""
    row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def list_runs(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    """List recent evaluation runs."""
    rows = conn.execute(
        "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_results_for_run(
    conn: sqlite3.Connection, run_id: str
) -> list[dict[str, Any]]:
    """Get all results for a given run."""
    rows = conn.execute(
        "SELECT * FROM eval_results WHERE run_id = ? ORDER BY metric_name, sample_index",
        (run_id,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_aggregated_scores(
    conn: sqlite3.Connection, run_id: str
) -> dict[str, dict[str, float]]:
    """Get aggregated scores per metric for a run."""
    rows = conn.execute(
        """
        SELECT metric_name,
               AVG(score) as avg_score,
               MIN(score) as min_score,
               MAX(score) as max_score,
               COUNT(*) as count
        FROM eval_results
        WHERE run_id = ?
        GROUP BY metric_name
        """,
        (run_id,),
    ).fetchall()
    return {
        row["metric_name"]: {
            "avg": row["avg_score"],
            "min": row["min_score"],
            "max": row["max_score"],
            "count": row["count"],
        }
        for row in rows
    }


def save_baseline(conn: sqlite3.Connection, run_id: str, name: str) -> str:
    """Save a run as a named baseline."""
    baseline_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO baselines (id, run_id, name, created_at) VALUES (?, ?, ?, ?)",
        (baseline_id, run_id, name, now),
    )
    conn.commit()
    return baseline_id


def get_baseline(conn: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    """Get a baseline by name."""
    row = conn.execute("SELECT * FROM baselines WHERE name = ?", (name,)).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict."""
    d: dict[str, Any] = dict(row)
    for key in ("config_json", "details_json"):
        if key in d and isinstance(d[key], str):
            d[key] = json.loads(d[key])
    return d
