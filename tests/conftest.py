"""Shared test fixtures."""

import sqlite3

import pytest

from ml_eval.db import init_db


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn
