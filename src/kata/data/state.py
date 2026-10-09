"""Per-item pipeline status, so every stage is resumable and idempotent."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    stage VARCHAR, key VARCHAR, status VARCHAR, attempts INTEGER,
    error VARCHAR, info JSON, updated_at TIMESTAMP,
    PRIMARY KEY (stage, key));
CREATE TABLE IF NOT EXISTS runs (
    stage VARCHAR, started_at TIMESTAMP, seconds DOUBLE, args VARCHAR, summary JSON);
"""


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute(SCHEMA)

    def close(self) -> None:
        self.con.close()

    def done(self, stage: str, statuses: tuple[str, ...] = ("ok",)) -> set[str]:
        rows = self.con.execute(
            f"SELECT key FROM items WHERE stage = ? AND status IN ({','.join('?' * len(statuses))})",
            [stage, *statuses]).fetchall()
        return {r[0] for r in rows}

    def mark(self, stage: str, key: str, status: str, error: str | None = None,
             info: dict | None = None) -> None:
        self.con.execute(
            """INSERT INTO items VALUES (?, ?, ?, 1, ?, ?, now())
               ON CONFLICT (stage, key) DO UPDATE SET status = excluded.status,
                 attempts = items.attempts + 1, error = excluded.error, info = excluded.info,
                 updated_at = excluded.updated_at""",
            [stage, key, status, error, json.dumps(info) if info is not None else None])

    def info(self, stage: str) -> dict[str, dict]:
        rows = self.con.execute(
            "SELECT key, info FROM items WHERE stage = ? AND status = 'ok'", [stage]).fetchall()
        return {k: (json.loads(v) if v else {}) for k, v in rows}

    def counts(self, stage: str) -> dict[str, int]:
        rows = self.con.execute(
            "SELECT status, count(*) FROM items WHERE stage = ? GROUP BY status", [stage]).fetchall()
        return dict(rows)

    def errors(self, stage: str, limit: int = 20) -> list[tuple[str, str]]:
        return self.con.execute(
            "SELECT key, error FROM items WHERE stage = ? AND status <> 'ok' LIMIT ?",
            [stage, limit]).fetchall()

    @contextmanager
    def run(self, stage: str, args: str = ""):
        started = time.time()
        summary: dict = {}
        yield summary
        self.con.execute(
            "INSERT INTO runs VALUES (?, to_timestamp(?), ?, ?, ?)",
            [stage, started, time.time() - started, args, json.dumps(summary)])
