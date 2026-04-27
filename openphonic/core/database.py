from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class JobRecord:
    id: str
    status: str
    original_filename: str
    input_path: str
    output_path: str | None
    transcript_path: str | None
    error_message: str | None
    current_stage: str | None
    progress: int
    config_json: str
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        try:
            data["config"] = json.loads(self.config_json)
        except json.JSONDecodeError:
            data["config"] = {}
        return data


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  original_filename TEXT NOT NULL,
  input_path TEXT NOT NULL,
  output_path TEXT,
  transcript_path TEXT,
  error_message TEXT,
  current_stage TEXT,
  progress INTEGER NOT NULL DEFAULT 0,
  config_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
"""

UPDATE_FIELDS = {
    "status",
    "output_path",
    "transcript_path",
    "error_message",
    "current_stage",
    "progress",
    "started_at",
    "completed_at",
}


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Path) -> None:
    with connect(db_path) as connection:
        connection.executescript(SCHEMA)


def create_job(
    db_path: Path,
    *,
    job_id: str,
    original_filename: str,
    input_path: Path,
    config: dict[str, Any] | None = None,
) -> JobRecord:
    now = utc_now()
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
              id, status, original_filename, input_path, progress,
              config_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                "queued",
                original_filename,
                str(input_path),
                0,
                json.dumps(config or {}, sort_keys=True),
                now,
                now,
            ),
        )
    record = get_job(db_path, job_id)
    if record is None:  # pragma: no cover - impossible after successful insert
        raise RuntimeError(f"Failed to create job {job_id}")
    return record


def get_job(db_path: Path, job_id: str) -> JobRecord | None:
    with connect(db_path) as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return JobRecord(**dict(row)) if row else None


def list_jobs(db_path: Path, limit: int = 100) -> list[JobRecord]:
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [JobRecord(**dict(row)) for row in rows]


def update_job(db_path: Path, job_id: str, **fields: Any) -> JobRecord:
    unknown = set(fields) - UPDATE_FIELDS
    if unknown:
        raise ValueError(f"Unknown job update fields: {', '.join(sorted(unknown))}")
    fields["updated_at"] = utc_now()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [job_id]
    with connect(db_path) as connection:
        connection.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", values)
    record = get_job(db_path, job_id)
    if record is None:
        raise KeyError(job_id)
    return record
