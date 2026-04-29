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


@dataclass(frozen=True)
class RetryClaim:
    previous: JobRecord
    current: JobRecord


@dataclass(frozen=True)
class RetentionClaim:
    previous: JobRecord
    current: JobRecord


RETENTION_CLAIM_STATUSES = {
    "succeeded": "retention_cleanup_succeeded",
    "failed": "retention_cleanup_failed",
}
RETENTION_ORIGINAL_STATUSES = {
    claim_status: original_status
    for original_status, claim_status in RETENTION_CLAIM_STATUSES.items()
}


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


def list_jobs_by_status(db_path: Path, statuses: tuple[str, ...]) -> list[JobRecord]:
    if not statuses:
        return []
    placeholders = ", ".join("?" for _ in statuses)
    with connect(db_path) as connection:
        rows = connection.execute(
            f"SELECT * FROM jobs WHERE status IN ({placeholders}) ORDER BY created_at ASC",
            statuses,
        ).fetchall()
    return [JobRecord(**dict(row)) for row in rows]


def list_completed_jobs_before(db_path: Path, cutoff: str) -> list[JobRecord]:
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT * FROM jobs
            WHERE status IN ('succeeded', 'failed')
              AND completed_at IS NOT NULL
              AND completed_at < ?
            ORDER BY completed_at ASC
            """,
            (cutoff,),
        ).fetchall()
    return [JobRecord(**dict(row)) for row in rows]


def list_retention_cleanup_candidates(
    db_path: Path,
    cutoff: str,
    claim_stale_cutoff: str,
) -> list[JobRecord]:
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT * FROM jobs
            WHERE completed_at IS NOT NULL
              AND (
                (
                  status IN ('succeeded', 'failed')
                  AND completed_at < ?
                )
                OR (
                  status IN ('retention_cleanup_succeeded', 'retention_cleanup_failed')
                  AND completed_at < ?
                  AND updated_at < ?
                )
              )
            ORDER BY completed_at ASC
            """,
            (cutoff, cutoff, claim_stale_cutoff),
        ).fetchall()
    return [JobRecord(**dict(row)) for row in rows]


def claim_completed_job_for_retention(
    db_path: Path,
    job_id: str,
    cutoff: str,
    claim_stale_cutoff: str | None = None,
) -> RetentionClaim | None:
    with connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if claim_stale_cutoff is None:
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE id = ?
                  AND status IN ('succeeded', 'failed')
                  AND completed_at IS NOT NULL
                  AND completed_at < ?
                """,
                (job_id, cutoff),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE id = ?
                  AND completed_at IS NOT NULL
                  AND (
                    (
                      status IN ('succeeded', 'failed')
                      AND completed_at < ?
                    )
                    OR (
                      status IN ('retention_cleanup_succeeded', 'retention_cleanup_failed')
                      AND completed_at < ?
                      AND updated_at < ?
                    )
                  )
                """,
                (job_id, cutoff, cutoff, claim_stale_cutoff),
            ).fetchone()
        if row is None:
            return None

        record = JobRecord(**dict(row))
        now = utc_now()

        if record.status in RETENTION_CLAIM_STATUSES:
            previous = record
            claim_status = RETENTION_CLAIM_STATUSES[record.status]
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = ?,
                    updated_at = ?
                WHERE id = ?
                  AND status = ?
                  AND completed_at IS NOT NULL
                  AND completed_at < ?
                """,
                (claim_status, now, job_id, record.status, cutoff),
            )
        else:
            previous_data = asdict(record)
            previous_data["status"] = RETENTION_ORIGINAL_STATUSES[record.status]
            previous = JobRecord(**previous_data)
            claim_status = record.status
            cursor = connection.execute(
                """
                UPDATE jobs
                SET updated_at = ?
                WHERE id = ?
                  AND status = ?
                  AND updated_at = ?
                  AND completed_at IS NOT NULL
                  AND completed_at < ?
                  AND updated_at < ?
                """,
                (now, job_id, record.status, record.updated_at, cutoff, claim_stale_cutoff),
            )
        if cursor.rowcount != 1:
            return None
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:  # pragma: no cover - impossible after successful update
            raise KeyError(job_id)
        current = JobRecord(**dict(row))
        if current.status != claim_status:  # pragma: no cover - guarded by BEGIN IMMEDIATE
            return None
        return RetentionClaim(previous=previous, current=current)


def delete_retention_claim(db_path: Path, claim: RetentionClaim) -> bool:
    with connect(db_path) as connection:
        cursor = connection.execute(
            """
            DELETE FROM jobs
            WHERE id = ?
              AND status = ?
              AND updated_at = ?
            """,
            (claim.current.id, claim.current.status, claim.current.updated_at),
        )
        return cursor.rowcount == 1


def restore_retention_claim(db_path: Path, claim: RetentionClaim) -> bool:
    with connect(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET status = ?,
                updated_at = ?
            WHERE id = ?
              AND status = ?
              AND updated_at = ?
            """,
            (
                claim.previous.status,
                claim.previous.updated_at,
                claim.current.id,
                claim.current.status,
                claim.current.updated_at,
            ),
        )
        return cursor.rowcount == 1


def claim_failed_job_for_retry(db_path: Path, job_id: str) -> RetryClaim | None:
    with connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        previous = JobRecord(**dict(row))
        if previous.status != "failed":
            return None

        now = utc_now()
        connection.execute(
            """
            UPDATE jobs
            SET status = ?,
                output_path = NULL,
                transcript_path = NULL,
                error_message = NULL,
                current_stage = ?,
                progress = ?,
                started_at = NULL,
                completed_at = NULL,
                updated_at = ?
            WHERE id = ? AND status = ?
            """,
            ("queued", "queued", 0, now, job_id, "failed"),
        )
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:  # pragma: no cover - impossible after successful update
            raise KeyError(job_id)
        return RetryClaim(previous=previous, current=JobRecord(**dict(row)))


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
