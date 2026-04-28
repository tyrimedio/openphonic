from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from openphonic.core.database import get_job, list_jobs, list_jobs_by_status, update_job, utc_now
from openphonic.core.logging import append_event, log_event
from openphonic.core.settings import get_settings
from openphonic.pipeline.config import PipelineConfig
from openphonic.pipeline.runner import PipelineRunner
from openphonic.services.storage import archive_job_attempt, job_dir, new_job_id, upload_path

logger = logging.getLogger(__name__)

INTERRUPTED_JOB_MESSAGE = (
    "Job was interrupted while Openphonic was not running. Retry the job to process it again."
)


class JobRetryError(RuntimeError):
    """Raised when a job cannot be retried."""


def reserve_upload(original_filename: str) -> tuple[str, Path]:
    settings = get_settings()
    job_id = new_job_id()
    return job_id, upload_path(settings, job_id, original_filename)


def recent_jobs(limit: int = 100):
    return list_jobs(get_settings().database_path, limit=limit)


def fetch_job(job_id: str):
    return get_job(get_settings().database_path, job_id)


def _attempt_archive_name() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"attempt-{timestamp}-{new_job_id()[:8]}"


def recover_interrupted_jobs() -> int:
    settings = get_settings()
    recovered = 0
    for record in list_jobs_by_status(settings.database_path, ("running", "queued")):
        work_dir = job_dir(settings, record.id)
        append_event(
            work_dir / "job-events.jsonl",
            "job.interrupted",
            job_id=record.id,
            previous_status=record.status,
            previous_stage=record.current_stage,
            previous_progress=record.progress,
        )
        update_job(
            settings.database_path,
            record.id,
            status="failed",
            error_message=INTERRUPTED_JOB_MESSAGE,
            current_stage="interrupted",
            completed_at=utc_now(),
        )
        log_event(
            logger,
            "job.interrupted",
            level=logging.WARNING,
            job_id=record.id,
            previous_status=record.status,
            previous_stage=record.current_stage,
        )
        recovered += 1
    return recovered


def retry_failed_job(job_id: str):
    settings = get_settings()
    record = get_job(settings.database_path, job_id)
    if record is None:
        raise KeyError(job_id)
    if record.status != "failed":
        raise JobRetryError(f"Only failed jobs can be retried. Current status: {record.status}.")

    archive_dir = archive_job_attempt(settings, job_id, _attempt_archive_name())
    work_dir = job_dir(settings, job_id)
    retried = update_job(
        settings.database_path,
        job_id,
        status="queued",
        output_path=None,
        transcript_path=None,
        error_message=None,
        current_stage="queued",
        progress=0,
        started_at=None,
        completed_at=None,
    )
    append_event(
        work_dir / "job-events.jsonl",
        "job.retry_queued",
        job_id=job_id,
        archived_to=archive_dir,
    )
    log_event(logger, "job.retry_queued", job_id=job_id, archived_to=archive_dir)
    return retried


def run_job(job_id: str) -> None:
    settings = get_settings()
    record = get_job(settings.database_path, job_id)
    if record is None:
        log_event(logger, "job.missing", level=logging.WARNING, job_id=job_id)
        return
    work_dir = job_dir(settings, job_id)
    job_events_path = work_dir / "job-events.jsonl"
    command_log_path = work_dir / "commands.jsonl"

    update_job(
        settings.database_path,
        job_id,
        status="running",
        current_stage="starting",
        progress=5,
        started_at=utc_now(),
        error_message=None,
    )
    append_event(job_events_path, "job.started", job_id=job_id, input_path=record.input_path)
    log_event(logger, "job.started", job_id=job_id, input_path=record.input_path)

    def on_stage(stage_name: str, progress: int) -> None:
        bounded_progress = max(0, min(progress, 99))
        update_job(
            settings.database_path,
            job_id,
            current_stage=stage_name,
            progress=bounded_progress,
        )
        append_event(
            job_events_path,
            "job.progress",
            job_id=job_id,
            current_stage=stage_name,
            progress=bounded_progress,
        )

    try:
        config = PipelineConfig.from_path(settings.pipeline_config)
        result = PipelineRunner(
            config,
            progress_callback=on_stage,
            command_log_path=command_log_path,
        ).run(
            Path(record.input_path),
            work_dir,
        )
        update_job(
            settings.database_path,
            job_id,
            status="succeeded",
            output_path=str(result.output_path),
            transcript_path=str(result.artifacts.get("transcript_json"))
            if "transcript_json" in result.artifacts
            else None,
            current_stage="complete",
            progress=100,
            completed_at=utc_now(),
        )
        append_event(
            job_events_path,
            "job.succeeded",
            job_id=job_id,
            output_path=result.output_path,
            artifacts=result.artifacts,
        )
        log_event(logger, "job.succeeded", job_id=job_id, output_path=result.output_path)
    except Exception as exc:  # pragma: no cover - exercised in integration tests
        log_event(
            logger,
            "job.failed",
            level=logging.ERROR,
            job_id=job_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        append_event(
            job_events_path,
            "job.failed",
            job_id=job_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        update_job(
            settings.database_path,
            job_id,
            status="failed",
            error_message=str(exc),
            current_stage="failed",
            completed_at=utc_now(),
        )
