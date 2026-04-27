from __future__ import annotations

import logging
from pathlib import Path

from openphonic.core.database import get_job, list_jobs, update_job, utc_now
from openphonic.core.logging import append_event, log_event
from openphonic.core.settings import get_settings
from openphonic.pipeline.config import PipelineConfig
from openphonic.pipeline.runner import PipelineRunner
from openphonic.services.storage import job_dir, new_job_id, upload_path

logger = logging.getLogger(__name__)


def reserve_upload(original_filename: str) -> tuple[str, Path]:
    settings = get_settings()
    job_id = new_job_id()
    return job_id, upload_path(settings, job_id, original_filename)


def recent_jobs(limit: int = 100):
    return list_jobs(get_settings().database_path, limit=limit)


def fetch_job(job_id: str):
    return get_job(get_settings().database_path, job_id)


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
