from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from openphonic.core.database import delete_completed_job_before, list_completed_jobs_before
from openphonic.core.logging import log_event
from openphonic.core.settings import get_settings
from openphonic.services.storage import delete_job_storage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionCleanupResult:
    deleted_job_ids: list[str] = field(default_factory=list)
    failed_job_ids: dict[str, str] = field(default_factory=dict)


def cleanup_expired_jobs(now: datetime | None = None) -> RetentionCleanupResult:
    settings = get_settings()
    if settings.retention_days <= 0:
        return RetentionCleanupResult()

    current_time = now or datetime.now(UTC)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=UTC)
    cutoff = (current_time - timedelta(days=settings.retention_days)).isoformat(timespec="seconds")

    deleted_job_ids: list[str] = []
    failed_job_ids: dict[str, str] = {}
    for candidate in list_completed_jobs_before(settings.database_path, cutoff):
        record = delete_completed_job_before(settings.database_path, candidate.id, cutoff)
        if record is None:
            continue
        try:
            delete_job_storage(settings, record.id)
        except Exception as exc:
            failed_job_ids[record.id] = str(exc)
            log_event(
                logger,
                "job.retention_cleanup_failed",
                level=logging.WARNING,
                job_id=record.id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            continue

        deleted_job_ids.append(record.id)
        log_event(
            logger,
            "job.retention_deleted",
            job_id=record.id,
            completed_at=record.completed_at,
            retention_days=settings.retention_days,
        )

    return RetentionCleanupResult(
        deleted_job_ids=deleted_job_ids,
        failed_job_ids=failed_job_ids,
    )
