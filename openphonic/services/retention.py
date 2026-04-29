from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from openphonic.core.database import (
    JobRecord,
    claim_completed_job_for_retention,
    delete_retention_claim,
    delete_stale_retention_claim,
    list_retention_cleanup_candidates,
    list_stale_retention_claims_to_restore,
    restore_retention_claim,
    restore_stale_retention_claim,
)
from openphonic.core.logging import log_event
from openphonic.core.settings import get_settings
from openphonic.services.storage import delete_job_storage

logger = logging.getLogger(__name__)
RETENTION_CLAIM_STALE_AFTER = timedelta(minutes=15)


@dataclass(frozen=True)
class RetentionCleanupResult:
    deleted_job_ids: list[str] = field(default_factory=list)
    failed_job_ids: dict[str, str] = field(default_factory=dict)


def _storage_root_survived(root: Path) -> bool:
    if not root.exists():
        return False
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Job storage root is not a directory: {root}")
    return True


def _retention_claim_storage_survived(settings, record: JobRecord) -> bool:
    if not _storage_root_survived(settings.uploads_dir / record.id):
        return False
    if not _storage_root_survived(settings.jobs_dir / record.id):
        return False

    required_files = [record.input_path, record.output_path, record.transcript_path]
    return all(Path(path).is_file() for path in required_files if path)


def cleanup_expired_jobs(now: datetime | None = None) -> RetentionCleanupResult:
    settings = get_settings()
    current_time = now or datetime.now(UTC)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=UTC)
    claim_stale_cutoff = (current_time - RETENTION_CLAIM_STALE_AFTER).isoformat(timespec="seconds")

    deleted_job_ids: list[str] = []
    failed_job_ids: dict[str, str] = {}

    if settings.retention_days <= 0:
        cutoff = datetime.min.replace(tzinfo=UTC).isoformat(timespec="seconds")
    else:
        cutoff = (current_time - timedelta(days=settings.retention_days)).isoformat(
            timespec="seconds"
        )

    for candidate in list_stale_retention_claims_to_restore(
        settings.database_path,
        cutoff,
        claim_stale_cutoff,
    ):
        try:
            if not _retention_claim_storage_survived(settings, candidate):
                delete_job_storage(settings, candidate.id)
                if delete_stale_retention_claim(
                    settings.database_path,
                    candidate,
                    cutoff,
                    claim_stale_cutoff,
                ):
                    deleted_job_ids.append(candidate.id)
                    log_event(
                        logger,
                        "job.retention_claim_deleted_missing_storage",
                        job_id=candidate.id,
                        completed_at=candidate.completed_at,
                        retention_days=settings.retention_days,
                    )
                continue

            restored = restore_stale_retention_claim(
                settings.database_path,
                candidate,
                cutoff,
                claim_stale_cutoff,
            )
        except Exception as exc:
            failed_job_ids[candidate.id] = str(exc)
            log_event(
                logger,
                "job.retention_claim_recovery_failed",
                level=logging.WARNING,
                job_id=candidate.id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            continue

        if restored is None:
            continue
        log_event(
            logger,
            "job.retention_claim_restored",
            job_id=restored.id,
            completed_at=restored.completed_at,
            retention_days=settings.retention_days,
        )

    if settings.retention_days <= 0:
        return RetentionCleanupResult(
            deleted_job_ids=deleted_job_ids,
            failed_job_ids=failed_job_ids,
        )

    for candidate in list_retention_cleanup_candidates(
        settings.database_path,
        cutoff,
        claim_stale_cutoff,
    ):
        claim = claim_completed_job_for_retention(
            settings.database_path,
            candidate.id,
            cutoff,
            claim_stale_cutoff,
        )
        if claim is None:
            continue
        try:
            delete_job_storage(settings, claim.current.id)
            if not delete_retention_claim(settings.database_path, claim):
                raise RuntimeError("Expired job changed before retention cleanup finalized.")
        except Exception as exc:
            restore_retention_claim(settings.database_path, claim)
            failed_job_ids[claim.current.id] = str(exc)
            log_event(
                logger,
                "job.retention_cleanup_failed",
                level=logging.WARNING,
                job_id=claim.current.id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            continue

        deleted_job_ids.append(claim.current.id)
        log_event(
            logger,
            "job.retention_deleted",
            job_id=claim.current.id,
            completed_at=claim.previous.completed_at,
            retention_days=settings.retention_days,
        )

    return RetentionCleanupResult(
        deleted_job_ids=deleted_job_ids,
        failed_job_ids=failed_job_ids,
    )
