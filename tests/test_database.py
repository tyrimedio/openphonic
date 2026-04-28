from pathlib import Path

from openphonic.core.database import (
    claim_failed_job_for_retry,
    create_job,
    delete_job,
    init_db,
    list_completed_jobs_before,
    list_jobs,
    list_jobs_by_status,
    update_job,
)


def test_job_record_lifecycle_is_durable(tmp_path) -> None:
    db_path = tmp_path / "openphonic.sqlite3"
    init_db(db_path)

    created = create_job(
        db_path,
        job_id="job-1",
        original_filename="input.wav",
        input_path=Path("/tmp/input.wav"),
        config={"preset": "podcast-default"},
    )

    assert created.status == "queued"
    assert created.to_dict()["config"] == {"preset": "podcast-default"}

    updated = update_job(db_path, "job-1", status="running", current_stage="ingest", progress=10)

    assert updated.status == "running"
    assert updated.current_stage == "ingest"
    assert list_jobs(db_path)[0].id == "job-1"
    assert list_jobs_by_status(db_path, ("running",))[0].id == "job-1"
    assert list_jobs_by_status(db_path, ("failed",)) == []


def test_failed_job_retry_claim_is_conditional(tmp_path) -> None:
    db_path = tmp_path / "openphonic.sqlite3"
    init_db(db_path)
    create_job(
        db_path,
        job_id="job-1",
        original_filename="input.wav",
        input_path=Path("/tmp/input.wav"),
    )
    update_job(
        db_path,
        "job-1",
        status="failed",
        output_path="/tmp/output.m4a",
        transcript_path="/tmp/transcript.json",
        error_message="boom",
        current_stage="failed",
        progress=72,
    )

    claimed = claim_failed_job_for_retry(db_path, "job-1")
    second_claim = claim_failed_job_for_retry(db_path, "job-1")

    assert claimed is not None
    assert claimed.previous.status == "failed"
    assert claimed.previous.error_message == "boom"
    assert claimed.current.status == "queued"
    assert claimed.current.output_path is None
    assert claimed.current.transcript_path is None
    assert claimed.current.error_message is None
    assert claimed.current.progress == 0
    assert second_claim is None


def test_completed_jobs_before_returns_only_terminal_expired_jobs(tmp_path) -> None:
    db_path = tmp_path / "openphonic.sqlite3"
    init_db(db_path)
    for job_id in ("old-success", "old-failed", "old-running", "fresh-success"):
        create_job(
            db_path,
            job_id=job_id,
            original_filename=f"{job_id}.wav",
            input_path=Path(f"/tmp/{job_id}.wav"),
        )

    update_job(
        db_path,
        "old-success",
        status="succeeded",
        completed_at="2026-01-01T00:00:00+00:00",
    )
    update_job(
        db_path,
        "old-failed",
        status="failed",
        completed_at="2026-01-02T00:00:00+00:00",
    )
    update_job(
        db_path,
        "old-running",
        status="running",
        completed_at="2026-01-01T00:00:00+00:00",
    )
    update_job(
        db_path,
        "fresh-success",
        status="succeeded",
        completed_at="2026-01-10T00:00:00+00:00",
    )

    expired = list_completed_jobs_before(db_path, "2026-01-05T00:00:00+00:00")

    assert [job.id for job in expired] == ["old-success", "old-failed"]
    assert delete_job(db_path, "old-success") is True
    assert delete_job(db_path, "old-success") is False
