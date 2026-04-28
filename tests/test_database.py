from pathlib import Path

from openphonic.core.database import (
    claim_failed_job_for_retry,
    create_job,
    init_db,
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
    assert claimed.status == "queued"
    assert claimed.output_path is None
    assert claimed.transcript_path is None
    assert claimed.error_message is None
    assert claimed.progress == 0
    assert second_claim is None
