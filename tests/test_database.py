from pathlib import Path

from openphonic.core.database import create_job, init_db, list_jobs, update_job


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
