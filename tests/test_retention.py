from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from openphonic.core.database import JobRecord, create_job, get_job, init_db, update_job
from openphonic.core.settings import get_settings
from openphonic.services.retention import cleanup_expired_jobs


def configure_tmp_settings(tmp_path, monkeypatch, *, retention_days: int) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "openphonic.sqlite3"
    monkeypatch.setenv("OPENPHONIC_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENPHONIC_DATABASE_PATH", str(db_path))
    monkeypatch.setenv("OPENPHONIC_PIPELINE_CONFIG", "config/default.yml")
    monkeypatch.setenv("OPENPHONIC_RETENTION_DAYS", str(retention_days))
    get_settings.cache_clear()
    init_db(db_path)
    return db_path


def create_completed_job(db_path: Path, job_id: str, *, completed_at: str) -> None:
    settings = get_settings()
    upload_root = settings.uploads_dir / job_id
    work_root = settings.jobs_dir / job_id
    upload_root.mkdir(parents=True)
    work_root.mkdir(parents=True)
    input_path = upload_root / "input.wav"
    input_path.write_bytes(b"input")
    (work_root / "pipeline_manifest.json").write_text("{}", encoding="utf-8")
    create_job(
        db_path,
        job_id=job_id,
        original_filename="input.wav",
        input_path=input_path,
    )
    update_job(
        db_path,
        job_id,
        status="succeeded",
        output_path=str(work_root / "output.m4a"),
        completed_at=completed_at,
    )


def test_cleanup_expired_jobs_removes_old_terminal_records_and_storage(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch, retention_days=7)
    create_completed_job(db_path, "old-job", completed_at="2026-04-01T00:00:00+00:00")
    create_completed_job(db_path, "fresh-job", completed_at="2026-04-25T00:00:00+00:00")

    result = cleanup_expired_jobs(now=datetime(2026, 4, 28, tzinfo=UTC))

    settings = get_settings()
    assert result.deleted_job_ids == ["old-job"]
    assert result.failed_job_ids == {}
    assert get_job(db_path, "old-job") is None
    assert get_job(db_path, "fresh-job") is not None
    assert not (settings.uploads_dir / "old-job").exists()
    assert not (settings.jobs_dir / "old-job").exists()
    assert (settings.uploads_dir / "fresh-job").exists()
    assert (settings.jobs_dir / "fresh-job").exists()


def test_cleanup_expired_jobs_is_disabled_when_retention_is_zero(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch, retention_days=0)
    create_completed_job(db_path, "old-job", completed_at="2026-04-01T00:00:00+00:00")

    result = cleanup_expired_jobs(now=datetime(2026, 4, 28, tzinfo=UTC))

    assert result.deleted_job_ids == []
    assert get_job(db_path, "old-job") is not None
    assert (get_settings().jobs_dir / "old-job").exists()


def test_cleanup_expired_jobs_preserves_row_when_storage_cleanup_fails(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch, retention_days=7)
    create_completed_job(db_path, "old-job", completed_at="2026-04-01T00:00:00+00:00")

    def fail_delete_storage(settings, job_id: str) -> None:
        _ = settings, job_id
        raise OSError("storage is busy")

    monkeypatch.setattr(
        "openphonic.services.retention.delete_job_storage",
        fail_delete_storage,
    )

    result = cleanup_expired_jobs(now=datetime(2026, 4, 28, tzinfo=UTC))

    settings = get_settings()
    assert result.deleted_job_ids == []
    assert result.failed_job_ids == {"old-job": "storage is busy"}
    assert get_job(db_path, "old-job") is not None
    assert (settings.uploads_dir / "old-job").exists()
    assert (settings.jobs_dir / "old-job").exists()


def test_cleanup_expired_jobs_does_not_hold_write_lock_during_storage_cleanup(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch, retention_days=7)
    create_completed_job(db_path, "old-job", completed_at="2026-04-01T00:00:00+00:00")
    create_job(
        db_path,
        job_id="active-job",
        original_filename="active.wav",
        input_path=Path("/tmp/active.wav"),
    )

    def delete_storage_while_writing(settings, job_id: str) -> None:
        assert job_id == "old-job"
        claimed = get_job(db_path, "old-job")
        assert claimed is not None
        assert claimed.status == "retention_cleanup_succeeded"
        update_job(
            db_path,
            "active-job",
            current_stage="storage_cleanup_overlap",
            progress=7,
        )
        from openphonic.services.storage import delete_job_storage

        delete_job_storage(settings, job_id)

    monkeypatch.setattr(
        "openphonic.services.retention.delete_job_storage",
        delete_storage_while_writing,
    )

    result = cleanup_expired_jobs(now=datetime(2026, 4, 28, tzinfo=UTC))

    active = get_job(db_path, "active-job")
    assert active is not None
    assert result.deleted_job_ids == ["old-job"]
    assert result.failed_job_ids == {}
    assert active.current_stage == "storage_cleanup_overlap"
    assert active.progress == 7


def test_cleanup_expired_jobs_skips_rows_that_changed_after_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch, retention_days=7)
    create_completed_job(db_path, "old-job", completed_at="2026-04-01T00:00:00+00:00")
    stale_record = get_job(db_path, "old-job")
    assert stale_record is not None
    update_job(
        db_path,
        "old-job",
        status="queued",
        output_path=None,
        completed_at=None,
        current_stage="queued",
    )

    def stale_expired_jobs(db_path_arg: Path, cutoff: str) -> list[JobRecord]:
        _ = db_path_arg, cutoff
        return [stale_record]

    monkeypatch.setattr(
        "openphonic.services.retention.list_completed_jobs_before",
        stale_expired_jobs,
    )

    result = cleanup_expired_jobs(now=datetime(2026, 4, 28, tzinfo=UTC))

    settings = get_settings()
    current = get_job(db_path, "old-job")
    assert current is not None
    assert current.status == "queued"
    assert result.deleted_job_ids == []
    assert result.failed_job_ids == {}
    assert (settings.uploads_dir / "old-job").exists()
    assert (settings.jobs_dir / "old-job").exists()
