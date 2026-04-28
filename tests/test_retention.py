from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from openphonic.core.database import create_job, get_job, init_db, update_job
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
