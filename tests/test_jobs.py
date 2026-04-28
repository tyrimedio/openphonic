from pathlib import Path

from openphonic.core.database import create_job, get_job, init_db, update_job
from openphonic.core.settings import get_settings
from openphonic.pipeline.runner import PipelineResult
from openphonic.services.jobs import (
    INTERRUPTED_JOB_MESSAGE,
    recover_interrupted_jobs,
    retry_failed_job,
    run_job,
)
from openphonic.services.storage import job_dir


class SuccessfulRunner:
    seen_command_log_path: Path | None = None

    def __init__(self, config, progress_callback=None, command_log_path=None) -> None:
        _ = config
        self.progress_callback = progress_callback
        SuccessfulRunner.seen_command_log_path = command_log_path

    def run(self, input_path: Path, work_dir: Path) -> PipelineResult:
        _ = input_path
        if self.progress_callback:
            self.progress_callback("ingest", 33)
        output_path = work_dir / "output.m4a"
        transcript_path = work_dir / "transcript.json"
        output_path.write_bytes(b"audio")
        transcript_path.write_text("{}", encoding="utf-8")
        return PipelineResult(
            output_path=output_path,
            artifacts={"transcript_json": transcript_path},
        )


class FailingRunner:
    def __init__(self, config, progress_callback=None, command_log_path=None) -> None:
        _ = config, progress_callback, command_log_path

    def run(self, input_path: Path, work_dir: Path) -> PipelineResult:
        _ = input_path, work_dir
        raise RuntimeError("boom")


def configure_tmp_settings(tmp_path, monkeypatch) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "openphonic.sqlite3"
    monkeypatch.setenv("OPENPHONIC_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENPHONIC_DATABASE_PATH", str(db_path))
    monkeypatch.setenv("OPENPHONIC_PIPELINE_CONFIG", "config/default.yml")
    get_settings.cache_clear()
    init_db(db_path)
    return db_path


def test_run_job_records_success_and_durable_events(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    monkeypatch.setattr("openphonic.services.jobs.PipelineRunner", SuccessfulRunner)

    run_job("job-1")

    record = get_job(db_path, "job-1")
    assert record is not None
    assert record.status == "succeeded"
    assert record.progress == 100
    assert record.output_path is not None
    assert record.transcript_path is not None
    assert SuccessfulRunner.seen_command_log_path == tmp_path / "data/jobs/job-1/commands.jsonl"
    assert (tmp_path / "data/jobs/job-1/job-events.jsonl").exists()


def test_run_job_records_failure_without_traceback_printing(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-2", original_filename="input.wav", input_path=input_path)
    monkeypatch.setattr("openphonic.services.jobs.PipelineRunner", FailingRunner)

    run_job("job-2")

    record = get_job(db_path, "job-2")
    assert record is not None
    assert record.status == "failed"
    assert record.current_stage == "failed"
    assert record.error_message == "boom"
    events = (tmp_path / "data/jobs/job-2/job-events.jsonl").read_text(encoding="utf-8")
    assert "job.failed" in events


def test_recover_interrupted_jobs_marks_abandoned_jobs_failed(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    running_input = tmp_path / "running.wav"
    queued_input = tmp_path / "queued.wav"
    running_input.write_bytes(b"audio")
    queued_input.write_bytes(b"audio")
    create_job(
        db_path,
        job_id="running-job",
        original_filename="running.wav",
        input_path=running_input,
    )
    create_job(
        db_path,
        job_id="queued-job",
        original_filename="queued.wav",
        input_path=queued_input,
    )
    update_job(db_path, "running-job", status="running", current_stage="loudness", progress=75)

    recovered = recover_interrupted_jobs()

    running = get_job(db_path, "running-job")
    queued = get_job(db_path, "queued-job")
    assert recovered == 2
    assert running is not None
    assert running.status == "failed"
    assert running.current_stage == "interrupted"
    assert running.error_message == INTERRUPTED_JOB_MESSAGE
    assert queued is not None
    assert queued.status == "failed"
    events = (tmp_path / "data/jobs/running-job/job-events.jsonl").read_text(encoding="utf-8")
    assert "job.interrupted" in events
    assert "loudness" in events


def test_retry_failed_job_archives_artifacts_and_resets_job(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-3", original_filename="input.wav", input_path=input_path)
    update_job(
        db_path,
        "job-3",
        status="failed",
        output_path="/old/output.m4a",
        transcript_path="/old/transcript.json",
        error_message="boom",
        current_stage="failed",
        progress=72,
    )
    work_dir = job_dir(get_settings(), "job-3")
    (work_dir / "commands.jsonl").write_text("old commands", encoding="utf-8")
    (work_dir / "pipeline_manifest.json").write_text("old manifest", encoding="utf-8")

    retried = retry_failed_job("job-3")

    archived_commands = list((work_dir / "attempts").glob("*/commands.jsonl"))
    record = get_job(db_path, "job-3")
    assert retried.status == "queued"
    assert record is not None
    assert record.status == "queued"
    assert record.output_path is None
    assert record.transcript_path is None
    assert record.error_message is None
    assert record.current_stage == "queued"
    assert record.progress == 0
    assert archived_commands
    assert archived_commands[0].read_text(encoding="utf-8") == "old commands"
    assert (work_dir / "job-events.jsonl").exists()
    retry_events = (work_dir / "job-events.jsonl").read_text(encoding="utf-8")
    assert "job.retry_queued" in retry_events


def test_retry_failed_job_claims_before_archiving(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-4", original_filename="input.wav", input_path=input_path)
    update_job(db_path, "job-4", status="failed", error_message="boom", current_stage="failed")
    work_dir = job_dir(get_settings(), "job-4")
    (work_dir / "commands.jsonl").write_text("old commands", encoding="utf-8")
    archived: list[str] = []

    def fake_archive_job_attempt(settings, job_id: str, archive_name: str):
        _ = settings
        record = get_job(db_path, job_id)
        assert record is not None
        assert record.status == "queued"
        archived.append(archive_name)
        return work_dir / "attempts" / archive_name

    monkeypatch.setattr("openphonic.services.jobs.archive_job_attempt", fake_archive_job_attempt)

    retry_failed_job("job-4")

    assert archived
