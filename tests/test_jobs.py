from pathlib import Path

from openphonic.core.database import create_job, get_job, init_db
from openphonic.core.settings import get_settings
from openphonic.pipeline.runner import PipelineResult
from openphonic.services.jobs import run_job


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
