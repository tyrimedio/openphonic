import asyncio
import shutil
from pathlib import Path

import pytest

from openphonic.core.settings import Settings
from openphonic.services.storage import (
    archive_job_attempt,
    delete_job_storage,
    job_artifact_path,
    job_dir,
    list_job_artifacts,
    safe_filename,
    save_upload_file,
)


class FakeUpload:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def read(self, size: int) -> bytes:
        _ = size
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def test_safe_filename_removes_path_and_shell_control_characters() -> None:
    assert safe_filename("../../show intro?.wav") == "show_intro_.wav"
    assert safe_filename("...") == "upload"


def test_save_upload_file_enforces_size_limit(tmp_path) -> None:
    destination = tmp_path / "upload.wav"

    with pytest.raises(ValueError, match="Upload exceeds"):
        asyncio.run(save_upload_file(FakeUpload([b"abc", b"def"]), destination, max_bytes=5))

    assert not destination.exists()


def test_save_upload_file_writes_chunks(tmp_path) -> None:
    destination = tmp_path / "upload.wav"

    written = asyncio.run(save_upload_file(FakeUpload([b"abc", b"def"]), destination, max_bytes=6))

    assert written == 6
    assert destination.read_bytes() == b"abcdef"


def test_list_job_artifacts_returns_relative_file_names(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "openphonic.sqlite3",
        pipeline_config=tmp_path / "config.yml",
        preset_dir=tmp_path / "data" / "presets",
        max_upload_mb=10,
        retention_days=0,
        public_base_url="http://127.0.0.1:8000",
        hf_token=None,
        whisper_model="small",
        whisper_device="auto",
        pyannote_model="pyannote/speaker-diarization-3.1",
        deepfilternet_bin="deepFilter",
    )
    root = job_dir(settings, "job-1")
    (root / "commands.jsonl").write_text("{}", encoding="utf-8")
    nested = root / "nested"
    nested.mkdir()
    (nested / "artifact.txt").write_text("artifact", encoding="utf-8")

    artifacts = list_job_artifacts(settings, "job-1")

    assert [artifact.name for artifact in artifacts] == [
        "commands.jsonl",
        "nested/artifact.txt",
    ]
    assert artifacts[0].size_bytes == 2


def test_job_artifact_path_rejects_traversal(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "openphonic.sqlite3",
        pipeline_config=tmp_path / "config.yml",
        preset_dir=tmp_path / "data" / "presets",
        max_upload_mb=10,
        retention_days=0,
        public_base_url="http://127.0.0.1:8000",
        hf_token=None,
        whisper_model="small",
        whisper_device="auto",
        pyannote_model="pyannote/speaker-diarization-3.1",
        deepfilternet_bin="deepFilter",
    )
    root = job_dir(settings, "job-1")
    (root / "commands.jsonl").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid artifact path"):
        job_artifact_path(settings, "job-1", "../openphonic.sqlite3")

    assert job_artifact_path(settings, "job-1", "commands.jsonl") == root / "commands.jsonl"


def test_archive_job_attempt_moves_current_artifacts_but_keeps_previous_attempts(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "openphonic.sqlite3",
        pipeline_config=tmp_path / "config.yml",
        preset_dir=tmp_path / "data" / "presets",
        max_upload_mb=10,
        retention_days=0,
        public_base_url="http://127.0.0.1:8000",
        hf_token=None,
        whisper_model="small",
        whisper_device="auto",
        pyannote_model="pyannote/speaker-diarization-3.1",
        deepfilternet_bin="deepFilter",
    )
    root = job_dir(settings, "job-1")
    (root / "commands.jsonl").write_text("{}", encoding="utf-8")
    previous = root / "attempts" / "attempt-old"
    previous.mkdir(parents=True)
    (previous / "commands.jsonl").write_text("old", encoding="utf-8")

    archive_dir = archive_job_attempt(settings, "job-1", "attempt-new")

    assert archive_dir == root / "attempts" / "attempt-new"
    assert not (root / "commands.jsonl").exists()
    assert (archive_dir / "commands.jsonl").read_text(encoding="utf-8") == "{}"
    assert (previous / "commands.jsonl").read_text(encoding="utf-8") == "old"


def test_archive_job_attempt_rolls_back_moved_artifacts_when_later_move_fails(
    tmp_path,
    monkeypatch,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "openphonic.sqlite3",
        pipeline_config=tmp_path / "config.yml",
        preset_dir=tmp_path / "data" / "presets",
        max_upload_mb=10,
        retention_days=0,
        public_base_url="http://127.0.0.1:8000",
        hf_token=None,
        whisper_model="small",
        whisper_device="auto",
        pyannote_model="pyannote/speaker-diarization-3.1",
        deepfilternet_bin="deepFilter",
    )
    root = job_dir(settings, "job-1")
    (root / "commands.jsonl").write_text("commands", encoding="utf-8")
    (root / "pipeline_manifest.json").write_text("manifest", encoding="utf-8")
    real_move = shutil.move
    forward_moves = 0

    def flaky_move(src: str, dst: str):
        nonlocal forward_moves
        source = Path(src)
        if source.parent == root:
            forward_moves += 1
            if forward_moves == 2:
                raise OSError("move failed")
        return real_move(src, dst)

    monkeypatch.setattr("openphonic.services.storage.shutil.move", flaky_move)

    with pytest.raises(OSError, match="move failed"):
        archive_job_attempt(settings, "job-1", "attempt-new")

    assert (root / "commands.jsonl").read_text(encoding="utf-8") == "commands"
    assert (root / "pipeline_manifest.json").read_text(encoding="utf-8") == "manifest"
    assert not (root / "attempts" / "attempt-new" / "commands.jsonl").exists()


def test_delete_job_storage_removes_upload_and_artifact_directories(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "openphonic.sqlite3",
        pipeline_config=tmp_path / "config.yml",
        preset_dir=tmp_path / "data" / "presets",
        max_upload_mb=10,
        retention_days=0,
        public_base_url="http://127.0.0.1:8000",
        hf_token=None,
        whisper_model="small",
        whisper_device="auto",
        pyannote_model="pyannote/speaker-diarization-3.1",
        deepfilternet_bin="deepFilter",
    )
    upload_root = settings.uploads_dir / "job-1"
    work_root = settings.jobs_dir / "job-1"
    upload_root.mkdir(parents=True)
    work_root.mkdir(parents=True)
    (upload_root / "input.wav").write_bytes(b"input")
    (work_root / "pipeline_manifest.json").write_text("{}", encoding="utf-8")

    delete_job_storage(settings, "job-1")

    assert not upload_root.exists()
    assert not work_root.exists()


def test_delete_job_storage_rejects_invalid_job_ids(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "openphonic.sqlite3",
        pipeline_config=tmp_path / "config.yml",
        preset_dir=tmp_path / "data" / "presets",
        max_upload_mb=10,
        retention_days=0,
        public_base_url="http://127.0.0.1:8000",
        hf_token=None,
        whisper_model="small",
        whisper_device="auto",
        pyannote_model="pyannote/speaker-diarization-3.1",
        deepfilternet_bin="deepFilter",
    )

    with pytest.raises(ValueError, match="Invalid job id"):
        delete_job_storage(settings, "../job-1")


def test_delete_job_storage_validates_roots_before_removing_anything(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "openphonic.sqlite3",
        pipeline_config=tmp_path / "config.yml",
        preset_dir=tmp_path / "data" / "presets",
        max_upload_mb=10,
        retention_days=0,
        public_base_url="http://127.0.0.1:8000",
        hf_token=None,
        whisper_model="small",
        whisper_device="auto",
        pyannote_model="pyannote/speaker-diarization-3.1",
        deepfilternet_bin="deepFilter",
    )
    upload_root = settings.uploads_dir / "job-1"
    work_root = settings.jobs_dir / "job-1"
    upload_root.mkdir(parents=True)
    settings.jobs_dir.mkdir(parents=True)
    work_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        delete_job_storage(settings, "job-1")

    assert upload_root.exists()
    assert work_root.exists()
