import asyncio

import pytest

from openphonic.core.settings import Settings
from openphonic.services.storage import (
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
        max_upload_mb=10,
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
        max_upload_mb=10,
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
