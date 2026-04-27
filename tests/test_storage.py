import asyncio

import pytest

from openphonic.services.storage import safe_filename, save_upload_file


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
