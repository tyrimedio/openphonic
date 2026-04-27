from __future__ import annotations

import re
import uuid
from pathlib import Path

from openphonic.core.settings import Settings

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def ensure_storage(settings: Settings) -> None:
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)


def new_job_id() -> str:
    return uuid.uuid4().hex


def safe_filename(filename: str | None) -> str:
    clean = SAFE_NAME.sub("_", filename or "upload")
    clean = clean.strip("._")
    return clean or "upload"


def upload_path(settings: Settings, job_id: str, filename: str) -> Path:
    directory = settings.uploads_dir / job_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory / safe_filename(filename)


def job_dir(settings: Settings, job_id: str) -> Path:
    directory = settings.jobs_dir / job_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


async def save_upload_file(upload_file, destination: Path, max_bytes: int) -> int:
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        while chunk := await upload_file.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                destination.unlink(missing_ok=True)
                raise ValueError("Upload exceeds configured maximum size.")
            handle.write(chunk)
    return total
