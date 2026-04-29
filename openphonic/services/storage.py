from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from openphonic.core.settings import Settings

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class JobArtifact:
    name: str
    path: Path
    size_bytes: int


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


def archive_job_attempt(settings: Settings, job_id: str, archive_name: str) -> Path | None:
    root = job_dir(settings, job_id)
    entries = sorted(path for path in root.iterdir() if path.name != "attempts")
    if not entries:
        return None

    archive_dir = root / "attempts" / safe_filename(archive_name)
    archive_dir.mkdir(parents=True, exist_ok=False)
    moved: list[tuple[Path, Path]] = []
    try:
        for path in entries:
            destination = archive_dir / path.name
            shutil.move(str(path), destination)
            moved.append((destination, path))
    except Exception:
        for destination, original in reversed(moved):
            if destination.exists() and not original.exists():
                shutil.move(str(destination), original)
        try:
            archive_dir.rmdir()
        except OSError:
            pass
        raise
    return archive_dir


def _job_root(settings: Settings, job_id: str) -> Path:
    if not job_id or Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError("Invalid job id.")
    return settings.jobs_dir / job_id


def _upload_root(settings: Settings, job_id: str) -> Path:
    if not job_id or Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError("Invalid job id.")
    return settings.uploads_dir / job_id


def delete_job_storage(settings: Settings, job_id: str) -> None:
    roots = [_upload_root(settings, job_id), _job_root(settings, job_id)]
    existing_roots: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"Job storage root is not a directory: {root}")
        existing_roots.append(root)
    for root in existing_roots:
        shutil.rmtree(root)


def list_job_artifacts(settings: Settings, job_id: str) -> list[JobArtifact]:
    root = _job_root(settings, job_id)
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError("Job artifact root is not a directory.")

    artifacts: list[JobArtifact] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        artifacts.append(
            JobArtifact(
                name=path.relative_to(root).as_posix(),
                path=path,
                size_bytes=path.stat().st_size,
            )
        )
    return artifacts


def job_artifact_path(settings: Settings, job_id: str, artifact_name: str) -> Path:
    root = _job_root(settings, job_id).resolve()
    requested = Path(artifact_name)
    if not artifact_name or requested.is_absolute() or ".." in requested.parts:
        raise ValueError("Invalid artifact path.")

    path = (root / requested).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Invalid artifact path.") from exc
    if not path.is_file():
        raise FileNotFoundError(artifact_name)
    return path


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
