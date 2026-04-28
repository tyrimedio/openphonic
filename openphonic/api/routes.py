from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from openphonic.core.database import create_job, init_db
from openphonic.core.settings import get_settings
from openphonic.services.jobs import (
    JobRetryError,
    fetch_job,
    recent_jobs,
    reserve_upload,
    retry_failed_job,
    run_job,
)
from openphonic.services.storage import (
    JobArtifact,
    job_artifact_path,
    list_job_artifacts,
    save_upload_file,
)

router = APIRouter()
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))
UploadFormFile = Annotated[UploadFile, File()]
PresetForm = Annotated[str, Form()]

COMMON_ARTIFACTS = [
    ("Job events", "job-events.jsonl", "/events"),
    ("Command log", "commands.jsonl", "/commands"),
    ("Media metadata", "00_media_metadata.json", "/metadata"),
    ("Pipeline manifest", "pipeline_manifest.json", "/manifest"),
]
MAX_ARTIFACT_PREVIEW_BYTES = 128 * 1024
TEXT_ARTIFACT_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".rttm",
    ".txt",
    ".vtt",
    ".yaml",
    ".yml",
}


def _ensure_ready() -> None:
    settings = get_settings()
    init_db(settings.database_path)


def _artifact_url(job_id: str, artifact_name: str) -> str:
    return f"/api/jobs/{job_id}/artifacts/{quote(artifact_name, safe='/')}"


def _artifact_page_url(job_id: str, artifact_name: str) -> str:
    return f"/jobs/{job_id}/artifacts/{quote(artifact_name, safe='/')}"


def _artifact_payload(job_id: str, artifact: JobArtifact) -> dict:
    return {
        "name": artifact.name,
        "size_bytes": artifact.size_bytes,
        "size": _format_bytes(artifact.size_bytes),
        "url": _artifact_url(job_id, artifact.name),
        "download_url": _artifact_url(job_id, artifact.name),
        "page_url": _artifact_page_url(job_id, artifact.name),
    }


def _require_job(job_id: str):
    record = fetch_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return record


def _artifact_response(job_id: str, artifact_name: str, media_type: str | None = None):
    _ensure_ready()
    _require_job(job_id)
    try:
        path = job_artifact_path(get_settings(), job_id, artifact_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found.") from exc
    return FileResponse(path, filename=path.name, media_type=media_type)


def _format_bytes(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def _format_seconds(value: Any) -> str:
    if value is None:
        return "-"
    try:
        total_ms = int(round(float(value) * 1000))
    except (TypeError, ValueError):
        return "-"
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    seconds, milliseconds = divmod(total_ms, 1000)
    if hours:
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
    return f"{minutes:02}:{seconds:02}.{milliseconds:03}"


def _format_probability(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _job_relative_artifact_name(job_id: str, path: Path) -> str | None:
    root = (get_settings().jobs_dir / job_id).resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return None


def _load_transcript_artifact(
    job_id: str, transcript_path: str | None
) -> tuple[dict[str, Any], str]:
    if not transcript_path:
        raise HTTPException(status_code=404, detail="Transcript not available.")

    artifact_name = _job_relative_artifact_name(job_id, Path(transcript_path))
    if artifact_name is None:
        raise HTTPException(status_code=404, detail="Transcript artifact not found.")

    try:
        path = job_artifact_path(get_settings(), job_id, artifact_name)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Transcript artifact not found.") from exc

    try:
        transcript = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Transcript artifact is invalid JSON.") from exc
    if not isinstance(transcript, dict):
        raise HTTPException(status_code=422, detail="Transcript artifact must be a JSON object.")
    return transcript, artifact_name


def _transcript_segments(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    rendered_segments: list[dict[str, Any]] = []
    for segment in transcript.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        words = []
        for word in segment.get("words") or []:
            if not isinstance(word, dict):
                continue
            words.append(
                {
                    "text": word.get("word") or "",
                    "start": _format_seconds(word.get("start")),
                    "end": _format_seconds(word.get("end")),
                    "probability": _format_probability(word.get("probability")),
                }
            )
        rendered_segments.append(
            {
                "id": segment.get("id"),
                "start": _format_seconds(segment.get("start")),
                "end": _format_seconds(segment.get("end")),
                "text": segment.get("text") or "",
                "words": words,
            }
        )
    return rendered_segments


def _artifact_preview(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in TEXT_ARTIFACT_EXTENSIONS:
        return {"available": False, "reason": "Preview unavailable for binary artifacts."}

    with path.open("rb") as handle:
        data = handle.read(MAX_ARTIFACT_PREVIEW_BYTES + 1)
    truncated = len(data) > MAX_ARTIFACT_PREVIEW_BYTES
    if truncated:
        data = data[:MAX_ARTIFACT_PREVIEW_BYTES]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"available": False, "reason": "Preview unavailable for non-UTF-8 artifacts."}

    if path.suffix.lower() == ".json" and not truncated:
        try:
            text = json.dumps(json.loads(text), indent=2)
        except json.JSONDecodeError:
            pass
    return {"available": True, "content": text, "truncated": truncated}


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    _ensure_ready()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "jobs": recent_jobs(50),
            "settings": get_settings(),
        },
    )


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    artifacts = list_job_artifacts(get_settings(), job_id)
    artifact_names = {artifact.name for artifact in artifacts}
    return templates.TemplateResponse(
        request,
        "job.html",
        {
            "request": request,
            "job": record,
            "artifacts": [_artifact_payload(job_id, artifact) for artifact in artifacts],
            "common_artifacts": [
                {
                    "label": label,
                    "name": artifact_name,
                    "url": _artifact_page_url(job_id, artifact_name),
                    "download_url": f"/api/jobs/{job_id}{suffix}",
                }
                for label, artifact_name, suffix in COMMON_ARTIFACTS
                if artifact_name in artifact_names
            ],
        },
    )


@router.get("/jobs/{job_id}/transcript", response_class=HTMLResponse)
def transcript_page(request: Request, job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    transcript, artifact_name = _load_transcript_artifact(job_id, record.transcript_path)
    segments = _transcript_segments(transcript)
    word_count = sum(len(segment["words"]) for segment in segments)
    vtt_name = "transcript.vtt"
    artifacts = {artifact.name for artifact in list_job_artifacts(get_settings(), job_id)}
    return templates.TemplateResponse(
        request,
        "transcript.html",
        {
            "request": request,
            "job": record,
            "transcript": transcript,
            "segments": segments,
            "word_count": word_count,
            "artifact_name": artifact_name,
            "download_url": _artifact_url(job_id, artifact_name),
            "vtt_url": _artifact_url(job_id, vtt_name) if vtt_name in artifacts else None,
            "language_probability": _format_probability(transcript.get("language_probability")),
            "duration": _format_seconds(transcript.get("duration")),
        },
    )


@router.get("/jobs/{job_id}/artifacts/{artifact_name:path}", response_class=HTMLResponse)
def artifact_page(request: Request, job_id: str, artifact_name: str):
    _ensure_ready()
    record = _require_job(job_id)
    try:
        path = job_artifact_path(get_settings(), job_id, artifact_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found.") from exc

    return templates.TemplateResponse(
        request,
        "artifact.html",
        {
            "request": request,
            "job": record,
            "artifact": {
                "name": artifact_name,
                "size": _format_bytes(path.stat().st_size),
                "download_url": _artifact_url(job_id, artifact_name),
            },
            "preview": _artifact_preview(path),
        },
    )


@router.get("/api/jobs")
def list_job_api() -> list[dict]:
    _ensure_ready()
    return [job.to_dict() for job in recent_jobs(100)]


@router.post("/api/jobs")
async def create_job_api(
    background_tasks: BackgroundTasks,
    file: UploadFormFile,
    preset: PresetForm = "podcast-default",
) -> dict[str, str]:
    return await _create_job(background_tasks, file, preset)


@router.post("/jobs")
async def create_job_form(
    background_tasks: BackgroundTasks,
    file: UploadFormFile,
    preset: PresetForm = "podcast-default",
):
    payload = await _create_job(background_tasks, file, preset)
    return RedirectResponse(f"/jobs/{payload['id']}", status_code=303)


@router.post("/api/jobs/{job_id}/retry")
def retry_job_api(background_tasks: BackgroundTasks, job_id: str) -> dict[str, str]:
    _ensure_ready()
    try:
        retry_failed_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    except JobRetryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(run_job, job_id)
    return {"id": job_id, "status_url": f"/api/jobs/{job_id}"}


@router.post("/jobs/{job_id}/retry")
def retry_job_form(background_tasks: BackgroundTasks, job_id: str):
    payload = retry_job_api(background_tasks, job_id)
    return RedirectResponse(f"/jobs/{payload['id']}", status_code=303)


async def _create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    preset: str,
) -> dict[str, str]:
    _ensure_ready()
    settings = get_settings()
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    job_id, destination = reserve_upload(file.filename)
    try:
        await save_upload_file(file, destination, settings.max_upload_mb * 1024 * 1024)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    create_job(
        settings.database_path,
        job_id=job_id,
        original_filename=file.filename,
        input_path=destination,
        config={"preset": preset},
    )
    background_tasks.add_task(run_job, job_id)
    return {"id": job_id, "status_url": f"/api/jobs/{job_id}"}


@router.get("/api/jobs/{job_id}")
def get_job_api(job_id: str) -> dict:
    _ensure_ready()
    record = _require_job(job_id)
    return record.to_dict()


@router.get("/api/jobs/{job_id}/download")
def download_job(job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    if record.status != "succeeded" or not record.output_path:
        raise HTTPException(status_code=409, detail="Job output is not ready.")
    path = Path(record.output_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file missing.")
    return FileResponse(path, filename=path.name)


@router.get("/api/jobs/{job_id}/transcript")
def download_transcript(job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    if not record.transcript_path:
        raise HTTPException(status_code=404, detail="Transcript not available.")
    path = Path(record.transcript_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript file missing.")
    return FileResponse(path, filename=path.name)


@router.get("/api/jobs/{job_id}/artifacts")
def list_artifacts_api(job_id: str) -> list[dict]:
    _ensure_ready()
    _require_job(job_id)
    try:
        artifacts = list_job_artifacts(get_settings(), job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [_artifact_payload(job_id, artifact) for artifact in artifacts]


@router.get("/api/jobs/{job_id}/artifacts/{artifact_name:path}")
def download_artifact(job_id: str, artifact_name: str):
    return _artifact_response(job_id, artifact_name)


@router.get("/api/jobs/{job_id}/events")
def download_job_events(job_id: str):
    return _artifact_response(job_id, "job-events.jsonl", media_type="application/x-ndjson")


@router.get("/api/jobs/{job_id}/commands")
def download_command_log(job_id: str):
    return _artifact_response(job_id, "commands.jsonl", media_type="application/x-ndjson")


@router.get("/api/jobs/{job_id}/metadata")
def download_media_metadata(job_id: str):
    return _artifact_response(job_id, "00_media_metadata.json", media_type="application/json")


@router.get("/api/jobs/{job_id}/manifest")
def download_pipeline_manifest(job_id: str):
    return _artifact_response(job_id, "pipeline_manifest.json", media_type="application/json")
