from __future__ import annotations

from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from openphonic.core.database import create_job, init_db
from openphonic.core.settings import get_settings
from openphonic.services.jobs import fetch_job, recent_jobs, reserve_upload, run_job
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


def _ensure_ready() -> None:
    settings = get_settings()
    init_db(settings.database_path)


def _artifact_url(job_id: str, artifact_name: str) -> str:
    return f"/api/jobs/{job_id}/artifacts/{quote(artifact_name, safe='/')}"


def _artifact_payload(job_id: str, artifact: JobArtifact) -> dict:
    return {
        "name": artifact.name,
        "size_bytes": artifact.size_bytes,
        "url": _artifact_url(job_id, artifact.name),
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
                    "url": f"/api/jobs/{job_id}{suffix}",
                }
                for label, artifact_name, suffix in COMMON_ARTIFACTS
                if artifact_name in artifact_names
            ],
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
