from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from openphonic.core.database import create_job, init_db
from openphonic.core.settings import get_settings
from openphonic.services.jobs import fetch_job, recent_jobs, reserve_upload, run_job
from openphonic.services.storage import save_upload_file

router = APIRouter()
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))
UploadFormFile = Annotated[UploadFile, File()]
PresetForm = Annotated[str, Form()]


def _ensure_ready() -> None:
    settings = get_settings()
    init_db(settings.database_path)


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
    record = fetch_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return templates.TemplateResponse(
        request,
        "job.html",
        {
            "request": request,
            "job": record,
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
    record = fetch_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return record.to_dict()


@router.get("/api/jobs/{job_id}/download")
def download_job(job_id: str):
    record = fetch_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if record.status != "succeeded" or not record.output_path:
        raise HTTPException(status_code=409, detail="Job output is not ready.")
    path = Path(record.output_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file missing.")
    return FileResponse(path, filename=path.name)


@router.get("/api/jobs/{job_id}/transcript")
def download_transcript(job_id: str):
    record = fetch_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not record.transcript_path:
        raise HTTPException(status_code=404, detail="Transcript not available.")
    path = Path(record.transcript_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript file missing.")
    return FileResponse(path, filename=path.name)
