from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from openphonic.api.routes import router
from openphonic.core.database import init_db
from openphonic.core.logging import configure_logging
from openphonic.core.settings import Settings, get_settings
from openphonic.pipeline.deepgram import DeepgramError, validate_deepgram_api_key
from openphonic.services.jobs import recover_interrupted_jobs
from openphonic.services.retention import cleanup_expired_jobs
from openphonic.services.storage import cleanup_artifact_bundle_snapshots, ensure_storage

PACKAGE_ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    validate_provider_setup(settings)
    ensure_storage(settings)
    cleanup_artifact_bundle_snapshots(settings)
    init_db(settings.database_path)
    recover_interrupted_jobs()
    cleanup_expired_jobs()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Openphonic", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(PACKAGE_ROOT / "static")), name="static")
    app.include_router(router)

    return app


def validate_provider_setup(settings: Settings) -> None:
    if settings.transcription_provider != "deepgram":
        return
    if not settings.deepgram_api_key:
        raise RuntimeError("Deepgram provider setup failed: DEEPGRAM_API_KEY is required.")
    try:
        validate_deepgram_api_key(settings.deepgram_api_key)
    except DeepgramError as exc:
        raise RuntimeError(f"Deepgram provider setup failed: {exc}") from exc


app = create_app()
