from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from openphonic.api.routes import router
from openphonic.core.database import init_db
from openphonic.core.logging import configure_logging
from openphonic.core.settings import get_settings
from openphonic.services.jobs import recover_interrupted_jobs
from openphonic.services.storage import ensure_storage

PACKAGE_ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    ensure_storage(settings)
    init_db(settings.database_path)
    recover_interrupted_jobs()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Openphonic", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(PACKAGE_ROOT / "static")), name="static")
    app.include_router(router)

    return app


app = create_app()
