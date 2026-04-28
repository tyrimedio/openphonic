from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is an install dependency
    load_dotenv = None


def _env_path(name: str, default: str | Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


DEFAULT_PIPELINE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yml"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_path: Path
    pipeline_config: Path
    preset_dir: Path
    max_upload_mb: int
    public_base_url: str
    hf_token: str | None
    whisper_model: str
    whisper_device: str
    pyannote_model: str
    deepfilternet_bin: str

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


@lru_cache
def get_settings() -> Settings:
    if load_dotenv is not None:
        load_dotenv()

    data_dir = _env_path("OPENPHONIC_DATA_DIR", "./data")
    return Settings(
        data_dir=data_dir,
        database_path=_env_path("OPENPHONIC_DATABASE_PATH", data_dir / "openphonic.sqlite3"),
        pipeline_config=_env_path("OPENPHONIC_PIPELINE_CONFIG", DEFAULT_PIPELINE_CONFIG),
        preset_dir=_env_path("OPENPHONIC_PRESET_DIR", data_dir / "presets"),
        max_upload_mb=_env_int("OPENPHONIC_MAX_UPLOAD_MB", 1024),
        public_base_url=os.getenv("OPENPHONIC_PUBLIC_BASE_URL", "http://127.0.0.1:8000"),
        hf_token=os.getenv("HF_TOKEN") or None,
        whisper_model=os.getenv("OPENPHONIC_WHISPER_MODEL", "small"),
        whisper_device=os.getenv("OPENPHONIC_WHISPER_DEVICE", "auto"),
        pyannote_model=os.getenv(
            "OPENPHONIC_PYANNOTE_MODEL",
            "pyannote/speaker-diarization-3.1",
        ),
        deepfilternet_bin=os.getenv("OPENPHONIC_DEEPFILTERNET_BIN", "deepFilter"),
    )
