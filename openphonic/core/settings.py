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


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    value = (os.getenv(name) or default).strip().lower()
    if not value:
        value = default
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {allowed}.")
    return value


DEFAULT_PIPELINE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yml"
TRANSCRIPTION_PROVIDERS = {"deepgram", "local"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_path: Path
    pipeline_config: Path
    preset_dir: Path
    max_upload_mb: int
    retention_days: int
    public_base_url: str
    hf_token: str | None
    whisper_model: str
    whisper_device: str
    pyannote_model: str
    deepfilternet_bin: str
    transcription_provider: str = "local"
    deepgram_api_key: str | None = None
    deepgram_model: str = "nova-3"

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
        retention_days=_env_int("OPENPHONIC_RETENTION_DAYS", 0),
        public_base_url=os.getenv("OPENPHONIC_PUBLIC_BASE_URL", "http://127.0.0.1:8000"),
        hf_token=os.getenv("HF_TOKEN") or None,
        whisper_model=os.getenv("OPENPHONIC_WHISPER_MODEL", "small"),
        whisper_device=os.getenv("OPENPHONIC_WHISPER_DEVICE", "auto"),
        pyannote_model=os.getenv(
            "OPENPHONIC_PYANNOTE_MODEL",
            "pyannote/speaker-diarization-3.1",
        ),
        deepfilternet_bin=os.getenv("OPENPHONIC_DEEPFILTERNET_BIN", "deepFilter"),
        transcription_provider=_env_choice(
            "TRANSCRIPTION_PROVIDER",
            "local",
            TRANSCRIPTION_PROVIDERS,
        ),
        deepgram_api_key=os.getenv("DEEPGRAM_API_KEY") or None,
        deepgram_model=os.getenv("OPENPHONIC_DEEPGRAM_MODEL") or "nova-3",
    )
