# Openphonic

Openphonic is a small-scale, self-hosted audio post-production platform inspired by Auphonic. It is designed for a single team or a handful of users first, with a clear path to heavier model-based processing when you want it.

The default pipeline is intentionally FFmpeg-first so the app is useful before you install GPU-sized dependencies. Optional stages are provided for Whisper/faster-whisper transcription, pyannote diarization, DeepFilterNet noise reduction, and Demucs source separation.

## What This Initial Version Includes

- FastAPI app with upload, job status, transcript, and download endpoints
- SQLite job database for small deployments
- Local filesystem storage under `data/`
- Config-driven audio pipeline
- FFprobe media validation plus FFmpeg ingest, silence trimming, and two-pass loudness normalization
- Optional adapters for faster-whisper, pyannote, DeepFilterNet CLI, and Demucs
- Docker and Docker Compose
- CLI entry point for processing one file without the web app
- Tests for media command builders, pipeline config, job transitions, storage behavior, and API routes

## Requirements

- Python 3.11+
- FFmpeg available on `PATH`
- Optional: GPU/CUDA if you enable heavier ML stages

On macOS:

```bash
brew install ffmpeg
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
openphonic process ./sample.wav --output ./processed.m4a
```

Run the web app:

```bash
uvicorn openphonic.main:app --reload
```

Then open `http://127.0.0.1:8000`.

## Docker

```bash
docker compose up --build
```

The app stores uploads, processed files, and SQLite data in the `openphonic-data` volume.

## Pipeline

The default preset lives in [config/default.yml](config/default.yml). It runs:

1. Validate media metadata and write a metadata artifact
2. Ingest to a normalized working WAV
3. Optional DeepFilterNet noise reduction
4. Optional Demucs separation
5. Optional silence trimming
6. FFmpeg two-pass loudness normalization
7. Optional faster-whisper transcription
8. Optional pyannote diarization

The optional ML stages are disabled by default because their install/runtime requirements differ by machine.

## Scaling Notes

For 10 people or fewer, the included SQLite + local worker model is enough to start. If you later need public multi-user hosting, the main changes are:

- Move job execution to a real queue such as Redis Queue, Dramatiq, or Celery
- Store media in S3-compatible object storage
- Put auth and per-user quotas in front of uploads
- Run ML workers separately from the API process
- Add GPU worker scheduling if you enable Demucs/WhisperX/pyannote at scale

## License Choice

This repository uses AGPL-3.0-or-later by default because hosted audio processing is naturally a service. If you prefer maximum permissiveness, switch to Apache-2.0 or MIT before accepting outside contributions.
