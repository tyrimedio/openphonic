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

## Local Smoke Test

Before enabling optional ML stages, run the local FFmpeg-first pipeline against a
generated test tone:

```bash
openphonic smoke-test
```

By default this writes the processed audio and trace artifacts under
`data/smoke-test/`. The command generates a tiny WAV with FFmpeg, runs the
configured pipeline, writes command logs, and prints the output and artifact
paths. You can override the output or work directory when you want to keep a
specific run:

```bash
openphonic smoke-test --output ./processed-smoke.m4a --work-dir ./data/smoke-test/work
```

Use this as the first local verification pass after setup. If it fails, fix the
FFmpeg/core pipeline path before testing transcription, diarization, denoise, or
source separation.

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
6. Optional intro/outro insertion from local media
7. FFmpeg two-pass loudness normalization
8. Optional faster-whisper transcription
9. Optional non-destructive cut suggestions from transcript timestamps
10. Optional pyannote diarization

The optional ML stages are disabled by default because their install/runtime requirements differ by machine.

The web uploader currently exposes these built-in presets:

- `podcast-default`: FFmpeg ingest, conservative silence trim, and two-pass loudness normalization.
- `speech-cleanup`: `podcast-default` plus DeepFilterNet speech enhancement.
- `vocal-isolation`: `podcast-default` plus Demucs vocal isolation with the `htdemucs` model and `vocals` stem.
- `transcript-review`: `podcast-default` plus faster-whisper transcription and non-destructive cut suggestions.
- `speaker-diarization`: `podcast-default` plus faster-whisper transcription and pyannote speaker labels.

The ML presets are explicit opt-ins. The uploader preflights enabled optional
stages before queueing a job and rejects presets whose local tools, Python
packages, model tokens, or configured assets are missing.

You can add per-show presets without changing application code by placing
`*.yml` or `*.yaml` files in `OPENPHONIC_PRESET_DIR`, which defaults to
`./data/presets`. The uploader lists these as `custom:<filename>` presets. A
custom preset can include optional display metadata:

```yaml
preset:
  label: Daily show
  description: Daily show production preset.
name: daily-show
target:
  sample_rate: 48000
  channels: 2
stages:
  intro_outro:
    enabled: true
    intro_path: ./assets/intro.wav
    outro_path: ./assets/outro.wav
  loudness:
    enabled: true
```

Relative intro/outro paths are resolved from the custom preset file location, so
a per-show preset can live alongside its local branding audio files.

Set `OPENPHONIC_RETENTION_DAYS` to a positive number to delete succeeded or
failed jobs older than that many days at startup, including their upload and job
artifact directories. The default `0` keeps all jobs.

To enable local ML stages, install the optional dependencies:

```bash
pip install -e ".[ml]"
```

Transcription uses `faster-whisper` and stores `transcript.json` plus `transcript.vtt`.
When `stages.filler_removal.enabled` is true, Openphonic writes
`cut_suggestions.json` from transcript word and timing gaps, but does not apply
cuts. Manual review is required before destructive edits are safe.
Diarization uses `pyannote.audio`, requires `HF_TOKEN` for common pretrained Hugging
Face pipelines such as `pyannote/speaker-diarization-3.1`, and stores both
`diarization.rttm` and `diarization.json`. Check the selected model license and
expected CPU/GPU runtime before enabling these stages for production jobs.

## Scaling Notes

For 10 people or fewer, the included SQLite + local worker model is enough to start. If you later need public multi-user hosting, the main changes are:

- Move job execution to a real queue such as Redis Queue, Dramatiq, or Celery
- Store media in S3-compatible object storage
- Put auth and per-user quotas in front of uploads
- Run ML workers separately from the API process
- Add GPU worker scheduling if you enable Demucs/WhisperX/pyannote at scale

## License Choice

This repository uses AGPL-3.0-or-later by default because hosted audio processing is naturally a service. If you prefer maximum permissiveness, switch to Apache-2.0 or MIT before accepting outside contributions.
