# Openphonic

Openphonic is a small-scale, self-hosted audio post-production platform inspired by Auphonic. It is designed for a single team or a handful of users first, with a clear path to heavier model-based processing when you want it.

The default pipeline is intentionally FFmpeg-first so the app is useful before you install GPU-sized dependencies. Optional stages are provided for Whisper/faster-whisper or Deepgram transcription, pyannote or Deepgram diarization, DeepFilterNet noise reduction, and Demucs source separation.

## What This Initial Version Includes

- FastAPI app with upload, job status, transcript, and download endpoints
- SQLite job database for small deployments
- Local filesystem storage under `data/`
- Config-driven audio pipeline
- FFprobe media validation plus FFmpeg ingest, silence trimming, and two-pass loudness normalization
- Optional adapters for faster-whisper, Deepgram, pyannote, DeepFilterNet CLI, and Demucs
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

Use a built-in or custom preset when you want to run real audio through the
same pipeline configuration exposed by the uploader:

```bash
openphonic process ./sample.wav --preset transcript-review --output ./processed.m4a
openphonic process ./sample.wav --preset custom:daily-show --output ./processed.wav
```

`process` runs the same optional-stage preflight as uploads and smoke tests
before creating work artifacts or starting the pipeline.

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

You can run the same generated input through any built-in or custom preset:

```bash
openphonic smoke-test --preset transcript-review
openphonic smoke-test --preset custom:daily-show
```

`smoke-test` runs the same optional-stage preflight used by the web uploader
before generating input or starting the pipeline. If a selected preset needs
missing local tools, Python packages, model tokens, or configured assets, the
command exits before doing work and prints the setup issue.

Use this as the first local verification pass after setup. If it fails, fix the
FFmpeg/core pipeline path before testing transcription, diarization, denoise, or
source separation.

For the complete local acceptance workflow, including strict artifact/log
inspection and the first optional-ML checks, see
[docs/local-verification.md](docs/local-verification.md).

## Job Inspection

After a smoke test, CLI process run, or web job finishes, inspect the work
directory and verify the manifest still points at existing files:

```bash
openphonic inspect-job ./data/smoke-test/work
openphonic inspect-job ./data/jobs/<job-id> --strict
```

The command reads `pipeline_manifest.json`, reports the pipeline status, input,
output, and artifact paths, and warns when expected files are missing. Use
`--strict` when local validation scripts should fail on missing artifacts.

## Job Event Inspection

Every web job writes `job-events.jsonl` with job starts, progress updates,
terminal job status, retries, and reviewed-cut apply events. Inspect it when you
need a quick lifecycle summary for a job:

```bash
openphonic inspect-events ./data/jobs/<job-id>/job-events.jsonl
openphonic inspect-events ./data/jobs/<job-id>/job-events.jsonl --strict
```

The command reports final job status, last progress, retries, failures,
interrupted jobs, malformed entries, and incomplete job or cut-apply runs. Use
`--strict` when local validation should fail on failed, interrupted, malformed,
or unterminated job event logs.

## Command Log Inspection

Each pipeline run writes `commands.jsonl` with FFmpeg and FFprobe starts,
successes, failures, return codes, and durations. Inspect it when a smoke test or
job artifact needs a quick command-level summary:

```bash
openphonic inspect-commands ./data/smoke-test/work/commands.jsonl
openphonic inspect-commands ./data/jobs/<job-id>/commands.jsonl --strict
```

The command reports process starts, successes, failures, malformed log entries,
executables used, and total recorded command duration. Use `--strict` when
failure or malformed command events should fail local validation.

## Preset Readiness

Check which built-in and custom presets can run on the current machine before
uploading a job or running a heavier smoke test:

```bash
openphonic readiness
```

The report marks each preset as `ready` or `blocked` and prints missing local
tools, Python packages, tokens, or configured assets. Limit the report to one
or more presets when you are preparing a specific real-audio run:

```bash
openphonic readiness --preset transcript-review
openphonic readiness --preset speech-cleanup --preset speaker-diarization
```

Use `--strict` when you want setup automation to fail if any reported preset is
blocked:

```bash
openphonic readiness --strict
```

## Transcript Inspection

After running a transcription preset on real audio, summarize the transcript
artifact before reviewing corrections or cut suggestions:

```bash
openphonic inspect-transcript ./data/jobs/<job-id>/transcript.json
```

The command reports segment counts, word counts, timed-word coverage, and
warnings for missing or invalid timestamps. Use `--strict` when validating ML
setup from a script:

```bash
openphonic inspect-transcript ./data/jobs/<job-id>/transcript.json --strict
```

## Diarization Inspection

After running a speaker diarization preset, summarize the speaker artifact before
reviewing labels or speaker exports:

```bash
openphonic inspect-diarization ./data/jobs/<job-id>/diarization.json
```

The command reports declared and detected speaker counts, segment timing
coverage, and total speaker time. Use `--duration` to validate turns against the
source audio length, and `--strict` when setup scripts should fail on warnings:

```bash
openphonic inspect-diarization ./data/jobs/<job-id>/diarization.json --duration 3600 --strict
```

## Cut Suggestion Inspection

After running the `transcript-review` preset, summarize the generated
non-destructive cut suggestions before manual review:

```bash
openphonic inspect-cut-suggestions ./data/jobs/<job-id>/cut_suggestions.json
```

The command reports review status, configured filler words, suggestion counts,
suggestion types, and proposed cut durations. Use `--duration` to check
suggestions against the source audio length, and `--strict` when scripts should
fail on malformed or non-reviewable suggestions:

```bash
openphonic inspect-cut-suggestions ./data/jobs/<job-id>/cut_suggestions.json --duration 3600 --strict
```

## Docker

```bash
docker compose up --build
```

The app stores uploads, processed files, and SQLite data in the `openphonic-data` volume.

## Hosted Deployment Direction

The planned hosted deployment is an operator-managed app for non-technical
users. The target shape is:

```text
Users
  |
  v
Cloudflare Tunnel + Access email allowlist
  |
  v
CPU VPS running Openphonic with Docker Compose
  |
  +--> local FFmpeg processing
  +--> local SQLite and filesystem artifacts
  +--> hosted ASR/diarization provider adapter
```

The selected hosted provider path is Deepgram Nova-3 for transcription and
diarization. Local processing remains the default. To use the hosted provider,
set `TRANSCRIPTION_PROVIDER=deepgram`, `DEEPGRAM_API_KEY`, and optionally
`OPENPHONIC_DEEPGRAM_MODEL=nova-3`. In Deepgram mode, transcription posts the
processed audio to Deepgram's pre-recorded `/v1/listen` API with smart
formatting and utterances enabled. When the selected preset enables
diarization, speaker labels are requested in the same Deepgram call and written
to Openphonic's existing `diarization.json` and RTTM artifacts instead of
running local pyannote. Startup and preset preflight validate the configured
Deepgram key with Deepgram's `/v1/auth/token` endpoint before queueing
Deepgram-backed jobs; this checks credentials without uploading audio.

The intended v1 hosted security boundary is Cloudflare Access in front of the
whole app, not per-user accounts inside Openphonic. The app should continue to
work locally without paid services, while future hosted-provider code remains
optional and explicit.

## Pipeline

The default preset lives in [config/default.yml](config/default.yml). It runs:

1. Validate media metadata and write a metadata artifact
2. Ingest to a normalized working WAV
3. Optional DeepFilterNet noise reduction
4. Optional Demucs separation
5. Optional silence trimming
6. Optional intro/outro insertion from local media
7. FFmpeg two-pass loudness normalization
8. Optional faster-whisper or Deepgram transcription
9. Optional non-destructive cut suggestions from transcript timestamps
10. Optional pyannote diarization, or Deepgram speaker labels when
    `TRANSCRIPTION_PROVIDER=deepgram`

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
Diarization uses `pyannote.audio` in local provider mode, requires `HF_TOKEN`
for common pretrained Hugging Face pipelines such as
`pyannote/speaker-diarization-3.1`, and stores both `diarization.rttm` and
`diarization.json`. In Deepgram provider mode, diarization is requested during
transcription and the local pyannote dependency is not required. Check the
selected model license, provider cost, privacy posture, and expected runtime
before enabling these stages for production jobs.

## Scaling Notes

For 10 people or fewer, the included SQLite + local worker model is enough to
start. The near-term hosted target is still a single CPU VPS, with Cloudflare
Tunnel and Access handling public ingress and authentication. In that mode,
FFmpeg stays local, but transcription and diarization should move to a hosted
provider adapter so non-technical users do not manage local ML dependencies.

The main changes before public multi-user hosting are:

- Harden the optional Deepgram transcription/diarization provider adapter with
  remote readiness checks and hosted deployment docs
- Keep local `faster-whisper` and `pyannote.audio` paths intact behind a local
  provider mode
- Document Cloudflare Tunnel + Access deployment and backup setup
- Move job execution to a real queue such as Redis Queue, Dramatiq, or Celery
- Store media in S3-compatible object storage
- Add app-native auth and per-user quotas only after the Cloudflare Access-gated
  v1 is stable
- Run ML workers separately from the API process
- Add GPU worker scheduling if you enable Demucs/WhisperX/pyannote at scale

## License Choice

This repository uses AGPL-3.0-or-later by default because hosted audio processing is naturally a service. If you prefer maximum permissiveness, switch to Apache-2.0 or MIT before accepting outside contributions.
