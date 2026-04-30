# Local Verification Guide

Use this guide after setup, after merging pipeline/storage changes, or before
starting optional ML validation. The goal is to prove the local FFmpeg-first path
is healthy before spending time on heavier transcription, diarization, denoise,
or source-separation runs.

## Baseline Setup Check

Confirm the CLI is installed and the preset preflight can inspect the host:

```bash
openphonic --help
openphonic readiness
```

On a default development install, `podcast-default` should be ready when FFmpeg
and FFprobe are on `PATH`. Optional presets can be blocked until their Python
packages, model tokens, or local CLIs are installed. That is expected; blocked
optional presets should fail before queueing work.

For a specific preset:

```bash
openphonic readiness --preset podcast-default --strict
openphonic readiness --preset transcript-review
openphonic readiness --preset speech-cleanup
openphonic readiness --preset speaker-diarization
```

## FFmpeg Smoke Test

Run the generated-tone smoke test first. It avoids needing a sample media file
and exercises media generation, validation, ingest, trimming, loudness
normalization, final encode, command logging, and the pipeline manifest.

```bash
openphonic smoke-test
```

The default output lives under `data/smoke-test/`:

```text
data/smoke-test/processed.m4a
data/smoke-test/work/
```

For an isolated run:

```bash
openphonic smoke-test \
  --output /tmp/openphonic-smoke/processed.m4a \
  --work-dir /tmp/openphonic-smoke/work
```

## Strict Artifact Inspection

After a smoke test, inspect the generated manifest and command log:

```bash
openphonic inspect-job data/smoke-test/work --strict
openphonic inspect-commands data/smoke-test/work/commands.jsonl --strict
```

The strict job inspection should report:

- `Status: succeeded`
- the input and output paths as `[ok]`
- all manifest artifacts present
- no warnings

The strict command inspection should report:

- matching started and terminal command counts
- `Failed: 0`
- `Unterminated: 0`
- `Malformed entries: 0`
- no warnings

## Real File CLI Check

When you have a small local media sample, run the real `process` command with
the default preset:

```bash
openphonic process ./sample.wav \
  --preset podcast-default \
  --output ./processed.m4a
```

If the command fails before work starts, inspect the printed preflight issue
first. If it fails after work starts, inspect the work directory printed or
derived from the output path:

```bash
openphonic inspect-job ./processed --strict
openphonic inspect-commands ./processed/commands.jsonl --strict
```

## Web Job Check

Run the web app:

```bash
uvicorn openphonic.main:app --reload
```

Then open `http://127.0.0.1:8000` and run a default-preset upload. After the
job finishes, check:

- the job detail page shows a terminal status
- processed audio downloads
- individual artifacts download
- the artifact ZIP downloads
- transcript/cut links only appear when their artifacts exist

Use the CLI inspectors against the created job directory:

```bash
openphonic inspect-job data/jobs/<job-id> --strict
openphonic inspect-commands data/jobs/<job-id>/commands.jsonl --strict
openphonic inspect-events data/jobs/<job-id>/job-events.jsonl --strict
```

For failed jobs, strict inspection should fail but still give actionable
diagnostics. Failed jobs should remain inspectable and retryable.

## Optional ML Reality Pass

Only start this after the FFmpeg-only smoke and web checks pass.

```bash
pip install -e ".[dev,ml]"
openphonic readiness --preset transcript-review --strict
openphonic process ./speech-sample.wav \
  --preset transcript-review \
  --output ./speech-processed.m4a
```

Then inspect the ML artifacts:

```bash
openphonic inspect-transcript ./speech-processed/transcript.json --strict
openphonic inspect-cut-suggestions ./speech-processed/cut_suggestions.json --strict
```

For diarization, configure `HF_TOKEN` and validate separately:

```bash
openphonic readiness --preset speaker-diarization
openphonic process ./speech-sample.wav \
  --preset speaker-diarization \
  --output ./speaker-processed.m4a
openphonic inspect-diarization ./speaker-processed/diarization.json --strict
```

For speech cleanup, install/configure DeepFilterNet and test on noisy speech:

```bash
openphonic readiness --preset speech-cleanup
openphonic process ./noisy-speech.wav \
  --preset speech-cleanup \
  --output ./cleaned-speech.m4a
```

Record runtime, CPU/GPU behavior, and whether the output is actually better,
not just whether the artifact exists.

## Ship Criteria

Before moving to larger ML or hosted-workflow changes, a local verification pass
should show:

- default FFmpeg-only smoke test succeeds
- manifest and command logs pass strict inspection
- web upload succeeds with the default preset
- web job event logs pass strict inspection
- failed jobs stay inspectable
- retries do not destroy previous artifacts
- optional ML presets either preflight cleanly or explain what is missing
