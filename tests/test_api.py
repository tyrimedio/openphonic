import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from openphonic.api.routes import (
    MAX_TRANSCRIPT_CORRECTION_FORM_BYTES,
    _read_limited_correction_form,
)
from openphonic.core.database import create_job, get_job, init_db, list_jobs, update_job
from openphonic.core.settings import get_settings
from openphonic.main import create_app
from openphonic.services.storage import job_dir


def configure_tmp_settings(tmp_path, monkeypatch) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "openphonic.sqlite3"
    monkeypatch.setenv("OPENPHONIC_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENPHONIC_DATABASE_PATH", str(db_path))
    monkeypatch.setenv("OPENPHONIC_PIPELINE_CONFIG", "config/default.yml")
    get_settings.cache_clear()
    init_db(db_path)
    return db_path


def write_test_diarization(work_dir: Path) -> dict:
    diarization = {
        "schema_version": 1,
        "engine": "pyannote.audio",
        "model": "pyannote/test-diarization",
        "speaker_count": 2,
        "segments": [
            {"start": 0.0, "end": 1.4, "speaker": "SPEAKER_00", "track": "A"},
            {"start": 1.4, "end": 2.8, "speaker": "SPEAKER_01", "track": "B"},
            {"start": 2.8, "end": 3.2, "speaker": "SPEAKER_00", "track": "C"},
        ],
    }
    (work_dir / "diarization.json").write_text(json.dumps(diarization), encoding="utf-8")
    (work_dir / "diarization.rttm").write_text(
        "SPEAKER test 1 0.000 1.400 <NA> <NA> SPEAKER_00 <NA> <NA>\n",
        encoding="utf-8",
    )
    return diarization


def write_test_cut_suggestions(work_dir: Path) -> dict:
    suggestions = {
        "schema_version": 1,
        "status": "not_applied",
        "source_artifact": "transcript.json",
        "configured_words": ["um", "uh"],
        "min_silence_seconds": 0.75,
        "suggestion_count": 2,
        "suggestions": [
            {
                "id": "cut-0001",
                "type": "filler_word",
                "start": 0.0,
                "end": 0.22,
                "duration": 0.22,
                "text": "Um",
                "normalized_text": "um",
                "segment_index": 0,
                "word_index": 0,
                "reason": "Matched configured filler word.",
            },
            {
                "id": "cut-0002",
                "type": "silence",
                "start": 0.9,
                "end": 1.8,
                "duration": 0.9,
                "source": "word_gap",
                "before_segment_index": 0,
                "before_word_index": 1,
                "after_segment_index": 1,
                "after_word_index": 0,
                "reason": "Detected a timestamp gap longer than the configured threshold.",
            },
        ],
    }
    (work_dir / "cut_suggestions.json").write_text(json.dumps(suggestions), encoding="utf-8")
    return suggestions


def write_many_cut_suggestions(work_dir: Path, count: int) -> dict:
    suggestions = {
        "schema_version": 1,
        "status": "not_applied",
        "source_artifact": "transcript.json",
        "configured_words": ["um"],
        "min_silence_seconds": 0.75,
        "suggestion_count": count,
        "suggestions": [
            {
                "id": f"cut-{index:04d}",
                "type": "filler_word",
                "start": float(index),
                "end": float(index) + 0.2,
                "duration": 0.2,
                "text": "um",
                "reason": "Matched configured filler word.",
            }
            for index in range(count)
        ],
    }
    (work_dir / "cut_suggestions.json").write_text(json.dumps(suggestions), encoding="utf-8")
    return suggestions


def artifact_version(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def test_index_route_renders(tmp_path, monkeypatch) -> None:
    configure_tmp_settings(tmp_path, monkeypatch)
    monkeypatch.setattr("openphonic.pipeline.preflight._binary_available", lambda binary: False)
    monkeypatch.setattr("openphonic.pipeline.preflight._module_available", lambda module: False)
    preset_dir = get_settings().preset_dir
    preset_dir.mkdir(parents=True)
    (preset_dir / "daily-show.yml").write_text(
        """
preset:
  label: Daily show
  description: Daily show production preset.
name: daily-show
stages:
  loudness:
    enabled: true
""",
        encoding="utf-8",
    )

    with TestClient(create_app()) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Openphonic" in response.text
    assert 'value="podcast-default"' in response.text
    assert 'value="speech-cleanup"' in response.text
    assert 'value="vocal-isolation"' in response.text
    assert 'value="transcript-review"' in response.text
    assert 'value="speaker-diarization"' in response.text
    assert 'value="custom:daily-show"' in response.text
    assert "Speech cleanup (unavailable)" in response.text
    assert "Transcript review (unavailable)" in response.text
    assert "Speaker diarization (unavailable)" in response.text
    assert "Daily show" in response.text


def test_create_job_rejects_unknown_preset(tmp_path, monkeypatch) -> None:
    configure_tmp_settings(tmp_path, monkeypatch)

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs",
            data={"preset": "missing"},
            files={"file": ("input.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown pipeline preset: missing"


def test_create_job_rejects_malformed_custom_preset(tmp_path, monkeypatch) -> None:
    configure_tmp_settings(tmp_path, monkeypatch)
    preset_dir = get_settings().preset_dir
    preset_dir.mkdir(parents=True)
    (preset_dir / "broken.yml").write_text("name: [broken\n", encoding="utf-8")
    (preset_dir / "scalar-stage.yml").write_text(
        """
name: scalar-stage
stages:
  loudness: true
""",
        encoding="utf-8",
    )
    started: list[str] = []

    def fake_run_job(job_id: str) -> None:
        started.append(job_id)

    monkeypatch.setattr("openphonic.api.routes.run_job", fake_run_job)

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs",
            data={"preset": "custom:broken"},
            files={"file": ("input.wav", b"audio", "audio/wav")},
            follow_redirects=False,
        )
        scalar_response = client.post(
            "/jobs",
            data={"preset": "custom:scalar-stage"},
            files={"file": ("input.wav", b"audio", "audio/wav")},
            follow_redirects=False,
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown pipeline preset: custom:broken"
    assert scalar_response.status_code == 400
    assert scalar_response.json()["detail"] == "Unknown pipeline preset: custom:scalar-stage"
    assert started == []


def test_create_job_stores_selected_preset(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    started: list[str] = []

    def fake_run_job(job_id: str) -> None:
        started.append(job_id)

    monkeypatch.setattr("openphonic.api.routes.run_job", fake_run_job)
    monkeypatch.setattr("openphonic.pipeline.preflight._binary_available", lambda binary: True)

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs",
            data={"preset": "speech-cleanup"},
            files={"file": ("input.wav", b"audio", "audio/wav")},
            follow_redirects=False,
        )

    assert response.status_code == 303
    job_id = response.headers["location"].removeprefix("/jobs/")
    record = get_job(db_path, job_id)
    assert record is not None
    assert record.to_dict()["config"] == {
        "preset": "speech-cleanup",
        "preset_label": "Speech cleanup",
    }
    assert started == [job_id]


def test_create_job_rejects_unavailable_ml_preset_before_queueing(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    started: list[str] = []

    def fake_run_job(job_id: str) -> None:
        started.append(job_id)

    monkeypatch.setattr("openphonic.api.routes.run_job", fake_run_job)
    monkeypatch.setattr("openphonic.pipeline.preflight._binary_available", lambda binary: False)

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs",
            data={"preset": "speech-cleanup"},
            files={"file": ("input.wav", b"audio", "audio/wav")},
            follow_redirects=False,
        )

    assert response.status_code == 400
    assert "DeepFilterNet noise reduction is enabled" in response.json()["detail"]
    assert list_jobs(db_path) == []
    assert started == []


def test_create_job_stores_transcript_review_preset_when_ml_is_available(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    started: list[str] = []

    def fake_run_job(job_id: str) -> None:
        started.append(job_id)

    monkeypatch.setattr("openphonic.api.routes.run_job", fake_run_job)
    monkeypatch.setattr("openphonic.pipeline.preflight._module_available", lambda module: True)

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs",
            data={"preset": "transcript-review"},
            files={"file": ("input.wav", b"audio", "audio/wav")},
            follow_redirects=False,
        )

    assert response.status_code == 303
    job_id = response.headers["location"].removeprefix("/jobs/")
    record = get_job(db_path, job_id)
    assert record is not None
    assert record.to_dict()["config"] == {
        "preset": "transcript-review",
        "preset_label": "Transcript review",
    }
    assert started == [job_id]


def test_create_job_stores_speaker_diarization_preset_when_ml_is_available(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    get_settings.cache_clear()
    started: list[str] = []

    def fake_run_job(job_id: str) -> None:
        started.append(job_id)

    monkeypatch.setattr("openphonic.api.routes.run_job", fake_run_job)
    monkeypatch.setattr("openphonic.pipeline.preflight._module_available", lambda module: True)

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs",
            data={"preset": "speaker-diarization"},
            files={"file": ("input.wav", b"audio", "audio/wav")},
            follow_redirects=False,
        )

    assert response.status_code == 303
    job_id = response.headers["location"].removeprefix("/jobs/")
    record = get_job(db_path, job_id)
    assert record is not None
    assert record.to_dict()["config"] == {
        "preset": "speaker-diarization",
        "preset_label": "Speaker diarization",
    }
    assert started == [job_id]


def test_create_job_stores_custom_preset(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    preset_dir = get_settings().preset_dir
    preset_dir.mkdir(parents=True)
    (preset_dir / "daily-show.yml").write_text(
        """
preset:
  label: Daily show
name: daily-show
stages:
  loudness:
    enabled: true
""",
        encoding="utf-8",
    )
    started: list[str] = []

    def fake_run_job(job_id: str) -> None:
        started.append(job_id)

    monkeypatch.setattr("openphonic.api.routes.run_job", fake_run_job)

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs",
            data={"preset": "custom:daily-show"},
            files={"file": ("input.wav", b"audio", "audio/wav")},
            follow_redirects=False,
        )

    assert response.status_code == 303
    job_id = response.headers["location"].removeprefix("/jobs/")
    record = get_job(db_path, job_id)
    assert record is not None
    assert record.to_dict()["config"] == {
        "preset": "custom:daily-show",
        "preset_label": "Daily show",
    }
    assert started == [job_id]


def test_job_artifact_routes_list_and_serve_files(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    (work_dir / "job-events.jsonl").write_text('{"event":"job.started"}\n', encoding="utf-8")
    (work_dir / "commands.jsonl").write_text('{"event":"process.started"}\n', encoding="utf-8")
    (work_dir / "00_media_metadata.json").write_text('{"format_name":"wav"}', encoding="utf-8")
    (work_dir / "pipeline_manifest.json").write_text('{"status":"failed"}', encoding="utf-8")

    with TestClient(create_app()) as client:
        list_response = client.get("/api/jobs/job-1/artifacts")
        manifest_response = client.get("/api/jobs/job-1/manifest")
        events_response = client.get("/api/jobs/job-1/events")
        page_response = client.get("/jobs/job-1")

    assert list_response.status_code == 200
    artifacts = list_response.json()
    assert [artifact["name"] for artifact in artifacts] == [
        "00_media_metadata.json",
        "commands.jsonl",
        "job-events.jsonl",
        "pipeline_manifest.json",
    ]
    assert artifacts[0]["url"] == "/api/jobs/job-1/artifacts/00_media_metadata.json"
    assert manifest_response.status_code == 200
    assert manifest_response.json() == {"status": "failed"}
    assert events_response.status_code == 200
    assert "job.started" in events_response.text
    assert page_response.status_code == 200
    assert "Pipeline manifest" in page_response.text
    assert "job-events.jsonl" in page_response.text
    assert "/jobs/job-1/artifacts/pipeline_manifest.json" in page_response.text


def test_artifact_page_previews_text_artifacts(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    (work_dir / "pipeline_manifest.json").write_text('{"status":"failed"}', encoding="utf-8")

    with TestClient(create_app()) as client:
        response = client.get("/jobs/job-1/artifacts/pipeline_manifest.json")

    assert response.status_code == 200
    assert "pipeline_manifest.json" in response.text
    assert "status" in response.text
    assert "failed" in response.text
    assert "/api/jobs/job-1/artifacts/pipeline_manifest.json" in response.text


def test_artifact_page_rejects_missing_and_traversal_paths(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    job_dir(get_settings(), "job-1")

    with TestClient(create_app()) as client:
        missing_response = client.get("/jobs/job-1/artifacts/missing.json")
        traversal_response = client.get("/jobs/job-1/artifacts/%2E%2E/openphonic.sqlite3")

    assert missing_response.status_code == 404
    assert traversal_response.status_code == 400


def test_transcript_page_renders_segments_and_word_timestamps(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    transcript_path = work_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "engine": "faster-whisper",
                "model": "tiny",
                "device": "cpu",
                "language": "en",
                "language_probability": 0.98,
                "duration": 1.25,
                "segments": [
                    {
                        "id": 1,
                        "start": 0.0,
                        "end": 1.25,
                        "text": " Hello world",
                        "words": [
                            {"start": 0.0, "end": 0.4, "word": " Hello", "probability": 0.95},
                            {"start": 0.5, "end": 1.2, "word": " world", "probability": 0.93},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "transcript.vtt").write_text("WEBVTT\n", encoding="utf-8")
    update_job(
        db_path,
        "job-1",
        status="succeeded",
        transcript_path=str(transcript_path),
        current_stage="complete",
        progress=100,
    )

    with TestClient(create_app()) as client:
        response = client.get("/jobs/job-1/transcript")

    assert response.status_code == 200
    assert "Transcript" in response.text
    assert "Hello world" in response.text
    assert "Hello" in response.text
    assert "00:00.000 - 00:01.250" in response.text
    assert "98.0%" in response.text
    assert "/api/jobs/job-1/artifacts/transcript.vtt" in response.text


def test_transcript_page_renders_speaker_labels_from_diarization(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    transcript_path = work_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "engine": "faster-whisper",
                "model": "tiny",
                "language": "en",
                "duration": 2.8,
                "segments": [
                    {"id": 1, "start": 0.0, "end": 1.2, "text": " Welcome back"},
                    {"id": 2, "start": 1.45, "end": 2.7, "text": " Thanks for having me"},
                ],
            }
        ),
        encoding="utf-8",
    )
    write_test_diarization(work_dir)
    (work_dir / "speaker_corrections.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_artifact": "diarization.json",
                "speakers": [{"speaker": "SPEAKER_00", "label": "Host"}],
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "transcript_corrections.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_artifact": "transcript.json",
                "segments": [
                    {
                        "segment_index": 1,
                        "segment_id": 2,
                        "start": 1.45,
                        "end": 2.7,
                        "original_text": " Thanks for having me",
                        "text": "Great to be here\n\nthanks",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    update_job(
        db_path,
        "job-1",
        status="succeeded",
        transcript_path=str(transcript_path),
        current_stage="complete",
        progress=100,
    )

    with TestClient(create_app()) as client:
        response = client.get("/jobs/job-1/transcript")
        speaker_transcript_response = client.get("/api/jobs/job-1/transcript/speakers.txt")

    assert response.status_code == 200
    assert "Speakers" in response.text
    assert "Host" in response.text
    assert "SPEAKER_01" in response.text
    assert "/jobs/job-1/speakers" in response.text
    assert "Speaker TXT" in response.text
    assert speaker_transcript_response.status_code == 200
    assert speaker_transcript_response.headers["content-type"].startswith("text/plain")
    assert speaker_transcript_response.text == (
        "[00:00.000 - 00:01.200] Host: Welcome back\n"
        "[00:01.450 - 00:02.700] SPEAKER_01: Great to be here thanks\n"
    )


def test_transcript_edit_page_saves_corrections_artifact(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    transcript_path = work_dir / "transcript.json"
    transcript = {
        "schema_version": 1,
        "engine": "faster-whisper",
        "model": "tiny",
        "device": "cpu",
        "language": "en",
        "language_probability": 0.98,
        "duration": 2.4,
        "segments": [
            {
                "id": 1,
                "start": 0.0,
                "end": 1.2,
                "text": " Original intro",
                "words": [],
            },
            {
                "id": 2,
                "start": 1.2,
                "end": 2.4,
                "text": " Second segment",
                "words": [],
            },
        ],
    }
    transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
    update_job(
        db_path,
        "job-1",
        status="succeeded",
        transcript_path=str(transcript_path),
        current_stage="complete",
        progress=100,
    )

    with TestClient(create_app()) as client:
        edit_response = client.get("/jobs/job-1/transcript/edit")
        save_response = client.post(
            "/jobs/job-1/transcript/corrections",
            data={
                "corrections_version": "missing",
                "segment_0_text": "Corrected intro\n\nwith context",
                "segment_1_text": " Second segment",
            },
            follow_redirects=False,
        )
        transcript_response = client.get("/jobs/job-1/transcript")
        artifact_response = client.get("/api/jobs/job-1/artifacts/transcript_corrections.json")
        corrected_json_response = client.get("/api/jobs/job-1/transcript/corrected.json")
        corrected_vtt_response = client.get("/api/jobs/job-1/transcript/corrected.vtt")

    corrections_path = work_dir / "transcript_corrections.json"
    assert edit_response.status_code == 200
    assert 'name="corrections_version" value="missing"' in edit_response.text
    assert 'name="segment_0_text"' in edit_response.text
    assert "Original intro" in edit_response.text
    assert save_response.status_code == 303
    assert save_response.headers["location"] == "/jobs/job-1/transcript"
    assert corrections_path.exists()
    assert json.loads(corrections_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "source_artifact": "transcript.json",
        "segments": [
            {
                "segment_index": 0,
                "segment_id": 1,
                "start": 0.0,
                "end": 1.2,
                "original_text": " Original intro",
                "text": "Corrected intro\n\nwith context",
            }
        ],
    }
    assert json.loads(transcript_path.read_text(encoding="utf-8")) == transcript
    assert transcript_response.status_code == 200
    assert "Corrected intro" in transcript_response.text
    assert "Corrections JSON" in transcript_response.text
    assert "Corrected JSON" in transcript_response.text
    assert "Corrected VTT" in transcript_response.text
    assert "1 corrected" in transcript_response.text
    assert artifact_response.status_code == 200
    assert corrected_json_response.status_code == 200
    assert corrected_json_response.headers["content-type"] == "application/json"
    assert (
        corrected_json_response.json()["segments"][0]["text"] == "Corrected intro\n\nwith context"
    )
    assert corrected_json_response.json()["segments"][1]["text"] == " Second segment"
    assert corrected_vtt_response.status_code == 200
    assert corrected_vtt_response.headers["content-type"].startswith("text/vtt")
    assert "WEBVTT" in corrected_vtt_response.text
    assert "Corrected intro with context" in corrected_vtt_response.text
    assert "Corrected intro\n\nwith context" not in corrected_vtt_response.text
    assert "Second segment" in corrected_vtt_response.text


def test_transcript_corrections_reject_stale_edits(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    transcript_path = work_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "segments": [
                    {"id": 1, "start": 0.0, "end": 1.0, "text": " Original intro"},
                    {"id": 2, "start": 1.0, "end": 2.0, "text": " Original outro"},
                ],
            }
        ),
        encoding="utf-8",
    )
    update_job(
        db_path,
        "job-1",
        status="succeeded",
        transcript_path=str(transcript_path),
        current_stage="complete",
        progress=100,
    )

    existing_corrections = {
        "schema_version": 1,
        "source_artifact": "transcript.json",
        "segments": [
            {
                "segment_index": 0,
                "segment_id": 1,
                "start": 0.0,
                "end": 1.0,
                "original_text": " Original intro",
                "text": "Corrected intro",
            }
        ],
    }
    corrections_path = work_dir / "transcript_corrections.json"
    corrections_path.write_text(json.dumps(existing_corrections), encoding="utf-8")

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs/job-1/transcript/corrections",
            data={
                "corrections_version": "missing",
                "segment_0_text": " Original intro",
                "segment_1_text": "Corrected outro",
            },
        )

    assert response.status_code == 409
    assert json.loads(corrections_path.read_text(encoding="utf-8")) == existing_corrections


def test_transcript_corrections_reject_oversized_forms(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    transcript_path = work_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps({"schema_version": 1, "segments": [{"id": 1, "text": " Short"}]}),
        encoding="utf-8",
    )
    update_job(
        db_path,
        "job-1",
        status="succeeded",
        transcript_path=str(transcript_path),
        current_stage="complete",
        progress=100,
    )

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs/job-1/transcript/corrections",
            data={
                "corrections_version": "missing",
                "segment_0_text": "x" * (MAX_TRANSCRIPT_CORRECTION_FORM_BYTES + 1),
            },
        )

    assert response.status_code == 413
    assert not (work_dir / "transcript_corrections.json").exists()


def test_transcript_corrections_reject_oversized_stream_chunk_before_buffering() -> None:
    class OversizedChunk:
        def __len__(self) -> int:
            return MAX_TRANSCRIPT_CORRECTION_FORM_BYTES + 1

        def __iter__(self):
            raise AssertionError("oversized chunk was buffered")

    class FakeRequest:
        headers = {"content-type": "application/x-www-form-urlencoded"}

        async def stream(self):
            yield OversizedChunk()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_read_limited_correction_form(FakeRequest()))

    assert exc_info.value.status_code == 413


def test_transcript_corrections_return_404_when_transcript_is_missing(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)

    with TestClient(create_app()) as client:
        edit_response = client.get("/jobs/job-1/transcript/edit")
        save_response = client.post(
            "/jobs/job-1/transcript/corrections",
            data={"segment_0_text": "No transcript"},
        )

    assert edit_response.status_code == 404
    assert save_response.status_code == 404


def test_speaker_pages_save_corrections_artifact(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    diarization = write_test_diarization(work_dir)
    update_job(db_path, "job-1", status="succeeded", current_stage="complete", progress=100)

    with TestClient(create_app()) as client:
        job_response = client.get("/jobs/job-1")
        speakers_response = client.get("/jobs/job-1/speakers")
        edit_response = client.get("/jobs/job-1/speakers/edit")
        save_response = client.post(
            "/jobs/job-1/speakers/corrections",
            data={
                "corrections_version": "missing",
                "speaker_0_id": "SPEAKER_00",
                "speaker_0_label": "Host",
                "speaker_1_id": "SPEAKER_01",
                "speaker_1_label": "SPEAKER_01",
            },
            follow_redirects=False,
        )
        corrected_response = client.get("/jobs/job-1/speakers")
        artifact_response = client.get("/api/jobs/job-1/artifacts/speaker_corrections.json")

    corrections_path = work_dir / "speaker_corrections.json"
    assert job_response.status_code == 200
    assert 'href="/jobs/job-1/speakers"' in job_response.text
    assert speakers_response.status_code == 200
    assert "SPEAKER_00" in speakers_response.text
    assert "Diarization RTTM" in speakers_response.text
    assert "0 corrected" in speakers_response.text
    assert edit_response.status_code == 200
    assert 'name="corrections_version" value="missing"' in edit_response.text
    assert 'name="speaker_0_label"' in edit_response.text
    assert save_response.status_code == 303
    assert save_response.headers["location"] == "/jobs/job-1/speakers"
    assert json.loads(corrections_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "source_artifact": "diarization.json",
        "speakers": [{"speaker": "SPEAKER_00", "label": "Host"}],
    }
    assert json.loads((work_dir / "diarization.json").read_text(encoding="utf-8")) == diarization
    assert corrected_response.status_code == 200
    assert "Host" in corrected_response.text
    assert "Speaker Corrections JSON" in corrected_response.text
    assert "1 corrected" in corrected_response.text
    assert artifact_response.status_code == 200


def test_speaker_corrections_reject_stale_edits(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    write_test_diarization(work_dir)
    update_job(db_path, "job-1", status="succeeded", current_stage="complete", progress=100)

    existing_corrections = {
        "schema_version": 1,
        "source_artifact": "diarization.json",
        "speakers": [{"speaker": "SPEAKER_00", "label": "Host"}],
    }
    corrections_path = work_dir / "speaker_corrections.json"
    corrections_path.write_text(json.dumps(existing_corrections), encoding="utf-8")

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs/job-1/speakers/corrections",
            data={
                "corrections_version": "missing",
                "speaker_0_id": "SPEAKER_00",
                "speaker_0_label": "Guest",
            },
        )

    assert response.status_code == 409
    assert json.loads(corrections_path.read_text(encoding="utf-8")) == existing_corrections


def test_speaker_pages_return_404_when_diarization_is_missing(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)

    with TestClient(create_app()) as client:
        page_response = client.get("/jobs/job-1/speakers")
        edit_response = client.get("/jobs/job-1/speakers/edit")
        save_response = client.post(
            "/jobs/job-1/speakers/corrections",
            data={"corrections_version": "missing"},
        )

    assert page_response.status_code == 404
    assert edit_response.status_code == 404
    assert save_response.status_code == 404


def test_cut_review_page_saves_review_artifact(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    suggestions = write_test_cut_suggestions(work_dir)
    update_job(db_path, "job-1", status="succeeded", current_stage="complete", progress=100)

    with TestClient(create_app()) as client:
        job_response = client.get("/jobs/job-1")
        page_response = client.get("/jobs/job-1/cuts")
        save_response = client.post(
            "/jobs/job-1/cuts/review",
            data={
                "suggestions_version": artifact_version(work_dir / "cut_suggestions.json"),
                "review_version": "missing",
                "suggestion_0_id": "cut-0001",
                "suggestion_0_decision": "approved",
                "suggestion_0_note": "remove",
                "suggestion_1_id": "cut-0002",
                "suggestion_1_decision": "rejected",
                "suggestion_1_note": "",
            },
            follow_redirects=False,
        )
        reviewed_response = client.get("/jobs/job-1/cuts")
        artifact_response = client.get("/api/jobs/job-1/artifacts/cut_review.json")

    review_path = work_dir / "cut_review.json"
    assert job_response.status_code == 200
    assert 'href="/jobs/job-1/cuts"' in job_response.text
    assert page_response.status_code == 200
    assert "Cut Review" in page_response.text
    assert 'name="review_version" value="missing"' in page_response.text
    assert 'name="suggestion_0_decision"' in page_response.text
    assert "cut-0001" in page_response.text
    assert save_response.status_code == 303
    assert save_response.headers["location"] == "/jobs/job-1/cuts"
    assert json.loads(review_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "source_artifact": "cut_suggestions.json",
        "decisions": [
            {
                "suggestion_id": "cut-0001",
                "decision": "approved",
                "type": "filler_word",
                "start": 0.0,
                "end": 0.22,
                "duration": 0.22,
                "note": "remove",
            },
            {
                "suggestion_id": "cut-0002",
                "decision": "rejected",
                "type": "silence",
                "start": 0.9,
                "end": 1.8,
                "duration": 0.9,
            },
        ],
    }
    saved_suggestions = json.loads((work_dir / "cut_suggestions.json").read_text(encoding="utf-8"))
    assert saved_suggestions == suggestions
    assert reviewed_response.status_code == 200
    assert "2 decided" in reviewed_response.text
    assert "Review JSON" in reviewed_response.text
    assert artifact_response.status_code == 200


def test_cut_review_rejects_stale_review_saves(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    write_test_cut_suggestions(work_dir)
    update_job(db_path, "job-1", status="succeeded", current_stage="complete", progress=100)

    existing_review = {
        "schema_version": 1,
        "source_artifact": "cut_suggestions.json",
        "decisions": [{"suggestion_id": "cut-0001", "decision": "approved"}],
    }
    review_path = work_dir / "cut_review.json"
    review_path.write_text(json.dumps(existing_review), encoding="utf-8")

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs/job-1/cuts/review",
            data={
                "suggestions_version": artifact_version(work_dir / "cut_suggestions.json"),
                "review_version": "missing",
                "suggestion_0_id": "cut-0001",
                "suggestion_0_decision": "rejected",
            },
        )

    assert response.status_code == 409
    assert json.loads(review_path.read_text(encoding="utf-8")) == existing_review


def test_cut_review_allows_large_suggestion_forms(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    suggestion_count = 3_333
    write_many_cut_suggestions(work_dir, suggestion_count)
    update_job(db_path, "job-1", status="succeeded", current_stage="complete", progress=100)

    form = {
        "suggestions_version": artifact_version(work_dir / "cut_suggestions.json"),
        "review_version": "missing",
    }
    for index in range(suggestion_count):
        form[f"suggestion_{index}_id"] = f"cut-{index:04d}"
        form[f"suggestion_{index}_decision"] = "approved" if index == 0 else "pending"
        form[f"suggestion_{index}_note"] = ""

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs/job-1/cuts/review",
            data=form,
            follow_redirects=False,
        )

    review = json.loads((work_dir / "cut_review.json").read_text(encoding="utf-8"))
    assert response.status_code == 303
    assert response.headers["location"] == "/jobs/job-1/cuts"
    assert review["decisions"] == [
        {
            "suggestion_id": "cut-0000",
            "decision": "approved",
            "type": "filler_word",
            "start": 0.0,
            "end": 0.2,
            "duration": 0.2,
        }
    ]


def test_cut_review_allows_large_decided_review_artifacts(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    suggestion_count = 7_000
    write_many_cut_suggestions(work_dir, suggestion_count)
    update_job(db_path, "job-1", status="succeeded", current_stage="complete", progress=100)

    form = {
        "suggestions_version": artifact_version(work_dir / "cut_suggestions.json"),
        "review_version": "missing",
    }
    for index in range(suggestion_count):
        form[f"suggestion_{index}_id"] = f"cut-{index:04d}"
        form[f"suggestion_{index}_decision"] = "approved"
        form[f"suggestion_{index}_note"] = ""

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs/job-1/cuts/review",
            data=form,
            follow_redirects=False,
        )

    review = json.loads((work_dir / "cut_review.json").read_text(encoding="utf-8"))
    assert response.status_code == 303
    assert response.headers["location"] == "/jobs/job-1/cuts"
    assert len(review["decisions"]) == suggestion_count
    assert review["decisions"][0]["suggestion_id"] == "cut-0000"
    assert review["decisions"][-1]["suggestion_id"] == "cut-6999"
    assert len(json.dumps(review, indent=2).encode("utf-8")) > 1024 * 1024


def test_cut_review_page_applies_approved_cuts(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    write_test_cut_suggestions(work_dir)
    output_path = work_dir / "06_loudnorm.m4a"
    output_path.write_bytes(b"audio")
    review_path = work_dir / "cut_review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_artifact": "cut_suggestions.json",
                "decisions": [{"suggestion_id": "cut-0001", "decision": "approved"}],
            }
        ),
        encoding="utf-8",
    )
    update_job(
        db_path,
        "job-1",
        status="succeeded",
        output_path=str(output_path),
        current_stage="complete",
        progress=100,
    )
    applied: list[dict] = []

    def fake_apply_approved_cuts(**kwargs) -> None:
        applied.append(kwargs)

    monkeypatch.setattr("openphonic.api.routes.apply_approved_cuts", fake_apply_approved_cuts)

    with TestClient(create_app()) as client:
        page_response = client.get("/jobs/job-1/cuts")
        response = client.post(
            "/jobs/job-1/cuts/apply",
            data={
                "suggestions_version": artifact_version(work_dir / "cut_suggestions.json"),
                "review_version": artifact_version(review_path),
            },
            follow_redirects=False,
        )

    assert page_response.status_code == 200
    assert "Apply approved cuts" in page_response.text
    assert response.status_code == 303
    assert response.headers["location"] == "/jobs/job-1/cuts"
    assert len(applied) == 1
    assert applied[0]["job_id"] == "job-1"
    assert applied[0]["input_path"] == output_path
    assert [cut.suggestion_id for cut in applied[0]["cuts"]] == ["cut-0001"]


def test_cut_review_hides_stale_reviewed_audio_after_review_changes(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    write_test_cut_suggestions(work_dir)
    output_path = work_dir / "06_loudnorm.m4a"
    output_path.write_bytes(b"audio")
    reviewed_path = work_dir / "cut_reviewed.m4a"
    reviewed_path.write_bytes(b"reviewed audio")
    review_path = work_dir / "cut_review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_artifact": "cut_suggestions.json",
                "decisions": [{"suggestion_id": "cut-0001", "decision": "approved"}],
            }
        ),
        encoding="utf-8",
    )
    suggestions_version = artifact_version(work_dir / "cut_suggestions.json")
    applied_review_version = artifact_version(review_path)
    (work_dir / "cut_apply_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "succeeded",
                "suggestions_version": suggestions_version,
                "review_version": applied_review_version,
                "output_artifact": "cut_reviewed.m4a",
            }
        ),
        encoding="utf-8",
    )
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_artifact": "cut_suggestions.json",
                "decisions": [{"suggestion_id": "cut-0001", "decision": "rejected"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    update_job(
        db_path,
        "job-1",
        status="succeeded",
        output_path=str(output_path),
        current_stage="complete",
        progress=100,
    )

    with TestClient(create_app()) as client:
        response = client.get("/jobs/job-1/cuts")

    assert response.status_code == 200
    assert "Apply manifest" in response.text
    assert "Download reviewed audio" not in response.text
    assert "/api/jobs/job-1/artifacts/cut_reviewed.m4a" not in response.text


def test_cut_review_apply_requires_approved_cuts(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    write_test_cut_suggestions(work_dir)
    output_path = work_dir / "06_loudnorm.m4a"
    output_path.write_bytes(b"audio")
    review_path = work_dir / "cut_review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_artifact": "cut_suggestions.json",
                "decisions": [{"suggestion_id": "cut-0001", "decision": "rejected"}],
            }
        ),
        encoding="utf-8",
    )
    update_job(
        db_path,
        "job-1",
        status="succeeded",
        output_path=str(output_path),
        current_stage="complete",
        progress=100,
    )

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs/job-1/cuts/apply",
            data={
                "suggestions_version": artifact_version(work_dir / "cut_suggestions.json"),
                "review_version": artifact_version(review_path),
            },
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "No approved cuts to apply."


def test_cut_review_rejects_stale_suggestions(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    write_test_cut_suggestions(work_dir)
    stale_version = artifact_version(work_dir / "cut_suggestions.json")
    update_job(db_path, "job-1", status="succeeded", current_stage="complete", progress=100)
    changed_suggestions = write_test_cut_suggestions(work_dir)
    changed_suggestions["suggestions"].append(
        {
            "id": "cut-0003",
            "type": "filler_word",
            "start": 2.0,
            "end": 2.2,
            "duration": 0.2,
            "text": "uh",
        }
    )
    (work_dir / "cut_suggestions.json").write_text(
        json.dumps(changed_suggestions), encoding="utf-8"
    )

    with TestClient(create_app()) as client:
        response = client.post(
            "/jobs/job-1/cuts/review",
            data={
                "suggestions_version": stale_version,
                "review_version": "missing",
                "suggestion_0_id": "cut-0001",
                "suggestion_0_decision": "approved",
            },
        )

    assert response.status_code == 409
    assert not (work_dir / "cut_review.json").exists()


def test_cut_review_returns_404_when_suggestions_are_missing(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)

    with TestClient(create_app()) as client:
        page_response = client.get("/jobs/job-1/cuts")
        save_response = client.post(
            "/jobs/job-1/cuts/review",
            data={"suggestions_version": "missing", "review_version": "missing"},
        )

    assert page_response.status_code == 404
    assert save_response.status_code == 404


def test_transcript_page_returns_404_when_missing(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)

    with TestClient(create_app()) as client:
        response = client.get("/jobs/job-1/transcript")

    assert response.status_code == 404


def test_artifact_download_rejects_missing_and_traversal_paths(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    job_dir(get_settings(), "job-1")

    with TestClient(create_app()) as client:
        missing_response = client.get("/api/jobs/job-1/artifacts/missing.json")
        traversal_response = client.get("/api/jobs/job-1/artifacts/%2E%2E/openphonic.sqlite3")

    assert missing_response.status_code == 404
    assert traversal_response.status_code == 400


def test_retry_failed_job_route_requeues_and_runs_background_task(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)

    update_job(db_path, "job-1", status="failed", error_message="boom", current_stage="failed")
    work_dir = job_dir(get_settings(), "job-1")
    (work_dir / "commands.jsonl").write_text("old commands", encoding="utf-8")
    ran: list[str] = []

    def fake_run_job(job_id: str) -> None:
        ran.append(job_id)

    monkeypatch.setattr("openphonic.api.routes.run_job", fake_run_job)

    with TestClient(create_app()) as client:
        response = client.post("/api/jobs/job-1/retry")

    record = get_job(db_path, "job-1")
    assert response.status_code == 200
    assert response.json()["id"] == "job-1"
    assert ran == ["job-1"]
    assert record is not None
    assert record.status == "queued"
    assert list((work_dir / "attempts").glob("*/commands.jsonl"))


def test_retry_preflights_stored_preset_before_mutating_job(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(
        db_path,
        job_id="job-1",
        original_filename="input.wav",
        input_path=input_path,
        config={"preset": "speech-cleanup", "preset_label": "Speech cleanup"},
    )
    update_job(db_path, "job-1", status="failed", error_message="boom", current_stage="failed")
    work_dir = job_dir(get_settings(), "job-1")
    (work_dir / "commands.jsonl").write_text("old commands", encoding="utf-8")
    ran: list[str] = []

    def fake_run_job(job_id: str) -> None:
        ran.append(job_id)

    monkeypatch.setattr("openphonic.api.routes.run_job", fake_run_job)
    monkeypatch.setattr("openphonic.pipeline.preflight._binary_available", lambda binary: False)

    with TestClient(create_app()) as client:
        response = client.post("/api/jobs/job-1/retry")

    record = get_job(db_path, "job-1")
    assert response.status_code == 400
    assert "DeepFilterNet noise reduction is enabled" in response.json()["detail"]
    assert ran == []
    assert record is not None
    assert record.status == "failed"
    assert not (work_dir / "attempts").exists()
    assert (work_dir / "commands.jsonl").read_text(encoding="utf-8") == "old commands"


def test_retry_wraps_malformed_preset_preflight_errors(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    broken_config = tmp_path / "broken-default.yml"
    broken_config.write_text(
        """
name: broken
stages: true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENPHONIC_PIPELINE_CONFIG", str(broken_config))
    get_settings.cache_clear()
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    update_job(db_path, "job-1", status="failed", error_message="boom", current_stage="failed")
    work_dir = job_dir(get_settings(), "job-1")
    (work_dir / "commands.jsonl").write_text("old commands", encoding="utf-8")
    ran: list[str] = []

    def fake_run_job(job_id: str) -> None:
        ran.append(job_id)

    monkeypatch.setattr("openphonic.api.routes.run_job", fake_run_job)

    with TestClient(create_app()) as client:
        response = client.post("/api/jobs/job-1/retry")

    record = get_job(db_path, "job-1")
    assert response.status_code == 400
    assert "Invalid pipeline preset" in response.json()["detail"]
    assert ran == []
    assert record is not None
    assert record.status == "failed"
    assert not (work_dir / "attempts").exists()


def test_retry_rejects_non_failed_jobs(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    update_job(db_path, "job-1", status="succeeded", current_stage="complete", progress=100)

    with TestClient(create_app()) as client:
        response = client.post("/api/jobs/job-1/retry")

    assert response.status_code == 409
