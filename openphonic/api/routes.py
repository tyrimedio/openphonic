from __future__ import annotations

import json
import math
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import parse_qs, quote

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from openphonic.core.database import create_job, init_db
from openphonic.core.settings import get_settings
from openphonic.pipeline.config import (
    PipelineConfig,
    TargetFormat,
    available_presets,
    load_pipeline_config_for_preset,
    preset_by_id,
)
from openphonic.pipeline.preflight import format_preflight_issues, pipeline_preflight_issues
from openphonic.services.cuts import (
    CUT_APPLY_MANIFEST_ARTIFACT,
    CutApplyError,
    apply_approved_cuts,
    approved_cuts_from_review,
)
from openphonic.services.jobs import (
    JobRetryError,
    fetch_job,
    recent_jobs,
    reserve_upload,
    retry_failed_job,
    run_job,
)
from openphonic.services.storage import (
    JobArtifact,
    job_artifact_path,
    job_dir,
    list_job_artifacts,
    save_upload_file,
)

router = APIRouter()
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))
UploadFormFile = Annotated[UploadFile, File()]
PresetForm = Annotated[str, Form()]

COMMON_ARTIFACTS = [
    ("Job events", "job-events.jsonl", "/events"),
    ("Command log", "commands.jsonl", "/commands"),
    ("Media metadata", "00_media_metadata.json", "/metadata"),
    ("Pipeline manifest", "pipeline_manifest.json", "/manifest"),
]
MAX_ARTIFACT_PREVIEW_BYTES = 128 * 1024
TEXT_ARTIFACT_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".rttm",
    ".txt",
    ".vtt",
    ".yaml",
    ".yml",
}
TRANSCRIPT_CORRECTIONS_ARTIFACT = "transcript_corrections.json"
SPEAKER_CORRECTIONS_ARTIFACT = "speaker_corrections.json"
CUT_SUGGESTIONS_ARTIFACT = "cut_suggestions.json"
CUT_REVIEW_ARTIFACT = "cut_review.json"
MAX_CORRECTION_FORM_BYTES = 1024 * 1024
MAX_CORRECTION_FIELDS = 10_000
MAX_CUT_REVIEW_FORM_BYTES = 8 * 1024 * 1024
MAX_CUT_REVIEW_FORM_FIELDS = 200_000
MAX_CUT_REVIEW_ARTIFACT_BYTES = MAX_CUT_REVIEW_FORM_BYTES
CUT_REVIEW_FORM_STATIC_FIELDS = 2
CUT_REVIEW_FORM_FIELDS_PER_SUGGESTION = 3
CUT_REVIEW_FORM_BYTES_PER_SUGGESTION = 256
MAX_TRANSCRIPT_CORRECTION_FORM_BYTES = MAX_CORRECTION_FORM_BYTES
MAX_TRANSCRIPT_CORRECTION_FIELDS = MAX_CORRECTION_FIELDS


def _ensure_ready() -> None:
    settings = get_settings()
    init_db(settings.database_path)


def _artifact_url(job_id: str, artifact_name: str) -> str:
    return f"/api/jobs/{job_id}/artifacts/{quote(artifact_name, safe='/')}"


def _artifact_page_url(job_id: str, artifact_name: str) -> str:
    return f"/jobs/{job_id}/artifacts/{quote(artifact_name, safe='/')}"


def _artifact_payload(job_id: str, artifact: JobArtifact) -> dict:
    return {
        "name": artifact.name,
        "size_bytes": artifact.size_bytes,
        "size": _format_bytes(artifact.size_bytes),
        "url": _artifact_url(job_id, artifact.name),
        "download_url": _artifact_url(job_id, artifact.name),
        "page_url": _artifact_page_url(job_id, artifact.name),
    }


def _require_job(job_id: str):
    record = fetch_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return record


def _artifact_response(job_id: str, artifact_name: str, media_type: str | None = None):
    _ensure_ready()
    _require_job(job_id)
    try:
        path = job_artifact_path(get_settings(), job_id, artifact_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found.") from exc
    return FileResponse(path, filename=path.name, media_type=media_type)


def _artifact_bundle_response(job_id: str):
    _ensure_ready()
    _require_job(job_id)
    try:
        artifacts = list_job_artifacts(get_settings(), job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not artifacts:
        raise HTTPException(status_code=404, detail="No artifacts available.")

    bundle = tempfile.NamedTemporaryFile(prefix=f"{job_id}-artifacts-", suffix=".zip", delete=False)
    bundle_path = Path(bundle.name)
    bundle.close()
    try:
        with zipfile.ZipFile(bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for artifact in artifacts:
                path = job_artifact_path(get_settings(), job_id, artifact.name)
                archive.write(path, arcname=artifact.name)
    except Exception:
        bundle_path.unlink(missing_ok=True)
        raise

    return FileResponse(
        bundle_path,
        filename=f"{job_id}-artifacts.zip",
        media_type="application/zip",
        background=BackgroundTask(lambda: bundle_path.unlink(missing_ok=True)),
    )


def _format_bytes(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def _format_seconds(value: Any) -> str:
    if value is None:
        return "-"
    try:
        total_ms = int(round(float(value) * 1000))
    except (TypeError, ValueError):
        return "-"
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    seconds, milliseconds = divmod(total_ms, 1000)
    if hours:
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
    return f"{minutes:02}:{seconds:02}.{milliseconds:03}"


def _format_probability(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _vtt_timestamp(value: Any) -> str:
    seconds = _numeric_seconds(value)
    if seconds is None or seconds < 0:
        raise ValueError("Transcript segment timestamps must be finite non-negative numbers.")
    total_ms = int(round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    seconds_part, milliseconds = divmod(total_ms, 1000)
    return f"{hours:02}:{minutes:02}:{seconds_part:02}.{milliseconds:03}"


def _job_relative_artifact_name(job_id: str, path: Path) -> str | None:
    root = (get_settings().jobs_dir / job_id).resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return None


def _load_transcript_artifact(
    job_id: str, transcript_path: str | None
) -> tuple[dict[str, Any], str]:
    if not transcript_path:
        raise HTTPException(status_code=404, detail="Transcript not available.")

    artifact_name = _job_relative_artifact_name(job_id, Path(transcript_path))
    if artifact_name is None:
        raise HTTPException(status_code=404, detail="Transcript artifact not found.")

    try:
        path = job_artifact_path(get_settings(), job_id, artifact_name)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Transcript artifact not found.") from exc

    try:
        transcript = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Transcript artifact is invalid JSON.") from exc
    if not isinstance(transcript, dict):
        raise HTTPException(status_code=422, detail="Transcript artifact must be a JSON object.")
    return transcript, artifact_name


def _load_diarization_artifact(job_id: str) -> tuple[dict[str, Any], str]:
    artifact_name = "diarization.json"
    try:
        path = job_artifact_path(get_settings(), job_id, artifact_name)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Diarization artifact not found.") from exc

    try:
        diarization = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail="Diarization artifact is invalid JSON.",
        ) from exc
    if not isinstance(diarization, dict):
        raise HTTPException(status_code=422, detail="Diarization artifact must be a JSON object.")
    return diarization, artifact_name


def _corrections_path(
    job_id: str,
    artifact_name: str = TRANSCRIPT_CORRECTIONS_ARTIFACT,
) -> Path:
    return job_dir(get_settings(), job_id) / artifact_name


def _corrections_version(
    job_id: str,
    artifact_name: str = TRANSCRIPT_CORRECTIONS_ARTIFACT,
) -> str:
    return _artifact_version(job_id, artifact_name)


def _artifact_version(job_id: str, artifact_name: str) -> str:
    try:
        path = job_artifact_path(get_settings(), job_id, artifact_name)
    except FileNotFoundError:
        return "missing"
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _load_corrections_artifact(
    job_id: str,
    artifact_name: str,
    label: str,
) -> dict[str, Any] | None:
    try:
        path = job_artifact_path(get_settings(), job_id, artifact_name)
    except FileNotFoundError:
        return None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        corrections = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{label} corrections are invalid JSON.",
        ) from exc
    if not isinstance(corrections, dict):
        raise HTTPException(status_code=422, detail=f"{label} corrections must be a JSON object.")
    return corrections


def _load_transcript_corrections(job_id: str) -> dict[str, Any] | None:
    return _load_corrections_artifact(job_id, TRANSCRIPT_CORRECTIONS_ARTIFACT, "Transcript")


def _load_speaker_corrections(job_id: str) -> dict[str, Any] | None:
    return _load_corrections_artifact(job_id, SPEAKER_CORRECTIONS_ARTIFACT, "Speaker")


def _optional_diarization_artifact(job_id: str) -> dict[str, Any] | None:
    return _load_optional_json_artifact(job_id, "diarization.json", "Diarization")


def _load_json_artifact(job_id: str, artifact_name: str, label: str) -> dict[str, Any]:
    try:
        path = job_artifact_path(get_settings(), job_id, artifact_name)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=f"{label} artifact not found.") from exc

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{label} artifact is invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail=f"{label} artifact must be a JSON object.")
    return payload


def _load_optional_json_artifact(
    job_id: str, artifact_name: str, label: str
) -> dict[str, Any] | None:
    try:
        path = job_artifact_path(get_settings(), job_id, artifact_name)
    except FileNotFoundError:
        return None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{label} artifact is invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail=f"{label} artifact must be a JSON object.")
    return payload


def _load_cut_suggestions(job_id: str) -> dict[str, Any]:
    return _load_json_artifact(job_id, CUT_SUGGESTIONS_ARTIFACT, "Cut suggestions")


def _load_cut_review(job_id: str) -> dict[str, Any] | None:
    return _load_optional_json_artifact(job_id, CUT_REVIEW_ARTIFACT, "Cut review")


def _load_cut_apply_manifest(job_id: str) -> dict[str, Any] | None:
    return _load_optional_json_artifact(job_id, CUT_APPLY_MANIFEST_ARTIFACT, "Cut apply")


def _cut_apply_output_url(job_id: str, manifest: dict[str, Any] | None) -> str | None:
    if manifest is None or manifest.get("status") != "succeeded":
        return None
    if manifest.get("suggestions_version") != _artifact_version(job_id, CUT_SUGGESTIONS_ARTIFACT):
        return None
    if manifest.get("review_version") != _artifact_version(job_id, CUT_REVIEW_ARTIFACT):
        return None
    output_artifact = manifest.get("output_artifact")
    if not isinstance(output_artifact, str):
        return None
    try:
        job_artifact_path(get_settings(), job_id, output_artifact)
    except (ValueError, FileNotFoundError):
        return None
    return _artifact_url(job_id, output_artifact)


def _job_target_format(job_id: str) -> TargetFormat:
    manifest = _load_optional_json_artifact(job_id, "pipeline_manifest.json", "Pipeline manifest")
    if manifest is not None:
        target = manifest.get("target")
        if isinstance(target, dict):
            fields = {
                key: target[key]
                for key in ("sample_rate", "channels", "codec", "container", "bitrate")
                if key in target
            }
            try:
                return TargetFormat(**fields)
            except (TypeError, ValueError):
                pass
    record = fetch_job(job_id)
    job_config = record.to_dict().get("config", {}) if record is not None else {}
    preset = job_config.get("preset") if isinstance(job_config, dict) else None
    return load_pipeline_config_for_preset(
        preset,
        default_path=get_settings().pipeline_config,
        preset_dir=get_settings().preset_dir,
    ).target


def _correction_text_by_index(corrections: dict[str, Any] | None) -> dict[int, str]:
    if corrections is None:
        return {}

    corrected: dict[int, str] = {}
    for segment in corrections.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        try:
            segment_index = int(segment["segment_index"])
        except (KeyError, TypeError, ValueError):
            continue
        text = segment.get("text")
        if isinstance(text, str):
            corrected[segment_index] = text
    return corrected


def _transcript_segments(
    transcript: dict[str, Any],
    corrections: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rendered_segments: list[dict[str, Any]] = []
    corrected_text = _correction_text_by_index(corrections)
    for index, segment in enumerate(transcript.get("segments") or []):
        if not isinstance(segment, dict):
            continue
        original_text = segment.get("text") or ""
        text = corrected_text.get(index, original_text)
        words = []
        for word in segment.get("words") or []:
            if not isinstance(word, dict):
                continue
            words.append(
                {
                    "text": word.get("word") or "",
                    "start": _format_seconds(word.get("start")),
                    "end": _format_seconds(word.get("end")),
                    "probability": _format_probability(word.get("probability")),
                }
            )
        rendered_segments.append(
            {
                "index": index,
                "id": segment.get("id"),
                "start": _format_seconds(segment.get("start")),
                "end": _format_seconds(segment.get("end")),
                "start_seconds": segment.get("start"),
                "end_seconds": segment.get("end"),
                "text": text,
                "original_text": original_text,
                "is_corrected": text != original_text,
                "words": words,
            }
        )
    return rendered_segments


def _numeric_seconds(value: Any) -> float | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds):
        return None
    return seconds


def _annotate_transcript_speakers(
    segments: list[dict[str, Any]],
    diarization: dict[str, Any] | None,
    corrections: dict[str, Any] | None,
) -> int | None:
    if diarization is None:
        return None

    labels = _speaker_label_map(corrections)
    turns: list[dict[str, Any]] = []
    for turn in diarization.get("segments") or []:
        if not isinstance(turn, dict):
            continue
        start = _numeric_seconds(turn.get("start"))
        end = _numeric_seconds(turn.get("end"))
        speaker = str(turn.get("speaker") or "")
        if start is None or end is None or end <= start or not speaker:
            continue
        turns.append({"start": start, "end": end, "speaker": speaker})

    if not turns:
        return 0

    speaker_count = len({turn["speaker"] for turn in turns})
    for segment in segments:
        start = _numeric_seconds(segment.get("start_seconds"))
        end = _numeric_seconds(segment.get("end_seconds"))
        if start is None or end is None or end <= start:
            continue

        overlaps: dict[str, float] = {}
        for turn in turns:
            overlap = min(end, turn["end"]) - max(start, turn["start"])
            if overlap <= 0:
                continue
            speaker = turn["speaker"]
            overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap

        if not overlaps:
            continue
        speaker = max(overlaps, key=overlaps.get)
        segment["speaker"] = {
            "speaker": speaker,
            "label": labels.get(speaker, speaker),
            "overlap_seconds": overlaps[speaker],
        }
    return speaker_count


def _build_transcript_corrections(
    transcript: dict[str, Any],
    artifact_name: str,
    form: dict[str, Any],
) -> dict[str, Any]:
    corrections: list[dict[str, Any]] = []
    for index, segment in enumerate(transcript.get("segments") or []):
        if not isinstance(segment, dict):
            continue
        original_text = str(segment.get("text") or "")
        corrected_text = str(form.get(f"segment_{index}_text", original_text))
        if corrected_text == original_text:
            continue
        corrections.append(
            {
                "segment_index": index,
                "segment_id": segment.get("id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "original_text": original_text,
                "text": corrected_text,
            }
        )

    return {
        "schema_version": 1,
        "source_artifact": artifact_name,
        "segments": corrections,
    }


def _corrected_transcript(
    transcript: dict[str, Any],
    corrections: dict[str, Any] | None,
) -> dict[str, Any]:
    corrected = json.loads(json.dumps(transcript))
    segments = corrected.get("segments")
    if not isinstance(segments, list):
        return corrected

    for segment_index, text in _correction_text_by_index(corrections).items():
        if segment_index < 0 or segment_index >= len(segments):
            continue
        segment = segments[segment_index]
        if isinstance(segment, dict):
            segment["text"] = text
    return corrected


def _transcript_vtt(transcript: dict[str, Any]) -> str:
    lines = ["WEBVTT", ""]
    for index, segment in enumerate(transcript.get("segments") or [], start=1):
        if not isinstance(segment, dict):
            continue
        try:
            timing = (
                f"{_vtt_timestamp(segment.get('start'))} --> {_vtt_timestamp(segment.get('end'))}"
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        lines.extend([str(index), timing, _vtt_cue_text(segment.get("text")), ""])
    return "\n".join(lines)


def _vtt_cue_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _speaker_transcript_text(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for segment in segments:
        text = " ".join(str(segment.get("text") or "").split())
        if not text:
            continue
        speaker = segment.get("speaker")
        if isinstance(speaker, dict):
            label = str(speaker.get("label") or speaker.get("speaker") or "Unknown")
        else:
            label = "Unknown"
        lines.append(f"[{segment['start']} - {segment['end']}] {label}: {text}")
    return "\n".join(lines) + ("\n" if lines else "")


def _speaker_label_map(corrections: dict[str, Any] | None) -> dict[str, str]:
    if corrections is None:
        return {}

    labels: dict[str, str] = {}
    for speaker in corrections.get("speakers") or []:
        if not isinstance(speaker, dict):
            continue
        speaker_id = speaker.get("speaker")
        label = speaker.get("label")
        if isinstance(speaker_id, str) and isinstance(label, str):
            labels[speaker_id] = label
    return labels


def _diarization_speaker_rows(
    diarization: dict[str, Any],
    corrections: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    labels = _speaker_label_map(corrections)
    speakers: dict[str, dict[str, Any]] = {}
    for segment in diarization.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        speaker_id = str(segment.get("speaker") or "")
        if not speaker_id:
            continue
        row = speakers.setdefault(
            speaker_id,
            {
                "speaker": speaker_id,
                "label": labels.get(speaker_id, speaker_id),
                "turn_count": 0,
            },
        )
        row["turn_count"] += 1
    return sorted(speakers.values(), key=lambda row: row["speaker"])


def _diarization_turn_rows(
    diarization: dict[str, Any],
    corrections: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    labels = _speaker_label_map(corrections)
    turns: list[dict[str, Any]] = []
    for index, segment in enumerate(diarization.get("segments") or []):
        if not isinstance(segment, dict):
            continue
        speaker_id = str(segment.get("speaker") or "")
        turns.append(
            {
                "index": index,
                "start": _format_seconds(segment.get("start")),
                "end": _format_seconds(segment.get("end")),
                "speaker": speaker_id,
                "label": labels.get(speaker_id, speaker_id),
                "track": segment.get("track"),
            }
        )
    return turns


def _build_speaker_corrections(
    diarization: dict[str, Any],
    artifact_name: str,
    form: dict[str, str],
) -> dict[str, Any]:
    known_speakers = {row["speaker"] for row in _diarization_speaker_rows(diarization)}
    speakers: list[dict[str, str]] = []
    index = 0
    while f"speaker_{index}_id" in form:
        speaker_id = form[f"speaker_{index}_id"]
        label = form.get(f"speaker_{index}_label", "").strip()
        if speaker_id in known_speakers and label and label != speaker_id:
            speakers.append({"speaker": speaker_id, "label": label})
        index += 1

    return {
        "schema_version": 1,
        "source_artifact": artifact_name,
        "speakers": speakers,
    }


def _review_decision_map(review: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    if review is None:
        return {}

    decisions: dict[str, dict[str, str]] = {}
    for decision in review.get("decisions") or []:
        if not isinstance(decision, dict):
            continue
        suggestion_id = decision.get("suggestion_id")
        state = decision.get("decision")
        if not isinstance(suggestion_id, str) or state not in {"approved", "rejected"}:
            continue
        note = decision.get("note")
        decisions[suggestion_id] = {
            "decision": state,
            "note": note if isinstance(note, str) else "",
        }
    return decisions


def _cut_suggestion_rows(
    suggestions: dict[str, Any],
    review: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    decisions = _review_decision_map(review)
    rows: list[dict[str, Any]] = []
    for index, suggestion in enumerate(suggestions.get("suggestions") or []):
        if not isinstance(suggestion, dict):
            continue
        suggestion_id = suggestion.get("id")
        if not isinstance(suggestion_id, str) or not suggestion_id:
            continue
        decision = decisions.get(suggestion_id, {"decision": "pending", "note": ""})
        rows.append(
            {
                "index": index,
                "id": suggestion_id,
                "type": suggestion.get("type") or "-",
                "source": suggestion.get("source") or "-",
                "start": _format_seconds(suggestion.get("start")),
                "end": _format_seconds(suggestion.get("end")),
                "duration": _format_seconds(suggestion.get("duration")),
                "start_seconds": suggestion.get("start"),
                "end_seconds": suggestion.get("end"),
                "duration_seconds": suggestion.get("duration"),
                "text": suggestion.get("text") or "",
                "reason": suggestion.get("reason") or "",
                "decision": decision["decision"],
                "note": decision["note"],
            }
        )
    return rows


def _build_cut_review(
    suggestions: dict[str, Any],
    artifact_name: str,
    form: dict[str, str],
) -> dict[str, Any]:
    known_suggestions = {row["id"]: row for row in _cut_suggestion_rows(suggestions)}
    decisions: list[dict[str, Any]] = []
    index = 0
    while f"suggestion_{index}_id" in form:
        suggestion_id = form[f"suggestion_{index}_id"]
        suggestion = known_suggestions.get(suggestion_id)
        decision = form.get(f"suggestion_{index}_decision")
        if suggestion is None or decision not in {"approved", "rejected"}:
            index += 1
            continue
        note = form.get(f"suggestion_{index}_note", "").strip()
        item: dict[str, Any] = {
            "suggestion_id": suggestion_id,
            "decision": decision,
            "type": suggestion["type"],
            "start": suggestion["start_seconds"],
            "end": suggestion["end_seconds"],
            "duration": suggestion["duration_seconds"],
        }
        if note:
            item["note"] = note
        decisions.append(item)
        index += 1

    return {
        "schema_version": 1,
        "source_artifact": artifact_name,
        "decisions": decisions,
    }


def _cut_review_form_limits(suggestions: dict[str, Any]) -> tuple[int, int]:
    suggestion_count = len(_cut_suggestion_rows(suggestions))
    expected_fields = (
        CUT_REVIEW_FORM_STATIC_FIELDS + suggestion_count * CUT_REVIEW_FORM_FIELDS_PER_SUGGESTION
    )
    expected_bytes = (
        MAX_CORRECTION_FORM_BYTES + suggestion_count * CUT_REVIEW_FORM_BYTES_PER_SUGGESTION
    )
    return (
        min(MAX_CUT_REVIEW_FORM_BYTES, max(MAX_CORRECTION_FORM_BYTES, expected_bytes)),
        min(MAX_CUT_REVIEW_FORM_FIELDS, max(MAX_CORRECTION_FIELDS, expected_fields)),
    )


async def _read_limited_urlencoded_form(
    request: Request,
    label: str,
    *,
    max_bytes: int = MAX_CORRECTION_FORM_BYTES,
    max_num_fields: int = MAX_CORRECTION_FIELDS,
) -> dict[str, str]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type != "application/x-www-form-urlencoded":
        raise HTTPException(
            status_code=415,
            detail=f"{label} must be submitted as a URL-encoded form.",
        )

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length.") from exc
        if declared_size > max_bytes:
            raise HTTPException(status_code=413, detail=f"{label} form is too large.")

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_bytes:
            raise HTTPException(status_code=413, detail=f"{label} form is too large.")
        body.extend(chunk)

    try:
        decoded = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{label} form is not UTF-8.") from exc

    try:
        parsed = parse_qs(
            decoded,
            keep_blank_values=True,
            max_num_fields=max_num_fields,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{label} form is invalid.") from exc
    return {key: values[-1] if values else "" for key, values in parsed.items()}


async def _read_limited_correction_form(
    request: Request,
    label: str = "Transcript",
) -> dict[str, str]:
    return await _read_limited_urlencoded_form(request, f"{label} corrections")


def _ensure_fresh_artifact(
    job_id: str,
    submitted_version: str | None,
    artifact_name: str,
    label: str,
) -> None:
    if submitted_version != _artifact_version(job_id, artifact_name):
        raise HTTPException(
            status_code=409,
            detail=f"{label} changed. Reload the page and try again.",
        )


def _ensure_fresh_corrections(
    job_id: str,
    submitted_version: str | None,
    artifact_name: str = TRANSCRIPT_CORRECTIONS_ARTIFACT,
    label: str = "Transcript",
) -> None:
    _ensure_fresh_artifact(
        job_id,
        submitted_version,
        artifact_name,
        f"{label} corrections",
    )


def _artifact_preview(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in TEXT_ARTIFACT_EXTENSIONS:
        return {"available": False, "reason": "Preview unavailable for binary artifacts."}

    with path.open("rb") as handle:
        data = handle.read(MAX_ARTIFACT_PREVIEW_BYTES + 1)
    truncated = len(data) > MAX_ARTIFACT_PREVIEW_BYTES
    if truncated:
        data = data[:MAX_ARTIFACT_PREVIEW_BYTES]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"available": False, "reason": "Preview unavailable for non-UTF-8 artifacts."}

    if path.suffix.lower() == ".json" and not truncated:
        try:
            text = json.dumps(json.loads(text), indent=2)
        except json.JSONDecodeError:
            pass
    return {"available": True, "content": text, "truncated": truncated}


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    _ensure_ready()
    settings = get_settings()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "jobs": recent_jobs(50),
            "settings": settings,
            "presets": _preset_options(settings),
        },
    )


def _preset_options(settings) -> list[dict[str, str | bool]]:
    options: list[dict[str, str | bool]] = []
    for preset in available_presets(settings.pipeline_config, settings.preset_dir):
        try:
            config = PipelineConfig.from_path(preset.path)
            issues = pipeline_preflight_issues(config, settings)
        except Exception as exc:
            available = False
            readiness = f"Preset cannot be loaded: {exc}"
        else:
            available = not issues
            readiness = "Ready" if available else format_preflight_issues(issues)
        options.append(
            {
                "id": preset.id,
                "label": preset.label,
                "description": preset.description,
                "available": available,
                "readiness": readiness,
                "title": f"{preset.description} {readiness}",
            }
        )
    return options


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    artifacts = list_job_artifacts(get_settings(), job_id)
    artifact_names = {artifact.name for artifact in artifacts}
    return templates.TemplateResponse(
        request,
        "job.html",
        {
            "request": request,
            "job": record,
            "artifacts": [_artifact_payload(job_id, artifact) for artifact in artifacts],
            "artifact_bundle_url": f"/api/jobs/{job_id}/artifacts.zip" if artifacts else None,
            "speaker_url": f"/jobs/{job_id}/speakers"
            if "diarization.json" in artifact_names
            else None,
            "cut_review_url": f"/jobs/{job_id}/cuts"
            if CUT_SUGGESTIONS_ARTIFACT in artifact_names
            else None,
            "common_artifacts": [
                {
                    "label": label,
                    "name": artifact_name,
                    "url": _artifact_page_url(job_id, artifact_name),
                    "download_url": f"/api/jobs/{job_id}{suffix}",
                }
                for label, artifact_name, suffix in COMMON_ARTIFACTS
                if artifact_name in artifact_names
            ],
        },
    )


@router.get("/jobs/{job_id}/transcript", response_class=HTMLResponse)
def transcript_page(request: Request, job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    transcript, artifact_name = _load_transcript_artifact(job_id, record.transcript_path)
    corrections = _load_transcript_corrections(job_id)
    segments = _transcript_segments(transcript, corrections)
    diarization = _optional_diarization_artifact(job_id)
    speaker_corrections = _load_speaker_corrections(job_id) if diarization is not None else None
    speaker_count = _annotate_transcript_speakers(segments, diarization, speaker_corrections)
    word_count = sum(len(segment["words"]) for segment in segments)
    vtt_name = "transcript.vtt"
    artifacts = {artifact.name for artifact in list_job_artifacts(get_settings(), job_id)}
    speaker_url = f"/jobs/{job_id}/speakers" if diarization is not None else None
    return templates.TemplateResponse(
        request,
        "transcript.html",
        {
            "request": request,
            "job": record,
            "transcript": transcript,
            "segments": segments,
            "word_count": word_count,
            "artifact_name": artifact_name,
            "download_url": _artifact_url(job_id, artifact_name),
            "vtt_url": _artifact_url(job_id, vtt_name) if vtt_name in artifacts else None,
            "speaker_url": speaker_url,
            "speaker_transcript_url": f"/api/jobs/{job_id}/transcript/speakers.txt"
            if diarization is not None
            else None,
            "edit_url": f"/jobs/{job_id}/transcript/edit",
            "corrected_json_url": f"/api/jobs/{job_id}/transcript/corrected.json"
            if corrections is not None
            else None,
            "corrected_vtt_url": f"/api/jobs/{job_id}/transcript/corrected.vtt"
            if corrections is not None
            else None,
            "corrections_url": _artifact_url(job_id, TRANSCRIPT_CORRECTIONS_ARTIFACT)
            if corrections is not None
            else None,
            "correction_count": sum(1 for segment in segments if segment["is_corrected"]),
            "speaker_count": speaker_count,
            "language_probability": _format_probability(transcript.get("language_probability")),
            "duration": _format_seconds(transcript.get("duration")),
        },
    )


@router.get("/jobs/{job_id}/transcript/edit", response_class=HTMLResponse)
def transcript_edit_page(request: Request, job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    transcript, artifact_name = _load_transcript_artifact(job_id, record.transcript_path)
    corrections = _load_transcript_corrections(job_id)
    segments = _transcript_segments(transcript, corrections)
    return templates.TemplateResponse(
        request,
        "transcript_edit.html",
        {
            "request": request,
            "job": record,
            "segments": segments,
            "artifact_name": artifact_name,
            "corrections_version": _corrections_version(job_id),
            "save_url": f"/jobs/{job_id}/transcript/corrections",
        },
    )


@router.post("/jobs/{job_id}/transcript/corrections")
async def save_transcript_corrections(request: Request, job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    transcript, artifact_name = _load_transcript_artifact(job_id, record.transcript_path)
    form = await _read_limited_correction_form(request)
    _ensure_fresh_corrections(job_id, form.get("corrections_version"))
    corrections = _build_transcript_corrections(transcript, artifact_name, form)
    serialized = json.dumps(corrections, indent=2)
    if len(serialized.encode("utf-8")) > MAX_CORRECTION_FORM_BYTES:
        raise HTTPException(status_code=413, detail="Transcript corrections artifact is too large.")
    _corrections_path(job_id).write_text(serialized, encoding="utf-8")
    return RedirectResponse(f"/jobs/{job_id}/transcript", status_code=303)


@router.get("/jobs/{job_id}/speakers", response_class=HTMLResponse)
def speakers_page(request: Request, job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    diarization, artifact_name = _load_diarization_artifact(job_id)
    corrections = _load_speaker_corrections(job_id)
    speakers = _diarization_speaker_rows(diarization, corrections)
    turns = _diarization_turn_rows(diarization, corrections)
    rttm_name = "diarization.rttm"
    artifacts = {artifact.name for artifact in list_job_artifacts(get_settings(), job_id)}
    return templates.TemplateResponse(
        request,
        "speakers.html",
        {
            "request": request,
            "job": record,
            "diarization": diarization,
            "speakers": speakers,
            "turns": turns,
            "artifact_name": artifact_name,
            "download_url": _artifact_url(job_id, artifact_name),
            "rttm_url": _artifact_url(job_id, rttm_name) if rttm_name in artifacts else None,
            "edit_url": f"/jobs/{job_id}/speakers/edit",
            "corrections_url": _artifact_url(job_id, SPEAKER_CORRECTIONS_ARTIFACT)
            if corrections is not None
            else None,
            "correction_count": sum(
                1 for speaker in speakers if speaker["label"] != speaker["speaker"]
            ),
        },
    )


@router.get("/jobs/{job_id}/speakers/edit", response_class=HTMLResponse)
def speakers_edit_page(request: Request, job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    diarization, artifact_name = _load_diarization_artifact(job_id)
    corrections = _load_speaker_corrections(job_id)
    speakers = _diarization_speaker_rows(diarization, corrections)
    return templates.TemplateResponse(
        request,
        "speakers_edit.html",
        {
            "request": request,
            "job": record,
            "speakers": speakers,
            "artifact_name": artifact_name,
            "corrections_version": _corrections_version(job_id, SPEAKER_CORRECTIONS_ARTIFACT),
            "save_url": f"/jobs/{job_id}/speakers/corrections",
        },
    )


@router.post("/jobs/{job_id}/speakers/corrections")
async def save_speaker_corrections(request: Request, job_id: str):
    _ensure_ready()
    _require_job(job_id)
    diarization, artifact_name = _load_diarization_artifact(job_id)
    form = await _read_limited_correction_form(request, "Speaker")
    _ensure_fresh_corrections(
        job_id,
        form.get("corrections_version"),
        SPEAKER_CORRECTIONS_ARTIFACT,
        "Speaker",
    )
    corrections = _build_speaker_corrections(diarization, artifact_name, form)
    serialized = json.dumps(corrections, indent=2)
    if len(serialized.encode("utf-8")) > MAX_CORRECTION_FORM_BYTES:
        raise HTTPException(status_code=413, detail="Speaker corrections artifact is too large.")
    _corrections_path(job_id, SPEAKER_CORRECTIONS_ARTIFACT).write_text(
        serialized,
        encoding="utf-8",
    )
    return RedirectResponse(f"/jobs/{job_id}/speakers", status_code=303)


@router.get("/jobs/{job_id}/cuts", response_class=HTMLResponse)
def cut_review_page(request: Request, job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    suggestions = _load_cut_suggestions(job_id)
    review = _load_cut_review(job_id)
    apply_manifest = _load_cut_apply_manifest(job_id)
    rows = _cut_suggestion_rows(suggestions, review)
    approved_count = sum(1 for row in rows if row["decision"] == "approved")
    rejected_count = sum(1 for row in rows if row["decision"] == "rejected")
    return templates.TemplateResponse(
        request,
        "cuts.html",
        {
            "request": request,
            "job": record,
            "suggestions": suggestions,
            "rows": rows,
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "pending_count": len(rows) - approved_count - rejected_count,
            "suggestions_url": _artifact_url(job_id, CUT_SUGGESTIONS_ARTIFACT),
            "review_url": _artifact_url(job_id, CUT_REVIEW_ARTIFACT)
            if review is not None
            else None,
            "apply_manifest": apply_manifest,
            "apply_manifest_url": _artifact_url(job_id, CUT_APPLY_MANIFEST_ARTIFACT)
            if apply_manifest is not None
            else None,
            "reviewed_output_url": _cut_apply_output_url(job_id, apply_manifest),
            "review_version": _artifact_version(job_id, CUT_REVIEW_ARTIFACT),
            "suggestions_version": _artifact_version(job_id, CUT_SUGGESTIONS_ARTIFACT),
            "save_url": f"/jobs/{job_id}/cuts/review",
            "apply_url": f"/jobs/{job_id}/cuts/apply",
        },
    )


@router.post("/jobs/{job_id}/cuts/review")
async def save_cut_review(request: Request, job_id: str):
    _ensure_ready()
    _require_job(job_id)
    suggestions = _load_cut_suggestions(job_id)
    max_bytes, max_num_fields = _cut_review_form_limits(suggestions)
    form = await _read_limited_urlencoded_form(
        request,
        "Cut review",
        max_bytes=max_bytes,
        max_num_fields=max_num_fields,
    )
    _ensure_fresh_artifact(
        job_id,
        form.get("suggestions_version"),
        CUT_SUGGESTIONS_ARTIFACT,
        "Cut suggestions",
    )
    _ensure_fresh_artifact(
        job_id,
        form.get("review_version"),
        CUT_REVIEW_ARTIFACT,
        "Cut review",
    )
    review = _build_cut_review(suggestions, CUT_SUGGESTIONS_ARTIFACT, form)
    serialized = json.dumps(review, indent=2)
    if len(serialized.encode("utf-8")) > MAX_CUT_REVIEW_ARTIFACT_BYTES:
        raise HTTPException(status_code=413, detail="Cut review artifact is too large.")
    _corrections_path(job_id, CUT_REVIEW_ARTIFACT).write_text(serialized, encoding="utf-8")
    return RedirectResponse(f"/jobs/{job_id}/cuts", status_code=303)


@router.post("/jobs/{job_id}/cuts/apply")
async def apply_cut_review(
    request: Request,
    background_tasks: BackgroundTasks,
    job_id: str,
):
    _ensure_ready()
    record = _require_job(job_id)
    if record.status != "succeeded" or not record.output_path:
        raise HTTPException(status_code=409, detail="Job output is not ready.")
    input_path = Path(record.output_path)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Output file missing.")

    suggestions = _load_cut_suggestions(job_id)
    review = _load_cut_review(job_id)
    if review is None:
        raise HTTPException(status_code=409, detail="Cut review has not been saved.")

    form = await _read_limited_urlencoded_form(
        request,
        "Cut apply",
        max_bytes=MAX_CORRECTION_FORM_BYTES,
        max_num_fields=4,
    )
    suggestions_version = form.get("suggestions_version")
    review_version = form.get("review_version")
    _ensure_fresh_artifact(
        job_id,
        suggestions_version,
        CUT_SUGGESTIONS_ARTIFACT,
        "Cut suggestions",
    )
    _ensure_fresh_artifact(
        job_id,
        review_version,
        CUT_REVIEW_ARTIFACT,
        "Cut review",
    )
    try:
        approved_cuts = approved_cuts_from_review(suggestions, review)
    except CutApplyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not approved_cuts:
        raise HTTPException(status_code=409, detail="No approved cuts to apply.")

    target = _job_target_format(job_id)
    background_tasks.add_task(
        apply_approved_cuts,
        job_id=job_id,
        input_path=input_path,
        cuts=approved_cuts,
        target=target,
        suggestions_version=suggestions_version or "",
        review_version=review_version or "",
        source_suggestions_artifact=CUT_SUGGESTIONS_ARTIFACT,
        source_review_artifact=CUT_REVIEW_ARTIFACT,
    )
    return RedirectResponse(f"/jobs/{job_id}/cuts", status_code=303)


@router.get("/jobs/{job_id}/artifacts/{artifact_name:path}", response_class=HTMLResponse)
def artifact_page(request: Request, job_id: str, artifact_name: str):
    _ensure_ready()
    record = _require_job(job_id)
    try:
        path = job_artifact_path(get_settings(), job_id, artifact_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found.") from exc

    return templates.TemplateResponse(
        request,
        "artifact.html",
        {
            "request": request,
            "job": record,
            "artifact": {
                "name": artifact_name,
                "size": _format_bytes(path.stat().st_size),
                "download_url": _artifact_url(job_id, artifact_name),
            },
            "preview": _artifact_preview(path),
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


@router.post("/api/jobs/{job_id}/retry")
def retry_job_api(background_tasks: BackgroundTasks, job_id: str) -> dict[str, str]:
    _ensure_ready()
    record = fetch_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if record.status != "failed":
        raise HTTPException(
            status_code=409,
            detail=f"Only failed jobs can be retried. Current status: {record.status}.",
        )
    job_config = record.to_dict().get("config", {})
    preset = job_config.get("preset") if isinstance(job_config, dict) else None
    _preflight_preset_or_400(preset)
    try:
        retry_failed_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    except JobRetryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(run_job, job_id)
    return {"id": job_id, "status_url": f"/api/jobs/{job_id}"}


@router.post("/jobs/{job_id}/retry")
def retry_job_form(background_tasks: BackgroundTasks, job_id: str):
    payload = retry_job_api(background_tasks, job_id)
    return RedirectResponse(f"/jobs/{payload['id']}", status_code=303)


def _preflight_preset_or_400(preset: str | None) -> None:
    settings = get_settings()
    label = "Podcast default"
    try:
        if preset:
            preset_info = preset_by_id(
                preset,
                default_path=settings.pipeline_config,
                preset_dir=settings.preset_dir,
            )
            label = preset_info.label
            pipeline_config = PipelineConfig.from_path(preset_info.path)
        else:
            pipeline_config = PipelineConfig.from_path(settings.pipeline_config)
        preflight_issues = pipeline_preflight_issues(pipeline_config, settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid pipeline preset: {exc}") from exc

    if preflight_issues:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Preset '{label}' cannot run on this host. "
                f"{format_preflight_issues(preflight_issues)}"
            ),
        )


async def _create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    preset: str,
) -> dict[str, str]:
    _ensure_ready()
    settings = get_settings()
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    try:
        preset_info = preset_by_id(
            preset,
            default_path=settings.pipeline_config,
            preset_dir=settings.preset_dir,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _preflight_preset_or_400(preset_info.id)

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
        config={"preset": preset_info.id, "preset_label": preset_info.label},
    )
    background_tasks.add_task(run_job, job_id)
    return {"id": job_id, "status_url": f"/api/jobs/{job_id}"}


@router.get("/api/jobs/{job_id}")
def get_job_api(job_id: str) -> dict:
    _ensure_ready()
    record = _require_job(job_id)
    return record.to_dict()


@router.get("/api/jobs/{job_id}/download")
def download_job(job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    if record.status != "succeeded" or not record.output_path:
        raise HTTPException(status_code=409, detail="Job output is not ready.")
    path = Path(record.output_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file missing.")
    return FileResponse(path, filename=path.name)


@router.get("/api/jobs/{job_id}/transcript")
def download_transcript(job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    if not record.transcript_path:
        raise HTTPException(status_code=404, detail="Transcript not available.")
    path = Path(record.transcript_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript file missing.")
    return FileResponse(path, filename=path.name)


@router.get("/api/jobs/{job_id}/transcript/corrected.json")
def download_corrected_transcript_json(job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    transcript, _artifact_name = _load_transcript_artifact(job_id, record.transcript_path)
    corrected = _corrected_transcript(transcript, _load_transcript_corrections(job_id))
    return Response(
        json.dumps(corrected, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="transcript_corrected.json"'},
    )


@router.get("/api/jobs/{job_id}/transcript/corrected.vtt")
def download_corrected_transcript_vtt(job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    transcript, _artifact_name = _load_transcript_artifact(job_id, record.transcript_path)
    corrected = _corrected_transcript(transcript, _load_transcript_corrections(job_id))
    return Response(
        _transcript_vtt(corrected),
        media_type="text/vtt",
        headers={"Content-Disposition": 'attachment; filename="transcript_corrected.vtt"'},
    )


@router.get("/api/jobs/{job_id}/transcript/speakers.txt")
def download_speaker_transcript(job_id: str):
    _ensure_ready()
    record = _require_job(job_id)
    transcript, _artifact_name = _load_transcript_artifact(job_id, record.transcript_path)
    transcript_corrections = _load_transcript_corrections(job_id)
    diarization, _diarization_name = _load_diarization_artifact(job_id)
    speaker_corrections = _load_speaker_corrections(job_id)
    segments = _transcript_segments(transcript, transcript_corrections)
    _annotate_transcript_speakers(segments, diarization, speaker_corrections)
    return Response(
        _speaker_transcript_text(segments),
        media_type="text/plain",
        headers={"Content-Disposition": 'attachment; filename="speaker_transcript.txt"'},
    )


@router.get("/api/jobs/{job_id}/artifacts")
def list_artifacts_api(job_id: str) -> list[dict]:
    _ensure_ready()
    _require_job(job_id)
    try:
        artifacts = list_job_artifacts(get_settings(), job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [_artifact_payload(job_id, artifact) for artifact in artifacts]


@router.get("/api/jobs/{job_id}/artifacts.zip")
def download_artifact_bundle(job_id: str):
    return _artifact_bundle_response(job_id)


@router.get("/api/jobs/{job_id}/artifacts/{artifact_name:path}")
def download_artifact(job_id: str, artifact_name: str):
    return _artifact_response(job_id, artifact_name)


@router.get("/api/jobs/{job_id}/events")
def download_job_events(job_id: str):
    return _artifact_response(job_id, "job-events.jsonl", media_type="application/x-ndjson")


@router.get("/api/jobs/{job_id}/commands")
def download_command_log(job_id: str):
    return _artifact_response(job_id, "commands.jsonl", media_type="application/x-ndjson")


@router.get("/api/jobs/{job_id}/metadata")
def download_media_metadata(job_id: str):
    return _artifact_response(job_id, "00_media_metadata.json", media_type="application/json")


@router.get("/api/jobs/{job_id}/manifest")
def download_pipeline_manifest(job_id: str):
    return _artifact_response(job_id, "pipeline_manifest.json", media_type="application/json")
