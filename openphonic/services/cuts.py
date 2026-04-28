from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openphonic.core.logging import append_event, log_event, utc_timestamp
from openphonic.core.settings import get_settings
from openphonic.pipeline.config import TargetFormat
from openphonic.pipeline.ffmpeg import build_apply_cuts_command, run_command
from openphonic.pipeline.stages import require_artifact
from openphonic.services.storage import job_dir, new_job_id

logger = logging.getLogger(__name__)

CUT_APPLY_MANIFEST_ARTIFACT = "cut_apply_manifest.json"
CUT_REVIEWED_OUTPUT_PREFIX = "cut_reviewed"


class CutApplyError(RuntimeError):
    """Raised when reviewed cut decisions cannot be applied."""


@dataclass(frozen=True)
class ApprovedCut:
    suggestion_id: str
    cut_type: str
    start: float
    end: float
    duration: float | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def approved_cuts_from_review(
    suggestions: dict[str, Any],
    review: dict[str, Any] | None,
) -> list[ApprovedCut]:
    if review is None:
        raise CutApplyError("Cut review has not been saved.")

    approved_ids = {
        decision.get("suggestion_id")
        for decision in review.get("decisions") or []
        if isinstance(decision, dict) and decision.get("decision") == "approved"
    }
    cuts: list[ApprovedCut] = []
    for suggestion in suggestions.get("suggestions") or []:
        if not isinstance(suggestion, dict):
            continue
        suggestion_id = suggestion.get("id")
        if not isinstance(suggestion_id, str) or suggestion_id not in approved_ids:
            continue
        cuts.append(_approved_cut_from_suggestion(suggestion_id, suggestion))
    return cuts


def merged_cut_ranges(cuts: list[ApprovedCut]) -> list[tuple[float, float]]:
    ranges = sorted((cut.start, cut.end) for cut in cuts)
    merged: list[tuple[float, float]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def apply_approved_cuts(
    *,
    job_id: str,
    input_path: Path,
    cuts: list[ApprovedCut],
    target: TargetFormat,
    suggestions_version: str,
    review_version: str,
    source_suggestions_artifact: str,
    source_review_artifact: str,
) -> Path:
    if not cuts:
        raise CutApplyError("At least one approved cut is required.")

    settings = get_settings()
    work_dir = job_dir(settings, job_id)
    command_log_path = work_dir / "commands.jsonl"
    job_events_path = work_dir / "job-events.jsonl"
    manifest_path = work_dir / CUT_APPLY_MANIFEST_ARTIFACT
    output_artifact = f"{CUT_REVIEWED_OUTPUT_PREFIX}.{target.container}"
    output_path = work_dir / output_artifact
    temporary_path = work_dir / f".{CUT_REVIEWED_OUTPUT_PREFIX}.{new_job_id()}.{target.container}"
    cut_ranges = merged_cut_ranges(cuts)

    manifest_base = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "job_id": job_id,
        "source_audio": str(input_path),
        "source_suggestions_artifact": source_suggestions_artifact,
        "source_review_artifact": source_review_artifact,
        "suggestions_version": suggestions_version,
        "review_version": review_version,
        "approved_count": len(cuts),
        "cut_count": len(cut_ranges),
        "approved_cuts": [cut.to_manifest() for cut in cuts],
        "cut_ranges": [{"start": start, "end": end} for start, end in cut_ranges],
        "output_artifact": output_artifact,
        "output_path": str(output_path),
    }
    _write_cut_apply_manifest(manifest_path, {**manifest_base, "status": "running"})
    append_event(
        job_events_path,
        "cut_apply.started",
        job_id=job_id,
        approved_count=len(cuts),
        output_path=output_path,
    )
    log_event(logger, "cut_apply.started", job_id=job_id, approved_count=len(cuts))

    try:
        run_command(
            build_apply_cuts_command(
                input_path,
                temporary_path,
                cut_ranges=cut_ranges,
                target=target,
            ),
            log_path=command_log_path,
        )
        require_artifact(temporary_path, "Cut apply")
        temporary_path.replace(output_path)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        _write_cut_apply_manifest(
            manifest_path,
            {
                **manifest_base,
                "status": "failed",
                "failed_at": utc_timestamp(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        append_event(
            job_events_path,
            "cut_apply.failed",
            job_id=job_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        log_event(
            logger,
            "cut_apply.failed",
            level=logging.ERROR,
            job_id=job_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise

    _write_cut_apply_manifest(
        manifest_path,
        {
            **manifest_base,
            "status": "succeeded",
            "completed_at": utc_timestamp(),
            "output_size_bytes": output_path.stat().st_size,
        },
    )
    append_event(
        job_events_path,
        "cut_apply.succeeded",
        job_id=job_id,
        output_path=output_path,
    )
    log_event(logger, "cut_apply.succeeded", job_id=job_id, output_path=output_path)
    return output_path


def _approved_cut_from_suggestion(suggestion_id: str, suggestion: dict[str, Any]) -> ApprovedCut:
    start = _required_timestamp(suggestion.get("start"), suggestion_id, "start")
    end = _required_timestamp(suggestion.get("end"), suggestion_id, "end")
    if end <= start:
        raise CutApplyError(f"Approved suggestion {suggestion_id} has an invalid time range.")

    duration = suggestion.get("duration")
    parsed_duration = None
    if duration is not None:
        parsed_duration = _required_timestamp(duration, suggestion_id, "duration")

    return ApprovedCut(
        suggestion_id=suggestion_id,
        cut_type=str(suggestion.get("type") or "unknown"),
        start=start,
        end=end,
        duration=parsed_duration,
    )


def _required_timestamp(value: Any, suggestion_id: str, field: str) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise CutApplyError(f"Approved suggestion {suggestion_id} has invalid {field}.") from exc
    if not math.isfinite(timestamp) or timestamp < 0:
        raise CutApplyError(f"Approved suggestion {suggestion_id} has invalid {field}.")
    return timestamp


def _write_cut_apply_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
