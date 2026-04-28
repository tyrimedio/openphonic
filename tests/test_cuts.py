import json
from pathlib import Path

import pytest

from openphonic.core.settings import get_settings
from openphonic.pipeline.config import TargetFormat
from openphonic.services.cuts import (
    CUT_APPLY_MANIFEST_ARTIFACT,
    ApprovedCut,
    CutApplyError,
    apply_approved_cuts,
    approved_cuts_from_review,
    merged_cut_ranges,
)


def configure_tmp_settings(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("OPENPHONIC_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENPHONIC_DATABASE_PATH", str(data_dir / "openphonic.sqlite3"))
    monkeypatch.setenv("OPENPHONIC_PIPELINE_CONFIG", "config/default.yml")
    get_settings.cache_clear()


def test_approved_cuts_from_review_uses_suggestions_as_source() -> None:
    suggestions = {
        "suggestions": [
            {"id": "cut-1", "type": "filler_word", "start": 0.2, "end": 0.4, "duration": 0.2},
            {"id": "cut-2", "type": "silence", "start": 1.0, "end": 2.0, "duration": 1.0},
        ]
    }
    review = {
        "decisions": [
            {"suggestion_id": "cut-1", "decision": "approved", "start": 99, "end": 100},
            {"suggestion_id": "cut-2", "decision": "rejected"},
        ]
    }

    cuts = approved_cuts_from_review(suggestions, review)

    assert cuts == [
        ApprovedCut(
            suggestion_id="cut-1",
            cut_type="filler_word",
            start=0.2,
            end=0.4,
            duration=0.2,
        )
    ]


def test_approved_cuts_from_review_rejects_invalid_approved_timing() -> None:
    suggestions = {"suggestions": [{"id": "cut-1", "start": 2.0, "end": 1.0}]}
    review = {"decisions": [{"suggestion_id": "cut-1", "decision": "approved"}]}

    with pytest.raises(CutApplyError, match="invalid time range"):
        approved_cuts_from_review(suggestions, review)


def test_merged_cut_ranges_collapses_overlaps() -> None:
    ranges = merged_cut_ranges(
        [
            ApprovedCut("cut-2", "silence", 1.0, 2.0),
            ApprovedCut("cut-1", "filler_word", 0.2, 0.4),
            ApprovedCut("cut-3", "filler_word", 1.5, 2.5),
        ]
    )

    assert ranges == [(0.2, 0.4), (1.0, 2.5)]


def test_apply_approved_cuts_writes_reviewed_output_and_manifest(tmp_path, monkeypatch) -> None:
    configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "source.m4a"
    input_path.write_bytes(b"source")
    seen_commands: list[list[str]] = []

    def fake_run_command(args, cwd=None, log_path=None):
        _ = cwd, log_path
        seen_commands.append(args)
        Path(args[-1]).write_bytes(b"reviewed")

    monkeypatch.setattr("openphonic.services.cuts.run_command", fake_run_command)

    output_path = apply_approved_cuts(
        job_id="job-1",
        input_path=input_path,
        cuts=[
            ApprovedCut("cut-1", "filler_word", 0.0, 0.2, 0.2),
            ApprovedCut("cut-2", "silence", 1.0, 2.0, 1.0),
        ],
        target=TargetFormat(),
        suggestions_version="suggestions-v1",
        review_version="review-v1",
        source_suggestions_artifact="cut_suggestions.json",
        source_review_artifact="cut_review.json",
    )

    manifest_path = get_settings().jobs_dir / "job-1" / CUT_APPLY_MANIFEST_ARTIFACT
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert output_path == get_settings().jobs_dir / "job-1" / "cut_reviewed.m4a"
    assert output_path.read_bytes() == b"reviewed"
    assert manifest["status"] == "succeeded"
    assert manifest["approved_count"] == 2
    assert manifest["output_artifact"] == "cut_reviewed.m4a"
    assert manifest["suggestions_version"] == "suggestions-v1"
    assert manifest["review_version"] == "review-v1"
    assert "between(t,0,0.2)" in seen_commands[0][seen_commands[0].index("-af") + 1]
