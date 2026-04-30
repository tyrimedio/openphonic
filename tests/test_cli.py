import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from openphonic.cli import (
    inspect_cut_suggestions,
    inspect_diarization,
    inspect_job,
    inspect_transcript,
    process_file,
    readiness,
    smoke_test,
)
from openphonic.core.settings import get_settings


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("OPENPHONIC_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENPHONIC_DATABASE_PATH", str(data_dir / "openphonic.sqlite3"))
    monkeypatch.setenv("OPENPHONIC_PIPELINE_CONFIG", "config/default.yml")
    get_settings.cache_clear()
    yield data_dir
    get_settings.cache_clear()


def test_process_file_loads_custom_presets(
    tmp_path,
    tmp_settings,
    monkeypatch,
    capsys,
) -> None:
    preset_dir = tmp_settings / "presets"
    preset_dir.mkdir(parents=True)
    (preset_dir / "daily-show.yml").write_text(
        """
name: daily-show
target:
  codec: pcm_s16le
  container: wav
stages:
  loudness:
    enabled: true
""",
        encoding="utf-8",
    )
    input_path = tmp_path / "sample.wav"
    output_path = tmp_path / "out" / "processed.wav"
    loaded: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, config, command_log_path: Path | None = None) -> None:
            loaded["name"] = config.name
            loaded["container"] = config.target.container
            loaded["command_log_path"] = command_log_path

        def run(self, input_path_arg: Path, work_dir: Path):
            loaded["input_path"] = input_path_arg
            loaded["work_dir"] = work_dir
            output = work_dir / "final.wav"
            output.write_bytes(b"preset-output")
            return SimpleNamespace(output_path=output, artifacts={"final_audio": output})

    monkeypatch.setattr("openphonic.cli.PipelineRunner", FakeRunner)

    result = process_file(
        argparse.Namespace(
            input=str(input_path),
            output=str(output_path),
            config=None,
            preset="custom:daily-show",
            work_dir=None,
        )
    )

    assert result == 0
    assert loaded == {
        "name": "daily-show",
        "container": "wav",
        "command_log_path": output_path.with_suffix("") / "commands.jsonl",
        "input_path": input_path.resolve(),
        "work_dir": output_path.with_suffix(""),
    }
    assert output_path.read_bytes() == b"preset-output"
    assert "Processed audio:" in capsys.readouterr().out


def test_process_file_reports_unknown_presets_before_work(
    tmp_path,
    tmp_settings,
    monkeypatch,
    capsys,
) -> None:
    output_path = tmp_path / "out" / "processed.m4a"

    class FailRunner:
        def __init__(self, config, command_log_path: Path | None = None) -> None:
            _ = config, command_log_path
            raise AssertionError("pipeline should not run after config failure")

    monkeypatch.setattr("openphonic.cli.PipelineRunner", FailRunner)

    result = process_file(
        argparse.Namespace(
            input=str(tmp_path / "sample.wav"),
            output=str(output_path),
            config=None,
            preset="missing",
            work_dir=str(tmp_path / "work"),
        )
    )

    assert result == 2
    assert "Process config failed: Unknown pipeline preset: missing" in capsys.readouterr().err
    assert not (tmp_path / "work").exists()
    assert not output_path.exists()


def test_process_file_preflights_before_work(
    tmp_path,
    tmp_settings,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "missing-intro.yml"
    config_path.write_text(
        """
name: missing-intro
stages:
  intro_outro:
    enabled: true
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "out" / "processed.m4a"

    class FailRunner:
        def __init__(self, config, command_log_path: Path | None = None) -> None:
            _ = config, command_log_path
            raise AssertionError("pipeline should not run after preflight failure")

    monkeypatch.setattr("openphonic.cli.PipelineRunner", FailRunner)

    result = process_file(
        argparse.Namespace(
            input=str(tmp_path / "sample.wav"),
            output=str(output_path),
            config=str(config_path),
            preset=None,
            work_dir=str(tmp_path / "work"),
        )
    )

    assert result == 2
    captured = capsys.readouterr()
    assert "Process preflight failed:" in captured.err
    assert "Intro/outro insertion requires intro_path or outro_path." in captured.err
    assert not (tmp_path / "work").exists()
    assert not output_path.exists()


def test_inspect_transcript_reports_word_timestamp_coverage(
    tmp_path,
    capsys,
) -> None:
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        """
{
  "engine": "faster-whisper",
  "model": "tiny",
  "language": "en",
  "duration": 1.4,
  "segments": [
    {
      "start": 0.0,
      "end": 1.4,
      "text": "Hello world",
      "words": [
        {"start": 0.0, "end": 0.5, "word": "Hello"},
        {"start": 0.6, "end": 1.1, "word": "world"}
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )

    result = inspect_transcript(argparse.Namespace(transcript=str(transcript_path), strict=False))

    assert result == 0
    captured = capsys.readouterr()
    assert f"Transcript: {transcript_path.resolve()}" in captured.out
    assert "Engine: faster-whisper" in captured.out
    assert "Model: tiny" in captured.out
    assert "Language: en" in captured.out
    assert "Duration: 1.400s" in captured.out
    assert "Segments: 1" in captured.out
    assert "Segments with words: 1/1" in captured.out
    assert "Words: 2" in captured.out
    assert "Timed words: 2/2 (100.0%)" in captured.out
    assert "Warnings:" not in captured.out


def test_inspect_transcript_strict_fails_on_missing_word_timestamps(
    tmp_path,
    capsys,
) -> None:
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        """
{
  "segments": [
    {"start": 0.0, "end": 1.0, "text": "No words", "words": []}
  ]
}
""",
        encoding="utf-8",
    )

    result = inspect_transcript(argparse.Namespace(transcript=str(transcript_path), strict=True))

    assert result == 2
    captured = capsys.readouterr()
    assert "Segments: 1" in captured.out
    assert "Words: 0" in captured.out
    assert "Timed words: 0/0 (0.0%)" in captured.out
    assert "Transcript has no word timestamps." in captured.out


def test_inspect_transcript_strict_fails_on_invalid_word_timestamps(
    tmp_path,
    capsys,
) -> None:
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        """
{
  "segments": [
    {
      "start": 0.0,
      "end": 1.0,
      "text": "Bad words",
      "words": [
        {"start": -0.5, "end": 0.1, "word": "negative"},
        {"start": 0.2, "end": 1.2, "word": "outside"}
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )

    result = inspect_transcript(argparse.Namespace(transcript=str(transcript_path), strict=True))

    assert result == 2
    captured = capsys.readouterr()
    assert "Words: 2" in captured.out
    assert "Timed words: 0/2 (0.0%)" in captured.out
    assert "Some transcript words are missing valid timestamps." in captured.out
    assert "2 transcript timing value(s) are invalid." in captured.out


def test_inspect_transcript_handles_overflowing_timestamp_values(
    tmp_path,
    capsys,
) -> None:
    huge_timestamp = "1" + ("0" * 1000)
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        f"""
{{
  "duration": {huge_timestamp},
  "segments": [
    {{
      "start": 0.0,
      "end": {huge_timestamp},
      "text": "Huge timestamps",
      "words": [
        {{"start": 0.0, "end": {huge_timestamp}, "word": "huge"}}
      ]
    }}
  ]
}}
""",
        encoding="utf-8",
    )

    result = inspect_transcript(argparse.Namespace(transcript=str(transcript_path), strict=True))

    assert result == 2
    captured = capsys.readouterr()
    assert "Duration: -" in captured.out
    assert "Timed words: 0/1 (0.0%)" in captured.out
    assert "Some transcript words are missing valid timestamps." in captured.out
    assert "3 transcript timing value(s) are invalid." in captured.out
    assert "Traceback" not in captured.err


def test_inspect_transcript_strict_fails_on_timestamps_after_duration(
    tmp_path,
    capsys,
) -> None:
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        """
{
  "duration": 1.0,
  "segments": [
    {
      "start": 1.1,
      "end": 1.2,
      "text": "After audio",
      "words": [
        {"start": 1.1, "end": 1.2, "word": "late"}
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )

    result = inspect_transcript(argparse.Namespace(transcript=str(transcript_path), strict=True))

    assert result == 2
    captured = capsys.readouterr()
    assert "Duration: 1.000s" in captured.out
    assert "Timed words: 0/1 (0.0%)" in captured.out
    assert "Some transcript words are missing valid timestamps." in captured.out
    assert "2 transcript timing value(s) are invalid." in captured.out


def test_inspect_transcript_rejects_invalid_artifacts(
    tmp_path,
    capsys,
) -> None:
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text("[]", encoding="utf-8")

    result = inspect_transcript(argparse.Namespace(transcript=str(transcript_path), strict=False))

    assert result == 2
    assert (
        "Transcript inspection failed: transcript must be a JSON object." in capsys.readouterr().err
    )


def test_inspect_diarization_reports_speaker_summary(
    tmp_path,
    capsys,
) -> None:
    diarization_path = tmp_path / "diarization.json"
    diarization_path.write_text(
        """
{
  "engine": "pyannote.audio",
  "model": "pyannote/test-diarization",
  "speaker_count": 2,
  "segments": [
    {"start": 0.0, "end": 0.8, "speaker": "SPEAKER_00", "track": "A"},
    {"start": 1.0, "end": 1.5, "speaker": "SPEAKER_01", "track": "B"}
  ]
}
""",
        encoding="utf-8",
    )

    result = inspect_diarization(
        argparse.Namespace(diarization=str(diarization_path), duration=2.0, strict=False)
    )

    assert result == 0
    captured = capsys.readouterr()
    assert f"Diarization: {diarization_path.resolve()}" in captured.out
    assert "Engine: pyannote.audio" in captured.out
    assert "Model: pyannote/test-diarization" in captured.out
    assert "Declared speakers: 2" in captured.out
    assert "Detected speakers: 2" in captured.out
    assert "Segments: 2" in captured.out
    assert "Timed segments: 2/2" in captured.out
    assert "Total speaker time: 1.300s" in captured.out
    assert "Warnings:" not in captured.out


def test_inspect_diarization_strict_fails_on_invalid_segments(
    tmp_path,
    capsys,
) -> None:
    diarization_path = tmp_path / "diarization.json"
    diarization_path.write_text(
        """
{
  "speaker_count": 3,
  "segments": [
    {"start": -0.1, "end": 0.2, "speaker": "SPEAKER_00"},
    {"start": 0.3, "end": 2.1, "speaker": ""},
    {"start": "nan", "end": 0.5, "speaker": "SPEAKER_01"},
    []
  ]
}
""",
        encoding="utf-8",
    )

    result = inspect_diarization(
        argparse.Namespace(diarization=str(diarization_path), duration=2.0, strict=True)
    )

    assert result == 2
    captured = capsys.readouterr()
    assert "Declared speakers: 3" in captured.out
    assert "Detected speakers: 2" in captured.out
    assert "Segments: 4" in captured.out
    assert "Timed segments: 0/4" in captured.out
    assert "Segment 3 is not an object." in captured.out
    assert "Diarization speaker_count does not match detected speaker labels (3 != 2)." in (
        captured.out
    )
    assert "2 diarization segment(s) have no speaker label." in captured.out
    assert "4 diarization segment timing value(s) are invalid." in captured.out


def test_inspect_diarization_handles_overflowing_timestamp_values(
    tmp_path,
    capsys,
) -> None:
    huge_timestamp = "1" + ("0" * 1000)
    diarization_path = tmp_path / "diarization.json"
    diarization_path.write_text(
        f"""
{{
  "segments": [
    {{"start": 0.0, "end": {huge_timestamp}, "speaker": "SPEAKER_00"}}
  ]
}}
""",
        encoding="utf-8",
    )

    result = inspect_diarization(
        argparse.Namespace(diarization=str(diarization_path), duration=None, strict=True)
    )

    assert result == 2
    captured = capsys.readouterr()
    assert "Timed segments: 0/1" in captured.out
    assert "1 diarization segment timing value(s) are invalid." in captured.out
    assert "Traceback" not in captured.err


def test_inspect_diarization_reports_invalid_utf8(
    tmp_path,
    capsys,
) -> None:
    diarization_path = tmp_path / "diarization.json"
    diarization_path.write_bytes(b"\xff")

    result = inspect_diarization(
        argparse.Namespace(diarization=str(diarization_path), duration=None, strict=False)
    )

    assert result == 2
    captured = capsys.readouterr()
    assert "Diarization inspection failed: invalid UTF-8:" in captured.err
    assert "Traceback" not in captured.err


def test_inspect_diarization_reports_json_value_errors(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    diarization_path = tmp_path / "diarization.json"
    diarization_path.write_text('{"segments": []}', encoding="utf-8")

    def raise_value_error(text: str) -> None:
        _ = text
        raise ValueError("exceeds the limit for integer string conversion")

    monkeypatch.setattr("openphonic.cli.json.loads", raise_value_error)

    result = inspect_diarization(
        argparse.Namespace(diarization=str(diarization_path), duration=None, strict=False)
    )

    assert result == 2
    captured = capsys.readouterr()
    assert (
        "Diarization inspection failed: invalid JSON: "
        "exceeds the limit for integer string conversion"
    ) in captured.err
    assert "Traceback" not in captured.err


def test_inspect_diarization_rejects_invalid_artifacts(
    tmp_path,
    capsys,
) -> None:
    diarization_path = tmp_path / "diarization.json"
    diarization_path.write_text("[]", encoding="utf-8")

    result = inspect_diarization(
        argparse.Namespace(diarization=str(diarization_path), duration=None, strict=False)
    )

    assert result == 2
    assert (
        "Diarization inspection failed: diarization must be a JSON object."
        in capsys.readouterr().err
    )


def test_inspect_cut_suggestions_reports_review_summary(
    tmp_path,
    capsys,
) -> None:
    suggestions_path = tmp_path / "cut_suggestions.json"
    suggestions_path.write_text(
        """
{
  "status": "not_applied",
  "reason": "Suggestions only; manual review is required before any cuts are applied.",
  "source_artifact": "transcript.json",
  "configured_words": ["um", "uh"],
  "min_silence_seconds": 0.75,
  "suggestion_count": 2,
  "suggestions": [
    {
      "id": "cut-0001",
      "type": "filler_word",
      "start": 0.0,
      "end": 0.2,
      "duration": 0.2,
      "text": "Um",
      "reason": "Matched configured filler word."
    },
    {
      "id": "cut-0002",
      "type": "silence",
      "start": 0.9,
      "end": 1.8,
      "duration": 0.9,
      "source": "word_gap",
      "reason": "Detected a timestamp gap longer than the configured threshold."
    }
  ]
}
""",
        encoding="utf-8",
    )

    result = inspect_cut_suggestions(
        argparse.Namespace(cut_suggestions=str(suggestions_path), duration=2.0, strict=False)
    )

    assert result == 0
    captured = capsys.readouterr()
    assert f"Cut suggestions: {suggestions_path.resolve()}" in captured.out
    assert "Status: not_applied" in captured.out
    assert "Source artifact: transcript.json" in captured.out
    assert "Configured words: um, uh" in captured.out
    assert "Minimum silence: 0.750s" in captured.out
    assert "Suggestions: 2" in captured.out
    assert "Timed suggestions: 2/2" in captured.out
    assert "Types: filler_word=1, silence=1" in captured.out
    assert "Suggested duration: 1.100s" in captured.out
    assert "Merged cut duration: 1.100s" in captured.out
    assert "Warnings:" not in captured.out


def test_inspect_cut_suggestions_strict_fails_on_invalid_suggestions(
    tmp_path,
    capsys,
) -> None:
    suggestions_path = tmp_path / "cut_suggestions.json"
    suggestions_path.write_text(
        """
{
  "status": "not_available",
  "configured_words": ["um", ""],
  "min_silence_seconds": -0.1,
  "suggestion_count": 4,
  "suggestions": [
    {"id": "cut-0001", "type": "filler_word", "start": 0.0, "end": 0.2, "duration": 0.2},
    {"id": "cut-0001", "type": "", "start": 1.0, "end": 0.5, "duration": 0.5},
    {"type": "silence", "start": 2.0, "end": 2.5, "duration": 0.25},
    []
  ]
}
""",
        encoding="utf-8",
    )

    result = inspect_cut_suggestions(
        argparse.Namespace(cut_suggestions=str(suggestions_path), duration=3.0, strict=True)
    )

    assert result == 2
    captured = capsys.readouterr()
    assert "Status: not_available" in captured.out
    assert "Suggestions: 4" in captured.out
    assert "Timed suggestions: 1/4" in captured.out
    assert "Types: filler_word=1, silence=1" in captured.out
    assert "Cut suggestions are not ready for review: not_available." in captured.out
    assert "Cut suggestions configured_words must be a list of non-empty strings." in (captured.out)
    assert "Cut suggestions min_silence_seconds must be non-negative." in captured.out
    assert "2 cut suggestion(s) have no id." in captured.out
    assert "1 cut suggestion id(s) are duplicated." in captured.out
    assert "2 cut suggestion(s) have no type." in captured.out
    assert "2 cut suggestion timing value(s) are invalid." in captured.out
    assert "1 cut suggestion duration value(s) do not match start/end." in captured.out


def test_inspect_cut_suggestions_reports_invalid_utf8(
    tmp_path,
    capsys,
) -> None:
    suggestions_path = tmp_path / "cut_suggestions.json"
    suggestions_path.write_bytes(b"\xff")

    result = inspect_cut_suggestions(
        argparse.Namespace(cut_suggestions=str(suggestions_path), duration=None, strict=False)
    )

    assert result == 2
    captured = capsys.readouterr()
    assert "Cut suggestion inspection failed: invalid UTF-8:" in captured.err
    assert "Traceback" not in captured.err


def test_inspect_cut_suggestions_reports_json_value_errors(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    suggestions_path = tmp_path / "cut_suggestions.json"
    suggestions_path.write_text('{"suggestions": []}', encoding="utf-8")

    def raise_value_error(text: str) -> None:
        _ = text
        raise ValueError("exceeds the limit for integer string conversion")

    monkeypatch.setattr("openphonic.cli.json.loads", raise_value_error)

    result = inspect_cut_suggestions(
        argparse.Namespace(cut_suggestions=str(suggestions_path), duration=None, strict=False)
    )

    assert result == 2
    captured = capsys.readouterr()
    assert (
        "Cut suggestion inspection failed: invalid JSON: "
        "exceeds the limit for integer string conversion"
    ) in captured.err
    assert "Traceback" not in captured.err


def test_inspect_cut_suggestions_rejects_invalid_artifacts(
    tmp_path,
    capsys,
) -> None:
    suggestions_path = tmp_path / "cut_suggestions.json"
    suggestions_path.write_text("[]", encoding="utf-8")

    result = inspect_cut_suggestions(
        argparse.Namespace(cut_suggestions=str(suggestions_path), duration=None, strict=False)
    )

    assert result == 2
    assert (
        "Cut suggestion inspection failed: cut suggestions must be a JSON object."
        in capsys.readouterr().err
    )


def test_inspect_job_reports_manifest_artifacts(
    tmp_path,
    capsys,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    input_path = tmp_path / "input.wav"
    output_path = work_dir / "processed.m4a"
    metadata_path = work_dir / "00_media_metadata.json"
    manifest_path = work_dir / "pipeline_manifest.json"
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"output")
    metadata_path.write_text("{}", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": "2026-04-30T00:00:00+00:00",
                "status": "succeeded",
                "pipeline_name": "podcast-default",
                "input_path": str(input_path),
                "work_dir": str(work_dir),
                "output_path": str(output_path),
                "artifacts": {
                    "final_audio": str(output_path),
                    "media_metadata": str(metadata_path),
                },
            }
        ),
        encoding="utf-8",
    )

    result = inspect_job(argparse.Namespace(work_dir=str(work_dir), strict=True))

    assert result == 0
    captured = capsys.readouterr()
    assert f"Job work directory: {work_dir.resolve()}" in captured.out
    assert f"Manifest: {manifest_path.resolve()}" in captured.out
    assert "Status: succeeded" in captured.out
    assert "Pipeline: podcast-default" in captured.out
    assert f"Input: {input_path} [ok]" in captured.out
    assert f"Output: {output_path} [ok]" in captured.out
    assert "Artifacts: 2/2 present" in captured.out
    assert f"  - final_audio: {output_path} [ok]" in captured.out
    assert f"  - media_metadata: {metadata_path} [ok]" in captured.out
    assert "Warnings:" not in captured.out


def test_inspect_job_anchors_relative_manifest_paths_to_job_base(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    project_dir = tmp_path / "project"
    work_dir = project_dir / "data" / "jobs" / "job-1"
    upload_dir = project_dir / "data" / "uploads" / "job-1"
    unrelated_cwd = tmp_path / "elsewhere"
    work_dir.mkdir(parents=True)
    upload_dir.mkdir(parents=True)
    unrelated_cwd.mkdir()
    input_path = upload_dir / "input.wav"
    output_path = work_dir / "processed.m4a"
    metadata_path = work_dir / "00_media_metadata.json"
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"output")
    metadata_path.write_text("{}", encoding="utf-8")
    (work_dir / "pipeline_manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-04-30T00:00:00+00:00",
                "status": "succeeded",
                "pipeline_name": "podcast-default",
                "input_path": "data/uploads/job-1/input.wav",
                "work_dir": "data/jobs/job-1",
                "output_path": "data/jobs/job-1/processed.m4a",
                "artifacts": {
                    "final_audio": "data/jobs/job-1/processed.m4a",
                    "media_metadata": "data/jobs/job-1/00_media_metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(unrelated_cwd)

    result = inspect_job(argparse.Namespace(work_dir=str(work_dir.resolve()), strict=True))

    assert result == 0
    captured = capsys.readouterr()
    assert f"Input: {input_path} [ok]" in captured.out
    assert f"Output: {output_path} [ok]" in captured.out
    assert "Artifacts: 2/2 present" in captured.out
    assert "Warnings:" not in captured.out


def test_inspect_job_keeps_raw_relative_path_fallback_for_symlinked_data(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    project_dir = tmp_path / "project"
    real_data_dir = tmp_path / "real-data"
    symlink_data_dir = project_dir / "data"
    work_dir = symlink_data_dir / "jobs" / "job-1"
    upload_dir = symlink_data_dir / "uploads" / "job-1"
    project_dir.mkdir()
    real_data_dir.mkdir()
    symlink_data_dir.symlink_to(real_data_dir, target_is_directory=True)
    work_dir.mkdir(parents=True)
    upload_dir.mkdir(parents=True)
    input_path = Path("data/uploads/job-1/input.wav")
    output_path = Path("data/jobs/job-1/processed.m4a")
    metadata_path = Path("data/jobs/job-1/00_media_metadata.json")
    (project_dir / input_path).write_bytes(b"input")
    (project_dir / output_path).write_bytes(b"output")
    (project_dir / metadata_path).write_text("{}", encoding="utf-8")
    (work_dir / "pipeline_manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-04-30T00:00:00+00:00",
                "status": "succeeded",
                "pipeline_name": "podcast-default",
                "input_path": str(input_path),
                "work_dir": "data/jobs/job-1",
                "output_path": str(output_path),
                "artifacts": {
                    "final_audio": str(output_path),
                    "media_metadata": str(metadata_path),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project_dir)

    result = inspect_job(argparse.Namespace(work_dir="data/jobs/job-1", strict=True))

    assert result == 0
    captured = capsys.readouterr()
    assert f"Input: {input_path} [ok]" in captured.out
    assert f"Output: {output_path} [ok]" in captured.out
    assert "Artifacts: 2/2 present" in captured.out
    assert "Warnings:" not in captured.out


def test_inspect_job_ignores_raw_relative_paths_when_manifest_base_exists(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    project_dir = tmp_path / "project"
    work_dir = project_dir / "data" / "jobs" / "job-1"
    upload_dir = project_dir / "data" / "uploads" / "job-1"
    stale_cwd = tmp_path / "elsewhere"
    stale_job_dir = stale_cwd / "data" / "jobs" / "job-1"
    work_dir.mkdir(parents=True)
    upload_dir.mkdir(parents=True)
    stale_job_dir.mkdir(parents=True)
    input_path = upload_dir / "input.wav"
    output_path = work_dir / "processed.m4a"
    metadata_path = work_dir / "00_media_metadata.json"
    input_path.write_bytes(b"input")
    (stale_job_dir / "processed.m4a").write_bytes(b"stale-output")
    (stale_job_dir / "00_media_metadata.json").write_text("{}", encoding="utf-8")
    (work_dir / "pipeline_manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-04-30T00:00:00+00:00",
                "status": "succeeded",
                "pipeline_name": "podcast-default",
                "input_path": "data/uploads/job-1/input.wav",
                "work_dir": "data/jobs/job-1",
                "output_path": "data/jobs/job-1/processed.m4a",
                "artifacts": {
                    "final_audio": "data/jobs/job-1/processed.m4a",
                    "media_metadata": "data/jobs/job-1/00_media_metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(stale_cwd)

    result = inspect_job(argparse.Namespace(work_dir=str(work_dir.resolve()), strict=True))

    assert result == 2
    captured = capsys.readouterr()
    assert f"Pipeline output is missing: {output_path}" in captured.out
    assert f"Pipeline artifact final_audio is missing: {output_path}" in captured.out
    assert f"Pipeline artifact media_metadata is missing: {metadata_path}" in captured.out
    assert "Artifacts: 0/2 present" in captured.out


def test_inspect_job_strict_fails_on_missing_paths(
    tmp_path,
    capsys,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    manifest_path = work_dir / "pipeline_manifest.json"
    input_path = tmp_path / "input.wav"
    output_path = work_dir / "processed.m4a"
    artifact_path = work_dir / "00_media_metadata.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "pipeline_name": "podcast-default",
                "input_path": str(input_path),
                "work_dir": str(work_dir),
                "output_path": str(output_path),
                "artifacts": {"media_metadata": str(artifact_path)},
            }
        ),
        encoding="utf-8",
    )

    result = inspect_job(argparse.Namespace(work_dir=str(work_dir), strict=True))

    assert result == 2
    captured = capsys.readouterr()
    assert "Status: succeeded" in captured.out
    assert "Artifacts: 0/1 present" in captured.out
    assert f"Pipeline input is missing: {input_path}" in captured.out
    assert f"Pipeline output is missing: {output_path}" in captured.out
    assert f"Pipeline artifact media_metadata is missing: {artifact_path}" in captured.out


def test_inspect_job_rejects_malformed_manifest(
    tmp_path,
    capsys,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "pipeline_manifest.json").write_text("[]", encoding="utf-8")

    result = inspect_job(argparse.Namespace(work_dir=str(work_dir), strict=False))

    assert result == 2
    assert "Job inspection failed: manifest must be a JSON object." in capsys.readouterr().err


def test_smoke_test_generates_input_and_copies_output(
    tmp_path,
    tmp_settings,
    monkeypatch,
    capsys,
) -> None:
    generated: dict[str, list[str]] = {}
    runner_calls: dict[str, Path] = {}

    def fake_run_command(args: list[str], cwd=None, log_path=None) -> subprocess.CompletedProcess:
        _ = cwd, log_path
        generated["args"] = args
        Path(args[-1]).write_bytes(b"input")
        return subprocess.CompletedProcess(args=args, returncode=0)

    class FakeRunner:
        def __init__(self, config, command_log_path: Path | None = None) -> None:
            runner_calls["config_path"] = config.source_path
            runner_calls["command_log_path"] = command_log_path

        def run(self, input_path: Path, work_dir: Path):
            runner_calls["input_path"] = input_path
            runner_calls["work_dir"] = work_dir
            output_path = work_dir / "final.m4a"
            manifest_path = work_dir / "pipeline_manifest.json"
            output_path.write_bytes(b"processed")
            manifest_path.write_text("{}", encoding="utf-8")
            return SimpleNamespace(
                output_path=output_path,
                artifacts={
                    "final_audio": output_path,
                    "pipeline_manifest": manifest_path,
                },
            )

    monkeypatch.setattr("openphonic.cli.run_command", fake_run_command)
    monkeypatch.setattr("openphonic.cli.PipelineRunner", FakeRunner)
    output_path = tmp_path / "out" / "processed.m4a"

    result = smoke_test(
        argparse.Namespace(
            output=str(output_path),
            config=None,
            preset=None,
            work_dir=None,
            duration=0.5,
            frequency=440,
        )
    )

    assert result == 0
    assert output_path.read_bytes() == b"processed"
    assert generated["args"][:8] == [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=0.5",
    ]
    assert runner_calls["config_path"] == Path("config/default.yml")
    assert runner_calls["input_path"] == output_path.parent / "work" / "input.wav"
    assert runner_calls["command_log_path"] == output_path.parent / "work" / "commands.jsonl"
    assert "Smoke test processed audio:" in capsys.readouterr().out


def test_smoke_test_uses_data_dir_defaults(tmp_settings, monkeypatch) -> None:
    def fake_run_command(args: list[str], cwd=None, log_path=None) -> subprocess.CompletedProcess:
        _ = cwd, log_path
        Path(args[-1]).write_bytes(b"input")
        return subprocess.CompletedProcess(args=args, returncode=0)

    class FakeRunner:
        def __init__(self, config, command_log_path: Path | None = None) -> None:
            _ = config, command_log_path

        def run(self, input_path: Path, work_dir: Path):
            assert input_path == tmp_settings / "smoke-test" / "work" / "input.wav"
            output_path = work_dir / "final.m4a"
            output_path.write_bytes(b"processed")
            return SimpleNamespace(output_path=output_path, artifacts={})

    monkeypatch.setattr("openphonic.cli.run_command", fake_run_command)
    monkeypatch.setattr("openphonic.cli.PipelineRunner", FakeRunner)

    result = smoke_test(
        argparse.Namespace(
            output=None,
            config=None,
            preset=None,
            work_dir=None,
            duration=1.0,
            frequency=1000,
        )
    )

    assert result == 0
    assert (tmp_settings / "smoke-test" / "processed.m4a").read_bytes() == b"processed"


def test_smoke_test_uses_configured_container_for_default_output(
    tmp_path,
    tmp_settings,
    monkeypatch,
) -> None:
    config_path = tmp_path / "smoke-wav.yml"
    config_path.write_text(
        """
name: smoke-wav
target:
  codec: pcm_s16le
  container: wav
stages:
  loudness:
    enabled: true
""",
        encoding="utf-8",
    )

    def fake_run_command(args: list[str], cwd=None, log_path=None) -> subprocess.CompletedProcess:
        _ = cwd, log_path
        Path(args[-1]).write_bytes(b"input")
        return subprocess.CompletedProcess(args=args, returncode=0)

    class FakeRunner:
        def __init__(self, config, command_log_path: Path | None = None) -> None:
            assert config.target.container == "wav"
            _ = command_log_path

        def run(self, input_path: Path, work_dir: Path):
            _ = input_path
            output_path = work_dir / "final.wav"
            output_path.write_bytes(b"wav-bytes")
            return SimpleNamespace(output_path=output_path, artifacts={})

    monkeypatch.setattr("openphonic.cli.run_command", fake_run_command)
    monkeypatch.setattr("openphonic.cli.PipelineRunner", FakeRunner)

    result = smoke_test(
        argparse.Namespace(
            output=None,
            config=str(config_path),
            preset=None,
            work_dir=None,
            duration=1.0,
            frequency=1000,
        )
    )

    assert result == 0
    assert (tmp_settings / "smoke-test" / "processed.wav").read_bytes() == b"wav-bytes"
    assert not (tmp_settings / "smoke-test" / "processed.m4a").exists()


def test_smoke_test_loads_custom_presets(
    tmp_settings,
    monkeypatch,
) -> None:
    preset_dir = tmp_settings / "presets"
    preset_dir.mkdir(parents=True)
    (preset_dir / "daily-show.yml").write_text(
        """
name: daily-show
target:
  codec: pcm_s16le
  container: wav
stages:
  loudness:
    enabled: true
""",
        encoding="utf-8",
    )
    loaded: dict[str, str] = {}

    def fake_run_command(args: list[str], cwd=None, log_path=None) -> subprocess.CompletedProcess:
        _ = cwd, log_path
        Path(args[-1]).write_bytes(b"input")
        return subprocess.CompletedProcess(args=args, returncode=0)

    class FakeRunner:
        def __init__(self, config, command_log_path: Path | None = None) -> None:
            loaded["name"] = config.name
            loaded["container"] = config.target.container
            _ = command_log_path

        def run(self, input_path: Path, work_dir: Path):
            _ = input_path
            output_path = work_dir / "final.wav"
            output_path.write_bytes(b"preset-wav")
            return SimpleNamespace(output_path=output_path, artifacts={})

    monkeypatch.setattr("openphonic.cli.run_command", fake_run_command)
    monkeypatch.setattr("openphonic.cli.PipelineRunner", FakeRunner)

    result = smoke_test(
        argparse.Namespace(
            output=None,
            config=None,
            preset="custom:daily-show",
            work_dir=None,
            duration=1.0,
            frequency=1000,
        )
    )

    assert result == 0
    assert loaded == {"name": "daily-show", "container": "wav"}
    assert (tmp_settings / "smoke-test" / "processed.wav").read_bytes() == b"preset-wav"
    assert not (tmp_settings / "smoke-test" / "processed.m4a").exists()


def test_smoke_test_reports_unknown_presets_before_generating_input(
    tmp_settings,
    monkeypatch,
    capsys,
) -> None:
    def fail_run_command(args: list[str], cwd=None, log_path=None) -> subprocess.CompletedProcess:
        _ = args, cwd, log_path
        raise AssertionError("smoke input should not be generated after config failure")

    monkeypatch.setattr("openphonic.cli.run_command", fail_run_command)

    result = smoke_test(
        argparse.Namespace(
            output=None,
            config=None,
            preset="missing",
            work_dir=None,
            duration=1.0,
            frequency=1000,
        )
    )

    assert result == 2
    assert "Smoke test config failed: Unknown pipeline preset: missing" in capsys.readouterr().err


def test_smoke_test_preflights_before_generating_input(
    tmp_path,
    tmp_settings,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "missing-intro.yml"
    config_path.write_text(
        """
name: missing-intro
stages:
  intro_outro:
    enabled: true
""",
        encoding="utf-8",
    )

    def fail_run_command(args: list[str], cwd=None, log_path=None) -> subprocess.CompletedProcess:
        _ = args, cwd, log_path
        raise AssertionError("smoke input should not be generated after preflight failure")

    class FailRunner:
        def __init__(self, config, command_log_path: Path | None = None) -> None:
            _ = config, command_log_path
            raise AssertionError("pipeline should not run after preflight failure")

    monkeypatch.setattr("openphonic.cli.run_command", fail_run_command)
    monkeypatch.setattr("openphonic.cli.PipelineRunner", FailRunner)

    result = smoke_test(
        argparse.Namespace(
            output=None,
            config=str(config_path),
            preset=None,
            work_dir=None,
            duration=1.0,
            frequency=1000,
        )
    )

    assert result == 2
    captured = capsys.readouterr()
    assert "Smoke test preflight failed:" in captured.err
    assert "Intro/outro insertion requires intro_path or outro_path." in captured.err


def test_readiness_reports_preset_preflight_status(
    tmp_settings,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("openphonic.cli.shutil.which", lambda executable: f"/usr/bin/{executable}")
    monkeypatch.setattr("openphonic.pipeline.preflight._binary_available", lambda binary: False)
    monkeypatch.setattr("openphonic.pipeline.preflight._module_available", lambda module: False)

    result = readiness(argparse.Namespace(strict=False))

    assert result == 0
    captured = capsys.readouterr()
    assert "[ready] podcast-default - Podcast default" in captured.out
    assert "[blocked] speech-cleanup - Speech cleanup" in captured.out
    assert "DeepFilterNet noise reduction is enabled" in captured.out
    assert "[blocked] transcript-review - Transcript review" in captured.out
    assert "Transcription is enabled, but faster-whisper is not installed." in captured.out


def test_readiness_strict_returns_nonzero_for_blocked_presets(
    tmp_settings,
    monkeypatch,
) -> None:
    monkeypatch.setattr("openphonic.cli.shutil.which", lambda executable: f"/usr/bin/{executable}")
    monkeypatch.setattr("openphonic.pipeline.preflight._binary_available", lambda binary: False)
    monkeypatch.setattr("openphonic.pipeline.preflight._module_available", lambda module: False)

    result = readiness(argparse.Namespace(strict=True))

    assert result == 2


def test_readiness_can_filter_to_requested_presets(
    tmp_settings,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("openphonic.cli.shutil.which", lambda executable: f"/usr/bin/{executable}")
    monkeypatch.setattr("openphonic.pipeline.preflight._module_available", lambda module: False)

    result = readiness(argparse.Namespace(preset=["transcript-review"], strict=False))

    assert result == 0
    captured = capsys.readouterr()
    assert "[blocked] transcript-review - Transcript review" in captured.out
    assert "Transcription is enabled, but faster-whisper is not installed." in captured.out
    assert "podcast-default" not in captured.out
    assert "speaker-diarization" not in captured.out


def test_readiness_reports_unknown_requested_presets(
    tmp_settings,
    capsys,
) -> None:
    result = readiness(argparse.Namespace(preset=["missing"], strict=False))

    assert result == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Readiness config failed: Unknown pipeline preset: missing" in captured.err


def test_readiness_reports_invalid_requested_custom_presets(
    tmp_settings,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("openphonic.cli.shutil.which", lambda executable: f"/usr/bin/{executable}")
    preset_dir = tmp_settings / "presets"
    preset_dir.mkdir(parents=True)
    (preset_dir / "daily.yml").write_text(
        """
name: daily
stages:
  intro_outro:
    enabled: true
    intro_path: missing-intro.wav
""",
        encoding="utf-8",
    )

    result = readiness(argparse.Namespace(preset=["custom:daily"], strict=False))

    assert result == 0
    captured = capsys.readouterr()
    assert "[blocked] custom:daily - Daily" in captured.out
    assert "Intro/outro intro_path does not exist:" in captured.out
    assert "Readiness config failed:" not in captured.err


def test_readiness_reports_malformed_requested_custom_presets(
    tmp_settings,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("openphonic.cli.shutil.which", lambda executable: f"/usr/bin/{executable}")
    preset_dir = tmp_settings / "presets"
    preset_dir.mkdir(parents=True)
    (preset_dir / "daily.yml").write_text("name: [", encoding="utf-8")

    result = readiness(argparse.Namespace(preset=["custom:daily"], strict=False))

    assert result == 0
    captured = capsys.readouterr()
    assert "[blocked] custom:daily - Daily" in captured.out
    assert "Preset config could not be inspected:" in captured.out
    assert "Readiness config failed:" not in captured.err


def test_readiness_reports_schema_invalid_requested_custom_presets(
    tmp_settings,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("openphonic.cli.shutil.which", lambda executable: f"/usr/bin/{executable}")
    preset_dir = tmp_settings / "presets"
    preset_dir.mkdir(parents=True)
    (preset_dir / "bad.yml").write_text(
        """
name: bad
stages:
  loudness: true
""",
        encoding="utf-8",
    )

    result = readiness(argparse.Namespace(preset=["custom:bad"], strict=False))

    assert result == 0
    captured = capsys.readouterr()
    assert "[blocked] custom:bad - Bad" in captured.out
    assert "Preset stage 'loudness' must be a mapping." in captured.out
    assert "Readiness config failed:" not in captured.err


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[]", "Preset config must be a mapping."),
        ("false", "Preset config must be a mapping."),
        ("stages: []", "Preset stages must be a mapping."),
    ],
)
def test_readiness_reports_raw_schema_invalid_requested_custom_presets(
    tmp_settings,
    monkeypatch,
    capsys,
    content,
    message,
) -> None:
    monkeypatch.setattr("openphonic.cli.shutil.which", lambda executable: f"/usr/bin/{executable}")
    preset_dir = tmp_settings / "presets"
    preset_dir.mkdir(parents=True)
    (preset_dir / "bad.yml").write_text(content, encoding="utf-8")

    result = readiness(argparse.Namespace(preset=["custom:bad"], strict=True))

    assert result == 2
    captured = capsys.readouterr()
    assert "[blocked] custom:bad - Bad" in captured.out
    assert message in captured.out
    assert "Readiness config failed:" not in captured.err


def test_readiness_lists_custom_presets(
    tmp_settings,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("openphonic.cli.shutil.which", lambda executable: f"/usr/bin/{executable}")
    preset_dir = tmp_settings / "presets"
    preset_dir.mkdir(parents=True)
    (preset_dir / "daily-show.yml").write_text(
        """
preset:
  label: Daily show
  description: Daily show production preset.
name: daily-show
target:
  codec: pcm_s16le
  container: wav
stages:
  loudness:
    enabled: true
""",
        encoding="utf-8",
    )

    result = readiness(argparse.Namespace(strict=False))

    assert result == 0
    assert "[ready] custom:daily-show - Daily show" in capsys.readouterr().out


def test_readiness_reports_missing_core_media_tools(
    tmp_settings,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("openphonic.cli.shutil.which", lambda executable: None)

    result = readiness(argparse.Namespace(strict=False))

    assert result == 0
    captured = capsys.readouterr()
    assert "[blocked] podcast-default - Podcast default" in captured.out
    assert "ffmpeg is required but was not found on PATH." in captured.out
    assert "ffprobe is required but was not found on PATH." in captured.out


def test_readiness_reports_malformed_default_config(
    tmp_path,
    tmp_settings,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "broken.yml"
    config_path.write_text("name: [", encoding="utf-8")
    monkeypatch.setenv("OPENPHONIC_PIPELINE_CONFIG", str(config_path))
    monkeypatch.setattr("openphonic.cli.shutil.which", lambda executable: f"/usr/bin/{executable}")
    get_settings.cache_clear()

    result = readiness(argparse.Namespace(strict=False))

    assert result == 0
    captured = capsys.readouterr()
    assert "[blocked] podcast-default - Podcast default" in captured.out
    assert "Preset config could not be inspected:" in captured.out
