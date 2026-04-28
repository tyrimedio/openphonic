import json
from pathlib import Path

import pytest

from openphonic.pipeline.config import PipelineConfig
from openphonic.pipeline.ffmpeg import AudioStreamMetadata, MediaMetadata
from openphonic.pipeline.runner import PipelineRunner
from openphonic.pipeline.stages import StageError


class FakeIngestStage:
    seen_command_log_path: Path | None = None

    def __init__(self, config: PipelineConfig, command_log_path: Path | None = None) -> None:
        _ = config
        self.command_log_path = command_log_path

    def run(self, input_path: Path, work_dir: Path) -> Path:
        _ = input_path
        FakeIngestStage.seen_command_log_path = self.command_log_path
        output_path = work_dir / "01_ingest.wav"
        output_path.write_bytes(b"audio")
        return output_path


class FailingTranscriptionStage:
    def __init__(self, config: PipelineConfig, command_log_path: Path | None = None) -> None:
        _ = config, command_log_path

    def run(self, input_path: Path, work_dir: Path) -> dict[str, Path]:
        _ = input_path, work_dir
        raise RuntimeError("transcription unavailable")


class SuggestingTranscriptionStage:
    def __init__(self, config: PipelineConfig, command_log_path: Path | None = None) -> None:
        _ = config, command_log_path

    def run(self, input_path: Path, work_dir: Path) -> dict[str, Path]:
        _ = input_path
        transcript_path = work_dir / "transcript.json"
        transcript_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "segments": [
                        {
                            "id": 1,
                            "start": 0.0,
                            "end": 0.9,
                            "text": " Um hello",
                            "words": [
                                {
                                    "start": 0.0,
                                    "end": 0.22,
                                    "word": " Um",
                                    "probability": 0.91,
                                },
                                {
                                    "start": 0.3,
                                    "end": 0.9,
                                    "word": " hello",
                                    "probability": 0.95,
                                },
                            ],
                        },
                        {
                            "id": 2,
                            "start": 1.8,
                            "end": 2.6,
                            "text": "uh there",
                            "words": [
                                {
                                    "start": 1.8,
                                    "end": 2.0,
                                    "word": "uh",
                                    "probability": 0.89,
                                },
                                {
                                    "start": 2.05,
                                    "end": 2.6,
                                    "word": "there",
                                    "probability": 0.96,
                                },
                            ],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {"transcript_json": transcript_path}


def test_runner_validates_media_and_writes_metadata_artifact(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"not real audio")
    command_log_path = tmp_path / "commands.jsonl"

    def fake_probe_media(path: Path, log_path: Path | None = None) -> MediaMetadata:
        assert path == input_path
        assert log_path == command_log_path
        return MediaMetadata(
            path=path,
            format_name="wav",
            duration_seconds=1.0,
            audio_streams=[
                AudioStreamMetadata(
                    index=0,
                    codec_name="pcm_s16le",
                    sample_rate=48000,
                    channels=2,
                    duration_seconds=1.0,
                )
            ],
        )

    monkeypatch.setattr("openphonic.pipeline.runner.probe_media", fake_probe_media)
    monkeypatch.setattr("openphonic.pipeline.runner.IngestStage", FakeIngestStage)

    config = PipelineConfig(
        name="test",
        stages={
            "silence_trim": {"enabled": False},
            "loudness": {"enabled": False},
        },
    )
    progress: list[tuple[str, int]] = []

    result = PipelineRunner(
        config,
        progress_callback=lambda stage, percent: progress.append((stage, percent)),
        command_log_path=command_log_path,
    ).run(input_path, tmp_path / "work")

    assert result.output_path.name == "01_ingest.wav"
    assert result.artifacts["media_metadata"].exists()
    assert result.artifacts["ingest_wav"] == result.output_path
    assert result.artifacts["final_audio"] == result.output_path
    assert result.artifacts["pipeline_manifest"].exists()
    assert '"format_name": "wav"' in result.artifacts["media_metadata"].read_text()
    manifest = json.loads(result.artifacts["pipeline_manifest"].read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["pipeline_name"] == "test"
    assert manifest["output_path"] == str(result.output_path)
    assert manifest["artifacts"]["media_metadata"] == str(result.artifacts["media_metadata"])
    assert manifest["artifacts"]["final_audio"] == str(result.output_path)
    assert progress[:2] == [("metadata", 8), ("ingest", 10)]
    assert FakeIngestStage.seen_command_log_path == command_log_path


def test_runner_writes_failed_manifest_with_partial_artifacts(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"not real audio")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    stale_suggestions = work_dir / "cut_suggestions.json"
    stale_suggestions.write_text('{"status": "stale"}', encoding="utf-8")

    def fake_probe_media(path: Path, log_path: Path | None = None) -> MediaMetadata:
        _ = log_path
        return MediaMetadata(
            path=path,
            format_name="wav",
            duration_seconds=1.0,
            audio_streams=[
                AudioStreamMetadata(
                    index=0,
                    codec_name="pcm_s16le",
                    sample_rate=48000,
                    channels=2,
                    duration_seconds=1.0,
                )
            ],
        )

    monkeypatch.setattr("openphonic.pipeline.runner.probe_media", fake_probe_media)
    monkeypatch.setattr("openphonic.pipeline.runner.IngestStage", FakeIngestStage)
    monkeypatch.setattr("openphonic.pipeline.runner.TranscriptionStage", FailingTranscriptionStage)

    config = PipelineConfig(
        name="test",
        stages={
            "silence_trim": {"enabled": False},
            "loudness": {"enabled": False},
            "transcription": {"enabled": True},
        },
    )

    with pytest.raises(RuntimeError, match="transcription unavailable"):
        PipelineRunner(config).run(input_path, work_dir)

    manifest_path = work_dir / "pipeline_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["output_path"] == str(work_dir / "01_ingest.wav")
    assert manifest["artifacts"]["media_metadata"] == str(work_dir / "00_media_metadata.json")
    assert manifest["artifacts"]["ingest_wav"] == str(work_dir / "01_ingest.wav")
    assert "cut_suggestions" not in manifest["artifacts"]
    assert manifest["error"] == {
        "type": "RuntimeError",
        "message": "transcription unavailable",
    }


def test_runner_writes_cut_suggestions_from_current_transcript(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"not real audio")
    work_dir = tmp_path / "work"

    def fake_probe_media(path: Path, log_path: Path | None = None) -> MediaMetadata:
        _ = log_path
        return MediaMetadata(
            path=path,
            format_name="wav",
            duration_seconds=1.0,
            audio_streams=[
                AudioStreamMetadata(
                    index=0,
                    codec_name="pcm_s16le",
                    sample_rate=48000,
                    channels=2,
                    duration_seconds=1.0,
                )
            ],
        )

    monkeypatch.setattr("openphonic.pipeline.runner.probe_media", fake_probe_media)
    monkeypatch.setattr("openphonic.pipeline.runner.IngestStage", FakeIngestStage)
    monkeypatch.setattr(
        "openphonic.pipeline.runner.TranscriptionStage",
        SuggestingTranscriptionStage,
    )

    config = PipelineConfig(
        name="test",
        stages={
            "silence_trim": {"enabled": False},
            "loudness": {"enabled": False},
            "transcription": {"enabled": True},
            "filler_removal": {
                "enabled": True,
                "words": ["um", "uh"],
                "min_silence_seconds": 0.75,
            },
        },
    )
    progress: list[tuple[str, int]] = []

    result = PipelineRunner(
        config,
        progress_callback=lambda stage, percent: progress.append((stage, percent)),
    ).run(input_path, work_dir)

    suggestions_path = work_dir / "cut_suggestions.json"
    assert result.output_path == work_dir / "01_ingest.wav"
    assert result.artifacts["cut_suggestions"] == suggestions_path
    assert progress[-2:] == [("cut_suggestions", 92), ("complete", 99)]
    manifest = json.loads((work_dir / "pipeline_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["artifacts"]["cut_suggestions"] == str(suggestions_path)
    suggestions = json.loads(suggestions_path.read_text(encoding="utf-8"))
    assert suggestions["status"] == "not_applied"
    assert suggestions["reason"] == (
        "Suggestions only; manual review is required before any cuts are applied."
    )
    assert suggestions["configured_words"] == ["um", "uh"]
    assert suggestions["suggestion_count"] == 3
    assert suggestions["suggestions"] == [
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
            "confidence": 0.91,
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
        {
            "id": "cut-0003",
            "type": "filler_word",
            "start": 1.8,
            "end": 2.0,
            "duration": 0.2,
            "text": "uh",
            "normalized_text": "uh",
            "segment_index": 1,
            "word_index": 0,
            "reason": "Matched configured filler word.",
            "confidence": 0.89,
        },
    ]


def test_runner_preserves_unavailable_cut_suggestion_artifact(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"not real audio")
    work_dir = tmp_path / "work"

    def fake_probe_media(path: Path, log_path: Path | None = None) -> MediaMetadata:
        _ = log_path
        return MediaMetadata(
            path=path,
            format_name="wav",
            duration_seconds=1.0,
            audio_streams=[
                AudioStreamMetadata(
                    index=0,
                    codec_name="pcm_s16le",
                    sample_rate=48000,
                    channels=2,
                    duration_seconds=1.0,
                )
            ],
        )

    monkeypatch.setattr("openphonic.pipeline.runner.probe_media", fake_probe_media)
    monkeypatch.setattr("openphonic.pipeline.runner.IngestStage", FakeIngestStage)

    config = PipelineConfig(
        name="test",
        stages={
            "silence_trim": {"enabled": False},
            "loudness": {"enabled": False},
            "filler_removal": {"enabled": True, "words": ["um"]},
        },
    )

    with pytest.raises(StageError, match="no transcript artifact"):
        PipelineRunner(config).run(input_path, work_dir)

    suggestions_path = work_dir / "cut_suggestions.json"
    assert suggestions_path.exists()
    manifest = json.loads((work_dir / "pipeline_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["artifacts"]["cut_suggestions"] == str(suggestions_path)
    suggestions = json.loads(suggestions_path.read_text(encoding="utf-8"))
    assert suggestions["status"] == "not_available"
    assert suggestions["suggestion_count"] == 0
    assert "Transcript word timestamps" in suggestions["reason"]
