import json
import shutil
import subprocess
from pathlib import Path

import pytest

from openphonic.pipeline.config import PipelineConfig, TargetFormat
from openphonic.pipeline.ffmpeg import (
    build_apply_cuts_command,
    build_intro_outro_command,
    probe_media,
    run_command,
)
from openphonic.pipeline.runner import PipelineRunner

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and FFprobe are required for the real-media smoke test.",
)


def _make_sine_wave(path: Path, *, duration: float = 1.0, frequency: int = 1000) -> None:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:duration={duration}",
            "-ar",
            "44100",
            "-ac",
            "1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_default_pipeline_processes_real_audio_and_writes_trace_artifacts(tmp_path) -> None:
    input_path = tmp_path / "input.wav"
    work_dir = tmp_path / "work"
    command_log_path = work_dir / "commands.jsonl"
    _make_sine_wave(input_path)

    config = PipelineConfig.from_path("config/default.yml")
    progress: list[tuple[str, int]] = []

    result = PipelineRunner(
        config,
        progress_callback=lambda stage, percent: progress.append((stage, percent)),
        command_log_path=command_log_path,
    ).run(input_path, work_dir)

    assert result.output_path.exists()
    assert result.output_path.stat().st_size > 0
    assert result.output_path.suffix == ".m4a"
    assert progress[-1] == ("complete", 99)

    expected_artifacts = {
        "media_metadata",
        "ingest_wav",
        "silence_trimmed_wav",
        "loudness_normalized_audio",
        "final_audio",
        "pipeline_manifest",
    }
    assert expected_artifacts.issubset(result.artifacts)
    for artifact_name in expected_artifacts:
        assert result.artifacts[artifact_name].exists()
        assert result.artifacts[artifact_name].stat().st_size > 0

    metadata = json.loads(result.artifacts["media_metadata"].read_text(encoding="utf-8"))
    assert metadata["audio_streams"][0]["sample_rate"] == 44100
    assert metadata["audio_streams"][0]["channels"] == 1

    manifest = json.loads(result.artifacts["pipeline_manifest"].read_text(encoding="utf-8"))
    assert manifest["pipeline_name"] == "podcast-default"
    assert manifest["input_path"] == str(input_path)
    assert manifest["output_path"] == str(result.output_path)
    assert manifest["artifacts"]["final_audio"] == str(result.output_path)
    assert manifest["target"]["sample_rate"] == 48000

    command_events = [
        json.loads(line)
        for line in command_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    started = [event for event in command_events if event["event"] == "process.started"]
    assert [event["executable"] for event in started] == [
        "ffprobe",
        "ffmpeg",
        "ffmpeg",
        "ffmpeg",
        "ffmpeg",
    ]
    assert all("argv" in event for event in started)


def test_apply_cuts_command_processes_real_audio(tmp_path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "reviewed.wav"
    _make_sine_wave(input_path)

    run_command(
        build_apply_cuts_command(
            input_path,
            output_path,
            cut_ranges=[(0.25, 0.75)],
            target=TargetFormat(codec="pcm_s16le", container="wav"),
        )
    )

    input_metadata = probe_media(input_path)
    output_metadata = probe_media(output_path)
    assert output_path.stat().st_size > 0
    assert input_metadata.duration_seconds is not None
    assert output_metadata.duration_seconds is not None
    assert output_metadata.duration_seconds < input_metadata.duration_seconds - 0.3
    assert output_metadata.duration_seconds > 0.3


def test_intro_outro_command_processes_real_audio(tmp_path) -> None:
    intro_path = tmp_path / "intro.wav"
    input_path = tmp_path / "input.wav"
    outro_path = tmp_path / "outro.wav"
    output_path = tmp_path / "branded.wav"
    _make_sine_wave(intro_path, duration=0.2, frequency=440)
    _make_sine_wave(input_path, duration=0.5, frequency=880)
    _make_sine_wave(outro_path, duration=0.3, frequency=660)

    run_command(
        build_intro_outro_command(
            input_path,
            output_path,
            target=TargetFormat(codec="pcm_s16le", container="wav"),
            intro_path=intro_path,
            outro_path=outro_path,
        )
    )

    output_metadata = probe_media(output_path)
    assert output_path.stat().st_size > 0
    assert output_metadata.duration_seconds is not None
    assert 0.95 <= output_metadata.duration_seconds <= 1.05
    assert output_metadata.audio_streams[0].sample_rate == 48000
    assert output_metadata.audio_streams[0].channels == 2
