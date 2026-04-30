import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from openphonic.cli import (
    inspect_diarization,
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
