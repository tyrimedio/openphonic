import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from openphonic.cli import readiness, smoke_test
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
    monkeypatch.setattr("openphonic.pipeline.preflight._binary_available", lambda binary: False)
    monkeypatch.setattr("openphonic.pipeline.preflight._module_available", lambda module: False)

    result = readiness(argparse.Namespace(strict=True))

    assert result == 2


def test_readiness_lists_custom_presets(
    tmp_settings,
    capsys,
) -> None:
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
