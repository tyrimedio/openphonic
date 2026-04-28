import subprocess
from pathlib import Path

import pytest

from openphonic.pipeline.config import PipelineConfig, TargetFormat
from openphonic.pipeline.ffmpeg import (
    FFmpegError,
    MediaValidationError,
    build_apply_cuts_command,
    build_ffprobe_command,
    build_ingest_command,
    build_loudnorm_apply_command,
    build_silence_trim_command,
    parse_loudnorm_json,
    parse_media_metadata,
)
from openphonic.pipeline.stages import DeepFilterNetStage, IngestStage, StageError


def test_build_ingest_command_sets_working_format() -> None:
    command = build_ingest_command(Path("in.mp3"), Path("out.wav"), TargetFormat())

    assert "-ar" in command
    assert "48000" in command
    assert "-c:a" in command
    assert "pcm_s16le" in command


def test_build_ffprobe_command_reads_stream_metadata() -> None:
    command = build_ffprobe_command(Path("input.mp4"))

    assert command[:2] == ["ffprobe", "-v"]
    assert "-show_streams" in command
    assert "-show_format" in command


def test_build_silence_trim_command_is_conservative() -> None:
    command = build_silence_trim_command(
        Path("in.wav"),
        Path("out.wav"),
        start_threshold_db=-50,
        stop_threshold_db=-50,
        min_silence_seconds=0.35,
    )

    filter_index = command.index("-af") + 1
    assert "start_periods=1" in command[filter_index]
    assert "stop_periods=1" in command[filter_index]


def test_parse_loudnorm_json_from_ffmpeg_stderr() -> None:
    stderr = """
    [Parsed_loudnorm_0 @ 0x123] something
    {
      "input_i" : "-19.11",
      "input_tp" : "-2.35",
      "input_lra" : "4.20",
      "input_thresh" : "-29.54",
      "output_i" : "-16.02",
      "output_tp" : "-1.51",
      "output_lra" : "3.90",
      "output_thresh" : "-26.22",
      "normalization_type" : "dynamic",
      "target_offset" : "0.02"
    }
    """

    parsed = parse_loudnorm_json(stderr)

    assert parsed["input_i"] == "-19.11"
    assert parsed["target_offset"] == "0.02"


def test_parse_media_metadata_extracts_audio_streams() -> None:
    metadata = parse_media_metadata(
        Path("input.wav"),
        """
        {
          "streams": [
            {
              "index": 0,
              "codec_type": "audio",
              "codec_name": "pcm_s16le",
              "sample_rate": "48000",
              "channels": 2,
              "duration": "12.5"
            }
          ],
          "format": {
            "format_name": "wav",
            "duration": "12.5"
          }
        }
        """,
    )

    assert metadata.duration_seconds == 12.5
    assert metadata.audio_streams[0].sample_rate == 48000
    assert metadata.audio_streams[0].channels == 2


def test_parse_media_metadata_rejects_files_without_audio() -> None:
    stdout = '{"streams": [{"codec_type": "video"}], "format": {"format_name": "mp4"}}'

    try:
        parse_media_metadata(Path("video.mp4"), stdout)
    except MediaValidationError as exc:
        assert "no audio streams" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Expected MediaValidationError.")


def test_loudnorm_apply_command_uses_measured_values() -> None:
    command = build_loudnorm_apply_command(
        Path("in.wav"),
        Path("out.m4a"),
        stage={"integrated_lufs": -16, "true_peak_db": -1.5, "loudness_range_lu": 11},
        target=TargetFormat(),
        measured={
            "input_i": "-19.11",
            "input_tp": "-2.35",
            "input_lra": "4.20",
            "input_thresh": "-29.54",
            "target_offset": "0.02",
        },
    )

    filter_spec = command[command.index("-af") + 1]
    assert "measured_I=-19.11" in filter_spec
    assert "offset=0.02" in filter_spec


def test_build_apply_cuts_command_removes_approved_ranges() -> None:
    command = build_apply_cuts_command(
        Path("in.m4a"),
        Path("out.m4a"),
        cut_ranges=[(0.0, 0.22), (1.5, 2.0)],
        target=TargetFormat(),
    )

    filter_spec = command[command.index("-af") + 1]
    assert command[:4] == ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    assert "aselect='not(" in filter_spec
    assert "between(t,0,0.22)" in filter_spec
    assert "between(t,1.5,2)" in filter_spec
    assert "asetpts=N/SR/TB" in filter_spec
    assert command[-1] == "out.m4a"


def test_build_apply_cuts_command_rejects_invalid_ranges() -> None:
    with pytest.raises(FFmpegError, match="Invalid cut range"):
        build_apply_cuts_command(
            Path("in.m4a"),
            Path("out.m4a"),
            cut_ranges=[(1.0, 1.0)],
            target=TargetFormat(),
        )


def test_stage_fails_when_command_does_not_produce_artifact(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")

    def fake_run_command(*args, **kwargs) -> None:
        _ = args, kwargs

    monkeypatch.setattr("openphonic.pipeline.stages.run_command", fake_run_command)

    with pytest.raises(StageError, match="Ingest stage did not produce expected artifact"):
        IngestStage(config=PipelineConfig("test")).run(input_path, tmp_path)


def test_deepfilternet_stage_passes_configured_attenuation(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    seen_commands: list[list[str]] = []

    def fake_subprocess_run(args, check, text, capture_output):
        _ = check, text, capture_output
        seen_commands.append(args)
        output_dir = Path(args[args.index("-o") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "enhanced.wav").write_bytes(b"enhanced")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("openphonic.pipeline.stages.shutil.which", lambda binary: f"/bin/{binary}")
    monkeypatch.setattr("openphonic.pipeline.stages.subprocess.run", fake_subprocess_run)

    output_path = DeepFilterNetStage(
        PipelineConfig(
            name="test",
            stages={"noise_reduction": {"enabled": True, "attenuation_db": 8}},
        )
    ).run(input_path, tmp_path)

    assert output_path == tmp_path / "02_noise_reduced.wav"
    assert output_path.read_bytes() == b"enhanced"
    assert seen_commands == [
        [
            "deepFilter",
            "--atten-lim",
            "8",
            str(input_path),
            "-o",
            str(tmp_path / "02_deepfilternet"),
        ]
    ]


def test_deepfilternet_stage_rejects_invalid_attenuation(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    monkeypatch.setattr("openphonic.pipeline.stages.shutil.which", lambda binary: f"/bin/{binary}")

    with pytest.raises(StageError, match="attenuation_db must be greater than zero"):
        DeepFilterNetStage(
            PipelineConfig(
                name="test",
                stages={"noise_reduction": {"enabled": True, "attenuation_db": 0}},
            )
        ).run(input_path, tmp_path)
