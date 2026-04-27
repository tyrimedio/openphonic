from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openphonic.core.logging import append_event, log_event
from openphonic.pipeline.config import TargetFormat

logger = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    """Raised when FFmpeg-family command execution or parsing fails."""


class MediaValidationError(FFmpegError):
    """Raised when uploaded media is missing required audio metadata."""


@dataclass(frozen=True)
class AudioStreamMetadata:
    index: int
    codec_name: str | None
    sample_rate: int | None
    channels: int | None
    duration_seconds: float | None


@dataclass(frozen=True)
class MediaMetadata:
    path: Path
    format_name: str | None
    duration_seconds: float | None
    audio_streams: list[AudioStreamMetadata]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise FFmpegError(f"{name} is required but was not found on PATH.")


def require_ffmpeg() -> None:
    require_executable("ffmpeg")


def run_command(
    args: list[str],
    cwd: Path | None = None,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if not args:
        raise FFmpegError("Cannot run an empty command.")

    executable = Path(args[0]).name
    require_executable(executable)
    event_fields = {"executable": executable, "argv": args, "cwd": cwd}
    log_event(logger, "process.started", **event_fields)
    if log_path is not None:
        append_event(log_path, "process.started", **event_fields)

    started = time.monotonic()
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    duration_ms = int((time.monotonic() - started) * 1000)

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        failure_fields = {
            **event_fields,
            "returncode": completed.returncode,
            "duration_ms": duration_ms,
            "stderr_tail": completed.stderr[-4000:],
        }
        log_event(logger, "process.failed", level=logging.ERROR, **failure_fields)
        if log_path is not None:
            append_event(log_path, "process.failed", **failure_fields)
        raise FFmpegError(
            f"Command failed ({completed.returncode}): {shlex.join(args)}\n{detail}"
        )

    success_fields = {
        **event_fields,
        "returncode": completed.returncode,
        "duration_ms": duration_ms,
    }
    log_event(logger, "process.succeeded", **success_fields)
    if log_path is not None:
        append_event(log_path, "process.succeeded", **success_fields)
    return completed


def build_ffprobe_command(input_path: Path) -> list[str]:
    return [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(input_path),
    ]


def build_ingest_command(input_path: Path, output_path: Path, target: TargetFormat) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ar",
        str(target.sample_rate),
        "-ac",
        str(target.channels),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def build_silence_trim_command(
    input_path: Path,
    output_path: Path,
    *,
    start_threshold_db: float,
    stop_threshold_db: float,
    min_silence_seconds: float,
) -> list[str]:
    filter_spec = (
        "silenceremove="
        f"start_periods=1:start_duration={min_silence_seconds}:start_threshold={start_threshold_db}dB:"
        f"stop_periods=1:stop_duration={min_silence_seconds}:stop_threshold={stop_threshold_db}dB"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-af",
        filter_spec,
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def loudnorm_filter(stage: dict[str, Any], measured: dict[str, Any] | None = None) -> str:
    integrated = stage.get("integrated_lufs", -16)
    true_peak = stage.get("true_peak_db", -1.5)
    loudness_range = stage.get("loudness_range_lu", 11)
    pieces = [f"I={integrated}", f"TP={true_peak}", f"LRA={loudness_range}"]
    if measured:
        pieces.extend(
            [
                f"measured_I={measured['input_i']}",
                f"measured_TP={measured['input_tp']}",
                f"measured_LRA={measured['input_lra']}",
                f"measured_thresh={measured['input_thresh']}",
                f"offset={measured['target_offset']}",
                "linear=true",
                "print_format=summary",
            ]
        )
    else:
        pieces.append("print_format=json")
    return "loudnorm=" + ":".join(pieces)


def build_loudnorm_probe_command(input_path: Path, stage: dict[str, Any]) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-af",
        loudnorm_filter(stage),
        "-f",
        "null",
        "-",
    ]


def build_loudnorm_apply_command(
    input_path: Path,
    output_path: Path,
    *,
    stage: dict[str, Any],
    target: TargetFormat,
    measured: dict[str, Any],
) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-af",
        loudnorm_filter(stage, measured),
        "-ar",
        str(target.sample_rate),
        "-ac",
        str(target.channels),
        "-c:a",
        target.codec,
        "-b:a",
        target.bitrate,
        str(output_path),
    ]


def _parse_float(value: Any) -> float | None:
    if value in (None, "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> int | None:
    if value in (None, "N/A"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_media_metadata(input_path: Path, stdout: str) -> MediaMetadata:
    try:
        raw = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise MediaValidationError("ffprobe returned invalid JSON.") from exc

    audio_streams = [
        AudioStreamMetadata(
            index=_parse_int(stream.get("index")) or 0,
            codec_name=stream.get("codec_name"),
            sample_rate=_parse_int(stream.get("sample_rate")),
            channels=_parse_int(stream.get("channels")),
            duration_seconds=_parse_float(stream.get("duration")),
        )
        for stream in raw.get("streams", [])
        if stream.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise MediaValidationError("Input media contains no audio streams.")

    media_format = raw.get("format") or {}
    return MediaMetadata(
        path=input_path,
        format_name=media_format.get("format_name"),
        duration_seconds=_parse_float(media_format.get("duration")),
        audio_streams=audio_streams,
    )


def probe_media(input_path: Path, log_path: Path | None = None) -> MediaMetadata:
    if not input_path.exists():
        raise MediaValidationError(f"Input media does not exist: {input_path}")
    completed = run_command(build_ffprobe_command(input_path), log_path=log_path)
    return parse_media_metadata(input_path, completed.stdout)


def parse_loudnorm_json(stderr: str) -> dict[str, Any]:
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise FFmpegError("Could not find loudnorm JSON in FFmpeg output.")
    return json.loads(stderr[start : end + 1])
