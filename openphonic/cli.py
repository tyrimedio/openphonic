from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from openphonic.core.logging import configure_logging
from openphonic.core.settings import Settings, get_settings
from openphonic.pipeline.config import (
    CUSTOM_PRESET_ID,
    PipelineConfig,
    PipelinePreset,
    available_presets,
    load_pipeline_config_for_preset,
    preset_by_id,
)
from openphonic.pipeline.ffmpeg import run_command
from openphonic.pipeline.preflight import format_preflight_issues, pipeline_preflight_issues
from openphonic.pipeline.runner import PipelineRunner


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("must be finite and non-negative") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _generate_smoke_input(input_path: Path, *, duration: float, frequency: int) -> None:
    input_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
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
            str(input_path),
        ]
    )


def _configured_output_suffix(config: PipelineConfig) -> str:
    container = config.target.container.strip().lstrip(".")
    return f".{container}" if container else ".m4a"


def _load_pipeline_config(args: argparse.Namespace, settings: Settings) -> PipelineConfig:
    if getattr(args, "preset", None):
        return load_pipeline_config_for_preset(
            args.preset,
            default_path=settings.pipeline_config,
            preset_dir=settings.preset_dir,
        )
    return PipelineConfig.from_path(Path(getattr(args, "config", None) or settings.pipeline_config))


def _core_media_messages() -> list[str]:
    return [
        f"{executable} is required but was not found on PATH."
        for executable in ("ffmpeg", "ffprobe")
        if shutil.which(executable) is None
    ]


def _raw_config_schema_messages(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return ["Preset config must be a mapping."]

    stages = raw.get("stages") if "stages" in raw else {}
    if not isinstance(stages, dict):
        return ["Preset stages must be a mapping."]
    return [
        f"Preset stage {stage_name!r} must be a mapping."
        for stage_name, stage_config in stages.items()
        if not isinstance(stage_config, dict)
    ]


def _readiness_messages(preset: PipelinePreset, settings: Settings) -> list[str]:
    messages = _core_media_messages()
    try:
        schema_messages = _raw_config_schema_messages(preset.path)
        if schema_messages:
            messages.extend(schema_messages)
            return messages
        config = PipelineConfig.from_path(preset.path)
        messages.extend(issue.message for issue in pipeline_preflight_issues(config, settings))
    except (OSError, TypeError, ValueError, AttributeError, yaml.YAMLError) as exc:
        messages.append(f"Preset config could not be inspected: {exc}")
    return messages


def _custom_readiness_preset(preset_id: str, preset_dir: Path) -> PipelinePreset | None:
    prefix = "custom:"
    if not preset_id.startswith(prefix):
        return None
    stem = preset_id.removeprefix(prefix)
    if not CUSTOM_PRESET_ID.fullmatch(stem):
        return None

    directory = Path(preset_dir).expanduser()
    for path in (directory / f"{stem}.yml", directory / f"{stem}.yaml"):
        if path.is_file():
            label = stem.replace("_", " ").replace("-", " ").strip().title() or stem
            return PipelinePreset(
                id=preset_id,
                label=label,
                description=f"Custom preset from {path.name}.",
                path=path,
            )
    return None


def _readiness_preset_by_id(preset_id: str, settings: Settings) -> PipelinePreset:
    try:
        return preset_by_id(
            preset_id,
            default_path=settings.pipeline_config,
            preset_dir=settings.preset_dir,
        )
    except ValueError:
        if custom_preset := _custom_readiness_preset(preset_id, settings.preset_dir):
            return custom_preset
        raise


def _readiness_presets(args: argparse.Namespace, settings: Settings) -> list[PipelinePreset]:
    requested = getattr(args, "preset", None)
    if not requested:
        return available_presets(settings.pipeline_config, settings.preset_dir)
    return [_readiness_preset_by_id(preset_id, settings) for preset_id in requested]


def process_file(args: argparse.Namespace) -> int:
    configure_logging()
    settings = get_settings()
    try:
        config = _load_pipeline_config(args, settings)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Process config failed: {exc}", file=sys.stderr)
        return 2
    try:
        preflight_issues = pipeline_preflight_issues(config, settings)
    except (TypeError, ValueError, AttributeError) as exc:
        print(f"Process preflight failed: {exc}", file=sys.stderr)
        return 2
    if preflight_issues:
        print(
            f"Process preflight failed: {format_preflight_issues(preflight_issues)}",
            file=sys.stderr,
        )
        return 2

    output_path = Path(args.output).expanduser().resolve()
    work_dir = Path(args.work_dir or output_path.with_suffix("")).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    result = PipelineRunner(config, command_log_path=work_dir / "commands.jsonl").run(
        Path(args.input).expanduser().resolve(),
        work_dir,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result.output_path, output_path)
    print(f"Processed audio: {output_path}")
    for name, path in result.artifacts.items():
        print(f"{name}: {path}")
    return 0


def readiness(args: argparse.Namespace) -> int:
    configure_logging()
    settings = get_settings()
    blocked = 0
    try:
        presets = _readiness_presets(args, settings)
    except ValueError as exc:
        print(f"Readiness config failed: {exc}", file=sys.stderr)
        return 2

    for preset in presets:
        messages = _readiness_messages(preset, settings)
        if messages:
            blocked += 1
            print(f"[blocked] {preset.id} - {preset.label}")
            for message in messages:
                print(f"  - {message}")
        else:
            print(f"[ready] {preset.id} - {preset.label}")

    if args.strict and blocked:
        return 2
    return 0


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _inspect_transcript(transcript: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Transcript segments must be a list.")

    warnings: list[str] = []
    segment_count = 0
    segments_with_words = 0
    wordless_segments = 0
    word_count = 0
    timed_words = 0
    invalid_timing = 0
    duration = _finite_float(transcript.get("duration"))
    if "duration" in transcript and (duration is None or duration < 0):
        invalid_timing += 1
    duration_bound = duration if duration is not None and duration >= 0 else None

    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            warnings.append(f"Segment {segment_index} is not an object.")
            continue
        segment_count += 1
        start = _finite_float(segment.get("start"))
        end = _finite_float(segment.get("end"))
        segment_timing_valid = (
            start is not None and end is not None and start >= 0 and end >= 0 and end >= start
        )
        segment_within_duration = duration_bound is None or (
            segment_timing_valid and end is not None and end <= duration_bound
        )
        if not segment_timing_valid or not segment_within_duration:
            invalid_timing += 1

        words = segment.get("words") or []
        if not isinstance(words, list):
            warnings.append(f"Segment {segment_index} words must be a list.")
            words = []
        if words:
            segments_with_words += 1
        else:
            wordless_segments += 1
        for word in words:
            if not isinstance(word, dict):
                warnings.append(f"Segment {segment_index} contains a non-object word.")
                continue
            word_count += 1
            word_start = _finite_float(word.get("start"))
            word_end = _finite_float(word.get("end"))
            word_timing_valid = (
                word_start is not None
                and word_end is not None
                and word_start >= 0
                and word_end >= 0
                and word_end >= word_start
                and (duration_bound is None or word_end <= duration_bound)
                and (
                    not segment_timing_valid
                    or (
                        start is not None
                        and end is not None
                        and word_start >= start
                        and word_end <= end
                    )
                )
            )
            if not word_timing_valid:
                invalid_timing += 1
            else:
                timed_words += 1

    if segment_count == 0:
        warnings.append("Transcript has no segments.")
    if word_count == 0 and segment_count > 0:
        warnings.append("Transcript has no word timestamps.")
    elif timed_words < word_count:
        warnings.append("Some transcript words are missing valid timestamps.")
    if wordless_segments:
        warnings.append(f"{wordless_segments} transcript segment(s) have no words.")
    if invalid_timing:
        warnings.append(f"{invalid_timing} transcript timing value(s) are invalid.")

    coverage = (timed_words / word_count * 100.0) if word_count else 0.0
    return (
        {
            "engine": transcript.get("engine") or "-",
            "model": transcript.get("model") or "-",
            "language": transcript.get("language") or "-",
            "duration": duration,
            "segment_count": segment_count,
            "segments_with_words": segments_with_words,
            "word_count": word_count,
            "timed_words": timed_words,
            "word_coverage": coverage,
        },
        warnings,
    )


def inspect_transcript(args: argparse.Namespace) -> int:
    configure_logging()
    transcript_path = Path(args.transcript).expanduser().resolve()
    try:
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"Transcript inspection failed: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Transcript inspection failed: invalid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(transcript, dict):
        print("Transcript inspection failed: transcript must be a JSON object.", file=sys.stderr)
        return 2
    try:
        summary, warnings = _inspect_transcript(transcript)
    except ValueError as exc:
        print(f"Transcript inspection failed: {exc}", file=sys.stderr)
        return 2

    print(f"Transcript: {transcript_path}")
    print(f"Engine: {summary['engine']}")
    print(f"Model: {summary['model']}")
    print(f"Language: {summary['language']}")
    duration = summary["duration"]
    print(f"Duration: {duration:.3f}s" if duration is not None else "Duration: -")
    print(f"Segments: {summary['segment_count']}")
    print(f"Segments with words: {summary['segments_with_words']}/{summary['segment_count']}")
    print(f"Words: {summary['word_count']}")
    print(
        f"Timed words: {summary['timed_words']}/{summary['word_count']} "
        f"({summary['word_coverage']:.1f}%)"
    )
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if args.strict and warnings:
        return 2
    return 0


def _valid_speaker_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _inspect_diarization(
    diarization: dict[str, Any],
    *,
    duration_bound: float | None = None,
) -> tuple[dict[str, Any], list[str]]:
    segments = diarization.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Diarization segments must be a list.")

    warnings: list[str] = []
    speakers: set[str] = set()
    timed_segments = 0
    invalid_timing = 0
    missing_speakers = 0
    total_speaker_time = 0.0

    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            warnings.append(f"Segment {segment_index} is not an object.")
            invalid_timing += 1
            missing_speakers += 1
            continue

        speaker = segment.get("speaker")
        if isinstance(speaker, str) and speaker.strip():
            speakers.add(speaker.strip())
        else:
            missing_speakers += 1

        start = _finite_float(segment.get("start"))
        end = _finite_float(segment.get("end"))
        segment_timing_valid = (
            start is not None and end is not None and start >= 0 and end >= 0 and end >= start
        )
        segment_within_duration = duration_bound is None or (
            segment_timing_valid and end is not None and end <= duration_bound
        )
        if (
            segment_timing_valid
            and segment_within_duration
            and start is not None
            and end is not None
        ):
            timed_segments += 1
            total_speaker_time += end - start
        else:
            invalid_timing += 1

    declared_speaker_count = _valid_speaker_count(diarization.get("speaker_count"))
    if "speaker_count" in diarization and declared_speaker_count is None:
        warnings.append("Diarization speaker_count must be a non-negative integer.")
    if declared_speaker_count is not None and declared_speaker_count != len(speakers):
        warnings.append(
            "Diarization speaker_count does not match detected speaker labels "
            f"({declared_speaker_count} != {len(speakers)})."
        )
    if not segments:
        warnings.append("Diarization has no segments.")
    if missing_speakers:
        warnings.append(f"{missing_speakers} diarization segment(s) have no speaker label.")
    if invalid_timing:
        warnings.append(f"{invalid_timing} diarization segment timing value(s) are invalid.")

    return (
        {
            "engine": diarization.get("engine") or "-",
            "model": diarization.get("model") or "-",
            "declared_speaker_count": declared_speaker_count,
            "detected_speaker_count": len(speakers),
            "segment_count": len(segments),
            "timed_segments": timed_segments,
            "total_speaker_time": total_speaker_time,
        },
        warnings,
    )


def inspect_diarization(args: argparse.Namespace) -> int:
    configure_logging()
    diarization_path = Path(args.diarization).expanduser().resolve()
    try:
        diarization = json.loads(diarization_path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"Diarization inspection failed: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Diarization inspection failed: invalid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(diarization, dict):
        print("Diarization inspection failed: diarization must be a JSON object.", file=sys.stderr)
        return 2
    try:
        summary, warnings = _inspect_diarization(
            diarization,
            duration_bound=getattr(args, "duration", None),
        )
    except ValueError as exc:
        print(f"Diarization inspection failed: {exc}", file=sys.stderr)
        return 2

    print(f"Diarization: {diarization_path}")
    print(f"Engine: {summary['engine']}")
    print(f"Model: {summary['model']}")
    declared_speaker_count = summary["declared_speaker_count"]
    print(
        f"Declared speakers: {declared_speaker_count}"
        if declared_speaker_count is not None
        else "Declared speakers: -"
    )
    print(f"Detected speakers: {summary['detected_speaker_count']}")
    print(f"Segments: {summary['segment_count']}")
    print(f"Timed segments: {summary['timed_segments']}/{summary['segment_count']}")
    print(f"Total speaker time: {summary['total_speaker_time']:.3f}s")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if args.strict and warnings:
        return 2
    return 0


def smoke_test(args: argparse.Namespace) -> int:
    configure_logging()
    settings = get_settings()
    try:
        config = _load_pipeline_config(args, settings)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Smoke test config failed: {exc}", file=sys.stderr)
        return 2
    preflight_issues = pipeline_preflight_issues(config, settings)
    if preflight_issues:
        print(
            f"Smoke test preflight failed: {format_preflight_issues(preflight_issues)}",
            file=sys.stderr,
        )
        return 2
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (
            settings.data_dir / "smoke-test" / f"processed{_configured_output_suffix(config)}"
        ).resolve()
    )
    work_dir = (
        Path(args.work_dir).expanduser().resolve() if args.work_dir else output_path.parent / "work"
    )
    input_path = work_dir / "input.wav"
    command_log_path = work_dir / "commands.jsonl"

    _generate_smoke_input(input_path, duration=args.duration, frequency=args.frequency)
    result = PipelineRunner(config, command_log_path=command_log_path).run(input_path, work_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if result.output_path.resolve() != output_path:
        shutil.copy2(result.output_path, output_path)
    print(f"Smoke test processed audio: {output_path}")
    print(f"Work directory: {work_dir}")
    print(f"Command log: {command_log_path}")
    for name, path in result.artifacts.items():
        print(f"{name}: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="openphonic")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process a single audio file.")
    process.add_argument("input", help="Input audio/video file.")
    process.add_argument("--output", required=True, help="Output media path.")
    process_config = process.add_mutually_exclusive_group()
    process_config.add_argument("--config", help="Pipeline config path.")
    process_config.add_argument("--preset", help="Built-in or custom pipeline preset id.")
    process.add_argument("--work-dir", help="Directory for intermediate files.")
    process.set_defaults(func=process_file)

    smoke = subparsers.add_parser(
        "smoke-test",
        help="Generate a tiny local input and run the configured pipeline.",
    )
    smoke.add_argument("--output", help="Processed output path.")
    smoke_config = smoke.add_mutually_exclusive_group()
    smoke_config.add_argument("--config", help="Pipeline config path.")
    smoke_config.add_argument("--preset", help="Built-in or custom pipeline preset id.")
    smoke.add_argument("--work-dir", help="Directory for generated input and artifacts.")
    smoke.add_argument(
        "--duration",
        type=_positive_float,
        default=1.0,
        help="Generated input duration in seconds.",
    )
    smoke.add_argument(
        "--frequency",
        type=_positive_int,
        default=1000,
        help="Generated sine wave frequency in Hz.",
    )
    smoke.set_defaults(func=smoke_test)

    readiness_check = subparsers.add_parser(
        "readiness",
        help="Report which pipeline presets can run on this host.",
    )
    readiness_check.add_argument(
        "--preset",
        action="append",
        help="Built-in or custom pipeline preset id to inspect. Can be provided more than once.",
    )
    readiness_check.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code if any preset is blocked.",
    )
    readiness_check.set_defaults(func=readiness)

    transcript = subparsers.add_parser(
        "inspect-transcript",
        help="Summarize transcript artifact timing and word timestamp coverage.",
    )
    transcript.add_argument("transcript", help="Path to transcript.json.")
    transcript.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when transcript quality warnings are present.",
    )
    transcript.set_defaults(func=inspect_transcript)

    diarization = subparsers.add_parser(
        "inspect-diarization",
        help="Summarize diarization artifact speaker and timing coverage.",
    )
    diarization.add_argument("diarization", help="Path to diarization.json.")
    diarization.add_argument(
        "--duration",
        type=_non_negative_float,
        help="Optional source audio duration in seconds for timing bounds.",
    )
    diarization.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when diarization quality warnings are present.",
    )
    diarization.set_defaults(func=inspect_diarization)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
