from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict, deque
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


def _valid_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _finite_timestamp(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    return _finite_float(value)


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
    except UnicodeDecodeError as exc:
        print(f"Diarization inspection failed: invalid UTF-8: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Diarization inspection failed: invalid JSON: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
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


def _merged_range_duration(ranges: list[tuple[float, float]]) -> float:
    if not ranges:
        return 0.0

    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return sum(end - start for start, end in merged)


def _inspect_cut_suggestions(
    cut_suggestions: dict[str, Any],
    *,
    duration_bound: float | None = None,
) -> tuple[dict[str, Any], list[str]]:
    suggestions = cut_suggestions.get("suggestions")
    if not isinstance(suggestions, list):
        raise ValueError("Cut suggestions must include a suggestions list.")

    warnings: list[str] = []
    type_counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    duplicate_ids = 0
    missing_ids = 0
    missing_types = 0
    invalid_timing = 0
    duration_mismatches = 0
    timed_suggestions = 0
    suggested_ranges: list[tuple[float, float]] = []
    suggested_duration = 0.0

    status = cut_suggestions.get("status")
    if not isinstance(status, str) or not status:
        warnings.append("Cut suggestions status must be a non-empty string.")
    elif status != "not_applied":
        warnings.append(f"Cut suggestions are not ready for review: {status}.")

    declared_suggestion_count = _valid_non_negative_int(cut_suggestions.get("suggestion_count"))
    if "suggestion_count" not in cut_suggestions or declared_suggestion_count is None:
        warnings.append("Cut suggestions suggestion_count must be a non-negative integer.")
    if declared_suggestion_count is not None and declared_suggestion_count != len(suggestions):
        warnings.append(
            "Cut suggestions suggestion_count does not match the suggestions list "
            f"({declared_suggestion_count} != {len(suggestions)})."
        )

    configured_words = cut_suggestions.get("configured_words")
    if not isinstance(configured_words, list) or any(
        not isinstance(word, str) or not word for word in configured_words
    ):
        warnings.append("Cut suggestions configured_words must be a list of non-empty strings.")
        configured_words = []

    min_silence_seconds = _finite_timestamp(cut_suggestions.get("min_silence_seconds"))
    if "min_silence_seconds" in cut_suggestions and (
        min_silence_seconds is None or min_silence_seconds < 0
    ):
        warnings.append("Cut suggestions min_silence_seconds must be non-negative.")
        min_silence_seconds = None

    for suggestion_index, suggestion in enumerate(suggestions):
        if not isinstance(suggestion, dict):
            warnings.append(f"Suggestion {suggestion_index} is not an object.")
            invalid_timing += 1
            missing_ids += 1
            missing_types += 1
            continue

        suggestion_id = suggestion.get("id")
        if isinstance(suggestion_id, str) and suggestion_id:
            if suggestion_id in seen_ids:
                duplicate_ids += 1
            seen_ids.add(suggestion_id)
        else:
            missing_ids += 1

        suggestion_type = suggestion.get("type")
        if isinstance(suggestion_type, str) and suggestion_type:
            type_counts[suggestion_type] = type_counts.get(suggestion_type, 0) + 1
        else:
            missing_types += 1

        start = _finite_timestamp(suggestion.get("start"))
        end = _finite_timestamp(suggestion.get("end"))
        duration = _finite_timestamp(suggestion.get("duration"))
        timing_valid = (
            start is not None
            and end is not None
            and duration is not None
            and start >= 0
            and end >= 0
            and duration >= 0
            and end >= start
            and (duration_bound is None or end <= duration_bound)
        )
        if not timing_valid:
            invalid_timing += 1
            continue

        expected_duration = end - start
        if abs(duration - expected_duration) > 0.01:
            duration_mismatches += 1
            continue

        timed_suggestions += 1
        suggested_duration += duration
        suggested_ranges.append((start, end))

    if missing_ids:
        warnings.append(f"{missing_ids} cut suggestion(s) have no id.")
    if duplicate_ids:
        warnings.append(f"{duplicate_ids} cut suggestion id(s) are duplicated.")
    if missing_types:
        warnings.append(f"{missing_types} cut suggestion(s) have no type.")
    if invalid_timing:
        warnings.append(f"{invalid_timing} cut suggestion timing value(s) are invalid.")
    if duration_mismatches:
        warnings.append(
            f"{duration_mismatches} cut suggestion duration value(s) do not match start/end."
        )

    return (
        {
            "status": status if isinstance(status, str) and status else "-",
            "reason": cut_suggestions.get("reason") or "-",
            "source_artifact": cut_suggestions.get("source_artifact") or "-",
            "configured_words": configured_words,
            "min_silence_seconds": min_silence_seconds,
            "suggestion_count": len(suggestions),
            "declared_suggestion_count": declared_suggestion_count,
            "timed_suggestions": timed_suggestions,
            "type_counts": type_counts,
            "suggested_duration": suggested_duration,
            "merged_cut_duration": _merged_range_duration(suggested_ranges),
        },
        warnings,
    )


def inspect_cut_suggestions(args: argparse.Namespace) -> int:
    configure_logging()
    suggestions_path = Path(args.cut_suggestions).expanduser().resolve()
    try:
        cut_suggestions = json.loads(suggestions_path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"Cut suggestion inspection failed: {exc}", file=sys.stderr)
        return 2
    except UnicodeDecodeError as exc:
        print(f"Cut suggestion inspection failed: invalid UTF-8: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Cut suggestion inspection failed: invalid JSON: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Cut suggestion inspection failed: invalid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(cut_suggestions, dict):
        print(
            "Cut suggestion inspection failed: cut suggestions must be a JSON object.",
            file=sys.stderr,
        )
        return 2
    try:
        summary, warnings = _inspect_cut_suggestions(
            cut_suggestions,
            duration_bound=getattr(args, "duration", None),
        )
    except ValueError as exc:
        print(f"Cut suggestion inspection failed: {exc}", file=sys.stderr)
        return 2

    print(f"Cut suggestions: {suggestions_path}")
    print(f"Status: {summary['status']}")
    print(f"Reason: {summary['reason']}")
    print(f"Source artifact: {summary['source_artifact']}")
    configured_words = summary["configured_words"]
    print(f"Configured words: {', '.join(configured_words) if configured_words else '-'}")
    min_silence_seconds = summary["min_silence_seconds"]
    print(
        f"Minimum silence: {min_silence_seconds:.3f}s"
        if min_silence_seconds is not None
        else "Minimum silence: -"
    )
    print(f"Suggestions: {summary['suggestion_count']}")
    print(f"Timed suggestions: {summary['timed_suggestions']}/{summary['suggestion_count']}")
    type_counts = summary["type_counts"]
    type_summary = ", ".join(
        f"{suggestion_type}={count}" for suggestion_type, count in sorted(type_counts.items())
    )
    print(f"Types: {type_summary if type_summary else '-'}")
    print(f"Suggested duration: {summary['suggested_duration']:.3f}s")
    print(f"Merged cut duration: {summary['merged_cut_duration']:.3f}s")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if args.strict and warnings:
        return 2
    return 0


def _path_exists(path: Path, *, expected: str) -> bool:
    if expected == "file":
        return path.is_file()
    if expected == "dir":
        return path.is_dir()
    return path.exists()


def _path_status(
    path_value: Any,
    *,
    expected: str = "any",
    base_dir: Path | None = None,
    fallback_dir: Path | None = None,
) -> tuple[str, Path | None]:
    if not isinstance(path_value, str) or not path_value:
        return "missing", None
    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        candidates = []
        if base_dir is not None:
            candidates.append(base_dir / raw_path)
        if fallback_dir is not None:
            candidates.append(fallback_dir / raw_path)
        if base_dir is None and raw_path not in candidates:
            candidates.append(raw_path)

    for candidate in candidates:
        if _path_exists(candidate, expected=expected):
            return "ok", candidate
    return "missing", candidates[0]


def _relative_manifest_base(work_dir_value: Any, *, inspected_work_dir: Path) -> Path | None:
    if not isinstance(work_dir_value, str) or not work_dir_value:
        return None
    manifest_work_dir = Path(work_dir_value).expanduser()
    if manifest_work_dir.is_absolute():
        return None

    parts = manifest_work_dir.parts
    if not parts or parts == (".",):
        return inspected_work_dir
    if (
        len(inspected_work_dir.parts) >= len(parts)
        and inspected_work_dir.parts[-len(parts) :] == parts
    ):
        return inspected_work_dir.parents[len(parts) - 1]
    return None


def _load_job_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read manifest: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"manifest is not valid UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is invalid JSON: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"manifest is invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object.")
    return manifest


def _inspect_job_manifest(
    manifest: dict[str, Any],
    *,
    work_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []

    status = manifest.get("status")
    if not isinstance(status, str) or not status:
        warnings.append("Pipeline manifest status must be a non-empty string.")
        status = "-"

    pipeline_name = manifest.get("pipeline_name")
    if not isinstance(pipeline_name, str) or not pipeline_name:
        pipeline_name = "-"

    created_at = manifest.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        created_at = "-"

    manifest_work_dir_value = manifest.get("work_dir")
    relative_base = _relative_manifest_base(manifest_work_dir_value, inspected_work_dir=work_dir)
    manifest_work_dir_status, manifest_work_dir = _path_status(
        manifest_work_dir_value,
        expected="dir",
        base_dir=relative_base,
    )
    if manifest_work_dir is None:
        warnings.append("Pipeline manifest work_dir must be a non-empty path.")
    elif manifest_work_dir.resolve() != work_dir:
        warnings.append(
            f"Pipeline manifest work_dir does not match inspected directory: {work_dir}"
        )

    input_status, input_path = _path_status(
        manifest.get("input_path"),
        expected="file",
        base_dir=relative_base,
        fallback_dir=work_dir,
    )
    if input_path is None:
        warnings.append("Pipeline manifest input_path must be a non-empty path.")
    elif input_status != "ok":
        warnings.append(f"Pipeline input is missing: {input_path}")

    output_status, output_path = _path_status(
        manifest.get("output_path"),
        expected="file",
        base_dir=relative_base,
        fallback_dir=work_dir,
    )
    if status == "succeeded" and output_path is None:
        warnings.append("Succeeded pipeline manifest must include output_path.")
    elif output_path is not None and output_status != "ok":
        warnings.append(f"Pipeline output is missing: {output_path}")

    artifacts = manifest.get("artifacts")
    artifact_rows: list[dict[str, Any]] = []
    if not isinstance(artifacts, dict):
        warnings.append("Pipeline manifest artifacts must be a mapping.")
        artifacts = {}

    for name, path_value in sorted(artifacts.items()):
        if not isinstance(name, str) or not name:
            warnings.append("Pipeline manifest contains an artifact with an invalid name.")
            continue
        path_status, path = _path_status(
            path_value,
            expected="file",
            base_dir=relative_base,
            fallback_dir=work_dir,
        )
        if path is None:
            warnings.append(f"Pipeline artifact {name} must be a non-empty path.")
            artifact_rows.append({"name": name, "path": "-", "status": "missing"})
            continue
        if path_status != "ok":
            warnings.append(f"Pipeline artifact {name} is missing: {path}")
        artifact_rows.append({"name": name, "path": path, "status": path_status})

    existing_artifacts = sum(1 for artifact in artifact_rows if artifact["status"] == "ok")
    return (
        {
            "status": status,
            "pipeline_name": pipeline_name,
            "created_at": created_at,
            "work_dir": work_dir,
            "manifest_work_dir": manifest_work_dir,
            "manifest_work_dir_status": manifest_work_dir_status,
            "input_path": input_path,
            "input_status": input_status,
            "output_path": output_path,
            "output_status": output_status,
            "artifact_rows": artifact_rows,
            "artifact_count": len(artifact_rows),
            "existing_artifacts": existing_artifacts,
        },
        warnings,
    )


def inspect_job(args: argparse.Namespace) -> int:
    configure_logging()
    work_dir = Path(args.work_dir).expanduser().resolve()
    if not work_dir.is_dir():
        print(f"Job inspection failed: work directory does not exist: {work_dir}", file=sys.stderr)
        return 2

    manifest_path = work_dir / "pipeline_manifest.json"
    if not manifest_path.is_file():
        print(
            f"Job inspection failed: pipeline manifest not found: {manifest_path}", file=sys.stderr
        )
        return 2

    try:
        manifest = _load_job_manifest(manifest_path)
        summary, warnings = _inspect_job_manifest(manifest, work_dir=work_dir)
    except ValueError as exc:
        print(f"Job inspection failed: {exc}", file=sys.stderr)
        return 2

    print(f"Job work directory: {summary['work_dir']}")
    print(f"Manifest: {manifest_path}")
    print(f"Status: {summary['status']}")
    print(f"Pipeline: {summary['pipeline_name']}")
    print(f"Created: {summary['created_at']}")
    input_path = summary["input_path"]
    print(f"Input: {input_path if input_path is not None else '-'} [{summary['input_status']}]")
    output_path = summary["output_path"]
    print(f"Output: {output_path if output_path is not None else '-'} [{summary['output_status']}]")
    print(f"Artifacts: {summary['existing_artifacts']}/{summary['artifact_count']} present")
    for artifact in summary["artifact_rows"]:
        print(f"  - {artifact['name']}: {artifact['path']} [{artifact['status']}]")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if args.strict and warnings:
        return 2
    return 0


def _valid_returncode(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _command_signature(row: dict[str, Any]) -> tuple[str, tuple[str, ...], str]:
    executable = row.get("executable")
    executable_name = executable if isinstance(executable, str) and executable else "-"

    argv_value = row.get("argv")
    argv = (
        tuple(argv_value)
        if isinstance(argv_value, list) and all(isinstance(arg, str) for arg in argv_value)
        else ()
    )

    cwd_value = row.get("cwd")
    cwd = cwd_value if isinstance(cwd_value, str) else ""
    return executable_name, argv, cwd


def _inspect_command_log(command_log_path: Path) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    event_counts: Counter[str] = Counter()
    executable_counts: Counter[str] = Counter()
    open_starts: defaultdict[tuple[str, tuple[str, ...], str], deque[int]] = defaultdict(deque)
    failure_rows: list[dict[str, Any]] = []
    entry_count = 0
    malformed_entries = 0
    total_duration_seconds = 0.0

    try:
        with command_log_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    malformed_entries += 1
                    warnings.append(f"Line {line_number} is blank.")
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    malformed_entries += 1
                    warnings.append(f"Line {line_number} is invalid JSON: {exc}")
                    continue
                except ValueError as exc:
                    malformed_entries += 1
                    warnings.append(f"Line {line_number} is invalid JSON: {exc}")
                    continue
                if not isinstance(row, dict):
                    malformed_entries += 1
                    warnings.append(f"Line {line_number} must be a JSON object.")
                    continue

                entry_count += 1
                event = row.get("event")
                if not isinstance(event, str) or not event:
                    warnings.append(f"Line {line_number} event must be a non-empty string.")
                    continue
                event_counts[event] += 1

                executable = row.get("executable")
                executable_name = executable if isinstance(executable, str) and executable else "-"
                command_signature = _command_signature(row)
                if event == "process.started":
                    if executable_name == "-":
                        warnings.append(f"Line {line_number} process.started has no executable.")
                    else:
                        executable_counts[executable_name] += 1
                    open_starts[command_signature].append(line_number)

                if event not in {"process.succeeded", "process.failed"}:
                    continue
                if open_starts[command_signature]:
                    open_starts[command_signature].popleft()
                else:
                    warnings.append(
                        f"Line {line_number} {event} has no matching process.started event."
                    )

                returncode = _valid_returncode(row.get("returncode"))
                if returncode is None:
                    warnings.append(f"Line {line_number} {event} has invalid returncode.")
                elif event == "process.succeeded" and returncode != 0:
                    warnings.append(
                        f"Line {line_number} process.succeeded recorded returncode {returncode}."
                    )
                elif event == "process.failed":
                    failure_rows.append(
                        {
                            "line_number": line_number,
                            "executable": executable_name,
                            "returncode": returncode,
                        }
                    )
                    warnings.append(
                        f"Line {line_number} process.failed recorded returncode {returncode}."
                    )

                duration_ms = _finite_timestamp(row.get("duration_ms"))
                if duration_ms is None or duration_ms < 0:
                    warnings.append(f"Line {line_number} {event} has invalid duration_ms.")
                else:
                    total_duration_seconds += duration_ms / 1000.0
    except OSError as exc:
        raise ValueError(f"could not read command log: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8: {exc}") from exc

    if entry_count == 0:
        warnings.append("Command log has no entries.")
    unterminated = sum(len(starts) for starts in open_starts.values())
    if unterminated:
        warnings.append(f"{unterminated} command start(s) have no matching terminal event.")

    return (
        {
            "entries": entry_count,
            "malformed_entries": malformed_entries,
            "started": event_counts["process.started"],
            "succeeded": event_counts["process.succeeded"],
            "failed": event_counts["process.failed"],
            "unterminated": unterminated,
            "executables": executable_counts,
            "failure_rows": failure_rows,
            "total_duration_seconds": total_duration_seconds,
        },
        warnings,
    )


def inspect_commands(args: argparse.Namespace) -> int:
    configure_logging()
    command_log_path = Path(args.command_log).expanduser().resolve()
    try:
        summary, warnings = _inspect_command_log(command_log_path)
    except ValueError as exc:
        print(f"Command log inspection failed: {exc}", file=sys.stderr)
        return 2

    print(f"Command log: {command_log_path}")
    print(f"Entries: {summary['entries']}")
    print(f"Started: {summary['started']}")
    print(f"Succeeded: {summary['succeeded']}")
    print(f"Failed: {summary['failed']}")
    print(f"Unterminated: {summary['unterminated']}")
    print(f"Malformed entries: {summary['malformed_entries']}")
    executable_summary = ", ".join(
        f"{executable}={count}" for executable, count in sorted(summary["executables"].items())
    )
    print(f"Executables: {executable_summary if executable_summary else '-'}")
    print(f"Total duration: {summary['total_duration_seconds']:.3f}s")
    if summary["failure_rows"]:
        print("Failures:")
        for failure in summary["failure_rows"]:
            print(
                f"  - line {failure['line_number']}: "
                f"{failure['executable']} returncode={failure['returncode']}"
            )
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

    cut_suggestions = subparsers.add_parser(
        "inspect-cut-suggestions",
        help="Summarize cut suggestion artifact timing and review readiness.",
    )
    cut_suggestions.add_argument("cut_suggestions", help="Path to cut_suggestions.json.")
    cut_suggestions.add_argument(
        "--duration",
        type=_non_negative_float,
        help="Optional source audio duration in seconds for timing bounds.",
    )
    cut_suggestions.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when cut suggestion warnings are present.",
    )
    cut_suggestions.set_defaults(func=inspect_cut_suggestions)

    job = subparsers.add_parser(
        "inspect-job",
        help="Summarize a pipeline work directory and verify manifest artifacts.",
    )
    job.add_argument("work_dir", help="Pipeline work directory containing pipeline_manifest.json.")
    job.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when job inspection warnings are present.",
    )
    job.set_defaults(func=inspect_job)

    command_log = subparsers.add_parser(
        "inspect-commands",
        help="Summarize a commands.jsonl process log and surface command failures.",
    )
    command_log.add_argument("command_log", help="Path to commands.jsonl.")
    command_log.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when command log warnings are present.",
    )
    command_log.set_defaults(func=inspect_commands)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
