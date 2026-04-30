from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

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

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
