from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from openphonic.core.logging import configure_logging
from openphonic.core.settings import get_settings
from openphonic.pipeline.config import PipelineConfig
from openphonic.pipeline.ffmpeg import run_command
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


def process_file(args: argparse.Namespace) -> int:
    configure_logging()
    settings = get_settings()
    config_path = Path(args.config or settings.pipeline_config)
    output_path = Path(args.output).expanduser().resolve()
    work_dir = Path(args.work_dir or output_path.with_suffix("")).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig.from_path(config_path)
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


def smoke_test(args: argparse.Namespace) -> int:
    configure_logging()
    settings = get_settings()
    config_path = Path(args.config or settings.pipeline_config)
    config = PipelineConfig.from_path(config_path)
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
    process.add_argument("--config", help="Pipeline config path.")
    process.add_argument("--work-dir", help="Directory for intermediate files.")
    process.set_defaults(func=process_file)

    smoke = subparsers.add_parser(
        "smoke-test",
        help="Generate a tiny local input and run the configured pipeline.",
    )
    smoke.add_argument("--output", help="Processed output path.")
    smoke.add_argument("--config", help="Pipeline config path.")
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

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
