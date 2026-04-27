from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from openphonic.core.logging import configure_logging
from openphonic.core.settings import get_settings
from openphonic.pipeline.config import PipelineConfig
from openphonic.pipeline.runner import PipelineRunner


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


def main() -> int:
    parser = argparse.ArgumentParser(prog="openphonic")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process a single audio file.")
    process.add_argument("input", help="Input audio/video file.")
    process.add_argument("--output", required=True, help="Output media path.")
    process.add_argument("--config", help="Pipeline config path.")
    process.add_argument("--work-dir", help="Directory for intermediate files.")
    process.set_defaults(func=process_file)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
