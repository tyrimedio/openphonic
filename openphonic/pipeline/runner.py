from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from openphonic.pipeline.config import PipelineConfig
from openphonic.pipeline.ffmpeg import probe_media
from openphonic.pipeline.stages import (
    DeepFilterNetStage,
    DemucsStage,
    DiarizationStage,
    FillerRemovalStage,
    IngestStage,
    LoudnessStage,
    SilenceTrimStage,
    TranscriptionStage,
)

ProgressCallback = Callable[[str, int], None]


@dataclass
class PipelineResult:
    output_path: Path
    artifacts: dict[str, Path] = field(default_factory=dict)


class PipelineRunner:
    def __init__(
        self,
        config: PipelineConfig,
        progress_callback: ProgressCallback | None = None,
        command_log_path: Path | None = None,
    ) -> None:
        self.config = config
        self.progress_callback = progress_callback
        self.command_log_path = command_log_path

    def _progress(self, stage: str, progress: int) -> None:
        if self.progress_callback:
            self.progress_callback(stage, progress)

    def run(self, input_path: Path, work_dir: Path) -> PipelineResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Path] = {}

        self._progress("metadata", 8)
        metadata = probe_media(input_path, log_path=self.command_log_path)
        metadata_path = work_dir / "00_media_metadata.json"
        metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
        artifacts["media_metadata"] = metadata_path

        self._progress("ingest", 10)
        current = IngestStage(self.config, self.command_log_path).run(input_path, work_dir)

        if self.config.enabled("noise_reduction"):
            self._progress("noise_reduction", 25)
            current = DeepFilterNetStage(self.config, self.command_log_path).run(current, work_dir)

        if self.config.enabled("music_separation"):
            self._progress("music_separation", 35)
            current = DemucsStage(self.config, self.command_log_path).run(current, work_dir)

        if self.config.enabled("silence_trim", default=True):
            self._progress("silence_trim", 50)
            current = SilenceTrimStage(self.config, self.command_log_path).run(current, work_dir)

        if self.config.enabled("filler_removal"):
            self._progress("filler_removal", 60)
            current = FillerRemovalStage(self.config, self.command_log_path).run(current, work_dir)

        if self.config.enabled("loudness", default=True):
            self._progress("loudness", 75)
            current = LoudnessStage(self.config, self.command_log_path).run(current, work_dir)

        if self.config.enabled("transcription"):
            self._progress("transcription", 88)
            artifacts.update(
                TranscriptionStage(self.config, self.command_log_path).run(current, work_dir)
            )

        if self.config.enabled("diarization"):
            self._progress("diarization", 94)
            artifacts.update(
                DiarizationStage(self.config, self.command_log_path).run(current, work_dir)
            )

        self._progress("complete", 99)
        return PipelineResult(output_path=current, artifacts=artifacts)
