from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from openphonic.core.logging import utc_timestamp
from openphonic.pipeline.config import PipelineConfig
from openphonic.pipeline.ffmpeg import probe_media
from openphonic.pipeline.stages import (
    DeepFilterNetStage,
    DemucsStage,
    DiarizationStage,
    FillerRemovalStage,
    IngestStage,
    IntroOutroStage,
    LoudnessStage,
    SilenceTrimStage,
    StageError,
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

    def _write_manifest(
        self,
        *,
        input_path: Path,
        work_dir: Path,
        output_path: Path | None,
        artifacts: dict[str, Path],
        status: str,
        error: BaseException | None = None,
    ) -> Path:
        manifest_path = work_dir / "pipeline_manifest.json"
        manifest = {
            "schema_version": 1,
            "created_at": utc_timestamp(),
            "status": status,
            "pipeline_name": self.config.name,
            "input_path": str(input_path),
            "work_dir": str(work_dir),
            "output_path": str(output_path) if output_path is not None else None,
            "target": asdict(self.config.target),
            "stages": self.config.stages,
            "artifacts": {name: str(path) for name, path in sorted(artifacts.items())},
        }
        if error is not None:
            manifest["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path

    def run(self, input_path: Path, work_dir: Path) -> PipelineResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Path] = {}
        current: Path | None = None

        try:
            self._progress("metadata", 8)
            metadata = probe_media(input_path, log_path=self.command_log_path)
            metadata_path = work_dir / "00_media_metadata.json"
            metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
            artifacts["media_metadata"] = metadata_path

            self._progress("ingest", 10)
            current = IngestStage(self.config, self.command_log_path).run(input_path, work_dir)
            artifacts["ingest_wav"] = current

            if self.config.enabled("noise_reduction"):
                self._progress("noise_reduction", 25)
                current = DeepFilterNetStage(self.config, self.command_log_path).run(
                    current, work_dir
                )
                artifacts["noise_reduced_wav"] = current

            if self.config.enabled("music_separation"):
                self._progress("music_separation", 35)
                current = DemucsStage(self.config, self.command_log_path).run(current, work_dir)
                artifacts["separated_wav"] = current

            if self.config.enabled("silence_trim", default=True):
                self._progress("silence_trim", 50)
                current = SilenceTrimStage(self.config, self.command_log_path).run(
                    current, work_dir
                )
                artifacts["silence_trimmed_wav"] = current

            if self.config.enabled("intro_outro"):
                self._progress("intro_outro", 62)
                current = IntroOutroStage(self.config, self.command_log_path).run(current, work_dir)
                artifacts["intro_outro_wav"] = current

            if self.config.enabled("loudness", default=True):
                self._progress("loudness", 75)
                current = LoudnessStage(self.config, self.command_log_path).run(current, work_dir)
                artifacts["loudness_normalized_audio"] = current

            if self.config.enabled("transcription"):
                self._progress("transcription", 88)
                artifacts.update(
                    TranscriptionStage(self.config, self.command_log_path).run(current, work_dir)
                )

            if self.config.enabled("filler_removal"):
                self._progress("cut_suggestions", 92)
                artifacts.update(
                    FillerRemovalStage(self.config, self.command_log_path).run(
                        artifacts.get("transcript_json"),
                        work_dir,
                    )
                )

            if self.config.enabled("diarization"):
                self._progress("diarization", 94)
                artifacts.update(
                    DiarizationStage(self.config, self.command_log_path).run(current, work_dir)
                )
        except Exception as exc:
            if isinstance(exc, StageError):
                artifacts.update(exc.artifacts)
            self._write_manifest(
                input_path=input_path,
                work_dir=work_dir,
                output_path=current,
                artifacts=artifacts,
                status="failed",
                error=exc,
            )
            raise

        if current is None:  # pragma: no cover - impossible after successful ingest
            raise RuntimeError("Pipeline completed without producing an output path.")
        artifacts["final_audio"] = current
        artifacts["pipeline_manifest"] = self._write_manifest(
            input_path=input_path,
            work_dir=work_dir,
            output_path=current,
            artifacts=artifacts,
            status="succeeded",
        )
        self._progress("complete", 99)
        return PipelineResult(output_path=current, artifacts=artifacts)
