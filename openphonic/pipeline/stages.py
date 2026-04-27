from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from openphonic.core.settings import get_settings
from openphonic.pipeline.config import PipelineConfig
from openphonic.pipeline.ffmpeg import (
    build_ingest_command,
    build_loudnorm_apply_command,
    build_loudnorm_probe_command,
    build_silence_trim_command,
    parse_loudnorm_json,
    run_command,
)


class StageError(RuntimeError):
    """Raised when a pipeline stage cannot produce a valid artifact."""

    def __init__(self, message: str, artifacts: dict[str, Path] | None = None) -> None:
        super().__init__(message)
        self.artifacts = artifacts or {}


def require_artifact(path: Path, stage_name: str, *, allow_empty: bool = False) -> Path:
    if not path.exists():
        raise StageError(f"{stage_name} stage did not produce expected artifact: {path}")
    if path.is_file() and not allow_empty and path.stat().st_size == 0:
        raise StageError(f"{stage_name} stage produced an empty artifact: {path}")
    return path


class PipelineStage:
    def __init__(self, config: PipelineConfig, command_log_path: Path | None = None) -> None:
        self.config = config
        self.command_log_path = command_log_path


class IngestStage(PipelineStage):
    def run(self, input_path: Path, work_dir: Path) -> Path:
        output_path = work_dir / "01_ingest.wav"
        run_command(
            build_ingest_command(input_path, output_path, self.config.target),
            log_path=self.command_log_path,
        )
        return require_artifact(output_path, "Ingest")


class DeepFilterNetStage(PipelineStage):
    def run(self, input_path: Path, work_dir: Path) -> Path:
        settings = get_settings()
        binary = settings.deepfilternet_bin
        if shutil.which(binary) is None:
            raise StageError(
                "DeepFilterNet stage is enabled, but the deepFilter CLI was not found. "
                "Install DeepFilterNet or disable stages.noise_reduction."
            )

        output_dir = work_dir / "02_deepfilternet"
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [binary, str(input_path), "-o", str(output_dir)],
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise StageError(completed.stderr.strip() or "DeepFilterNet failed.")

        candidates = sorted(output_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise StageError("DeepFilterNet completed but produced no WAV output.")
        output_path = work_dir / "02_noise_reduced.wav"
        shutil.copy2(candidates[-1], output_path)
        return require_artifact(output_path, "DeepFilterNet")


class DemucsStage(PipelineStage):
    def run(self, input_path: Path, work_dir: Path) -> Path:
        stage_config = self.config.stage("music_separation")
        model = stage_config.get("model")
        stem = stage_config.get("stem")
        if not model:
            raise StageError("Demucs stage requires stages.music_separation.model.")
        if not stem:
            raise StageError("Demucs stage requires stages.music_separation.stem.")
        output_root = work_dir / "03_demucs"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "demucs.separate",
                "-n",
                model,
                "--two-stems",
                stem,
                "-o",
                str(output_root),
                str(input_path),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise StageError(completed.stderr.strip() or "Demucs failed.")

        candidate = output_root / model / input_path.stem / f"{stem}.wav"
        if not candidate.exists():
            raise StageError(f"Demucs output was not found at {candidate}.")
        output_path = work_dir / "03_separated.wav"
        shutil.copy2(candidate, output_path)
        return require_artifact(output_path, "Demucs")


class SilenceTrimStage(PipelineStage):
    def run(self, input_path: Path, work_dir: Path) -> Path:
        stage = self.config.stage("silence_trim")
        output_path = work_dir / "04_silence_trimmed.wav"
        run_command(
            build_silence_trim_command(
                input_path,
                output_path,
                start_threshold_db=stage.get("start_threshold_db", -50),
                stop_threshold_db=stage.get("stop_threshold_db", -50),
                min_silence_seconds=stage.get("min_silence_seconds", 0.35),
            ),
            log_path=self.command_log_path,
        )
        return require_artifact(output_path, "Silence trim")


class FillerRemovalStage(PipelineStage):
    def run(self, input_path: Path, work_dir: Path) -> Path:
        manifest = work_dir / "05_filler_removal_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "status": "not_applied",
                    "reason": (
                        "Filler removal needs word timestamps and a review workflow "
                        "before cuts are safe."
                    ),
                    "configured_words": self.config.stage("filler_removal").get("words", []),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise StageError(
            (
                "Filler removal is configured but not available yet. "
                "It requires word timestamps and a manual review workflow before cuts are safe."
            ),
            artifacts={"filler_removal_manifest": manifest},
        )


class LoudnessStage(PipelineStage):
    def run(self, input_path: Path, work_dir: Path) -> Path:
        stage = self.config.stage("loudness")
        output_path = work_dir / f"06_loudnorm.{self.config.target.container}"
        probe = run_command(
            build_loudnorm_probe_command(input_path, stage),
            log_path=self.command_log_path,
        )
        measured = parse_loudnorm_json(probe.stderr)
        run_command(
            build_loudnorm_apply_command(
                input_path,
                output_path,
                stage=stage,
                target=self.config.target,
                measured=measured,
            ),
            log_path=self.command_log_path,
        )
        return require_artifact(output_path, "Loudness")


class TranscriptionStage(PipelineStage):
    def run(self, input_path: Path, work_dir: Path) -> dict[str, Path]:
        stage = self.config.stage("transcription")
        settings = get_settings()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise StageError(
                "Transcription is enabled, but faster-whisper is not installed. "
                'Install with pip install -e ".[ml]" or disable stages.transcription.'
            ) from exc

        model_name = stage.get("model") or settings.whisper_model
        device = settings.whisper_device
        if device == "auto":
            device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"

        model = WhisperModel(model_name, device=device)
        segments, info = model.transcribe(str(input_path), language=stage.get("language"))
        transcript = {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": [
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                for segment in segments
            ],
        }
        json_path = work_dir / "transcript.json"
        json_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
        require_artifact(json_path, "Transcription")

        vtt_path = work_dir / "transcript.vtt"
        vtt_path.write_text(_segments_to_vtt(transcript["segments"]), encoding="utf-8")
        require_artifact(vtt_path, "Transcription")
        return {"transcript_json": json_path, "transcript_vtt": vtt_path}


class DiarizationStage(PipelineStage):
    def run(self, input_path: Path, work_dir: Path) -> dict[str, Path]:
        stage = self.config.stage("diarization")
        settings = get_settings()
        if not settings.hf_token:
            raise StageError("Diarization requires HF_TOKEN for pyannote pretrained pipelines.")
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise StageError(
                "Diarization is enabled, but pyannote.audio is not installed. "
                'Install with pip install -e ".[ml]" or disable stages.diarization.'
            ) from exc

        pipeline = Pipeline.from_pretrained(
            stage.get("model") or settings.pyannote_model,
            use_auth_token=settings.hf_token,
        )
        diarization = pipeline(str(input_path))
        rttm_path = work_dir / "diarization.rttm"
        with rttm_path.open("w", encoding="utf-8") as handle:
            diarization.write_rttm(handle)
        require_artifact(rttm_path, "Diarization", allow_empty=True)
        return {"diarization_rttm": rttm_path}


def _timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds_value, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds_value:02}.{milliseconds:03}"


def _segments_to_vtt(segments: list[dict[str, Any]]) -> str:
    lines = ["WEBVTT", ""]
    for index, segment in enumerate(segments, start=1):
        lines.extend(
            [
                str(index),
                f"{_timestamp(segment['start'])} --> {_timestamp(segment['end'])}",
                segment["text"].strip(),
                "",
            ]
        )
    return "\n".join(lines)
