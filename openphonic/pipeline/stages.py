from __future__ import annotations

import json
import os
import shutil
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

from openphonic.core.settings import get_settings
from openphonic.pipeline.config import PipelineConfig
from openphonic.pipeline.ffmpeg import (
    build_ingest_command,
    build_intro_outro_command,
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
        stage = self.config.stage("noise_reduction")
        binary = settings.deepfilternet_bin
        if shutil.which(binary) is None:
            raise StageError(
                "DeepFilterNet stage is enabled, but the deepFilter CLI was not found. "
                "Install DeepFilterNet or disable stages.noise_reduction."
            )
        command = [binary]
        attenuation_db = stage.get("attenuation_db")
        if attenuation_db is not None:
            try:
                attenuation_limit = float(attenuation_db)
            except (TypeError, ValueError) as exc:
                raise StageError("DeepFilterNet attenuation_db must be an integer.") from exc
            if not attenuation_limit.is_integer():
                raise StageError("DeepFilterNet attenuation_db must be an integer.")
            if attenuation_limit <= 0:
                raise StageError("DeepFilterNet attenuation_db must be greater than zero.")
            command.extend(["--atten-lim", str(int(attenuation_limit))])

        output_dir = work_dir / "02_deepfilternet"
        output_dir.mkdir(parents=True, exist_ok=True)
        command.extend([str(input_path), "-o", str(output_dir)])
        completed = subprocess.run(
            command,
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


class IntroOutroStage(PipelineStage):
    def run(self, input_path: Path, work_dir: Path) -> Path:
        stage = self.config.stage("intro_outro")
        intro_path = self._asset_path(stage.get("intro_path"), "intro_path")
        outro_path = self._asset_path(stage.get("outro_path"), "outro_path")
        if intro_path is None and outro_path is None:
            raise StageError("Intro/outro stage requires intro_path or outro_path.")

        output_path = work_dir / "05_intro_outro.wav"
        run_command(
            build_intro_outro_command(
                input_path,
                output_path,
                target=self.config.target,
                intro_path=intro_path,
                outro_path=outro_path,
            ),
            log_path=self.command_log_path,
        )
        return require_artifact(output_path, "Intro/outro")

    def _asset_path(self, value: Any, field_name: str) -> Path | None:
        if value in (None, ""):
            return None
        if not isinstance(value, str):
            raise StageError(f"Intro/outro {field_name} must be a filesystem path.")
        path = self.config.resolve_path(value)
        if not path.exists():
            raise StageError(f"Intro/outro {field_name} does not exist: {path}")
        if not path.is_file():
            raise StageError(f"Intro/outro {field_name} is not a file: {path}")
        return path


class FillerRemovalStage(PipelineStage):
    def run(self, transcript_path: Path | None, work_dir: Path) -> dict[str, Path]:
        stage = self.config.stage("filler_removal")
        artifact_path = work_dir / "cut_suggestions.json"
        configured_words = _configured_filler_words(stage)
        min_silence_seconds = _positive_float(stage.get("min_silence_seconds"), 0.75)

        if transcript_path is None or not transcript_path.exists():
            _write_cut_suggestions(
                artifact_path,
                source_artifact=None,
                configured_words=configured_words,
                min_silence_seconds=min_silence_seconds,
                suggestions=[],
                status="not_available",
                reason="Transcript word timestamps are required before cuts can be suggested.",
            )
            raise StageError(
                (
                    "Filler removal is configured but no transcript artifact was produced. "
                    "Enable stages.transcription before stages.filler_removal."
                ),
                artifacts={"cut_suggestions": artifact_path},
            )

        try:
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _write_cut_suggestions(
                artifact_path,
                source_artifact=transcript_path.name,
                configured_words=configured_words,
                min_silence_seconds=min_silence_seconds,
                suggestions=[],
                status="not_available",
                reason="Transcript artifact is invalid JSON.",
            )
            raise StageError("Transcript artifact is invalid JSON.") from exc
        if not isinstance(transcript, dict):
            _write_cut_suggestions(
                artifact_path,
                source_artifact=transcript_path.name,
                configured_words=configured_words,
                min_silence_seconds=min_silence_seconds,
                suggestions=[],
                status="not_available",
                reason="Transcript artifact must be a JSON object.",
            )
            raise StageError("Transcript artifact must be a JSON object.")

        suggestions = _build_cut_suggestions(
            transcript,
            filler_words=configured_words,
            min_silence_seconds=min_silence_seconds,
        )
        _write_cut_suggestions(
            artifact_path,
            source_artifact=transcript_path.name,
            configured_words=configured_words,
            min_silence_seconds=min_silence_seconds,
            suggestions=suggestions,
            status="not_applied",
            reason="Suggestions only; manual review is required before any cuts are applied.",
        )
        require_artifact(artifact_path, "Filler suggestion")
        return {"cut_suggestions": artifact_path}


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
        if settings.transcription_provider == "deepgram":
            raise StageError(
                "Deepgram transcription provider is configured, but the Deepgram adapter "
                "is not implemented yet. Use TRANSCRIPTION_PROVIDER=local until the "
                "adapter lands."
            )
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
        segments, info = model.transcribe(
            str(input_path),
            language=stage.get("language"),
            word_timestamps=True,
        )
        transcript = {
            "schema_version": 1,
            "engine": "faster-whisper",
            "model": model_name,
            "device": device,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": getattr(info, "duration", None),
            "segments": [_segment_to_dict(segment) for segment in segments],
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
        if settings.transcription_provider == "deepgram":
            raise StageError(
                "Deepgram diarization must run through the transcription provider, but "
                "the Deepgram adapter is not implemented yet. Use TRANSCRIPTION_PROVIDER=local "
                "until the adapter lands."
            )
        if not settings.hf_token:
            raise StageError("Diarization requires HF_TOKEN for pyannote pretrained pipelines.")
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise StageError(
                "Diarization is enabled, but pyannote.audio is not installed. "
                'Install with pip install -e ".[ml]" or disable stages.diarization.'
            ) from exc

        model_name = stage.get("model") or settings.pyannote_model
        pipeline = _load_pyannote_pipeline(Pipeline, model_name, settings.hf_token)
        diarization = pipeline(str(input_path))
        annotation = _diarization_annotation(diarization)

        rttm_path = work_dir / "diarization.rttm"
        with rttm_path.open("w", encoding="utf-8") as handle:
            _write_diarization_rttm(diarization, annotation, handle)
        require_artifact(rttm_path, "Diarization", allow_empty=True)

        segments = _diarization_segments(annotation)
        json_path = work_dir / "diarization.json"
        json_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "engine": "pyannote.audio",
                    "model": model_name,
                    "speaker_count": len({segment["speaker"] for segment in segments}),
                    "segments": segments,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        require_artifact(json_path, "Diarization")
        return {"diarization_rttm": rttm_path, "diarization_json": json_path}


def _timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds_value, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds_value:02}.{milliseconds:03}"


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return parsed


def _normalize_word(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().strip(string.punctuation).lower()


def _configured_filler_words(stage: dict[str, Any]) -> list[str]:
    raw_words = stage.get("words", ["um", "uh", "erm", "ah"])
    if raw_words is None:
        raw_words = []
    if isinstance(raw_words, str):
        raw_words = [raw_words]
    elif not isinstance(raw_words, list | tuple | set):
        raw_words = []
    words: list[str] = []
    seen: set[str] = set()
    for word in raw_words:
        normalized = _normalize_word(word)
        if normalized and normalized not in seen:
            words.append(normalized)
            seen.add(normalized)
    return words


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_seconds(value: float) -> float:
    return round(value, 3)


def _transcript_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(transcript.get("segments") or []):
        if not isinstance(segment, dict):
            continue
        for word_index, word in enumerate(segment.get("words") or []):
            if not isinstance(word, dict):
                continue
            start = _optional_float(word.get("start"))
            end = _optional_float(word.get("end"))
            if start is None or end is None or end < start:
                continue
            text = str(word.get("word") or "").strip()
            words.append(
                {
                    "segment_index": segment_index,
                    "word_index": word_index,
                    "start": start,
                    "end": end,
                    "text": text,
                    "normalized_text": _normalize_word(text),
                    "probability": _optional_float(word.get("probability")),
                }
            )
    return sorted(words, key=lambda word: (word["start"], word["end"]))


def _transcript_segments_for_timing(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(transcript.get("segments") or []):
        if not isinstance(segment, dict):
            continue
        start = _optional_float(segment.get("start"))
        end = _optional_float(segment.get("end"))
        if start is None or end is None or end < start:
            continue
        segments.append(
            {
                "segment_index": segment_index,
                "word_index": None,
                "start": start,
                "end": end,
            }
        )
    return sorted(segments, key=lambda segment: (segment["start"], segment["end"]))


def _build_cut_suggestions(
    transcript: dict[str, Any],
    *,
    filler_words: list[str],
    min_silence_seconds: float,
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    words = _transcript_words(transcript)
    filler_word_set = set(filler_words)

    for word in words:
        if word["normalized_text"] not in filler_word_set:
            continue
        suggestion = {
            "type": "filler_word",
            "start": _round_seconds(word["start"]),
            "end": _round_seconds(word["end"]),
            "duration": _round_seconds(word["end"] - word["start"]),
            "text": word["text"],
            "normalized_text": word["normalized_text"],
            "segment_index": word["segment_index"],
            "word_index": word["word_index"],
            "reason": "Matched configured filler word.",
        }
        if word["probability"] is not None:
            suggestion["confidence"] = word["probability"]
        suggestions.append(suggestion)

    timed_segments = _transcript_segments_for_timing(transcript)
    word_segment_indexes = {word["segment_index"] for word in words}
    wordless_segments = [
        segment
        for segment in timed_segments
        if segment["segment_index"] not in word_segment_indexes
    ]

    if words:
        for before, after in zip(words, words[1:], strict=False):
            if _gap_overlaps_segments(before["end"], after["start"], wordless_segments):
                continue
            _append_silence_suggestion(
                suggestions,
                before=before,
                after=after,
                source="word_gap",
                min_silence_seconds=min_silence_seconds,
            )

    wordless_segment_indexes = {segment["segment_index"] for segment in wordless_segments}
    for before, after in zip(timed_segments, timed_segments[1:], strict=False):
        if words and (
            before["segment_index"] not in wordless_segment_indexes
            and after["segment_index"] not in wordless_segment_indexes
        ):
            continue
        _append_silence_suggestion(
            suggestions,
            before=before,
            after=after,
            source="segment_gap",
            min_silence_seconds=min_silence_seconds,
        )

    suggestions = sorted(
        suggestions,
        key=lambda suggestion: (
            suggestion["start"],
            suggestion["end"],
            suggestion["type"],
        ),
    )
    return [
        {
            "id": f"cut-{index:04d}",
            **suggestion,
        }
        for index, suggestion in enumerate(suggestions, start=1)
    ]


def _gap_overlaps_segments(
    start: float,
    end: float,
    segments: list[dict[str, Any]],
) -> bool:
    return any(segment["start"] < end and segment["end"] > start for segment in segments)


def _append_silence_suggestion(
    suggestions: list[dict[str, Any]],
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    source: str,
    min_silence_seconds: float,
) -> None:
    start = before["end"]
    end = after["start"]
    duration = end - start
    if duration < min_silence_seconds:
        return
    suggestions.append(
        {
            "type": "silence",
            "start": _round_seconds(start),
            "end": _round_seconds(end),
            "duration": _round_seconds(duration),
            "source": source,
            "before_segment_index": before["segment_index"],
            "before_word_index": before["word_index"],
            "after_segment_index": after["segment_index"],
            "after_word_index": after["word_index"],
            "reason": "Detected a timestamp gap longer than the configured threshold.",
        }
    )


def _write_cut_suggestions(
    artifact_path: Path,
    *,
    source_artifact: str | None,
    configured_words: list[str],
    min_silence_seconds: float,
    suggestions: list[dict[str, Any]],
    status: str,
    reason: str,
) -> None:
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "reason": reason,
                "source_artifact": source_artifact,
                "configured_words": configured_words,
                "min_silence_seconds": min_silence_seconds,
                "suggestion_count": len(suggestions),
                "suggestions": suggestions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _word_to_dict(word: Any) -> dict[str, Any]:
    return {
        "start": getattr(word, "start", None),
        "end": getattr(word, "end", None),
        "word": getattr(word, "word", ""),
        "probability": getattr(word, "probability", None),
    }


def _segment_to_dict(segment: Any) -> dict[str, Any]:
    return {
        "id": getattr(segment, "id", None),
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "words": [_word_to_dict(word) for word in (getattr(segment, "words", None) or [])],
    }


def _diarization_annotation(diarization: Any) -> Any:
    return getattr(diarization, "speaker_diarization", diarization)


def _load_pyannote_pipeline(pipeline_class: Any, model_name: str, token: str) -> Any:
    try:
        return pipeline_class.from_pretrained(model_name, token=token)
    except TypeError as exc:
        if "token" not in str(exc):
            raise
    return pipeline_class.from_pretrained(model_name, use_auth_token=token)


def _write_diarization_rttm(diarization: Any, annotation: Any, handle: Any) -> None:
    writer = getattr(annotation, "write_rttm", None) or getattr(diarization, "write_rttm", None)
    if writer is None:
        raise StageError("Diarization completed but the result cannot be exported as RTTM.")
    writer(handle)


def _diarization_segments(annotation: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for turn, track, speaker in _iter_diarization_turns(annotation):
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
                "track": None if track is None else str(track),
            }
        )
    return segments


def _iter_diarization_turns(annotation: Any) -> Any:
    itertracks = getattr(annotation, "itertracks", None)
    if callable(itertracks):
        yield from itertracks(yield_label=True)
        return

    for item in annotation:
        if len(item) == 2:
            turn, speaker = item
            yield turn, None, speaker
        else:
            turn, track, speaker = item
            yield turn, track, speaker


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
