from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass

from openphonic.core.settings import Settings, get_settings
from openphonic.pipeline.config import PipelineConfig
from openphonic.pipeline.deepgram import DeepgramError, validate_deepgram_api_key


@dataclass(frozen=True)
class PipelinePreflightIssue:
    stage: str
    message: str


def pipeline_preflight_issues(
    config: PipelineConfig,
    settings: Settings | None = None,
) -> list[PipelinePreflightIssue]:
    settings = settings or get_settings()
    issues: list[PipelinePreflightIssue] = []

    if config.enabled("noise_reduction"):
        if not _binary_available(settings.deepfilternet_bin):
            issues.append(
                PipelinePreflightIssue(
                    stage="noise_reduction",
                    message=(
                        "DeepFilterNet noise reduction is enabled, but the "
                        f"{settings.deepfilternet_bin} CLI was not found on PATH."
                    ),
                )
            )

    if config.enabled("music_separation"):
        music_separation = config.stage("music_separation")
        if not music_separation.get("model"):
            issues.append(
                PipelinePreflightIssue(
                    stage="music_separation",
                    message="Demucs source separation requires stages.music_separation.model.",
                )
            )
        if not music_separation.get("stem"):
            issues.append(
                PipelinePreflightIssue(
                    stage="music_separation",
                    message="Demucs source separation requires stages.music_separation.stem.",
                )
            )
        if not _module_available("demucs.separate"):
            issues.append(
                PipelinePreflightIssue(
                    stage="music_separation",
                    message=(
                        "Demucs source separation is enabled, but demucs is not installed. "
                        'Install with pip install -e ".[ml]" or disable '
                        "stages.music_separation."
                    ),
                )
            )

    if config.enabled("intro_outro"):
        issues.extend(_intro_outro_issues(config))

    if config.enabled("transcription"):
        if settings.transcription_provider == "deepgram":
            issues.extend(_deepgram_transcription_issues(settings))
        elif not _module_available("faster_whisper"):
            issues.append(
                PipelinePreflightIssue(
                    stage="transcription",
                    message=(
                        "Transcription is enabled, but faster-whisper is not installed. "
                        'Install with pip install -e ".[ml]" or disable stages.transcription.'
                    ),
                )
            )

    if config.enabled("filler_removal") and not config.enabled("transcription"):
        issues.append(
            PipelinePreflightIssue(
                stage="filler_removal",
                message=(
                    "Filler-word cut suggestions require stages.transcription so word "
                    "timestamps can be produced."
                ),
            )
        )

    if config.enabled("diarization"):
        if settings.transcription_provider == "deepgram":
            if not config.enabled("transcription"):
                issues.append(
                    PipelinePreflightIssue(
                        stage="diarization",
                        message=(
                            "Deepgram diarization must run through stages.transcription; "
                            "enable stages.transcription before stages.diarization."
                        ),
                    )
                )
            if not config.enabled("transcription"):
                issues.extend(_deepgram_transcription_issues(settings))
        else:
            if not settings.hf_token:
                issues.append(
                    PipelinePreflightIssue(
                        stage="diarization",
                        message="Diarization requires HF_TOKEN for pyannote pretrained pipelines.",
                    )
                )
            if not _module_available("pyannote.audio"):
                issues.append(
                    PipelinePreflightIssue(
                        stage="diarization",
                        message=(
                            "Diarization is enabled, but pyannote.audio is not installed. "
                            'Install with pip install -e ".[ml]" or disable stages.diarization.'
                        ),
                    )
                )

    return issues


def _deepgram_transcription_issues(settings: Settings) -> list[PipelinePreflightIssue]:
    issues: list[PipelinePreflightIssue] = []
    if not settings.deepgram_api_key:
        issues.append(
            PipelinePreflightIssue(
                stage="transcription",
                message="Deepgram transcription provider requires DEEPGRAM_API_KEY.",
            )
        )
        return issues
    try:
        validate_deepgram_api_key(settings.deepgram_api_key)
    except DeepgramError as exc:
        issues.append(
            PipelinePreflightIssue(
                stage="transcription",
                message=f"Deepgram API key could not be validated: {exc}",
            )
        )
    return issues


def format_preflight_issues(issues: list[PipelinePreflightIssue]) -> str:
    return " ".join(issue.message for issue in issues)


def _intro_outro_issues(config: PipelineConfig) -> list[PipelinePreflightIssue]:
    stage = config.stage("intro_outro")
    issues: list[PipelinePreflightIssue] = []
    configured = [
        (field_name, stage.get(field_name))
        for field_name in ("intro_path", "outro_path")
        if stage.get(field_name) not in (None, "")
    ]
    if not configured:
        return [
            PipelinePreflightIssue(
                stage="intro_outro",
                message="Intro/outro insertion requires intro_path or outro_path.",
            )
        ]

    for field_name, value in configured:
        if not isinstance(value, str):
            issues.append(
                PipelinePreflightIssue(
                    stage="intro_outro",
                    message=f"Intro/outro {field_name} must be a filesystem path.",
                )
            )
            continue
        path = config.resolve_path(value)
        if not path.exists():
            issues.append(
                PipelinePreflightIssue(
                    stage="intro_outro",
                    message=f"Intro/outro {field_name} does not exist: {path}",
                )
            )
        elif not path.is_file():
            issues.append(
                PipelinePreflightIssue(
                    stage="intro_outro",
                    message=f"Intro/outro {field_name} is not a file: {path}",
                )
            )
    return issues


def _binary_available(binary: str) -> bool:
    return shutil.which(binary) is not None


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False
