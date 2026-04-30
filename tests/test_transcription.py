import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from openphonic.core.settings import get_settings
from openphonic.pipeline.config import PipelineConfig
from openphonic.pipeline.stages import StageError, TranscriptionStage


def test_transcription_stage_writes_word_timestamp_artifacts(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    calls: dict[str, object] = {}

    class FakeWhisperModel:
        def __init__(self, model_name: str, *, device: str) -> None:
            calls["model"] = {"model_name": model_name, "device": device}

        def transcribe(self, path: str, **kwargs: object):
            calls["transcribe"] = {"path": path, **kwargs}
            return (
                [
                    SimpleNamespace(
                        id=1,
                        start=0.0,
                        end=1.1,
                        text=" Hello world",
                        words=[
                            SimpleNamespace(
                                start=0.0,
                                end=0.42,
                                word=" Hello",
                                probability=0.97,
                            ),
                            SimpleNamespace(
                                start=0.43,
                                end=1.1,
                                word=" world",
                                probability=0.93,
                            ),
                        ],
                    )
                ],
                SimpleNamespace(language="en", language_probability=0.99, duration=1.1),
            )

    fake_faster_whisper = ModuleType("faster_whisper")
    fake_faster_whisper.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_faster_whisper)
    monkeypatch.setenv("OPENPHONIC_WHISPER_DEVICE", "cpu")
    get_settings.cache_clear()

    config = PipelineConfig(
        name="test",
        stages={"transcription": {"enabled": True, "model": "tiny", "language": "en"}},
    )

    try:
        artifacts = TranscriptionStage(config).run(input_path, tmp_path)
    finally:
        get_settings.cache_clear()

    assert calls["model"] == {"model_name": "tiny", "device": "cpu"}
    assert calls["transcribe"] == {
        "path": str(input_path),
        "language": "en",
        "word_timestamps": True,
    }
    assert artifacts == {
        "transcript_json": tmp_path / "transcript.json",
        "transcript_vtt": tmp_path / "transcript.vtt",
    }

    transcript = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert transcript == {
        "schema_version": 1,
        "engine": "faster-whisper",
        "model": "tiny",
        "device": "cpu",
        "language": "en",
        "language_probability": 0.99,
        "duration": 1.1,
        "segments": [
            {
                "id": 1,
                "start": 0.0,
                "end": 1.1,
                "text": " Hello world",
                "words": [
                    {"start": 0.0, "end": 0.42, "word": " Hello", "probability": 0.97},
                    {"start": 0.43, "end": 1.1, "word": " world", "probability": 0.93},
                ],
            }
        ],
    }

    assert (tmp_path / "transcript.vtt").read_text(encoding="utf-8") == (
        "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.100\nHello world\n"
    )


def test_transcription_stage_rejects_unimplemented_deepgram_provider(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    monkeypatch.setenv("TRANSCRIPTION_PROVIDER", "deepgram")
    get_settings.cache_clear()

    try:
        with pytest.raises(StageError, match="Deepgram transcription provider"):
            TranscriptionStage(PipelineConfig(name="test")).run(input_path, tmp_path)
    finally:
        get_settings.cache_clear()
