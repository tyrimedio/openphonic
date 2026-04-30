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


def test_transcription_stage_writes_deepgram_transcript_and_diarization_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    monkeypatch.setenv("TRANSCRIPTION_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg_test")
    monkeypatch.setenv("OPENPHONIC_DEEPGRAM_MODEL", "nova-3")
    get_settings.cache_clear()
    calls: dict[str, object] = {}

    def fake_transcribe(path, options):
        calls["path"] = path
        calls["options"] = options
        return {
            "metadata": {"duration": 2.5},
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "Hello there. Hi back.",
                                "confidence": 0.98,
                                "words": [
                                    {
                                        "word": "hello",
                                        "punctuated_word": "Hello",
                                        "start": 0.0,
                                        "end": 0.4,
                                        "confidence": 0.97,
                                        "speaker": 0,
                                        "speaker_confidence": 0.82,
                                    },
                                    {
                                        "word": "there",
                                        "punctuated_word": "there.",
                                        "start": 0.45,
                                        "end": 0.9,
                                        "confidence": 0.96,
                                        "speaker": 0,
                                    },
                                    {
                                        "word": "hi",
                                        "punctuated_word": "Hi",
                                        "start": 1.2,
                                        "end": 1.4,
                                        "confidence": 0.94,
                                        "speaker": 1,
                                    },
                                ],
                            }
                        ]
                    }
                ],
                "utterances": [
                    {
                        "id": "utt-1",
                        "start": 0.0,
                        "end": 0.9,
                        "confidence": 0.95,
                        "speaker": 0,
                        "transcript": "Hello there.",
                        "words": [
                            {
                                "word": "hello",
                                "punctuated_word": "Hello",
                                "start": 0.0,
                                "end": 0.4,
                                "confidence": 0.97,
                                "speaker": 0,
                            },
                            {
                                "word": "there",
                                "punctuated_word": "there.",
                                "start": 0.45,
                                "end": 0.9,
                                "confidence": 0.96,
                                "speaker": 0,
                            },
                        ],
                    },
                    {
                        "id": "utt-2",
                        "start": 1.2,
                        "end": 1.4,
                        "confidence": 0.94,
                        "speaker": 1,
                        "transcript": "Hi back.",
                        "words": [
                            {
                                "word": "hi",
                                "punctuated_word": "Hi",
                                "start": 1.2,
                                "end": 1.4,
                                "confidence": 0.94,
                                "speaker": 1,
                            }
                        ],
                    },
                ],
            },
        }

    monkeypatch.setattr("openphonic.pipeline.stages.transcribe_deepgram_file", fake_transcribe)
    config = PipelineConfig(
        name="test",
        stages={
            "transcription": {"enabled": True, "model": "small", "language": "en"},
            "diarization": {"enabled": True},
        },
    )

    try:
        artifacts = TranscriptionStage(config).run(input_path, tmp_path)
    finally:
        get_settings.cache_clear()

    options = calls["options"]
    assert calls["path"] == input_path
    assert options.api_key == "dg_test"
    assert options.model == "nova-3"
    assert options.language == "en"
    assert options.diarize is True
    assert artifacts == {
        "transcript_json": tmp_path / "transcript.json",
        "transcript_vtt": tmp_path / "transcript.vtt",
        "deepgram_response": tmp_path / "deepgram_response.json",
        "diarization_json": tmp_path / "diarization.json",
        "diarization_rttm": tmp_path / "diarization.rttm",
    }

    transcript = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["engine"] == "deepgram"
    assert transcript["model"] == "nova-3"
    assert transcript["language"] == "en"
    assert transcript["duration"] == 2.5
    assert transcript["segments"] == [
        {
            "id": "utt-1",
            "start": 0.0,
            "end": 0.9,
            "text": "Hello there.",
            "words": [
                {
                    "start": 0.0,
                    "end": 0.4,
                    "word": "Hello",
                    "probability": 0.97,
                    "speaker": "SPEAKER_00",
                },
                {
                    "start": 0.45,
                    "end": 0.9,
                    "word": "there.",
                    "probability": 0.96,
                    "speaker": "SPEAKER_00",
                },
            ],
            "confidence": 0.95,
            "speaker": "SPEAKER_00",
        },
        {
            "id": "utt-2",
            "start": 1.2,
            "end": 1.4,
            "text": "Hi back.",
            "words": [
                {
                    "start": 1.2,
                    "end": 1.4,
                    "word": "Hi",
                    "probability": 0.94,
                    "speaker": "SPEAKER_01",
                }
            ],
            "confidence": 0.94,
            "speaker": "SPEAKER_01",
        },
    ]
    assert (tmp_path / "transcript.vtt").read_text(encoding="utf-8") == (
        "WEBVTT\n\n1\n00:00:00.000 --> 00:00:00.900\nHello there.\n\n"
        "2\n00:00:01.200 --> 00:00:01.400\nHi back.\n"
    )

    diarization = json.loads((tmp_path / "diarization.json").read_text(encoding="utf-8"))
    assert diarization == {
        "schema_version": 1,
        "engine": "deepgram",
        "model": "nova-3",
        "speaker_count": 2,
        "segments": [
            {
                "start": 0.0,
                "end": 0.9,
                "speaker": "SPEAKER_00",
                "track": None,
                "confidence": 0.95,
            },
            {
                "start": 1.2,
                "end": 1.4,
                "speaker": "SPEAKER_01",
                "track": None,
                "confidence": 0.94,
            },
        ],
    }
    assert (tmp_path / "diarization.rttm").read_text(encoding="utf-8") == (
        "SPEAKER input 1 0.000 0.900 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        "SPEAKER input 1 1.200 0.200 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
    )


def test_transcription_stage_requires_deepgram_api_key(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    monkeypatch.setenv("TRANSCRIPTION_PROVIDER", "deepgram")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    get_settings.cache_clear()

    try:
        with pytest.raises(StageError, match="DEEPGRAM_API_KEY"):
            TranscriptionStage(PipelineConfig(name="test")).run(input_path, tmp_path)
    finally:
        get_settings.cache_clear()


def test_transcription_stage_preserves_unparseable_deepgram_response(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    monkeypatch.setenv("TRANSCRIPTION_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg_test")
    get_settings.cache_clear()

    def fake_transcribe(path, options):
        _ = path, options
        return {"results": {"channels": []}}

    monkeypatch.setattr("openphonic.pipeline.stages.transcribe_deepgram_file", fake_transcribe)

    try:
        with pytest.raises(StageError, match="did not include any result channels") as raised:
            TranscriptionStage(PipelineConfig(name="test")).run(input_path, tmp_path)
    finally:
        get_settings.cache_clear()

    raw_path = tmp_path / "deepgram_response.json"
    assert raised.value.artifacts == {"deepgram_response": raw_path}
    assert json.loads(raw_path.read_text(encoding="utf-8")) == {"results": {"channels": []}}
