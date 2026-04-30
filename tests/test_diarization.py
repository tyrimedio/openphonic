import json
import sys
from types import ModuleType

import pytest

from openphonic.core.settings import get_settings
from openphonic.pipeline.config import PipelineConfig
from openphonic.pipeline.stages import DiarizationStage, StageError


def test_diarization_stage_writes_rttm_and_json_artifacts(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    calls: dict[str, object] = {}

    class FakeTurn:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class FakeDiarization:
        def itertracks(self, *, yield_label: bool):
            assert yield_label is True
            yield FakeTurn(0.0, 1.4), "A", "SPEAKER_00"
            yield FakeTurn(1.4, 2.8), "B", "SPEAKER_01"
            yield FakeTurn(2.8, 3.2), "C", "SPEAKER_00"

        def write_rttm(self, handle) -> None:
            handle.write("SPEAKER input 1 0.000 1.400 <NA> <NA> SPEAKER_00 <NA> <NA>\n")
            handle.write("SPEAKER input 1 1.400 1.400 <NA> <NA> SPEAKER_01 <NA> <NA>\n")

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_name: str, *, use_auth_token: str):
            calls["from_pretrained"] = {
                "model_name": model_name,
                "use_auth_token": use_auth_token,
            }
            return cls()

        def __call__(self, path: str) -> FakeDiarization:
            calls["path"] = path
            return FakeDiarization()

    fake_audio = ModuleType("pyannote.audio")
    fake_audio.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    get_settings.cache_clear()

    config = PipelineConfig(
        name="test",
        stages={"diarization": {"enabled": True, "model": "pyannote/test-diarization"}},
    )

    try:
        artifacts = DiarizationStage(config).run(input_path, tmp_path)
    finally:
        get_settings.cache_clear()

    assert calls == {
        "from_pretrained": {
            "model_name": "pyannote/test-diarization",
            "use_auth_token": "hf_test",
        },
        "path": str(input_path),
    }
    assert artifacts == {
        "diarization_rttm": tmp_path / "diarization.rttm",
        "diarization_json": tmp_path / "diarization.json",
    }
    assert "SPEAKER_00" in (tmp_path / "diarization.rttm").read_text(encoding="utf-8")

    diarization = json.loads((tmp_path / "diarization.json").read_text(encoding="utf-8"))
    assert diarization == {
        "schema_version": 1,
        "engine": "pyannote.audio",
        "model": "pyannote/test-diarization",
        "speaker_count": 2,
        "segments": [
            {"start": 0.0, "end": 1.4, "speaker": "SPEAKER_00", "track": "A"},
            {"start": 1.4, "end": 2.8, "speaker": "SPEAKER_01", "track": "B"},
            {"start": 2.8, "end": 3.2, "speaker": "SPEAKER_00", "track": "C"},
        ],
    }


def test_diarization_stage_supports_token_pipeline_outputs(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    calls: dict[str, object] = {}

    class FakeTurn:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class FakeOutput:
        speaker_diarization = [
            (FakeTurn(0.25, 1.0), "SPEAKER_00"),
            (FakeTurn(1.2, 2.0), "SPEAKER_01"),
        ]

        def write_rttm(self, handle) -> None:
            handle.write("SPEAKER input 1 0.250 0.750 <NA> <NA> SPEAKER_00 <NA> <NA>\n")

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_name: str, *, token: str):
            calls["from_pretrained"] = {"model_name": model_name, "token": token}
            return cls()

        def __call__(self, path: str) -> FakeOutput:
            calls["path"] = path
            return FakeOutput()

    fake_audio = ModuleType("pyannote.audio")
    fake_audio.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    get_settings.cache_clear()

    config = PipelineConfig(
        name="test",
        stages={"diarization": {"enabled": True, "model": "pyannote/current-diarization"}},
    )

    try:
        artifacts = DiarizationStage(config).run(input_path, tmp_path)
    finally:
        get_settings.cache_clear()

    assert calls == {
        "from_pretrained": {"model_name": "pyannote/current-diarization", "token": "hf_test"},
        "path": str(input_path),
    }
    assert artifacts["diarization_json"] == tmp_path / "diarization.json"

    diarization = json.loads((tmp_path / "diarization.json").read_text(encoding="utf-8"))
    assert diarization["speaker_count"] == 2
    assert diarization["segments"] == [
        {"start": 0.25, "end": 1.0, "speaker": "SPEAKER_00", "track": None},
        {"start": 1.2, "end": 2.0, "speaker": "SPEAKER_01", "track": None},
    ]


def test_diarization_stage_requires_hugging_face_token(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    get_settings.cache_clear()

    try:
        with pytest.raises(StageError, match="requires HF_TOKEN"):
            DiarizationStage(PipelineConfig(name="test")).run(input_path, tmp_path)
    finally:
        get_settings.cache_clear()


def test_diarization_stage_reuses_deepgram_artifacts_from_transcription(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    monkeypatch.setenv("TRANSCRIPTION_PROVIDER", "deepgram")
    (tmp_path / "diarization.json").write_text('{"segments": []}', encoding="utf-8")
    (tmp_path / "diarization.rttm").write_text("", encoding="utf-8")
    get_settings.cache_clear()

    try:
        artifacts = DiarizationStage(PipelineConfig(name="test")).run(input_path, tmp_path)
    finally:
        get_settings.cache_clear()

    assert artifacts == {
        "diarization_json": tmp_path / "diarization.json",
        "diarization_rttm": tmp_path / "diarization.rttm",
    }


def test_diarization_stage_requires_deepgram_transcription_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    monkeypatch.setenv("TRANSCRIPTION_PROVIDER", "deepgram")
    get_settings.cache_clear()

    try:
        with pytest.raises(StageError, match="did not produce expected artifact"):
            DiarizationStage(PipelineConfig(name="test")).run(input_path, tmp_path)
    finally:
        get_settings.cache_clear()
