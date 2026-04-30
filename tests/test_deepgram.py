import json
from urllib.parse import parse_qs, urlparse

import pytest

from openphonic.pipeline.deepgram import (
    DeepgramError,
    DeepgramOptions,
    deepgram_response_to_transcript,
    transcribe_deepgram_file,
)


def test_transcribe_deepgram_file_posts_prerecorded_audio_request(
    tmp_path,
    monkeypatch,
) -> None:
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")
    calls: dict[str, object] = {}

    class FakeResponse:
        status = 200
        reason = "OK"

        def read(self) -> bytes:
            return json.dumps({"results": {"channels": []}}).encode()

    class FakeConnection:
        def __init__(self, host: str, port: int | None = None, *, timeout: int) -> None:
            calls["connection"] = {"host": host, "port": port, "timeout": timeout}
            self.headers: dict[str, str] = {}
            self.body = bytearray()

        def putrequest(self, method: str, path: str) -> None:
            calls["request"] = {"method": method, "path": path}

        def putheader(self, name: str, value: str) -> None:
            self.headers[name] = value
            calls["headers"] = self.headers

        def endheaders(self) -> None:
            calls["headers_ended"] = True

        def send(self, chunk: bytes) -> None:
            self.body.extend(chunk)
            calls["body"] = bytes(self.body)

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr("openphonic.pipeline.deepgram.http.client.HTTPSConnection", FakeConnection)

    payload = transcribe_deepgram_file(
        audio_path,
        DeepgramOptions(
            api_key="dg_test",
            model="nova-3",
            language="en",
            diarize=True,
            endpoint="https://api.deepgram.com/v1/listen",
            timeout_seconds=12,
        ),
    )

    assert payload == {"results": {"channels": []}}
    assert calls["connection"] == {"host": "api.deepgram.com", "port": None, "timeout": 12}
    assert calls["request"]["method"] == "POST"
    request_path = calls["request"]["path"]
    assert request_path.startswith("/v1/listen?")
    assert "model=nova-3" in request_path
    assert "smart_format=true" in request_path
    assert "utterances=true" in request_path
    assert "diarize=true" in request_path
    assert "language=en" in request_path
    assert "detect_language=true" not in request_path
    assert calls["headers"]["Authorization"] == "Token dg_test"
    assert calls["headers"]["Content-Type"] in {"audio/x-wav", "audio/wav"}
    assert calls["headers"]["Content-Length"] == "5"
    assert calls["body"] == b"audio"
    assert calls["closed"] is True


def test_transcribe_deepgram_file_detects_language_when_not_configured(
    tmp_path,
    monkeypatch,
) -> None:
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")
    calls: dict[str, object] = {}

    class FakeResponse:
        status = 200
        reason = "OK"

        def read(self) -> bytes:
            return json.dumps({"results": {"channels": []}}).encode()

    class FakeConnection:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def putrequest(self, method: str, path: str) -> None:
            calls["request"] = {"method": method, "path": path}

        def putheader(self, name: str, value: str) -> None:
            pass

        def endheaders(self) -> None:
            pass

        def send(self, chunk: bytes) -> None:
            pass

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr("openphonic.pipeline.deepgram.http.client.HTTPSConnection", FakeConnection)

    transcribe_deepgram_file(
        audio_path,
        DeepgramOptions(api_key="dg_test", model="nova-3", language=None),
    )

    query = parse_qs(urlparse(calls["request"]["path"]).query)
    assert query["detect_language"] == ["true"]
    assert "language" not in query


def test_transcribe_deepgram_file_reports_http_errors(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")

    class FakeResponse:
        status = 401
        reason = "Unauthorized"

        def read(self) -> bytes:
            return b'{"err_msg":"bad key"}'

    class FakeConnection:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def putrequest(self, method: str, path: str) -> None:
            pass

        def putheader(self, name: str, value: str) -> None:
            pass

        def endheaders(self) -> None:
            pass

        def send(self, chunk: bytes) -> None:
            pass

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr("openphonic.pipeline.deepgram.http.client.HTTPSConnection", FakeConnection)

    with pytest.raises(DeepgramError, match="HTTP 401"):
        transcribe_deepgram_file(audio_path, DeepgramOptions(api_key="dg_test"))


def test_deepgram_transcript_uses_channel_detected_language() -> None:
    transcript = deepgram_response_to_transcript(
        {
            "metadata": {"duration": 1.2},
            "results": {
                "channels": [
                    {
                        "detected_language": "es",
                        "language_confidence": 0.87,
                        "alternatives": [
                            {
                                "transcript": "Hola mundo.",
                                "confidence": 0.94,
                                "words": [
                                    {
                                        "word": "hola",
                                        "punctuated_word": "Hola",
                                        "start": 0.0,
                                        "end": 0.4,
                                        "confidence": 0.95,
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        },
        model="nova-3",
        language=None,
    )

    assert transcript["language"] == "es"
    assert transcript["language_probability"] == 0.87
    assert transcript["segments"][0]["text"] == "Hola mundo."
