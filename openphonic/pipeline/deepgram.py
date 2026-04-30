from __future__ import annotations

import http.client
import json
import math
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

DEFAULT_DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"
DEFAULT_DEEPGRAM_MODEL = "nova-3"
DEFAULT_DEEPGRAM_TIMEOUT_SECONDS = 650
_CHUNK_BYTES = 1024 * 1024


class DeepgramError(RuntimeError):
    """Raised when the Deepgram provider cannot produce a valid transcript."""


@dataclass(frozen=True)
class DeepgramOptions:
    api_key: str
    model: str = DEFAULT_DEEPGRAM_MODEL
    language: str | None = None
    diarize: bool = False
    endpoint: str = DEFAULT_DEEPGRAM_API_URL
    timeout_seconds: int = DEFAULT_DEEPGRAM_TIMEOUT_SECONDS


def transcribe_deepgram_file(audio_path: Path, options: DeepgramOptions) -> dict[str, Any]:
    if not options.api_key:
        raise DeepgramError("Deepgram transcription provider requires DEEPGRAM_API_KEY.")
    params = {
        "model": options.model,
        "smart_format": "true",
        "utterances": "true",
    }
    if options.diarize:
        params["diarize"] = "true"
    if options.language:
        params["language"] = options.language
    return _post_audio_file(
        endpoint=options.endpoint,
        api_key=options.api_key,
        params=params,
        audio_path=audio_path,
        timeout_seconds=options.timeout_seconds,
    )


def deepgram_response_to_transcript(
    response: dict[str, Any],
    *,
    model: str,
    language: str | None,
) -> dict[str, Any]:
    alternative = _primary_alternative(response)
    utterances = _response_utterances(response)
    segments = (
        [_utterance_to_segment(index, utterance) for index, utterance in enumerate(utterances)]
        if utterances
        else _alternative_segments(alternative)
    )
    metadata = _mapping(response.get("metadata"))
    return {
        "schema_version": 1,
        "engine": "deepgram",
        "model": model,
        "language": alternative.get("language") or language,
        "language_probability": alternative.get("language_confidence"),
        "duration": _finite_float(metadata.get("duration")),
        "segments": segments,
    }


def deepgram_response_to_diarization(
    response: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    utterances = _response_utterances(response)
    if any(
        _utterance_has_speech(utterance) and _speaker_label(utterance.get("speaker")) is None
        for utterance in utterances
    ):
        raise DeepgramError("Deepgram diarization response did not include speaker labels.")
    segments = _diarization_segments_from_utterances(utterances)
    if not segments:
        segments = _diarization_segments_from_words(_primary_alternative(response))
    speakers = {segment["speaker"] for segment in segments}
    if _has_speech_units(response) and not speakers:
        raise DeepgramError("Deepgram diarization response did not include speaker labels.")
    return {
        "schema_version": 1,
        "engine": "deepgram",
        "model": model,
        "speaker_count": len(speakers),
        "segments": segments,
    }


def diarization_to_rttm(diarization: dict[str, Any], *, source_name: str) -> str:
    file_id = _rttm_file_id(source_name)
    lines: list[str] = []
    for segment in diarization.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        start = _finite_float(segment.get("start"))
        end = _finite_float(segment.get("end"))
        speaker = str(segment.get("speaker") or "")
        if start is None or end is None or end < start or not speaker:
            continue
        lines.append(
            f"SPEAKER {file_id} 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _post_audio_file(
    *,
    endpoint: str,
    api_key: str,
    params: dict[str, str],
    audio_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise DeepgramError(f"Invalid Deepgram endpoint: {endpoint}")

    query = urlencode(params)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}&{query}"
    else:
        path = f"{path}?{query}"

    content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    try:
        content_length = audio_path.stat().st_size
    except OSError as exc:
        raise DeepgramError(f"Deepgram input audio cannot be read: {audio_path}") from exc

    connection_class = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    connection = connection_class(parsed.hostname, parsed.port, timeout=timeout_seconds)
    try:
        connection.putrequest("POST", path)
        connection.putheader("Authorization", f"Token {api_key}")
        connection.putheader("Content-Type", content_type)
        connection.putheader("Content-Length", str(content_length))
        connection.putheader("Accept", "application/json")
        connection.endheaders()
        with audio_path.open("rb") as handle:
            while chunk := handle.read(_CHUNK_BYTES):
                connection.send(chunk)

        response = connection.getresponse()
        response_body = response.read()
    except OSError as exc:
        raise DeepgramError(f"Deepgram request failed: {exc}") from exc
    finally:
        connection.close()

    body_text = response_body.decode("utf-8", errors="replace")
    if response.status < 200 or response.status >= 300:
        detail = body_text.strip()[:500] or response.reason
        raise DeepgramError(f"Deepgram request failed with HTTP {response.status}: {detail}")

    try:
        payload = json.loads(body_text)
    except json.JSONDecodeError as exc:
        raise DeepgramError("Deepgram response was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise DeepgramError("Deepgram response must be a JSON object.")
    return payload


def _primary_alternative(response: dict[str, Any]) -> dict[str, Any]:
    results = _mapping(response.get("results"))
    channels = results.get("channels")
    if not isinstance(channels, list) or not channels:
        raise DeepgramError("Deepgram response did not include any result channels.")
    if not isinstance(channels[0], dict):
        raise DeepgramError("Deepgram response channel must be a JSON object.")
    channel = channels[0]
    alternatives = channel.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        raise DeepgramError("Deepgram response did not include a transcription alternative.")
    if not isinstance(alternatives[0], dict):
        raise DeepgramError("Deepgram transcription alternative must be a JSON object.")
    return alternatives[0]


def _response_utterances(response: dict[str, Any]) -> list[dict[str, Any]]:
    utterances = _mapping(response.get("results")).get("utterances")
    if not isinstance(utterances, list):
        return []
    return [utterance for utterance in utterances if isinstance(utterance, dict)]


def _utterance_to_segment(index: int, utterance: dict[str, Any]) -> dict[str, Any]:
    start = _finite_float(utterance.get("start"))
    end = _finite_float(utterance.get("end"))
    segment = {
        "id": utterance.get("id") or index + 1,
        "start": 0.0 if start is None else start,
        "end": 0.0 if end is None else end,
        "text": str(utterance.get("transcript") or ""),
        "words": _word_list(utterance.get("words")),
    }
    confidence = _finite_float(utterance.get("confidence"))
    if confidence is not None:
        segment["confidence"] = confidence
    speaker = _speaker_label(utterance.get("speaker"))
    if speaker is not None:
        segment["speaker"] = speaker
    return segment


def _alternative_segments(alternative: dict[str, Any]) -> list[dict[str, Any]]:
    words = _word_list(alternative.get("words"))
    transcript = str(alternative.get("transcript") or "")
    if not transcript and not words:
        return []
    start = words[0]["start"] if words else 0.0
    end = words[-1]["end"] if words else start
    segment: dict[str, Any] = {
        "id": 1,
        "start": start,
        "end": end,
        "text": transcript,
        "words": words,
    }
    confidence = _finite_float(alternative.get("confidence"))
    if confidence is not None:
        segment["confidence"] = confidence
    return [segment]


def _word_list(raw_words: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_words, list):
        return []
    words: list[dict[str, Any]] = []
    for raw_word in raw_words:
        if not isinstance(raw_word, dict):
            continue
        start = _finite_float(raw_word.get("start"))
        end = _finite_float(raw_word.get("end"))
        if start is None or end is None or end < start:
            continue
        word = {
            "start": start,
            "end": end,
            "word": str(raw_word.get("punctuated_word") or raw_word.get("word") or ""),
            "probability": _finite_float(raw_word.get("confidence")),
        }
        speaker = _speaker_label(raw_word.get("speaker"))
        if speaker is not None:
            word["speaker"] = speaker
        speaker_confidence = _finite_float(raw_word.get("speaker_confidence"))
        if speaker_confidence is not None:
            word["speaker_confidence"] = speaker_confidence
        words.append(word)
    return words


def _diarization_segments_from_utterances(
    utterances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for utterance in utterances:
        speaker = _speaker_label(utterance.get("speaker"))
        start = _finite_float(utterance.get("start"))
        end = _finite_float(utterance.get("end"))
        if speaker is None or start is None or end is None or end < start:
            continue
        segment: dict[str, Any] = {
            "start": start,
            "end": end,
            "speaker": speaker,
            "track": None,
        }
        confidence = _finite_float(utterance.get("confidence"))
        if confidence is not None:
            segment["confidence"] = confidence
        segments.append(segment)
    return segments


def _diarization_segments_from_words(alternative: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for word in _word_list(alternative.get("words")):
        speaker = word.get("speaker")
        if not isinstance(speaker, str):
            continue
        if segments and segments[-1]["speaker"] == speaker:
            segments[-1]["end"] = word["end"]
            continue
        segments.append(
            {
                "start": word["start"],
                "end": word["end"],
                "speaker": speaker,
                "track": None,
            }
        )
    return segments


def _has_speech_units(response: dict[str, Any]) -> bool:
    if _response_utterances(response):
        return True
    try:
        return bool(_primary_alternative(response).get("words"))
    except DeepgramError:
        return False


def _utterance_has_speech(utterance: dict[str, Any]) -> bool:
    if str(utterance.get("transcript") or "").strip():
        return True
    return bool(_word_list(utterance.get("words")))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _speaker_label(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        speaker_number = int(value)
    except (TypeError, ValueError):
        label = str(value).strip()
        return label or None
    return f"SPEAKER_{speaker_number:02d}"


def _rttm_file_id(source_name: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "._-" else "_" for character in source_name
    ).strip("._-")
    return normalized or "audio"
