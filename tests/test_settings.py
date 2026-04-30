import pytest

from openphonic.core.settings import get_settings


def test_get_settings_reads_transcription_provider(monkeypatch) -> None:
    monkeypatch.setenv("TRANSCRIPTION_PROVIDER", "Deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg_test")
    get_settings.cache_clear()

    try:
        settings = get_settings()
    finally:
        get_settings.cache_clear()

    assert settings.transcription_provider == "deepgram"
    assert settings.deepgram_api_key == "dg_test"


def test_get_settings_rejects_unknown_transcription_provider(monkeypatch) -> None:
    monkeypatch.setenv("TRANSCRIPTION_PROVIDER", "hosted")
    get_settings.cache_clear()

    try:
        with pytest.raises(ValueError, match="TRANSCRIPTION_PROVIDER must be one of"):
            get_settings()
    finally:
        get_settings.cache_clear()
