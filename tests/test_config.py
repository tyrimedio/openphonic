from pathlib import Path

import pytest

from openphonic.pipeline.config import (
    PipelineConfig,
    available_presets,
    load_pipeline_config_for_preset,
    preset_by_id,
)


def test_default_config_loads() -> None:
    config = PipelineConfig.from_path(Path("config/default.yml"))

    assert config.name == "podcast-default"
    assert config.target.sample_rate == 48000
    assert config.enabled("loudness") is True
    assert config.enabled("transcription") is False


def test_builtin_presets_load_conservative_audio_polish_configs() -> None:
    presets = available_presets(Path("config/default.yml"))
    speech_cleanup = load_pipeline_config_for_preset(
        "speech-cleanup",
        default_path=Path("config/default.yml"),
    )
    vocal_isolation = load_pipeline_config_for_preset(
        "vocal-isolation",
        default_path=Path("config/default.yml"),
    )

    assert [preset.id for preset in presets] == [
        "podcast-default",
        "speech-cleanup",
        "vocal-isolation",
    ]
    assert speech_cleanup.name == "speech-cleanup"
    assert speech_cleanup.enabled("noise_reduction") is True
    assert speech_cleanup.stage("noise_reduction")["attenuation_db"] == 8
    assert speech_cleanup.enabled("music_separation") is False
    assert vocal_isolation.name == "vocal-isolation"
    assert vocal_isolation.enabled("music_separation") is True
    assert vocal_isolation.stage("music_separation")["stem"] == "vocals"
    assert vocal_isolation.enabled("noise_reduction") is False


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown pipeline preset"):
        preset_by_id("missing", default_path=Path("config/default.yml"))
