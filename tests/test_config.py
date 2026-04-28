from pathlib import Path

import pytest

from openphonic.core.settings import DEFAULT_PIPELINE_CONFIG
from openphonic.pipeline.config import (
    CONFIG_ROOT,
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
    presets = available_presets(DEFAULT_PIPELINE_CONFIG)
    speech_cleanup = load_pipeline_config_for_preset(
        "speech-cleanup",
        default_path=DEFAULT_PIPELINE_CONFIG,
    )
    vocal_isolation = load_pipeline_config_for_preset(
        "vocal-isolation",
        default_path=DEFAULT_PIPELINE_CONFIG,
    )

    assert [preset.id for preset in presets] == [
        "podcast-default",
        "speech-cleanup",
        "vocal-isolation",
    ]
    assert DEFAULT_PIPELINE_CONFIG.exists()
    assert presets[1].path.is_relative_to(CONFIG_ROOT)
    assert presets[2].path.is_relative_to(CONFIG_ROOT)
    assert presets[1].path.exists()
    assert presets[2].path.exists()
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
        preset_by_id("missing", default_path=DEFAULT_PIPELINE_CONFIG)
