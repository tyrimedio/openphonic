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


def test_custom_presets_are_discovered_and_loaded(tmp_path) -> None:
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()
    (preset_dir / "daily-show.yml").write_text(
        """
preset:
  label: Daily show
  description: Daily show production preset.
name: daily-show
target:
  sample_rate: 48000
  channels: 2
stages:
  silence_trim:
    enabled: false
  loudness:
    enabled: true
""",
        encoding="utf-8",
    )
    (preset_dir / "ignored preset.yml").write_text("name: ignored\n", encoding="utf-8")
    (preset_dir / "broken.yml").write_text("name: [broken\n", encoding="utf-8")
    (preset_dir / "bad-target.yml").write_text(
        """
name: bad-target
target:
  unsupported: true
""",
        encoding="utf-8",
    )

    presets = available_presets(DEFAULT_PIPELINE_CONFIG, preset_dir)
    custom = preset_by_id(
        "custom:daily-show",
        default_path=DEFAULT_PIPELINE_CONFIG,
        preset_dir=preset_dir,
    )
    config = load_pipeline_config_for_preset(
        "custom:daily-show",
        default_path=DEFAULT_PIPELINE_CONFIG,
        preset_dir=preset_dir,
    )

    assert [preset.id for preset in presets] == [
        "podcast-default",
        "speech-cleanup",
        "vocal-isolation",
        "custom:daily-show",
    ]
    assert custom.label == "Daily show"
    assert custom.description == "Daily show production preset."
    assert custom.path == preset_dir / "daily-show.yml"
    assert config.name == "daily-show"
    assert config.enabled("silence_trim", default=True) is False
    assert config.enabled("loudness") is True
    with pytest.raises(ValueError, match="Unknown pipeline preset"):
        preset_by_id(
            "custom:broken",
            default_path=DEFAULT_PIPELINE_CONFIG,
            preset_dir=preset_dir,
        )


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown pipeline preset"):
        preset_by_id("missing", default_path=DEFAULT_PIPELINE_CONFIG)
