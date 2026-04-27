from pathlib import Path

from openphonic.pipeline.config import PipelineConfig


def test_default_config_loads() -> None:
    config = PipelineConfig.from_path(Path("config/default.yml"))

    assert config.name == "podcast-default"
    assert config.target.sample_rate == 48000
    assert config.enabled("loudness") is True
    assert config.enabled("transcription") is False
