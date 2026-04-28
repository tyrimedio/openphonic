from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"


@dataclass(frozen=True)
class TargetFormat:
    sample_rate: int = 48000
    channels: int = 2
    codec: str = "aac"
    container: str = "m4a"
    bitrate: str = "160k"


@dataclass(frozen=True)
class PipelinePreset:
    id: str
    label: str
    description: str
    path: Path

    def to_dict(self) -> dict[str, str]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    target: TargetFormat = field(default_factory=TargetFormat)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path) -> PipelineConfig:
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        target = TargetFormat(**(raw.get("target") or {}))
        stages = raw.get("stages") or {}
        return cls(name=raw.get("name", "default"), target=target, stages=stages)

    def stage(self, name: str) -> dict[str, Any]:
        return self.stages.get(name, {})

    def enabled(self, name: str, default: bool = False) -> bool:
        return bool(self.stage(name).get("enabled", default))


BUILTIN_PRESETS = (
    PipelinePreset(
        id="podcast-default",
        label="Podcast default",
        description="FFmpeg ingest, conservative silence trim, and loudness normalization.",
        path=CONFIG_ROOT / "default.yml",
    ),
    PipelinePreset(
        id="speech-cleanup",
        label="Speech cleanup",
        description="Podcast default plus optional DeepFilterNet speech enhancement.",
        path=CONFIG_ROOT / "presets" / "speech-cleanup.yml",
    ),
    PipelinePreset(
        id="vocal-isolation",
        label="Vocal isolation",
        description="Podcast default plus optional Demucs vocal isolation for mixed audio.",
        path=CONFIG_ROOT / "presets" / "vocal-isolation.yml",
    ),
)


def available_presets(default_path: str | Path | None = None) -> list[PipelinePreset]:
    default_config_path = Path(default_path).expanduser() if default_path is not None else None
    presets = list(BUILTIN_PRESETS)
    if default_config_path is not None:
        presets[0] = PipelinePreset(
            id=presets[0].id,
            label=presets[0].label,
            description=presets[0].description,
            path=default_config_path,
        )
    return presets


def preset_by_id(
    preset_id: str,
    *,
    default_path: str | Path | None = None,
) -> PipelinePreset:
    for preset in available_presets(default_path):
        if preset.id == preset_id:
            return preset
    raise ValueError(f"Unknown pipeline preset: {preset_id}")


def load_pipeline_config_for_preset(
    preset_id: str | None,
    *,
    default_path: str | Path,
) -> PipelineConfig:
    if not preset_id:
        return PipelineConfig.from_path(default_path)
    return PipelineConfig.from_path(preset_by_id(preset_id, default_path=default_path).path)
