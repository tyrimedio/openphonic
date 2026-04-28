from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
CUSTOM_PRESET_ID = re.compile(r"^[A-Za-z0-9._-]+$")


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


def available_presets(
    default_path: str | Path | None = None,
    preset_dir: str | Path | None = None,
) -> list[PipelinePreset]:
    default_config_path = Path(default_path).expanduser() if default_path is not None else None
    presets = list(BUILTIN_PRESETS)
    if default_config_path is not None:
        presets[0] = PipelinePreset(
            id=presets[0].id,
            label=presets[0].label,
            description=presets[0].description,
            path=default_config_path,
        )
    presets.extend(_custom_presets(preset_dir))
    return presets


def preset_by_id(
    preset_id: str,
    *,
    default_path: str | Path | None = None,
    preset_dir: str | Path | None = None,
) -> PipelinePreset:
    for preset in available_presets(default_path, preset_dir):
        if preset.id == preset_id:
            return preset
    raise ValueError(f"Unknown pipeline preset: {preset_id}")


def load_pipeline_config_for_preset(
    preset_id: str | None,
    *,
    default_path: str | Path,
    preset_dir: str | Path | None = None,
) -> PipelineConfig:
    if not preset_id:
        return PipelineConfig.from_path(default_path)
    return PipelineConfig.from_path(
        preset_by_id(preset_id, default_path=default_path, preset_dir=preset_dir).path
    )


def _custom_presets(preset_dir: str | Path | None) -> list[PipelinePreset]:
    if preset_dir is None:
        return []
    directory = Path(preset_dir).expanduser()
    if not directory.exists():
        return []
    if not directory.is_dir():
        return []

    presets: list[PipelinePreset] = []
    seen: set[str] = {preset.id for preset in BUILTIN_PRESETS}
    candidates = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix in {".yml", ".yaml"}
    )
    for path in candidates:
        stem = path.stem
        if not CUSTOM_PRESET_ID.fullmatch(stem):
            continue
        preset_id = f"custom:{stem}"
        if preset_id in seen:
            continue
        seen.add(preset_id)
        label, description = _custom_preset_metadata(path)
        presets.append(
            PipelinePreset(
                id=preset_id,
                label=label,
                description=description,
                path=path,
            )
        )
    return presets


def _custom_preset_metadata(path: Path) -> tuple[str, str]:
    label = _label_from_stem(path.stem)
    description = f"Custom preset from {path.name}."
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return label, description
    if not isinstance(raw, dict):
        return label, description

    preset_metadata = raw.get("preset")
    if isinstance(preset_metadata, dict):
        configured_label = preset_metadata.get("label")
        configured_description = preset_metadata.get("description")
    else:
        configured_label = raw.get("label")
        configured_description = raw.get("description")

    if isinstance(configured_label, str) and configured_label.strip():
        label = configured_label.strip()
    if isinstance(configured_description, str) and configured_description.strip():
        description = configured_description.strip()
    return label, description


def _label_from_stem(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip().title() or stem
