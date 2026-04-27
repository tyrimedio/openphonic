from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TargetFormat:
    sample_rate: int = 48000
    channels: int = 2
    codec: str = "aac"
    container: str = "m4a"
    bitrate: str = "160k"


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
