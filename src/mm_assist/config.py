from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]


class WindowCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title_substr: str = "瑪奇"
    client_size: tuple[int, int] = (1920, 1440)
    dpi_aware: bool = True
    capture_method: str = "auto"
    background: bool = True
    input_trace: bool = False


class PathsCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")

    knowledge: str = "knowledge"
    templates: str = "knowledge/templates"
    captures: str = "knowledge/captures"
    logs: str = "logs"


class PerceptionCfg(BaseModel):
    ocr_min_confidence: float = 0.55
    template_threshold: float = 0.88
    min_peak_margin: float = 0.05
    allow_ocr_click: bool = False
    single_scale: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    window: WindowCfg = Field(default_factory=WindowCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)
    perception: PerceptionCfg = Field(default_factory=PerceptionCfg)
    read_only_only: bool = True
    # Testing phase pauses an incomplete dispatched plan; later operation may skip it.
    incomplete_plan_policy: Literal["pause", "skip"] = "pause"

    def resolve(self, rel: str | Path) -> Path:
        p = Path(rel)
        if p.is_absolute():
            return p
        return ROOT / p


def load_config(path: Path | None = None) -> AppConfig:
    import yaml

    cfg_path = path or ROOT / "config.yaml"
    raw: dict[str, Any] = {}
    if cfg_path.is_file():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)
