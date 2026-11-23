"""Register matting model config keys → YAML files under configs/auto_labeling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import anylabeling


@dataclass(frozen=True)
class MattingModelSpec:
    key: str
    display_name: str
    yaml_name: str
    default_for_fine: bool = False
    default_for_fast: bool = False


def _config_dir() -> Path:
    return Path(anylabeling.__file__).resolve().parent / "configs" / "auto_labeling"


def config_path_for(yaml_name: str) -> Path:
    return _config_dir() / yaml_name


MATTING_MODELS: Tuple[MattingModelSpec, ...] = (
    MattingModelSpec(
        key="rmbg_v20",
        display_name="RMBG v2.0",
        yaml_name="rmbg_v20.yaml",
        default_for_fine=True,
    ),
    MattingModelSpec(
        key="rmbg_v20_quant",
        display_name="RMBG v2.0 Quantized",
        yaml_name="rmbg_v20_quant.yaml",
        default_for_fast=True,
    ),
    MattingModelSpec(
        key="rmbg_v14",
        display_name="RMBG v1.4",
        yaml_name="rmbg_v14.yaml",
    ),
)

MODEL_BY_KEY: Dict[str, MattingModelSpec] = {m.key: m for m in MATTING_MODELS}


def default_model_key(*, fine: bool) -> str:
    for spec in MATTING_MODELS:
        if fine and spec.default_for_fine:
            return spec.key
        if not fine and spec.default_for_fast:
            return spec.key
    return MATTING_MODELS[0].key


def list_model_keys() -> List[str]:
    return [m.key for m in MATTING_MODELS]