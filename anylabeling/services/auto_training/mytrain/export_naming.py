"""ONNX export filename helpers for Ultralytics training runs."""

from __future__ import annotations

import os
import re
from typing import Optional


def sanitize_name_part(value: str, *, default: str = "x") -> str:
    """Normalize a filename segment: spaces→_, drop path separators."""
    text = (value or "").strip()
    text = text.replace("\\", "_").replace("/", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or default


def model_stem_from_path(model_path: str) -> str:
    """``/path/yolo26n.pt`` → ``yolo26n`` (never best/last)."""
    base = os.path.basename(model_path or "")
    stem, _ = os.path.splitext(base)
    stem = sanitize_name_part(stem, default="model")
    if stem.lower() in ("best", "last"):
        return "model"
    return stem


def build_onnx_export_name(
    model: str,
    imgsz: int,
    num_classes: int,
    project: str,
    subdir: str,
    version: str,
) -> str:
    """
    ``{model}_{imgsz}_{N}cls_{project}_{subdir}_{version}.onnx``

    Example: ``yolo26n_1280_8cls_engineering_base_0_0_scene_v3.onnx``
    """
    m = sanitize_name_part(model, default="model")
    p = sanitize_name_part(project, default="project")
    s = sanitize_name_part(subdir, default="data")
    v = sanitize_name_part(version or "v1", default="v1")
    try:
        size = int(imgsz)
    except (TypeError, ValueError):
        size = 640
    try:
        ncls = max(0, int(num_classes))
    except (TypeError, ValueError):
        ncls = 0
    return f"{m}_{size}_{ncls}cls_{p}_{s}_{v}.onnx"


def resolve_export_meta(
    *,
    model_path: str = "",
    imgsz: int = 640,
    num_classes: int = 0,
    dataset_root: str = "",
    subdirs: Optional[list] = None,
) -> dict:
    """Build default fields for the export dialog."""
    root_name = os.path.basename((dataset_root or "").rstrip("/\\")) or "project"
    sub = "-".join(subdirs or []) if subdirs else "data"
    return {
        "model_stem": model_stem_from_path(model_path),
        "imgsz": int(imgsz) if imgsz else 640,
        "num_classes": int(num_classes) if num_classes else 0,
        "project": root_name,
        "subdir": sub,
        "version": "v1",
    }
