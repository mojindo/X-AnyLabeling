"""Run dataset analysis pipeline for a single image."""

from __future__ import annotations

import json
import os.path as osp
from typing import Any, Dict, Optional

import cv2

from anylabeling.views.labeling.classifier.utils import get_label_file_path
from anylabeling.views.labeling.advanced.dataset_analysis.analysis_store import (
    ANALYSIS_KEY,
    merge_analysis_into_label_json,
)
from anylabeling.views.labeling.advanced.dataset_analysis.metrics import compute_image_metrics
from anylabeling.views.labeling.advanced.dataset_analysis.vlm import generate_caption


def analyze_one_image(
    image_path: str,
    output_dir: str | None,
    *,
    max_side: int,
    run_metrics: bool,
    run_vlm: bool,
    vlm_overwrite_existing: bool,
    api_base: str,
    api_key: str,
    model: str,
    vlm_prompt: str,
    vlm_max_tokens: int,
) -> Dict[str, Any]:
    """
    Analyze one image, merge results into label JSON, return analysis block.
    """
    base = osp.basename(image_path)
    label_path = get_label_file_path(image_path, output_dir)

    metrics = None
    vlm_block = None
    per_image_error = None

    try:
        if run_metrics:
            metrics = compute_image_metrics(image_path, max_side=max_side)
            ih = metrics.get("original_height")
            iw = metrics.get("original_width")
        else:
            ih = iw = None
            im = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if im is not None:
                ih, iw = int(im.shape[0]), int(im.shape[1])
    except Exception as e:
        ih = iw = None
        per_image_error = f"metrics_error: {e}"

    if run_vlm:
        try:
            should_call_vlm = True
            if not vlm_overwrite_existing and osp.isfile(label_path):
                try:
                    with open(label_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    block0 = existing.get(ANALYSIS_KEY)
                    if isinstance(block0, dict):
                        vlm0 = block0.get("vlm")
                        if isinstance(vlm0, dict):
                            cap0 = vlm0.get("caption")
                            if isinstance(cap0, str) and cap0.strip():
                                should_call_vlm = False
                except Exception:
                    should_call_vlm = True

            if should_call_vlm:
                ak = (api_key or "").strip()
                if not ak and (
                    "localhost" in (api_base or "").lower()
                    or "127.0.0.1" in (api_base or "")
                ):
                    ak = "ollama"
                if not ak:
                    raise ValueError("VLM enabled but API key is empty.")
                vlm_block = generate_caption(
                    image_path,
                    api_key=ak,
                    base_url=api_base or "https://api.openai.com/v1",
                    model=model,
                    prompt=vlm_prompt,
                    max_side=max_side,
                    max_tokens=vlm_max_tokens,
                )
        except Exception as e:
            per_image_error = (
                (per_image_error + "; " if per_image_error else "") + f"vlm_error: {e}"
            )

    merge_analysis_into_label_json(
        label_path,
        metrics=metrics if run_metrics else None,
        vlm=vlm_block if run_vlm else None,
        skip_existing=False,
        create_if_missing=True,
        image_basename=base,
        image_width=iw,
        image_height=ih,
    )

    if per_image_error:
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            block = data.get(ANALYSIS_KEY) or {}
            if isinstance(block, dict):
                block["error"] = per_image_error
                data[ANALYSIS_KEY] = block
                with open(label_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    return load_analysis_block(label_path)


def load_analysis_block(label_json_path: str) -> Dict[str, Any]:
    """Read xal_dataset_analysis from label JSON."""
    if not osp.isfile(label_json_path):
        return {}
    try:
        with open(label_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        block = data.get(ANALYSIS_KEY)
        return dict(block) if isinstance(block, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
