"""Read/write XLABEL JSON with dataset analysis block.

Named analysis_store to avoid shadowing the stdlib ``io`` module.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from anylabeling.app_info import __version__
from anylabeling.views.labeling.schema import create_xlabel_template

ANALYSIS_KEY = "xal_dataset_analysis"


def merge_analysis_into_label_json(
    label_json_path: str,
    *,
    metrics: Optional[Dict[str, Any]] = None,
    vlm: Optional[Dict[str, Any]] = None,
    skip_existing: bool = False,
    create_if_missing: bool = True,
    image_basename: Optional[str] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> None:
    """Merge xal_dataset_analysis into label JSON (create minimal JSON if needed)."""
    if osp.exists(label_json_path):
        with open(label_json_path, "r", encoding="utf-8-sig") as f:
            raw = f.read()
        stripped = raw.strip()
        if not stripped:
            if not create_if_missing:
                return
            if not image_basename:
                raise ValueError(
                    f"Label file is empty ({label_json_path}); "
                    "image_basename is required to create label content."
                )
            data = create_xlabel_template(
                version=__version__,
                flags={},
                shapes=[],
                image_path=image_basename,
                image_data=None,
                image_height=image_height if image_height is not None else -1,
                image_width=image_width if image_width is not None else -1,
            )
        else:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in label file ({label_json_path}): {e}"
                ) from e
        if skip_existing and data.get(ANALYSIS_KEY):
            return
    else:
        if not create_if_missing:
            return
        if not image_basename:
            raise ValueError("image_basename required to create label file")
        data = create_xlabel_template(
            version=__version__,
            flags={},
            shapes=[],
            image_path=image_basename,
            image_data=None,
            image_height=image_height if image_height is not None else -1,
            image_width=image_width if image_width is not None else -1,
        )

    prior = data.get(ANALYSIS_KEY)
    if isinstance(prior, dict):
        block = dict(prior)
    else:
        block = {}
    block["schema_version"] = 1
    block["generated_at"] = datetime.now(timezone.utc).isoformat()
    if metrics is not None:
        block["metrics"] = metrics
    if vlm is not None:
        block["vlm"] = vlm
    data[ANALYSIS_KEY] = block

    out_dir = osp.dirname(label_json_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(label_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
