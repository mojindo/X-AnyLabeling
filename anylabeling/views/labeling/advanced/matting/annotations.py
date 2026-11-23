"""Load label shapes for matting export."""
from __future__ import annotations

import json
import os.path as osp
from dataclasses import dataclass
from typing import List, Optional, Set

from anylabeling.views.labeling.classifier.utils import get_label_file_path


@dataclass(frozen=True)
class MattingShape:
    label: str
    shape_type: str
    points: List[List[float]]
    index: int


def _load_label_json(label_path: str) -> Optional[dict]:
    if not label_path or not osp.isfile(label_path):
        return None
    try:
        with open(label_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def iter_matting_shapes(
    image_path: str,
    *,
    output_dir: Optional[str],
    allowed_labels: Optional[Set[str]],
) -> List[MattingShape]:
    label_path = get_label_file_path(image_path, output_dir)
    data = _load_label_json(label_path)
    if not data:
        return []

    allowed = None if allowed_labels is None else {str(x).strip() for x in allowed_labels if str(x).strip()}
    out: List[MattingShape] = []
    for idx, shape in enumerate(data.get("shapes") or []):
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label") or "").strip()
        if not label:
            continue
        if allowed is not None and label not in allowed:
            continue
        st = str(shape.get("shape_type") or shape.get("type") or "").strip()
        if not st:
            continue
        pts = []
        for p in shape.get("points") or []:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append([float(p[0]), float(p[1])])
        if len(pts) < 1:
            continue
        if st in ("rectangle", "quadrilateral") and len(pts) < 2:
            continue
        if st == "rotation" and len(pts) < 4:
            continue
        if st == "polygon" and len(pts) < 3:
            continue
        out.append(MattingShape(label=label, shape_type=st, points=pts, index=idx))
    return out


def image_hw_from_label(
    image_path: str, *, output_dir: Optional[str], image_bgr
) -> tuple[int, int]:
    label_path = get_label_file_path(image_path, output_dir)
    data = _load_label_json(label_path)
    if data:
        try:
            iw = int(data.get("imageWidth") or 0)
            ih = int(data.get("imageHeight") or 0)
            if iw > 0 and ih > 0:
                return iw, ih
        except Exception:
            pass
    h, w = image_bgr.shape[:2]
    return w, h