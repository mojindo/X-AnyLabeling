from __future__ import annotations

import os
import os.path as osp
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from anylabeling.config import get_work_directory
from anylabeling.views.labeling.label_converter import LabelConverter


def _shape_to_mask(
    shape: Dict[str, Any],
    *,
    h: int,
    w: int,
) -> Optional[np.ndarray]:
    st = shape.get("shape_type")
    pts = shape.get("points") or []
    if not isinstance(pts, list) or len(pts) < 2:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    try:
        if st in ("rectangle", "rotation") and len(pts) >= 2:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            xmin, xmax = int(round(min(x1, x2))), int(round(max(x1, x2)))
            ymin, ymax = int(round(min(y1, y2))), int(round(max(y1, y2)))
            xmin = max(0, min(xmin, w - 1))
            xmax = max(0, min(xmax, w - 1))
            ymin = max(0, min(ymin, h - 1))
            ymax = max(0, min(ymax, h - 1))
            cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, thickness=-1)
            return mask
        if st == "circle" and len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[0], pts[1]
            cx, cy = int(round(x1)), int(round(y1))
            r = int(round(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))
            cv2.circle(mask, (cx, cy), max(1, r), 255, thickness=-1)
            return mask
        if st in ("polygon", "quadrilateral") and len(pts) >= 3:
            arr = np.array(pts, dtype=np.float32)
            arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
            arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
            poly = arr.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [poly], 255)
            return mask
    except Exception:
        return None
    return None


def compose_masked_image(
    image_path: str,
    shapes: List[Dict[str, Any]],
    *,
    fill_bgr: Tuple[int, int, int] = (30, 30, 30),
    out_dir: Optional[str] = None,
) -> str:
    """
    Create an image where all annotated regions are replaced by a solid color.
    Returns the saved masked image path (PNG).
    """
    bgr, _size = LabelConverter.imread_unicode(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = bgr.shape[:2]

    union = np.zeros((h, w), dtype=np.uint8)
    for s in shapes:
        if not isinstance(s, dict):
            continue
        m = _shape_to_mask(s, h=h, w=w)
        if m is None:
            continue
        union = cv2.bitwise_or(union, m)

    out = bgr.copy()
    out[union > 0] = np.array(fill_bgr, dtype=np.uint8)

    if out_dir is None:
        root = get_work_directory()
        out_dir = osp.join(root, "xanylabeling_data", "annotation_inspector", "masked")
    os.makedirs(out_dir, exist_ok=True)
    fn = f"masked_{uuid.uuid4().hex}.png"
    out_path = osp.join(out_dir, fn)
    ok = cv2.imwrite(out_path, out)
    if not ok:
        raise ValueError("Failed to write masked image.")
    return out_path

