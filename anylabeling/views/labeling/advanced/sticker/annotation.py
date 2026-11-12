from __future__ import annotations

import json
import logging
import os
import os.path as osp
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from anylabeling.views.labeling.classifier.utils import get_label_file_path

logger = logging.getLogger(__name__)


def _load_or_init_label_json(label_path: str, *, image_path: str, w: int, h: int) -> Dict[str, Any]:
    if osp.isfile(label_path):
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                obj.setdefault("shapes", [])
                return obj
        except Exception:
            # Don't silently overwrite a possibly valuable/corrupted file.
            logger.exception("[Sticker] Failed to parse label JSON: %s", label_path)
            try:
                bak_path = label_path + ".bak"
                if osp.isfile(bak_path):
                    # Avoid clobbering an existing backup.
                    bak_path = label_path + ".bak2"
                os.replace(label_path, bak_path)
                logger.warning("[Sticker] Backed up invalid label JSON to: %s", bak_path)
            except Exception:
                logger.exception("[Sticker] Failed to backup invalid label JSON: %s", label_path)
    return {
        "version": "4.0.0",
        "flags": {},
        "shapes": [],
        "imagePath": osp.basename(image_path),
        "imageData": None,
        "imageHeight": int(h),
        "imageWidth": int(w),
    }


def _save_label_json(label_path: str, obj: Dict[str, Any]) -> None:
    # Atomic write to avoid partial files on crash/interrupt.
    dirpath = osp.dirname(label_path) or "."
    os.makedirs(dirpath, exist_ok=True)
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=dirpath)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, label_path)
    finally:
        try:
            if fd is not None:
                os.close(fd)
        except Exception:
            pass
        try:
            if tmp_path and osp.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _mask_to_polygon(mask: np.ndarray) -> Optional[List[List[float]]]:
    if mask is None or mask.size == 0:
        return None
    m = (mask > 0).astype(np.uint8) * 255
    cnts, _hier = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 3:
        return None
    eps = 0.005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    pts = approx.reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    return [[float(x), float(y)] for x, y in pts.tolist()]


def _mask_to_bbox(mask: np.ndarray) -> Optional[List[List[float]]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    xmin = float(np.min(xs))
    ymin = float(np.min(ys))
    xmax = float(np.max(xs))
    ymax = float(np.max(ys))
    if xmax <= xmin or ymax <= ymin:
        return None
    # rectangle 4 points
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


def _mask_to_obb(
    mask: np.ndarray,
) -> Optional[Tuple[List[List[float]], float]]:
    """
    Build an oriented bounding box (rotation shape) that wraps the sticker mask.

    Returns:
        (points, direction) where points is [[x,y]*4] in clockwise order and
        direction is the rotation angle in radians used by shape_type='rotation'.
    """
    if mask is None or mask.size == 0:
        return None
    m = (mask > 0).astype(np.uint8) * 255
    cnts, _hier = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 3:
        return None

    rect = cv2.minAreaRect(cnt)
    corner_points = cv2.boxPoints(rect)
    # Sort clockwise around center (matches project OBB convention).
    cx, cy = float(np.mean(corner_points[:, 0])), float(np.mean(corner_points[:, 1]))
    ordered = sorted(
        corner_points,
        key=lambda p: np.arctan2(float(p[1]) - cy, float(p[0]) - cx),
    )
    pts = [[float(p[0]), float(p[1])] for p in ordered]
    if len(pts) != 4:
        return None

    try:
        from anylabeling.services.auto_labeling.utils import (
            calculate_rotation_theta,
        )

        direction = float(calculate_rotation_theta(np.asarray(pts, dtype=np.float64)))
    except Exception:
        # Fallback: angle of the first edge.
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        direction = float(np.arctan2(y2 - y1, x2 - x1))
        if direction < 0:
            direction += 2.0 * np.pi

    return pts, direction


def write_annotation_for_sticker(
    *,
    base_image_path: str,
    output_dir: Optional[str],
    label: str,
    anno_mode: str,
    mask: np.ndarray,
) -> None:
    """
    anno_mode:
      - 'hbb': create rectangle shape_type='rectangle'
      - 'obb': create rotation shape_type='rotation' (oriented box)
      - 'seg': create polygon shape_type='polygon'
    """
    if not label:
        label = "object"

    # read image size quickly
    img = cv2.imdecode(np.fromfile(base_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w = img.shape[:2]
    label_path = get_label_file_path(base_image_path, output_dir)
    obj = _load_or_init_label_json(label_path, image_path=base_image_path, w=w, h=h)
    shapes = obj.get("shapes", [])
    if not isinstance(shapes, list):
        shapes = []
        obj["shapes"] = shapes

    mode = (anno_mode or "hbb").lower().strip()
    if mode in ("seg", "polygon", "mask"):
        pts = _mask_to_polygon(mask)
        if not pts:
            return
        shapes.append(
            {
                "label": label,
                "points": pts,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        )
    elif mode in ("obb", "rotation", "rotated"):
        result = _mask_to_obb(mask)
        if not result:
            return
        pts, direction = result
        shapes.append(
            {
                "label": label,
                "points": pts,
                "group_id": None,
                "description": "",
                "shape_type": "rotation",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
                "direction": direction,
            }
        )
    else:
        pts = _mask_to_bbox(mask)
        if not pts:
            return
        shapes.append(
            {
                "label": label,
                "points": pts,
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        )

    _save_label_json(label_path, obj)

