"""Crop utilities for Annotation Inspector."""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from anylabeling.views.labeling.label_converter import LabelConverter


@dataclass(frozen=True)
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def w(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def h(self) -> int:
        return max(0, self.ymax - self.ymin)


def shape_to_bbox(
    points: List[List[float]],
    img_w: int,
    img_h: int,
    *,
    padding_px: int = 4,
) -> Optional[BBox]:
    """Axis-aligned bbox for points, clipped to image, with padding."""
    if not points:
        return None
    xs, ys = [], []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
        except Exception:
            continue
    if not xs or not ys:
        return None
    xmin = int(np.floor(min(xs))) - int(padding_px)
    ymin = int(np.floor(min(ys))) - int(padding_px)
    xmax = int(np.ceil(max(xs))) + int(padding_px)
    ymax = int(np.ceil(max(ys))) + int(padding_px)
    xmin = max(0, min(xmin, img_w))
    ymin = max(0, min(ymin, img_h))
    xmax = max(0, min(xmax, img_w))
    ymax = max(0, min(ymax, img_h))
    if xmax <= xmin or ymax <= ymin:
        return None
    return BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def read_image_bgr(image_path: str) -> np.ndarray:
    """Read image supporting Unicode paths on Windows."""
    if not osp.isfile(image_path):
        raise FileNotFoundError(image_path)
    img, _size = LabelConverter.imread_unicode(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img


def crop_image_bgr(image_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
    return image_bgr[bbox.ymin : bbox.ymax, bbox.xmin : bbox.xmax].copy()


def crop_polygon_bgra(
    image_bgr: np.ndarray, *, bbox: BBox, polygon_points: List[List[float]]
) -> np.ndarray:
    """
    Crop by bbox and apply polygon alpha mask so that pixels outside polygon are transparent.
    Returns BGRA image.
    """
    crop_bgr = crop_image_bgr(image_bgr, bbox)
    h, w = crop_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((0, 0, 4), dtype=np.uint8)

    # Translate points into crop-local coordinates and clip.
    pts = []
    for p in polygon_points or []:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            x = float(p[0]) - float(bbox.xmin)
            y = float(p[1]) - float(bbox.ymin)
        except Exception:
            continue
        pts.append([x, y])
    if len(pts) < 3:
        bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
        return bgra

    poly = np.array(pts, dtype=np.int32)
    alpha = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(alpha, [poly], 255)
    bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra

