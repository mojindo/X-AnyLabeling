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


def make_shape_key(
    *,
    label: object,
    shape_type: object,
    points: Optional[List] = None,
    bbox: Optional[dict] = None,
) -> str:
    """
    Stable identity for a shape ↔ crop binding.

    Prefer rounded vertex coordinates so the key survives list-index shifts
    after delete/reorder. Fall back to axis-aligned bbox when points are absent
    (legacy cache rows).
    """
    lab = str(label or "").strip()
    st = str(shape_type or "").strip()
    coords: List[str] = []
    for p in points or []:
        try:
            if hasattr(p, "x") and hasattr(p, "y"):
                x, y = float(p.x()), float(p.y())
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                x, y = float(p[0]), float(p[1])
            else:
                continue
            coords.append(f"{int(round(x))},{int(round(y))}")
        except Exception:
            continue
    if coords:
        return f"{lab}\x1f{st}\x1fp:{'|'.join(coords)}"
    if isinstance(bbox, dict) and bbox:
        try:
            return (
                f"{lab}\x1f{st}\x1fb:"
                f"{int(bbox.get('xmin', 0))},"
                f"{int(bbox.get('ymin', 0))},"
                f"{int(bbox.get('xmax', 0))},"
                f"{int(bbox.get('ymax', 0))}"
            )
        except Exception:
            pass
    return f"{lab}\x1f{st}\x1f"


def shape_key_from_shape(shape) -> str:
    """Build shape_key from a live canvas Shape (vertex-based)."""
    pts = []
    try:
        pts = list(getattr(shape, "points", None) or [])
    except Exception:
        pts = []
    return make_shape_key(
        label=getattr(shape, "label", ""),
        shape_type=getattr(shape, "shape_type", ""),
        points=pts,
    )


def shape_bbox_key_from_shape(
    shape,
    image_w: int,
    image_h: int,
    *,
    padding_px: int = 4,
) -> str:
    """Legacy/compat key from padded axis-aligned bbox (matches old cache rows)."""
    pts = []
    try:
        for p in list(getattr(shape, "points", None) or []):
            pts.append([float(p.x()), float(p.y())])
    except Exception:
        pts = []
    bbox = shape_to_bbox(pts, int(image_w), int(image_h), padding_px=padding_px)
    if bbox is None:
        return ""
    return make_shape_key(
        label=getattr(shape, "label", ""),
        shape_type=getattr(shape, "shape_type", ""),
        bbox={
            "xmin": bbox.xmin,
            "ymin": bbox.ymin,
            "xmax": bbox.xmax,
            "ymax": bbox.ymax,
        },
    )


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


CROP_MODE_HBB = "hbb"
CROP_MODE_OBB = "obb"
CROP_MODE_SEG = "seg"


def shape_eligible_for_crop_mode(shape_type: str, crop_mode: str) -> bool:
    st = str(shape_type or "").strip().lower()
    mode = str(crop_mode or CROP_MODE_HBB).strip().lower()
    if mode == CROP_MODE_HBB:
        return st in ("rectangle", "rotation", "quadrilateral")
    if mode == CROP_MODE_OBB:
        return st == "rotation"
    if mode == CROP_MODE_SEG:
        return st == "polygon"
    return False


def _quad_points_as_float(points: List[List[float]]) -> Optional[np.ndarray]:
    pts: List[List[float]] = []
    for p in points or []:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            pts.append([float(p[0]), float(p[1])])
        except Exception:
            continue
    if len(pts) != 4:
        return None
    return np.asarray(pts, dtype=np.float32)


def crop_rotation_masked_bgra(
    image_bgr: np.ndarray,
    points: List[List[float]],
    *,
    img_w: int,
    img_h: int,
    padding_px: int = 4,
) -> Optional[np.ndarray]:
    """Oriented box: axis-aligned crop window + quad alpha mask (keeps rotation)."""
    bbox = shape_to_bbox(points, img_w, img_h, padding_px=padding_px)
    if bbox is None:
        return None
    out = crop_polygon_bgra(image_bgr, bbox=bbox, polygon_points=points)
    return out if out.size else None


def crop_rotation_rectified_bgra(
    image_bgr: np.ndarray,
    points: List[List[float]],
) -> Optional[np.ndarray]:
    """Perspective-rectify OBB using annotation point order for width/height."""
    quad = _quad_points_as_float(points)
    if quad is None:
        return None

    edge01 = float(np.linalg.norm(quad[1] - quad[0]))
    edge12 = float(np.linalg.norm(quad[2] - quad[1]))
    if edge01 < 1.0 or edge12 < 1.0:
        return None
    width = max(1, int(round(edge01)))
    height = max(1, int(round(edge12)))

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(
        image_bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if warped.size == 0:
        return None
    bgra = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = 255
    return bgra


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


def crop_shape_to_array(
    image_bgr: np.ndarray,
    *,
    shape_type: str,
    points: List[List[float]],
    img_w: int,
    img_h: int,
    crop_mode: str = CROP_MODE_HBB,
    obb_rectify: bool = False,
    padding_px: int = 4,
) -> Optional[np.ndarray]:
    """Return crop (BGR or BGRA) for one shape under the selected crop mode."""
    mode = str(crop_mode or CROP_MODE_HBB).lower()
    st = str(shape_type or "")
    if not shape_eligible_for_crop_mode(st, mode):
        return None
    if mode == CROP_MODE_SEG:
        bbox = shape_to_bbox(points, img_w, img_h, padding_px=padding_px)
        if bbox is None:
            return None
        out = crop_polygon_bgra(image_bgr, bbox=bbox, polygon_points=points)
        return out if out.size else None
    if mode == CROP_MODE_OBB:
        if obb_rectify:
            return crop_rotation_rectified_bgra(image_bgr, points)
        return crop_rotation_masked_bgra(
            image_bgr, points, img_w=img_w, img_h=img_h, padding_px=padding_px
        )
    bbox = shape_to_bbox(points, img_w, img_h, padding_px=padding_px)
    if bbox is None:
        return None
    out = crop_image_bgr(image_bgr, bbox)
    return out if out.size else None


def crop_output_extension(crop_mode: str, shape_type: str = "") -> str:
    mode = str(crop_mode or CROP_MODE_HBB).lower()
    if mode in (CROP_MODE_OBB, CROP_MODE_SEG):
        return ".png"
    return ".jpg"


def write_crop_image(
    path: str,
    image: np.ndarray,
    *,
    jpeg_quality: int = 85,
) -> None:
    """Write a crop file; ``imencode`` + ``tofile`` supports Unicode paths on Windows."""
    ext = osp.splitext(path)[1].lower()
    if ext == ".png":
        ok, buf = cv2.imencode(".png", image)
    else:
        ok, buf = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
    if not ok:
        raise OSError(f"Failed to encode crop image: {path}")
    buf.tofile(path)

