"""Mosaic / pixelation for dataset desensitization."""

from __future__ import annotations

import os.path as osp
from typing import Optional, Tuple

import cv2
import numpy as np

from anylabeling.views.labeling.label_converter import LabelConverter


def apply_solid_gray_bgr(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    gray: int = 128,
) -> np.ndarray:
    """Fill ``image_bgr`` with uniform gray (0–255) where ``mask`` > 0."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    if mask is None or mask.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        return image_bgr
    sel = mask > 0
    if not np.any(sel):
        return image_bgr
    level = max(0, min(255, int(gray)))
    out = image_bgr.copy()
    out[sel] = (level, level, level)
    return out


def apply_mosaic_bgr(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    block_size: int = 12,
) -> np.ndarray:
    """Pixelate ``image_bgr`` only where ``mask`` > 0."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    if mask is None or mask.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        return image_bgr
    sel = mask > 0
    if not np.any(sel):
        return image_bgr

    bs = max(2, int(block_size))
    small_w = max(1, (w + bs - 1) // bs)
    small_h = max(1, (h + bs - 1) // bs)
    small = cv2.resize(
        image_bgr, (small_w, small_h), interpolation=cv2.INTER_LINEAR
    )
    mosaic = cv2.resize(
        small, (w, h), interpolation=cv2.INTER_NEAREST
    )
    out = image_bgr.copy()
    out[sel] = mosaic[sel]
    return out


def read_image_bgr(path: str) -> np.ndarray:
    bgr, _ = LabelConverter.imread_unicode(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return bgr


def write_image_bgr(path: str, bgr: np.ndarray) -> None:
    ext = osp.splitext(path)[1].lower() or ".jpg"
    ok, enc = cv2.imencode(ext, bgr)
    if not ok:
        ok, enc = cv2.imencode(".jpg", bgr)
    if not ok:
        raise OSError(f"Failed to encode image: {path}")
    enc.tofile(path)


def apply_mosaic_to_file(
    image_path: str,
    mask: np.ndarray,
    *,
    block_size: int = 12,
) -> Tuple[int, int]:
    """Mosaic masked pixels and overwrite ``image_path``. Returns (h, w)."""
    return apply_desensitize_to_file(
        image_path,
        mask,
        effect="mosaic",
        block_size=block_size,
    )


def apply_desensitize_to_file(
    image_path: str,
    mask: np.ndarray,
    *,
    effect: str = "mosaic",
    block_size: int = 12,
    gray_level: int = 128,
) -> Tuple[int, int]:
    """Apply mask effect and overwrite ``image_path``. Returns (h, w)."""
    bgr = read_image_bgr(image_path)
    h, w = bgr.shape[:2]
    if mask.shape[:2] != (h, w):
        raise ValueError("Mask size does not match image.")
    mode = str(effect or "mosaic").lower()
    if mode in ("solid", "solid_gray", "gray"):
        out = apply_solid_gray_bgr(bgr, mask, gray=gray_level)
    else:
        out = apply_mosaic_bgr(bgr, mask, block_size=block_size)
    write_image_bgr(image_path, out)
    return h, w