"""Match sticker RGB to surrounding background before warp/blend."""
from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

MATCH_STRENGTH = 0.65
GAIN_RANGE = (0.75, 1.30)
BIAS_LIMIT = 25.0
MIN_STD = 5.0
SAMPLE_RADIUS_RATIO = 0.06
MIN_TARGET_SAMPLES = 30
MIN_SOURCE_SAMPLES = 30
VISIBLE_ALPHA_THRESHOLD = 128


def _quad_corners_xy(quad: Sequence[object]) -> np.ndarray:
    if not quad or len(quad) < 4:
        raise ValueError("quad needs at least 4 points")
    pts = []
    for p in quad[0:4]:
        pts.append([float(p.x()), float(p.y())])
    return np.array(pts, dtype=np.float32)


def adapt_sticker_rgba_to_background(
    rgba_bgra: np.ndarray,
    base_bgr: np.ndarray,
    quad: Sequence[object],
    *,
    enabled: bool = True,
    strength: float = MATCH_STRENGTH,
) -> np.ndarray:
    """
    Adjust sticker BGR channels from local background ring statistics (image coords).
    Alpha is preserved. Returns a copy when disabled or sampling is insufficient.
    """
    if not enabled or rgba_bgra is None or rgba_bgra.size == 0:
        return rgba_bgra
    if base_bgr is None or base_bgr.size == 0:
        return rgba_bgra

    result = rgba_bgra.copy()
    h, w = base_bgr.shape[:2]
    q = _quad_corners_xy(quad)
    q_draw = np.round(q).astype(np.int32).reshape((-1, 1, 2))

    sticker_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(sticker_mask, q_draw, 255)

    xs = q[:, 0]
    ys = q[:, 1]
    bw = float(np.max(xs) - np.min(xs))
    bh = float(np.max(ys) - np.min(ys))
    short_side = max(1.0, min(bw, bh))
    radius = max(3, int(short_side * SAMPLE_RADIUS_RATIO))
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded = cv2.dilate(sticker_mask, kernel)
    sample_mask = cv2.subtract(expanded, sticker_mask)

    target_pixels = base_bgr[sample_mask > 0]
    if target_pixels.shape[0] < MIN_TARGET_SAMPLES:
        return result

    target_pixels = target_pixels.astype(np.float32)
    target_mean = np.mean(target_pixels, axis=0)
    target_std = np.maximum(np.std(target_pixels, axis=0), MIN_STD)

    src_bgr = rgba_bgra[:, :, :3].astype(np.float32)
    alpha = rgba_bgra[:, :, 3]
    visible = alpha > VISIBLE_ALPHA_THRESHOLD
    source_pixels = src_bgr[visible]
    if source_pixels.shape[0] < MIN_SOURCE_SAMPLES:
        return result

    source_mean = np.mean(source_pixels, axis=0)
    source_std = np.maximum(np.std(source_pixels, axis=0), MIN_STD)

    gain = np.clip(target_std / source_std, GAIN_RANGE[0], GAIN_RANGE[1])
    bias = np.clip(target_mean - source_mean * gain, -BIAS_LIMIT, BIAS_LIMIT)

    s = float(np.clip(strength, 0.0, 1.0))
    adjusted = (src_bgr - source_mean) * (1.0 + (gain - 1.0) * s) + source_mean
    adjusted += bias * s
    result[:, :, :3] = np.clip(adjusted, 0, 255).astype(np.uint8)
    result[:, :, 3] = rgba_bgra[:, :, 3]
    return result