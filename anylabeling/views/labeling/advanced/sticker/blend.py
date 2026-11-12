from __future__ import annotations

import logging
import os.path as osp
from typing import Tuple

import cv2
import numpy as np

from anylabeling.views.labeling.label_converter import LabelConverter

logger = logging.getLogger(__name__)


def _read_image_bgr(path: str) -> np.ndarray:
    img, _ = LabelConverter.imread_unicode(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def _read_image_rgba(path: str) -> np.ndarray:
    if not osp.isfile(path):
        raise FileNotFoundError(path)
    # Use imdecode to support Unicode paths and keep alpha.
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        bgr = img
        a = np.full((bgr.shape[0], bgr.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([bgr, a], axis=2)
    return img


def _write_image_bgr(path: str, bgr: np.ndarray) -> None:
    ext = osp.splitext(path)[1].lower() or ".jpg"
    ok, enc = cv2.imencode(ext, bgr)
    if not ok:
        ok, enc = cv2.imencode(".jpg", bgr)
    enc.tofile(path)


def _blend_alpha(base: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)[..., None]
    out = base.astype(np.float32) * (1.0 - a) + fg.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _blend_multiply(base: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    b = base.astype(np.float32) / 255.0
    f = fg.astype(np.float32) / 255.0
    mul = (b * f) * 255.0
    return _blend_alpha(base, mul.astype(np.uint8), alpha)


def _blend_screen(base: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    b = base.astype(np.float32) / 255.0
    f = fg.astype(np.float32) / 255.0
    scr = (1.0 - (1.0 - b) * (1.0 - f)) * 255.0
    return _blend_alpha(base, scr.astype(np.uint8), alpha)


def _blend_overlay(base: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    b = base.astype(np.float32) / 255.0
    f = fg.astype(np.float32) / 255.0
    out = np.where(b <= 0.5, 2.0 * b * f, 1.0 - 2.0 * (1.0 - b) * (1.0 - f))
    out = (out * 255.0).astype(np.uint8)
    return _blend_alpha(base, out, alpha)


def apply_sticker_overwrite(
    *,
    base_image_path: str,
    material_path: str,
    quad: Tuple[object, ...],
    blend_mode: str = "alpha",
) -> np.ndarray:
    """
    Warp material (RGBA) to quad and blend into base image. Overwrite base file.

    Returns: uint8 mask (H,W) where sticker is applied (0/255).
    """
    base = _read_image_bgr(base_image_path)
    rgba = _read_image_rgba(material_path)
    h, w = base.shape[:2]

    def _alpha_compose(dst_bgra: np.ndarray, src_bgra: np.ndarray) -> np.ndarray:
        sa = (src_bgra[:, :, 3].astype(np.float32) / 255.0)[..., None]
        da = (dst_bgra[:, :, 3].astype(np.float32) / 255.0)[..., None]
        out_a = sa + da * (1.0 - sa)
        src_rgb = src_bgra[:, :, :3].astype(np.float32)
        dst_rgb = dst_bgra[:, :, :3].astype(np.float32)
        denom = np.clip(out_a, 1e-6, 1.0)
        out_rgb = (src_rgb * sa + dst_rgb * da * (1.0 - sa)) / denom
        out = dst_bgra.copy()
        out[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        out[:, :, 3] = np.clip(out_a[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
        return out

    def _warp_rgba_mesh_8pts(rgba_bgra: np.ndarray, pts8: Tuple[object, ...]) -> np.ndarray:
        # pts8 order matches Canvas: 0 tl,1 tr,2 br,3 bl,4 top,5 right,6 bottom,7 left
        p0, p1, p2, p3, mt, mr, mb, ml = pts8
        cx = (mt.x() + mr.x() + mb.x() + ml.x()) / 4.0
        cy = (mt.y() + mr.y() + mb.y() + ml.y()) / 4.0
        c = type(mt)(cx, cy)
        grid = (p0, mt, p1, ml, c, mr, p3, mb, p2)  # 3x3 row-major

        mh, mw = rgba_bgra.shape[:2]
        x0, x1, x2 = 0.0, float(mw - 1) / 2.0, float(mw - 1)
        y0, y1, y2 = 0.0, float(mh - 1) / 2.0, float(mh - 1)
        xs = [x0, x1, x2]
        ys = [y0, y1, y2]
        out = np.zeros((h, w, 4), dtype=np.uint8)

        for gy in range(2):
            for gx in range(2):
                sx0, sx1 = xs[gx], xs[gx + 1]
                sy0, sy1 = ys[gy], ys[gy + 1]
                ix0, ix1 = int(round(sx0)), int(round(sx1))
                iy0, iy1 = int(round(sy0)), int(round(sy1))
                ix1 = max(ix1, ix0 + 1)
                iy1 = max(iy1, iy0 + 1)
                cell = rgba_bgra[iy0:iy1, ix0:ix1]
                ch, cw = cell.shape[:2]
                if ch <= 1 or cw <= 1:
                    continue

                src = np.array(
                    [
                        [0.0, 0.0],
                        [float(cw - 1), 0.0],
                        [float(cw - 1), float(ch - 1)],
                        [0.0, float(ch - 1)],
                    ],
                    dtype=np.float32,
                )
                p00 = grid[gy * 3 + gx]
                p10 = grid[gy * 3 + (gx + 1)]
                p11 = grid[(gy + 1) * 3 + (gx + 1)]
                p01 = grid[(gy + 1) * 3 + gx]
                dst = np.array(
                    [
                        [float(p00.x()), float(p00.y())],
                        [float(p10.x()), float(p10.y())],
                        [float(p11.x()), float(p11.y())],
                        [float(p01.x()), float(p01.y())],
                    ],
                    dtype=np.float32,
                )
                M = cv2.getPerspectiveTransform(src, dst)
                warped_cell = cv2.warpPerspective(
                    cell,
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0),
                )
                out = _alpha_compose(out, warped_cell)
        return out

    if isinstance(quad, tuple) and len(quad) == 8:
        warped = _warp_rgba_mesh_8pts(rgba, quad)
    else:
        src = np.array(
            [
                [0.0, 0.0],
                [float(rgba.shape[1] - 1), 0.0],
                [float(rgba.shape[1] - 1), float(rgba.shape[0] - 1)],
                [0.0, float(rgba.shape[0] - 1)],
            ],
            dtype=np.float32,
        )
        dst = np.array([[float(p.x()), float(p.y())] for p in quad], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            rgba,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
    fg = warped[:, :, :3]
    a = warped[:, :, 3].astype(np.float32) / 255.0
    mask = (a > 0.01).astype(np.uint8) * 255

    mode = (blend_mode or "alpha").lower().strip()
    if mode == "multiply":
        out = _blend_multiply(base, fg, a)
    elif mode == "screen":
        out = _blend_screen(base, fg, a)
    elif mode == "overlay":
        out = _blend_overlay(base, fg, a)
    elif mode in ("poisson", "seamless", "seamlessclone"):
        # Poisson blending ignores alpha; use mask as region
        try:
            # center must be computed regardless of 4-pt vs 8-pt warp.
            if isinstance(quad, tuple) and len(quad) == 8:
                # Use the 4 corners for a stable center.
                cx = float(sum(p.x() for p in quad[0:4])) / 4.0
                cy = float(sum(p.y() for p in quad[0:4])) / 4.0
            else:
                cx = float(np.mean(dst[:, 0]))
                cy = float(np.mean(dst[:, 1]))
            center = (int(round(cx)), int(round(cy)))
            out = cv2.seamlessClone(
                fg, base, mask, center, cv2.NORMAL_CLONE
            )
        except Exception:
            logger.exception("[Sticker] poisson blend failed; falling back to alpha")
            out = _blend_alpha(base, fg, a)
    else:
        out = _blend_alpha(base, fg, a)

    _write_image_bgr(base_image_path, out)
    return mask

