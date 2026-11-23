"""Crop matte RGBA by HBB / OBB / SEG annotations."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from anylabeling.views.labeling.advanced.batch_annotation_check.cropper import (
    CROP_MODE_HBB,
    CROP_MODE_OBB,
    CROP_MODE_SEG,
    BBox,
    crop_image_bgr,
    crop_polygon_bgra,
    crop_rotation_masked_bgra,
    crop_rotation_rectified_bgra,
    crop_shape_to_array,
    shape_to_bbox,
)


def crop_mode_for_shape_type(shape_type: str) -> str:
    st = str(shape_type or "").strip().lower()
    if st == "polygon":
        return CROP_MODE_SEG
    if st == "rotation":
        return CROP_MODE_OBB
    return CROP_MODE_HBB


def _alpha_masked_patch(
    alpha: np.ndarray,
    *,
    shape_type: str,
    points: List[List[float]],
    img_w: int,
    img_h: int,
    padding_px: int,
) -> Optional[np.ndarray]:
    a3 = cv2.merge([alpha, alpha, alpha])
    patch = crop_rotation_masked_bgra(
        a3, points, img_w=img_w, img_h=img_h, padding_px=padding_px
    )
    if patch is None or patch.size == 0:
        return None
    return patch[:, :, 3]


def crop_shape_from_matte(
    full_bgra: np.ndarray,
    *,
    shape_type: str,
    points: List[List[float]],
    img_w: int,
    img_h: int,
    padding_px: int = 4,
    obb_rectify: bool = False,
    mask_dilate_px: int = 0,
) -> Optional[np.ndarray]:
    """Crop a shape region from a full-image matte (BGRA)."""
    if full_bgra is None or full_bgra.size == 0:
        return None
    bgr = full_bgra[:, :, :3]
    alpha = full_bgra[:, :, 3]
    mode = crop_mode_for_shape_type(shape_type)

    if mode == CROP_MODE_HBB:
        bbox = shape_to_bbox(points, img_w, img_h, padding_px=padding_px)
        if bbox is None or bbox.w <= 0 or bbox.h <= 0:
            return None
        patch_bgr = crop_image_bgr(bgr, bbox)
        patch_a = alpha[bbox.ymin : bbox.ymax, bbox.xmin : bbox.xmax].copy()
    elif mode == CROP_MODE_SEG:
        bbox = shape_to_bbox(points, img_w, img_h, padding_px=padding_px)
        if bbox is None or bbox.w <= 0 or bbox.h <= 0:
            return None
        local_pts = [
            [float(p[0]) - bbox.xmin, float(p[1]) - bbox.ymin] for p in points
        ]
        patch_bgr = bgr[bbox.ymin : bbox.ymax, bbox.xmin : bbox.xmax].copy()
        poly = crop_polygon_bgra(
            patch_bgr,
            bbox=BBox(0, 0, patch_bgr.shape[1], patch_bgr.shape[0]),
            polygon_points=local_pts,
            mask_dilate_px=int(mask_dilate_px),
        )
        if poly is None or poly.size == 0:
            return None
        patch_a = poly[:, :, 3]
        patch_a = (
            patch_a.astype(np.float32)
            * alpha[bbox.ymin : bbox.ymax, bbox.xmin : bbox.xmax].astype(np.float32)
            / 255.0
        ).astype(np.uint8)
    elif mode == CROP_MODE_OBB:
        if obb_rectify:
            patch_bgr = crop_rotation_rectified_bgra(bgr, points)
            if patch_bgr is None or patch_bgr.size == 0:
                return None
            patch = crop_rotation_rectified_bgra(
                cv2.merge([alpha, alpha, alpha]), points
            )
            patch_a = patch[:, :, 3] if patch is not None else np.full(patch_bgr.shape[:2], 255, np.uint8)
        else:
            patch_bgr = crop_rotation_masked_bgra(
                bgr, points, img_w=img_w, img_h=img_h, padding_px=padding_px
            )
            if patch_bgr is None or patch_bgr.size == 0:
                return None
            patch_bgr = patch_bgr[:, :, :3]
            patch_a = _alpha_masked_patch(
                alpha,
                shape_type=shape_type,
                points=points,
                img_w=img_w,
                img_h=img_h,
                padding_px=padding_px,
            )
            if patch_a is None:
                patch_a = patch_bgr[:, :, 3] if patch_bgr.shape[2] == 4 else np.full(patch_bgr.shape[:2], 255, np.uint8)
    else:
        return None

    out = cv2.cvtColor(patch_bgr[:, :, :3], cv2.COLOR_BGR2BGRA)
    out[:, :, 3] = patch_a
    return out


def crop_shape_precut_bgra(
    image_bgr: np.ndarray,
    *,
    shape_type: str,
    points: List[List[float]],
    img_w: int,
    img_h: int,
    padding_px: int = 4,
    obb_rectify: bool = False,
    mask_dilate_px: int = 0,
) -> Optional[np.ndarray]:
    """First-stage crop along annotation (fine mode, before RMBG)."""
    mode = crop_mode_for_shape_type(shape_type)
    arr = crop_shape_to_array(
        image_bgr,
        shape_type=shape_type,
        points=points,
        img_w=img_w,
        img_h=img_h,
        crop_mode=mode,
        obb_rectify=obb_rectify,
        padding_px=padding_px,
        mask_dilate_px=mask_dilate_px,
    )
    if arr is None or arr.size == 0:
        return None
    if arr.shape[2] == 3:
        bgra = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255
        return bgra
    return arr