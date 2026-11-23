"""Fine / fast matting pipelines."""
from __future__ import annotations

import os
import os.path as osp
import re
from typing import Callable, List, Optional, Set, Tuple

import cv2
import numpy as np

from anylabeling.views.labeling.advanced.batch_annotation_check.cropper import (
    read_image_bgr,
    write_crop_image,
)
from anylabeling.views.labeling.advanced.matting.annotations import (
    image_hw_from_label,
    iter_matting_shapes,
)
from anylabeling.views.labeling.advanced.matting.backends.base import (
    MattingBackend,
    apply_mask_to_bgra,
)
from anylabeling.views.labeling.advanced.matting.crops import (
    crop_shape_from_matte,
    crop_shape_precut_bgra,
)


MODE_FINE = "fine"
MODE_FAST = "fast"


def _safe_name(text: str) -> str:
    t = re.sub(r'[<>:"/\\|?*]+', "_", str(text or "").strip())
    return t or "object"


def matte_full_bgra(
    image_bgr: np.ndarray, backend: MattingBackend
) -> np.ndarray:
    mask = backend.predict_mask(image_bgr)
    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    return bgra


def refine_crop_bgra(
    crop_bgra: np.ndarray, backend: MattingBackend
) -> np.ndarray:
    mask = backend.predict_mask(crop_bgra[:, :, :3])
    return apply_mask_to_bgra(crop_bgra, mask)


def export_bgra(output_dir: str, label: str, stem: str, shape_index: int, bgra: np.ndarray) -> str:
    folder = osp.join(output_dir, _safe_name(label))
    os.makedirs(folder, exist_ok=True)
    name = f"{stem}_{shape_index:04d}.png"
    path = osp.join(folder, name)
    write_crop_image(path, bgra)
    return path


def process_one_image(
    image_path: str,
    backend: MattingBackend,
    *,
    mode: str,
    output_dir: str,
    output_label_dir: Optional[str],
    allowed_labels: Optional[Set[str]],
    padding_px: int,
    obb_rectify: bool,
    mask_dilate_px: int = 0,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Tuple[int, int]:
    shapes = iter_matting_shapes(
        image_path, output_dir=output_label_dir, allowed_labels=allowed_labels
    )
    if not shapes:
        return 0, 0

    image_bgr = read_image_bgr(image_path)
    img_w, img_h = image_hw_from_label(
        image_path, output_dir=output_label_dir, image_bgr=image_bgr
    )
    stem = osp.splitext(osp.basename(image_path))[0]
    written = 0

    if mode == MODE_FAST:
        if should_cancel and should_cancel():
            return len(shapes), written
        full_bgra = matte_full_bgra(image_bgr, backend)
        for sh in shapes:
            if should_cancel and should_cancel():
                break
            patch = crop_shape_from_matte(
                full_bgra,
                shape_type=sh.shape_type,
                points=sh.points,
                img_w=img_w,
                img_h=img_h,
                padding_px=padding_px,
                obb_rectify=obb_rectify,
                mask_dilate_px=mask_dilate_px,
            )
            if patch is None or patch.size == 0:
                continue
            export_bgra(output_dir, sh.label, stem, sh.index, patch)
            written += 1
        return len(shapes), written

    for sh in shapes:
        if should_cancel and should_cancel():
            break
        crop = crop_shape_precut_bgra(
            image_bgr,
            shape_type=sh.shape_type,
            points=sh.points,
            img_w=img_w,
            img_h=img_h,
            padding_px=padding_px,
            obb_rectify=obb_rectify,
            mask_dilate_px=mask_dilate_px,
        )
        if crop is None or crop.size == 0:
            continue
        refined = refine_crop_bgra(crop, backend)
        export_bgra(output_dir, sh.label, stem, sh.index, refined)
        written += 1
    return len(shapes), written


def run_matting_dataset(
    image_paths: List[str],
    backend: MattingBackend,
    *,
    mode: str,
    output_dir: str,
    output_label_dir: Optional[str],
    allowed_labels: Optional[Set[str]],
    padding_px: int,
    obb_rectify: bool,
    mask_dilate_px: int = 0,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Tuple[int, int, int, bool]:
    total_shapes = 0
    total_written = 0
    n = len(image_paths)
    for i, path in enumerate(image_paths):
        if should_cancel and should_cancel():
            break
        if on_progress:
            on_progress(i + 1, n, osp.basename(path))
        try:
            sc, wr = process_one_image(
                path,
                backend,
                mode=mode,
                output_dir=output_dir,
                output_label_dir=output_label_dir,
                allowed_labels=allowed_labels,
                padding_px=padding_px,
                obb_rectify=obb_rectify,
                mask_dilate_px=mask_dilate_px,
                should_cancel=should_cancel,
            )
            total_shapes += sc
            total_written += wr
        except Exception:
            continue
    cancelled = bool(should_cancel and should_cancel())
    return n, total_shapes, total_written, cancelled