"""Dataset filter helpers: match images by labels or empty annotations."""

from __future__ import annotations

import json
import os
import os.path as osp
import shutil
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from anylabeling.views.labeling.classifier.utils import get_label_file_path


def _load_label_json(label_path: str) -> Optional[dict]:
    if not label_path or not os.path.isfile(label_path):
        return None
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def collect_shape_labels(label_path: str) -> Set[str]:
    data = _load_label_json(label_path)
    if not data:
        return set()
    found: Set[str] = set()
    for shape in data.get("shapes") or []:
        if not isinstance(shape, dict):
            continue
        label = shape.get("label")
        if label is None:
            continue
        text = str(label).strip()
        if text:
            found.add(text)
    return found


def is_empty_annotation(
    image_path: str, *, output_dir: Optional[str] = None
) -> bool:
    """True when there is no JSON or the JSON has no shape annotations."""
    label_path = get_label_file_path(str(image_path), output_dir)
    data = _load_label_json(label_path)
    if not data:
        return True
    shapes = data.get("shapes") or []
    if not isinstance(shapes, list):
        return True
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label") or "").strip()
        pts = shape.get("points")
        if label or (isinstance(pts, list) and pts):
            return False
    return True


def image_matches_filter(
    image_path: str,
    *,
    labels: Iterable[str],
    include_empty: bool,
    output_dir: Optional[str] = None,
) -> bool:
    wanted = {str(x).strip() for x in labels if str(x).strip()}
    if include_empty and is_empty_annotation(image_path, output_dir=output_dir):
        return True
    if not wanted:
        return False
    label_path = get_label_file_path(str(image_path), output_dir)
    present = collect_shape_labels(label_path)
    return bool(present & wanted)


def _dedupe_path(path: str) -> str:
    if not osp.exists(path):
        return path
    base, ext = osp.splitext(path)
    k = 1
    while True:
        cand = f"{base}_{k}{ext}"
        if not osp.exists(cand):
            return cand
        k += 1


def transfer_image_and_label(
    image_path: str,
    *,
    target_dir: str,
    mode: str,
    output_dir: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Copy or move image + label JSON into ``target_dir``.

    Returns (ok, message). On success message is the destination image path.
    """
    if not image_path or not osp.isfile(image_path):
        return False, "missing image"
    os.makedirs(target_dir, exist_ok=True)

    dst_img = _dedupe_path(osp.join(target_dir, osp.basename(image_path)))
    label_path = get_label_file_path(image_path, output_dir)
    has_label = bool(label_path and osp.isfile(label_path))
    dst_json = osp.splitext(dst_img)[0] + ".json"

    op = shutil.move if str(mode).lower().strip() == "move" else shutil.copy2
    try:
        op(image_path, dst_img)
    except Exception as e:
        return False, str(e)

    if has_label:
        try:
            op(label_path, dst_json)
        except Exception as e:
            # Best-effort rollback for the image on failure.
            try:
                if str(mode).lower().strip() == "move" and osp.isfile(dst_img):
                    shutil.move(dst_img, image_path)
                elif osp.isfile(dst_img):
                    os.remove(dst_img)
            except Exception:
                pass
            return False, str(e)

    return True, dst_img


def filter_matching_images(
    image_list: Sequence[str],
    *,
    labels: Sequence[str],
    include_empty: bool,
    output_dir: Optional[str] = None,
) -> List[str]:
    matched: List[str] = []
    for path in image_list or []:
        if not path:
            continue
        if image_matches_filter(
            str(path),
            labels=labels,
            include_empty=include_empty,
            output_dir=output_dir,
        ):
            matched.append(str(path))
    return matched
