"""Dataset filter helpers: match images by labels, suspect marks, or empty."""

from __future__ import annotations

import json
import os
import os.path as osp
import shutil
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from anylabeling.views.labeling.advanced.compare.annotation_compare import (
    PENDING_REVIEW_STATUSES,
    STATUS_MODEL_FALSE,
    STATUS_MODEL_MISS,
    STATUS_SUSPECT_FN,
    STATUS_SUSPECT_FP,
    get_compare_status,
)
from anylabeling.views.labeling.classifier.utils import get_label_file_path

# UI keys for suspect-attribute group (checkbox UserRole / settings).
SUSPECT_ATTR_NORMAL = "normal"
SUSPECT_ATTR_SUSPECT_FN = "suspect_fn"
SUSPECT_ATTR_SUSPECT_FP = "suspect_fp"
SUSPECT_ATTR_MODEL_FALSE = "model_false"
SUSPECT_ATTR_MODEL_MISS = "model_miss"

SUSPECT_ATTR_ALL = (
    SUSPECT_ATTR_NORMAL,
    SUSPECT_ATTR_SUSPECT_FN,
    SUSPECT_ATTR_SUSPECT_FP,
    SUSPECT_ATTR_MODEL_FALSE,
    SUSPECT_ATTR_MODEL_MISS,
)


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


def _shape_has_geometry(shape: dict) -> bool:
    pts = shape.get("points")
    return isinstance(pts, list) and len(pts) > 0


def is_normal_annotation_shape(shape: dict) -> bool:
    """No pending suspect / model-eval review mark on the shape."""
    status = get_compare_status(shape)
    return status not in PENDING_REVIEW_STATUSES


def is_strict_empty_annotation(
    image_path: str, *, output_dir: Optional[str] = None
) -> bool:
    """
    Empty image: no normal annotation and no suspect-tagged shapes.

    A shape with label or geometry counts as normal unless it is only a
    placeholder with neither (ignored). Any pending compare status counts as
    suspect-tagged.
    """
    label_path = get_label_file_path(str(image_path), output_dir)
    data = _load_label_json(label_path)
    if not data:
        return True
    shapes = data.get("shapes") or []
    if not isinstance(shapes, list) or not shapes:
        return True
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        if get_compare_status(shape) in PENDING_REVIEW_STATUSES:
            return False
        label = str(shape.get("label") or "").strip()
        if label or _shape_has_geometry(shape):
            return False
    return True


def is_empty_annotation(
    image_path: str, *, output_dir: Optional[str] = None
) -> bool:
    """Legacy helper — alias for strict empty (dataset filter semantics)."""
    return is_strict_empty_annotation(image_path, output_dir=output_dir)


def image_has_any_suspect_mark(shapes: Sequence[dict]) -> bool:
    """True if any shape carries a pending suspect / model-eval mark."""
    return any(
        get_compare_status(shape) in PENDING_REVIEW_STATUSES for shape in shapes
    )


def shape_matches_suspect_status(
    shape: dict, selected_shape_attrs: Set[str]
) -> bool:
    """Per-shape suspect types (excludes image-level ``normal``)."""
    if not selected_shape_attrs:
        return False
    status = get_compare_status(shape)
    if (
        SUSPECT_ATTR_SUSPECT_FN in selected_shape_attrs
        and status == STATUS_SUSPECT_FN
    ):
        return True
    if (
        SUSPECT_ATTR_SUSPECT_FP in selected_shape_attrs
        and status == STATUS_SUSPECT_FP
    ):
        return True
    if (
        SUSPECT_ATTR_MODEL_FALSE in selected_shape_attrs
        and status == STATUS_MODEL_FALSE
    ):
        return True
    if (
        SUSPECT_ATTR_MODEL_MISS in selected_shape_attrs
        and status == STATUS_MODEL_MISS
    ):
        return True
    return False


def suspect_attr_group_satisfied(
    shapes: Sequence[dict], wanted_attrs: Set[str]
) -> bool:
    """
    OR within the suspect group.

    ``normal`` means the whole image has no pending suspect marks.
    Other options match if any shape has that status.
    """
    if not wanted_attrs:
        return False
    if SUSPECT_ATTR_NORMAL in wanted_attrs and not image_has_any_suspect_mark(
        shapes
    ):
        return True
    shape_attrs = wanted_attrs - {SUSPECT_ATTR_NORMAL}
    if shape_attrs and any(
        shape_matches_suspect_status(shape, shape_attrs) for shape in shapes
    ):
        return True
    return False


def _load_shapes(
    image_path: str, *, output_dir: Optional[str] = None
) -> List[dict]:
    label_path = get_label_file_path(str(image_path), output_dir)
    data = _load_label_json(label_path)
    if not data:
        return []
    shapes = data.get("shapes") or []
    if not isinstance(shapes, list):
        return []
    return [s for s in shapes if isinstance(s, dict)]


def _matches_content_groups(
    shapes: Sequence[dict],
    *,
    wanted_labels: Set[str],
    wanted_attrs: Set[str],
    shape_attrs: Set[str],
    has_normal_attr: bool,
) -> bool:
    """AND across category labels and suspect attrs (empty excluded)."""
    active_labels = bool(wanted_labels)
    active_suspect = bool(wanted_attrs)
    if not active_labels and not active_suspect:
        return False

    def label_ok(shape: dict) -> bool:
        text = str(shape.get("label") or "").strip()
        return bool(text) and text in wanted_labels

    same_shape_types_only = (
        active_labels and bool(shape_attrs) and not has_normal_attr
    )

    results: List[bool] = []
    if same_shape_types_only:
        results.append(
            any(
                label_ok(shape) and shape_matches_suspect_status(shape, shape_attrs)
                for shape in shapes
            )
        )
    elif active_labels:
        results.append(any(label_ok(shape) for shape in shapes))

    if active_suspect and not same_shape_types_only:
        results.append(suspect_attr_group_satisfied(shapes, wanted_attrs))

    return all(results)


def image_matches_filter(
    image_path: str,
    *,
    labels: Iterable[str],
    include_empty: bool,
    suspect_attrs: Optional[Iterable[str]] = None,
    output_dir: Optional[str] = None,
) -> bool:
    """
    Group logic:
    - OR inside each active group.
    - AND across category labels and suspect attrs (not empty).
    - OR between strict empty and all other active groups combined.
    - Category labels + per-shape suspect types (no ``normal``): one shape must
      satisfy both (same box).
    - ``normal`` is image-level: no shape may have a pending suspect mark.
    """
    wanted_labels = {str(x).strip() for x in labels if str(x).strip()}
    wanted_attrs = {
        str(x).strip()
        for x in (suspect_attrs or [])
        if str(x).strip() in SUSPECT_ATTR_ALL
    }
    shape_attrs = wanted_attrs - {SUSPECT_ATTR_NORMAL}
    has_normal_attr = SUSPECT_ATTR_NORMAL in wanted_attrs

    active_empty = bool(include_empty)
    active_labels = bool(wanted_labels)
    active_suspect = bool(wanted_attrs)
    if not active_empty and not active_labels and not active_suspect:
        return False

    shapes = _load_shapes(image_path, output_dir=output_dir)
    content_active = active_labels or active_suspect

    empty_ok = False
    if active_empty:
        empty_ok = is_strict_empty_annotation(image_path, output_dir=output_dir)

    if not content_active:
        return empty_ok

    content_ok = _matches_content_groups(
        shapes,
        wanted_labels=wanted_labels,
        wanted_attrs=wanted_attrs,
        shape_attrs=shape_attrs,
        has_normal_attr=has_normal_attr,
    )

    if active_empty:
        return empty_ok or content_ok
    return content_ok


def transfer_image_and_label(
    image_path: str,
    *,
    target_dir: str,
    mode: str,
    output_dir: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Copy or move image + label JSON into ``target_dir``.

    Existing same-name files in ``target_dir`` are overwritten.
    Returns (ok, message). On success message is the destination image path.
    """
    if not image_path or not osp.isfile(image_path):
        return False, "missing image"
    os.makedirs(target_dir, exist_ok=True)

    dst_img = osp.join(target_dir, osp.basename(image_path))
    label_path = get_label_file_path(image_path, output_dir)
    has_label = bool(label_path and osp.isfile(label_path))
    dst_json = osp.splitext(dst_img)[0] + ".json"

    is_move = str(mode).lower().strip() == "move"
    op = shutil.move if is_move else shutil.copy2
    try:
        if osp.exists(dst_img):
            os.remove(dst_img)
        op(image_path, dst_img)
    except Exception as e:
        return False, str(e)

    if has_label:
        try:
            if osp.exists(dst_json):
                os.remove(dst_json)
            op(label_path, dst_json)
        except Exception as e:
            try:
                if is_move and osp.isfile(dst_img):
                    shutil.move(dst_img, image_path)
                elif osp.isfile(dst_img):
                    os.remove(dst_img)
            except Exception:
                pass
            return False, str(e)

    return True, dst_img


def delete_image_and_label(
    image_path: str,
    *,
    output_dir: Optional[str] = None,
) -> Tuple[bool, str]:
    """Delete one image and its label JSON when that file exists.

    The annotation is removed first. If that fails, the image is left in place.
    """
    if not image_path or not osp.isfile(image_path):
        return False, "missing image"
    label_path = get_label_file_path(image_path, output_dir)
    has_label = bool(label_path and osp.isfile(label_path))
    if has_label:
        try:
            os.remove(label_path)
        except Exception as e:
            return False, str(e)
    try:
        os.remove(image_path)
    except Exception as e:
        return False, str(e)
    return True, image_path


def count_label_files(
    image_paths: Sequence[str],
    *,
    output_dir: Optional[str] = None,
) -> int:
    """How many of ``image_paths`` currently have a label JSON."""
    count = 0
    for path in image_paths or []:
        if not path:
            continue
        label_path = get_label_file_path(str(path), output_dir)
        if label_path and osp.isfile(label_path):
            count += 1
    return count


def filter_matching_images(
    image_list: Sequence[str],
    *,
    labels: Sequence[str],
    include_empty: bool,
    suspect_attrs: Optional[Sequence[str]] = None,
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
            suspect_attrs=suspect_attrs,
            output_dir=output_dir,
        ):
            matched.append(str(path))
    return matched