"""Remove duplicate bounding-box annotations (same label + high IoU)."""

from __future__ import annotations

import json
import os.path as osp
from typing import Any, Dict, List, Optional, Set, Tuple

from anylabeling.views.labeling.logger import logger

BBox = Tuple[float, float, float, float]


def _read_json(path: str) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def bbox_from_shape(shape: Dict[str, Any]) -> Optional[BBox]:
    pts = shape.get("points")
    if not isinstance(pts, list) or not pts:
        return None
    xs: List[float] = []
    ys: List[float] = []
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
        except (TypeError, ValueError):
            continue
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_area(box: BBox) -> float:
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w * h


def bbox_iou(a: BBox, b: BBox) -> float:
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _union_find_parent(parent: List[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union_find_union(parent: List[int], a: int, b: int) -> None:
    ra = _union_find_parent(parent, a)
    rb = _union_find_parent(parent, b)
    if ra != rb:
        parent[rb] = ra


def indices_to_remove(
    shapes: List[Dict[str, Any]],
    iou_threshold: float,
) -> Set[int]:
    """
    Among shapes with the same label and IoU > threshold, keep the largest
    area box; mark others for removal. Works for clusters of 3+ duplicates.
    """
    n = len(shapes)
    if n <= 1:
        return set()

    thr = float(iou_threshold)
    labels = [str(s.get("label") or "") for s in shapes]
    boxes = [bbox_from_shape(s) for s in shapes]
    areas = [bbox_area(b) if b else -1.0 for b in boxes]

    parent = list(range(n))
    for i in range(n):
        if boxes[i] is None:
            continue
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                continue
            if boxes[j] is None:
                continue
            if bbox_iou(boxes[i], boxes[j]) > thr:
                _union_find_union(parent, i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = _union_find_parent(parent, i)
        groups.setdefault(root, []).append(i)

    remove: Set[int] = set()
    for members in groups.values():
        if len(members) <= 1:
            continue
        valid = [i for i in members if boxes[i] is not None and areas[i] >= 0]
        if len(valid) <= 1:
            continue
        keep = max(valid, key=lambda i: areas[i])
        for i in valid:
            if i != keep:
                remove.add(i)
    return remove


def dedup_shapes(
    shapes: List[Dict[str, Any]],
    iou_threshold: float,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return deduplicated shape list and number of removed shapes."""
    if not shapes:
        return [], 0
    remove = indices_to_remove(shapes, iou_threshold)
    if not remove:
        return shapes, 0
    kept = [s for i, s in enumerate(shapes) if i not in remove]
    return kept, len(remove)


def dedup_label_file(
    label_path: str,
    iou_threshold: float,
) -> Tuple[int, int, bool]:
    """
    Deduplicate shapes in a label JSON file in place.

    Returns:
        (removed_count, remaining_count, file_was_written)
    """
    if not label_path or not osp.isfile(label_path):
        return 0, 0, False

    data = _read_json(label_path)
    shapes = data.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        return 0, 0, False

    new_shapes, removed = dedup_shapes(shapes, iou_threshold)
    if removed <= 0:
        return 0, len(shapes), False

    data["shapes"] = new_shapes
    try:
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.error("Failed to write deduped label file %s: %s", label_path, e)
        raise
    return removed, len(new_shapes), True
