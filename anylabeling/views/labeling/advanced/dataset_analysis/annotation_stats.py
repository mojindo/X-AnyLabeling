"""Annotation statistics for dataset analysis.

This module reads existing XLABEL JSON files and aggregates box-like shapes
into dataset-level statistics and chart-ready arrays.
"""

from __future__ import annotations

import json
import os.path as osp
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import cv2
import numpy as np

from anylabeling.views.labeling.classifier.utils import get_label_file_path


BOX_SHAPE_TYPES = ("rectangle", "rotation", "polygon")


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _points_to_bbox(points: List[List[float]]) -> Optional[Tuple[float, float, float, float]]:
    if not points:
        return None
    xs = [float(p[0]) for p in points if len(p) >= 2]
    ys = [float(p[1]) for p in points if len(p) >= 2]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _polygon_area(points: List[List[float]]) -> Optional[float]:
    if not points or len(points) < 3:
        return None
    arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    area = float(cv2.contourArea(arr))
    return abs(area)


@dataclass(frozen=True)
class AnnotationStats:
    image_total: int
    images_with_any_box: int
    images_with_zero_box: int
    class_names: List[str]
    per_class_box_count: Dict[str, int]
    per_image_class_counts: List[Dict[str, int]]
    per_image_names: List[str]
    per_box_area_ratio: Dict[str, List[float]]
    per_class_centers: Dict[str, List[Tuple[float, float]]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_total": self.image_total,
            "images_with_any_box": self.images_with_any_box,
            "images_with_zero_box": self.images_with_zero_box,
            "class_names": self.class_names,
            "per_class_box_count": dict(self.per_class_box_count),
            "per_image_class_counts": self.per_image_class_counts,
            "per_image_names": self.per_image_names,
            "per_box_area_ratio": {k: v for k, v in self.per_box_area_ratio.items()},
            "per_class_centers": {
                k: [{"x": float(x), "y": float(y)} for x, y in pts]
                for k, pts in self.per_class_centers.items()
            },
        }


def collect_annotation_stats(
    image_paths: List[str],
    *,
    output_dir: Optional[str] = None,
    include_shape_types: Tuple[str, ...] = BOX_SHAPE_TYPES,
) -> AnnotationStats:
    """Aggregate annotation stats from existing label JSON files."""
    per_class_box: Counter[str] = Counter()
    per_image_counts: List[Dict[str, int]] = []
    per_image_names: List[str] = []
    area_ratios: DefaultDict[str, List[float]] = defaultdict(list)
    centers: DefaultDict[str, List[Tuple[float, float]]] = defaultdict(list)

    images_with_any = 0
    images_with_zero = 0

    class_set = set()

    for image_path in image_paths:
        label_path = get_label_file_path(image_path, output_dir)
        data = _load_json(label_path)
        per_image_names.append(osp.basename(image_path))
        if not data:
            per_image_counts.append({})
            images_with_zero += 1
            continue

        shapes = data.get("shapes") or []
        iw = data.get("imageWidth")
        ih = data.get("imageHeight")
        try:
            iw = int(iw) if iw is not None else None
            ih = int(ih) if ih is not None else None
        except Exception:
            iw = ih = None
        image_area = float(iw * ih) if iw and ih and iw > 0 and ih > 0 else None

        img_counter: Counter[str] = Counter()

        for s in shapes:
            try:
                label = s.get("label")
                st = s.get("shape_type")
                points = s.get("points") or []
                if not isinstance(label, str) or not label:
                    continue
                if st not in include_shape_types:
                    continue
                if not isinstance(points, list):
                    continue
            except Exception:
                continue

            bbox = _points_to_bbox(points)
            area = _polygon_area(points)
            if bbox is None:
                continue
            xmin, ymin, xmax, ymax = bbox
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0

            img_counter[label] += 1
            per_class_box[label] += 1
            class_set.add(label)

            if image_area and area is not None:
                area_ratios[label].append(float(area / image_area))

            if iw and ih and iw > 0 and ih > 0:
                centers[label].append((float(cx / iw), float(cy / ih)))

        per_image_counts.append(dict(img_counter))
        if sum(img_counter.values()) > 0:
            images_with_any += 1
        else:
            images_with_zero += 1

    class_names = sorted(class_set)
    return AnnotationStats(
        image_total=len(image_paths),
        images_with_any_box=images_with_any,
        images_with_zero_box=images_with_zero,
        class_names=class_names,
        per_class_box_count=dict(per_class_box),
        per_image_class_counts=per_image_counts,
        per_image_names=per_image_names,
        per_box_area_ratio=dict(area_ratios),
        per_class_centers=dict(centers),
    )

