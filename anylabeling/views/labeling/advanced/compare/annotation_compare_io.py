"""Backup and read/write for annotation compare workflow."""

from __future__ import annotations

import json
import os
import os.path as osp
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from anylabeling.config import get_work_directory
from anylabeling.views.labeling.classifier.utils import get_label_file_path
from anylabeling.views.labeling.label_converter import LabelConverter
from anylabeling.views.labeling.logger import logger

from .annotation_compare import (
    PENDING_GT_STATUSES,
    PENDING_PRED_STATUSES,
    PENDING_REVIEW_STATUSES,
    REVIEW_MODE_ANNOTATION,
    REVIEW_MODE_MODEL_EVAL,
    STATUS_SUSPECT_FN,
    STATUS_SUSPECT_FP,
    CompareSettings,
    CompareStats,
    DetValMetrics,
    clear_compare_results_shapes,
    compare_shapes,
    compute_det_val_metrics,
    get_compare_status,
    shape_has_compare_metadata,
    labels_in_shapes,
    load_classes_file,
    parse_yolo_txt,
)


@dataclass
class BatchCompareResult:
    images_processed: int = 0
    stats_total: CompareStats = field(default_factory=CompareStats)
    val_metrics: Optional[DetValMetrics] = None
    backup_dir: str = ""
    errors: List[str] = field(default_factory=list)


@dataclass
class ClearCompareResult:
    files_updated: int = 0
    fn_removed: int = 0
    fp_cleared: int = 0
    backup_dir: str = ""
    errors: List[str] = field(default_factory=list)


def backup_label_files(label_paths: List[str]) -> str:
    """Copy label JSON files into a timestamped backup folder."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    root = osp.join(get_work_directory(), "annotation_compare_backup", ts)
    os.makedirs(root, exist_ok=True)
    for path in label_paths:
        if not path or not osp.isfile(path):
            continue
        dest = osp.join(root, osp.basename(path))
        if osp.isfile(dest):
            base, ext = osp.splitext(osp.basename(path))
            dest = osp.join(root, f"{base}_{abs(hash(path)) & 0xFFFF:x}{ext}")
        try:
            shutil.copy2(path, dest)
        except OSError as e:
            logger.warning("Backup failed for %s: %s", path, e)
    return root


def read_label_json(path: str) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def write_label_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def image_size(image_path: str) -> Tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            w, h = im.size
        return int(w), int(h)
    except Exception:
        return LabelConverter().get_image_size(image_path)


def label_json_has_pending_compare(
    label_path: str,
    *,
    statuses: Optional[Tuple[str, ...]] = None,
) -> bool:
    """True if label JSON contains unreviewed shapes with given compare statuses.

    Default *statuses* is all pending annotation-compare / model-eval marks.
    """
    if not label_path or not osp.isfile(label_path):
        return False
    wanted = statuses or PENDING_REVIEW_STATUSES
    data = read_label_json(label_path)
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return False
    for shape in shapes:
        if get_compare_status(shape) in wanted:
            return True
    return False


def find_next_pending_compare_image(
    image_paths: List[str],
    *,
    current_path: Optional[str],
    output_dir: Optional[str],
    statuses: Optional[Tuple[str, ...]] = None,
) -> Optional[str]:
    """Next image after *current_path* (wrap) whose JSON still has suspect boxes.

    *statuses* filters by compare status (e.g. only ``suspect_fn`` or
    ``suspect_fp``). Default is both.
    """
    if not image_paths:
        return None
    n = len(image_paths)
    start = 0
    if current_path:
        cur = osp.normpath(str(current_path))
        for i, path in enumerate(image_paths):
            if osp.normpath(str(path)) == cur:
                start = i
                break
    for offset in range(1, n + 1):
        idx = (start + offset) % n
        label_path = get_label_file_path(image_paths[idx], output_dir)
        if label_json_has_pending_compare(label_path, statuses=statuses):
            return image_paths[idx]
    return None


def collect_gt_labels_from_images(
    image_paths: List[str],
    output_dir: Optional[str],
) -> set:
    labels: set = set()
    for image_path in image_paths:
        lp = get_label_file_path(image_path, output_dir)
        if not osp.isfile(lp):
            continue
        data = read_label_json(lp)
        shapes = data.get("shapes")
        if isinstance(shapes, list):
            labels |= labels_in_shapes(shapes)
    return labels


def process_one_image(
    *,
    image_path: str,
    label_path: str,
    pred_dir: str,
    classes: List[str],
    settings: CompareSettings,
    common_labels: Optional[set] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], CompareStats, Optional[str]]:
    stats = CompareStats()
    if not osp.isfile(label_path):
        return None, stats, f"Missing label: {label_path}"
    if not osp.isfile(image_path):
        return None, stats, f"Missing image: {image_path}"

    data = read_label_json(label_path)
    gt_shapes = data.get("shapes")
    if not isinstance(gt_shapes, list):
        gt_shapes = []

    stem = osp.splitext(osp.basename(image_path))[0]
    pred_txt = osp.join(pred_dir, stem + ".txt")
    img_w, img_h = image_size(image_path)
    pred_shapes: List[Dict[str, Any]] = []
    if osp.isfile(pred_txt):
        pred_shapes = parse_yolo_txt(pred_txt, classes, img_w, img_h)

    merged, st = compare_shapes(
        gt_shapes,
        pred_shapes,
        settings,
        common_labels=common_labels,
        classes=classes,
    )
    return merged, st, None


def write_review_label(
    label_path: str,
    merged_shapes: List[Dict[str, Any]],
    image_path: str,
) -> None:
    data = read_label_json(label_path)
    if not data:
        data = {
            "version": "0.4.0",
            "flags": {},
            "shapes": [],
            "imagePath": osp.basename(image_path),
            "imageData": None,
            "imageHeight": 0,
            "imageWidth": 0,
        }
    data["shapes"] = merged_shapes
    data["imagePath"] = osp.basename(image_path)
    try:
        img_w, img_h = image_size(image_path)
        data["imageHeight"] = img_h
        data["imageWidth"] = img_w
    except Exception:
        pass
    write_label_json(label_path, data)


def run_batch_compare(
    *,
    image_paths: List[str],
    output_dir: Optional[str],
    pred_dir: str,
    classes_path: str,
    settings: CompareSettings,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> BatchCompareResult:
    result = BatchCompareResult()
    classes = load_classes_file(classes_path)
    if not classes:
        result.errors.append("classes.txt is empty")
        return result

    pairs: List[Tuple[str, str]] = []
    label_paths: List[str] = []
    for image_path in image_paths:
        lp = get_label_file_path(image_path, output_dir)
        label_paths.append(lp)
        pairs.append((image_path, lp))

    result.backup_dir = backup_label_files(label_paths)
    gt_labels_all = collect_gt_labels_from_images(image_paths, output_dir)
    class_set = {str(c).strip() for c in classes if str(c).strip()}
    common_labels = gt_labels_all & class_set
    if gt_labels_all and class_set and not common_labels:
        result.errors.append(
            "No overlapping labels between ground-truth JSON and classes.txt; "
            "compare skipped for all shapes."
        )

    total = len(pairs)
    val_records: List[Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]] = []
    for i, (image_path, label_path) in enumerate(pairs):
        if progress_cb:
            progress_cb(i, total, osp.basename(image_path))
        merged, st, err = process_one_image(
            image_path=image_path,
            label_path=label_path,
            pred_dir=pred_dir,
            classes=classes,
            settings=settings,
            common_labels=common_labels,
        )
        if err:
            result.errors.append(f"{osp.basename(image_path)}: {err}")
            continue
        if merged is None:
            continue
        try:
            # Collect GT/pred for YOLO-val metrics (same label filter as compare).
            data = read_label_json(label_path)
            gt_shapes = data.get("shapes")
            if not isinstance(gt_shapes, list):
                gt_shapes = []
            stem = osp.splitext(osp.basename(image_path))[0]
            pred_txt = osp.join(pred_dir, stem + ".txt")
            img_w, img_h = image_size(image_path)
            pred_shapes: List[Dict[str, Any]] = []
            if osp.isfile(pred_txt):
                pred_shapes = parse_yolo_txt(pred_txt, classes, img_w, img_h)
            val_records.append((i, gt_shapes, pred_shapes))

            write_review_label(label_path, merged, image_path)
            result.images_processed += 1
            result.stats_total.matched += st.matched
            result.stats_total.suspect_fn += st.suspect_fn
            result.stats_total.suspect_fp += st.suspect_fp
        except OSError as e:
            result.errors.append(f"{osp.basename(image_path)}: {e}")

    try:
        result.val_metrics = compute_det_val_metrics(
            val_records,
            match_mode=settings.match_mode,
            common_labels=common_labels if common_labels else None,
        )
    except Exception as e:
        logger.exception("Failed to compute YOLO-val metrics")
        result.errors.append(f"YOLO-val metrics: {e}")

    return result


def _shapes_have_compare_marks(shapes: List[Dict[str, Any]]) -> bool:
    for shape in shapes:
        if get_compare_status(shape) in PENDING_REVIEW_STATUSES:
            return True
        if shape_has_compare_metadata(shape):
            return True
    return False


def label_json_has_compare_marks(label_path: str) -> bool:
    if not label_path or not osp.isfile(label_path):
        return False
    data = read_label_json(label_path)
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return False
    return _shapes_have_compare_marks(shapes)


def collect_images_with_compare_marks(
    image_paths: List[str],
    output_dir: Optional[str],
) -> List[str]:
    """Image basenames whose label JSON still has suspect_fn / suspect_fp marks."""
    marked: List[str] = []
    for image_path in image_paths:
        label_path = get_label_file_path(image_path, output_dir)
        if label_json_has_compare_marks(label_path):
            marked.append(osp.basename(image_path))
    return marked


def run_batch_clear_compare_results(
    *,
    image_paths: List[str],
    output_dir: Optional[str],
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> ClearCompareResult:
    """Clear annotation-compare and model-eval marks from JSON.

    Removes pred-side boxes (漏标/误报); keeps GT-side boxes and strips flags (误标/漏报).
    """
    result = ClearCompareResult()
    label_paths: List[str] = []
    pairs: List[Tuple[str, str]] = []
    for image_path in image_paths:
        lp = get_label_file_path(image_path, output_dir)
        label_paths.append(lp)
        pairs.append((image_path, lp))

    result.backup_dir = backup_label_files(label_paths)
    total = len(pairs)
    for i, (image_path, label_path) in enumerate(pairs):
        if progress_cb:
            progress_cb(i, total, osp.basename(image_path))
        if not osp.isfile(label_path):
            continue
        try:
            data = read_label_json(label_path)
            shapes = data.get("shapes")
            if not isinstance(shapes, list) or not _shapes_have_compare_marks(shapes):
                continue
            cleaned, fn_n, fp_n = clear_compare_results_shapes(shapes)
            data["shapes"] = cleaned
            write_label_json(label_path, data)
            result.files_updated += 1
            result.fn_removed += fn_n
            result.fp_cleared += fp_n
        except OSError as e:
            result.errors.append(f"{osp.basename(image_path)}: {e}")

    return result
