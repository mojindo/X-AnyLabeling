"""Compare GT (LabelMe JSON) vs prediction (YOLO txt) for annotation review."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.advanced.ops.label_dedup import bbox_area, bbox_from_shape

COMPARE_STATUS = "_compare_status"
STATUS_MATCHED = "matched"
STATUS_SUSPECT_FN = "suspect_fn"  # pred only — 疑是漏标 (annotation compare)
STATUS_SUSPECT_FP = "suspect_fp"  # gt only — 疑是误标 (annotation compare)
# Model evaluation: same geometry, different semantics
STATUS_MODEL_FALSE = "model_false"  # pred only — 误报
STATUS_MODEL_MISS = "model_miss"  # gt only — 漏报

REVIEW_MODE_ANNOTATION = "annotation"
REVIEW_MODE_MODEL_EVAL = "model_eval"

# Review highlight colors (RGB): pred-side=yellow, gt-side=red
COLOR_SUSPECT_FN = (255, 220, 0)
COLOR_SUSPECT_FP = (255, 51, 51)
COLOR_MODEL_FALSE = COLOR_SUSPECT_FN
COLOR_MODEL_MISS = COLOR_SUSPECT_FP

# Pred-side pending marks (added from predictions; cleared by deleting the box)
PENDING_PRED_STATUSES = (STATUS_SUSPECT_FN, STATUS_MODEL_FALSE)
# GT-side pending marks (flag on existing GT; cleared by stripping flags)
PENDING_GT_STATUSES = (STATUS_SUSPECT_FP, STATUS_MODEL_MISS)
PENDING_REVIEW_STATUSES = PENDING_PRED_STATUSES + PENDING_GT_STATUSES

FN_ACCEPT_LABEL = "fn_accept_label"
FN_ACCEPT_NEW_LABEL = "fn_accept_new_label"
FN_REJECT = "fn_reject"

FP_CONFIRM_DELETE = "fp_confirm_delete"
FP_CONFIRM_RELABEL = "fp_confirm_relabel"
FP_KEEP = "fp_keep"

_PRIORITY = {
    STATUS_MATCHED: 3,
    STATUS_SUSPECT_FN: 2,
    STATUS_MODEL_FALSE: 2,
    STATUS_SUSPECT_FP: 1,
    STATUS_MODEL_MISS: 1,
}

BBox = Tuple[float, float, float, float]


@dataclass
class CompareSettings:
    match_mode: str = "ios"  # ios | iou
    match_thresh: float = 0.3
    dedup_ios_thresh: float = 0.85
    same_class_only: bool = True
    # When False, that kind of suspect mark is not written to JSON.
    check_fp: bool = True  # unmatched GT → 误标 / 漏报
    check_fn: bool = True  # unmatched pred → 漏标 / 误报
    review_mode: str = REVIEW_MODE_ANNOTATION


@dataclass
class CompareStats:
    matched: int = 0
    suspect_fn: int = 0
    suspect_fp: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class DetValMetrics:
    """YOLO-val style detection metrics (overlap mode = ios or iou)."""

    match_mode: str = "iou"
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    num_images: int = 0
    num_gt: int = 0
    num_pred: int = 0
    num_classes: int = 0

    def format_summary(self) -> str:
        mode = (self.match_mode or "iou").upper()
        return (
            f"YOLO-val metrics (overlap={mode}):\n"
            f"  Precision (准确率): {self.precision:.4f}\n"
            f"  Recall (召回率):    {self.recall:.4f}\n"
            f"  mAP50:             {self.map50:.4f}\n"
            f"  mAP50-95:          {self.map50_95:.4f}\n"
            f"  ({self.num_images} images, {self.num_gt} GT, "
            f"{self.num_pred} pred, {self.num_classes} class(es))"
        )


def load_classes_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def parse_yolo_txt(
    path: str,
    classes: Sequence[str],
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    """Parse YOLO HBB txt to shape dicts (rectangle, pixel coords)."""
    shapes: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
    except OSError as e:
        logger.warning("Cannot read YOLO file %s: %s", path, e)
        return shapes

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_index = int(float(parts[0]))
            cx, cy, nw, nh = (float(parts[i]) for i in range(1, 5))
            score = float(parts[5]) if len(parts) >= 6 else 1.0
        except (TypeError, ValueError):
            continue
        if class_index < 0 or class_index >= len(classes):
            continue
        label = classes[class_index]
        xmin = (cx - nw / 2) * img_w
        ymin = (cy - nh / 2) * img_h
        xmax = (cx + nw / 2) * img_w
        ymax = (cy + nh / 2) * img_h
        shape = _make_rect_shape(label, xmin, ymin, xmax, ymax, status=None)
        shape["score"] = float(score)
        shapes.append(shape)
    return shapes


def _make_rect_shape(
    label: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    status: Optional[str],
    source: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if source:
        shape = copy.deepcopy(source)
        shape["label"] = label
        shape["shape_type"] = "rectangle"
        shape["points"] = [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ]
    else:
        shape = {
            "label": label,
            "score": None,
            "points": [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ],
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {},
            "kie_linking": [],
        }
    attrs = shape.setdefault("attributes", {})
    if status:
        attrs[COMPARE_STATUS] = status
    elif COMPARE_STATUS in attrs:
        del attrs[COMPARE_STATUS]
    return shape


def box_overlap_matrix(
    boxes_gt: np.ndarray,
    boxes_pred: np.ndarray,
    mode: str = "ios",
    eps: float = 1e-7,
) -> np.ndarray:
    """Pairwise overlap (M_gt, N_pred)."""
    if boxes_gt.size == 0 or boxes_pred.size == 0:
        return np.zeros((len(boxes_gt), len(boxes_pred)), dtype=np.float64)
    g = boxes_gt.astype(np.float64).reshape(-1, 1, 4)
    p = boxes_pred.astype(np.float64).reshape(1, -1, 4)
    ix1 = np.maximum(g[..., 0], p[..., 0])
    iy1 = np.maximum(g[..., 1], p[..., 1])
    ix2 = np.minimum(g[..., 2], p[..., 2])
    iy2 = np.minimum(g[..., 3], p[..., 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_g = np.maximum(0.0, g[..., 2] - g[..., 0]) * np.maximum(
        0.0, g[..., 3] - g[..., 1]
    )
    area_p = np.maximum(0.0, p[..., 2] - p[..., 0]) * np.maximum(
        0.0, p[..., 3] - p[..., 1]
    )
    if mode == "ios":
        denom = np.minimum(area_g, area_p)
        return np.where(denom > eps, inter / np.maximum(denom, eps), 0.0)
    union = area_g + area_p - inter
    return np.where(union > eps, inter / np.maximum(union, eps), 0.0)


def _boxes_from_shapes(shapes: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[int]]:
    boxes: List[List[float]] = []
    valid_idx: List[int] = []
    for i, s in enumerate(shapes):
        b = bbox_from_shape(s)
        if b is None:
            continue
        boxes.append(list(b))
        valid_idx.append(i)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float64), valid_idx
    return np.array(boxes, dtype=np.float64), valid_idx


def _labels_aligned(
    gt_shapes: List[Dict[str, Any]],
    pred_shapes: List[Dict[str, Any]],
    gi: int,
    pi: int,
    same_class_only: bool,
) -> bool:
    if not same_class_only:
        return True
    return str(gt_shapes[gi].get("label", "")) == str(
        pred_shapes[pi].get("label", "")
    )


def labels_in_shapes(shapes: Sequence[Dict[str, Any]]) -> set:
    return {
        str(s.get("label", "")).strip()
        for s in shapes
        if str(s.get("label", "")).strip()
    }


def _pair_overlap(a: BBox, b: BBox, mode: str, eps: float = 1e-7) -> float:
    mat = box_overlap_matrix(
        np.array([a], dtype=np.float64),
        np.array([b], dtype=np.float64),
        mode=mode,
        eps=eps,
    )
    return float(mat[0, 0]) if mat.size else 0.0


def _shape_score(shape: Dict[str, Any]) -> float:
    score = shape.get("score")
    try:
        if score is None:
            return 1.0
        return float(score)
    except (TypeError, ValueError):
        return 1.0


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-point interpolated AP (area under precision-recall curve)."""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], np.asarray(recall, dtype=np.float64), [1.0]))
    mpre = np.concatenate(([1.0], np.asarray(precision, dtype=np.float64), [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _ap_precision_recall_for_class(
    *,
    preds: List[Tuple[int, BBox, float]],
    gts_by_image: Dict[int, List[BBox]],
    thr: float,
    match_mode: str,
) -> Tuple[float, float, float, int, int]:
    """Return (ap, precision, recall, tp_count, fp_count) for one class."""
    n_gt = sum(len(v) for v in gts_by_image.values())
    if n_gt == 0:
        return 0.0, 0.0, 0.0, 0, len(preds)

    preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)
    matched: Dict[int, List[bool]] = {
        img_id: [False] * len(boxes) for img_id, boxes in gts_by_image.items()
    }
    tps = np.zeros(len(preds_sorted), dtype=np.float64)
    fps = np.zeros(len(preds_sorted), dtype=np.float64)

    for i, (img_id, box, _score) in enumerate(preds_sorted):
        gt_boxes = gts_by_image.get(img_id) or []
        best_ov = 0.0
        best_j = -1
        for j, gt_box in enumerate(gt_boxes):
            if matched[img_id][j]:
                continue
            ov = _pair_overlap(gt_box, box, match_mode)
            if ov > best_ov:
                best_ov = ov
                best_j = j
        if best_j >= 0 and best_ov >= thr:
            matched[img_id][best_j] = True
            tps[i] = 1.0
        else:
            fps[i] = 1.0

    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recall_curve = tp_cum / float(n_gt)
    ap = _compute_ap(recall_curve, precision_curve)

    tp_total = int(tp_cum[-1]) if tp_cum.size else 0
    fp_total = int(fp_cum[-1]) if fp_cum.size else 0
    precision = float(tp_total / max(tp_total + fp_total, 1))
    recall = float(tp_total / max(n_gt, 1))
    return ap, precision, recall, tp_total, fp_total


def compute_det_val_metrics(
    image_records: Sequence[
        Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]
    ],
    *,
    match_mode: str = "iou",
    common_labels: Optional[set] = None,
) -> DetValMetrics:
    """Compute Precision / Recall / mAP50 / mAP50-95 like YOLO val.

    ``image_records``: iterable of ``(image_id, gt_shapes, pred_shapes)``.
    Overlap uses ``match_mode`` (``ios`` or ``iou``) from compare settings.
    mAP thresholds follow COCO/YOLO: 0.50 and 0.50:0.05:0.95.
    Precision/Recall are reported at overlap threshold 0.50.
    """
    mode = "ios" if str(match_mode).lower() == "ios" else "iou"
    metrics = DetValMetrics(match_mode=mode)
    if not image_records:
        return metrics

    # label -> img_id -> gt boxes
    gts: Dict[str, Dict[int, List[BBox]]] = {}
    # label -> list of (img_id, box, score)
    preds: Dict[str, List[Tuple[int, BBox, float]]] = {}

    for img_id, gt_shapes, pred_shapes in image_records:
        metrics.num_images += 1
        gt_work = [
            s for s in gt_shapes if get_compare_status(s) != STATUS_SUSPECT_FN
        ]
        if common_labels is not None:
            gt_cmp, _ = _partition_by_common_labels(gt_work, common_labels)
            pred_cmp, _ = _partition_by_common_labels(pred_shapes, common_labels)
        else:
            gt_cmp, pred_cmp = gt_work, pred_shapes

        for shape in gt_cmp:
            box = bbox_from_shape(shape)
            if box is None:
                continue
            label = _shape_label(shape)
            if not label:
                continue
            gts.setdefault(label, {}).setdefault(img_id, []).append(box)
            metrics.num_gt += 1

        for shape in pred_cmp:
            box = bbox_from_shape(shape)
            if box is None:
                continue
            label = _shape_label(shape)
            if not label:
                continue
            preds.setdefault(label, []).append(
                (img_id, box, _shape_score(shape))
            )
            metrics.num_pred += 1

    labels = sorted(set(gts.keys()) | set(preds.keys()))
    # mAP averages only over classes that have GT (COCO/YOLO style)
    eval_labels = [lb for lb in labels if sum(len(v) for v in gts.get(lb, {}).values()) > 0]
    metrics.num_classes = len(eval_labels)
    if not eval_labels:
        return metrics

    thr_list = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95
    ap50_list: List[float] = []
    ap_all: List[float] = []
    tp50 = 0
    fp50 = 0
    gt50 = 0

    for label in eval_labels:
        gts_by_img = gts.get(label, {})
        pred_list = preds.get(label, [])
        gt50 += sum(len(v) for v in gts_by_img.values())
        class_aps: List[float] = []
        for thr in thr_list:
            ap, precision, recall, tp, fp = _ap_precision_recall_for_class(
                preds=pred_list,
                gts_by_image=gts_by_img,
                thr=thr,
                match_mode=mode,
            )
            class_aps.append(ap)
            if abs(thr - 0.5) < 1e-9:
                ap50_list.append(ap)
                tp50 += tp
                fp50 += fp
        ap_all.append(float(np.mean(class_aps)) if class_aps else 0.0)

    metrics.map50 = float(np.mean(ap50_list)) if ap50_list else 0.0
    metrics.map50_95 = float(np.mean(ap_all)) if ap_all else 0.0
    metrics.precision = float(tp50 / max(tp50 + fp50, 1))
    metrics.recall = float(tp50 / max(gt50, 1))
    return metrics


def common_compare_labels(
    gt_shapes: Sequence[Dict[str, Any]],
    pred_shapes: Sequence[Dict[str, Any]],
    classes: Optional[Sequence[str]] = None,
) -> set:
    """Labels present in both GT and prediction vocabulary (classes.txt)."""
    gt_labels = labels_in_shapes(gt_shapes)
    if classes:
        pred_vocab = {str(c).strip() for c in classes if str(c).strip()}
    else:
        pred_vocab = labels_in_shapes(pred_shapes)
    return gt_labels & pred_vocab


def _shape_label(shape: Dict[str, Any]) -> str:
    return str(shape.get("label", "")).strip()


def _partition_by_common_labels(
    shapes: List[Dict[str, Any]],
    common: set,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (comparable, skipped) shapes by label membership in *common*."""
    comparable: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for s in shapes:
        if _shape_label(s) in common:
            comparable.append(s)
        else:
            skipped.append(s)
    return comparable, skipped


def match_gt_pred(
    gt_shapes: List[Dict[str, Any]],
    pred_shapes: List[Dict[str, Any]],
    settings: CompareSettings,
) -> Tuple[np.ndarray, np.ndarray]:
    """Many-to-one class-aware matching (val_error_export style)."""
    n_gt = len(gt_shapes)
    n_pr = len(pred_shapes)
    gt_matched = np.zeros(n_gt, dtype=bool)
    pred_matched = np.zeros(n_pr, dtype=bool)
    if n_gt == 0 or n_pr == 0:
        return gt_matched, pred_matched

    gt_boxes, gt_idx = _boxes_from_shapes(gt_shapes)
    pr_boxes, pr_idx = _boxes_from_shapes(pred_shapes)
    if gt_boxes.shape[0] == 0 or pr_boxes.shape[0] == 0:
        return gt_matched, pred_matched

    overlap = box_overlap_matrix(gt_boxes, pr_boxes, mode=settings.match_mode)
    thr = float(settings.match_thresh)

    for gi_local, gi in enumerate(gt_idx):
        for pi_local, pi in enumerate(pr_idx):
            if not _labels_aligned(
                gt_shapes, pred_shapes, gi, pi, settings.same_class_only
            ):
                overlap[gi_local, pi_local] = 0.0

    for pi_local, pi in enumerate(pr_idx):
        pred_matched[pi] = bool((overlap[:, pi_local] >= thr).any())

    for gi_local, gi in enumerate(gt_idx):
        gt_matched[gi] = bool((overlap[gi_local, :] >= thr).any())

    return gt_matched, pred_matched


def _status_pair_for_mode(review_mode: str) -> Tuple[str, str]:
    """Return (pred_side_status, gt_side_status) for the review mode."""
    if str(review_mode) == REVIEW_MODE_MODEL_EVAL:
        return STATUS_MODEL_FALSE, STATUS_MODEL_MISS
    return STATUS_SUSPECT_FN, STATUS_SUSPECT_FP


def build_review_shapes(
    gt_shapes: List[Dict[str, Any]],
    pred_shapes: List[Dict[str, Any]],
    gt_matched: np.ndarray,
    pred_matched: np.ndarray,
    *,
    check_fp: bool = True,
    check_fn: bool = True,
    review_mode: str = REVIEW_MODE_ANNOTATION,
) -> List[Dict[str, Any]]:
    """Only unmatched boxes get review marks; matched boxes keep GT as-is.

    ``check_fp`` / ``check_fn`` gate which suspect marks are written. Unmatched
    GT without ``check_fp`` stays as a normal box (no compare flag). Unmatched
    pred without ``check_fn`` is not appended.

    Annotation compare: pred→漏标, GT→误标.
    Model eval: pred→误报, GT→漏报.
    """
    pred_status, gt_status = _status_pair_for_mode(review_mode)
    out: List[Dict[str, Any]] = []
    for i, gt in enumerate(gt_shapes):
        s = strip_compare_metadata(gt)
        if not gt_matched[i] and check_fp:
            attrs = s.setdefault("attributes", {})
            if not isinstance(attrs, dict):
                attrs = {}
                s["attributes"] = attrs
            attrs[COMPARE_STATUS] = gt_status
        out.append(s)

    if check_fn:
        for j, pred in enumerate(pred_shapes):
            if pred_matched[j]:
                continue
            s = copy.deepcopy(pred)
            attrs = s.setdefault("attributes", {})
            if not isinstance(attrs, dict):
                attrs = {}
                s["attributes"] = attrs
            attrs[COMPARE_STATUS] = pred_status
            out.append(s)
    return out


def _shape_status(shape: Dict[str, Any]) -> Optional[str]:
    attrs = shape.get("attributes") or {}
    return attrs.get(COMPARE_STATUS)


def _bbox_ios(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    smaller = min(bbox_area(a), bbox_area(b))
    return inter / smaller if smaller > 0 else 0.0


def dedup_shapes_ios(
    shapes: List[Dict[str, Any]],
    ios_threshold: float,
) -> List[Dict[str, Any]]:
    """Remove lower-priority overlapping shapes (matched > fn > fp)."""
    if len(shapes) <= 1:
        return shapes

    n = len(shapes)
    boxes = [bbox_from_shape(s) for s in shapes]
    remove: set[int] = set()
    thr = float(ios_threshold)

    for i in range(n):
        if i in remove or boxes[i] is None:
            continue
        li = str(shapes[i].get("label", ""))
        si = _shape_status(shapes[i]) or STATUS_MATCHED
        pi = _PRIORITY.get(si, 0)
        for j in range(i + 1, n):
            if j in remove or boxes[j] is None:
                continue
            if li != str(shapes[j].get("label", "")):
                continue
            if _bbox_ios(boxes[i], boxes[j]) <= thr:
                continue
            sj = _shape_status(shapes[j]) or STATUS_MATCHED
            pj = _PRIORITY.get(sj, 0)
            if pi > pj:
                remove.add(j)
            elif pj > pi:
                remove.add(i)
                break
            else:
                if bbox_area(boxes[i]) >= bbox_area(boxes[j]):
                    remove.add(j)
                else:
                    remove.add(i)
                    break

    return [s for i, s in enumerate(shapes) if i not in remove]


def strip_compare_metadata(shape: Dict[str, Any]) -> Dict[str, Any]:
    s = copy.deepcopy(shape)
    attrs = s.get("attributes")
    if isinstance(attrs, dict):
        for k in [k for k in attrs if str(k).startswith("_compare")]:
            del attrs[k]
        if not attrs:
            s["attributes"] = {}
    return s


def shape_has_compare_metadata(shape: Any) -> bool:
    if hasattr(shape, "attributes"):
        attrs = shape.attributes
    else:
        attrs = (shape.get("attributes") if isinstance(shape, dict) else None)
    if not isinstance(attrs, dict):
        return False
    return any(str(k).startswith("_compare") for k in attrs)


def strip_compare_metadata_in_place(shape: Any) -> bool:
    """Remove internal _compare_* keys from a live Shape. Returns True if changed."""
    if not hasattr(shape, "attributes"):
        return False
    attrs = shape.attributes
    if not isinstance(attrs, dict):
        return False
    keys = [k for k in attrs if str(k).startswith("_compare")]
    if not keys:
        return False
    for k in keys:
        del attrs[k]
    return True


def finalize_shape(shape: Dict[str, Any]) -> Dict[str, Any]:
    return strip_compare_metadata(shape)


def accept_pending_compare_shapes(
    shapes: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Keep all pending pred/GT review boxes; strip flags (pred-side becomes formal GT)."""
    fn_accepted = 0
    fp_accepted = 0
    out: List[Dict[str, Any]] = []
    for shape in shapes:
        status = get_compare_status(shape)
        if status in PENDING_PRED_STATUSES:
            fn_accepted += 1
            accepted = apply_review_action(shape, FN_ACCEPT_LABEL)
            if accepted is not None:
                out.append(accepted)
            continue
        if status in PENDING_GT_STATUSES:
            fp_accepted += 1
            accepted = apply_review_action(shape, FP_KEEP)
            if accepted is not None:
                out.append(accepted)
            continue
        out.append(shape)
    return out, fn_accepted, fp_accepted


def clear_compare_results_shapes(
    shapes: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Remove pred-side marks (漏标/误报); keep GT-side boxes and strip flags.

    Also clears annotation-compare and model-eval marks.
    """
    fn_removed = 0
    fp_cleared = 0
    out: List[Dict[str, Any]] = []
    for shape in shapes:
        status = get_compare_status(shape)
        if status in PENDING_PRED_STATUSES:
            fn_removed += 1
            continue
        if status in PENDING_GT_STATUSES:
            fp_cleared += 1
            out.append(strip_compare_metadata(shape))
            continue
        if shape_has_compare_metadata(shape):
            out.append(strip_compare_metadata(shape))
            continue
        out.append(shape)
    return out, fn_removed, fp_cleared


def get_compare_status(shape: Any) -> Optional[str]:
    if hasattr(shape, "attributes"):
        attrs = shape.attributes or {}
    else:
        attrs = shape.get("attributes") or {}
    return attrs.get(COMPARE_STATUS)


def is_pred_side_status(status: Optional[str]) -> bool:
    return status in PENDING_PRED_STATUSES


def is_gt_side_status(status: Optional[str]) -> bool:
    return status in PENDING_GT_STATUSES


def compare_review_rgb(status: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if status in PENDING_PRED_STATUSES:
        return COLOR_SUSPECT_FN
    if status in PENDING_GT_STATUSES:
        return COLOR_SUSPECT_FP
    return None


def compare_status_display_text(status: Any) -> Optional[str]:
    if status == STATUS_SUSPECT_FN:
        return "疑是漏标"
    if status == STATUS_SUSPECT_FP:
        return "疑是误标"
    if status == STATUS_MODEL_FALSE:
        return "误报"
    if status == STATUS_MODEL_MISS:
        return "漏报"
    return None


def apply_review_action(
    shape: Any,
    action: str,
    *,
    new_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Apply user review choice. Returns updated shape dict, or None to delete."""
    if hasattr(shape, "to_dict"):
        data = shape.to_dict()
    else:
        data = copy.deepcopy(shape)

    status = get_compare_status(data)
    if status in PENDING_PRED_STATUSES:
        if action == FN_REJECT:
            return None
        s = finalize_shape(data)
        if action == FN_ACCEPT_NEW_LABEL and new_label:
            s["label"] = new_label.strip()
        return s
    if status in PENDING_GT_STATUSES:
        if action == FP_CONFIRM_DELETE:
            return None
        s = finalize_shape(data)
        if action == FP_CONFIRM_RELABEL and new_label:
            s["label"] = new_label.strip()
        return s
    return finalize_shape(data)


def compare_shapes(
    gt_shapes: List[Dict[str, Any]],
    pred_shapes: List[Dict[str, Any]],
    settings: CompareSettings,
    *,
    common_labels: Optional[set] = None,
    classes: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], CompareStats]:
    review_mode = getattr(settings, "review_mode", REVIEW_MODE_ANNOTATION)
    pred_status, gt_status = _status_pair_for_mode(review_mode)
    # Pred-side pending boxes are not GT; exclude them on re-run.
    gt_work = [
        s for s in gt_shapes if get_compare_status(s) not in PENDING_PRED_STATUSES
    ]
    common = (
        common_labels
        if common_labels is not None
        else common_compare_labels(gt_work, pred_shapes, classes)
    )
    if not common:
        passthrough = [strip_compare_metadata(copy.deepcopy(s)) for s in gt_work]
        return passthrough, CompareStats()

    gt_skip = [
        strip_compare_metadata(copy.deepcopy(s))
        for s in gt_work
        if _shape_label(s) not in common
    ]
    gt_cmp, _ = _partition_by_common_labels(gt_work, common)
    pred_cmp, _ = _partition_by_common_labels(pred_shapes, common)

    gt_matched, pred_matched = match_gt_pred(gt_cmp, pred_cmp, settings)
    n_matched = int(gt_matched.sum())
    merged = build_review_shapes(
        gt_cmp,
        pred_cmp,
        gt_matched,
        pred_matched,
        check_fp=bool(settings.check_fp),
        check_fn=bool(settings.check_fn),
        review_mode=review_mode,
    )
    merged = dedup_shapes_ios(merged, settings.dedup_ios_thresh)
    merged = gt_skip + merged
    stats = CompareStats(
        matched=n_matched,
        suspect_fn=sum(1 for s in merged if _shape_status(s) == pred_status),
        suspect_fp=sum(1 for s in merged if _shape_status(s) == gt_status),
    )
    return merged, stats


def count_pending_review(shapes: Sequence[Any]) -> int:
    return sum(
        1
        for s in shapes
        if get_compare_status(s) in PENDING_REVIEW_STATUSES
    )
