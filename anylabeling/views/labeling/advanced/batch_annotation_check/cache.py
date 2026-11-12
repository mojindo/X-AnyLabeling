"""Disk cache for Annotation Inspector crops."""

from __future__ import annotations

import json
import os
import os.path as osp
import shutil
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

from anylabeling.config import get_work_directory
from anylabeling.views.labeling.classifier.utils import get_label_file_path
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.advanced.batch_annotation_check.cropper import (
    BBox,
    crop_polygon_bgra,
    crop_image_bgr,
    read_image_bgr,
    shape_to_bbox,
)


@dataclass(frozen=True)
class CropItem:
    image_path: str
    label_json_path: str
    shape_index: int
    label: str
    shape_type: str
    bbox: Dict[str, int]  # xmin,ymin,xmax,ymax
    crop_path: str
    fp_backend: Optional[str] = None
    fp_is_match: Optional[bool] = None
    fp_confidence: Optional[float] = None
    fp_reason: Optional[str] = None
    fp_error: Optional[str] = None
    # False-negative (miss) results, rendered as a second row (yellow).
    fn_source_image_path: Optional[str] = None
    fn_label: Optional[str] = None
    fn_bbox01: Optional[List[float]] = None
    fn_score: Optional[float] = None


def new_session_cache_dir() -> str:
    root = get_work_directory()
    sid = str(uuid.uuid4())
    return osp.join(root, "xanylabeling_data", "annotation_inspector", sid)


def _same_image_path(a: object, b: object) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    try:
        return osp.normcase(osp.normpath(osp.abspath(a))) == osp.normcase(
            osp.normpath(osp.abspath(b))
        )
    except Exception:
        return str(a) == str(b)


def _replace_items_for_image_keep_order(
    old_items: List[Dict[str, Any]],
    image_path: str,
    new_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Put new_items where this image's crops were, not at the end of the list."""
    insert_pos = len(old_items)
    for i, it in enumerate(old_items):
        if isinstance(it, dict) and _same_image_path(it.get("image_path"), image_path):
            insert_pos = i
            break
    filtered = [
        it
        for it in old_items
        if isinstance(it, dict)
        and not _same_image_path(it.get("image_path"), image_path)
    ]
    return filtered[:insert_pos] + list(new_items) + filtered[insert_pos:]


class InspectorCache:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.crops_dir = osp.join(root_dir, "crops")
        self.index_path = osp.join(root_dir, "index.json")
        self._index: Dict[str, Any] = {}
        self._index_mtime: Optional[float] = None
        os.makedirs(self.crops_dir, exist_ok=True)

    def load_index(self) -> Dict[str, Any]:
        try:
            mtime = osp.getmtime(self.index_path) if osp.isfile(self.index_path) else None
        except Exception:
            mtime = None
        # Avoid re-reading index.json on every image switch if unchanged.
        if self._index and (mtime is not None) and (self._index_mtime == mtime):
            return self._index
        if osp.isfile(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                self._index = json.load(f)
            self._index_mtime = mtime
        else:
            self._index = {"version": 1, "items": []}
            self._index_mtime = None
        self._normalize_index_paths()
        return self._index

    def _normalize_index_paths(self) -> None:
        """Make imported cache portable across machines/paths."""
        idx = self._index
        if not isinstance(idx, dict):
            return
        items = idx.get("items")
        if not isinstance(items, list):
            return
        for it in items:
            if not isinstance(it, dict):
                continue
            cp = it.get("crop_path")
            if isinstance(cp, str) and cp:
                if not osp.isfile(cp):
                    bn = osp.basename(cp)
                    cand = osp.join(self.crops_dir, bn)
                    if osp.isfile(cand):
                        it["crop_path"] = cand
            ip = it.get("image_path")
            if isinstance(ip, str) and ip:
                it.setdefault("image_basename", osp.basename(ip))
            # Normalize fn crop paths too (if present)
            cp2 = it.get("fn_crop_path")
            if isinstance(cp2, str) and cp2:
                if not osp.isfile(cp2):
                    bn2 = osp.basename(cp2)
                    cand2 = osp.join(self.root_dir, "crops_fn", bn2)
                    if osp.isfile(cand2):
                        it["fn_crop_path"] = cand2

    def is_valid_cache_dir(self) -> bool:
        """Best-effort validation for an imported cache directory."""
        try:
            if not osp.isdir(self.root_dir):
                return False
            if not osp.isfile(self.index_path):
                return False
            if not osp.isdir(self.crops_dir):
                return False
            idx = self.load_index()
            items = idx.get("items") if isinstance(idx, dict) else None
            return isinstance(items, list)
        except Exception:
            return False

    def save_index(self) -> None:
        os.makedirs(osp.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)
        try:
            self._index_mtime = osp.getmtime(self.index_path)
        except Exception:
            self._index_mtime = None

    def set_vlm_config(self, cfg: Dict[str, Any]) -> None:
        """Persist VLM config for inspector (best-effort)."""
        self.load_index()
        if not isinstance(self._index, dict):
            return
        safe = {}
        for k in ("base_url", "api_key", "model"):
            v = cfg.get(k)
            if isinstance(v, str) and v.strip():
                safe[k] = v.strip()
        if safe:
            self._index["vlm_config"] = safe
            self.save_index()

    def get_vlm_config(self) -> Dict[str, str]:
        self.load_index()
        idx = self._index
        if not isinstance(idx, dict):
            return {}
        cfg = idx.get("vlm_config")
        if not isinstance(cfg, dict):
            return {}
        out: Dict[str, str] = {}
        for k in ("base_url", "api_key", "model"):
            v = cfg.get(k)
            if isinstance(v, str) and v.strip():
                out[k] = v.strip()
        return out

    def set_fp_result(
        self,
        *,
        image_basename: str,
        shape_index: int,
        backend: str,
        is_match: Optional[bool],
        confidence: Optional[float],
        reason: Optional[str],
        error: Optional[str],
    ) -> None:
        """Persist false-positive check result onto the matching item."""
        self.load_index()
        items = self._index.get("items") or []
        if not isinstance(items, list):
            return
        for it in items:
            if not isinstance(it, dict):
                continue
            ib = it.get("image_basename") or ""
            if ib != image_basename:
                continue
            try:
                si = int(it.get("shape_index"))
            except Exception:
                continue
            if si != int(shape_index):
                continue
            it["fp_backend"] = str(backend)
            it["fp_is_match"] = is_match
            it["fp_confidence"] = confidence
            it["fp_reason"] = reason
            it["fp_error"] = error
            break
        self.save_index()

    def set_fn_results_for_image(
        self,
        *,
        image_basename: str,
        results: List[Dict[str, Any]],
    ) -> None:
        """
        Store FN detections for a given image.
        Each result should be:
          {"fn_crop_path": str, "fn_label": str, "fn_bbox01": [x1,y1,x2,y2], "fn_score": float|None}
        """
        self.load_index()
        idx = self._index
        if not isinstance(idx, dict):
            return
        idx.setdefault("fn_results", {})
        if not isinstance(idx.get("fn_results"), dict):
            idx["fn_results"] = {}
        idx["fn_results"][image_basename] = results
        self.save_index()

    def get_fn_items_for_image(self, image_path: str) -> List[CropItem]:
        """Return FN crop items for second-row display."""
        self.load_index()
        target_base = osp.basename(image_path or "")
        idx = self._index
        out: List[CropItem] = []
        if not isinstance(idx, dict):
            return out
        fnr = idx.get("fn_results") or {}
        if not isinstance(fnr, dict):
            return out
        rows = fnr.get(target_base) or []
        if not isinstance(rows, list):
            return out
        for r in rows:
            if not isinstance(r, dict):
                continue
            cp = r.get("fn_crop_path") or ""
            if isinstance(cp, str) and cp and not osp.isfile(cp):
                bn = osp.basename(cp)
                cand = osp.join(self.root_dir, "crops_fn", bn)
                if osp.isfile(cand):
                    cp = cand
            try:
                out.append(
                    CropItem(
                        image_path=image_path,
                        label_json_path="",
                        shape_index=-1,
                        label="",
                        shape_type="",
                        bbox={},
                        crop_path=str(cp),
                        fn_source_image_path=image_path,
                        fn_label=str(r.get("fn_label") or ""),
                        fn_bbox01=r.get("fn_bbox01"),
                        fn_score=r.get("fn_score"),
                    )
                )
            except Exception:
                continue
        return out

    def get_all_crop_items(self) -> List[CropItem]:
        """All annotation crops in the cache (whole dataset)."""
        self.load_index()
        out: List[CropItem] = []
        for i in self._index.get("items", []) or []:
            if not isinstance(i, dict):
                continue
            try:
                out.append(
                    CropItem(
                        image_path=i["image_path"],
                        label_json_path=i["label_json_path"],
                        shape_index=int(i["shape_index"]),
                        label=str(i.get("label", "")),
                        shape_type=str(i.get("shape_type", "")),
                        bbox=dict(i.get("bbox") or {}),
                        crop_path=i["crop_path"],
                        fp_backend=i.get("fp_backend"),
                        fp_is_match=i.get("fp_is_match"),
                        fp_confidence=i.get("fp_confidence"),
                        fp_reason=i.get("fp_reason"),
                        fp_error=i.get("fp_error"),
                    )
                )
            except Exception:
                continue
        return out

    def get_all_fn_items(self) -> List[CropItem]:
        """All FN candidate crops across images (basename-keyed fn_results)."""
        self.load_index()
        out: List[CropItem] = []
        idx = self._index
        if not isinstance(idx, dict):
            return out
        fnr = idx.get("fn_results") or {}
        if not isinstance(fnr, dict):
            return out
        base_to_path: Dict[str, str] = {}
        for i in idx.get("items", []) or []:
            if not isinstance(i, dict):
                continue
            ip = i.get("image_path")
            if not isinstance(ip, str) or not ip:
                continue
            ib = i.get("image_basename") or osp.basename(ip)
            base_to_path[ib] = ip

        for basename, rows in fnr.items():
            if not isinstance(rows, list):
                continue
            ip = base_to_path.get(str(basename))
            if not ip:
                continue
            for r in rows:
                if not isinstance(r, dict):
                    continue
                cp = r.get("fn_crop_path") or ""
                if isinstance(cp, str) and cp and not osp.isfile(cp):
                    bn = osp.basename(cp)
                    cand = osp.join(self.root_dir, "crops_fn", bn)
                    if osp.isfile(cand):
                        cp = cand
                try:
                    out.append(
                        CropItem(
                            image_path=ip,
                            label_json_path="",
                            shape_index=-1,
                            label="",
                            shape_type="",
                            bbox={},
                            crop_path=str(cp),
                            fn_source_image_path=ip,
                            fn_label=str(r.get("fn_label") or ""),
                            fn_bbox01=r.get("fn_bbox01"),
                            fn_score=r.get("fn_score"),
                        )
                    )
                except Exception:
                    continue
        return out

    def clear(self) -> None:
        try:
            shutil.rmtree(self.root_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("Failed to clear inspector cache: %s", e)

    def export_to(self, dst_dir: str) -> None:
        if not dst_dir:
            return
        if osp.abspath(dst_dir) == osp.abspath(self.root_dir):
            return
        if osp.exists(dst_dir) and os.listdir(dst_dir):
            raise ValueError("Destination directory is not empty.")
        shutil.copytree(self.root_dir, dst_dir, dirs_exist_ok=True)

    def build_index(
        self,
        image_paths: Sequence[str],
        *,
        output_dir: Optional[str],
        labels_filter: Sequence[str],
        force: bool = False,
        padding_px: int = 4,
        jpeg_quality: int = 85,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Build crops for all images for selected labels."""
        labels_set = {l for l in labels_filter if isinstance(l, str) and l}
        if force or not self._index:
            self._index = {"version": 1, "items": [], "errors": []}
        items: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        total = len(image_paths)
        for i_img, image_path in enumerate(image_paths, start=1):
            if should_cancel is not None and should_cancel():
                raise RuntimeError("Canceled")
            if on_progress is not None:
                try:
                    on_progress(i_img, total, image_path)
                except Exception:
                    pass
            label_json_path = get_label_file_path(image_path, output_dir)
            if not osp.isfile(label_json_path):
                continue
            try:
                with open(label_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                errors.append(
                    {"image": image_path, "label_json": label_json_path, "error": str(e)}
                )
                continue
            shapes = data.get("shapes") or []
            if not isinstance(shapes, list) or not shapes:
                continue
            iw, ih = data.get("imageWidth"), data.get("imageHeight")
            try:
                iw = int(iw)
                ih = int(ih)
            except Exception:
                iw = ih = None
            if should_cancel is not None and should_cancel():
                raise RuntimeError("Canceled")
            try:
                img_bgr = read_image_bgr(image_path)
            except Exception as e:
                errors.append({"image": image_path, "error": str(e)})
                continue
            if iw is None or ih is None:
                ih, iw = img_bgr.shape[:2]

            base = osp.splitext(osp.basename(image_path))[0]
            for idx, s in enumerate(shapes):
                if should_cancel is not None and should_cancel():
                    raise RuntimeError("Canceled")
                try:
                    label = s.get("label")
                    if labels_set and label not in labels_set:
                        continue
                    points = s.get("points") or []
                    if not isinstance(points, list):
                        continue
                except Exception:
                    continue
                bbox = shape_to_bbox(points, iw, ih, padding_px=padding_px)
                if bbox is None:
                    continue
                shape_type = str(s.get("shape_type", "") or "")
                if shape_type == "polygon":
                    crop = crop_polygon_bgra(img_bgr, bbox=bbox, polygon_points=points)
                    if crop.size == 0:
                        continue
                    fn = f"{base}__{idx:06d}.png"
                    out_path = osp.join(self.crops_dir, fn)
                    # PNG + alpha, use imencode to support Unicode paths.
                    try:
                        enc = cv2.imencode(".png", crop)[1]
                        enc.tofile(out_path)
                    except Exception:
                        cv2.imwrite(out_path, crop)
                else:
                    crop = crop_image_bgr(img_bgr, bbox)
                if crop.size == 0:
                    continue
                if shape_type != "polygon":
                    fn = f"{base}__{idx:06d}.jpg"
                    out_path = osp.join(self.crops_dir, fn)
                    cv2.imwrite(
                        out_path,
                        crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                    )
                items.append(
                    dict(
                        image_path=image_path,
                        image_basename=osp.basename(image_path),
                        label_json_path=label_json_path,
                        shape_index=idx,
                        label=str(label),
                        shape_type=shape_type,
                        bbox=dict(
                            xmin=bbox.xmin,
                            ymin=bbox.ymin,
                            xmax=bbox.xmax,
                            ymax=bbox.ymax,
                        ),
                        crop_path=out_path,
                    )
                )

        self._index["items"] = items
        self._index["errors"] = errors
        self._index["labels_filter"] = list(labels_set)
        self.save_index()

    def update_for_image(
        self,
        image_path: str,
        *,
        output_dir: Optional[str],
        labels_filter: Sequence[str],
        padding_px: int = 4,
    ) -> None:
        """Rebuild cache entries for a single image."""
        self.load_index()
        snapshot = list(self._index.get("items") or [])
        self.build_index(
            [image_path],
            output_dir=output_dir,
            labels_filter=labels_filter,
            force=False,
            padding_px=padding_px,
        )
        new_items = list(self._index.get("items") or [])
        self._index["items"] = _replace_items_for_image_keep_order(
            snapshot, image_path, new_items
        )
        self.save_index()

    def update_for_image_live(
        self,
        image_path: str,
        *,
        output_dir: Optional[str],
        labels_filter: Sequence[str],
        shapes: Sequence[Dict[str, Any]],
        image_w: int,
        image_h: int,
        padding_px: int = 4,
        jpeg_quality: int = 85,
    ) -> None:
        """Update crops for a single image from in-memory shapes (unsaved)."""
        self.load_index()
        snapshot = list(self._index.get("items") or [])
        labels_set = {l for l in labels_filter if isinstance(l, str) and l}
        label_json_path = get_label_file_path(image_path, output_dir)

        base = osp.splitext(osp.basename(image_path))[0]
        try:
            for fn in os.listdir(self.crops_dir):
                if fn.startswith(f"{base}__") and (
                    fn.lower().endswith(".jpg") or fn.lower().endswith(".png")
                ):
                    try:
                        os.remove(osp.join(self.crops_dir, fn))
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            img_bgr = read_image_bgr(image_path)
        except Exception as e:
            self._index.setdefault("errors", []).append(
                {"image": image_path, "error": str(e)}
            )
            self.save_index()
            return

        new_items: List[Dict[str, Any]] = []
        for idx, s in enumerate(shapes):
            try:
                label = s.get("label")
                if labels_set and label not in labels_set:
                    continue
                points = s.get("points") or []
                if not isinstance(points, list):
                    continue
            except Exception:
                continue
            bbox = shape_to_bbox(points, int(image_w), int(image_h), padding_px=padding_px)
            if bbox is None:
                continue
            shape_type = str(s.get("shape_type", "") or "")
            if shape_type == "polygon":
                crop = crop_polygon_bgra(img_bgr, bbox=bbox, polygon_points=points)
            else:
                crop = crop_image_bgr(img_bgr, bbox)
            if crop.size == 0:
                continue
            if shape_type == "polygon":
                fn = f"{base}__{idx:06d}.png"
                out_path = osp.join(self.crops_dir, fn)
                try:
                    enc = cv2.imencode(".png", crop)[1]
                    enc.tofile(out_path)
                except Exception:
                    cv2.imwrite(out_path, crop)
            else:
                fn = f"{base}__{idx:06d}.jpg"
                out_path = osp.join(self.crops_dir, fn)
                # cv2.imwrite may fail on Windows Unicode paths; use tofile instead.
                try:
                    enc = cv2.imencode(
                        ".jpg",
                        crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                    )[1]
                    enc.tofile(out_path)
                except Exception:
                    # Fallback to imwrite (best effort)
                    cv2.imwrite(
                        out_path,
                        crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                    )
            new_items.append(
                dict(
                    image_path=image_path,
                    image_basename=osp.basename(image_path),
                    label_json_path=label_json_path,
                    shape_index=idx,
                    label=str(label),
                    shape_type=shape_type,
                    bbox=dict(
                        xmin=bbox.xmin,
                        ymin=bbox.ymin,
                        xmax=bbox.xmax,
                        ymax=bbox.ymax,
                    ),
                    crop_path=out_path,
                )
            )

        self._index["items"] = _replace_items_for_image_keep_order(
            snapshot, image_path, new_items
        )
        self._index["labels_filter"] = list(labels_set)
        self.save_index()

    def get_items_for_image(self, image_path: str) -> List[CropItem]:
        self.load_index()
        out: List[CropItem] = []
        target_base = osp.basename(image_path or "")
        for i in self._index.get("items", []) or []:
            ip = i.get("image_path")
            ib = i.get("image_basename") or (
                osp.basename(ip) if isinstance(ip, str) else ""
            )
            if ip != image_path and ib != target_base:
                continue
            try:
                out.append(
                    CropItem(
                        image_path=i["image_path"],
                        label_json_path=i["label_json_path"],
                        shape_index=int(i["shape_index"]),
                        label=str(i.get("label", "")),
                        shape_type=str(i.get("shape_type", "")),
                        bbox=dict(i.get("bbox") or {}),
                        crop_path=i["crop_path"],
                        fp_backend=i.get("fp_backend"),
                        fp_is_match=i.get("fp_is_match"),
                        fp_confidence=i.get("fp_confidence"),
                        fp_reason=i.get("fp_reason"),
                        fp_error=i.get("fp_error"),
                    )
                )
            except Exception:
                continue
        return out

