"""Free inference helpers: scan trained weights, prepare sources, cleanup."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from anylabeling.services.auto_training.mytrain.config import (
    TASK_TYPES,
    get_default_project_dir,
    get_trainer_root_dir,
)
from anylabeling.views.labeling.logger import logger

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
WEIGHT_EXTS = {".pt", ".onnx"}


@dataclass
class WeightEntry:
    """A discoverable or uploaded weight for free inference."""

    display: str
    project_id: str
    run_dir: str
    weight_name: str
    weight_path: str
    source: str = "scan"  # scan | upload
    remote: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FreeInferJob:
    job_id: str
    job_dir: str
    source_dir: str
    source_kind: str  # image | folder | video
    weight: WeightEntry
    imgsz: int = 640
    created_at: str = ""
    labels_dir: str = ""
    meta: Dict = field(default_factory=dict)

    def save(self) -> None:
        os.makedirs(self.job_dir, exist_ok=True)
        path = os.path.join(self.job_dir, "meta.json")
        payload = {
            "job_id": self.job_id,
            "job_dir": self.job_dir,
            "source_dir": self.source_dir,
            "source_kind": self.source_kind,
            "weight": self.weight.to_dict(),
            "imgsz": self.imgsz,
            "created_at": self.created_at,
            "labels_dir": self.labels_dir,
            "meta": self.meta,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def get_free_infer_root() -> str:
    return os.path.join(get_trainer_root_dir(), "free_infer")


def list_local_train_projects() -> List[Tuple[str, str]]:
    """Return [(project_id, run_dir), ...] under runs/{task}/{name}."""
    root = Path(get_default_project_dir())
    out: List[Tuple[str, str]] = []
    if not root.is_dir():
        return out
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        # Skip non-task folders (profiles, etc.)
        if task_dir.name.lower() not in {t.lower() for t in TASK_TYPES} and (
            task_dir.name.lower()
            not in {"classify", "detect", "obb", "segment", "pose"}
        ):
            # Still allow any first-level folder that contains run children
            pass
        for run_dir in sorted(task_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            weights = run_dir / "weights"
            if weights.is_dir():
                out.append((f"{task_dir.name}/{run_dir.name}", str(run_dir)))
    return out


def list_local_free_infer_projects() -> List[Tuple[str, str]]:
    root = Path(get_free_infer_root())
    out: List[Tuple[str, str]] = []
    if not root.is_dir():
        return out
    for job_dir in sorted(root.iterdir()):
        if job_dir.is_dir() and not job_dir.name.startswith("."):
            out.append((f"free_infer/{job_dir.name}", str(job_dir)))
    return out


def scan_local_weights(
    *,
    task_filter: Optional[str] = None,
) -> List[WeightEntry]:
    """Scan trained projects for .pt / .onnx under weights/."""
    entries: List[WeightEntry] = []
    for project_id, run_dir in list_local_train_projects():
        if task_filter:
            task = project_id.split("/", 1)[0].lower()
            if task != task_filter.lower():
                continue
        weights_dir = os.path.join(run_dir, "weights")
        if not os.path.isdir(weights_dir):
            continue
        names: List[str] = []
        for preferred in ("best.pt", "last.pt"):
            p = os.path.join(weights_dir, preferred)
            if os.path.isfile(p):
                names.append(preferred)
        for name in sorted(os.listdir(weights_dir)):
            ext = os.path.splitext(name)[1].lower()
            if ext in WEIGHT_EXTS and name not in names:
                names.append(name)
        for name in names:
            path = os.path.join(weights_dir, name)
            entries.append(
                WeightEntry(
                    display=f"{project_id}/weights/{name}",
                    project_id=project_id,
                    run_dir=run_dir,
                    weight_name=name,
                    weight_path=path,
                    source="scan",
                    remote=False,
                )
            )
    return entries


def scan_remote_weights(
    client,
    home: str,
    *,
    tasks: Optional[Sequence[str]] = None,
    remote_base: str = "",
) -> List[WeightEntry]:
    """Scan remote ``_runs/{task}/{name}/weights`` for .pt/.onnx."""
    from anylabeling.services.auto_training.mytrain.remote.remote_paths import (
        runs_remote_dir,
    )

    task_list = list(tasks) if tasks else [
        t.lower() for t in ("detect", "classify", "obb", "segment", "pose")
    ]
    entries: List[WeightEntry] = []
    for task in task_list:
        base = runs_remote_dir(task, home, remote_base)
        try:
            names = client.listdir(base)
        except Exception:
            continue
        for run_name in sorted(names):
            if run_name in (".", "..") or run_name.startswith("."):
                continue
            run_dir = f"{base.rstrip('/')}/{run_name}"
            weights_dir = f"{run_dir}/weights"
            try:
                if not client.exists(weights_dir):
                    continue
                weight_names = client.listdir(weights_dir)
            except Exception:
                continue
            ordered: List[str] = []
            for preferred in ("best.pt", "last.pt"):
                if preferred in weight_names:
                    ordered.append(preferred)
            for name in sorted(weight_names):
                ext = os.path.splitext(name)[1].lower()
                if ext in WEIGHT_EXTS and name not in ordered:
                    ordered.append(name)
            project_id = f"{task}/{run_name}"
            for name in ordered:
                path = f"{weights_dir}/{name}"
                entries.append(
                    WeightEntry(
                        display=f"{project_id}/weights/{name}",
                        project_id=project_id,
                        run_dir=run_dir,
                        weight_name=name,
                        weight_path=path,
                        source="scan",
                        remote=True,
                    )
                )
    return entries


def create_free_infer_job_dir(prefix: str = "job") -> Tuple[str, str]:
    root = get_free_infer_root()
    os.makedirs(root, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    job_id = f"{prefix}_{stamp}"
    job_dir = os.path.join(root, job_id)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(os.path.join(job_dir, "source"), exist_ok=True)
    os.makedirs(os.path.join(job_dir, "weights"), exist_ok=True)
    return job_id, job_dir


def prepare_source_images(
    material_path: str,
    dest_source_dir: str,
    *,
    video_interval: int = 1,
    progress=None,
) -> Tuple[bool, str, str]:
    """
    Prepare inference images under ``dest_source_dir``.

    Returns (ok, source_kind, message_or_source_dir).
    """
    material_path = os.path.abspath(material_path or "")
    if not material_path or not os.path.exists(material_path):
        return False, "", f"Material not found: {material_path}"

    os.makedirs(dest_source_dir, exist_ok=True)
    # Clear previous source images
    for name in os.listdir(dest_source_dir):
        p = os.path.join(dest_source_dir, name)
        try:
            if os.path.isfile(p) or os.path.islink(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass

    if os.path.isfile(material_path):
        ext = os.path.splitext(material_path)[1].lower()
        if ext in IMAGE_EXTS:
            dst = os.path.join(dest_source_dir, os.path.basename(material_path))
            shutil.copy2(material_path, dst)
            return True, "image", dest_source_dir
        if ext in VIDEO_EXTS:
            ok, msg = extract_video_frames(
                material_path,
                dest_source_dir,
                interval=max(1, int(video_interval)),
                progress=progress,
            )
            if not ok:
                return False, "video", msg
            return True, "video", dest_source_dir
        return False, "", f"Unsupported file type: {ext}"

    if os.path.isdir(material_path):
        n = 0
        for root, _dirs, files in os.walk(material_path):
            for name in files:
                if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                    continue
                src = os.path.join(root, name)
                # Keep relative structure flat with unique names on collision
                dst_name = name
                dst = os.path.join(dest_source_dir, dst_name)
                if os.path.exists(dst):
                    stem, e = os.path.splitext(name)
                    rel = os.path.relpath(root, material_path).replace(
                        "\\", "_"
                    ).replace("/", "_")
                    dst_name = f"{rel}_{stem}{e}" if rel != "." else f"{stem}_dup{e}"
                    dst = os.path.join(dest_source_dir, dst_name)
                shutil.copy2(src, dst)
                n += 1
        if n <= 0:
            return False, "folder", f"No images found under: {material_path}"
        return True, "folder", dest_source_dir

    return False, "", f"Invalid material: {material_path}"


def extract_video_frames(
    video_path: str,
    out_dir: str,
    *,
    interval: int = 1,
    prefix: str = "frame",
    seq_len: int = 6,
    progress=None,
) -> Tuple[bool, str]:
    """Extract frames with OpenCV (no Qt dialog)."""
    try:
        import cv2
    except Exception as e:
        return False, f"OpenCV not available: {e}"

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, f"Failed to open video: {video_path}"

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % interval == 0:
                name = f"{prefix}_{saved:0{seq_len}d}.jpg"
                path = os.path.join(out_dir, name)
                if not cv2.imwrite(path, frame):
                    return False, f"Failed to write frame: {path}"
                saved += 1
                if progress and total > 0 and saved % 20 == 0:
                    progress(f"Extracted {saved} frames…")
            idx += 1
        if saved <= 0:
            return False, "No frames extracted from video"
        logger.info(f"Extracted {saved} frames from {video_path} → {out_dir}")
        return True, f"Extracted {saved} frames"
    finally:
        cap.release()


def resolve_model_path(weight: WeightEntry, job_dir: str) -> Tuple[bool, str]:
    """Return absolute local model path; copy upload into job weights if needed."""
    if weight.source == "upload" or (
        not weight.remote and os.path.isabs(weight.weight_path)
    ):
        src = weight.weight_path
        if not os.path.isfile(src):
            return False, f"Weight file not found: {src}"
        # If already under job, use it
        weights_dir = os.path.join(job_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        dst = os.path.join(weights_dir, os.path.basename(src))
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        return True, dst
    if weight.remote:
        return False, "Remote weight path cannot be used as local model"
    if not os.path.isfile(weight.weight_path):
        return False, f"Weight file not found: {weight.weight_path}"
    return True, weight.weight_path


def describe_infer_project(project_path: str) -> Dict:
    """Locate result images and YOLO labels for a free-inference job directory.

    Prefer annotated images saved beside ``labels/``. Otherwise use the source
    images so boxes can be drawn from the label files.
    """
    project_path = os.path.abspath(project_path or "")
    images_dir = os.path.join(project_path, "source")
    labels_dir = os.path.join(project_path, "batch_infer", "labels")
    meta_path = os.path.join(project_path, "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if data.get("source_dir"):
                    images_dir = str(data["source_dir"])
                if data.get("labels_dir"):
                    labels_dir = str(data["labels_dir"])
        except (OSError, json.JSONDecodeError):
            pass

    def _images_in(folder: str) -> List[str]:
        if not folder or not os.path.isdir(folder):
            return []
        names = []
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                names.append(path)
        return names

    annotated_dir = os.path.dirname(labels_dir) if labels_dir else ""
    annotated = _images_in(annotated_dir)
    if annotated:
        return {
            "images": annotated,
            "labels_dir": labels_dir,
            "annotated": True,
            "images_dir": annotated_dir,
        }
    return {
        "images": _images_in(images_dir),
        "labels_dir": labels_dir,
        "annotated": False,
        "images_dir": images_dir,
    }


def delete_local_project(path: str) -> Tuple[bool, str]:
    path = os.path.abspath(path or "")
    if not path or not os.path.isdir(path):
        return False, f"Project not found: {path}"
    # Safety: only allow under trainer root
    root = os.path.abspath(get_trainer_root_dir())
    if not path.startswith(root + os.sep) and path != root:
        return False, f"Refusing to delete outside trainer root: {path}"
    try:
        shutil.rmtree(path)
        return True, path
    except Exception as e:
        return False, str(e)


def package_results(
    source_dir: str,
    labels_dir: str,
    dest_dir: str,
) -> Tuple[bool, str]:
    """Copy images + matching label txt into dest_dir/images and dest_dir/labels."""
    source_dir = os.path.abspath(source_dir or "")
    labels_dir = os.path.abspath(labels_dir or "")
    dest_dir = os.path.abspath(dest_dir or "")
    if not os.path.isdir(source_dir):
        return False, f"Source images not found: {source_dir}"
    if not os.path.isdir(labels_dir):
        return False, f"Labels not found: {labels_dir}"
    img_out = os.path.join(dest_dir, "images")
    lab_out = os.path.join(dest_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lab_out, exist_ok=True)
    n_img = 0
    n_lab = 0
    for name in os.listdir(source_dir):
        src = os.path.join(source_dir, name)
        if not os.path.isfile(src):
            continue
        if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
            continue
        shutil.copy2(src, os.path.join(img_out, name))
        n_img += 1
        stem = os.path.splitext(name)[0]
        lab = os.path.join(labels_dir, f"{stem}.txt")
        if os.path.isfile(lab):
            shutil.copy2(lab, os.path.join(lab_out, f"{stem}.txt"))
            n_lab += 1
    return True, f"Copied {n_img} images and {n_lab} labels to {dest_dir}"
