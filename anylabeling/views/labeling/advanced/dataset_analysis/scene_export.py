"""Export random sample images per scene cluster for dataset analysis."""

from __future__ import annotations

import os
import os.path as osp
import random
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SceneImageExportResult:
    clusters_processed: int = 0
    images_copied: int = 0
    errors: List[str] = field(default_factory=list)


def _basename_index(image_list: List[str]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for path in image_list:
        if not path:
            continue
        key = osp.basename(str(path))
        if not key:
            continue
        index.setdefault(key, []).append(str(path))
    return index


def resolve_scene_image_file(
    image_path_field: str,
    image_list: List[str],
) -> Optional[str]:
    """Map cluster imagePath (often basename) to an on-disk image file."""
    field = str(image_path_field or "").strip()
    if not field:
        return None
    norm_field = osp.normpath(field)
    base = osp.basename(norm_field)
    by_name = _basename_index(image_list)
    if base in by_name:
        return by_name[base][0]
    for path in image_list:
        p = str(path)
        if osp.normpath(p) == norm_field:
            return p
        if osp.basename(p) == base:
            return p
    if osp.isfile(field):
        return field
    return None


def _cluster_image_paths(
    cluster: Dict[str, Any],
    image_list: List[str],
) -> List[str]:
    resolved: List[str] = []
    seen: set[str] = set()
    for item in cluster.get("images") or []:
        if not isinstance(item, dict):
            continue
        raw = item.get("imagePath") or ""
        src = resolve_scene_image_file(str(raw), image_list)
        if not src or not osp.isfile(src):
            continue
        key = osp.normpath(src).lower()
        if key in seen:
            continue
        seen.add(key)
        resolved.append(src)
    return resolved


def export_scene_cluster_images(
    export_dir: str,
    scene_clusters: List[Dict[str, Any]],
    image_list: List[str],
    per_cluster_count: int,
    *,
    seed: Optional[int] = None,
) -> SceneImageExportResult:
    """Randomly copy up to *per_cluster_count* images from each scene cluster."""
    result = SceneImageExportResult()
    n = max(1, int(per_cluster_count))
    rng = random.Random(seed)

    try:
        os.makedirs(export_dir, exist_ok=True)
    except OSError as e:
        result.errors.append(str(e))
        return result

    for cluster in scene_clusters:
        if not isinstance(cluster, dict):
            continue
        try:
            cid = int(cluster.get("cluster_id", 0))
        except (TypeError, ValueError):
            cid = 0

        paths = _cluster_image_paths(cluster, image_list)
        if not paths:
            continue

        pick_n = min(n, len(paths))
        picked = rng.sample(paths, pick_n)
        dest_root = osp.join(export_dir, f"cluster_{cid}")
        try:
            os.makedirs(dest_root, exist_ok=True)
        except OSError as e:
            result.errors.append(f"cluster_{cid}: {e}")
            continue

        copied_here = 0
        for src in picked:
            base = osp.basename(src)
            dest = osp.join(dest_root, base)
            if osp.exists(dest):
                stem, ext = osp.splitext(base)
                for i in range(1, 1000):
                    cand = osp.join(dest_root, f"{stem}_{i}{ext}")
                    if not osp.exists(cand):
                        dest = cand
                        break
            try:
                shutil.copy2(src, dest)
                copied_here += 1
            except OSError as e:
                result.errors.append(f"{base}: {e}")

        if copied_here > 0:
            result.clusters_processed += 1
            result.images_copied += copied_here

    return result
