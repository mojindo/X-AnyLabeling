"""Batch-rename dataset images and sync label JSON files."""

from __future__ import annotations

import json
import os
import os.path as osp
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from anylabeling.views.labeling.classifier.utils import get_label_file_path
from anylabeling.views.labeling.logger import logger


@dataclass
class DatasetRenameResult:
    renamed_images: int = 0
    renamed_labels: int = 0
    updated_json: int = 0
    skipped_same_name: int = 0
    path_map: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def default_prefix_from_image_paths(image_paths: List[str]) -> str:
    """
    Default prefix = parent folder name of the image directory.

    e.g. ``.../river/base_0_0/image/foo.jpg`` or ``.../base_0_0/images/foo.jpg``
    → ``base_0_0`` (regardless of the last folder name).
    """
    if not image_paths:
        return ""
    image_dir = osp.normpath(osp.dirname(osp.abspath(image_paths[0])))
    parent_dir = osp.dirname(image_dir)
    if parent_dir:
        return osp.basename(parent_dir)
    return osp.basename(image_dir)


def build_rename_plan(
    image_paths: List[str],
    prefix: str,
    *,
    start_index: int = 1,
    num_digits: int = 6,
) -> List[Tuple[str, str]]:
    """Return ordered (old_path, new_path) pairs under each image's directory."""
    prefix = (prefix or "").strip()
    if not prefix:
        raise ValueError("文件名前缀不能为空")

    if start_index < 0:
        raise ValueError("起始序号不能为负数")
    if num_digits < 1 or num_digits > 12:
        raise ValueError("序号位数须在 1~12 之间")

    plan: List[Tuple[str, str]] = []
    seen_new: set[str] = set()
    index = int(start_index)

    for old_path in image_paths:
        old_path = osp.normpath(old_path)
        if not osp.isfile(old_path):
            raise FileNotFoundError(f"图片不存在: {old_path}")
        ext = osp.splitext(old_path)[1]
        if not ext:
            ext = ".jpg"
        new_name = f"{prefix}_{index:0{num_digits}d}{ext}"
        new_path = osp.normpath(osp.join(osp.dirname(old_path), new_name))
        if new_path in seen_new:
            raise ValueError(f"重命名目标重复: {new_name}")
        seen_new.add(new_path)
        plan.append((old_path, new_path))
        index += 1

    return plan


def _read_json(path: str) -> dict:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            continue
    raise ValueError(f"无法读取 JSON: {path}")


def _write_json(path: str, data: dict) -> None:
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _update_label_json_image_path(label_path: str, image_basename: str) -> bool:
    if not osp.isfile(label_path):
        return False
    data = _read_json(label_path)
    data["imagePath"] = image_basename
    _write_json(label_path, data)
    return True


def rename_dataset(
    image_paths: List[str],
    prefix: str,
    output_dir: Optional[str] = None,
    *,
    start_index: int = 1,
    num_digits: int = 6,
) -> DatasetRenameResult:
    """
    Rename images to ``{prefix}_{index:06d}.ext`` and sync JSON label files.
    Uses a two-phase rename to avoid name collisions.
    """
    result = DatasetRenameResult()
    plan = build_rename_plan(
        image_paths,
        prefix,
        start_index=start_index,
        num_digits=num_digits,
    )
    if not plan:
        return result

    old_set = {osp.normpath(o) for o, _ in plan}
    for _old, new in plan:
        if osp.exists(new) and new not in old_set:
            raise FileExistsError(f"目标文件已存在: {osp.basename(new)}")

    to_move = [(o, n) for o, n in plan if osp.normpath(o) != osp.normpath(n)]
    result.skipped_same_name = len(plan) - len(to_move)

    temp_moves: List[Tuple[str, str, str]] = []
    for old, new in to_move:
        token = uuid.uuid4().hex[:10]
        tmp = f"{old}.xal_tmp_{token}"
        temp_moves.append((old, tmp, new))

    try:
        for old, tmp, _new in temp_moves:
            os.rename(old, tmp)
        for _old, tmp, new in temp_moves:
            os.rename(tmp, new)
            result.renamed_images += 1
    except Exception:
        for _old, tmp, new in temp_moves:
            if osp.isfile(tmp) and not osp.isfile(_old):
                try:
                    os.rename(tmp, _old)
                except OSError:
                    pass
            elif osp.isfile(new) and not osp.isfile(_old):
                try:
                    os.rename(new, _old)
                except OSError:
                    pass
        raise

    for old, new in plan:
        result.path_map[old] = new
        old_label = get_label_file_path(old, output_dir)
        new_label = get_label_file_path(new, output_dir)
        image_base = osp.basename(new)

        if osp.isfile(old_label):
            try:
                if osp.normpath(old_label) != osp.normpath(new_label):
                    if osp.isfile(new_label):
                        raise FileExistsError(
                            f"标注文件已存在: {osp.basename(new_label)}"
                        )
                    os.rename(old_label, new_label)
                    result.renamed_labels += 1
                if _update_label_json_image_path(new_label, image_base):
                    result.updated_json += 1
            except Exception as e:
                msg = f"{osp.basename(old_label)}: {e}"
                result.errors.append(msg)
                logger.error("Dataset rename label failed: %s", msg)
        elif osp.isfile(new_label):
            try:
                if _update_label_json_image_path(new_label, image_base):
                    result.updated_json += 1
            except Exception as e:
                msg = f"{osp.basename(new_label)}: {e}"
                result.errors.append(msg)
                logger.error("Dataset rename label update failed: %s", msg)

    return result
