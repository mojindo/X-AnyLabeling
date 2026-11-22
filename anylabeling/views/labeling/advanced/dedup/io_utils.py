"""图片 / labelme JSON 的扫描、配对与按需解码。

关键坑：本数据集 JSON 里 imageWidth/imageHeight 是标注时的画布尺寸（如 1280x720），
实际图片文件是 4000x2250 等更高分辨率。框坐标属于 JSON 画布坐标系，
必须按 JSON 自身宽高归一化，不能用图片实际尺寸。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass
class Box:
    label: str
    score: float
    # 归一化坐标 [x1, y1, x2, y2]，范围 0~1，来自 JSON 画布宽高
    xyxy: tuple[float, float, float, float]


@dataclass
class ImageRecord:
    index: int
    image_path: str
    json_path: str | None
    image_md5: str = ""
    json_w: int = 0
    json_h: int = 0
    image_w: int = 0
    image_h: int = 0
    boxes: list[Box] = field(default_factory=list)


def _stem(path: str) -> str:
    return Path(path).stem.lower()


def scan_dataset(data_dir: str, image_exts: tuple[str, ...], seed: int = 42) -> list[ImageRecord]:
    """扫描目录，把图片与同名 JSON 配对，返回按文件名稳定排序的记录。"""
    data_dir = str(Path(data_dir))
    exts = {e.lower() for e in image_exts}
    images: dict[str, str] = {}
    jsons: dict[str, str] = {}
    for name in os.listdir(data_dir):
        p = os.path.join(data_dir, name)
        if not os.path.isfile(p):
            continue
        ext = Path(name).suffix.lower()
        if ext in exts:
            images[_stem(name)] = p
        elif ext == ".json":
            jsons[_stem(name)] = p

    stems = sorted(set(images) | set(jsons))
    recs: list[ImageRecord] = []
    for i, stem in enumerate(stems):
        img = images.get(stem)
        js = jsons.get(stem)
        if img is None and js is None:
            continue
        if img is None:  # 只有 JSON 没图片，跳过（无法参与图像判重）
            continue
        rec = ImageRecord(index=i, image_path=img, json_path=js)
        if js is not None:
            _load_labelme(rec)
        rec.image_md5 = md5_file(img)
        recs.append(rec)
    return recs


def _load_labelme(rec: ImageRecord) -> None:
    try:
        with open(rec.json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return
    w = float(d.get("imageWidth") or 0)
    h = float(d.get("imageHeight") or 0)
    rec.json_w, rec.json_h = int(w), int(h)
    if w <= 0 or h <= 0:
        return
    for sh in d.get("shapes", []):
        pts = sh.get("points") or []
        if len(pts) < 2:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x1, x2 = min(xs) / w, max(xs) / w
        y1, y2 = min(ys) / h, max(ys) / h
        x1, x2 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))
        y1, y2 = max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        rec.boxes.append(Box(
            label=str(sh.get("label", "")),
            score=float(sh.get("score", 1.0) or 1.0),
            xyxy=(x1, y1, x2, y2),
        ))


def records_from_paths(image_paths: list[str], label_for) -> list[ImageRecord]:
    """Build records from the images currently opened in the app.

    ``label_for(image_path)`` returns the annotation JSON path or None.
    """
    recs: list[ImageRecord] = []
    for i, img in enumerate(image_paths):
        if not img or not os.path.isfile(img):
            continue
        js = label_for(img)
        if js is not None and not os.path.isfile(js):
            js = None
        rec = ImageRecord(index=len(recs), image_path=img, json_path=js)
        if js is not None:
            _load_labelme(rec)
        rec.image_md5 = md5_file(img)
        recs.append(rec)
    return recs


def md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def decode_thumb(path: str, max_side: int = 256) -> np.ndarray:
    """解码并返回 RGB uint8 缩略图。JPEG 用 draft() 先降采样，大图速度提升明显。"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        if max(w, h) > max_side * 2:
            try:
                im.draft("RGB", (max_side, max_side))
            except Exception:
                pass
        scale = max_side / max(im.size)
        if scale < 1.0:
            nw = max(1, int(round(im.width * scale)))
            nh = max(1, int(round(im.height * scale)))
            im = im.resize((nw, nh), Image.Resampling.BILINEAR)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def decode_verify(path: str, max_side: int = 1024) -> np.ndarray:
    """L2 几何验证用的较大图，按需解码，不常驻内存。"""
    return decode_thumb(path, max_side=max_side)


def fullres_pixel_md5(path: str) -> str:
    """全分辨率解码像素 MD5：解码后 RGB 原始像素逐字节一致才算精确重复。

    与字节 MD5 的区别：改元数据/重容器但像素不变时，字节 MD5 变、本值不变。
    与缩略 MD5 的区别：本值对分辨率敏感（缩略相同不代表全分辨率相同）。
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return hashlib.md5(arr.tobytes()).hexdigest()
