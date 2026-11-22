"""L0/L1 特征提取：感知哈希、框空间签名、质量指标。

全部在 256 长边缩略图上计算，与图片实际分辨率无关；
质量指标公式与 xai_dataset_analysis.metrics 对齐，便于复用已有阈值。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Config
from .io_utils import ImageRecord


@dataclass
class Features:
    index: int
    pixel_hash: str = ""                 # L0：解码后像素哈希（挡重编码）
    phash: int = 0                       # L1：8x8 DCT 感知哈希
    dhash: int = 0                       # L1：8x7 差分哈希
    box_sig: np.ndarray | None = None    # L1：person 框空间金字塔签名
    quality: dict[str, float] = field(default_factory=dict)


def phash(gray: np.ndarray, hash_bits: int = 64) -> int:
    """8x8 DCT 低频哈希，返回位整数。"""
    g = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(g)[:8, :8]
    med = float(np.median(dct[1:]))
    bits = (dct > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def dhash(gray: np.ndarray) -> int:
    """8x7 相邻像素差分哈希，64 bit。"""
    g = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def pixel_hash(rgb: np.ndarray) -> str:
    """解码像素哈希：固定尺寸灰度量化后 MD5，挡住改元数据/重编码的同图。"""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)
    return hashlib.md5(g.tobytes()).hexdigest()


def box_signature(rec: ImageRecord, cfg: Config) -> np.ndarray | None:
    """检测框的 3x3 空间金字塔签名，支持多标签 one-hot。

    one-hot（box_multilabel=True）：每个类别各占 grid*grid 个格位，
    向量维度 = 9*C + 3。类别集合由 cfg.feature.box_classes 全局固定，
    保证每张图维度一致；"1人+1车"与"2人"不再混淆。
    非 one-hot（旧行为）：所有类混投同一套 9 格，维度 9+3=12。

    末尾 3 维为跨类全局量：总框数、平均宽高比、平均框面积。
    L2 归一化后用余弦相似度召回。
    """
    grid = cfg.feature.box_grid
    filt = cfg.feature.box_label_filter
    boxes = [b for b in rec.boxes if (b.label in filt if filt else True)]
    boxes = boxes[:cfg.feature.box_max_per_image]
    if not boxes:
        return None

    if cfg.feature.box_multilabel:
        classes = list(cfg.feature.box_classes)
        c2i = {c: i for i, c in enumerate(classes)}
        cells = np.zeros(len(classes) * grid * grid, dtype=np.float32)
        for b in boxes:
            ci = c2i.get(b.label)
            if ci is None:
                continue
            x1, y1, x2, y2 = b.xyxy
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            gx = min(grid - 1, max(0, int(cx * grid)))
            gy = min(grid - 1, max(0, int(cy * grid)))
            cells[ci * grid * grid + gy * grid + gx] += float(b.score)
    else:
        cells = np.zeros(grid * grid, dtype=np.float32)
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            gx = min(grid - 1, max(0, int(cx * grid)))
            gy = min(grid - 1, max(0, int(cy * grid)))
            cells[gy * grid + gx] += float(b.score)

    areas, ratios = [], []
    for b in boxes:
        w, h = b.xyxy[2] - b.xyxy[0], b.xyxy[3] - b.xyxy[1]
        areas.append(w * h)
        ratios.append(w / max(h, 1e-6))
    extra = np.array([
        len(boxes) / 16.0,
        float(np.mean(ratios)) / 3.0,
        float(np.mean(areas)) / 0.25,
    ], dtype=np.float32)
    vec = np.concatenate([cells, extra])
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else None


def _calc_quality(rgb: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    f = gray.astype(np.float32)

    lap = cv2.Laplacian(f, cv2.CV_32F)
    sharpness = float(np.log1p(lap.var()))

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    brightness = float(lab[:, :, 0].mean())

    contrast = float(np.sqrt(np.mean(f * f)))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = float(hsv[:, :, 1].mean())
    value = float(hsv[:, :, 2].mean())

    # 噪声：中值滤波残差的 MAD
    med = cv2.medianBlur(gray, 3).astype(np.float32)
    residual = f - med
    noise = float(np.median(np.abs(residual - np.median(residual))))

    # artifact 启发式：JPEG 块效应强度（8x8 块边界梯度 vs 块内梯度）
    h, w = gray.shape[:2]
    h2 = (h // 8) * 8
    w2 = (w // 8) * 8
    if h2 >= 16 and w2 >= 16:
        crop = f[:h2, :w2].reshape(h2 // 8, 8, w2 // 8, 8)
        border = np.abs(crop[:, 0, :, :]).mean() + np.abs(crop[:, :, :, 0]).mean()
        inner = np.abs(np.diff(crop, axis=1)).mean() + np.abs(np.diff(crop, axis=3)).mean()
        artifact = float(border / max(inner, 1e-6))
    else:
        artifact = 0.0

    # brisque：opencv 标准包通常不含 quality 模块，默认关闭，缺失返回 0
    try:
        brisque = float(cv2.quality.QualityBRISQUE_create().compute([rgb])[0])
    except Exception:
        brisque = 0.0

    return {
        "sharpness_laplacian_var_log1p": sharpness,
        "brightness_l_mean": brightness,
        "contrast_rms_gray": contrast,
        "saturation_mean": saturation,
        "value_mean": value,
        "noise_mad_residual": noise,
        "artifact_heuristic_score": artifact,
        "brisque": brisque,
    }


def extract_one(rec: ImageRecord, cfg: Config, thumb: np.ndarray) -> Features:
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    return Features(
        index=rec.index,
        pixel_hash=pixel_hash(thumb),
        phash=phash(gray),
        dhash=dhash(gray),
        box_sig=box_signature(rec, cfg),
        quality=_calc_quality(thumb),
    )


def extract_all(recs: list[ImageRecord], cfg: Config, thumbs: list[np.ndarray]) -> list[Features]:
    return [extract_one(rec, cfg, thumb) for rec, thumb in zip(recs, thumbs)]
