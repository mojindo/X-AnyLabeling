"""Gate2 视差量：从背景单应 H 分解出平移/尺度/旋转，表征"场景变化幅度"。

注意：verify.py 里的 distortion = |H[2,2]-1| 在 RANSAC 归一化后恒为 0，
不能用来表征画面变化。这里用画面中心与四角在单应下的位移/尺度/旋转来量。

H 由 verify.py 拟合时屏蔽人框内点得到（纯背景单应），直接复用，
不再单独跑 SIFT——既省时，又保证 Gate2/Gate3 用的是同一个几何对齐。
"""
from __future__ import annotations

import cv2
import numpy as np

from .verify import VerifyResult


def homography_motion(H: np.ndarray, w: int, h: int) -> tuple[float, float, float]:
    """返回 (中心位移归一化, 尺度变化绝对值, 旋转弧度绝对值)。"""
    pts = np.float32([[w / 2, h / 2], [0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    base = pts.reshape(-1, 2)
    shift = float(np.linalg.norm(mapped[0] - base[0]) / max(min(w, h), 1))
    d0 = np.linalg.norm(base[1:] - base[0], axis=1)
    d1 = np.linalg.norm(mapped[1:] - mapped[0], axis=1)
    scale_change = abs(float(np.median(d1 / np.maximum(d0, 1))) - 1.0)
    v0 = base[1] - base[0]
    v1 = mapped[1] - mapped[0]
    rotation = float(abs(np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])))
    return shift, scale_change, rotation


def view_motion(v: VerifyResult) -> tuple[float, float, float] | None:
    """从验证结果里的背景单应直接算视差；无 H 返回 None。"""
    if v.H is None or v.img_size[0] <= 0:
        return None
    return homography_motion(v.H, v.img_size[0], v.img_size[1])
