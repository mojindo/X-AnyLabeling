"""L2 几何验证（Gate1）：SIFT + KNN ratio test + RANSAC 单应矩阵。

这是"场景完全匹配"判据：
- 真重复：两图是同一场景的平移/缩放/光照/裁剪变化，单应内点密集
- 不同场景：拟合不出空间一致的单应，内点比例低

关键改动：拟合单应时屏蔽标注框内的 SIFT 点，只用背景点。
理由——画面里的人在移动，人身上特征点会污染 H，使 H 描述"跟着人动"
而非"跟着场景动"，导致后续 H 补偿后的实例真位移失真。
拟合得到的 H 会保存在结果里，供 Gate2（视差）和 Gate3（真位移）复用。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .io_utils import Box, ImageRecord, decode_verify


@dataclass
class VerifyResult:
    a: int
    b: int
    passed: bool
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    overlap: float = 0.0
    distortion: float = 0.0
    reason: str = ""
    # Gate2/Gate3 复用的背景单应矩阵（A 坐标系 -> B 坐标系）与两图 verify 尺寸
    H: np.ndarray | None = None
    img_size: tuple[int, int] = (0, 0)      # A 图 (w, h)
    img_size_b: tuple[int, int] = (0, 0)    # B 图 (w, h)


# SIFT 提取器复用（线程内共享，避免每张图重建）
_EXTRACTOR = None


def _get_sift(nfeatures: int):
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = cv2.SIFT_create(nfeatures=nfeatures)
    return _EXTRACTOR


def _features(rgb: np.ndarray, sift, max_side: int):
    scale = max_side / max(rgb.shape[:2])
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des, gray.shape[:2][::-1]  # (w, h)


def _mask_background(good: list, kp1, kp2, boxes_a: list[Box], boxes_b: list[Box],
                     size1: tuple[int, int], size2: tuple[int, int]) -> list:
    """剔除落在标注框内的匹配点，只保留背景点。

    框坐标是归一化的 [0,1]，乘以对应图尺寸得到像素坐标。
    任一端落在框内即剔除。
    """
    if not boxes_a and not boxes_b:
        return good

    def in_box(pt, boxes, size, pad=0.0):
        x, y = pt
        w, h = size
        nx, ny = x / max(w, 1), y / max(h, 1)
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy
            if (x1 - pad) <= nx <= (x2 + pad) and (y1 - pad) <= ny <= (y2 + pad):
                return True
        return False

    kept = []
    for m in good:
        p1 = kp1[m.queryIdx].pt
        p2 = kp2[m.trainIdx].pt
        if in_box(p1, boxes_a, size1, 0.01) or in_box(p2, boxes_b, size2, 0.01):
            continue
        kept.append(m)
    return kept


def _quad_area(pts: np.ndarray) -> float:
    """四边形面积（shoelace）。"""
    p = pts.reshape(4, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def verify_pair(rec_a: ImageRecord, rec_b: ImageRecord, cfg: Config,
                cache: dict[int, tuple] | None = None) -> VerifyResult:
    vc = cfg.verify
    sift = _get_sift(vc.sift_nfeatures)

    def get(r: ImageRecord):
        if cache is not None and r.index in cache:
            return cache[r.index]
        v = _features(decode_verify(r.image_path, cfg.feature.verify_max_side), sift, cfg.feature.verify_max_side)
        if cache is not None:
            cache[r.index] = v
        return v

    kp1, des1, size1 = get(rec_a)
    kp2, des2, size2 = get(rec_b)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return VerifyResult(rec_a.index, rec_b.index, False, reason="no_features")

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < vc.ratio_test * n.distance]
    if len(good) < vc.min_inliers:
        return VerifyResult(rec_a.index, rec_b.index, False,
                            good_matches=len(good), reason="few_good_matches")

    # 背景点过滤：屏蔽人框内特征点，防止运动目标污染场景单应
    fit_matches = good
    if vc.mask_boxes_when_fitting_h:
        bg = _mask_background(good, kp1, kp2, rec_a.boxes, rec_b.boxes, size1, size2)
        if len(bg) >= vc.min_good_after_mask:
            fit_matches = bg

    src = np.float32([kp1[m.queryIdx].pt for m in fit_matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in fit_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return VerifyResult(rec_a.index, rec_b.index, False,
                            good_matches=len(good), reason="homography_failed")

    inliers = int(mask.sum())
    ratio = inliers / max(len(fit_matches), 1)
    h_norm = float(H[2, 2])
    distortion = abs(h_norm - 1.0) / max(abs(h_norm), 1e-6)

    # 重叠率：把 A 的四个角映射到 B 坐标系，求与 B 画面的交面积
    w1, h1 = size1
    w2, h2 = size2
    corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    clip = np.clip(mapped, [0, 0], [w2, h2])
    inter_area = _quad_area(clip)
    smaller_area = min(w1 * h1, w2 * h2)
    overlap = inter_area / max(smaller_area, 1.0)
    # 对称检查：反向单应，防止 A 大部分在 B 外仍通过
    try:
        Hinv = np.linalg.inv(H)
        back = cv2.perspectiveTransform(np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        clip_back = np.clip(back, [0, 0], [w1, h1])
        overlap = min(overlap, _quad_area(clip_back) / max(smaller_area, 1.0))
    except Exception:
        pass

    passed = (inliers >= vc.min_inliers
              and ratio >= vc.min_inlier_ratio
              and overlap >= vc.min_overlap
              and distortion <= vc.max_homography_distortion)
    reason = "ok" if passed else (
        "few_inliers" if inliers < vc.min_inliers else
        "low_inlier_ratio" if ratio < vc.min_inlier_ratio else
        "low_overlap" if overlap < vc.min_overlap else
        "high_distortion"
    )
    return VerifyResult(rec_a.index, rec_b.index, passed, len(good), inliers,
                        round(ratio, 3), round(overlap, 3), round(distortion, 3), reason,
                        H=H, img_size=size1, img_size_b=size2)
