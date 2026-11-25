"""Gate3 实例层判定：在场景单应 H 对齐后判断实例是否同一、是否几乎没变。

判据（用户定义）：
1. 按类计数一致 —— 个数不等直接判多样性
2. 一对一配对 —— 匈牙利最优匹配，代价 = H 补偿后真位移 + 外观距离
3. 真位移小 —— 用 H 把 A 框中心投影到 B 坐标系，扣除场景运动后的残差
4. 尺度一致 —— 框面积比
5. 姿态一致 —— 框宽高比（姿态代理）
6. 外观一致（可选）—— 框内 SIFT 内点数 + 分块颜色直方图（ReID 纯 CPU 替代）

关键：真位移不是原始框中心位移。若相机摇动、人跟着平移，H 补偿后真位移≈0，
判重复；若相机不动、人自己走动，补偿后位移大，判多样性保留。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Config
from .io_utils import Box, ImageRecord
from .verify import VerifyResult


@dataclass
class InstanceResult:
    passed: bool
    reason: str = ""
    count_a: dict[str, int] = field(default_factory=dict)
    count_b: dict[str, int] = field(default_factory=dict)
    matched: int = 0
    max_true_shift: float = 0.0
    max_scale_diff: float = 0.0
    max_pose_diff: float = 0.0
    mean_hist_sim: float = 0.0
    min_sift_good: int = 0


def _filtered_boxes(rec: ImageRecord, cfg: Config) -> list[Box]:
    ic = cfg.strict.instance
    filt = cfg.feature.box_label_filter
    return [b for b in rec.boxes
            if b.score >= ic.min_box_score and (b.label in filt if filt else True)]


def _class_counts(boxes: list[Box]) -> dict[str, int]:
    return dict(Counter(b.label for b in boxes))


def _project_center(box: Box, H: np.ndarray, w: int, h: int) -> tuple[float, float]:
    """把归一化框中心经 H 投影到 B 的像素坐标。"""
    cx = (box.xyxy[0] + box.xyxy[2]) / 2.0 * w
    cy = (box.xyxy[1] + box.xyxy[3]) / 2.0 * h
    p = H @ np.array([cx, cy, 1.0], dtype=np.float64)
    p = p / max(p[2], 1e-9)
    return float(p[0]), float(p[1])


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """最小化总代价的一对一匹配（Jonker-Volgenant 风格，O(n^3)）。

    输入 cost[n,m]（n<=m），返回 [(row, col), ...]。框数很少（通常<10），
    纯 numpy 实现足够，避免引入 scipy 依赖。
    """
    n, m = cost.shape
    if n == 0:
        return []
    if n > m:
        rows, cols = _hungarian(cost.T)
        return [(c, r) for r, c in rows]
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    result = []
    for j in range(1, m + 1):
        if p[j] > 0:
            result.append((int(p[j]) - 1, j - 1))
    return result


def _crop_features(rgb: np.ndarray, box: Box) -> tuple[np.ndarray | None, np.ndarray | None]:
    """从 verify 尺度图上按归一化框裁剪，返回 (灰度块, 直方图特征)。"""
    h, w = rgb.shape[:2]
    x1 = int(max(0, box.xyxy[0] * w))
    y1 = int(max(0, box.xyxy[1] * h))
    x2 = int(min(w, box.xyxy[2] * w))
    y2 = int(min(h, box.xyxy[3] * h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None, None
    crop = rgb[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    # 分块颜色直方图：3x3 块，每块 HSV 的 H(16bin)+S(8bin)，拼成向量
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    feats = []
    for gy in range(3):
        for gx in range(3):
            yy1, yy2 = gy * crop.shape[0] // 3, (gy + 1) * crop.shape[0] // 3
            xx1, xx2 = gx * crop.shape[1] // 3, (gx + 1) * crop.shape[1] // 3
            cell = hsv[yy1:yy2, xx1:xx2]
            if cell.size == 0:
                feats.extend([0.0] * 24)
                continue
            hh = np.histogram(cell[:, :, 0], bins=16, range=(0, 180), density=True)[0]
            ss = np.histogram(cell[:, :, 1], bins=8, range=(0, 256), density=True)[0]
            feats.extend(hh.tolist())
            feats.extend(ss.tolist())
    vec = np.asarray(feats, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return gray, vec / norm if norm > 0 else vec


def judge_instance(rec_a: ImageRecord, rec_b: ImageRecord, v: VerifyResult,
                   rgb_a: np.ndarray, rgb_b: np.ndarray, cfg: Config) -> InstanceResult:
    """判定一对（已过 G1/G2）的实例层一致性。"""
    cfg_i = cfg.strict.instance
    ba = _filtered_boxes(rec_a, cfg)
    bb = _filtered_boxes(rec_b, cfg)
    ca, cb = _class_counts(ba), _class_counts(bb)
    res = InstanceResult(False, count_a=ca, count_b=cb)

    if not ba and not bb:
        # 两边都无参与实例判定的框：实例层不拦截，交给场景/画质门控。
        return InstanceResult(True, reason="no_boxes_both_pass", count_a=ca, count_b=cb)

    # 一边有框、一边没有：不做实例“直通”，与双有框一样走下面判定。
    if not ba or not bb:
        res.reason = "instance_presence_mismatch"
        return res

    # 按类计数一致
    if cfg_i.require_count_equal and ca != cb:
        res.reason = "count_mismatch"
        return res

    H = v.H
    if H is None:
        res.reason = "no_homography"
        return res
    wa, ha = v.img_size
    wb, hb = v.img_size_b

    # 按类分组做一对一配对（不同类不互相配）
    total_true_shift = 0.0
    total_scale = 0.0
    total_pose = 0.0
    total_hist = 0.0
    total_sift = 0
    n_match = 0
    max_true_shift = max_scale = max_pose = 0.0

    for label in ca:
        ga = [b for b in ba if b.label == label]
        gb = [b for b in bb if b.label == label]
        if not gb:
            continue

        # 外观特征（可选）：预计算每框特征与 descriptor
        desc_a: list = [None] * len(ga)
        desc_b: list = [None] * len(gb)
        hist_a: list = [None] * len(ga)
        hist_b: list = [None] * len(gb)
        if cfg_i.use_appearance:
            sift = cv2.SIFT_create(nfeatures=256)
            for i, b in enumerate(ga):
                gray, hv = _crop_features(rgb_a, b)
                hist_a[i] = hv
                if gray is not None:
                    _, desc_a[i] = sift.detectAndCompute(gray, None)
            for j, b in enumerate(gb):
                gray, hv = _crop_features(rgb_b, b)
                hist_b[j] = hv
                if gray is not None:
                    _, desc_b[j] = sift.detectAndCompute(gray, None)

        # 代价矩阵：真位移为主，外观距离为辅
        cost = np.zeros((len(ga), len(gb)), dtype=np.float64)
        for i, a in enumerate(ga):
            pa = _project_center(a, H, wa, ha)
            for j, b in enumerate(gb):
                pb = ((b.xyxy[0] + b.xyxy[2]) / 2.0 * wb,
                      (b.xyxy[1] + b.xyxy[3]) / 2.0 * hb)
                shift = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]) / max(min(wb, hb), 1))
                c = shift
                if cfg_i.use_appearance and hist_a[i] is not None and hist_b[j] is not None:
                    sim = float(np.dot(hist_a[i], hist_b[j]))
                    c += 0.5 * (1.0 - sim)
                cost[i, j] = c

        pairs = _hungarian(cost)
        for i, j in pairs:
            a, b = ga[i], gb[j]
            pa = _project_center(a, H, wa, ha)
            pb = ((b.xyxy[0] + b.xyxy[2]) / 2.0 * wb,
                  (b.xyxy[1] + b.xyxy[3]) / 2.0 * hb)
            shift = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]) / max(min(wb, hb), 1))
            area_a = (a.xyxy[2] - a.xyxy[0]) * (a.xyxy[3] - a.xyxy[1])
            area_b = (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1])
            scale = abs(float(np.log(max(area_a, 1e-9) / max(area_b, 1e-9))))
            ar_a = (a.xyxy[2] - a.xyxy[0]) / max(a.xyxy[3] - a.xyxy[1], 1e-9)
            ar_b = (b.xyxy[2] - b.xyxy[0]) / max(b.xyxy[3] - b.xyxy[1], 1e-9)
            pose = abs(float(np.log(max(ar_a, 1e-9) / max(ar_b, 1e-9))))

            max_true_shift = max(max_true_shift, shift)
            max_scale = max(max_scale, scale)
            max_pose = max(max_pose, pose)
            total_true_shift += shift
            total_scale += scale
            total_pose += pose
            n_match += 1

            if cfg_i.use_appearance:
                if hist_a[i] is not None and hist_b[j] is not None:
                    total_hist += float(np.dot(hist_a[i], hist_b[j]))
                if desc_a[i] is not None and desc_b[j] is not None:
                    bf = cv2.BFMatcher(cv2.NORM_L2)
                    knn = bf.knnMatch(desc_a[i], desc_b[j], k=2)
                    good = [m for m, n in knn if m.distance < 0.72 * n.distance]
                    total_sift += len(good)

    res.matched = n_match
    res.max_true_shift = round(max_true_shift, 4)
    res.max_scale_diff = round(max_scale, 4)
    res.max_pose_diff = round(max_pose, 4)
    res.mean_hist_sim = round(total_hist / max(n_match, 1), 4)
    res.min_sift_good = total_sift

    if n_match == 0:
        res.reason = "no_match"
        return res

    if max_true_shift > cfg_i.max_true_shift:
        res.reason = "true_shift_large"
        return res
    if max_scale > cfg_i.max_scale_diff:
        res.reason = "scale_diff_large"
        return res
    if max_pose > cfg_i.max_pose_diff:
        res.reason = "pose_diff_large"
        return res
    if cfg_i.use_appearance and cfg_i.appearance_is_hard:
        if total_sift < cfg_i.min_sift_good:
            res.reason = "sift_appearance_mismatch"
            return res
        if total_hist / max(n_match, 1) < cfg_i.min_hist_sim:
            res.reason = "hist_appearance_mismatch"
            return res

    res.passed = True
    res.reason = "ok"
    return res
