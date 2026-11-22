"""Image augmentation. One source image receives one method."""

from __future__ import annotations

import json
import os
import os.path as osp
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class AugmentStep:
    op_id: str
    enabled: bool
    p: float
    params: Dict[str, Any] = field(default_factory=dict)


def assign_one_method(index: int, enabled: Sequence[AugmentStep]) -> Optional[AugmentStep]:
    """Round-robin: image i uses exactly one enabled method, never a stack."""
    if not enabled:
        return None
    return enabled[int(index) % len(enabled)]


def _odd(value: int, lo: int = 3, hi: int = 101) -> int:
    n = max(lo, min(hi, int(value)))
    if n % 2 == 0:
        n += 1 if n < hi else -1
    return n


def _imread_bgr(path: str) -> Optional[np.ndarray]:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path: str, bgr: np.ndarray) -> None:
    ext = osp.splitext(path)[1].lower() or ".jpg"
    ok, enc = cv2.imencode(ext, bgr)
    if not ok:
        ok, enc = cv2.imencode(".jpg", bgr)
        ext = ".jpg"
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=ext, dir=osp.dirname(path) or ".")
    try:
        os.close(fd)
        enc.tofile(tmp)
        os.replace(tmp, path)
    finally:
        if osp.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _warp_shapes(obj: dict, mapper: Callable[[np.ndarray], np.ndarray], width: int, height: int) -> None:
    """Move annotation points with a geometric augmentation. Rectangles stay axis-aligned."""
    shapes = obj.get("shapes")
    if not isinstance(shapes, list):
        return
    kept = []
    x_max = max(0, int(width) - 1)
    y_max = max(0, int(height) - 1)
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        points = shape.get("points")
        if not isinstance(points, list) or not points:
            kept.append(shape)
            continue
        arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        kind = str(shape.get("shape_type") or "polygon")
        if kind == "rectangle" and len(arr) >= 2:
            x1, y1 = float(arr[0, 0]), float(arr[0, 1])
            x2, y2 = float(arr[1, 0]), float(arr[1, 1])
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
            warped = mapper(corners)
            arr = np.array(
                [[float(warped[:, 0].min()), float(warped[:, 1].min())], [float(warped[:, 0].max()), float(warped[:, 1].max())]],
                dtype=np.float64,
            )
        else:
            arr = np.asarray(mapper(arr), dtype=np.float64).reshape(-1, 2)
        arr[:, 0] = np.clip(arr[:, 0], 0, x_max)
        arr[:, 1] = np.clip(arr[:, 1], 0, y_max)
        if kind == "rectangle" and (abs(arr[1, 0] - arr[0, 0]) < 1 or abs(arr[1, 1] - arr[0, 1]) < 1):
            continue
        shape["points"] = [[float(x), float(y)] for x, y in arr]
        kept.append(shape)
    obj["shapes"] = kept


def _write_label(
    src_json: Optional[str],
    dst_image: str,
    width: int,
    height: int,
    mapper: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> None:
    if not src_json or not osp.isfile(src_json):
        return
    try:
        with open(src_json, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(obj, dict):
        return
    obj["imagePath"] = osp.basename(dst_image)
    obj["imageData"] = None
    obj["imageWidth"] = int(width)
    obj["imageHeight"] = int(height)
    if mapper is not None:
        _warp_shapes(obj, mapper, width, height)
    dst_json = osp.splitext(dst_image)[0] + ".json"
    with open(dst_json, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def _range_float(rng: np.random.Generator, lo: float, hi: float) -> float:
    lo_f, hi_f = float(lo), float(hi)
    if lo_f > hi_f:
        lo_f, hi_f = hi_f, lo_f
    if hi_f <= lo_f:
        return lo_f
    return float(rng.uniform(lo_f, hi_f))


def _range_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    lo_i, hi_i = int(lo), int(hi)
    if lo_i > hi_i:
        lo_i, hi_i = hi_i, lo_i
    return int(rng.integers(lo_i, hi_i + 1))


def _motion_kernel(kernel_size: int, angle_deg: float) -> np.ndarray:
    """Symmetric motion streak. 0 degrees is horizontal, 90 is vertical."""
    size = _odd(kernel_size, lo=3, hi=51)
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    angle = np.deg2rad(float(angle_deg))
    dx = float(np.cos(angle))
    dy = float(np.sin(angle))
    half = size // 2
    samples = np.linspace(-half, half, size)
    xs = np.clip(np.rint(center + dx * samples), 0, size - 1).astype(np.int32)
    ys = np.clip(np.rint(center + dy * samples), 0, size - 1).astype(np.int32)
    kernel[ys, xs] = 1.0
    total = float(kernel.sum())
    if total <= 0:
        kernel[center, :] = 1.0
        total = float(kernel.sum())
    kernel /= total
    return kernel


def _lut_gamma(bgr: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(0.05, float(gamma))
    table = np.clip((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(bgr, table)


def _clahe_bgr(bgr: np.ndarray, clip_limit: float, tile: int) -> np.ndarray:
    tile = max(1, int(tile))
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clip_limit)), tileGridSize=(tile, tile))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _iso_noise(bgr: np.ndarray, color_shift: float, intensity: float, rng: np.random.Generator) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    _, stddev = cv2.meanStdDev(hls)
    lam = max(float(stddev[1, 0]) * float(intensity), 1e-6)
    luminance_noise = rng.poisson(lam, size=hls.shape[:2]).astype(np.float32)
    color_noise = rng.normal(0.0, abs(float(color_shift)) * float(intensity), size=hls.shape[:2]).astype(np.float32)
    hls = hls.copy()
    hls[..., 0] = np.mod(hls[..., 0] + color_noise * 360.0, 360.0)
    hls[..., 1] = np.clip(hls[..., 1] + luminance_noise * float(intensity) * (1.0 - hls[..., 1]), 0.0, 1.0)
    out = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _rgb_shift(bgr: np.ndarray, r_shift: int, g_shift: int, b_shift: int) -> np.ndarray:
    out = bgr.astype(np.int16)
    out[:, :, 0] = np.clip(out[:, :, 0] + int(b_shift), 0, 255)
    out[:, :, 1] = np.clip(out[:, :, 1] + int(g_shift), 0, 255)
    out[:, :, 2] = np.clip(out[:, :, 2] + int(r_shift), 0, 255)
    return out.astype(np.uint8)


def _coarse_dropout(bgr: np.ndarray, holes: Sequence[Tuple[int, int, int, int]], fill: int) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    value = int(np.clip(fill, 0, 255))
    for x1, y1, x2, y2 in holes:
        xa, xb = sorted((int(np.clip(x1, 0, w)), int(np.clip(x2, 0, w))))
        ya, yb = sorted((int(np.clip(y1, 0, h)), int(np.clip(y2, 0, h))))
        if xb > xa and yb > ya:
            out[ya:yb, xa:xb] = value
    return out


def _inverse_remap_mapper(map_x: np.ndarray, map_y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """map_* stores the source coordinate sampled by each output pixel."""
    height, width = map_x.shape[:2]
    step = 1 if max(height, width) <= 640 else 4
    ys = np.arange(0, height, step)
    xs = np.arange(0, width, step)
    sample_x = map_x[np.ix_(ys, xs)].ravel()
    sample_y = map_y[np.ix_(ys, xs)].ravel()
    out_y, out_x = np.meshgrid(ys, xs, indexing="ij")
    out_x = out_x.ravel().astype(np.float64)
    out_y = out_y.ravel().astype(np.float64)

    def mapper(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        moved = np.empty_like(pts)
        for index, (px, py) in enumerate(pts):
            dist = (sample_x - px) ** 2 + (sample_y - py) ** 2
            nearest = int(np.argmin(dist))
            moved[index, 0] = out_x[nearest]
            moved[index, 1] = out_y[nearest]
        return moved

    return mapper


def _separable_inverse_mapper(xx: np.ndarray, yy: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def mapper(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        out = pts.copy()
        x_order = np.argsort(xx)
        y_order = np.argsort(yy)
        out[:, 0] = np.interp(pts[:, 0], xx[x_order], np.arange(len(xx), dtype=np.float64)[x_order])
        out[:, 1] = np.interp(pts[:, 1], yy[y_order], np.arange(len(yy), dtype=np.float64)[y_order])
        return out

    return mapper


def _grid_axes(length: int, num_steps: int, steps: Sequence[float]) -> np.ndarray:
    step = max(1, int(length) // int(num_steps))
    axis = np.zeros(int(length), np.float32)
    prev = 0.0
    for idx, scale in enumerate(steps):
        start = min(int(length), idx * step)
        end = min(int(length), start + step)
        if end <= start:
            break
        cur = prev + step * float(scale)
        if idx == len(steps) - 1:
            cur = float(length)
        axis[start:end] = np.linspace(prev, cur, end - start, dtype=np.float32)
        prev = cur
    if length > 1:
        last = int(np.flatnonzero(axis).max()) if np.any(axis) else 0
        if last < length - 1:
            axis[last:] = np.linspace(float(axis[last]), float(length), length - last, dtype=np.float32)
    return axis


def _normalize_steps(length: int, num_steps: int, steps: List[float]) -> List[float]:
    step = max(1, length // num_steps)
    last = min(length, (num_steps + 1) * step) - num_steps * step
    if step > 0 and steps:
        steps[-1] *= last / step
    total = float(sum(steps)) or 1.0
    scale = (length / float(step)) / total
    return [float(item) * scale for item in steps]


def _grid_distortion_maps(
    height: int,
    width: int,
    num_steps: int,
    distort_limit: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps_n = max(2, int(num_steps))
    limit = float(np.clip(distort_limit, 0.0, 0.9))
    x_steps = _normalize_steps(width, steps_n, [1.0 + _range_float(rng, -limit, limit) for _ in range(steps_n + 1)])
    y_steps = _normalize_steps(height, steps_n, [1.0 + _range_float(rng, -limit, limit) for _ in range(steps_n + 1)])
    xx = _grid_axes(width, steps_n, x_steps)
    yy = _grid_axes(height, steps_n, y_steps)
    map_x, map_y = np.meshgrid(xx, yy)
    return map_x.astype(np.float32), map_y.astype(np.float32), xx, yy


def _optical_maps(
    height: int,
    width: int,
    k: float,
    shift_x: float,
    shift_y: float,
) -> Tuple[np.ndarray, np.ndarray]:
    center_x = width * 0.5 + float(shift_x)
    center_y = height * 0.5 + float(shift_y)
    camera = np.array([[width, 0, center_x], [0, height, center_y], [0, 0, 1]], dtype=np.float32)
    distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)
    map_x, map_y = cv2.initUndistortRectifyMap(camera, distortion, None, None, (width, height), cv2.CV_32FC1)
    return map_x, map_y


def _elastic_maps(
    height: int,
    width: int,
    alpha: float,
    sigma: float,
    alpha_affine: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    affine = None
    if float(alpha_affine) > 0 and min(height, width) > 8:
        center = np.array([width, height], dtype=np.float32) * 0.5
        square = float(min(height, width) // 3)
        pts1 = np.array(
            [center + square, [center[0] + square, center[1] - square], center - square],
            dtype=np.float32,
        )
        pts2 = pts1 + rng.uniform(-float(alpha_affine), float(alpha_affine), size=pts1.shape).astype(np.float32)
        affine = cv2.getAffineTransform(pts1, pts2)
    noise_x = rng.standard_normal((height, width)).astype(np.float32)
    noise_y = rng.standard_normal((height, width)).astype(np.float32)
    peak = max(float(np.max(np.abs(noise_x))), float(np.max(np.abs(noise_y))), 1e-6)
    noise_x /= peak
    noise_y /= peak
    sigma_v = max(0.1, float(sigma))
    dx = cv2.GaussianBlur(noise_x, (0, 0), sigmaX=sigma_v) * float(alpha)
    dy = cv2.GaussianBlur(noise_y, (0, 0), sigmaX=sigma_v) * float(alpha)
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return xs + dx, ys + dy, affine


def _compose_mapper(
    affine: Optional[np.ndarray],
    pixel_mapper: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    def mapper(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if affine is not None:
            try:
                inverse = cv2.invertAffineTransform(affine)
            except cv2.error:
                inverse = None
            if inverse is not None:
                moved = cv2.transform(pts.reshape(-1, 1, 2).astype(np.float32), inverse).reshape(-1, 2)
                pts = moved.astype(np.float64)
        return pixel_mapper(pts)

    return mapper


def _apply_one(
    bgr: np.ndarray,
    step: AugmentStep,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[Callable[[np.ndarray], np.ndarray]]]:
    op = step.op_id
    p = step.params or {}
    if op == "random_fog":
        lo = float(p.get("fog_coef_lo", 0.3))
        hi = float(p.get("fog_coef_hi", 1.0))
        alpha = float(p.get("alpha_coef", 0.08))
        coef = float(rng.uniform(min(lo, hi), max(lo, hi))) * alpha
        return cv2.addWeighted(bgr, 1.0 - coef, np.full_like(bgr, 255), coef, 0), None
    if op == "random_rain":
        out = bgr.copy()
        h, w = out.shape[:2]
        length = max(1, int(p.get("drop_length", 20)))
        width = max(1, int(p.get("drop_width", 1)))
        slant_lo = int(p.get("slant_lower", -10))
        slant_hi = int(p.get("slant_upper", 10))
        slant = int(rng.integers(min(slant_lo, slant_hi), max(slant_lo, slant_hi) + 1))
        count = max(20, (h * w) // 8000)
        for _ in range(count):
            x1 = int(rng.integers(0, w))
            y1 = int(rng.integers(0, h))
            x2 = int(np.clip(x1 + slant, 0, w - 1))
            y2 = int(np.clip(y1 + length, 0, h - 1))
            cv2.line(out, (x1, y1), (x2, y2), (230, 230, 230), width)
        blur = max(0, int(p.get("blur_value", 3)))
        if blur > 0:
            k = _odd(blur)
            out = cv2.GaussianBlur(out, (k, k), 0)
        bright = float(p.get("brightness_coefficient", 0.9))
        return cv2.convertScaleAbs(out, alpha=bright, beta=0), None
    if op == "random_snow":
        lo = float(p.get("snow_point_lo", 0.1))
        hi = float(p.get("snow_point_hi", 0.3))
        thresh = float(rng.uniform(min(lo, hi), max(lo, hi)))
        mask = rng.random(bgr.shape[:2]) < thresh * 0.05
        out = bgr.copy()
        out[mask] = 255
        gain = float(p.get("brightness_coeff", 2.0))
        return cv2.convertScaleAbs(out, alpha=min(1.4, 0.7 + 0.15 * gain), beta=8), None
    if op == "random_shadow":
        out = bgr.copy()
        h, w = out.shape[:2]
        n_lo = int(p.get("num_shadows_lower", 1))
        n_hi = int(p.get("num_shadows_upper", 3))
        n = int(rng.integers(min(n_lo, n_hi), max(n_lo, n_hi) + 1))
        lo = float(p.get("shadow_intensity_lower", 0.2))
        hi = float(p.get("shadow_intensity_upper", 0.6))
        overlay = out.copy()
        for _ in range(max(0, n)):
            pts = np.array(
                [[int(rng.integers(0, w)), int(rng.integers(0, h))] for _ in range(4)],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(overlay, pts, (0, 0, 0))
        strength = float(rng.uniform(min(lo, hi), max(lo, hi)))
        return cv2.addWeighted(out, 1.0 - strength, overlay, strength, 0), None
    if op == "random_sun_flare":
        out = bgr.copy()
        h, w = out.shape[:2]
        x1, y1, x2, y2 = p.get("flare_roi", (0.2, 0.0, 0.8, 0.5))
        cx = int(rng.uniform(min(x1, x2), max(x1, x2)) * (w - 1))
        cy = int(rng.uniform(min(y1, y2), max(y1, y2)) * (h - 1))
        n_lo = int(p.get("num_circles_lower", 6))
        n_hi = int(p.get("num_circles_upper", 10))
        n = int(rng.integers(min(n_lo, n_hi), max(n_lo, n_hi) + 1))
        radius = max(4, int(p.get("src_radius", 200)) // 8)
        angle = float(rng.uniform(float(p.get("angle_lower", 0.0)), float(p.get("angle_upper", 1.0)))) * np.pi
        for i in range(max(1, n)):
            px = int(cx + np.cos(angle) * i * radius)
            py = int(cy + np.sin(angle) * i * radius * 0.6)
            cv2.circle(out, (px, py), max(2, radius // (i + 1)), (255, 240, 180), -1)
        return cv2.addWeighted(bgr, 0.75, out, 0.25, 0), None
    if op == "brightness_contrast":
        b = float(p.get("brightness_limit", 0.2))
        c = float(p.get("contrast_limit", 0.2))
        alpha = 1.0 + float(rng.uniform(-c, c))
        beta = float(rng.uniform(-b, b)) * 255.0
        return cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta), None
    if op == "hsv":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.integers(-abs(int(p.get("hue", 10))), abs(int(p.get("hue", 10))) + 1))) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + int(rng.integers(-abs(int(p.get("sat", 15))), abs(int(p.get("sat", 15))) + 1)), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + int(rng.integers(-abs(int(p.get("val", 10))), abs(int(p.get("val", 10))) + 1)), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), None
    if op == "motion_blur":
        k = _odd(int(p.get("blur_limit", 7)))
        angle = _range_float(rng, float(p.get("angle_lower", 0.0)), float(p.get("angle_upper", 0.0)))
        return cv2.filter2D(bgr, -1, _motion_kernel(k, angle)), None
    if op == "gaussian_blur":
        lo, hi = p.get("blur_limit", (3, 7))
        k = _odd(int(rng.integers(int(min(lo, hi)), int(max(lo, hi)) + 1)))
        return cv2.GaussianBlur(bgr, (k, k), 0), None
    if op == "blur":
        lo, hi = p.get("blur_limit", (3, 7))
        k = _odd(int(rng.integers(int(min(lo, hi)), int(max(lo, hi)) + 1)))
        return cv2.blur(bgr, (k, k)), None
    if op == "median_blur":
        lo, hi = p.get("blur_limit", (3, 7))
        k = _odd(int(rng.integers(int(min(lo, hi)), int(max(lo, hi)) + 1)))
        return cv2.medianBlur(bgr, k), None
    if op == "defocus":
        rlo, rhi = p.get("radius", (2, 4))
        radius = int(rng.integers(int(min(rlo, rhi)), int(max(rlo, rhi)) + 1))
        k = _odd(radius * 2 + 1)
        return cv2.GaussianBlur(bgr, (k, k), 0), None
    if op == "gauss_noise":
        vlo, vhi = p.get("var_limit", (10.0, 50.0))
        var = float(rng.uniform(float(min(vlo, vhi)), float(max(vlo, vhi))))
        noise = rng.normal(0.0, np.sqrt(max(var, 0.0)), bgr.shape)
        return np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8), None
    if op == "iso_noise":
        color_shift = _range_float(rng, float(p.get("color_shift_lo", 0.01)), float(p.get("color_shift_hi", 0.05)))
        intensity = _range_float(rng, float(p.get("intensity_lo", 0.1)), float(p.get("intensity_hi", 0.5)))
        return _iso_noise(bgr, color_shift, intensity, rng), None
    if op == "rgb_shift":
        limit = abs(int(p.get("shift_limit", 20)))
        return (
            _rgb_shift(
                bgr,
                _range_int(rng, -limit, limit),
                _range_int(rng, -limit, limit),
                _range_int(rng, -limit, limit),
            ),
            None,
        )
    if op == "random_gamma":
        lo = float(p.get("gamma_lo", 80))
        hi = float(p.get("gamma_hi", 120))
        return _lut_gamma(bgr, _range_float(rng, lo, hi) / 100.0), None
    if op == "clahe":
        clip = _range_float(rng, float(p.get("clip_lo", 1.0)), float(p.get("clip_hi", 4.0)))
        return _clahe_bgr(bgr, clip, int(p.get("tile_grid", 8))), None
    if op == "coarse_dropout":
        h, w = bgr.shape[:2]
        n = _range_int(rng, int(p.get("holes_lo", 1)), int(p.get("holes_hi", 8)))
        holes = []
        for _ in range(max(0, n)):
            hole_w = max(1, int(round(w * _range_float(rng, float(p.get("width_lo", 0.02)), float(p.get("width_hi", 0.1))))))
            hole_h = max(1, int(round(h * _range_float(rng, float(p.get("height_lo", 0.02)), float(p.get("height_hi", 0.1))))))
            x1 = int(rng.integers(0, max(1, w - hole_w + 1)))
            y1 = int(rng.integers(0, max(1, h - hole_h + 1)))
            holes.append((x1, y1, x1 + hole_w, y1 + hole_h))
        return _coarse_dropout(bgr, holes, int(p.get("fill_value", 0))), None
    if op == "image_compression":
        qlo = int(p.get("quality_lower", 30))
        qhi = int(p.get("quality_upper", 95))
        q = int(rng.integers(min(qlo, qhi), max(qlo, qhi) + 1))
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(q, 1, 100))])
        if not ok:
            return bgr, None
        decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return (bgr if decoded is None else decoded), None
    if op == "elastic_transform":
        h, w = bgr.shape[:2]
        map_x, map_y, affine = _elastic_maps(
            h,
            w,
            float(p.get("alpha", 1.0)),
            float(p.get("sigma", 50.0)),
            float(p.get("alpha_affine", 50.0)),
            rng,
        )
        image = bgr
        if affine is not None:
            image = cv2.warpAffine(image, affine, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return image, _compose_mapper(affine, _inverse_remap_mapper(map_x, map_y))
    if op == "grid_distortion":
        h, w = bgr.shape[:2]
        map_x, map_y, xx, yy = _grid_distortion_maps(
            h,
            w,
            int(p.get("num_steps", 5)),
            float(p.get("distort_limit", 0.3)),
            rng,
        )
        image = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return image, _separable_inverse_mapper(xx, yy)
    if op == "optical_distortion":
        h, w = bgr.shape[:2]
        limit = abs(float(p.get("distort_limit", 0.05)))
        k = _range_float(rng, -limit, limit)
        shift = abs(float(p.get("shift_limit", 0.05)))
        map_x, map_y = _optical_maps(
            h,
            w,
            k,
            _range_float(rng, -shift, shift) * w,
            _range_float(rng, -shift, shift) * h,
        )
        image = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return image, _inverse_remap_mapper(map_x, map_y)
    return bgr, None


def apply_one_method_dataset(
    *,
    image_paths: List[str],
    steps: List[AugmentStep],
    overwrite: bool,
    output_dir: Optional[str],
    label_for: Optional[Callable[[str], Optional[str]]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Tuple[int, int]:
    """Write one augmented image per source. Unchanged originals are not copied."""
    enabled = [step for step in steps if step.enabled and float(step.p) > 0]
    if not image_paths or not enabled:
        return (0, 0)
    try:
        root = osp.commonpath([osp.dirname(p) for p in image_paths])
    except ValueError:
        root = osp.dirname(image_paths[0])
    stamp = datetime.now().strftime("%Y%m%d%H")
    total = 0
    ok = 0
    for index, img_path in enumerate(image_paths):
        if should_cancel and should_cancel():
            break
        total += 1
        if on_progress:
            on_progress(index + 1, len(image_paths), img_path)
        step = assign_one_method(index, enabled)
        if step is None:
            continue
        rng = np.random.default_rng(abs(hash((img_path, step.op_id, index))) % (2**32))
        if float(rng.random()) > float(step.p):
            continue
        bgr = _imread_bgr(img_path)
        if bgr is None:
            continue
        out, mapper = _apply_one(bgr, step, rng)
        if overwrite:
            out_path = img_path
        else:
            if not output_dir:
                continue
            rel = osp.relpath(img_path, root)
            stem = osp.splitext(osp.basename(rel))[0]
            out_name = f"{stem}_{step.op_id}_{stamp}.jpg"
            out_path = osp.join(output_dir, osp.dirname(rel), out_name)
        _write_image(out_path, out)
        if mapper is not None or not overwrite:
            src_json = label_for(img_path) if label_for else None
            if src_json is None:
                side = osp.splitext(img_path)[0] + ".json"
                src_json = side if osp.isfile(side) else None
            h, w = out.shape[:2]
            _write_label(src_json, out_path, w, h, mapper)
        ok += 1
    return total, ok
