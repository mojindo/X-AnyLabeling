"""No-reference image metrics for dataset analysis (OpenCV + NumPy)."""

from __future__ import annotations

import math
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

_BRISQUE_DATA_DIR = osp.join(osp.dirname(__file__), "data")
_BRISQUE_MODEL_YML = osp.join(_BRISQUE_DATA_DIR, "brisque_model_live.yml")
_BRISQUE_RANGE_YML = osp.join(_BRISQUE_DATA_DIR, "brisque_range_live.yml")

import cv2
import numpy as np
from PIL import ExifTags, Image

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.label_converter import LabelConverter

ORIENTATION_TAG_NAMES = {v: k for k, v in ExifTags.TAGS.items()}


def _resize_max_side(bgr: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side or max_side <= 0:
        return bgr, 1.0
    scale = max_side / float(m)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return resized, scale


def _circular_mean_hue_deg(h_deg: np.ndarray, mask: Optional[np.ndarray] = None):
    """h_deg in [0, 360)."""
    if mask is None:
        hx = np.cos(np.deg2rad(h_deg))
        hy = np.sin(np.deg2rad(h_deg))
    else:
        m = mask.astype(bool)
        if not np.any(m):
            return None
        hx = np.cos(np.deg2rad(h_deg[m]))
        hy = np.sin(np.deg2rad(h_deg[m]))
    mean_rad = math.atan2(float(np.mean(hy)), float(np.mean(hx)))
    deg = math.degrees(mean_rad) % 360.0
    return deg


def _read_orientation_value(exif) -> Optional[int]:
    """Tag 274 (Orientation): main IFD, then Exif sub-IFD."""
    if exif is None:
        return None
    ori_id = ORIENTATION_TAG_NAMES.get("Orientation") or 274
    val = exif.get(ori_id)
    if val is not None:
        return int(val)
    try:
        ifd_exif = getattr(ExifTags, "IFD", None)
        if ifd_exif is not None:
            sub = exif.get_ifd(ifd_exif.Exif)
            if sub:
                v = sub.get(ori_id)
                if v is not None:
                    return int(v)
    except Exception:
        pass
    return None


def _exif_orientation(image_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "orientation_tag": None,
        "orientation_label": None,
    }
    labels = {
        1: "horizontal(normal)",
        2: "mirrored_horizontal",
        3: "rotate_180",
        4: "mirrored_vertical",
        5: "mirrored_horizontal_then_rotate_270",
        6: "rotate_90",
        7: "mirrored_horizontal_then_rotate_90",
        8: "rotate_270",
    }
    try:
        with Image.open(image_path) as im:
            exif = im.getexif()
            if exif is not None and len(exif) == 0 and "exif" in im.info:
                try:
                    raw = Image.Exif()
                    raw.load(im.info["exif"])
                    exif = raw
                except Exception:
                    pass
            val = _read_orientation_value(exif)
            out["orientation_tag"] = val
            if val is not None:
                out["orientation_label"] = labels.get(val, f"tag_{val}")
    except Exception as e:
        logger.debug("EXIF read failed for %s: %s", image_path, e)
    return out


def _estimate_roll_deg(gray: np.ndarray) -> Optional[float]:
    """Rough horizon / roll from dominant line angle (degrees)."""
    try:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(30, min(gray.shape) // 20),
            minLineLength=max(20, min(gray.shape) // 10),
            maxLineGap=12,
        )
        if lines is None or len(lines) == 0:
            return None
        angles = []
        for ln in lines[:, 0]:
            x1, y1, x2, y2 = ln
            if x2 == x1:
                continue
            ang = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
            while ang > 90:
                ang -= 180
            while ang <= -90:
                ang += 180
            if -45 <= ang <= 45:
                angles.append(ang)
        if not angles:
            return None
        return float(np.median(np.array(angles, dtype=np.float32)))
    except Exception:
        return None


def _noise_mad(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    resid = gray.astype(np.float32) - blur.astype(np.float32)
    med = np.median(resid)
    return float(np.median(np.abs(resid - med)) * 1.4826)


def _artifact_heuristic_score(gray: np.ndarray) -> float:
    """
    Heuristic 0–100: higher may indicate stronger blocking/compression artifacts.
    Not a perceptual benchmark.
    """
    h, w = gray.shape[:2]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    eh, ew = h - (h % 8), w - (w % 8)
    if eh < 8 or ew < 8:
        return 0.0
    mag = mag[:eh, :ew]
    blocked = mag.reshape(eh // 8, 8, ew // 8, 8).mean(axis=(1, 3))
    block_var = float(np.var(blocked))
    edge_mean = float(np.mean(mag))
    raw = math.log1p(block_var * 2.0) * 15.0 + math.log1p(edge_mean) * 5.0
    return float(max(0.0, min(100.0, raw)))


def _try_brisque(bgr: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    """
    OpenCV 4.13+ requires model_file_path and range_file_path (bundled under data/).
    """
    try:
        qmod = getattr(cv2, "quality", None)
        if qmod is None:
            return None, "cv2.quality not available in this OpenCV build"
        br_fn = getattr(qmod, "QualityBRISQUE_compute", None)
        if br_fn is None:
            return None, "QualityBRISQUE_compute not available"
        if not osp.isfile(_BRISQUE_MODEL_YML) or not osp.isfile(
            _BRISQUE_RANGE_YML
        ):
            return (
                None,
                "BRISQUE YAML models missing under dataset_analysis/data/ "
                "(brisque_model_live.yml, brisque_range_live.yml)",
            )
        # OpenCV expects BGR, same as imread
        out = br_fn(bgr, _BRISQUE_MODEL_YML, _BRISQUE_RANGE_YML)
        if isinstance(out, (tuple, list)) and len(out) > 0:
            return float(out[0]), None
        return float(out), None
    except Exception as e:
        return None, str(e)


def numeric_keys_for_charts() -> List[str]:
    return [
        "original_width",
        "original_height",
        "megapixels",
        "aspect_ratio",
        "sharpness_laplacian_var_log1p",
        "brightness_l_mean",
        "contrast_rms_gray",
        "hue_mean_deg",
        "saturation_mean",
        "value_mean",
        "noise_mad_residual",
        "artifact_heuristic_score",
        "dominant_line_roll_deg",
        "brisque",
    ]


def compute_image_metrics(
    image_path: str, max_side: int = 1024
) -> Dict[str, Any]:
    """
    Compute no-reference metrics on an image file.

    Returns a dict for xal_dataset_analysis["metrics"].
    """
    if not osp.isfile(image_path):
        raise FileNotFoundError(image_path)

    # Support Unicode paths on Windows by using imdecode+fromfile
    bgr, _size = LabelConverter.imread_unicode(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    orig_h, orig_w = bgr.shape[:2]
    megapixels = float(orig_w * orig_h) / 1_000_000.0
    aspect_ratio = float(orig_w) / float(orig_h) if orig_h else None
    work, scale = _resize_max_side(bgr, max_side)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_log = float(np.log1p(lap_var))

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    l_mean = float(np.mean(lab[:, :, 0]))

    contrast_rms = float(np.std(gray.astype(np.float64)))

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0].astype(np.float32) * 2.0
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))

    hue_mean_deg = _circular_mean_hue_deg(h_ch)

    brisque, brisque_err = _try_brisque(work)

    noise_mad = _noise_mad(gray)
    artifact_score = _artifact_heuristic_score(gray)

    exif = _exif_orientation(image_path)
    roll_deg = _estimate_roll_deg(gray)

    return {
        "image_path_resolved": osp.basename(image_path),
        "original_width": int(orig_w),
        "original_height": int(orig_h),
        "megapixels": megapixels,
        "aspect_ratio": aspect_ratio,
        "analysis_max_side": int(max_side),
        "analysis_scale": float(scale),
        "sharpness_laplacian_var_log1p": sharpness_log,
        "brightness_l_mean": l_mean,
        "contrast_rms_gray": contrast_rms,
        "hue_mean_deg": hue_mean_deg,
        "saturation_mean": s_mean,
        "value_mean": v_mean,
        "noise_mad_residual": noise_mad,
        "artifact_heuristic_score": artifact_score,
        "artifact_method": "heuristic_block_edge_energy",
        "brisque": brisque,
        "brisque_error": brisque_err,
        "exif_orientation": exif,
        "dominant_line_roll_deg": roll_deg,
        "methods_note": (
            "Sharpness: log1p(Laplacian variance); contrast: RMS on gray; "
            "BRISQUE may be null without OpenCV quality models; "
            "artifact_score is a heuristic only."
        ),
    }
