"""Chinese-CLIP (ONNX) false-positive check for Batch Annotation Check.

Uses the same ViT-B-16 weights / download paths as EdgeSAM-CN-CLIP
(``edge_sam_with_chinese_clip.yaml``). Inference is CPU-only.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from urllib.parse import urlparse

import anylabeling.configs as configs_pkg
from anylabeling.config import get_work_directory
from anylabeling.services.auto_labeling.__base__.clip import ChineseClipONNX
from anylabeling.services.auto_labeling.model import Model, load_model_config
from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.views.labeling.logger import logger

CLIP_CONFIG_NAME = "edge_sam_with_chinese_clip.yaml"
BACKEND_NAME = "chinese_clip_onnx:ViT-B-16:cpu:contrastive"
# Minimum softmax probability for the positive phrase among
# positive + negative distractors. Absolute cosine alone is too weak.
DEFAULT_THRESHOLD = 0.40

# Suggested Chinese phrases when the annotation label is English/code-like.
DEFAULT_LABEL_PROMPTS: Dict[str, str] = {
    "person": "人",
    "people": "人",
    "pedestrian": "行人",
    "man": "男人",
    "woman": "女人",
    "child": "小孩",
    "car": "汽车",
    "truck": "卡车",
    "bus": "公交车",
    "bicycle": "自行车",
    "motorcycle": "摩托车",
    "bike": "自行车",
    "dog": "狗",
    "cat": "猫",
}

# Distractors so cabinets/pipes/background lose to "人" less often.
DEFAULT_NEGATIVE_PROMPTS: List[str] = [
    "电气柜",
    "配电柜",
    "柜子",
    "设备",
    "机器",
    "管道",
    "栏杆",
    "柱子",
    "墙壁",
    "地面",
    "金属结构",
    "背景",
    "空场景",
    "其他物体",
]


def resolve_label_prompt(
    label: str, label_prompts: Optional[Dict[str, str]] = None
) -> str:
    """Map annotation label to CLIP text; fall back to label itself."""
    raw = str(label or "").strip()
    if not raw:
        return "object"
    if label_prompts:
        mapped = label_prompts.get(raw)
        if isinstance(mapped, str) and mapped.strip():
            return mapped.strip()
        mapped = label_prompts.get(raw.lower())
        if isinstance(mapped, str) and mapped.strip():
            return mapped.strip()
    hint = DEFAULT_LABEL_PROMPTS.get(raw.lower())
    if hint:
        return hint
    return raw

_CLIP_LOCK = threading.Lock()
_CLIP_STATE: Dict[str, Any] = {
    "net": None,
    "error": None,
}


@dataclass(frozen=True)
class ClipFpResult:
    similarity: Optional[float]  # positive-class softmax probability
    passed: Optional[bool]
    backend: str
    error: Optional[str] = None
    reason: Optional[str] = None
    best_label: Optional[str] = None


class _ClipWeightHelper(Model):
    """Concrete Model used only to reuse ``get_model_abs_path`` downloads."""

    class Meta(Model.Meta):
        required_config_names = []

    def predict_shapes(self, image, filename=None):
        return AutoLabelingResult([], replace=False)

    def unload(self):
        return None


def _load_cn_clip_config() -> Dict[str, Any]:
    resource_path = pkg_resources.files(configs_pkg).joinpath(
        "auto_labeling", CLIP_CONFIG_NAME
    )
    content = resource_path.read_text(encoding="utf-8")
    cfg = load_model_config(content)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Invalid CLIP config: {CLIP_CONFIG_NAME}")
    cfg["config_file"] = str(resource_path)
    return cfg


def _filename_from_url_or_path(value: str) -> str:
    if str(value).startswith(("http://", "https://")):
        return os.path.basename(urlparse(value).path)
    return os.path.basename(str(value))


def _local_model_cache_path(model_name: str, filename: str) -> str:
    """Same layout as ``Model.get_model_abs_path`` cache folder."""
    work_dir = get_work_directory()
    for data_dir in ("xanylabeling_data", "anylabeling_data"):
        path = os.path.abspath(
            os.path.join(work_dir, data_dir, "models", model_name, filename)
        )
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    # Preferred default location when missing (download target).
    return os.path.abspath(
        os.path.join(
            work_dir, "xanylabeling_data", "models", model_name, filename
        )
    )


def _resolve_cn_clip_weight(
    helper: _ClipWeightHelper,
    field: str,
    *,
    on_message: Optional[Callable[[str], None]] = None,
) -> str:
    """Prefer existing local cache; only download when the file is missing."""
    cfg = helper.config
    raw = str(cfg.get(field) or "")
    filename = _filename_from_url_or_path(raw)
    model_name = str(cfg.get("name") or "edge_sam_with_chinese_clip-r20240131")
    local = _local_model_cache_path(model_name, filename)
    if os.path.isfile(local) and os.path.getsize(local) > 0:
        if on_message:
            try:
                on_message(f"Using local Chinese-CLIP weight: {filename}")
            except Exception:
                pass
        logger.info("Using local Chinese-CLIP weight: %s", local)
        return local
    return helper.get_model_abs_path(cfg, field)


def _read_crop_rgb(image_path: str) -> np.ndarray:
    """Load crop as RGB uint8 HWC (handles Unicode paths + optional alpha)."""
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(image_path)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255, dtype=np.float32)
        bgr_f = bgr.astype(np.float32)
        comp = bgr_f * alpha + white * (1.0 - alpha)
        img = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def ensure_chinese_clip_loaded(
    *,
    on_message: Optional[Callable[[str], None]] = None,
    force_reload: bool = False,
) -> ChineseClipONNX:
    """Load ChineseClipONNX on CPU; download weights if missing."""
    with _CLIP_LOCK:
        if force_reload:
            _CLIP_STATE["net"] = None
            _CLIP_STATE["error"] = None
        if _CLIP_STATE["net"] is not None:
            return _CLIP_STATE["net"]
        # Allow retry after a previous load failure (e.g. ORT init error).
        _CLIP_STATE["error"] = None

        def _msg(text: str) -> None:
            if on_message:
                try:
                    on_message(str(text))
                except Exception:
                    pass
            logger.info(text)

        try:
            cfg = _load_cn_clip_config()
            helper = _ClipWeightHelper(cfg, on_message=_msg)
            # Prefer ~/xanylabeling_data/models/<name>/...; download only if absent.
            txt_model = _resolve_cn_clip_weight(
                helper, "txt_model_path", on_message=_msg
            )
            _ = _resolve_cn_clip_weight(
                helper, "txt_extra_path", on_message=_msg
            )
            img_model = _resolve_cn_clip_weight(
                helper, "img_model_path", on_message=_msg
            )
            _ = _resolve_cn_clip_weight(
                helper, "img_extra_path", on_message=_msg
            )
            model_arch = str(helper.config.get("model_arch") or "ViT-B-16")
            _msg("Loading Chinese-CLIP (ONNX, CPU)...")
            net = ChineseClipONNX(
                txt_model,
                img_model,
                model_arch,
                device="cpu",
            )
            _CLIP_STATE["net"] = net
            _CLIP_STATE["error"] = None
            return net
        except Exception as e:  # noqa: BLE001
            _CLIP_STATE["net"] = None
            _CLIP_STATE["error"] = str(e)
            raise


def reset_chinese_clip_cache() -> None:
    """Drop cached session (e.g. after a failed partial download)."""
    with _CLIP_LOCK:
        _CLIP_STATE["net"] = None
        _CLIP_STATE["error"] = None


def _softmax_probs(img_feat: np.ndarray, txt_feat: np.ndarray) -> np.ndarray:
    """Chinese-CLIP style image-text probabilities (temperature 100)."""
    logits = 100.0 * np.dot(img_feat, txt_feat.T)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def clip_contrastive_match(
    image_path: str,
    *,
    positive_text: str,
    negative_texts: Optional[Sequence[str]] = None,
    min_prob: float = DEFAULT_THRESHOLD,
    on_message: Optional[Callable[[str], None]] = None,
    net: Optional[ChineseClipONNX] = None,
    txt_feat_cache: Optional[Dict[str, np.ndarray]] = None,
) -> ClipFpResult:
    """
    Contrastive FP score: positive phrase vs distractors.

    Marks as match only when the positive class is argmax AND its softmax
    probability is >= ``min_prob``. Absolute single-text cosine is too weak
    for cabinets/pipes labeled as person.
    """
    try:
        if net is None:
            net = ensure_chinese_clip_loaded(on_message=on_message)
    except Exception as e:  # noqa: BLE001
        return ClipFpResult(
            similarity=None,
            passed=None,
            backend=BACKEND_NAME,
            error=str(e),
        )
    try:
        pos = str(positive_text or "").strip() or "object"
        negs = [
            str(x).strip()
            for x in (negative_texts if negative_texts is not None else DEFAULT_NEGATIVE_PROMPTS)
            if str(x).strip() and str(x).strip() != pos
        ]
        # De-dupe while keeping order; positive always index 0.
        texts: List[str] = [pos]
        seen = {pos}
        for n in negs:
            if n not in seen:
                texts.append(n)
                seen.add(n)

        image = _read_crop_rgb(image_path)
        img_feat = net.img_pipeline(image)

        feats = []
        cache = txt_feat_cache if txt_feat_cache is not None else {}
        for t in texts:
            if t in cache:
                feats.append(cache[t])
            else:
                f = net.txt_pipeline([t])
                cache[t] = f
                feats.append(f)
        txt_feat = np.concatenate(feats, axis=0)
        probs = _softmax_probs(img_feat, txt_feat).reshape(-1)
        best_i = int(np.argmax(probs))
        pos_prob = float(probs[0])
        best_prob = float(probs[best_i])
        best_label = texts[best_i]
        margin = float(pos_prob - (best_prob if best_i != 0 else float(np.partition(probs, -2)[-2]) if len(probs) > 1 else 0.0))
        passed = bool(best_i == 0 and pos_prob >= float(min_prob))
        reason = (
            f"pos={pos} p={pos_prob:.3f}; best={best_label} "
            f"p={best_prob:.3f}; margin={margin:.3f}"
        )
        return ClipFpResult(
            similarity=pos_prob,
            passed=passed,
            backend=BACKEND_NAME,
            error=None,
            reason=reason,
            best_label=best_label,
        )
    except Exception as e:  # noqa: BLE001
        return ClipFpResult(
            similarity=None,
            passed=None,
            backend=BACKEND_NAME,
            error=str(e),
        )


# Back-compat alias
def clip_similarity(
    image_path: str,
    *,
    label_text: str,
    threshold: float = DEFAULT_THRESHOLD,
    on_message: Optional[Callable[[str], None]] = None,
) -> ClipFpResult:
    return clip_contrastive_match(
        image_path,
        positive_text=label_text,
        min_prob=threshold,
        on_message=on_message,
    )


def run_fp_check_on_crops(
    items: Sequence[Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    label_prompts: Optional[Dict[str, str]] = None,
    negative_prompts: Optional[Sequence[str]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Run contrastive FP check on crop items.

    Each result dict:
      image_path, label_json_path, shape_index, shape_key, label, prompt,
      similarity (pos prob), passed, backend, reason, best_label, error
    """
    results: List[Dict[str, Any]] = []
    total = len(items)
    if total == 0:
        return results
    prompts = dict(label_prompts or {})
    negatives = list(
        negative_prompts
        if negative_prompts is not None
        else DEFAULT_NEGATIVE_PROMPTS
    )

    try:
        net = ensure_chinese_clip_loaded(on_message=on_message)
    except Exception as e:  # noqa: BLE001
        err = str(e)
        for it in items:
            label = str(getattr(it, "label", "") or "")
            results.append(
                {
                    "image_path": getattr(it, "image_path", ""),
                    "label_json_path": getattr(it, "label_json_path", ""),
                    "shape_index": int(getattr(it, "shape_index", -1)),
                    "shape_key": str(getattr(it, "shape_key", "") or ""),
                    "label": label,
                    "prompt": resolve_label_prompt(label, prompts),
                    "similarity": None,
                    "passed": None,
                    "backend": BACKEND_NAME,
                    "reason": None,
                    "best_label": None,
                    "error": err,
                }
            )
        return results

    txt_cache: Dict[str, np.ndarray] = {}
    for i, it in enumerate(items):
        if should_cancel and should_cancel():
            break
        crop_path = str(getattr(it, "crop_path", "") or "")
        label = str(getattr(it, "label", "") or "")
        prompt = resolve_label_prompt(label, prompts)
        if on_progress:
            try:
                on_progress(i + 1, total, os.path.basename(crop_path) or label)
            except Exception:
                pass
        if not crop_path or not os.path.isfile(crop_path):
            res = ClipFpResult(
                similarity=None,
                passed=None,
                backend=BACKEND_NAME,
                error="Crop file missing.",
            )
        else:
            res = clip_contrastive_match(
                crop_path,
                positive_text=prompt,
                negative_texts=negatives,
                min_prob=threshold,
                on_message=on_message,
                net=net,
                txt_feat_cache=txt_cache,
            )
        results.append(
            {
                "image_path": getattr(it, "image_path", ""),
                "label_json_path": getattr(it, "label_json_path", ""),
                "shape_index": int(getattr(it, "shape_index", -1)),
                "shape_key": str(getattr(it, "shape_key", "") or ""),
                "label": label,
                "prompt": prompt,
                "similarity": res.similarity,
                "passed": res.passed,
                "backend": res.backend,
                "reason": res.reason,
                "best_label": res.best_label,
                "error": res.error,
            }
        )
    return results
