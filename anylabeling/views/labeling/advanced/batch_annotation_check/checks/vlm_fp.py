from __future__ import annotations

import json
import os
import os.path as osp
import tempfile
from typing import Any, Dict, Optional, Tuple

from anylabeling.views.labeling.advanced.dataset_analysis.vlm import generate_caption
from anylabeling.views.labeling.label_converter import LabelConverter


_FP_JSON_PROMPT = (
    "你是标注质检助手。给定一张裁剪小图与“期望标签”，判断该小图是否属于该标签。\n"
    "必须仅输出严格 JSON（不要输出任何额外文本），格式如下：\n"
    "{{"
    '"is_match": true/false, '
    '"confidence": 0.0-1.0, '
    '"reason": "简要原因"'
    "}}\n"
    "期望标签：{label}\n"
)


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    # Try direct JSON first
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Best-effort: locate first {...}
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(s[i : j + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _resize_to_224_letterbox(image_path: str) -> Tuple[str, Optional[str]]:
    """
    Returns (path_to_send, tmp_path_to_cleanup).
    If resize fails, returns (image_path, None).
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return image_path, None

    try:
        bgr = LabelConverter.imread_unicode(image_path)
        if bgr is None:
            return image_path, None
        h, w = bgr.shape[:2]
        if h <= 0 or w <= 0:
            return image_path, None

        target = 224
        scale = min(target / float(w), target / float(h))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.full((target, target, 3), 0, dtype=np.uint8)
        x0 = (target - nw) // 2
        y0 = (target - nh) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized

        fd, tmp_path = tempfile.mkstemp(prefix="xanylabeling_fp224_", suffix=".jpg")
        os.close(fd)
        ok = cv2.imwrite(tmp_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return image_path, None
        return tmp_path, tmp_path
    except Exception:
        return image_path, None


def judge_crop_matches_label(
    crop_path: str,
    *,
    label: str,
    api_key: str,
    base_url: str,
    model: str,
    max_side: int = 512,
    max_tokens: int = 256,
    timeout: float = 120.0,
    prompt_template: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "is_match": bool|None,
        "confidence": float|None,
        "reason": str|None,
        "raw_text": str,
        "error": str|None,
        "prompt": str,
      }
    """
    tpl = prompt_template if isinstance(prompt_template, str) and prompt_template.strip() else _FP_JSON_PROMPT
    prompt = tpl.format(label=str(label))
    send_path, tmp_path = _resize_to_224_letterbox(crop_path)
    try:
        out = generate_caption(
            send_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt=prompt,
            max_side=int(max_side),
            max_tokens=int(max_tokens),
            timeout=float(timeout),
        )
        raw = str(out.get("caption") or "").strip()
        obj = _extract_json_obj(raw)
        if not obj:
            return {
                "is_match": None,
                "confidence": None,
                "reason": None,
                "raw_text": raw,
                "error": "VLM did not return valid JSON.",
                "prompt": prompt,
            }
        is_match = obj.get("is_match")
        conf = obj.get("confidence")
        reason = obj.get("reason")
        return {
            "is_match": bool(is_match) if isinstance(is_match, bool) else None,
            "confidence": float(conf) if isinstance(conf, (int, float)) else None,
            "reason": str(reason) if isinstance(reason, str) else None,
            "raw_text": raw,
            "error": None,
            "prompt": prompt,
        }
    except Exception as e:
        return {
            "is_match": None,
            "confidence": None,
            "reason": None,
            "raw_text": "",
            "error": str(e),
            "prompt": prompt,
        }
    finally:
        if tmp_path and osp.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

