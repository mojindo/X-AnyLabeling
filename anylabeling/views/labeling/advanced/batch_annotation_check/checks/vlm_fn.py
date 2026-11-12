from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from anylabeling.views.labeling.advanced.dataset_analysis.vlm import generate_caption


_FN_JSON_PROMPT = (
    "你是目标检测助手。给定一张图片（其中已遮挡现有标注区域），请判断是否仍然存在目标。\n"
    "必须仅输出严格 JSON（不要输出任何额外文本）。\n"
    "要求：所有 bbox 坐标必须使用 0-1 归一化（相对于整张图宽高），并确保 0<=x1<x2<=1 且 0<=y1<y2<=1。\n"
    "输出格式：\n"
    "{\n"
    '  "norm": "0-1",\n'
    '  "detections": [\n'
    '    {"label": "xxx", "bbox": [x1,y1,x2,y2], "score": 0.0-1.0}\n'
    "  ]\n"
    "}\n"
    "需要检测的目标标签（可多类）：{labels}\n"
)


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        i = s.find("{")
        j = s.rfind("}")
        if i >= 0 and j > i:
            try:
                obj = json.loads(s[i : j + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def detect_missing_objects(
    masked_image_path: str,
    *,
    labels: List[str],
    api_key: str,
    base_url: str,
    model: str,
    max_side: int = 1024,
    max_tokens: int = 768,
    timeout: float = 120.0,
    prompt_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "detections": [{"label": str, "bbox01": [x1,y1,x2,y2], "score": float|None}],
        "raw_text": str,
        "error": str|None,
        "prompt": str,
      }
    """
    lbl = ", ".join([str(x) for x in (labels or []) if str(x).strip()])
    if isinstance(prompt_override, str) and prompt_override.strip():
        prompt = prompt_override
    else:
        prompt = _FN_JSON_PROMPT.format(labels=lbl)
    try:
        out = generate_caption(
            masked_image_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt=prompt,
            max_side=int(max_side),
            max_tokens=int(max_tokens),
            timeout=float(timeout),
        )
        raw = str(out.get("caption") or "").strip()
        obj = _parse_json(raw)
        if not obj:
            return {
                "detections": [],
                "raw_text": raw,
                "error": "VLM did not return valid JSON.",
                "prompt": prompt,
            }
        dets_in = obj.get("detections") or []
        dets: List[Dict[str, Any]] = []
        if isinstance(dets_in, list):
            for d in dets_in:
                if not isinstance(d, dict):
                    continue
                label = d.get("label")
                bbox = d.get("bbox")
                score = d.get("score")
                if not isinstance(label, str) or not label.strip():
                    continue
                if (
                    not isinstance(bbox, (list, tuple))
                    or len(bbox) != 4
                ):
                    continue
                try:
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                except Exception:
                    continue
                # Expect 0-1; clamp and validate.
                x1, y1, x2, y2 = _clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                sc = None
                if isinstance(score, (int, float)):
                    sc = float(score)
                dets.append(
                    {
                        "label": label.strip(),
                        "bbox01": [x1, y1, x2, y2],
                        "score": sc,
                    }
                )
        return {
            "detections": dets,
            "raw_text": raw,
            "error": None,
            "prompt": prompt,
        }
    except Exception as e:
        return {
            "detections": [],
            "raw_text": "",
            "error": str(e),
            "prompt": prompt,
        }

