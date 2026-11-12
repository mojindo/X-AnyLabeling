"""Optional multimodal caption via OpenAI-compatible vision API."""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional

import cv2
import numpy as np
from openai import OpenAI
from anylabeling.views.labeling.label_converter import LabelConverter


def _jpeg_base64(bgr, max_side: int = 1024) -> str:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m > max_side > 0:
        s = max_side / float(m)
        bgr = cv2.resize(
            bgr,
            (int(round(w * s)), int(round(h * s))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def generate_caption(
    image_path: str = "",
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_side: int = 1024,
    max_tokens: int = 2048,
    timeout: float = 120.0,
    image_bgr: Optional[np.ndarray] = None,
    image_base64: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Call OpenAI-compatible chat completions with a single image (base64 data URL).

    Image source (first match wins):
    1. ``image_base64`` — already JPEG/PNG base64 (no re-encode / max_side)
    2. ``image_bgr`` — numpy BGR array (JPEG + optional max_side resize)
    3. ``image_path`` — load from disk
    """
    if image_base64:
        b64 = str(image_base64).strip()
    elif image_bgr is not None:
        b64 = _jpeg_base64(image_bgr, max_side=max_side)
    else:
        if not image_path:
            raise ValueError("image_path, image_bgr, or image_base64 is required")
        bgr, _size = LabelConverter.imread_unicode(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")
        b64 = _jpeg_base64(bgr, max_side=max_side)

    data_url = f"data:image/jpeg;base64,{b64}"

    user_content: list = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    messages: list = []
    sys_text = (system_prompt or "").strip()
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    messages.append({"role": "user", "content": user_content})

    client = OpenAI(api_key=api_key or "dummy", base_url=base_url, timeout=timeout)
    create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max(1, int(max_tokens)),
    }
    if temperature is not None:
        create_kwargs["temperature"] = float(temperature)

    resp = client.chat.completions.create(**create_kwargs)
    text = ""
    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        pass
    return {
        "caption": text,
        "model": model,
        "base_url": base_url,
        "prompt": prompt,
        "system_prompt": sys_text,
        "prompt_version": 2,
        "max_tokens": int(max_tokens),
    }
