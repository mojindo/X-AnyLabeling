from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ClipFpResult:
    similarity: Optional[float]
    passed: Optional[bool]
    backend: str
    error: Optional[str] = None


def _try_load_open_clip():
    """
    Best-effort local CLIP backend.

    This function intentionally avoids adding hard dependencies. If users want
    local inference, they can install:
      - torch
      - open_clip_torch
      - pillow
    """
    try:
        import torch  # type: ignore
        import open_clip  # type: ignore
    except Exception as e:  # noqa: BLE001
        return None, None, None, f"Missing dependency for local CLIP: {e}"
    return torch, open_clip, None, None


_CLIP_SINGLETON = {
    "loaded": False,
    "error": None,
    "torch": None,
    "open_clip": None,
    "model": None,
    "preprocess": None,
    "tokenizer": None,
    "device": None,
}


def _ensure_clip_loaded(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    if _CLIP_SINGLETON["loaded"]:
        return
    torch, open_clip, _unused, err = _try_load_open_clip()
    if err:
        _CLIP_SINGLETON["loaded"] = True
        _CLIP_SINGLETON["error"] = err
        return
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval().to(device)
        _CLIP_SINGLETON.update(
            {
                "loaded": True,
                "error": None,
                "torch": torch,
                "open_clip": open_clip,
                "model": model,
                "preprocess": preprocess,
                "tokenizer": tokenizer,
                "device": device,
            }
        )
    except Exception as e:  # noqa: BLE001
        _CLIP_SINGLETON["loaded"] = True
        _CLIP_SINGLETON["error"] = str(e)


def clip_similarity(
    image_path: str,
    *,
    label_text: str,
    threshold: float = 0.23,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> ClipFpResult:
    """
    Compute cosine similarity between image and label text.

    Returns a result with similarity in [-1, 1] (open_clip normalized features).
    """
    _ensure_clip_loaded(model_name=model_name, pretrained=pretrained)
    err = _CLIP_SINGLETON.get("error")
    if err:
        return ClipFpResult(
            similarity=None,
            passed=None,
            backend="open_clip",
            error=err,
        )
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # noqa: BLE001
        return ClipFpResult(
            similarity=None,
            passed=None,
            backend="open_clip",
            error=f"Missing dependency for local CLIP (PIL): {e}",
        )

    torch = _CLIP_SINGLETON["torch"]
    model = _CLIP_SINGLETON["model"]
    preprocess = _CLIP_SINGLETON["preprocess"]
    tokenizer = _CLIP_SINGLETON["tokenizer"]
    device = _CLIP_SINGLETON["device"]
    if not all([torch, model, preprocess, tokenizer, device]):
        return ClipFpResult(
            similarity=None,
            passed=None,
            backend="open_clip",
            error="Local CLIP backend not initialized.",
        )
    try:
        img = Image.open(image_path).convert("RGB")
        image = preprocess(img).unsqueeze(0).to(device)
        text = tokenizer([str(label_text)]).to(device)

        with torch.no_grad():
            im_feat = model.encode_image(image)
            tx_feat = model.encode_text(text)
            im_feat = im_feat / im_feat.norm(dim=-1, keepdim=True)
            tx_feat = tx_feat / tx_feat.norm(dim=-1, keepdim=True)
            sim = float((im_feat @ tx_feat.T).squeeze().item())
        return ClipFpResult(
            similarity=sim,
            passed=bool(sim >= float(threshold)),
            backend=f"open_clip:{model_name}:{pretrained}",
            error=None,
        )
    except Exception as e:  # noqa: BLE001
        return ClipFpResult(
            similarity=None,
            passed=None,
            backend="open_clip",
            error=str(e),
        )

