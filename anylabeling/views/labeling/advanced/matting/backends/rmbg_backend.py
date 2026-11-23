from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np

from anylabeling.services.auto_labeling.model import load_model_config
from anylabeling.services.auto_labeling.rmbg import RMBG
from anylabeling.views.labeling.advanced.matting.backends.base import MattingBackend
from anylabeling.views.labeling.advanced.matting.backends.registry import (
    MattingModelSpec,
    config_path_for,
)


class RmbgMattingBackend(MattingBackend):
    def __init__(
        self,
        spec: MattingModelSpec,
        *,
        on_message: Optional[Callable[..., None]] = None,
    ) -> None:
        path = config_path_for(spec.yaml_name)
        with open(path, "r", encoding="utf-8") as fp:
            cfg = load_model_config(fp)
        cfg["config_file"] = str(path)
        self._spec = spec
        self._rmbg = RMBG(cfg, on_message or (lambda *_a, **_k: None))

    @property
    def display_name(self) -> str:
        return self._spec.display_name

    def predict_mask(self, bgr: np.ndarray) -> np.ndarray:
        if bgr is None or bgr.size == 0:
            raise ValueError("empty image")
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        blob = self._rmbg.preprocess(rgb)
        output = self._rmbg.forward(blob)
        return self._rmbg.postprocess(output, rgb.shape[:2])

    def unload(self) -> None:
        try:
            self._rmbg.unload()
        except Exception:
            pass


def create_matting_backend(
    model_key: str,
    *,
    on_message: Optional[Callable[..., None]] = None,
) -> MattingBackend:
    from anylabeling.views.labeling.advanced.matting.backends.registry import (
        MODEL_BY_KEY,
    )

    spec = MODEL_BY_KEY.get(model_key)
    if spec is None:
        raise ValueError(f"Unknown matting model: {model_key}")
    return RmbgMattingBackend(spec, on_message=on_message)