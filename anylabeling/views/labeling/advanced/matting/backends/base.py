"""Pluggable matting backends (RMBG today; more later)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class MattingBackend(ABC):
    """Remove background and return an alpha mask for an RGB/BGR crop."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        ...

    @abstractmethod
    def predict_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Return uint8 alpha HxW aligned with ``bgr`` (0–255)."""

    def unload(self) -> None:
        pass


def combine_alpha(existing: np.ndarray, refined: np.ndarray) -> np.ndarray:
    """Multiply existing crop alpha with a new matting mask."""
    if existing.shape != refined.shape:
        raise ValueError("alpha shape mismatch")
    ex = existing.astype(np.float32)
    rv = refined.astype(np.float32)
    out = (ex * rv / 255.0).clip(0, 255).astype(np.uint8)
    return out


def apply_mask_to_bgra(bgra: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if bgra.shape[:2] != mask.shape:
        raise ValueError("mask/bgra size mismatch")
    out = bgra.copy()
    if out.shape[2] == 3:
        out = np.dstack([out, np.full(out.shape[:2], 255, dtype=np.uint8)])
    existing = out[:, :, 3]
    out[:, :, 3] = combine_alpha(existing, mask)
    return out