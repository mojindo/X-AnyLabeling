"""Ultralytics YOLO ``model.predict`` options for MyTrain inference."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple


def parse_classes_value(text: str) -> Optional[Tuple[int, ...]]:
    raw = (text or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in re.split(r"[,;\s]+", raw) if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            continue
    return tuple(out) if out else None


def parse_predict_cli(text: str) -> Dict[str, Any]:
    """Parse ``key=value`` tokens (space-separated) into predict kwargs."""
    out: Dict[str, Any] = {}
    for token in (text or "").split():
        token = token.strip()
        if not token or "=" not in token:
            continue
        key, _, val = token.partition("=")
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        if key == "classes":
            parsed = parse_classes_value(val)
            if parsed is not None:
                out[key] = list(parsed)
            continue
        low = val.lower()
        if low in ("true", "yes", "1"):
            out[key] = True
        elif low in ("false", "no", "0"):
            out[key] = False
        else:
            try:
                if "." in val or "e" in low:
                    out[key] = float(val)
                else:
                    out[key] = int(val)
            except ValueError:
                out[key] = val
    return out


@dataclass
class PredictOptions:
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300
    classes: Optional[Tuple[int, ...]] = None
    agnostic_nms: bool = False
    save: bool = True
    save_txt: bool = True
    save_conf: bool = False
    save_crop: bool = False
    save_frames: bool = False
    show: bool = False
    show_labels: bool = True
    show_conf: bool = True
    show_boxes: bool = True
    line_width: Optional[int] = None
    visualize: bool = False
    rect: bool = False
    verbose: bool = True
    exist_ok: bool = True
    custom_cli: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_labels_only(cls, *, imgsz: int = 640) -> "PredictOptions":
        """Training-tab infer: txt labels only, no annotated images."""
        return cls(
            imgsz=int(imgsz),
            save=False,
            save_txt=True,
            save_conf=False,
            save_crop=False,
            show=False,
        )

    def with_cli_overrides(self) -> "PredictOptions":
        if not (self.custom_cli or "").strip():
            return self
        names = {f.name for f in fields(self)} - {"extra"}
        merged = {name: getattr(self, name) for name in names}
        extra = dict(self.extra)
        cli = parse_predict_cli(self.custom_cli)
        for key, val in cli.items():
            if key in merged:
                merged[key] = val
            else:
                extra[key] = val
        return PredictOptions(**merged, extra=extra)

    def predict_kwargs(self, **fixed: Any) -> Dict[str, Any]:
        """Keyword arguments for ``YOLO.predict`` (after CLI merge)."""
        opts = self.with_cli_overrides()
        kw: Dict[str, Any] = {
            "imgsz": int(opts.imgsz),
            "conf": float(opts.conf),
            "iou": float(opts.iou),
            "max_det": int(opts.max_det),
            "agnostic_nms": bool(opts.agnostic_nms),
            "save": bool(opts.save),
            "save_txt": bool(opts.save_txt),
            "save_conf": bool(opts.save_conf),
            "save_crop": bool(opts.save_crop),
            "save_frames": bool(opts.save_frames),
            "show": bool(opts.show),
            "show_labels": bool(opts.show_labels),
            "show_conf": bool(opts.show_conf),
            "show_boxes": bool(opts.show_boxes),
            "visualize": bool(opts.visualize),
            "rect": bool(opts.rect),
            "verbose": bool(opts.verbose),
            "exist_ok": bool(opts.exist_ok),
        }
        if opts.classes:
            kw["classes"] = list(opts.classes)
        if opts.line_width is not None and int(opts.line_width) > 0:
            kw["line_width"] = int(opts.line_width)
        kw.update(opts.extra)
        kw.update(fixed)
        return kw

    def render_predict_call(
        self,
        *,
        source_expr: str,
        project_expr: str = "run_dir",
        name_expr: str = "'predict'",
        indent: str = "    ",
    ) -> str:
        kw = self.predict_kwargs()
        lines = [f"{indent}model.predict("]
        lines.append(f"{indent}    source={source_expr},")
        lines.append(f"{indent}    project={project_expr},")
        lines.append(f"{indent}    name={name_expr},")
        for key, val in kw.items():
            if key in ("source", "project", "name"):
                continue
            lines.append(f"{indent}    {key}={repr(val)},")
        lines.append(f"{indent})")
        return "\n".join(lines)