"""Probe remote NVIDIA GPUs and suggest safe training batch size."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Leave this much free VRAM for other users / system (MiB).
RESERVED_MIB_FOR_OTHERS = 1024
# Absolute minimum free VRAM (after reserve) required to start training.
MIN_USABLE_MIB = 1536

# Rough per-sample training VRAM at imgsz=640 for Detect (MiB / batch-item),
# plus a fixed model+overhead base. Tuned conservatively for YOLOv8.
MODEL_VRAM_PROFILES = {
    "n": {"base": 700, "per": 180},
    "s": {"base": 1100, "per": 260},
    "m": {"base": 1800, "per": 380},
    "l": {"base": 2600, "per": 500},
    "x": {"base": 3600, "per": 650},
    "default": {"base": 1400, "per": 300},
}


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    util_gpu: int = 0
    util_mem: int = 0

    @property
    def usable_mib(self) -> int:
        return max(0, int(self.free_mib) - RESERVED_MIB_FOR_OTHERS)


def mib_to_gb(mib: float, digits: int = 1) -> str:
    """Format MiB as decimal GB (GiB), e.g. 24576 -> '24.0'."""
    try:
        return f"{float(mib) / 1024.0:.{digits}f}"
    except (TypeError, ValueError):
        return "0.0"


@dataclass
class GpuProbeResult:
    ok: bool
    error: str = ""
    gpus: List[GpuInfo] = field(default_factory=list)
    selected_index: Optional[int] = None
    recommend_batch: int = 1
    recommend_imgsz: int = 640
    allow_train: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "gpus": [asdict(g) for g in self.gpus],
            "selected_index": self.selected_index,
            "recommend_batch": self.recommend_batch,
            "recommend_imgsz": self.recommend_imgsz,
            "allow_train": self.allow_train,
            "reason": self.reason,
        }

    def summary_text(self) -> str:
        if not self.ok:
            return f"GPU probe failed: {self.error}"
        if not self.gpus:
            return "No NVIDIA GPU detected on remote host"
        lines = []
        for g in self.gpus:
            used_gb = mib_to_gb(g.used_mib)
            free_gb = mib_to_gb(g.free_mib)
            total_gb = mib_to_gb(g.total_mib)
            usable_gb = mib_to_gb(g.usable_mib)
            lines.append(
                f"GPU{g.index} {g.name}: "
                f"used {used_gb} GB / total {total_gb} GB, "
                f"free {free_gb} GB (util {g.util_gpu}%), "
                f"usable~{usable_gb} GB"
            )
        if self.allow_train:
            lines.append(
                f"Selected GPU{self.selected_index}: "
                f"suggested batch={self.recommend_batch}, imgsz={self.recommend_imgsz}"
            )
        else:
            lines.append(f"Training blocked: {self.reason}")
        return "\n".join(lines)


def infer_model_size_key(model_path: str) -> str:
    name = (model_path or "").lower()
    # yolov8n / yolov8s.pt / yolo11m etc.
    m = re.search(r"yolo[v]?\d*([nslmx])\b", name)
    if m:
        return m.group(1)
    for key in ("n", "s", "m", "l", "x"):
        if f"yolo{key}" in name or f"_{key}." in name or f"-{key}." in name:
            return key
    return "default"


def estimate_batch(
    usable_mib: int,
    model_path: str = "",
    imgsz: int = 640,
    max_batch: int = 16,
) -> Tuple[int, int, str]:
    """
    Return (batch, imgsz, reason).
    May lower imgsz to 512/416 if VRAM is tight but still above minimum.
    """
    profile = MODEL_VRAM_PROFILES.get(
        infer_model_size_key(model_path), MODEL_VRAM_PROFILES["default"]
    )
    scale = (float(imgsz) / 640.0) ** 2
    base = int(profile["base"] * scale)
    per = max(80, int(profile["per"] * scale))

    if usable_mib < MIN_USABLE_MIB:
        return 0, imgsz, (
            f"Usable VRAM {mib_to_gb(usable_mib)} GB < minimum "
            f"{mib_to_gb(MIN_USABLE_MIB)} GB "
            f"(free minus {mib_to_gb(RESERVED_MIB_FOR_OTHERS)} GB reserve)"
        )

    # Prefer keeping imgsz; if batch would be 0, try smaller imgsz.
    for try_imgsz in (imgsz, 512, 416):
        sc = (float(try_imgsz) / 640.0) ** 2
        b = int(profile["base"] * sc)
        p = max(80, int(profile["per"] * sc))
        remain = usable_mib - b
        if remain < p:
            continue
        batch = max(1, min(max_batch, remain // p))
        # Cap batch so we leave a little slack (~10%)
        while batch > 1 and (b + p * batch) > int(usable_mib * 0.9):
            batch -= 1
        if batch >= 1 and (b + p * batch) <= usable_mib:
            reason = (
                f"usable={mib_to_gb(usable_mib)} GB, "
                f"estimate base={mib_to_gb(b)}+{mib_to_gb(p)}*batch GB, "
                f"batch={batch}, imgsz={try_imgsz}"
            )
            return batch, try_imgsz, reason

    return 0, imgsz, (
        f"Usable VRAM {mib_to_gb(usable_mib)} GB cannot fit minimum batch=1 "
        f"(need ~{mib_to_gb(base + per)} GB at imgsz={imgsz})"
    )


def parse_nvidia_smi_csv(text: str) -> List[GpuInfo]:
    """
    Parse output of:
      nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,
                  utilization.gpu,utilization.memory --format=csv,noheader,nounits
    """
    gpus: List[GpuInfo] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            total = int(float(parts[2]))
            used = int(float(parts[3]))
            free = int(float(parts[4]))
            util_gpu = int(float(parts[5])) if len(parts) > 5 else 0
            util_mem = int(float(parts[6])) if len(parts) > 6 else 0
        except (ValueError, IndexError):
            continue
        gpus.append(
            GpuInfo(
                index=idx,
                name=name,
                total_mib=total,
                used_mib=used,
                free_mib=free,
                util_gpu=util_gpu,
                util_mem=util_mem,
            )
        )
    return gpus


NVIDIA_SMI_QUERY = (
    "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,"
    "utilization.gpu,utilization.memory --format=csv,noheader,nounits"
)


def probe_gpus_from_smi_text(
    text: str,
    *,
    model_path: str = "",
    imgsz: int = 640,
) -> GpuProbeResult:
    gpus = parse_nvidia_smi_csv(text)
    if not gpus:
        return GpuProbeResult(
            ok=False,
            error="Failed to parse nvidia-smi output (no GPUs?)",
            allow_train=False,
            reason="No GPU data",
        )

    # Prefer GPU with most usable free memory, then lowest util
    ordered = sorted(
        gpus, key=lambda g: (g.usable_mib, -g.util_gpu), reverse=True
    )
    best = ordered[0]
    batch, out_imgsz, reason = estimate_batch(
        best.usable_mib, model_path=model_path, imgsz=imgsz
    )
    allow = batch >= 1
    return GpuProbeResult(
        ok=True,
        gpus=gpus,
        selected_index=best.index if allow else None,
        recommend_batch=batch if allow else 0,
        recommend_imgsz=out_imgsz,
        allow_train=allow,
        reason=reason if allow else reason,
    )


def probe_remote_gpus(
    ssh_client,
    *,
    model_path: str = "",
    imgsz: int = 640,
) -> GpuProbeResult:
    """Run nvidia-smi on remote host and build recommendation."""
    try:
        code, out, err = ssh_client.exec(NVIDIA_SMI_QUERY, timeout=30)
    except Exception as e:
        return GpuProbeResult(
            ok=False,
            error=str(e),
            allow_train=False,
            reason="SSH exec failed",
        )
    if code != 0:
        return GpuProbeResult(
            ok=False,
            error=(err or out or f"nvidia-smi exit {code}").strip(),
            allow_train=False,
            reason="nvidia-smi failed (driver/CUDA missing?)",
        )
    return probe_gpus_from_smi_text(
        out or "", model_path=model_path, imgsz=imgsz
    )
