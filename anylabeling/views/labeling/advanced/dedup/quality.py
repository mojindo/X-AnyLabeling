"""L4 质量 gating：判断结构重复对是否"严格重复"。

需求语义（已与用户确认）：质量指标是收紧判重的辅助条件，不是删除低质量图。
- 两图结构一致（L2 通过）但某启用指标差异超容差 -> 不算严格重复
- 这就是截图里"亮度差 20% 以内算重复，超过算非重复"的实现

相对差异公式带 eps 兜底 + 绝对容差双判据，避免两值都接近 0 时相对差异爆炸。
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config, QualityMetric
from .features import Features


@dataclass
class GateResult:
    passed: bool
    worst_key: str = ""
    worst_diff: float = 0.0
    details: dict[str, float] | None = None


def metric_diff(v1: float, v2: float, m: QualityMetric, eps: float) -> float:
    """返回归一化差异（0 表示完全一致，>1 表示超容差）。"""
    rel = abs(v1 - v2) / (max(abs(v1), abs(v2)) + eps)
    # 两值都接近 0 时相对差异不可靠，改用绝对差 / abs_tol
    if max(abs(v1), abs(v2)) < 1e-3:
        rel = abs(v1 - v2) / max(m.abs_tol, eps)
    return rel / max(m.rel_tol, eps)


def quality_gate(f1: Features, f2: Features, cfg: Config) -> GateResult:
    qc = cfg.quality
    if not qc.enabled:
        return GateResult(True, details={})
    details: dict[str, float] = {}
    worst_key, worst = "", 0.0
    for m in qc.metrics:
        if not m.enabled:
            continue
        d = metric_diff(f1.quality.get(m.key, 0.0), f2.quality.get(m.key, 0.0), m, qc.eps)
        details[m.key] = round(d, 3)
        if d > worst:
            worst, worst_key = d, m.key
    return GateResult(worst <= 1.0, worst_key, round(worst, 3), details)
