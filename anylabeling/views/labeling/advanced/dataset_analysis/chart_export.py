"""Export dataset analysis histograms to PNG files (headless matplotlib Agg)."""

from __future__ import annotations

import os
import os.path as osp
import re
from typing import Any, Dict, List, Optional, Tuple

from anylabeling.views.labeling.advanced.dataset_analysis.metrics import numeric_keys_for_charts


def _safe_filename_part(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name, flags=re.ASCII)
    return s.strip("_")[:80] or "metric"


def export_dataset_analysis_charts(
    export_dir: str,
    numeric_series: Dict[str, List[float]],
    diversity: Dict[str, Any],
    annotation_stats: Optional[Dict[str, Any]] = None,
    resolution_counts: Optional[Dict[str, int]] = None,
    *,
    title_caption_len: str = "caption character length",
) -> Tuple[List[str], Optional[str]]:
    """
    Save all metric histograms (+ caption length if present) under export_dir.

    Returns:
        (list of saved file paths, error message or None)
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure
    except ImportError as e:
        return [], str(e)

    try:
        os.makedirs(export_dir, exist_ok=True)
    except OSError as e:
        return [], str(e)

    saved: List[str] = []

    try:
        # Resolution: export a single WxH count chart; skip the 4 resolution numeric histograms.
        if isinstance(resolution_counts, dict) and resolution_counts:
            items = list(resolution_counts.items())
            max_n = 30
            shown = items[:max_n]
            other = sum(v for _, v in items[max_n:])
            if other:
                shown.append(("other", other))
            labels = [k for k, _v in shown]
            vals = [int(v) for _k, v in shown]
            import numpy as np

            x = np.arange(len(labels), dtype=np.float32)
            fig = Figure(figsize=(10, 5.0), dpi=120)
            ax = fig.add_subplot(111)
            bars = ax.bar(x, vals, width=0.6, color="#0066FF")
            ax.bar_label(bars, padding=2, fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("image count")
            ax.set_title("Resolution counts (WxH)")
            fig.tight_layout()
            path = osp.join(export_dir, "resolution_counts.png")
            fig.savefig(path, bbox_inches="tight")
            saved.append(path)

        skip_keys = {"original_width", "original_height", "megapixels", "aspect_ratio"}
        for key in numeric_keys_for_charts():
            if key in skip_keys:
                continue
            values = numeric_series.get(key) or []
            if not values:
                continue
            fig = Figure(figsize=(8, 4.5), dpi=120)
            ax = fig.add_subplot(111)
            ax.hist(
                values,
                bins=min(30, max(10, len(values) // 5)),
                color="#0066FF",
                alpha=0.85,
            )
            ax.set_title(key)
            ax.set_xlabel("value")
            ax.set_ylabel("count")
            fig.tight_layout()
            fn = f"hist_{_safe_filename_part(key)}.png"
            path = osp.join(export_dir, fn)
            fig.savefig(path, bbox_inches="tight")
            saved.append(path)

        cap_lens = diversity.get("caption_lengths") or []
        if (
            cap_lens
            and isinstance(cap_lens, list)
            and all(isinstance(x, (int, float)) for x in cap_lens)
        ):
            lens = [float(x) for x in cap_lens]
            fig = Figure(figsize=(8, 4.5), dpi=120)
            ax = fig.add_subplot(111)
            ax.hist(
                lens,
                bins=min(25, max(8, len(lens) // 4)),
                color="#00A86B",
                alpha=0.85,
            )
            ax.set_title(title_caption_len)
            ax.set_xlabel("length")
            ax.set_ylabel("count")
            fig.tight_layout()
            path = osp.join(export_dir, "hist_caption_character_length.png")
            fig.savefig(path, bbox_inches="tight")
            saved.append(path)

        # Annotation stats charts (if present)
        if isinstance(annotation_stats, dict) and annotation_stats:
            class_names = annotation_stats.get("class_names") or []
            per_class_box = annotation_stats.get("per_class_box_count") or {}
            images_total = int(annotation_stats.get("image_total") or 0)
            images_with_any = int(annotation_stats.get("images_with_any_box") or 0)
            images_zero = int(annotation_stats.get("images_with_zero_box") or 0)

            # Summary figure
            fig = Figure(figsize=(10, 4.5), dpi=120)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            bars1 = ax1.bar(
                ["total", "with_box", "zero_box"],
                [images_total, images_with_any, images_zero],
                color=["#6C757D", "#0066FF", "#FF6B6B"],
                width=0.6,
            )
            ax1.bar_label(bars1, padding=2, fontsize=9)
            ax1.set_title("Images")
            ax1.set_ylabel("count")
            if class_names:
                y = [int(per_class_box.get(c, 0)) for c in class_names]
                import numpy as np

                xs = np.arange(len(class_names), dtype=np.float32)
                bars2 = ax2.bar(xs, y, color="#00A86B", width=0.6)
                ax2.set_xticks(xs)
                ax2.set_xticklabels(class_names)
                ax2.bar_label(bars2, padding=2, fontsize=8)
                ax2.set_title("Boxes per class")
                ax2.tick_params(axis="x", labelrotation=60)
            else:
                ax2.text(0.5, 0.5, "No classes", ha="center", va="center")
                ax2.set_axis_off()
            fig.tight_layout()
            path = osp.join(export_dir, "annot_summary.png")
            fig.savefig(path, bbox_inches="tight")
            saved.append(path)

            # Boxes-per-image distribution (per class, one chart)
            per_image_counts = annotation_stats.get("per_image_class_counts") or []
            if class_names and isinstance(per_image_counts, list) and per_image_counts:
                class_totals = {c: int(per_class_box.get(c, 0)) for c in class_names}
                top_classes = [
                    c
                    for c, _v in sorted(
                        class_totals.items(), key=lambda kv: kv[1], reverse=True
                    )
                    if class_totals.get(c, 0) > 0
                ][:6]
                if top_classes:
                    max_k = 0
                    per_class_k_counts: Dict[str, Dict[int, int]] = {}
                    for c in top_classes:
                        kk: Dict[int, int] = {}
                        for d in per_image_counts:
                            v = 0
                            if isinstance(d, dict):
                                v = int(d.get(c, 0) or 0)
                            kk[v] = kk.get(v, 0) + 1
                            max_k = max(max_k, v)
                        per_class_k_counts[c] = kk
                    ks = list(range(0, max_k + 1))
                    x = np.arange(len(ks), dtype=np.float32)
                    width = 0.12
                    fig = Figure(figsize=(10, 5.0), dpi=120)
                    ax = fig.add_subplot(111)
                    palette = ["#0066FF", "#00A86B", "#845EF7", "#FF6B6B", "#F59F00", "#12B886"]
                    for idx, c in enumerate(top_classes):
                        offs = (idx - (len(top_classes) - 1) / 2.0) * width
                        ys = [int(per_class_k_counts[c].get(k, 0)) for k in ks]
                        bars = ax.bar(
                            x + offs,
                            ys,
                            width=width,
                            color=palette[idx % len(palette)],
                            label=c,
                        )
                        ax.bar_label(bars, padding=1, fontsize=8)
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(k) for k in ks])
                    ax.set_xlabel("boxes per image (k)")
                    ax.set_ylabel("image count")
                    ax.set_title("Images distribution by boxes-per-image (per class)")
                    ax.legend(loc="best", fontsize=8)
                    fig.tight_layout()
                    path = osp.join(export_dir, "annot_boxes_per_image.png")
                    fig.savefig(path, bbox_inches="tight")
                    saved.append(path)

            # Area ratio distribution bins (per class, one chart)
            ratios = annotation_stats.get("per_box_area_ratio") or {}
            if class_names and isinstance(ratios, dict):
                class_totals = {c: int(per_class_box.get(c, 0)) for c in class_names}
                top_classes = [
                    c
                    for c, _v in sorted(
                        class_totals.items(), key=lambda kv: kv[1], reverse=True
                    )
                    if class_totals.get(c, 0) > 0
                ][:6]
                if top_classes:
                    all_vals: List[float] = []
                    for c in top_classes:
                        all_vals.extend(
                            [
                                float(x)
                                for x in (ratios.get(c) or [])
                                if isinstance(x, (int, float))
                            ]
                        )
                    all_vals = [v for v in all_vals if 0.0 <= v <= 1.0]
                    if len(all_vals) >= 5:
                        qs = np.quantile(
                            np.array(all_vals),
                            [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                        )
                        edges = sorted(set([0.0, *[float(x) for x in qs], 1.0]))
                    else:
                        edges = [0.0, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
                    if len(edges) < 3:
                        edges = [0.0, 0.1, 1.0]
                    bin_labels = [
                        f"{edges[i]*100:.2f}%–{edges[i+1]*100:.2f}%"
                        for i in range(len(edges) - 1)
                    ]
                    x = np.arange(len(bin_labels), dtype=np.float32)
                    width = 0.12
                    fig = Figure(figsize=(10, 5.0), dpi=120)
                    ax = fig.add_subplot(111)
                    palette = ["#0066FF", "#00A86B", "#845EF7", "#FF6B6B", "#F59F00", "#12B886"]
                    for idx, c in enumerate(top_classes):
                        vals = [
                            float(v)
                            for v in (ratios.get(c) or [])
                            if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0
                        ]
                        hist, _ = np.histogram(
                            np.array(vals, dtype=np.float64),
                            bins=np.array(edges, dtype=np.float64),
                        )
                        offs = (idx - (len(top_classes) - 1) / 2.0) * width
                        bars = ax.bar(
                            x + offs,
                            hist.tolist(),
                            width=width,
                            color=palette[idx % len(palette)],
                            label=c,
                        )
                        ax.bar_label(bars, padding=1, fontsize=8)
                    ax.set_xticks(x)
                    ax.set_xticklabels(bin_labels, rotation=35, ha="right")
                    ax.set_xlabel("box area ratio bins")
                    ax.set_ylabel("box count")
                    ax.set_title("Box area ratio distribution (per class)")
                    ax.legend(loc="best", fontsize=8)
                    fig.tight_layout()
                    path = osp.join(export_dir, "annot_area_ratio_bins.png")
                    fig.savefig(path, bbox_inches="tight")
                    saved.append(path)

            # Centers per class
            centers = annotation_stats.get("per_class_centers") or {}
            if isinstance(centers, dict) and class_names:
                for c in class_names:
                    pts = centers.get(c) or []
                    if not isinstance(pts, list) or not pts:
                        continue
                    xs, ys = [], []
                    for p in pts:
                        if isinstance(p, dict):
                            x, y = p.get("x"), p.get("y")
                        else:
                            try:
                                x, y = p
                            except Exception:
                                continue
                        if isinstance(x, (int, float)) and isinstance(
                            y, (int, float)
                        ):
                            xs.append(float(x))
                            ys.append(float(y))
                    if not xs:
                        continue
                    fig = Figure(figsize=(7, 6), dpi=120)
                    ax = fig.add_subplot(111)
                    hb = ax.hexbin(
                        xs,
                        ys,
                        gridsize=40,
                        extent=(0, 1, 0, 1),
                        cmap="viridis",
                    )
                    ax.set_title(f"Center distribution: {c}")
                    ax.set_xlabel("cx / W")
                    ax.set_ylabel("cy / H")
                    fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
                    fig.tight_layout()
                    path = osp.join(
                        export_dir, f"annot_center_{_safe_filename_part(c)}.png"
                    )
                    fig.savefig(path, bbox_inches="tight")
                    saved.append(path)

    except Exception as e:
        return saved, str(e)

    return saved, None


def export_optimization_suggestions_txt(
    export_dir: str,
    optimization_suggestions: Optional[Dict[str, Any]],
    *,
    filename: str = "optimization_suggestions.txt",
) -> Tuple[Optional[str], Optional[str]]:
    """Write heuristic optimization suggestions to a UTF-8 text file."""
    if not isinstance(optimization_suggestions, dict):
        return None, None

    err_msg = optimization_suggestions.get("error")
    if isinstance(err_msg, str) and err_msg.strip():
        return None, err_msg.strip()

    sugg_list = optimization_suggestions.get("suggestions_zh")
    if not isinstance(sugg_list, list) or not sugg_list:
        return None, None

    lines: List[str] = []
    for item in sugg_list:
        if isinstance(item, str) and item.strip():
            lines.append(f"- {item.strip()}")
    if not lines:
        return None, None

    method = optimization_suggestions.get("method")
    if isinstance(method, str) and method.strip():
        lines.insert(0, f"# method: {method.strip()}")
        lines.insert(1, "")

    try:
        os.makedirs(export_dir, exist_ok=True)
        out_path = osp.join(export_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return out_path, None
    except OSError as e:
        return None, str(e)
