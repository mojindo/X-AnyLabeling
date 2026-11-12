"""Aggregate per-image analysis for charts and dataset-level summaries."""

from __future__ import annotations

import json
import math
import os.path as osp
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from anylabeling.views.labeling.advanced.dataset_analysis.analysis_store import ANALYSIS_KEY
from anylabeling.views.labeling.advanced.dataset_analysis.metrics import (
    numeric_keys_for_charts,
)

def collect_resolution_counts(label_json_paths: List[str]) -> Dict[str, int]:
    """Count occurrences of each WxH resolution from per-image metrics."""
    counts: Dict[str, int] = {}
    for path in label_json_paths:
        data = _load_json(path)
        if not data:
            continue
        m = extract_metrics_from_label_data(data)
        if not m:
            continue
        w = m.get("original_width")
        h = m.get("original_height")
        try:
            wi = int(w)
            hi = int(h)
        except Exception:
            continue
        if wi <= 0 or hi <= 0:
            continue
        key = f"{wi}x{hi}"
        counts[key] = counts.get(key, 0) + 1
    # Stable-ish order: by count desc then name
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def extract_metrics_from_label_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    block = data.get(ANALYSIS_KEY)
    if not isinstance(block, dict):
        return None
    m = block.get("metrics")
    return m if isinstance(m, dict) else None


def aggregate_numeric_metrics(
    label_json_paths: List[str],
) -> Tuple[Dict[str, List[float]], int]:
    """
    Collect numeric metric lists from label JSON files.
    Skips None / missing values.
    """
    keys = numeric_keys_for_charts()
    series: Dict[str, List[float]] = {k: [] for k in keys}
    count = 0
    for path in label_json_paths:
        data = _load_json(path)
        if not data:
            continue
        m = extract_metrics_from_label_data(data)
        if not m:
            continue
        count += 1
        for k in keys:
            v = m.get(k)
            if v is None:
                continue
            try:
                series[k].append(float(v))
            except (TypeError, ValueError):
                continue
    return series, count


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[\w\u4e00-\u9fff]+", text)


def build_optimization_suggestions(
    *,
    numeric_series: Dict[str, List[float]],
    resolution_counts: Dict[str, int],
    diversity: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Heuristic dataset optimization suggestions based on metric distributions
    and caption/cluster diversity.
    """

    def _safe_stats(xs: List[float]) -> Optional[Dict[str, float]]:
        if not xs:
            return None
        arr = np.array(xs, dtype=np.float64)
        if arr.size <= 0:
            return None
        return {
            "n": float(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)) if arr.size >= 2 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    suggestions: List[str] = []
    evidence: Dict[str, Any] = {}

    # Resolution diversity
    if resolution_counts:
        total = int(sum(resolution_counts.values()))
        uniq = int(len(resolution_counts))
        top_res, top_cnt = next(iter(resolution_counts.items()))
        top_ratio = float(top_cnt) / float(total) if total > 0 else 0.0
        evidence["resolution"] = {
            "unique": uniq,
            "top": top_res,
            "top_ratio": top_ratio,
        }
        if uniq <= 2 or top_ratio >= 0.85:
            suggestions.append(
                f"分辨率较单一（Top 为 {top_res}，占比 {top_ratio:.1%}）。"
                "建议补充不同分辨率/不同拍摄距离的数据，或在训练中加入多尺度增强。"
            )

    # Aspect ratio diversity
    ar = numeric_series.get("aspect_ratio") or []
    st_ar = _safe_stats(ar)
    if st_ar:
        evidence["aspect_ratio"] = st_ar
        if st_ar["std"] <= 0.05:
            suggestions.append(
                f"宽高比波动很小（std≈{st_ar['std']:.3f}）。"
                "建议补充不同横竖构图/不同裁剪比例的样本，以增强泛化。"
            )

    # Brightness / contrast concentration
    b = numeric_series.get("brightness_l_mean") or []
    c = numeric_series.get("contrast_rms_gray") or []
    st_b = _safe_stats(b)
    st_c = _safe_stats(c)
    if st_b:
        evidence["brightness_l_mean"] = st_b
        if st_b["std"] <= 8.0:
            suggestions.append(
                f"亮度分布较集中（L 均值 std≈{st_b['std']:.2f}）。"
                "建议补充更暗/更亮的样本，或在训练中增强曝光/伽马扰动。"
            )
        if st_b["mean"] <= 60:
            suggestions.append(
                f"整体偏暗（L 均值≈{st_b['mean']:.1f}）。建议补充白天/强光样本或做亮度增强。"
            )
        if st_b["mean"] >= 180:
            suggestions.append(
                f"整体偏亮（L 均值≈{st_b['mean']:.1f}）。建议补充弱光样本或做曝光抑制增强。"
            )
    if st_c:
        evidence["contrast_rms_gray"] = st_c
        if st_c["std"] <= 10.0:
            suggestions.append(
                f"对比度分布较集中（RMS std≈{st_c['std']:.2f}）。"
                "建议补充低对比/高对比场景（逆光、雾霾、夜景等），或加入对比度增强。"
            )

    # Quality metrics concentration (sharpness / brisque)
    sh = numeric_series.get("sharpness_laplacian_var_log1p") or []
    br = numeric_series.get("brisque") or []
    st_sh = _safe_stats(sh)
    st_br = _safe_stats(br)
    if st_sh:
        evidence["sharpness_laplacian_var_log1p"] = st_sh
        if st_sh["std"] <= 0.35:
            suggestions.append(
                f"清晰度分布较集中（std≈{st_sh['std']:.2f}）。"
                "建议补充轻微模糊/运动模糊/失焦样本，或加入模糊增强以提升鲁棒性。"
            )
    if st_br and st_br["n"] >= 3:
        evidence["brisque"] = st_br
        if st_br["mean"] >= 40:
            suggestions.append(
                f"整体无参考质量偏低（BRISQUE 均值≈{st_br['mean']:.1f}，越高越差）。"
                "建议优先清洗严重压缩/噪声/模糊样本，或单独标注为低质域进行域适配。"
            )

    # Scene diversity via caption clustering + keyword coverage
    captions = diversity.get("caption_count", 0) or 0
    cluster_info = diversity.get("caption_tfidf_kmeans") or {}
    if isinstance(cluster_info, dict) and cluster_info.get("cluster_sizes"):
        sizes = cluster_info.get("cluster_sizes") or []
        try:
            sizes_i = [int(x) for x in sizes if int(x) >= 0]
        except Exception:
            sizes_i = []
        if sizes_i:
            tot = sum(sizes_i)
            mx = max(sizes_i) if tot > 0 else 0
            mx_ratio = float(mx) / float(tot) if tot > 0 else 0.0
            evidence["caption_clusters"] = {
                "n_clusters": int(cluster_info.get("n_clusters", 0) or 0),
                "max_cluster_ratio": mx_ratio,
            }
            if mx_ratio >= 0.7:
                suggestions.append(
                    f"场景描述聚类较集中（最大簇占比≈{mx_ratio:.1%}）。"
                    "建议补充更多场景类型/背景变化的数据，降低场景偏置。"
                )

    # Keyword coverage from center captions (best-effort)
    centers = []
    if isinstance(cluster_info, dict):
        centers = cluster_info.get("center_captions") or []
    if isinstance(centers, list) and centers:
        text = " ".join([t for t in centers if isinstance(t, str)])
        t_low = text.lower()
        groups = {
            "昼夜/时段": [
                "白天",
                "夜",
                "夜晚",
                "黄昏",
                "傍晚",
                "清晨",
                "黎明",
                "日落",
                "日出",
            ],
            "天气": ["雨", "下雨", "雪", "雾", "雾霾", "大雾", "阴天", "晴", "多云"],
            "光照条件": ["逆光", "强光", "弱光", "暗", "曝光", "过曝", "欠曝"],
        }
        missing = []
        present = {}
        for g, kws in groups.items():
            hit = any(k in t_low for k in kws)
            present[g] = bool(hit)
            if not hit:
                missing.append(g)
        evidence["caption_keyword_coverage"] = present
        if captions and missing:
            suggestions.append(
                "从场景描述代表文本看，可能缺少以下多样性维度："
                + "、".join(missing)
                + "。建议补充对应场景（如雨/雪/雾、黑天白夜、不同光照）以提升泛化。"
            )

    if not suggestions:
        suggestions.append(
            "当前指标未发现明显单一性/偏置（仅基于启发式统计）。可继续通过抽样人工复核代表性与标注一致性。"
        )

    return {
        "suggestions_zh": suggestions,
        "evidence": evidence,
        "method": "heuristic_v1",
    }


def build_diversity_summary(
    label_json_paths: List[str],
) -> Dict[str, Any]:
    """
    Dataset-level caption / hue diversity; TF-IDF+KMeans when sklearn is installed.
    """
    caption_items: List[Dict[str, Any]] = []
    hue_values: List[float] = []
    for path in label_json_paths:
        data = _load_json(path)
        if not data:
            continue
        image_path_field = data.get("imagePath") or data.get("image_path") or ""
        block = data.get(ANALYSIS_KEY)
        if not isinstance(block, dict):
            continue
        vlm = block.get("vlm")
        if isinstance(vlm, dict):
            cap = vlm.get("caption")
            if isinstance(cap, str) and cap.strip():
                caption_items.append(
                    {
                        "label_json_path": path,
                        "imagePath": image_path_field,
                        "caption": cap.strip(),
                    }
                )
        m = block.get("metrics")
        if isinstance(m, dict) and m.get("hue_mean_deg") is not None:
            try:
                hue_values.append(float(m["hue_mean_deg"]))
            except (TypeError, ValueError):
                pass

    captions = [it["caption"] for it in caption_items]
    n = len(captions)
    unique = len(set(captions)) if n else 0
    dedup_ratio = (unique / n) if n else 0.0

    all_words: List[str] = []
    for c in captions:
        all_words.extend(_tokenize(c))
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(12)

    hue_std = None
    if len(hue_values) >= 2:
        hue_std = float(np.std(np.array(hue_values, dtype=np.float64)))

    caption_lengths = [len(c) for c in captions]

    cluster_info = _optional_caption_tfidf_kmeans(captions)
    scene_clusters: List[Dict[str, Any]] = []
    if isinstance(cluster_info, dict) and "labels" in cluster_info:
        labels = cluster_info.get("labels") or []
        try:
            labels_i = [int(x) for x in labels]
        except Exception:
            labels_i = []
        if labels_i and len(labels_i) == len(caption_items):
            k = int(cluster_info.get("n_clusters", 0) or 0)
            centers = cluster_info.get("center_captions") or []
            for ci in range(k):
                imgs = []
                for j, lab in enumerate(labels_i):
                    if lab != ci:
                        continue
                    item = caption_items[j]
                    imgs.append(
                        {
                            "imagePath": item.get("imagePath") or "",
                            "label_json_path": item.get("label_json_path") or "",
                            "caption": item.get("caption") or "",
                        }
                    )
                scene_clusters.append(
                    {
                        "cluster_id": ci,
                        "count": len(imgs),
                        "center_caption": centers[ci] if ci < len(centers) else "",
                        "images": imgs,
                    }
                )

    summary_zh = (
        f"共 {len(label_json_paths)} 个标注文件；"
        f"有效图像描述 {n} 条，去重率 {dedup_ratio:.2%}；"
        f"色调角度标准差约为 {hue_std:.2f}°（越大通常表示场景色彩分布更分散）。"
        if hue_std is not None
        else f"共 {len(label_json_paths)} 个标注文件；"
        f"有效图像描述 {n} 条，去重率 {dedup_ratio:.2%}。"
    )
    if cluster_info:
        k = int(cluster_info.get("n_clusters", 0) or 0)
        summary_zh += f" 基于 TF-IDF 的 KMeans 场景簇数：{k}。"
        centers = cluster_info.get("center_captions") or []
        if isinstance(centers, list) and centers:
            parts = []
            for i, t in enumerate(centers[: min(8, len(centers))], start=1):
                if not isinstance(t, str) or not t.strip():
                    continue
                s = t.strip().replace("\n", " ")
                if len(s) > 40:
                    s = s[:40] + "…"
                parts.append(f"{i}) {s}")
            if parts:
                summary_zh += " 聚类中心文本：" + "；".join(parts) + "。"

    out: Dict[str, Any] = {
        "caption_count": n,
        "caption_unique_count": unique,
        "caption_deduplication_ratio": dedup_ratio,
        "caption_lengths": caption_lengths,
        "top_caption_words": [{"word": w, "count": c} for w, c in top_words],
        "hue_mean_deg_std": hue_std,
        "summary_zh": summary_zh,
        "scene_clusters": scene_clusters,
    }
    if cluster_info:
        out["caption_tfidf_kmeans"] = cluster_info
    return out


def _optional_caption_tfidf_kmeans(
    captions: List[str],
) -> Optional[Dict[str, Any]]:
    """Optional TF-IDF + KMeans when scikit-learn is installed (plan)."""
    if len(captions) < 4:
        return None
    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return None
    try:
        n_doc = len(captions)
        k = max(2, min(8, int(round(math.sqrt(n_doc)))))
        k = min(k, n_doc - 1)
        vec = TfidfVectorizer(max_features=min(500, max(50, n_doc * 5)))
        X = vec.fit_transform(captions)
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(X)
        sizes = np.bincount(labels, minlength=k).tolist()

        # Build "center text" per cluster: pick the caption nearest to centroid.
        centers = km.cluster_centers_
        center_captions: List[str] = []
        center_keywords: List[List[str]] = []
        try:
            vocab = vec.get_feature_names_out()
        except Exception:
            vocab = None
        for ci in range(k):
            idxs = np.where(labels == ci)[0]
            if idxs.size <= 0:
                center_captions.append("")
                center_keywords.append([])
                continue
            sims = cosine_similarity(
                X[idxs],
                centers[ci].reshape(1, -1),
            ).ravel()
            best_local = int(np.argmax(sims))
            best_idx = int(idxs[best_local])
            center_captions.append(captions[best_idx])

            if vocab is not None:
                w = centers[ci]
                top = np.argsort(w)[::-1][:8]
                kws = [str(vocab[j]) for j in top if w[j] > 0]
                center_keywords.append(kws)
            else:
                center_keywords.append([])
        return {
            "n_clusters": k,
            "cluster_sizes": sizes,
            "labels": labels.tolist() if hasattr(labels, "tolist") else list(labels),
            "center_captions": center_captions,
            "center_keywords": center_keywords,
            "method": "tfidf_kmeans",
        }
    except Exception:
        return None


def write_summary_file(output_dir: str, summary: Dict[str, Any]) -> str:
    path = osp.join(output_dir, "xal_dataset_analysis_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path
