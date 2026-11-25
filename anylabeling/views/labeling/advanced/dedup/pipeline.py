"""Serial near-duplicate funnel for the images currently open in the app."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .cluster import build_dedup_groups
from .config import Config
from .features import Features, extract_one
from .instance import InstanceResult, judge_instance
from .io_utils import ImageRecord, decode_thumb, decode_verify, fullres_pixel_md5
from .motion import view_motion
from .quality import quality_gate
from .recall import collect_candidates
from .verify import VerifyResult, verify_pair


@dataclass
class ScanResult:
    recs: list[ImageRecord]
    groups: list[list[int]]
    gate_stats: dict[str, int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def group_paths(self) -> list[list[str]]:
        out = []
        for group in self.groups:
            if len(group) < 2:
                continue
            out.append([self.recs[i].image_path for i in group])
        return out


def run_records(cfg: Config, recs: list[ImageRecord], progress=None) -> ScanResult:
    t0 = time.time()
    timings: dict[str, float] = {}

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    labels = sorted({b.label for r in recs for b in r.boxes})
    filt = cfg.feature.box_label_filter
    if cfg.strict.use_instance_gate and filt:
        cfg.feature.box_classes = tuple(c for c in labels if c in filt)
    else:
        cfg.feature.box_classes = tuple(labels)
    log(
        f"labels: {list(cfg.feature.box_classes)} "
        f"(instance_gate={cfg.strict.use_instance_gate})"
    )

    t = time.time()
    feats: list[Features] = []
    total = len(recs)
    for i, rec in enumerate(recs):
        feats.append(extract_one(rec, cfg, decode_thumb(rec.image_path, cfg.feature.thumb_max_side)))
        if progress and (i + 1) % 20 == 0:
            log(f"features {i + 1}/{total}")
    timings["features"] = round(time.time() - t, 2)

    t = time.time()
    by_byte: dict[str, list[int]] = defaultdict(list)
    by_thumb: dict[str, list[int]] = defaultdict(list)
    for feat in feats:
        by_byte[recs[feat.index].image_md5].append(feat.index)
        by_thumb[feat.pixel_hash].append(feat.index)
    exact: dict[int, list[int]] = {}
    for group in by_byte.values():
        if len(group) > 1:
            for i in group:
                exact[i] = group
    fullres_cache: dict[int, str] = {}
    for group in by_thumb.values():
        if len(group) < 2 or all(i in exact for i in group):
            continue
        sub: dict[str, list[int]] = defaultdict(list)
        for i in group:
            if i not in fullres_cache:
                fullres_cache[i] = fullres_pixel_md5(recs[i].image_path)
            sub[fullres_cache[i]].append(i)
        for sg in sub.values():
            if len(sg) > 1:
                for i in sg:
                    exact[i] = sg
    uniq: dict[tuple[int, ...], list[int]] = {}
    for group in exact.values():
        key = tuple(sorted(set(group)))
        if len(key) > 1:
            uniq[key] = list(key)
    exact_groups = list(uniq.values())
    timings["L0"] = round(time.time() - t, 2)
    log(f"exact groups {len(exact_groups)}")

    t = time.time()
    candidates, stats = collect_candidates(feats, cfg)
    timings["recall"] = round(time.time() - t, 2)
    log(f"candidates {stats.get('after_cap', stats.get('union', 0))}")

    t = time.time()
    pairs_sorted = sorted(candidates)
    vcache: dict[tuple[int, int], VerifyResult] = {}
    feat_cache: dict[int, tuple] = {}
    for n, (a, b) in enumerate(pairs_sorted):
        vcache[(a, b)] = verify_pair(recs[a], recs[b], cfg, cache=feat_cache)
        if progress and (n + 1) % 20 == 0:
            log(f"scene {n + 1}/{len(pairs_sorted)}")
    g1_pass = [(a, b) for a, b in pairs_sorted if vcache[(a, b)].passed]
    timings["G1"] = round(time.time() - t, 2)

    gate_stats = {
        "images": len(recs),
        "L0_exact_groups": len(exact_groups),
        "L1_candidates": len(pairs_sorted),
        "G1_scene_match": len(g1_pass),
        "G3_no_boxes_bypass": 0,
        "G3_instance_presence_mismatch": 0,
    }
    g2 = cfg.strict.gate2
    g2_pass = []
    for a, b in g1_pass:
        motion = view_motion(vcache[(a, b)])
        if motion is None:
            continue
        shift, scale, rotation = motion
        if shift >= g2.max_translation or scale >= g2.max_scale or rotation >= g2.max_rotation_rad:
            continue
        g2_pass.append((a, b))
    gate_stats["G2_small_view_change"] = len(g2_pass)

    rgb_cache: dict[int, np.ndarray] = {}

    def get_rgb(i: int) -> np.ndarray:
        if i not in rgb_cache:
            rgb_cache[i] = decode_verify(recs[i].image_path, cfg.feature.verify_max_side)
        return rgb_cache[i]

    g3_pass = []
    if cfg.strict.use_instance_gate:
        for a, b in g2_pass:
            judged: InstanceResult = judge_instance(
                recs[a], recs[b], vcache[(a, b)], get_rgb(a), get_rgb(b), cfg
            )
            if judged.reason == "no_boxes_both_pass":
                gate_stats["G3_no_boxes_bypass"] += 1
            elif judged.reason == "instance_presence_mismatch":
                gate_stats["G3_instance_presence_mismatch"] += 1
            if judged.passed:
                g3_pass.append((a, b))
    else:
        g3_pass = list(g2_pass)
        gate_stats["G3_instance_skipped"] = len(g3_pass)
    gate_stats["G3_instance_same"] = len(g3_pass)

    g4_pass = []
    for a, b in g3_pass:
        ok = (not cfg.strict.require_quality) or quality_gate(feats[a], feats[b], cfg).passed
        if ok:
            g4_pass.append((a, b))
    gate_stats["G4_quality_ok"] = len(g4_pass)

    groups = build_dedup_groups(len(recs), exact_groups, g4_pass)
    groups = [g for g in groups if len(g) >= 2]
    gate_stats["groups"] = len(groups)
    timings["total"] = round(time.time() - t0, 2)
    return ScanResult(recs, groups, gate_stats, timings)