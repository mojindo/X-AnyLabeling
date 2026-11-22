"""Candidate recall: perceptual-hash LSH plus box-signature neighbors.

Oversized hash buckets are scanned in phash order with a fixed window
instead of being dropped. Box signatures use an exact cosine search only
while the set is small; larger sets use random-projection buckets.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import Config
from .features import Features


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def hash_pass(f1: Features, f2: Features, cfg: Config) -> bool:
    fc = cfg.feature
    ph = hamming(f1.phash, f2.phash) <= fc.phash_hamming_max
    dh = hamming(f1.dhash, f2.dhash) <= fc.dhash_hamming_max
    return (ph and dh) if fc.hash_require_both else (ph or dh)


def _add_hash_pairs(ids: list[int], feats: list[Features], cfg: Config, pairs: set) -> None:
    if len(ids) < 2:
        return
    if len(ids) <= 300:
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = sorted((ids[i], ids[j]))
                if hash_pass(feats[a], feats[b], cfg):
                    pairs.add((a, b))
        return
    ordered = sorted(ids, key=lambda i: feats[i].phash)
    window = 48
    for i in range(len(ordered)):
        stop = min(len(ordered), i + 1 + window)
        for j in range(i + 1, stop):
            a, b = sorted((ordered[i], ordered[j]))
            if hash_pass(feats[a], feats[b], cfg):
                pairs.add((a, b))


def hash_lsh_candidates(feats: list[Features], cfg: Config) -> set[tuple[int, int]]:
    bands = cfg.recall.lsh_bands
    bits = cfg.feature.hash_bits
    per = bits // bands
    mask = (1 << per) - 1
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for f in feats:
        for i in range(bands):
            key = (i, (f.phash >> (i * per)) & mask)
            buckets[key].append(f.index)
    pairs: set[tuple[int, int]] = set()
    for ids in buckets.values():
        _add_hash_pairs(ids, feats, cfg, pairs)
    return pairs


def _topk_from_sims(idxs: list[int], sims: np.ndarray, cfg: Config) -> set[tuple[int, int]]:
    k = min(cfg.recall.box_topk, len(idxs) - 1)
    pairs: set[tuple[int, int]] = set()
    if k <= 0:
        return pairs
    for r in range(len(idxs)):
        top = np.argpartition(sims[r], -k)[-k:]
        for c in top:
            if r == int(c):
                continue
            if float(sims[r, c]) >= cfg.recall.box_sim_min:
                a, b = sorted((idxs[r], int(idxs[c])))
                pairs.add((a, b))
    return pairs


def box_signature_candidates(feats: list[Features], cfg: Config) -> set[tuple[int, int]]:
    idxs = [f.index for f in feats if f.box_sig is not None]
    if len(idxs) < 2:
        return set()
    mat = np.stack([feats[i].box_sig for i in idxs]).astype(np.float32)
    if len(idxs) <= cfg.recall.box_exact_max:
        sims = mat @ mat.T
        np.fill_diagonal(sims, -1.0)
        return _topk_from_sims(idxs, sims, cfg)

    rng = np.random.default_rng(0)
    dim = mat.shape[1]
    pairs: set[tuple[int, int]] = set()
    tables = 4
    nbits = 8
    for _table in range(tables):
        planes = rng.normal(size=(nbits, dim)).astype(np.float32)
        planes /= np.linalg.norm(planes, axis=1, keepdims=True) + 1e-8
        bits = mat @ planes.T > 0
        weights = 1 << np.arange(nbits, dtype=np.int64)
        keys = bits.astype(np.int64) @ weights
        buckets: dict[int, list[int]] = defaultdict(list)
        for row, key in enumerate(keys.tolist()):
            buckets[int(key)].append(row)
        for rows in buckets.values():
            if len(rows) < 2 or len(rows) > 400:
                continue
            sub = mat[rows]
            sims = sub @ sub.T
            np.fill_diagonal(sims, -1.0)
            local = [idxs[r] for r in rows]
            pairs |= _topk_from_sims(local, sims, cfg)
    return pairs


def collect_candidates(feats: list[Features], cfg: Config) -> tuple[set[tuple[int, int]], dict[str, int]]:
    h = hash_lsh_candidates(feats, cfg)
    b = box_signature_candidates(feats, cfg)
    all_pairs = h | b
    stats = {"hash": len(h), "box": len(b), "union": len(all_pairs)}
    cap = cfg.recall.max_candidates_per_image
    if cap > 0:
        degree: dict[int, int] = defaultdict(int)
        kept: set[tuple[int, int]] = set()
        for a, b_ in sorted(all_pairs):
            if degree[a] < cap and degree[b_] < cap:
                kept.add((a, b_))
                degree[a] += 1
                degree[b_] += 1
        stats["after_cap"] = len(kept)
        return kept, stats
    return all_pairs, stats