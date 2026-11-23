"""Train / val / test split with optional label-stratified assignment."""

from __future__ import annotations

import json
import os.path as osp
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from anylabeling.views.labeling.classifier.utils import get_label_file_path

UNLABELED = "__unlabeled__"


def split_sizes(n: int, w_train: int, w_val: int, w_test: int) -> Tuple[int, int, int]:
    total_w = w_train + w_val + w_test
    if total_w <= 0:
        raise ValueError("invalid weights")
    n_train = n * w_train // total_w
    n_val = n * w_val // total_w
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class ImageLabelInfo:
    path: str
    labels: frozenset[str]
    box_counts: Tuple[Tuple[str, int], ...]

    @property
    def label_set(self) -> Set[str]:
        return set(self.labels) if self.labels else {UNLABELED}


def _labels_from_annotation_data(data: dict) -> Counter[str]:
    counter: Counter[str] = Counter()
    for shape in data.get("shapes") or []:
        if not isinstance(shape, dict):
            continue
        label = shape.get("label")
        if label is None:
            continue
        name = str(label).strip()
        if name:
            counter[name] += 1
    flags = data.get("flags")
    if isinstance(flags, dict):
        for key, value in flags.items():
            if value and str(key).strip():
                counter[str(key).strip()] += 1
    return counter


def load_image_label_info(
    image_path: str, *, output_dir: Optional[str] = None
) -> ImageLabelInfo:
    label_path = get_label_file_path(image_path, output_dir)
    data = _load_json(label_path)
    if not data:
        return ImageLabelInfo(path=image_path, labels=frozenset(), box_counts=())

    counter = _labels_from_annotation_data(data)
    labels = frozenset(counter.keys())
    box_counts = tuple(sorted(counter.items(), key=lambda x: (-x[1], x[0])))
    return ImageLabelInfo(path=image_path, labels=labels, box_counts=box_counts)


def is_unlabeled_image_info(info: ImageLabelInfo) -> bool:
    return not info.labels


def load_all_image_labels(
    image_paths: Sequence[str], *, output_dir: Optional[str] = None
) -> List[ImageLabelInfo]:
    return [load_image_label_info(p, output_dir=output_dir) for p in image_paths]


@dataclass
class SplitSetStats:
    image_count: int
    unlabeled_images: int
    per_class_images: Dict[str, int]
    per_class_boxes: Dict[str, int]

    def sorted_rows(self) -> List[Tuple[str, int, int]]:
        classes = set(self.per_class_images) | set(self.per_class_boxes)
        classes.discard(UNLABELED)
        rows = []
        for name in sorted(classes):
            rows.append(
                (
                    name,
                    int(self.per_class_images.get(name, 0)),
                    int(self.per_class_boxes.get(name, 0)),
                )
            )
        return rows


def compute_split_stats(
    image_paths: Sequence[str], *, output_dir: Optional[str] = None
) -> SplitSetStats:
    per_class_images: Counter[str] = Counter()
    per_class_boxes: Counter[str] = Counter()
    unlabeled = 0
    for p in image_paths:
        info = load_image_label_info(p, output_dir=output_dir)
        if not info.labels:
            unlabeled += 1
            continue
        for lb in info.labels:
            per_class_images[lb] += 1
        for lb, cnt in info.box_counts:
            per_class_boxes[lb] += int(cnt)
    return SplitSetStats(
        image_count=len(image_paths),
        unlabeled_images=unlabeled,
        per_class_images=dict(per_class_images),
        per_class_boxes=dict(per_class_boxes),
    )


def random_split_paths(
    paths: Sequence[str],
    *,
    w_train: int,
    w_val: int,
    w_test: int,
    shuffle: bool,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    items = list(paths)
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(items)
    n_tr, n_v, n_te = split_sizes(len(items), w_train, w_val, w_test)
    i0 = 0
    part_train = items[i0 : i0 + n_tr]
    i0 += n_tr
    part_val = items[i0 : i0 + n_v]
    i0 += n_v
    part_test = items[i0 : i0 + n_te]
    return part_train, part_val, part_test


def _stratified_assign_group(
    infos: Sequence[ImageLabelInfo],
    *,
    caps: Dict[str, int],
    w_train: int,
    w_val: int,
    w_test: int,
    rng: random.Random,
) -> Dict[str, List[str]]:
    """Greedy class balancing within fixed per-split capacity caps."""
    assignments: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    if not infos:
        return assignments

    sizes = {"train": 0, "val": 0, "test": 0}
    class_totals: Counter[str] = Counter()
    for info in infos:
        for lb in info.label_set:
            class_totals[lb] += 1

    total_w = w_train + w_val + w_test
    fracs = {
        "train": w_train / total_w,
        "val": w_val / total_w,
        "test": w_test / total_w,
    }
    counts: Dict[str, Dict[str, int]] = {
        lb: {"train": 0, "val": 0, "test": 0} for lb in class_totals
    }

    order = list(infos)
    rng.shuffle(order)
    order.sort(key=lambda info: min(class_totals[lb] for lb in info.label_set))

    for info in order:
        labs = info.label_set
        best_split: Optional[str] = None
        best_score: Optional[float] = None
        for split in ("train", "val", "test"):
            if sizes[split] >= caps[split]:
                continue
            score = 0.0
            for lb in labs:
                target = class_totals[lb] * fracs[split]
                projected = counts[lb][split] + 1
                score += (projected - target) ** 2
            fill = sizes[split] / max(1, caps[split])
            score += 0.001 * fill
            if best_score is None or score < best_score:
                best_score = score
                best_split = split

        if best_split is None:
            for split in ("train", "val", "test"):
                if sizes[split] < caps[split]:
                    best_split = split
                    break
        if best_split is None:
            best_split = "train"

        assignments[best_split].append(info.path)
        sizes[best_split] += 1
        for lb in labs:
            counts[lb][best_split] += 1

    return assignments


def stratified_split_paths(
    infos: Sequence[ImageLabelInfo],
    *,
    w_train: int,
    w_val: int,
    w_test: int,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Assign each image once to train/val/test while balancing class presence
    across splits (image counted for every class it contains).

    Images without any category are split in a dedicated pass with the same
    train/val/test ratio, then merged so empty images are not pushed entirely
    into val/test after labeled images fill train.
    """
    if not infos:
        return [], [], []

    rng = random.Random(int(seed))
    labeled = [i for i in infos if not is_unlabeled_image_info(i)]
    unlabeled = [i for i in infos if is_unlabeled_image_info(i)]

    tr_l, va_l, te_l = split_sizes(len(labeled), w_train, w_val, w_test)
    tr_u, va_u, te_u = split_sizes(len(unlabeled), w_train, w_val, w_test)

    caps_l = {"train": tr_l, "val": va_l, "test": te_l}
    caps_u = {"train": tr_u, "val": va_u, "test": te_u}

    part_l = _stratified_assign_group(
        labeled, caps=caps_l, w_train=w_train, w_val=w_val, w_test=w_test, rng=rng
    )
    part_u = _stratified_assign_group(
        unlabeled, caps=caps_u, w_train=w_train, w_val=w_val, w_test=w_test, rng=rng
    )

    def _merge(split: str) -> List[str]:
        merged = part_l[split] + part_u[split]
        rng.shuffle(merged)
        return merged

    return _merge("train"), _merge("val"), _merge("test")


def run_split(
    paths: Sequence[str],
    *,
    w_train: int,
    w_val: int,
    w_test: int,
    stratify: bool,
    shuffle: bool,
    seed: int,
    output_dir: Optional[str] = None,
) -> Tuple[List[str], List[str], List[str]]:
    if stratify:
        infos = load_all_image_labels(paths, output_dir=output_dir)
        return stratified_split_paths(
            infos, w_train=w_train, w_val=w_val, w_test=w_test, seed=seed
        )
    return random_split_paths(
        paths,
        w_train=w_train,
        w_val=w_val,
        w_test=w_test,
        shuffle=shuffle,
        seed=seed,
    )