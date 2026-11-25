"""Jump to prev/next image that contains any of the selected labels."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from anylabeling.views.labeling.advanced.compare.annotation_compare import (
    PENDING_REVIEW_STATUSES,
    get_compare_status,
)
from anylabeling.views.labeling.classifier.utils import get_label_file_path

# QListWidget UserRole value for the synthetic empty-image jump target.
EMPTY_IMAGE_JUMP_KEY = "__empty_image__"


def _load_label_json(label_path: str) -> Optional[dict]:
    if not label_path or not os.path.isfile(label_path):
        return None
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def collect_labels_from_json(label_path: str) -> Set[str]:
    """Return label names present in a LabelMe/xlabel JSON file."""
    data = _load_label_json(label_path)
    if not data:
        return set()

    found: Set[str] = set()
    for shape in data.get("shapes") or []:
        if not isinstance(shape, dict):
            continue
        label = shape.get("label")
        if label is None:
            continue
        text = str(label).strip()
        if text:
            found.add(text)

    flags = data.get("flags")
    if isinstance(flags, dict):
        for key, value in flags.items():
            if value and str(key).strip():
                found.add(str(key).strip())
    return found


def image_is_empty_for_jump(
    image_path: str,
    *,
    output_dir: Optional[str] = None,
) -> bool:
    """True when the image has no normal annotation labels.

    Suspect / model-eval marks are ignored; only non-review shapes count.
    """
    label_path = get_label_file_path(str(image_path), output_dir)
    data = _load_label_json(label_path)
    if not data:
        return True
    shapes = data.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        return True
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        if get_compare_status(shape) in PENDING_REVIEW_STATUSES:
            continue
        if str(shape.get("label") or "").strip():
            return False
    return True


def count_strict_empty_images(
    image_list: Sequence[str],
    *,
    output_dir: Optional[str] = None,
) -> int:
    """How many images have no normal annotation labels (for empty-image jump)."""
    n = 0
    for image_path in image_list or []:
        if not image_path:
            continue
        if image_is_empty_for_jump(str(image_path), output_dir=output_dir):
            n += 1
    return n


def image_matches_jump_targets(
    image_path: str,
    labels: Iterable[str],
    output_dir: Optional[str] = None,
) -> bool:
    """True if the image matches any selected jump target (labels and/or empty)."""
    wanted = {str(x).strip() for x in labels if str(x).strip()}
    if not wanted:
        return False
    want_empty = EMPTY_IMAGE_JUMP_KEY in wanted
    label_names = wanted - {EMPTY_IMAGE_JUMP_KEY}
    if want_empty and image_is_empty_for_jump(str(image_path), output_dir=output_dir):
        return True
    if label_names and image_has_any_label(
        str(image_path), label_names, output_dir=output_dir
    ):
        return True
    return False


def image_has_any_label(
    image_path: str,
    labels: Iterable[str],
    output_dir: Optional[str] = None,
) -> bool:
    """True if the image's label file contains any label in ``labels``."""
    wanted = {str(x).strip() for x in labels if str(x).strip()}
    if not wanted:
        return False
    label_path = get_label_file_path(image_path, output_dir)
    present = collect_labels_from_json(label_path)
    return bool(present & wanted)


def find_image_with_labels(
    image_list: Sequence[str],
    *,
    current_path: Optional[str],
    labels: Sequence[str],
    output_dir: Optional[str] = None,
    direction: int = 1,
) -> Optional[str]:
    """
    Find the next (direction>0) or previous (direction<0) image that contains
    at least one of ``labels``.
    """
    paths = [str(p) for p in (image_list or []) if p]
    wanted = [str(x).strip() for x in (labels or []) if str(x).strip()]
    if not paths or not wanted:
        return None

    cur = str(current_path) if current_path else ""
    try:
        cur_index = paths.index(cur) if cur else -1
    except ValueError:
        # Fall back to basename match
        cur_index = -1
        base = os.path.basename(cur)
        if base:
            for i, p in enumerate(paths):
                if os.path.basename(p) == base:
                    cur_index = i
                    break

    if direction >= 0:
        indices = range(cur_index + 1, len(paths))
    else:
        indices = range(cur_index - 1, -1, -1)

    for i in indices:
        if image_matches_jump_targets(paths[i], wanted, output_dir):
            return paths[i]
    return None


def collect_available_labels(
    image_list: Sequence[str],
    *,
    output_dir: Optional[str] = None,
    seed_labels: Optional[Sequence[str]] = None,
) -> List[str]:
    """Union of seed labels and labels found in the dataset JSON files."""
    return [
        name
        for name, _ in collect_available_labels_with_counts(
            image_list, output_dir=output_dir, seed_labels=seed_labels
        )
    ]


def collect_available_labels_with_counts(
    image_list: Sequence[str],
    *,
    output_dir: Optional[str] = None,
    seed_labels: Optional[Sequence[str]] = None,
) -> List[Tuple[str, int]]:
    """
    Labels available in the dataset with annotation-box counts.

    Count is the total number of shapes with that label across all JSON files.
    Seed / flag-only labels with no shapes appear with count 0.
    """
    counts: Dict[str, int] = {}
    for name in seed_labels or []:
        text = str(name).strip()
        if text:
            counts.setdefault(text, 0)

    for image_path in image_list or []:
        label_path = get_label_file_path(str(image_path), output_dir)
        data = _load_label_json(label_path)
        if not data:
            continue
        for shape in data.get("shapes") or []:
            if not isinstance(shape, dict):
                continue
            label = shape.get("label")
            if label is None:
                continue
            text = str(label).strip()
            if text:
                counts[text] = counts.get(text, 0) + 1
        flags = data.get("flags")
        if isinstance(flags, dict):
            for key, value in flags.items():
                if value and str(key).strip():
                    counts.setdefault(str(key).strip(), 0)

    return sorted(counts.items(), key=lambda kv: kv[0].lower())


def fuzzy_match_label(label: str, query: str) -> bool:
    """
    Fuzzy match a label against a search query.

    Rules (case-insensitive):
    1. Empty query matches everything.
    2. Substring containment.
    3. Subsequence match (query chars appear in order).
    4. Token AND: space/underscore/hyphen separated tokens all match as
       substring or subsequence.
    """
    text = str(label or "")
    q = str(query or "").strip()
    if not q:
        return True

    text_cf = text.casefold()
    q_cf = q.casefold()
    if q_cf in text_cf:
        return True
    if _is_subsequence(q_cf, text_cf):
        return True

    tokens = [
        t for t in q_cf.replace("_", " ").replace("-", " ").split() if t
    ]
    if len(tokens) <= 1:
        return False
    return all(
        (t in text_cf) or _is_subsequence(t, text_cf) for t in tokens
    )


def _is_subsequence(needle: str, haystack: str) -> bool:
    if not needle:
        return True
    i = 0
    for ch in haystack:
        if ch == needle[i]:
            i += 1
            if i >= len(needle):
                return True
    return False
