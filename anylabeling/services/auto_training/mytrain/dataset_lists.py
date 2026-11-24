"""Merge per-subdir YOLO split lists and write root data.yaml (Detect)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

SPLIT_FILES = ("train.txt", "val.txt", "test.txt", "all.txt")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def path_line_for_txt(p: Union[str, Path]) -> str:
    """Absolute path with POSIX-style separators for list txt / yaml."""
    return str(Path(p).resolve()).replace("\\", "/")


def path_line_for_txt_remote(abs_posix: str) -> str:
    """Normalize an already-absolute remote POSIX path for list/yaml lines."""
    return abs_posix.replace("\\", "/")


def load_classes(classes_path: Union[str, Path]) -> List[str]:
    with open(classes_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_name_lines(list_file: Union[str, Path]) -> List[str]:
    out: List[str] = []
    for line in Path(list_file).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(Path(s).name)
    return out


def list_candidate_subdirs(root: Union[str, Path]) -> List[str]:
    """First-level directories under ROOT that look like dataset versions."""
    root = Path(root)
    if not root.is_dir():
        return []
    names: List[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name.startswith("_"):
            continue
        # Prefer dirs that already have images/ or any split txt
        if (child / "images").is_dir() or any(
            (child / s).is_file() for s in SPLIT_FILES
        ):
            names.append(child.name)
    return names


def validate_dataset_root(
    root: Union[str, Path], subdirs: Sequence[str]
) -> Tuple[bool, str]:
    """Validate ROOT/classes.txt and each subdir's split lists."""
    root = Path(root)
    if not root.is_dir():
        return False, f"Dataset root does not exist: {root}"
    classes_path = root / "classes.txt"
    if not classes_path.is_file():
        return False, f"Missing classes.txt: {classes_path}"
    classes = load_classes(classes_path)
    if not classes:
        return False, f"No valid class names in {classes_path}"
    if not subdirs:
        return False, "Please select at least one sub-dataset version"
    for sd in subdirs:
        sub = root / sd
        if not sub.is_dir():
            return False, f"Sub-directory does not exist: {sub}"
        for split in SPLIT_FILES:
            p = sub / split
            if not p.is_file():
                return False, f"Missing {p}"
    return True, ""


def check_subdir_split_lists(root: Path, subdirs: List[str]) -> int:
    """CLI-compatible: return 0 if OK, else 1 (prints via return code only)."""
    ok, _ = validate_dataset_root(root, subdirs)
    return 0 if ok else 1


def remove_main_outputs_if_exist(root: Path, yaml_path: Path) -> List[str]:
    """Delete existing merge outputs; return names of removed files."""
    targets: List[Path] = [root / name for name in SPLIT_FILES] + [
        yaml_path.resolve()
    ]
    seen: set = set()
    removed: List[str] = []
    for p in targets:
        pr = p.resolve()
        if pr in seen:
            continue
        seen.add(pr)
        if pr.is_file():
            pr.unlink()
            removed.append(pr.name)
    return removed


def merge_paths_for_split(
    root: Path,
    subdirs: Sequence[str],
    split_name: str,
    *,
    path_formatter=None,
) -> List[str]:
    """Build absolute image paths for one split across subdirs."""
    if path_formatter is None:
        path_formatter = path_line_for_txt
    merged: List[str] = []
    root = Path(root)
    for sd in subdirs:
        sub_root = (root / sd).resolve()
        list_path = sub_root / split_name
        names = read_name_lines(list_path)
        if not names:
            continue
        images_dir = sub_root / "images"
        for name in names:
            full = (images_dir / name).resolve()
            merged.append(path_formatter(full))
    return merged


def write_lines(path: Path, lines: List[str]) -> None:
    path.write_text("".join(s + "\n" for s in lines), encoding="utf-8")


def resolve_yolo_label_path(image_path: Union[str, Path]) -> Path:
    """Map ``.../images/foo.jpg`` → ``.../labels/foo.txt``; else same-stem .txt."""
    p = Path(image_path)
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return p.with_suffix(".txt")


def read_image_paths_from_list(list_file: Union[str, Path]) -> List[str]:
    out: List[str] = []
    path = Path(list_file)
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            out.append(s)
    return out


def collect_all_image_paths(
    root: Union[str, Path],
    subdirs: Optional[Sequence[str]] = None,
) -> List[str]:
    """Prefer ROOT/all.txt; else concatenate selected (or candidate) subdir all.txt."""
    root = Path(root)
    root_all = root / "all.txt"
    if root_all.is_file():
        return read_image_paths_from_list(root_all)

    names = list(subdirs) if subdirs else list_candidate_subdirs(root)
    paths: List[str] = []
    seen: set = set()
    for sd in names:
        for p in read_image_paths_from_list(root / sd / "all.txt"):
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


def get_detect_dataset_summary_table(
    root: Union[str, Path],
    subdirs: Optional[Sequence[str]] = None,
) -> List[List[str]]:
    """Build summary rows from ROOT ``classes.txt`` + ``all.txt`` / YOLO labels.

    Columns: Class | ID | Instances | Images
    """
    root = Path(root)
    classes_path = root / "classes.txt"
    if not root.is_dir() or not classes_path.is_file():
        return []

    classes = load_classes(classes_path)
    if not classes:
        return []

    image_paths = collect_all_image_paths(root, subdirs)
    instance_counts = [0] * len(classes)
    image_counts = [0] * len(classes)

    for img in image_paths:
        img_path = Path(img)
        if not img_path.is_file():
            cand = root / img
            if cand.is_file():
                img_path = cand
        label_path = resolve_yolo_label_path(img_path)
        if not label_path.is_file():
            continue
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        ids_in_image: set = set()
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            if 0 <= cid < len(classes):
                instance_counts[cid] += 1
                ids_in_image.add(cid)
        for cid in ids_in_image:
            image_counts[cid] += 1

    rows: List[List[str]] = [["Class", "ID", "Instances", "Images"]]
    for i, name in enumerate(classes):
        rows.append(
            [name, str(i), str(instance_counts[i]), str(image_counts[i])]
        )
    rows.append(
        [
            "Total",
            "",
            str(sum(instance_counts)),
            str(len(image_paths)),
        ]
    )
    return rows


def count_subdir_images_and_labels(sub_dir: Union[str, Path]) -> Tuple[int, int]:
    """Count image files under ``images/`` and ``.txt`` files under ``labels/``."""
    sub_dir = Path(sub_dir)
    n_images = 0
    n_labels = 0
    images_root = sub_dir / "images"
    if images_root.is_dir():
        for _dirpath, dirnames, filenames in os.walk(images_root):
            dirnames[:] = [d for d in dirnames if d and not d.startswith(".")]
            for name in filenames:
                if not name or name.startswith("."):
                    continue
                if Path(name).suffix.lower() in IMAGE_EXTS:
                    n_images += 1
    labels_root = sub_dir / "labels"
    if labels_root.is_dir():
        for _dirpath, dirnames, filenames in os.walk(labels_root):
            dirnames[:] = [d for d in dirnames if d and not d.startswith(".")]
            for name in filenames:
                if not name or name.startswith("."):
                    continue
                if name.lower().endswith(".txt"):
                    n_labels += 1
    return n_images, n_labels


def audit_subdataset_images_labels(
    root: Union[str, Path], subdirs: Sequence[str]
) -> List[Dict[str, object]]:
    """Return sub-datasets whose images/labels are empty or unequal.

    Each item is ``{"subdir", "images", "labels"}``. An empty list means every
    selected sub-dataset has a non-empty images folder and a non-empty labels
    folder with the same file count.
    """
    root = Path(root)
    issues: List[Dict[str, object]] = []
    for sd in subdirs:
        n_images, n_labels = count_subdir_images_and_labels(root / sd)
        if n_images > 0 and n_labels > 0 and n_images == n_labels:
            continue
        issues.append(
            {"subdir": sd, "images": n_images, "labels": n_labels}
        )
    return issues


def yaml_escape_single_quoted(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def build_data_yaml(
    root: Union[str, Path],
    classes: List[str],
    *,
    path_formatter=None,
) -> str:
    if path_formatter is None:
        path_formatter = path_line_for_txt
    root = Path(root)
    train_p = path_formatter(root / "train.txt")
    val_p = path_formatter(root / "val.txt")
    test_p = path_formatter(root / "test.txt")
    lines: List[str] = [
        f"train: {train_p}",
        f"val: {val_p}",
        f"test: {test_p}",
        "",
        "# number of classes",
        f"nc: {len(classes)}",
        "",
        "# class names",
        "names: [",
    ]
    for i, name in enumerate(classes):
        sep = "," if i < len(classes) - 1 else ""
        lines.append(f"   {yaml_escape_single_quoted(name)}{sep}")
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


def merge_dataset_lists(
    root: Union[str, Path],
    subdirs: Sequence[str],
    *,
    yaml_name: str = "data.yaml",
    path_formatter=None,
) -> Dict[str, object]:
    """
    Merge subdir split lists into ROOT and write data.yaml.

    Returns dict with keys: ok, error, yaml_path, counts, removed, classes.
    """
    root = Path(root).resolve()
    ok, err = validate_dataset_root(root, subdirs)
    if not ok:
        return {"ok": False, "error": err}

    if path_formatter is None:
        path_formatter = path_line_for_txt

    classes = load_classes(root / "classes.txt")
    yaml_path = root / yaml_name
    removed = remove_main_outputs_if_exist(root, yaml_path)

    train_m = merge_paths_for_split(
        root, subdirs, "train.txt", path_formatter=path_formatter
    )
    val_m = merge_paths_for_split(
        root, subdirs, "val.txt", path_formatter=path_formatter
    )
    test_m = merge_paths_for_split(
        root, subdirs, "test.txt", path_formatter=path_formatter
    )
    all_m = merge_paths_for_split(
        root, subdirs, "all.txt", path_formatter=path_formatter
    )

    write_lines(root / "train.txt", train_m)
    write_lines(root / "val.txt", val_m)
    write_lines(root / "test.txt", test_m)
    write_lines(root / "all.txt", all_m)
    yaml_path.write_text(
        build_data_yaml(root, classes, path_formatter=path_formatter),
        encoding="utf-8",
    )
    image_label_issues = audit_subdataset_images_labels(root, subdirs)

    return {
        "ok": True,
        "error": "",
        "yaml_path": str(yaml_path),
        "root": str(root),
        "counts": {
            "train": len(train_m),
            "val": len(val_m),
            "test": len(test_m),
            "all": len(all_m),
        },
        "removed": removed,
        "classes": classes,
        "image_label_issues": image_label_issues,
    }


def build_remote_merge_script(
    remote_root: str, subdirs: Sequence[str], yaml_name: str = "data.yaml"
) -> str:
    """
    Python script source to run on the remote host (no local package deps).
    ``remote_root`` must be an absolute POSIX path.
    """
    template = r'''# -*- coding: utf-8 -*-
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__REMOTE_ROOT__)
SUBDIRS = __SUBDIRS__
YAML_NAME = __YAML_NAME__
SPLIT_FILES = ("train.txt", "val.txt", "test.txt", "all.txt")
IMAGE_EXTS = set([".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"])

def fail(msg, code=1):
    print("MERGE_ERROR: " + str(msg), flush=True)
    sys.exit(code)

def path_line(p):
    return str(p.resolve()).replace("\\", "/")

def load_classes(p):
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

def read_names(p):
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s:
            out.append(Path(s).name)
    return out

def merge_split(split_name):
    merged = []
    for sd in SUBDIRS:
        sub = (ROOT / sd).resolve()
        names = read_names(sub / split_name)
        images_dir = sub / "images"
        for name in names:
            merged.append(path_line(images_dir / name))
    return merged

def count_images_labels(sub):
    n_img = 0
    n_lbl = 0
    img_root = sub / "images"
    if img_root.is_dir():
        for _dirpath, dirnames, filenames in os.walk(str(img_root)):
            dirnames[:] = [d for d in dirnames if d and not d.startswith(".")]
            for name in filenames:
                if not name or name.startswith("."):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext in IMAGE_EXTS:
                    n_img += 1
    lab_root = sub / "labels"
    if lab_root.is_dir():
        for _dirpath, dirnames, filenames in os.walk(str(lab_root)):
            dirnames[:] = [d for d in dirnames if d and not d.startswith(".")]
            for name in filenames:
                if not name or name.startswith("."):
                    continue
                if name.lower().endswith(".txt"):
                    n_lbl += 1
    return n_img, n_lbl

def yaml_escape(s):
    return "'" + s.replace("'", "''") + "'"

def build_yaml(classes):
    lines = [
        "train: " + path_line(ROOT / "train.txt"),
        "val: " + path_line(ROOT / "val.txt"),
        "test: " + path_line(ROOT / "test.txt"),
        "",
        "# number of classes",
        "nc: " + str(len(classes)),
        "",
        "# class names",
        "names: [",
    ]
    for i, name in enumerate(classes):
        sep = "," if i < len(classes) - 1 else ""
        lines.append("   " + yaml_escape(name) + sep)
    lines.append("]")
    lines.append("")
    return "\n".join(lines)

def main():
    if not ROOT.is_dir():
        fail("ROOT missing: %s" % ROOT)
    classes_path = ROOT / "classes.txt"
    if not classes_path.is_file():
        fail("missing classes.txt under %s" % ROOT)
    classes = load_classes(classes_path)
    if not classes:
        fail("empty classes.txt")
    if not SUBDIRS:
        fail("no sub-dataset versions selected")
    for sd in SUBDIRS:
        sub = ROOT / sd
        if not sub.is_dir():
            fail("subdir missing: %s" % sub)
        for sp in SPLIT_FILES:
            p = sub / sp
            if not p.is_file():
                fail("missing %s" % p)

    yaml_path = ROOT / YAML_NAME
    for name in list(SPLIT_FILES) + [YAML_NAME]:
        p = ROOT / name
        if p.is_file():
            p.unlink()

    train_m = merge_split("train.txt")
    val_m = merge_split("val.txt")
    test_m = merge_split("test.txt")
    all_m = merge_split("all.txt")
    for name, lines in (
        ("train.txt", train_m),
        ("val.txt", val_m),
        ("test.txt", test_m),
        ("all.txt", all_m),
    ):
        (ROOT / name).write_text(
            "".join(x + "\n" for x in lines), encoding="utf-8"
        )
    yaml_path.write_text(build_yaml(classes), encoding="utf-8")
    for sd in SUBDIRS:
        n_img, n_lbl = count_images_labels(ROOT / sd)
        if n_img > 0 and n_lbl > 0 and n_img == n_lbl:
            continue
        print("DATASET_CHECK|%s|%d|%d" % (sd, n_img, n_lbl), flush=True)
    print(
        "OK yaml=%s train=%d val=%d test=%d all=%d nc=%d"
        % (
            yaml_path,
            len(train_m),
            len(val_m),
            len(test_m),
            len(all_m),
            len(classes),
        ),
        flush=True,
    )

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        fail(traceback.format_exc())
'''
    return (
        template.replace("__REMOTE_ROOT__", repr(remote_root))
        .replace("__SUBDIRS__", repr(list(subdirs)))
        .replace("__YAML_NAME__", repr(yaml_name))
    )
