import subprocess
import sys

from anylabeling.services.auto_training.mytrain.dataset_lists import (
    audit_subdataset_images_labels,
    build_remote_merge_script,
    merge_dataset_lists,
)
from anylabeling.services.auto_training.mytrain.remote.remote_trainer import (
    _parse_dataset_check_lines,
)


def _write_splits(sub, name="a.jpg"):
    for split in ("train.txt", "val.txt", "test.txt", "all.txt"):
        (sub / split).write_text(name + "\n", encoding="utf-8")


def _make_sub(root, subdir, n_images, n_labels):
    sub = root / subdir
    (sub / "images").mkdir(parents=True)
    (sub / "labels").mkdir()
    for i in range(n_images):
        (sub / "images" / f"{i}.jpg").write_bytes(b"img")
        (sub / "images" / f"{i}.json").write_text("{}", encoding="utf-8")
    for i in range(n_labels):
        (sub / "labels" / f"{i}.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
        )
    (sub / "labels" / "labels.cache").write_bytes(b"cache")
    _write_splits(sub)
    return sub


def test_audit_flags_empty_and_unequal_counts(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    _make_sub(root, "ok", 2, 2)
    _make_sub(root, "short_labels", 2, 1)
    _make_sub(root, "empty", 0, 0)

    issues = audit_subdataset_images_labels(
        root, ["ok", "short_labels", "empty"]
    )
    by_name = {item["subdir"]: item for item in issues}
    assert "ok" not in by_name
    assert by_name["short_labels"] == {
        "subdir": "short_labels",
        "images": 2,
        "labels": 1,
    }
    assert by_name["empty"] == {"subdir": "empty", "images": 0, "labels": 0}


def test_local_merge_keeps_lists_and_reports_mismatch(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    (root / "classes.txt").write_text("person\n", encoding="utf-8")
    _make_sub(root, "base", 1, 0)

    result = merge_dataset_lists(root, ["base"])
    assert result["ok"] is True
    assert (root / "data.yaml").is_file()
    assert (root / "train.txt").is_file()
    assert result["image_label_issues"] == [
        {"subdir": "base", "images": 1, "labels": 0}
    ]


def test_remote_merge_script_prints_the_same_check(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    (root / "classes.txt").write_text("person\n", encoding="utf-8")
    _make_sub(root, "base", 2, 1)
    script = tmp_path / "merge.py"
    script.write_text(
        build_remote_merge_script(str(root), ["base"]), encoding="utf-8"
    )

    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK yaml=" in proc.stdout
    assert _parse_dataset_check_lines(proc.stdout) == [
        {"subdir": "base", "images": 2, "labels": 1}
    ]
