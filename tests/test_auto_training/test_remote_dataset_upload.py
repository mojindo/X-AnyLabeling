import os
import tempfile
from pathlib import Path

from anylabeling.services.auto_training.mytrain.remote.remote_trainer import (
    UploadProgressReporter,
    _build_labels_splits_tar,
    _image_needs_upload,
    _local_upload_entries,
    _plan_upload_workload,
    _remote_image_rels_to_delete,
    _subdir_bulk_images_recommended,
)


def test_local_upload_entries_skips_json_under_images():
    with tempfile.TemporaryDirectory() as td:
        sd = os.path.join(td, "v1")
        img_dir = os.path.join(sd, "images")
        lbl_dir = os.path.join(sd, "labels")
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)
        open(os.path.join(td, "classes.txt"), "w", encoding="utf-8").write("cat\n")
        for name in ("train.txt", "val.txt", "test.txt", "all.txt"):
            open(os.path.join(sd, name), "w", encoding="utf-8").write("a.jpg\n")
        open(os.path.join(img_dir, "a.jpg"), "wb").write(b"\xff\xd8\xff")
        open(os.path.join(img_dir, "a.json"), "w", encoding="utf-8").write("{}")
        open(os.path.join(lbl_dir, "a.txt"), "w", encoding="utf-8").write("0 0.5 0.5 1 1\n")
        entries = _local_upload_entries(td, ["v1"])
        keys = set(entries)
        assert "classes.txt" in keys
        assert "v1/train.txt" in keys
        assert "v1/labels/a.txt" in keys
        assert "v1/images/a.jpg" in keys
        assert "v1/images/a.json" not in keys


def test_upload_progress_reporter_notify_partial():
    seen = []
    rep = UploadProgressReporter(
        4,
        {"v1": 2},
        callback=lambda o, d, s, sp: seen.append((o, sp)),
    )
    rep.step("labels", sd="v1")
    rep.notify("pack", sd="v1", partial=0.5)
    assert seen[-1][0] > 25.0
    assert seen[-1][1] == 75.0


def test_upload_progress_reporter_subdir_pct():
    steps = []
    rep = UploadProgressReporter(
        4,
        {"v1": 2},
        callback=lambda o, d, s, sp: steps.append((o, s, sp)),
    )
    rep.step("a", sd="v1")
    rep.step("b", sd="v1")
    rep.step("c")
    assert steps[0][0] == 25.0
    assert steps[1][2] == 100.0


def test_plan_upload_workload():
    with tempfile.TemporaryDirectory() as td:
        sd = os.path.join(td, "v1")
        os.makedirs(os.path.join(sd, "images"))
        os.makedirs(os.path.join(sd, "labels"))
        open(os.path.join(td, "classes.txt"), "w").write("x\n")
        open(os.path.join(sd, "train.txt"), "w").write("a.jpg\n")
        open(os.path.join(sd, "images", "a.jpg"), "wb").write(b"x")
        entries = _local_upload_entries(td, ["v1"])
        total, per = _plan_upload_workload(
            Path(td), ["v1"], entries, [], {}, None
        )
        assert per["v1"] >= 2
        assert total >= per["v1"] + 4


def test_build_labels_splits_tar():
    import tarfile

    with tempfile.TemporaryDirectory() as td:
        sd = os.path.join(td, "v1")
        os.makedirs(os.path.join(sd, "labels"))
        open(os.path.join(sd, "train.txt"), "w", encoding="utf-8").write("a.jpg\n")
        open(os.path.join(sd, "labels", "a.txt"), "w", encoding="utf-8").write("0\n")
        out = os.path.join(td, "bundle.tar.gz")
        assert _build_labels_splits_tar(Path(sd), out) is True
        with tarfile.open(out, "r:gz") as tar:
            names = set(tar.getnames())
        assert "train.txt" in names
        assert "labels/a.txt" in names


def test_subdir_bulk_images_recommended():
    entries = {
        "v1/images/a.jpg": {"size": 1, "mtime_ns": 1},
        "v1/images/b.jpg": {"size": 1, "mtime_ns": 1},
    }
    assert (
        _subdir_bulk_images_recommended(
            "v1", entries, {}, None, []
        )
        is True
    )
    many = {
        f"v1/images/{i}.jpg": {"size": 1, "mtime_ns": 1} for i in range(120)
    }
    remote = {f"v1/images/{i}.jpg": 999 for i in range(10)}
    assert (
        _subdir_bulk_images_recommended(
            "v1", many, {}, remote, []
        )
        is True
    )
    remote = {"v1/images/a.jpg": 1, "v1/images/b.jpg": 1}
    assert (
        _subdir_bulk_images_recommended(
            "v1", entries, {"v1/images/a.jpg": {"size": 1, "mtime_ns": 1}}, remote, []
        )
        is False
    )


def test_remote_image_rels_to_delete():
    entries = {
        "v1/images/keep.jpg": {"size": 1, "mtime_ns": 1},
        "v1/labels/keep.txt": {"size": 2, "mtime_ns": 2},
    }
    remote = {
        "v1/images/keep.jpg": 1,
        "v1/images/gone.jpg": 1,
        "v1/images/also_gone.png": 2,
        "v2/images/other.jpg": 9,
    }
    assert _remote_image_rels_to_delete(entries, remote, ["v1"]) == [
        "v1/images/also_gone.png",
        "v1/images/gone.jpg",
    ]
    assert _remote_image_rels_to_delete(entries, remote, ["v1", "v2"]) == [
        "v1/images/also_gone.png",
        "v1/images/gone.jpg",
        "v2/images/other.jpg",
    ]


def test_image_needs_upload_respects_manifest_mtime():
    meta = {"size": 100, "mtime_ns": 123}
    assert _image_needs_upload("v1/images/a.jpg", meta, {}, None) is True
    assert (
        _image_needs_upload(
            "v1/images/a.jpg",
            meta,
            {"v1/images/a.jpg": {"size": 100, "mtime_ns": 123}},
            {"v1/images/a.jpg": 100},
        )
        is False
    )
    assert (
        _image_needs_upload(
            "v1/images/a.jpg",
            meta,
            {"v1/images/a.jpg": {"size": 100, "mtime_ns": 0}},
            {"v1/images/a.jpg": 100},
        )
        is True
    )