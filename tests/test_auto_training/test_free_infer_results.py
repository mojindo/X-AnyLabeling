import os

from anylabeling.services.auto_training.mytrain.free_infer import (
    describe_infer_project,
)


def test_describe_infer_project_prefers_annotated_images(tmp_path):
    source = tmp_path / "source"
    labels = tmp_path / "batch_infer" / "labels"
    source.mkdir()
    labels.mkdir(parents=True)
    (source / "a.jpg").write_bytes(b"src")
    (tmp_path / "batch_infer" / "a.jpg").write_bytes(b"ann")
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    info = describe_infer_project(str(tmp_path))
    assert info["annotated"] is True
    assert os.path.basename(info["images"][0]) == "a.jpg"
    assert info["images"][0].endswith(os.path.join("batch_infer", "a.jpg"))


def test_describe_infer_project_uses_source_when_no_annotated(tmp_path):
    source = tmp_path / "source"
    labels = tmp_path / "batch_infer" / "labels"
    source.mkdir()
    labels.mkdir(parents=True)
    (source / "b.png").write_bytes(b"src")
    (labels / "b.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    info = describe_infer_project(str(tmp_path))
    assert info["annotated"] is False
    assert [os.path.basename(p) for p in info["images"]] == ["b.png"]
    assert info["labels_dir"].endswith("labels")
