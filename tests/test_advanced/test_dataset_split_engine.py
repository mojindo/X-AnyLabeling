import json
import os
import tempfile

from anylabeling.views.labeling.advanced.ops.dataset_split_engine import (
    ImageLabelInfo,
    load_image_label_info,
    run_split,
    split_sizes,
    stratified_split_paths,
)


def _write_label(path_img: str, labels: list[str]) -> None:
    data = {
        "version": "0.4.30",
        "shapes": [
            {
                "label": lb,
                "shape_type": "rectangle",
                "points": [[0, 0], [10, 10]],
            }
            for lb in labels
        ],
        "imagePath": os.path.basename(path_img),
    }
    json_path = os.path.splitext(path_img)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp)


def test_split_sizes_remainder_goes_to_test():
    assert split_sizes(10, 7, 2, 1) == (7, 2, 1)
    assert split_sizes(11, 7, 2, 1) == (7, 2, 2)


def test_stratified_spreads_rare_class():
  with tempfile.TemporaryDirectory() as td:
        paths = []
        infos = []
        for i in range(9):
            p = os.path.join(td, f"a_{i}.jpg")
            open(p, "wb").close()
            _write_label(p, ["cat"])
            paths.append(p)
            infos.append(load_image_label_info(p))
        for i in range(9):
            p = os.path.join(td, f"b_{i}.jpg")
            open(p, "wb").close()
            _write_label(p, ["dog"])
            paths.append(p)
            infos.append(load_image_label_info(p))

        tr, va, te = stratified_split_paths(infos, w_train=7, w_val=2, w_test=1, seed=1)
        assert len(tr) == 12
        assert len(va) + len(te) == 6

        def count_class(split_paths, cls):
            n = 0
            for p in split_paths:
                info = load_image_label_info(p)
                if cls in info.labels:
                    n += 1
            return n

        for cls in ("cat", "dog"):
            assert 5 <= count_class(tr, cls) <= 8
            assert count_class(va, cls) >= 1
            assert count_class(te, cls) >= 0


def test_empty_images_reach_train_under_stratify():
    with tempfile.TemporaryDirectory() as td:
        paths = []
        infos = []
        for i in range(50):
            p = os.path.join(td, f"lbl_{i}.jpg")
            open(p, "wb").close()
            _write_label(p, ["obj"])
            paths.append(p)
            infos.append(load_image_label_info(p))
        for i in range(30):
            p = os.path.join(td, f"empty_{i}.jpg")
            open(p, "wb").close()
            paths.append(p)
            infos.append(load_image_label_info(p))

        tr, va, te = stratified_split_paths(
            infos, w_train=7, w_val=2, w_test=1, seed=42
        )
        empty = {p for p in paths if "empty_" in os.path.basename(p)}
        assert any(p in empty for p in tr)
        assert any(p in empty for p in va)
        assert sum(1 for p in tr if p in empty) >= 15


def test_run_split_stratify_is_deterministic():
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(20):
            p = os.path.join(td, f"img_{i}.jpg")
            open(p, "wb").close()
            _write_label(p, ["x" if i % 2 == 0 else "y"])
            paths.append(p)
        a = run_split(
            paths,
            w_train=8,
            w_val=1,
            w_test=1,
            stratify=True,
            shuffle=False,
            seed=99,
        )
        b = run_split(
            paths,
            w_train=8,
            w_val=1,
            w_test=1,
            stratify=True,
            shuffle=False,
            seed=99,
        )
        assert a == b