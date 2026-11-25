from anylabeling.services.auto_training.mytrain.predict_options import (
    PredictOptions,
    parse_classes_value,
    parse_predict_cli,
)


def test_parse_classes_value():
    assert parse_classes_value("") is None
    assert parse_classes_value("0") == (0,)
    assert parse_classes_value("0, 1") == (0, 1)


def test_parse_predict_cli_overrides():
    cli = parse_predict_cli("conf=0.5 iou=0.55 classes=0 save=False")
    assert cli["conf"] == 0.5
    assert cli["iou"] == 0.55
    assert cli["classes"] == [0]
    assert cli["save"] is False


def test_labels_only_preset():
    opts = PredictOptions.for_labels_only(imgsz=640)
    kw = opts.predict_kwargs(source="x", project="p", name="predict")
    assert kw["save"] is False
    assert kw["save_txt"] is True
    assert kw["conf"] == 0.25


def test_custom_cli_merge():
    opts = PredictOptions(custom_cli="conf=0.9 max_det=50 rect=True")
    merged = opts.with_cli_overrides()
    assert merged.conf == 0.9
    assert merged.max_det == 50
    assert merged.rect is True
    assert merged.predict_kwargs()["rect"] is True


def test_batch_predict_script_snippet_syntax():
    opts = PredictOptions.for_labels_only()
    run_dir = r"C:\runs\job1"
    predict_call = opts.render_predict_call(
        source_expr="image_folder",
        project_expr=repr(run_dir),
        name_expr="'batch_infer'",
        indent="",
    )
    script = (
        "from ultralytics import YOLO\n"
        "image_folder = '.'\n"
        "model = YOLO('w.pt')\n"
        f"{predict_call}\n"
    )
    compile(script, "<test>", "exec")


def test_predict_stream_loop_syntax():
    opts = PredictOptions(rect=True, verbose=False)
    loop = opts.render_predict_stream_loop(
        source_expr="image_folder",
        project_expr=repr(r"C:\runs\job1"),
        name_expr="'batch_infer'",
        indent="",
    )
    assert "stream=True" in loop
    assert "PREDICT %d/%d" in loop
    assert "rect=True" in loop
    script = "import os\nimage_folder = '.'\nmodel = None\n" + loop + "\n"
    compile(script, "<test>", "exec")


def test_try_block_predict_script_syntax():
    opts = PredictOptions.for_labels_only()
    run_dir = r"C:\runs\job1"
    predict_call = opts.render_predict_call(
        source_expr="all_txt",
        project_expr=repr(run_dir),
        name_expr="'predict'",
        indent="    ",
    )
    script = (
        "try:\n"
        "    model = YOLO('w.pt')\n"
        "    all_txt = 'a.txt'\n"
        f"{predict_call}\n"
        "finally:\n"
        "    pass\n"
    )
    compile(script, "<test>", "exec")