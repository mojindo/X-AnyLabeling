"""PyInstaller hidden imports for the MyTrain training stack."""

MYTRAIN_HIDDENIMPORTS = [
    "anylabeling.views.training.mytrain_dialog",
    "anylabeling.views.training.widgets.mytrain_widgets",
    "anylabeling.views.training.widgets.mytrain_widgets.custom_dialogs",
    "anylabeling.views.training.widgets.mytrain_widgets.custom_widgets",
    "anylabeling.views.training.widgets.mytrain_widgets.free_infer_panel",
    "anylabeling.services.auto_training.mytrain",
    "anylabeling.services.auto_training.mytrain.trainer",
    "anylabeling.services.auto_training.mytrain.exporter",
    "anylabeling.services.auto_training.mytrain.free_infer",
    "anylabeling.services.auto_training.mytrain.local_interpreters",
    "anylabeling.services.auto_training.mytrain.logging_bridge",
    "anylabeling.services.auto_training.mytrain.remote",
    "anylabeling.services.auto_training.mytrain.remote.job_state",
    "anylabeling.services.auto_training.mytrain.remote.remote_runtime",
    "anylabeling.services.auto_training.mytrain.remote.remote_trainer",
    "anylabeling.services.auto_training.mytrain.remote.remote_paths",
    "anylabeling.services.auto_training.mytrain.remote.remote_gpu",
    "anylabeling.services.auto_training.mytrain.remote.ssh_client",
    "anylabeling.services.auto_training.mytrain.remote.ssh_profiles",
    "paramiko",
]