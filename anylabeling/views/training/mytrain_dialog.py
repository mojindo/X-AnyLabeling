import csv
import datetime
import glob
import os
import platform
import re
import shutil
import subprocess
import threading

from PyQt6.QtCore import Qt, QObject, QTimer, pyqtSignal, QCoreApplication
from PyQt6.QtGui import QIcon, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QPushButton,
    QLabel,
    QMessageBox,
    QScrollArea,
    QGroupBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QProgressBar,
    QTextEdit,
    QApplication,
    QSizePolicy,
    QRadioButton,
    QButtonGroup,
)

from anylabeling.config import get_config
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.qt import new_icon
from anylabeling.views.training.widgets.mytrain_widgets import *
from anylabeling.services.auto_training.mytrain._io import *
from anylabeling.services.auto_training.mytrain.config import *
from anylabeling.services.auto_training.mytrain.exporter import (
    ExportEventRedirector,
    ExportLogRedirector,
    get_export_manager,
)
from anylabeling.services.auto_training.mytrain.export_naming import (
    resolve_export_meta,
)
from anylabeling.services.auto_training.mytrain.general import (
    create_yolo_dataset,
    format_classes_display,
    parse_freeze_value,
    parse_string_to_digit_list,
)
from anylabeling.services.auto_training.mytrain.dataset_lists import (
    get_detect_dataset_summary_table,
    list_candidate_subdirs,
    load_classes,
    merge_dataset_lists,
    validate_dataset_root,
)
from anylabeling.services.auto_training.mytrain.style import *
from anylabeling.services.auto_training.mytrain.trainer import (
    TrainingEventRedirector,
    TrainingLogRedirector,
    get_training_manager,
)
from anylabeling.services.auto_training.mytrain.utils import *
from anylabeling.services.auto_training.mytrain.validators import (
    validate_basic_config,
    validate_data_file,
    validate_task_requirements,
)
from anylabeling.services.auto_training.mytrain.free_infer import (
    create_free_infer_job_dir,
    delete_local_project,
    list_local_free_infer_projects,
    list_local_train_projects,
    package_results,
    prepare_source_images,
    resolve_model_path,
    scan_local_weights,
    scan_remote_weights,
    FreeInferJob,
    WeightEntry,
)
from anylabeling.services.auto_training.mytrain.remote.ssh_client import (
    SSHClient,
    SSHError,
)
from anylabeling.services.auto_training.mytrain.remote.ssh_profiles import (
    delete_profile,
    empty_profile,
    get_active_profile,
    load_ssh_profiles,
    mark_test_result,
    profile_display_name,
    profiles_path,
    save_ssh_profiles,
    upsert_profile,
)
from anylabeling.services.auto_training.mytrain.remote.remote_paths import (
    dataset_remote_dir,
    posix_join,
    resolve_remote_base,
    runs_remote_dir,
)
from anylabeling.services.auto_training.mytrain.remote.remote_runtime import (
    list_remote_conda_envs,
    probe_remote_python,
    resolve_remote_conda_executable,
)
from anylabeling.services.auto_training.mytrain.remote.remote_trainer import (
    list_remote_weight_choices,
    remote_batch_predict_yolo,
    remote_export_onnx,
    remote_merge_lists,
    remote_predict_yolo,
    probe_remote_dataset_subdirs,
    validate_remote_dataset_root,
    UploadGate,
    UploadStopped,
    upload_dataset,
    upload_model_if_local,
    count_local_images,
)
from anylabeling.services.auto_training.mytrain.custom_training import (
    bind_command_to_python,
    bind_remote_command_to_python,
    build_path_context,
    format_train_log_terminal,
    is_windows_local_path,
    latest_log_session_text,
    parse_epochs_from_cli,
    posix_if_needed,
    substitute_custom_command,
)
from anylabeling.services.auto_training.mytrain.remote.job_state import (
    clear_local_job,
    load_local_job,
    load_remote_job,
    probe_local_job,
    probe_remote_job,
)
from anylabeling.services.auto_training.mytrain.remote.remote_gpu import (
    MIN_USABLE_MIB,
    probe_remote_gpus,
)
from anylabeling.services.auto_training.mytrain import (
    local_interpreters as local_interp,
)


class _BgTaskBridge(QObject):
    """Cross-thread bridge so long SSH/local jobs do not freeze the UI."""

    finished = pyqtSignal(str, bool, object)  # task_name, ok, payload
    log = pyqtSignal(str)


class MyTrainDialog(QDialog):
    def tr(self, source_text, /, disambiguation=None, n=-1):
        """Reuse UltralyticsDialog translations for the forked training UI."""
        return QCoreApplication.translate(
            "UltralyticsDialog", source_text, disambiguation, n
        )

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        # Allow labeling main window to stay interactive while training
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(*DEFAULT_WINDOW_SIZE)
        self.setMinimumSize(*DEFAULT_WINDOW_SIZE)

        self.image_list = getattr(parent, "image_list", None) or []
        self.output_dir = getattr(parent, "output_dir", None)
        self.supported_shape = getattr(parent, "supported_shape", None) or []
        self.selected_task_type = "Detect"
        self.config_widgets = {}
        self._classification_cache = None
        self._detection_cache = None
        self.task_type_buttons = {}
        self.names = []

        # Local / Interpreter / Remote dataset training
        self.train_mode = "local"  # local | interpreter | remote
        self.training_backend = "ultralytics"  # ultralytics | custom
        self.dataset_root = ""
        self.remote_dataset_root = ""
        self.remote_dataset_source = "upload"  # upload | remote
        self.data_yaml_path = ""
        self.ssh_client = SSHClient()
        self.subdir_checkboxes = {}
        self.tensorboard_url = ""
        self.remote_gpu_result = None
        self.remote_train_allowed = True  # local/interpreter unrestricted
        self.export_meta = {}
        self._last_batch_infer_labels = ""  # local or remote labels dir
        self._interp_profiles = {
            "version": 1,
            "active_id": "",
            "interpreters": [],
        }
        self._interp_row_widgets = []
        self._interp_loading = False
        self.interpreter_probe_info = None

        # Background jobs (export / predict / download) — keep UI responsive
        self._bg_busy = False
        self._bg_bridge = _BgTaskBridge()
        self._bg_bridge.finished.connect(
            self._on_bg_task_finished, Qt.ConnectionType.QueuedConnection
        )
        self._bg_bridge.log.connect(
            self.append_training_log, Qt.ConnectionType.QueuedConnection
        )
        self._upload_bridge = _BgTaskBridge()
        self._upload_bridge.log.connect(
            self._on_dataset_upload_progress, Qt.ConnectionType.QueuedConnection
        )
        self._upload_bridge.finished.connect(
            self._on_dataset_upload_finished, Qt.ConnectionType.QueuedConnection
        )
        self._upload_gate = UploadGate()
        self._dataset_upload_running = False

        # Training related attributes
        self.log_redirector = TrainingLogRedirector()
        self.log_redirector.log_signal.connect(
            self.append_training_log, Qt.ConnectionType.QueuedConnection
        )
        self.event_redirector = TrainingEventRedirector()
        self.event_redirector.training_event_signal.connect(
            self.on_training_event, Qt.ConnectionType.QueuedConnection
        )
        self.training_manager = get_training_manager()
        self.training_manager.callbacks = [
            self.event_redirector.emit_training_event
        ]

        # Export related attributes
        self.export_log_redirector = ExportLogRedirector()
        self.export_log_redirector.log_signal.connect(
            self.append_training_log, Qt.ConnectionType.QueuedConnection
        )
        self.export_event_redirector = ExportEventRedirector()
        self.export_event_redirector.export_event_signal.connect(
            self.on_export_event, Qt.ConnectionType.QueuedConnection
        )
        self.export_manager = get_export_manager()
        self.export_manager.callbacks = [
            self.export_event_redirector.emit_export_event
        ]

        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_training_progress)
        self.image_timer = QTimer()
        self.image_timer.timeout.connect(self.update_training_images)
        self.current_project_path = None
        self.training_status = "idle"  # idle, training, completed, error
        self.current_epochs = 0
        self._application_closing = False
        self._log_follow_tail = True
        self._log_scroll_programmatic = False

        app_config = get_config()
        train_cfg = app_config.get("training", {})
        mytrain_cfg = train_cfg.get("mytrain") or train_cfg.get("ultralytics", {})
        self.project_readonly = mytrain_cfg.get("project_readonly", True)

        self.init_ui()
        self.setStyleSheet(get_ultralytics_dialog_style())
        self.refresh_dataset_summary()
        QTimer.singleShot(600, self._prompt_saved_training_sessions)

    def init_ui(self):
        self.data_tab = QWidget()
        self.config_tab = QWidget()
        self.train_tab = QWidget()
        self.free_infer_tab = QWidget()

        workflow_row = QHBoxLayout()
        workflow_row.addWidget(QLabel(self.tr("Workspace:")))
        self.workflow_combo = CustomComboBox()
        self.workflow_combo.addItem(self.tr("Train"), "train")
        self.workflow_combo.addItem(self.tr("Free Inference"), "free_infer")
        self.workflow_combo.currentIndexChanged.connect(
            self.on_workflow_changed
        )
        workflow_row.addWidget(self.workflow_combo)
        workflow_row.addStretch()

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.data_tab, self.tr("Data"))
        self.tab_widget.addTab(self.config_tab, self.tr("Config"))
        self.tab_widget.addTab(self.train_tab, self.tr("Train"))
        self.tab_widget.addTab(self.free_infer_tab, self.tr("Free Infer"))
        self.tab_widget.tabBar().setEnabled(False)
        main_layout = QVBoxLayout(self)
        main_layout.addLayout(workflow_row)
        main_layout.addWidget(self.tab_widget)

        self.init_data_tab()
        self.init_config_tab()
        self.init_train_tab()
        self.init_free_infer_tab()
        self._apply_workflow_ui("train")

    def on_workflow_changed(self, _index: int = 0):
        mode = self.workflow_combo.currentData() or "train"
        self._apply_workflow_ui(str(mode))

    def _apply_workflow_ui(self, mode: str):
        """Show train wizard tabs or free-infer tab only."""
        is_infer = mode == "free_infer"
        self.tab_widget.setTabVisible(0, not is_infer)
        self.tab_widget.setTabVisible(1, not is_infer)
        self.tab_widget.setTabVisible(2, not is_infer)
        self.tab_widget.setTabVisible(3, is_infer)
        if is_infer:
            self.tab_widget.setCurrentIndex(3)
        else:
            self.tab_widget.setCurrentIndex(0)
        self.tab_widget.tabBar().setEnabled(False)

    def init_free_infer_tab(self):
        layout = QVBoxLayout(self.free_infer_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        self.free_infer_panel = FreeInferPanel(self)
        self.free_infer_panel.request_scan_weights.connect(
            self.free_infer_scan_weights
        )
        self.free_infer_panel.request_start.connect(self.free_infer_start)
        self.free_infer_panel.request_download.connect(
            self.free_infer_download
        )
        self.free_infer_panel.request_cleanup.connect(self.free_infer_cleanup)
        self.free_infer_panel.mode_changed.connect(
            self._on_free_infer_mode_changed
        )
        layout.addWidget(self.free_infer_panel)
        self._free_infer_mode = "local"

    def _on_free_infer_mode_changed(self, mode: str):
        self._free_infer_mode = mode or "local"
        mapping = {"local": 0, "interpreter": 1, "remote": 2}
        idx = mapping.get(self._free_infer_mode, 0)
        if hasattr(self, "mode_combo"):
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(idx)
            self.mode_combo.blockSignals(False)
            self.on_train_mode_changed(self.mode_combo.currentText())

    def free_infer_scan_weights(self):
        mode = self.free_infer_panel.current_env_mode()
        self.free_infer_panel.set_status(self.tr("Scanning weights…"))
        try:
            if mode == "remote":
                ok, msg = self._ensure_ssh_for_free_infer()
                if not ok:
                    QMessageBox.warning(self, self.tr("Remote"), self.tr(msg))
                    self.free_infer_panel.set_status(msg)
                    return
                home = self.ssh_client.home
                entries = scan_remote_weights(
                    self.ssh_client, home, remote_base=self._remote_base_dir()
                )
            else:
                entries = scan_local_weights()
            self.free_infer_panel.set_weight_entries(entries)
            self.free_infer_panel.set_status(
                self.tr("Found {n} weight file(s).").format(n=len(entries))
            )
        except Exception as e:
            logger.exception("free_infer_scan_weights failed")
            QMessageBox.warning(self, self.tr("Scan Failed"), str(e))
            self.free_infer_panel.set_status(str(e))

    def _ensure_ssh_for_free_infer(self):
        """Connect SSH using the active / selected profile from Data tab."""
        try:
            if self.ssh_client and self.ssh_client.connected:
                return True, ""
            profile = None
            if hasattr(self, "_selected_profile"):
                profile = self._selected_profile()
            if not profile:
                profiles = load_ssh_profiles()
                profile = get_active_profile(profiles)
            if not profile:
                return (
                    False,
                    "No SSH profile configured. Open Workspace → Train → Data "
                    "and Test Connection first.",
                )
            host = (profile.get("host") or "").strip()
            user = (profile.get("username") or "").strip()
            password = profile.get("password") or ""
            try:
                port = int(profile.get("port") or 22)
            except (TypeError, ValueError):
                port = 22
            if not host or not user:
                return False, "SSH profile missing host or username."
            self.ssh_client.connect(host, port, user, password)
            return True, ""
        except Exception as e:
            return False, str(e)

    def free_infer_start(self):
        panel = self.free_infer_panel
        weight = panel.current_weight()
        material = panel.material_path()
        mode = panel.current_env_mode()
        if weight is None:
            QMessageBox.warning(
                self,
                self.tr("Free Inference"),
                self.tr("Please scan or upload a weight file first."),
            )
            return
        if not material:
            QMessageBox.warning(
                self,
                self.tr("Free Inference"),
                self.tr("Please select inference material."),
            )
            return

        job_id, job_dir = create_free_infer_job_dir("infer")
        source_dir = os.path.join(job_dir, "source")
        imgsz = panel.imgsz()
        interval = panel.video_frame_interval()

        def _progress(msg: str):
            self._bg_bridge.log.emit(str(msg))

        def _job():
            ok, kind, msg = prepare_source_images(
                material,
                source_dir,
                video_interval=interval,
                progress=_progress,
            )
            if not ok:
                return False, msg

            if mode == "remote":
                ok_ssh, err = self._ensure_ssh_for_free_infer()
                if not ok_ssh:
                    return False, err
                home = self.ssh_client.home
                if weight.source == "upload" or not weight.remote:
                    remote_run = posix_join(
                        runs_remote_dir(
                            "free_infer", home, self._remote_base_dir()
                        ),
                        job_id,
                    )
                    local_w = weight.weight_path
                    ok_pred, payload = remote_batch_predict_yolo(
                        self.ssh_client,
                        remote_run,
                        source_dir,
                        weights_file=os.path.basename(local_w),
                        imgsz=imgsz,
                        progress=_progress,
                        local_weight_to_upload=local_w,
                        remote_python=self._remote_python_for_exec(),
                    )
                else:
                    remote_run = weight.run_dir
                    ok_pred, payload = remote_batch_predict_yolo(
                        self.ssh_client,
                        remote_run,
                        source_dir,
                        weights_file=weight.weight_name,
                        imgsz=imgsz,
                        progress=_progress,
                        model_path=weight.weight_path,
                        remote_python=self._remote_python_for_exec(),
                    )
                if not ok_pred:
                    return False, payload
                local_labels = os.path.join(job_dir, "batch_infer", "labels")
                os.makedirs(local_labels, exist_ok=True)
                _progress("Downloading remote labels…")
                self.ssh_client.download_directory(
                    payload, local_labels, progress=_progress
                )
                job = FreeInferJob(
                    job_id=job_id,
                    job_dir=job_dir,
                    source_dir=source_dir,
                    source_kind=kind,
                    weight=weight,
                    imgsz=imgsz,
                    created_at=datetime.datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    labels_dir=local_labels,
                    meta={"remote_labels": payload, "remote_run": remote_run},
                )
                job.save()
                return True, {
                    "job_dir": job_dir,
                    "source_dir": source_dir,
                    "labels_dir": local_labels,
                    "kind": kind,
                }

            ok_m, model_path = resolve_model_path(weight, job_dir)
            if not ok_m:
                return False, model_path
            if weight.source == "scan" and weight.run_dir:
                run_dir = weight.run_dir
            else:
                run_dir = job_dir
            py_exe = ""
            if mode == "interpreter":
                py_exe = self._selected_interpreter_python() or ""
                if not py_exe:
                    return False, "No interpreter selected on Data tab."
            ok_pred, payload = self._run_local_batch_predict(
                run_dir,
                source_dir,
                model_path,
                imgsz,
                python_executable=py_exe,
            )
            if not ok_pred:
                return False, payload
            local_labels = payload
            job_labels = os.path.join(job_dir, "batch_infer", "labels")
            if os.path.abspath(local_labels) != os.path.abspath(job_labels):
                os.makedirs(job_labels, exist_ok=True)
                if os.path.isdir(local_labels):
                    for name in os.listdir(local_labels):
                        if name.lower().endswith(".txt"):
                            shutil.copy2(
                                os.path.join(local_labels, name),
                                os.path.join(job_labels, name),
                            )
                local_labels = job_labels
            job = FreeInferJob(
                job_id=job_id,
                job_dir=job_dir,
                source_dir=source_dir,
                source_kind=kind,
                weight=weight,
                imgsz=imgsz,
                created_at=datetime.datetime.now().isoformat(
                    timespec="seconds"
                ),
                labels_dir=local_labels,
                meta={"model_path": model_path, "run_dir": run_dir},
            )
            job.save()
            return True, {
                "job_dir": job_dir,
                "source_dir": source_dir,
                "labels_dir": local_labels,
                "kind": kind,
            }

        panel.set_busy(True)
        panel.set_status(self.tr("Preparing material and running inference…"))
        self.append_training_log(
            f"[Free Infer] start job={job_id} mode={mode} material={material}"
        )
        if not self._start_bg_task("free_infer", _job):
            panel.set_busy(False)

    def free_infer_download(self):
        panel = self.free_infer_panel
        source_dir = panel.last_source_dir()
        labels_dir = panel.last_labels_dir()
        if not source_dir or not labels_dir:
            QMessageBox.information(
                self,
                self.tr("Download"),
                self.tr("No inference results available yet."),
            )
            return
        dest = QFileDialog.getExistingDirectory(
            self, self.tr("Select download folder"), ""
        )
        if not dest:
            return
        out_dir = os.path.join(
            dest,
            f"free_infer_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        def _job():
            return package_results(source_dir, labels_dir, out_dir)

        panel.set_busy(True)
        panel.set_status(self.tr("Packaging images and labels…"))
        if not self._start_bg_task("free_infer_download", _job):
            panel.set_busy(False)

    def free_infer_cleanup(self):
        import shlex

        train_projects = list_local_train_projects()
        infer_projects = list_local_free_infer_projects()
        mode = self.free_infer_panel.current_env_mode()
        if mode == "remote":
            ok, _err = self._ensure_ssh_for_free_infer()
            if ok:
                try:
                    home = self.ssh_client.home
                    for task in (
                        "detect",
                        "classify",
                        "obb",
                        "segment",
                        "pose",
                        "free_infer",
                    ):
                        base = runs_remote_dir(
                            task, home, self._remote_base_dir()
                        )
                        try:
                            names = self.ssh_client.listdir(base)
                        except Exception:
                            continue
                        for name in sorted(names):
                            if name in (".", "..") or name.startswith("."):
                                continue
                            path = f"{base.rstrip('/')}/{name}"
                            pid = f"remote:{task}/{name}"
                            if task == "free_infer":
                                infer_projects.append((pid, path))
                            else:
                                train_projects.append((pid, path))
                except Exception as e:
                    logger.warning(f"Remote cleanup list failed: {e}")

        dialog = CleanupProjectsDialog(
            self,
            train_projects=train_projects,
            infer_projects=infer_projects,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_projects()
        errors = []
        for project_id, path in selected:
            is_remote = str(project_id).startswith("remote:") or (
                isinstance(path, str)
                and path.startswith("/")
                and not os.path.isdir(path)
            )
            if is_remote:
                try:
                    ok_ssh, err = self._ensure_ssh_for_free_infer()
                    if not ok_ssh:
                        errors.append(f"{project_id}: {err}")
                        continue
                    self.ssh_client.exec(
                        f"rm -rf {shlex.quote(path)} 2>/dev/null || true"
                    )
                    self.append_training_log(
                        f"[Cleanup] deleted remote {project_id}"
                    )
                except Exception as e:
                    errors.append(f"{project_id}: {e}")
            else:
                ok, msg = delete_local_project(path)
                if ok:
                    self.append_training_log(
                        f"[Cleanup] deleted local {project_id}"
                    )
                else:
                    errors.append(f"{project_id}: {msg}")
        if errors:
            QMessageBox.warning(
                self,
                self.tr("Cleanup"),
                self.tr("Some deletions failed:\n") + "\n".join(errors),
            )
        else:
            QMessageBox.information(
                self,
                self.tr("Cleanup"),
                self.tr("Selected project(s) deleted."),
            )

    def save_training_logs_to_file(self):
        """Save training logs to a local file with timestamp"""
        if (
            not hasattr(self, "log_display")
            or not self.log_display.toPlainText().strip()
        ):
            return

        if not os.path.exists(self.current_project_path):
            return
        log_dir_path = os.path.join(self.current_project_path, "logs")
        os.makedirs(log_dir_path, exist_ok=True)

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{self.training_status}_{timestamp}.txt"
            log_file_path = os.path.join(log_dir_path, filename)

            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write(self.log_display.toPlainText())

            logger.info(f"Training logs saved to: {log_file_path}")

        except Exception as e:
            logger.error(f"Failed to save training logs: {str(e)}")

    def closeEvent(self, event):
        """Handle window close event.

        While training is running, hide instead of stopping so the user can
        keep labeling other datasets in the main window.
        """
        if (
            self.training_manager.is_training
            and self.training_status == "training"
            and not getattr(self, "_application_closing", False)
        ):
            self.append_training_log(
                self.tr(
                    "Training window hidden; training continues in the "
                    "background. Re-open via Train menu to show this window."
                )
            )
            try:
                self._save_ssh_settings()
            except Exception:
                pass
            self.hide()
            event.ignore()
            return

        if self.training_status in ["completed", "error", "stop"]:
            self.save_training_logs_to_file()

        self.clear_cache()
        try:
            self._save_ssh_settings()
        except Exception:
            pass
        try:
            self.ssh_client.disconnect()
        except Exception:
            pass
        # Drop parent singleton reference when really closed
        parent = self.parent()
        if parent is not None and getattr(parent, "mytrain_dialog", None) is self:
            parent.mytrain_dialog = None
        super().closeEvent(event)

    def prepare_for_application_close(self):
        if self.training_manager.is_training:
            self.training_manager.detach_for_app_exit()
        elif load_local_job() or load_remote_job():
            self.training_manager.detach_for_app_exit()
        export_active = getattr(self.export_manager, "is_exporting", False)
        if export_active:
            reply = QMessageBox.question(
                self.parent() or self,
                self.tr("Export Running"),
                self.tr(
                    "An export task is still running. Stop it and exit?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False
            self.export_manager.stop_export(wait=True)
        self._application_closing = True
        return True

    def go_to_specific_tab(self, index):
        """Go to specific tab by index"""
        self.tab_widget.setCurrentIndex(index)

    # Data Tab
    def show_pose_config(self):
        """Show the pose config field"""
        if hasattr(self, "pose_config_label"):
            self.pose_config_label.setVisible(True)
            self.config_widgets["pose_config"].setVisible(True)

            for i in range(self.pose_config_layout.count()):
                widget = self.pose_config_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)

    def hide_pose_config(self):
        """Hide the pose config field"""
        if hasattr(self, "pose_config_label"):
            self.pose_config_label.setVisible(False)
            self.config_widgets["pose_config"].setVisible(False)

            for i in range(self.pose_config_layout.count()):
                widget = self.pose_config_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(False)

    def on_task_type_selected(self, task_type):
        normalized_task_type = None
        for task in TASK_TYPES:
            if task.lower() == task_type.lower():
                normalized_task_type = task
                break

        if normalized_task_type is None:
            logger.warning(f"Unknown task type: {task_type}")
            return

        task_type = normalized_task_type

        if task_type not in self.task_type_buttons:
            logger.warning(f"Task type button not found: {task_type}")
            return

        if self.selected_task_type == task_type:
            self.selected_task_type = None
            self.task_type_buttons[task_type].set_selected(False)
            self.hide_pose_config()
        else:
            if self.selected_task_type:
                self.task_type_buttons[self.selected_task_type].set_selected(
                    False
                )
            self.selected_task_type = task_type
            self.task_type_buttons[task_type].set_selected(True)

            if task_type.lower() == "pose":
                self.show_pose_config()
            else:
                self.hide_pose_config()

        self.refresh_dataset_summary()

    def create_task_handler(self, task_type):
        def handler():
            self.on_task_type_selected(task_type)

        return handler

    def init_task_configuration(self, parent_layout):
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)

        task_type_layout = QHBoxLayout()
        task_type_layout.addWidget(QLabel(self.tr("Task Type:")))
        for task_type in TASK_TYPES:
            button = CustomQPushButton(task_type)
            button.clicked.connect(self.create_task_handler(task_type))
            task_type_layout.addWidget(button)
            self.task_type_buttons[task_type] = button
            if task_type == "Detect":
                button.set_selected(True)

        task_type_layout.addStretch()
        config_layout.addLayout(task_type_layout)
        parent_layout.addWidget(config_widget)

    def init_mode_and_remote(self, parent_layout):
        group = QGroupBox(self.tr("Training Mode"))
        layout = QVBoxLayout(group)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel(self.tr("Mode:")))
        self.mode_combo = CustomComboBox()
        self.mode_combo.addItems(["Local", "Interpreter", "Remote"])
        self.mode_combo.currentTextChanged.connect(self.on_train_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # ---- Local Interpreter (Conda / custom python.exe) ----
        self.interpreter_panel = QWidget()
        interp_layout = QVBoxLayout(self.interpreter_panel)
        interp_layout.setContentsMargins(0, 8, 0, 0)
        interp_layout.setSpacing(8)

        interp_select_row = QHBoxLayout()
        interp_select_row.addWidget(QLabel(self.tr("Selected interpreter:")))
        self.interp_profile_combo = CustomComboBox()
        self.interp_profile_combo.setMinimumWidth(220)
        self.interp_profile_combo.currentIndexChanged.connect(
            self._on_interp_profile_selected
        )
        interp_select_row.addWidget(self.interp_profile_combo, 1)
        self.interp_test_btn = SecondaryButton(self.tr("Test Environment"))
        self.interp_test_btn.clicked.connect(self.test_interpreter)
        interp_select_row.addWidget(self.interp_test_btn)
        interp_layout.addLayout(interp_select_row)

        self.interp_status_label = QLabel(
            self.tr("Pick a Conda/venv python.exe for local training.")
        )
        self.interp_status_label.setWordWrap(True)
        interp_layout.addWidget(self.interp_status_label)

        interp_layout.addWidget(QLabel(self.tr("Interpreter list:")))
        interp_header = QHBoxLayout()
        interp_header.setSpacing(6)
        for text, stretch in (
            (self.tr("Name"), 2),
            (self.tr("Python path"), 5),
            ("", 0),
            ("", 0),
        ):
            lab = QLabel(text)
            lab.setStyleSheet("font-weight: 600;")
            if stretch:
                interp_header.addWidget(lab, stretch)
            else:
                lab.setFixedWidth(56)
                interp_header.addWidget(lab)
        interp_layout.addLayout(interp_header)

        self.interp_rows_host = QWidget()
        self.interp_rows_layout = QVBoxLayout(self.interp_rows_host)
        self.interp_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.interp_rows_layout.setSpacing(4)
        interp_layout.addWidget(self.interp_rows_host)

        interp_btn_row = QHBoxLayout()
        self.interp_add_btn = PrimaryButton(self.tr("Add Interpreter"))
        self.interp_add_btn.clicked.connect(self.add_interp_profile)
        self.interp_save_btn = SecondaryButton(self.tr("Save"))
        self.interp_save_btn.clicked.connect(self.save_current_interp_profile)
        interp_btn_row.addWidget(self.interp_add_btn)
        interp_btn_row.addWidget(self.interp_save_btn)
        interp_btn_row.addStretch()
        interp_layout.addLayout(interp_btn_row)

        interp_tip = QLabel(
            self.tr(
                "Tip: choose Anaconda/Miniconda env python.exe (or Browse). "
                "App and training stay separate — like Remote, but on this PC."
            )
        )
        interp_tip.setWordWrap(True)
        interp_layout.addWidget(interp_tip)

        layout.addWidget(self.interpreter_panel)
        self.interpreter_panel.setVisible(False)

        self.remote_panel = QWidget()
        remote_layout = QVBoxLayout(self.remote_panel)
        remote_layout.setContentsMargins(0, 8, 0, 0)
        remote_layout.setSpacing(8)

        # Active server selection + actions
        select_row = QHBoxLayout()
        select_row.addWidget(QLabel(self.tr("Selected server:")))
        self.ssh_profile_combo = CustomComboBox()
        self.ssh_profile_combo.setMinimumWidth(220)
        self.ssh_profile_combo.currentIndexChanged.connect(
            self._on_ssh_profile_selected
        )
        select_row.addWidget(self.ssh_profile_combo, 1)
        self.ssh_test_btn = SecondaryButton(self.tr("Test Connection"))
        self.ssh_test_btn.clicked.connect(self.test_ssh_connection)
        self.ssh_reattach_btn = SecondaryButton(self.tr("Reattach Job"))
        self.ssh_reattach_btn.clicked.connect(self.reattach_training_job)
        self.ssh_reattach_btn.setEnabled(False)
        select_row.addWidget(self.ssh_test_btn)
        select_row.addWidget(self.ssh_reattach_btn)
        remote_layout.addLayout(select_row)

        self.ssh_status_label = QLabel(self.tr("Not connected"))
        self.ssh_status_label.setWordWrap(True)
        remote_layout.addWidget(self.ssh_status_label)

        self.remote_gpu_label = QLabel(
            self.tr("GPU: (connect to probe remote GPUs)")
        )
        self.remote_gpu_label.setWordWrap(True)
        remote_layout.addWidget(self.remote_gpu_label)

        base_row = QHBoxLayout()
        base_row.addWidget(QLabel(self.tr("Remote base directory:")))
        self.remote_base_edit = CustomLineEdit()
        self.remote_base_edit.setPlaceholderText(
            self.tr("Optional. Empty uses ~/xanylabeling_datasets")
        )
        self.remote_base_edit.editingFinished.connect(
            self._sync_remote_runtime_to_profile
        )
        base_row.addWidget(self.remote_base_edit, 1)
        self.remote_base_browse_btn = SecondaryButton(self.tr("Browse"))
        self.remote_base_browse_btn.clicked.connect(self.browse_remote_base_dir)
        base_row.addWidget(self.remote_base_browse_btn)
        remote_layout.addLayout(base_row)

        runtime_row = QHBoxLayout()
        runtime_row.addWidget(QLabel(self.tr("Remote runtime:")))
        self.remote_runtime_combo = CustomComboBox()
        self.remote_runtime_combo.addItem(
            self.tr("Shell (default)"), "shell"
        )
        self.remote_runtime_combo.addItem(
            self.tr("Conda environment"), "conda"
        )
        self.remote_runtime_combo.currentIndexChanged.connect(
            self._on_remote_runtime_changed
        )
        runtime_row.addWidget(self.remote_runtime_combo)
        runtime_row.addStretch()
        remote_layout.addLayout(runtime_row)

        self.remote_conda_panel = QWidget()
        conda_layout = QVBoxLayout(self.remote_conda_panel)
        conda_layout.setContentsMargins(0, 0, 0, 0)
        conda_layout.setSpacing(6)

        conda_exe_row = QHBoxLayout()
        conda_exe_row.addWidget(QLabel(self.tr("Conda executable:")))
        self.remote_conda_exe_edit = CustomLineEdit()
        self.remote_conda_exe_edit.setPlaceholderText(
            self.tr("Optional, e.g. /home/user/miniconda3/bin/conda")
        )
        self.remote_conda_exe_edit.editingFinished.connect(
            self._sync_remote_runtime_to_profile
        )
        conda_exe_row.addWidget(self.remote_conda_exe_edit, 1)
        self.remote_conda_scan_btn = SecondaryButton(self.tr("Scan Environments"))
        self.remote_conda_scan_btn.clicked.connect(self.scan_remote_conda_envs)
        conda_exe_row.addWidget(self.remote_conda_scan_btn)
        conda_layout.addLayout(conda_exe_row)

        conda_sel_row = QHBoxLayout()
        conda_sel_row.addWidget(QLabel(self.tr("Conda env:")))
        self.remote_conda_env_combo = CustomComboBox()
        self.remote_conda_env_combo.setMinimumWidth(200)
        self.remote_conda_env_combo.currentIndexChanged.connect(
            self._on_remote_conda_env_selected
        )
        conda_sel_row.addWidget(self.remote_conda_env_combo, 1)
        self.remote_conda_test_btn = SecondaryButton(self.tr("Test Environment"))
        self.remote_conda_test_btn.clicked.connect(self.test_remote_conda_env)
        conda_sel_row.addWidget(self.remote_conda_test_btn)
        conda_layout.addLayout(conda_sel_row)

        self.remote_conda_status_label = QLabel(
            self.tr(
                "Conda mode runs training with the env's python "
                "(e.g. …/envs/yolo/bin/python). Shell mode keeps "
                "legacy remote python3/python."
            )
        )
        self.remote_conda_status_label.setWordWrap(True)
        conda_layout.addWidget(self.remote_conda_status_label)
        remote_layout.addWidget(self.remote_conda_panel)
        self.remote_conda_panel.setVisible(False)
        self._remote_conda_envs_cache = []

        # Server table: one row per server
        remote_layout.addWidget(QLabel(self.tr("Server list:")))
        header = QHBoxLayout()
        header.setSpacing(6)
        for text, stretch in (
            (self.tr("Name"), 2),
            (self.tr("IP"), 2),
            (self.tr("Port"), 1),
            (self.tr("Username"), 2),
            (self.tr("Password"), 2),
            ("", 0),
        ):
            lab = QLabel(text)
            lab.setStyleSheet("font-weight: 600;")
            if stretch:
                header.addWidget(lab, stretch)
            else:
                lab.setFixedWidth(56)
                header.addWidget(lab)
        remote_layout.addLayout(header)

        self.ssh_rows_host = QWidget()
        self.ssh_rows_layout = QVBoxLayout(self.ssh_rows_host)
        self.ssh_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.ssh_rows_layout.setSpacing(4)
        remote_layout.addWidget(self.ssh_rows_host)

        list_btn_row = QHBoxLayout()
        self.ssh_add_btn = PrimaryButton(self.tr("Add Server"))
        self.ssh_add_btn.clicked.connect(self.add_ssh_profile)
        self.ssh_save_btn = SecondaryButton(self.tr("Save"))
        self.ssh_save_btn.clicked.connect(self.save_current_ssh_profile)
        list_btn_row.addWidget(self.ssh_add_btn)
        list_btn_row.addWidget(self.ssh_save_btn)
        list_btn_row.addStretch()
        remote_layout.addLayout(list_btn_row)

        tip = QLabel(
            self.tr(
                "Tip: each row is one server. Choose the target in "
                "“Selected server”, then Test Connection. Upload, training, "
                "and remote tests use that server. Remote base directory is "
                "the root for datasets and runs; leave it empty for "
                "~/xanylabeling_datasets."
            )
        )
        tip.setWordWrap(True)
        remote_layout.addWidget(tip)

        layout.addWidget(self.remote_panel)
        self.remote_panel.setVisible(False)
        parent_layout.addWidget(group)

        self._ssh_profiles = {"version": 2, "active_id": "", "servers": []}
        self._ssh_row_widgets = []
        self._ssh_loading_profile = False
        self._load_interp_settings()
        self._load_ssh_settings()

    def _is_local_filesystem_mode(self) -> bool:
        return self.train_mode in ("local", "interpreter")

    def on_train_mode_changed(self, text: str):
        t = (text or "").strip().lower()
        if t.startswith("remote"):
            self.train_mode = "remote"
        elif t.startswith("interpreter"):
            self.train_mode = "interpreter"
        else:
            self.train_mode = "local"
        if hasattr(self, "interpreter_panel"):
            self.interpreter_panel.setVisible(self.train_mode == "interpreter")
        if hasattr(self, "remote_panel"):
            self.remote_panel.setVisible(self.train_mode == "remote")
        if self._is_local_filesystem_mode():
            self.remote_train_allowed = True
            if self.train_mode == "local":
                self.remote_gpu_result = None
                self.interpreter_probe_info = None
        self._refresh_prepare_dataset_btn()
        self._sync_remote_dataset_source_ui()
        self._refresh_exec_cwd_hint()
        if (
            self.train_mode == "remote"
            and hasattr(self, "custom_exec_cwd_edit")
            and is_windows_local_path(self.custom_exec_cwd_edit.text())
        ):
            self.custom_exec_cwd_edit.clear()

    def _interp_settings_path(self):
        return local_interp.profiles_path()

    def _clear_interp_rows(self) -> None:
        while self.interp_rows_layout.count():
            item = self.interp_rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._interp_row_widgets = []

    def _make_interp_row_widget(self, profile: dict) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        name_edit = CustomLineEdit()
        name_edit.setPlaceholderText(self.tr("Name"))
        name_edit.setText(profile.get("name") or "")
        py_edit = CustomLineEdit()
        py_edit.setPlaceholderText(
            r"D:\miniconda3\envs\yolo\python.exe"
        )
        py_edit.setText(profile.get("python") or "")

        browse_btn = SecondaryButton(self.tr("Browse"))
        browse_btn.setFixedWidth(64)

        def _browse(_checked=False, edit=py_edit):
            path, _ = QFileDialog.getOpenFileName(
                self,
                self.tr("Select python.exe"),
                edit.text().strip() or "",
                self.tr("Python (python.exe);;All files (*)"),
            )
            if path:
                edit.setText(
                    local_interp.resolve_python_executable(path)
                )
                self._on_interp_row_field_changed()

        browse_btn.clicked.connect(_browse)

        del_btn = SecondaryButton(self.tr("Del"))
        del_btn.setFixedWidth(56)
        profile_id = profile.get("id") or ""
        del_btn.clicked.connect(
            lambda *_a, pid=profile_id: self.delete_interp_profile(pid)
        )

        name_edit.editingFinished.connect(self._on_interp_row_field_changed)
        py_edit.editingFinished.connect(self._on_interp_row_field_changed)

        h.addWidget(name_edit, 2)
        h.addWidget(py_edit, 5)
        h.addWidget(browse_btn)
        h.addWidget(del_btn)

        self._interp_row_widgets.append(
            {
                "id": profile_id,
                "widget": row,
                "name": name_edit,
                "python": py_edit,
            }
        )
        return row

    def _rebuild_interp_rows(self) -> None:
        self._clear_interp_rows()
        for profile in self._interp_profiles.get("interpreters") or []:
            self.interp_rows_layout.addWidget(
                self._make_interp_row_widget(profile)
            )

    def _interp_row_to_profile(self, row: dict) -> dict:
        name = row["name"].text().strip()
        py = local_interp.resolve_python_executable(
            row["python"].text().strip()
        )
        if not name:
            name = local_interp._default_name({"python": py})
        existing = None
        for s in self._interp_profiles.get("interpreters") or []:
            if s.get("id") == row.get("id"):
                existing = s
                break
        return {
            "id": row.get("id") or local_interp.empty_profile()["id"],
            "name": name,
            "python": py,
            "last_ok": (existing or {}).get("last_ok"),
            "last_tested_at": (existing or {}).get("last_tested_at") or "",
            "last_detail": (existing or {}).get("last_detail") or "",
        }

    def _sync_interp_rows_to_store(self) -> None:
        active_id = self._interp_profiles.get("active_id") or ""
        items = [
            self._interp_row_to_profile(r) for r in self._interp_row_widgets
        ]
        self._interp_profiles["interpreters"] = items
        if active_id and any(s.get("id") == active_id for s in items):
            self._interp_profiles["active_id"] = active_id
        elif items:
            self._interp_profiles["active_id"] = items[0]["id"]
        else:
            self._interp_profiles["active_id"] = ""

    def _selected_interp_id(self) -> str:
        idx = self.interp_profile_combo.currentIndex()
        if idx < 0:
            return self._interp_profiles.get("active_id") or ""
        return self.interp_profile_combo.itemData(idx) or ""

    def _selected_interp_profile(self) -> dict:
        self._sync_interp_rows_to_store()
        pid = self._selected_interp_id()
        for s in self._interp_profiles.get("interpreters") or []:
            if s.get("id") == pid:
                return s
        return local_interp.get_active_profile(self._interp_profiles) or {}

    def _selected_interpreter_python(self) -> str:
        if self.train_mode != "interpreter":
            return ""
        profile = self._selected_interp_profile()
        return local_interp.resolve_python_executable(
            (profile or {}).get("python") or ""
        )

    def _refresh_interp_profile_combo(self, select_id: str = "") -> None:
        select_id = (
            select_id
            or self._interp_profiles.get("active_id")
            or self._selected_interp_id()
            or ""
        )
        self._interp_loading = True
        try:
            self.interp_profile_combo.clear()
            for profile in self._interp_profiles.get("interpreters") or []:
                self.interp_profile_combo.addItem(
                    local_interp.profile_display_name(profile),
                    profile.get("id"),
                )
            if not self.interp_profile_combo.count():
                return
            idx = 0
            for i in range(self.interp_profile_combo.count()):
                if self.interp_profile_combo.itemData(i) == select_id:
                    idx = i
                    break
            self.interp_profile_combo.setCurrentIndex(idx)
            self._interp_profiles["active_id"] = (
                self.interp_profile_combo.itemData(idx) or ""
            )
        finally:
            self._interp_loading = False

    def _on_interp_row_field_changed(self) -> None:
        if self._interp_loading:
            return
        active = self._selected_interp_id()
        self._sync_interp_rows_to_store()
        self._refresh_interp_profile_combo(active)

    def _load_interp_settings(self):
        self._interp_profiles = local_interp.load_interpreter_profiles(
            self._interp_settings_path()
        )
        if not (self._interp_profiles.get("interpreters") or []):
            profile = local_interp.empty_profile(
                name=self.tr("Interpreter 1")
            )
            local_interp.upsert_profile(self._interp_profiles, profile)
            self._interp_profiles["active_id"] = profile["id"]
        self._rebuild_interp_rows()
        self._refresh_interp_profile_combo(
            self._interp_profiles.get("active_id") or ""
        )

    def _persist_interp_profiles(self) -> None:
        try:
            local_interp.save_interpreter_profiles(
                self._interp_profiles, self._interp_settings_path()
            )
        except Exception as e:
            logger.warning("Failed to save interpreter profiles: %s", e)

    def _save_interp_settings(self):
        self._sync_interp_rows_to_store()
        active = self._selected_interp_id()
        if active:
            self._interp_profiles["active_id"] = active
        self._persist_interp_profiles()
        self._refresh_interp_profile_combo(active)

    def _on_interp_profile_selected(self, index: int) -> None:
        if self._interp_loading or index < 0:
            return
        self._sync_interp_rows_to_store()
        profile_id = self.interp_profile_combo.itemData(index)
        if not profile_id:
            return
        self._interp_profiles["active_id"] = profile_id
        self._persist_interp_profiles()
        profile = None
        for s in self._interp_profiles.get("interpreters") or []:
            if s.get("id") == profile_id:
                profile = s
                break
        self.interp_status_label.setText(
            self.tr("Selected: %s — click Test Environment before train.")
            % local_interp.profile_display_name(profile or {"name": profile_id})
        )
        self.interpreter_probe_info = None

    def add_interp_profile(self):
        self._sync_interp_rows_to_store()
        n = len(self._interp_profiles.get("interpreters") or []) + 1
        profile = local_interp.empty_profile(
            name=self.tr("Interpreter %d") % n
        )
        local_interp.upsert_profile(self._interp_profiles, profile)
        if not self._interp_profiles.get("active_id"):
            self._interp_profiles["active_id"] = profile["id"]
        self._persist_interp_profiles()
        self._rebuild_interp_rows()
        self._refresh_interp_profile_combo(
            self._interp_profiles.get("active_id")
        )
        self.interp_status_label.setText(
            self.tr("Added a row — set python.exe, select it, then Test.")
        )

    def delete_interp_profile(self, profile_id: str = ""):
        self._sync_interp_rows_to_store()
        profile_id = profile_id or self._selected_interp_id()
        items = self._interp_profiles.get("interpreters") or []
        if len(items) <= 1:
            QMessageBox.information(
                self,
                self.tr("Interpreter"),
                self.tr("At least one interpreter profile is required."),
            )
            return
        target = None
        for s in items:
            if s.get("id") == profile_id:
                target = s
                break
        name = (target or {}).get("name") or profile_id
        reply = QMessageBox.question(
            self,
            self.tr("Delete Interpreter"),
            self.tr("Delete interpreter “%s”?") % name,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        local_interp.delete_profile(self._interp_profiles, profile_id)
        self._persist_interp_profiles()
        self._rebuild_interp_rows()
        self._refresh_interp_profile_combo(
            self._interp_profiles.get("active_id") or ""
        )
        self.interpreter_probe_info = None
        self.interp_status_label.setText(
            self.tr("Profile deleted.")
        )

    def save_current_interp_profile(self):
        self._sync_interp_rows_to_store()
        if not (self._interp_profiles.get("interpreters") or []):
            QMessageBox.warning(
                self,
                self.tr("Interpreter"),
                self.tr("No interpreters to save."),
            )
            return
        self._save_interp_settings()
        active = local_interp.get_active_profile(self._interp_profiles) or {}
        self.interp_status_label.setText(
            self.tr("Saved %d interpreter(s). Selected: %s")
            % (
                len(self._interp_profiles.get("interpreters") or []),
                local_interp.profile_display_name(active) if active else "-",
            )
        )
        QMessageBox.information(
            self,
            self.tr("Interpreter"),
            self.tr("Interpreter list saved."),
        )

    def _apply_interpreter_probe_devices(self, info: dict) -> None:
        """Refresh device combo / CUDA checkboxes from probe payload."""
        self.interpreter_probe_info = dict(info or {})
        n = int(info.get("device_count") or 0) if info.get("cuda") else 0
        if n <= 0:
            return
        if "device" not in self.config_widgets:
            return
        combo = self.config_widgets["device"]
        if combo.findText("cuda") < 0:
            return
        combo.setCurrentText("cuda")
        self.on_device_changed("cuda")
        if hasattr(self, "_cuda_layout") and self._cuda_layout:
            if self._cuda_layout.count() < n:
                self.setup_cuda_checkboxes(n)
            for i in range(self._cuda_layout.count()):
                w = self._cuda_layout.itemAt(i).widget()
                if w and hasattr(w, "setChecked"):
                    w.setChecked(i == 0)

    def test_interpreter(self):
        """Probe ultralytics/torch in the selected local interpreter."""
        profile = self._selected_interp_profile()
        if not profile:
            QMessageBox.warning(
                self,
                self.tr("Interpreter"),
                self.tr("Please select an interpreter first."),
            )
            return
        py = local_interp.resolve_python_executable(
            profile.get("python") or ""
        )
        if not py or not os.path.isfile(py):
            QMessageBox.warning(
                self,
                self.tr("Interpreter"),
                self.tr(
                    "Python executable not found. Browse to python.exe "
                    "(or a Conda env folder), then Test again."
                ),
            )
            return

        self._interp_profiles["active_id"] = profile["id"]
        profile["python"] = py
        local_interp.upsert_profile(self._interp_profiles, profile)

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            ok, msg, info = local_interp.probe_interpreter(
                py,
                require_ultralytics=self._custom_probe_require_ultralytics(),
            )
            local_interp.mark_test_result(
                self._interp_profiles, profile["id"], ok, msg
            )
            self._persist_interp_profiles()
            self._refresh_interp_profile_combo(profile["id"])
            if ok:
                self.interp_status_label.setText(
                    self.tr("Ready · %s\n%s")
                    % (local_interp.profile_display_name(profile), msg)
                )
                self._apply_interpreter_probe_devices(info)
                box = QMessageBox.information
                if not info.get("ultralytics") and not self._custom_probe_require_ultralytics():
                    box = QMessageBox.warning
                box(self, self.tr("Interpreter"), msg)
            else:
                self.interpreter_probe_info = None
                self.interp_status_label.setText(
                    self.tr("Failed · %s\n%s")
                    % (profile.get("name") or py, msg)
                )
                QMessageBox.critical(
                    self,
                    self.tr("Interpreter"),
                    self.tr(
                        "Environment check failed.\n\n%s\n\n"
                        "Install in that env:\n"
                        "  pip install torch (ultralytics optional for custom training)"
                    )
                    % msg,
                )
        finally:
            QApplication.restoreOverrideCursor()

    def _ensure_selected_interpreter_ready(self) -> bool:
        """Gate Start Training for Interpreter mode."""
        if self.train_mode != "interpreter":
            return True
        profile = self._selected_interp_profile()
        if not profile:
            QMessageBox.warning(
                self,
                self.tr("Interpreter"),
                self.tr("Please add and select a local interpreter."),
            )
            return False
        py = local_interp.resolve_python_executable(
            profile.get("python") or ""
        )
        if not py or not os.path.isfile(py):
            QMessageBox.warning(
                self,
                self.tr("Interpreter"),
                self.tr(
                    "Selected interpreter python.exe was not found:\n%s"
                )
                % (profile.get("python") or "(empty)"),
            )
            return False

        # Prefer a recent successful Test; otherwise probe now.
        if profile.get("last_ok") is True:
            return True

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            ok, msg, info = local_interp.probe_interpreter(
                py,
                require_ultralytics=self._custom_probe_require_ultralytics(),
            )
            local_interp.mark_test_result(
                self._interp_profiles, profile["id"], ok, msg
            )
            self._persist_interp_profiles()
            self._refresh_interp_profile_combo(profile["id"])
            if ok:
                self.interp_status_label.setText(
                    self.tr("Ready · %s\n%s")
                    % (local_interp.profile_display_name(profile), msg)
                )
                self._apply_interpreter_probe_devices(info)
                return True
            QMessageBox.critical(
                self,
                self.tr("Interpreter"),
                self.tr(
                    "Cannot start training — environment check failed:\n\n%s"
                )
                % msg,
            )
            return False
        finally:
            QApplication.restoreOverrideCursor()

    def _ssh_settings_path(self):
        return profiles_path()

    def _clear_ssh_rows(self) -> None:
        while self.ssh_rows_layout.count():
            item = self.ssh_rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._ssh_row_widgets = []

    def _make_ssh_row_widget(self, profile: dict) -> QWidget:
        from PyQt6.QtWidgets import QLineEdit

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        name_edit = CustomLineEdit()
        name_edit.setPlaceholderText(self.tr("Name"))
        name_edit.setText(profile.get("name") or "")
        host_edit = CustomLineEdit()
        host_edit.setPlaceholderText("192.168.1.10")
        host_edit.setText(profile.get("host") or "")
        port_edit = CustomLineEdit()
        port_edit.setPlaceholderText("22")
        port_edit.setText(str(profile.get("port") or 22))
        port_edit.setMaximumWidth(72)
        user_edit = CustomLineEdit()
        user_edit.setPlaceholderText("root")
        user_edit.setText(profile.get("username") or "")
        pass_edit = CustomLineEdit()
        pass_edit.setPlaceholderText(self.tr("Password"))
        pass_edit.setText(profile.get("password") or "")
        try:
            pass_edit.setEchoMode(QLineEdit.EchoMode.Password)
        except Exception:
            pass

        del_btn = SecondaryButton(self.tr("Del"))
        del_btn.setFixedWidth(56)
        profile_id = profile.get("id") or ""
        del_btn.clicked.connect(
            lambda *_a, pid=profile_id: self.delete_ssh_profile(pid)
        )

        # Sync name changes into the Selected-server combo labels
        for edit in (name_edit, host_edit, port_edit, user_edit, pass_edit):
            edit.editingFinished.connect(self._on_ssh_row_field_changed)

        h.addWidget(name_edit, 2)
        h.addWidget(host_edit, 2)
        h.addWidget(port_edit, 1)
        h.addWidget(user_edit, 2)
        h.addWidget(pass_edit, 2)
        h.addWidget(del_btn)

        self._ssh_row_widgets.append(
            {
                "id": profile_id,
                "widget": row,
                "name": name_edit,
                "host": host_edit,
                "port": port_edit,
                "user": user_edit,
                "password": pass_edit,
            }
        )
        return row

    def _rebuild_ssh_rows(self) -> None:
        self._clear_ssh_rows()
        for profile in self._ssh_profiles.get("servers") or []:
            self.ssh_rows_layout.addWidget(self._make_ssh_row_widget(profile))

    def _row_to_profile(self, row: dict) -> dict:
        name = row["name"].text().strip()
        host = row["host"].text().strip()
        user = row["user"].text().strip()
        if not name:
            name = f"{user}@{host}" if host and user else (host or "Server")
        try:
            port = int(row["port"].text().strip() or 22)
        except ValueError:
            port = 22
        existing = None
        for s in self._ssh_profiles.get("servers") or []:
            if s.get("id") == row.get("id"):
                existing = s
                break
        kept = existing or {}
        return {
            "id": row.get("id") or empty_profile()["id"],
            "name": name,
            "host": host,
            "port": port,
            "username": user,
            "password": row["password"].text(),
            "last_ok": kept.get("last_ok"),
            "last_tested_at": kept.get("last_tested_at") or "",
            "remote_runtime": kept.get("remote_runtime") or "shell",
            "conda_executable": kept.get("conda_executable") or "",
            "conda_env_name": kept.get("conda_env_name") or "",
            "remote_python": kept.get("remote_python") or "",
            "remote_base_dir": kept.get("remote_base_dir") or "",
        }

    def _sync_rows_to_store(self) -> None:
        """Write all table rows back into ``_ssh_profiles`` (keep active_id)."""
        active_id = self._ssh_profiles.get("active_id") or ""
        servers = [self._row_to_profile(r) for r in self._ssh_row_widgets]
        self._ssh_profiles["servers"] = servers
        if active_id and any(s.get("id") == active_id for s in servers):
            self._ssh_profiles["active_id"] = active_id
        elif servers:
            self._ssh_profiles["active_id"] = servers[0]["id"]
        else:
            self._ssh_profiles["active_id"] = ""

    def _selected_profile_id(self) -> str:
        idx = self.ssh_profile_combo.currentIndex()
        if idx < 0:
            return self._ssh_profiles.get("active_id") or ""
        return self.ssh_profile_combo.itemData(idx) or ""

    def _selected_profile(self) -> dict:
        self._sync_rows_to_store()
        pid = self._selected_profile_id()
        for s in self._ssh_profiles.get("servers") or []:
            if s.get("id") == pid:
                return s
        return get_active_profile(self._ssh_profiles) or {}

    def _refresh_ssh_profile_combo(self, select_id: str = "") -> None:
        select_id = (
            select_id
            or self._ssh_profiles.get("active_id")
            or self._selected_profile_id()
            or ""
        )
        self._ssh_loading_profile = True
        try:
            self.ssh_profile_combo.clear()
            for profile in self._ssh_profiles.get("servers") or []:
                self.ssh_profile_combo.addItem(
                    profile_display_name(profile), profile.get("id")
                )
            if not self.ssh_profile_combo.count():
                return
            idx = 0
            for i in range(self.ssh_profile_combo.count()):
                if self.ssh_profile_combo.itemData(i) == select_id:
                    idx = i
                    break
            self.ssh_profile_combo.setCurrentIndex(idx)
            self._ssh_profiles["active_id"] = (
                self.ssh_profile_combo.itemData(idx) or ""
            )
        finally:
            self._ssh_loading_profile = False

    def _on_ssh_row_field_changed(self) -> None:
        if self._ssh_loading_profile:
            return
        active = self._selected_profile_id()
        self._sync_rows_to_store()
        self._sync_remote_runtime_to_profile()
        self._refresh_ssh_profile_combo(active)

    def _load_ssh_settings(self):
        self._ssh_profiles = load_ssh_profiles(self._ssh_settings_path())
        if not (self._ssh_profiles.get("servers") or []):
            profile = empty_profile(name=self.tr("Server 1"))
            upsert_profile(self._ssh_profiles, profile)
            self._ssh_profiles["active_id"] = profile["id"]
        self._rebuild_ssh_rows()
        self._refresh_ssh_profile_combo(
            self._ssh_profiles.get("active_id") or ""
        )
        profile = self._selected_profile()
        if profile:
            self._load_remote_runtime_from_profile(profile)

    def _persist_ssh_profiles(self) -> None:
        try:
            save_ssh_profiles(self._ssh_profiles, self._ssh_settings_path())
        except Exception as e:
            logger.warning("Failed to save SSH profiles: %s", e)

    def _save_ssh_settings(self):
        self._sync_remote_runtime_to_profile()
        self._sync_rows_to_store()
        active = self._selected_profile_id()
        if active:
            self._ssh_profiles["active_id"] = active
        self._persist_ssh_profiles()
        self._refresh_ssh_profile_combo(active)

    def _disconnect_ssh_ui(self, status: str = "") -> None:
        try:
            self.ssh_client.disconnect()
        except Exception:
            pass
        self.ssh_status_label.setText(
            status or self.tr("Not connected")
        )
        self.ssh_reattach_btn.setEnabled(False)
        self.remote_train_allowed = False
        self.remote_gpu_result = None
        if hasattr(self, "remote_gpu_label"):
            self.remote_gpu_label.setText(
                self.tr("GPU: (connect to probe remote GPUs)")
            )

    def _remote_runtime_kind(self) -> str:
        if hasattr(self, "remote_runtime_combo"):
            idx = self.remote_runtime_combo.currentIndex()
            if idx >= 0:
                kind = self.remote_runtime_combo.itemData(idx)
                if kind in ("shell", "conda"):
                    return kind
        profile = self._selected_profile()
        return (
            "conda"
            if str(profile.get("remote_runtime") or "").strip() == "conda"
            else "shell"
        )

    def _remote_python_for_exec(self):
        """Python path for remote train/export/infer (per server profile)."""
        if self._remote_runtime_kind() != "conda":
            return None
        py = (self._selected_profile().get("remote_python") or "").strip()
        return py or None

    def _conda_python_from_combo(self) -> str:
        if not hasattr(self, "remote_conda_env_combo"):
            return ""
        idx = self.remote_conda_env_combo.currentIndex()
        if idx < 0:
            return ""
        data = self.remote_conda_env_combo.itemData(idx)
        if isinstance(data, dict):
            return str(data.get("python") or "").strip()
        return ""

    def _ensure_remote_conda_ready(self) -> bool:
        if self._remote_runtime_kind() != "conda":
            return True
        self._sync_remote_runtime_to_profile()
        py = (self._selected_profile().get("remote_python") or "").strip()
        if not py:
            py = self._conda_python_from_combo()
        if not py:
            QMessageBox.warning(
                self,
                self.tr("Conda"),
                self.tr(
                    "Conda runtime is selected. Scan environments, pick one, "
                    "and click Test Environment before training or inference."
                ),
            )
            return False
        return True

    def _remote_base_dir(self) -> str:
        if hasattr(self, "remote_base_edit"):
            return self.remote_base_edit.text().strip()
        profile = self._selected_profile() or {}
        return str(profile.get("remote_base_dir") or "").strip()

    def _apply_remote_project_dir(self) -> None:
        if not self.ssh_client.connected or "project" not in self.config_widgets:
            return
        project = runs_remote_dir(
            self.selected_task_type or "detect",
            self.ssh_client.home,
            self._remote_base_dir(),
        )
        self.config_widgets["project"].setText(project)

    def browse_remote_base_dir(self) -> None:
        if not self.ssh_client.connected:
            QMessageBox.warning(
                self,
                self.tr("SSH"),
                self.tr("Connect to the remote server first (Test Connection)."),
            )
            return
        start = self._remote_base_dir() or (self.ssh_client.home or "/")
        dialog = RemoteDirBrowserDialog(self, self.ssh_client, start)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        path = dialog.selected_path().strip()
        if path:
            self.remote_base_edit.setText(path)
            self._sync_remote_runtime_to_profile()

    def _on_remote_runtime_changed(self, _index: int = 0) -> None:
        if getattr(self, "_ssh_loading_profile", False):
            return
        kind = self._remote_runtime_kind()
        if hasattr(self, "remote_conda_panel"):
            self.remote_conda_panel.setVisible(kind == "conda")
        self._sync_remote_runtime_to_profile()

    def _sync_remote_runtime_to_profile(self) -> None:
        if getattr(self, "_ssh_loading_profile", False):
            return
        pid = self._selected_profile_id()
        if not pid:
            return
        runtime = self._remote_runtime_kind()
        conda_exe = ""
        conda_env = ""
        remote_py = ""
        if hasattr(self, "remote_conda_exe_edit"):
            conda_exe = self.remote_conda_exe_edit.text().strip()
        data = None
        if hasattr(self, "remote_conda_env_combo"):
            idx = self.remote_conda_env_combo.currentIndex()
            if idx >= 0:
                data = self.remote_conda_env_combo.itemData(idx)
        if isinstance(data, dict):
            conda_env = str(data.get("name") or "")
            remote_py = str(data.get("python") or "")
        for s in self._ssh_profiles.get("servers") or []:
            if s.get("id") == pid:
                s["remote_runtime"] = runtime
                s["conda_executable"] = conda_exe
                s["conda_env_name"] = conda_env
                s["remote_python"] = remote_py
                if hasattr(self, "remote_base_edit"):
                    s["remote_base_dir"] = self.remote_base_edit.text().strip()
                break
        self._persist_ssh_profiles()

    def _load_remote_runtime_from_profile(self, profile: dict) -> None:
        if not profile or not hasattr(self, "remote_runtime_combo"):
            return
        blocking = getattr(self, "_ssh_loading_profile", False)
        self._ssh_loading_profile = True
        runtime = profile.get("remote_runtime") or "shell"
        conda_exe = profile.get("conda_executable") or ""
        base_dir = profile.get("remote_base_dir") or ""
        name = profile.get("conda_env_name") or ""
        py = profile.get("remote_python") or ""
        try:
            if hasattr(self, "remote_conda_exe_edit"):
                self.remote_conda_exe_edit.setText(conda_exe)
            if hasattr(self, "remote_base_edit"):
                self.remote_base_edit.setText(base_dir)
            self.remote_conda_env_combo.blockSignals(True)
            self.remote_runtime_combo.blockSignals(True)
            self.remote_conda_env_combo.clear()
            if name or py:
                label = f"{name}  ({py})" if name and py else (name or py)
                self.remote_conda_env_combo.addItem(
                    label, {"name": name, "python": py}
                )
            combo_idx = 1 if runtime == "conda" else 0
            self.remote_runtime_combo.setCurrentIndex(combo_idx)
            if hasattr(self, "remote_conda_panel"):
                self.remote_conda_panel.setVisible(runtime == "conda")
        finally:
            self.remote_conda_env_combo.blockSignals(False)
            self.remote_runtime_combo.blockSignals(False)
            self._ssh_loading_profile = blocking

    def scan_remote_conda_envs(self) -> None:
        if not self.ssh_client.connected:
            QMessageBox.warning(
                self,
                self.tr("SSH"),
                self.tr("Connect to the remote server first (Test Connection)."),
            )
            return
        user_path = ""
        if hasattr(self, "remote_conda_exe_edit"):
            user_path = self.remote_conda_exe_edit.text().strip()
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            conda_exe, detail = resolve_remote_conda_executable(
                self.ssh_client, user_path
            )
            if not conda_exe:
                QMessageBox.warning(self, self.tr("Conda"), detail)
                return
            if hasattr(self, "remote_conda_exe_edit"):
                self.remote_conda_exe_edit.setText(conda_exe)
            envs, msg = list_remote_conda_envs(self.ssh_client, conda_exe)
            self._remote_conda_envs_cache = envs
            self.remote_conda_env_combo.clear()
            for env in envs:
                self.remote_conda_env_combo.addItem(
                    env.display,
                    {"name": env.name, "python": env.python},
                )
            self.remote_conda_status_label.setText(
                self.tr("%s\n%s") % (detail, msg)
            )
            self._sync_remote_runtime_to_profile()
        except SSHError as e:
            QMessageBox.critical(self, self.tr("Conda"), str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def _on_remote_conda_env_selected(self, _index: int = 0) -> None:
        if self._ssh_loading_profile:
            return
        self._sync_remote_runtime_to_profile()

    def test_remote_conda_env(self) -> None:
        if not self.ssh_client.connected:
            QMessageBox.warning(
                self,
                self.tr("SSH"),
                self.tr("Connect to the remote server first."),
            )
            return
        data = None
        idx = self.remote_conda_env_combo.currentIndex()
        if idx >= 0:
            data = self.remote_conda_env_combo.itemData(idx)
        py = ""
        if isinstance(data, dict):
            py = (data.get("python") or "").strip()
        if not py:
            QMessageBox.warning(
                self,
                self.tr("Conda"),
                self.tr("Select a conda environment first."),
            )
            return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            ok, message, _info = probe_remote_python(self.ssh_client, py)
            if ok:
                self.remote_conda_status_label.setText(message)
                QMessageBox.information(
                    self, self.tr("Conda"), message
                )
            else:
                self.remote_conda_status_label.setText(message)
                QMessageBox.warning(self, self.tr("Conda"), message)
            self._sync_remote_runtime_to_profile()
        except SSHError as e:
            QMessageBox.critical(self, self.tr("Conda"), str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def _on_ssh_profile_selected(self, index: int) -> None:
        if self._ssh_loading_profile or index < 0:
            return
        self._sync_rows_to_store()
        profile_id = self.ssh_profile_combo.itemData(index)
        if not profile_id:
            return
        self._ssh_profiles["active_id"] = profile_id
        self._persist_ssh_profiles()

        profile = None
        for s in self._ssh_profiles.get("servers") or []:
            if s.get("id") == profile_id:
                profile = s
                break
        if profile:
            self._load_remote_runtime_from_profile(profile)
        connected_host = getattr(self.ssh_client, "host", "") or ""
        if (
            self.ssh_client.connected
            and profile
            and connected_host == (profile.get("host") or "")
        ):
            self.ssh_status_label.setText(
                self.tr("Selected · %s (still connected)")
                % profile_display_name(profile)
            )
            return

        self._disconnect_ssh_ui(
            self.tr("Selected: %s (not connected — Test Connection)")
            % profile_display_name(profile or {"name": profile_id})
        )

    def add_ssh_profile(self):
        self._sync_rows_to_store()
        n = len(self._ssh_profiles.get("servers") or []) + 1
        profile = empty_profile(name=self.tr("Server %d") % n)
        upsert_profile(self._ssh_profiles, profile)
        # Keep current selection unless empty
        if not self._ssh_profiles.get("active_id"):
            self._ssh_profiles["active_id"] = profile["id"]
        self._persist_ssh_profiles()
        self._rebuild_ssh_rows()
        self._refresh_ssh_profile_combo(self._ssh_profiles.get("active_id"))
        self.ssh_status_label.setText(
            self.tr("Added a server row — fill fields, select it, then Test.")
        )

    def delete_ssh_profile(self, profile_id: str = ""):
        self._sync_rows_to_store()
        profile_id = profile_id or self._selected_profile_id()
        servers = self._ssh_profiles.get("servers") or []
        if len(servers) <= 1:
            QMessageBox.information(
                self,
                self.tr("SSH"),
                self.tr("At least one server profile is required."),
            )
            return
        target = None
        for s in servers:
            if s.get("id") == profile_id:
                target = s
                break
        name = (target or {}).get("name") or (target or {}).get("host") or profile_id
        reply = QMessageBox.question(
            self,
            self.tr("Delete Server"),
            self.tr("Delete server “%s”?") % name,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        was_active = self._ssh_profiles.get("active_id") == profile_id
        delete_profile(self._ssh_profiles, profile_id)
        self._persist_ssh_profiles()
        self._rebuild_ssh_rows()
        self._refresh_ssh_profile_combo(
            self._ssh_profiles.get("active_id") or ""
        )
        if was_active or (
            getattr(self.ssh_client, "host", "")
            == ((target or {}).get("host") or "")
        ):
            self._disconnect_ssh_ui(
                self.tr("Profile deleted. Not connected.")
            )

    def save_current_ssh_profile(self):
        self._sync_rows_to_store()
        if not (self._ssh_profiles.get("servers") or []):
            QMessageBox.warning(
                self, self.tr("SSH"), self.tr("No servers to save.")
            )
            return
        self._save_ssh_settings()
        active = get_active_profile(self._ssh_profiles) or {}
        self.ssh_status_label.setText(
            self.tr("Saved %d server(s). Selected: %s")
            % (
                len(self._ssh_profiles.get("servers") or []),
                profile_display_name(active) if active else "-",
            )
        )
        QMessageBox.information(
            self,
            self.tr("SSH"),
            self.tr("Server list saved."),
        )

    def test_ssh_connection(self):
        """Test SSH for the currently selected server only."""
        profile = self._selected_profile()
        if not profile:
            QMessageBox.warning(
                self, self.tr("SSH"), self.tr("Please select a server first.")
            )
            return
        host = (profile.get("host") or "").strip()
        user = (profile.get("username") or "").strip()
        password = profile.get("password") or ""
        try:
            port = int(profile.get("port") or 22)
        except (TypeError, ValueError):
            port = 22
        if not host or not user:
            QMessageBox.warning(
                self,
                self.tr("SSH"),
                self.tr(
                    "Selected server is missing host or username. "
                    "Fill the row, then Test again."
                ),
            )
            return

        # Ensure combo / active_id point at this profile before train uses it
        self._ssh_profiles["active_id"] = profile["id"]
        upsert_profile(self._ssh_profiles, profile)

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.ssh_client.connect(host, port, user, password)
            self.ssh_status_label.setText(
                self.tr("Connected · %s (%s)")
                % (profile.get("name") or host, self.ssh_client.home)
            )
            mark_test_result(self._ssh_profiles, profile["id"], True)
            self._persist_ssh_profiles()
            self._refresh_ssh_profile_combo(profile["id"])
            self._refresh_reattach_button()

            self._probe_and_apply_remote_gpu(show_popup=True)

            job = load_remote_job()
            status, msg = probe_remote_job(self.ssh_client, job)
            if status == "alive":
                reply = QMessageBox.question(
                    self,
                    self.tr("Remote Job Found"),
                    self.tr(
                        "A remote training job is still running:\n%s\n\n"
                        "Reattach to continue watching logs?"
                    )
                    % msg,
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.reattach_training_job()
                    return
        except SSHError as e:
            mark_test_result(self._ssh_profiles, profile["id"], False)
            self._persist_ssh_profiles()
            self._refresh_ssh_profile_combo(profile["id"])
            self.ssh_status_label.setText(
                self.tr("Connection failed · %s")
                % (profile.get("name") or host)
            )
            self.ssh_reattach_btn.setEnabled(False)
            self.remote_train_allowed = False
            self.remote_gpu_result = None
            if hasattr(self, "remote_gpu_label"):
                self.remote_gpu_label.setText(self.tr("GPU: (not connected)"))
            QMessageBox.critical(self, self.tr("SSH"), str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def _current_model_path(self) -> str:
        if "model" in self.config_widgets:
            return self.config_widgets["model"].text().strip()
        return ""

    def _current_imgsz(self) -> int:
        try:
            if "imgsz" in self.config_widgets:
                return int(self.config_widgets["imgsz"].value())
        except Exception:
            pass
        return int(DEFAULT_TRAINING_CONFIG.get("imgsz", 640))

    def _probe_and_apply_remote_gpu(self, *, show_popup: bool = False) -> bool:
        """Query remote nvidia-smi; update UI and gate training."""
        if not self.ssh_client.connected:
            self.remote_train_allowed = False
            return False

        result = probe_remote_gpus(
            self.ssh_client,
            model_path=self._current_model_path(),
            imgsz=self._current_imgsz(),
        )
        self.remote_gpu_result = result
        self.remote_train_allowed = bool(result.ok and result.allow_train)

        if hasattr(self, "remote_gpu_label"):
            self.remote_gpu_label.setText(result.summary_text())

        if self.remote_train_allowed:
            self._apply_remote_gpu_recommendations(result)
            if show_popup:
                QMessageBox.information(
                    self,
                    self.tr("Remote GPU"),
                    self.tr(
                        "GPU check OK.\n\n%s\n\n"
                        "Batch / image size were adjusted for remaining VRAM "
                        "(reserve kept for other users)."
                    )
                    % result.summary_text(),
                )
        elif show_popup:
            detail = result.summary_text()
            QMessageBox.warning(
                self,
                self.tr("Insufficient GPU Memory"),
                self.tr(
                    "Remote free VRAM is not enough for training "
                    "(need at least ~%.1f GB usable after reserve).\n\n"
                    "%s\n\n"
                    "You can still upload the dataset, but starting training "
                    "is blocked until enough GPU memory is available."
                )
                % (MIN_USABLE_MIB / 1024.0, detail),
            )
        return self.remote_train_allowed

    def _apply_remote_gpu_recommendations(self, result) -> None:
        if not result or not result.allow_train:
            return
        if "batch" in self.config_widgets and result.recommend_batch >= 1:
            self.config_widgets["batch"].setValue(int(result.recommend_batch))
        if "imgsz" in self.config_widgets and result.recommend_imgsz:
            self.config_widgets["imgsz"].setValue(int(result.recommend_imgsz))
        if (
            "device" in self.config_widgets
            and result.selected_index is not None
        ):
            combo = self.config_widgets["device"]
            idx = combo.findText("cuda")
            if idx >= 0:
                combo.setCurrentIndex(idx)
            self.on_device_changed("cuda")
            if hasattr(self, "_cuda_layout") and self._cuda_layout:
                need = int(result.selected_index) + 1
                if self._cuda_layout.count() < need:
                    self.setup_cuda_checkboxes(max(need, 1))
                for i in range(self._cuda_layout.count()):
                    w = self._cuda_layout.itemAt(i).widget()
                    if w and hasattr(w, "setChecked"):
                        w.setChecked(i == int(result.selected_index))

    def _ensure_remote_train_allowed(self) -> bool:
        if self.train_mode != "remote":
            return True
        # Refresh dropped idle sessions before GPU / train gates
        try:
            if not self.ssh_client.is_alive():
                profile = self._selected_profile()
                if profile.get("host") and profile.get("username"):
                    self.ssh_client.connect(
                        profile["host"],
                        int(profile.get("port") or 22),
                        profile.get("username") or "",
                        profile.get("password") or "",
                    )
                else:
                    self.ssh_client.ensure_connected()
                if hasattr(self, "ssh_status_label"):
                    self.ssh_status_label.setText(
                        self.tr("Reconnected · %s (%s)")
                        % (
                            profile.get("name")
                            or self.ssh_client.host
                            or "",
                            self.ssh_client.home,
                        )
                    )
            else:
                self.ssh_client.ensure_connected()
        except SSHError as e:
            QMessageBox.warning(
                self,
                self.tr("SSH"),
                self.tr(
                    "SSH session expired or dropped.\n\n%s\n\n"
                    "Please Test Connection again, then start training."
                )
                % str(e),
            )
            return False

        if not self.ssh_client.connected:
            QMessageBox.warning(
                self,
                self.tr("SSH"),
                self.tr("Please connect to the remote server first."),
            )
            return False

        allowed = self._probe_and_apply_remote_gpu(show_popup=False)
        if allowed:
            return True

        result = self.remote_gpu_result
        detail = result.summary_text() if result else ""
        # SSH / probe transport failures are not a VRAM problem
        if result is not None and not result.ok:
            QMessageBox.warning(
                self,
                self.tr("Remote GPU Check Failed"),
                self.tr(
                    "Could not check remote GPU status (SSH/command failed).\n\n"
                    "%s\n\n"
                    "The previous connection may have timed out. "
                    "Test Connection again, then retry training."
                )
                % detail,
            )
            return False

        QMessageBox.warning(
            self,
            self.tr("Insufficient GPU Memory"),
            self.tr(
                "Cannot start remote training: not enough free GPU memory "
                "for other users' safety.\n\n%s\n\n"
                "Dataset upload is still allowed. Free GPU memory or wait, "
                "then Test Connection again."
            )
            % detail,
        )
        return False

    def _refresh_reattach_button(self):
        enabled = False
        lst, _lmsg = probe_local_job(load_local_job())
        if lst == "alive":
            enabled = True
        elif self.ssh_client.connected:
            job = load_remote_job()
            status, _msg = probe_remote_job(self.ssh_client, job)
            enabled = status == "alive"
        self.ssh_reattach_btn.setEnabled(enabled)

    def _apply_training_job_to_ui(self, job: dict) -> None:
        if not job:
            return
        if job.get("data"):
            self.data_yaml_path = job.get("data") or ""
            if "data" in self.config_widgets:
                self.config_widgets["data"].setText(self.data_yaml_path)
        if job.get("project") and "project" in self.config_widgets:
            self.config_widgets["project"].setText(job["project"])
        if job.get("name") and "name" in self.config_widgets:
            self.config_widgets["name"].setText(job["name"])
        if job.get("tensorboard_url"):
            self.tensorboard_url = job["tensorboard_url"]
        project = (job.get("project") or "").replace("\\", "/")
        name = job.get("name") or "exp"
        self.current_project_path = posix_join(project, name)
        self._remote_run_dir_path = self.current_project_path

    def _load_finished_run_into_ui(self, job: dict) -> None:
        self._apply_training_job_to_ui(job)
        self.go_to_specific_tab(2)
        run_dir = job.get("run_dir") or self.current_project_path or ""
        log_path = job.get("log_path") or ""
        if log_path and os.path.isfile(log_path):
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    tail = f.read()[-120_000:]
                if tail.strip():
                    text = format_train_log_terminal(
                        latest_log_session_text(tail)
                    )
                    if text.strip() and hasattr(self, "log_display"):
                        self._set_training_log_mirror(text)
            except OSError:
                pass
        self.training_status = "completed"
        self.update_training_status_display()
        self._set_post_train_buttons(visible=True)
        self.progress_timer.stop()
        self.image_timer.stop()
        self.update_training_progress()
        self.update_training_images()
        clear_local_job()

    def _ssh_upload_progress(self, msg: str, status_widget=None) -> None:
        text = str(msg).strip()
        if not text:
            return
        if status_widget is not None:
            status_widget.setText(self.tr("Uploading: %s") % text)
            QApplication.processEvents()
        self.append_training_log(text)

    def _prompt_saved_training_sessions(self) -> None:
        if self.training_status == "training":
            return
        local = load_local_job()
        lst, lmsg = probe_local_job(local)
        if lst == "alive":
            reply = QMessageBox.question(
                self,
                self.tr("Training Job Found"),
                self.tr(
                    "A local training job is still running:\n%s\n\n"
                    "Reattach to continue watching logs?"
                )
                % lmsg,
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.reattach_training_job()
            return
        if lst == "finished" and local:
            reply = QMessageBox.question(
                self,
                self.tr("Previous Training"),
                self.tr(
                    "The last local training session has finished:\n%s\n\n"
                    "Open that run and show logs/metrics?"
                )
                % lmsg,
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._load_finished_run_into_ui(local)
            return

        remote = load_remote_job()
        if not remote:
            self._refresh_reattach_button()
            return
        if self.ssh_client.connected:
            rst, rmsg = probe_remote_job(self.ssh_client, remote)
            if rst == "alive":
                reply = QMessageBox.question(
                    self,
                    self.tr("Remote Job Found"),
                    self.tr(
                        "A remote training job is still running:\n%s\n\n"
                        "Reattach to continue watching logs?"
                    )
                    % rmsg,
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.reattach_training_job()
                return
            if rst == "finished":
                reply = QMessageBox.question(
                    self,
                    self.tr("Previous Remote Training"),
                    self.tr(
                        "The last remote training session has finished:\n%s\n\n"
                        "Restore project paths and show saved log offset?"
                    )
                    % rmsg,
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._load_finished_run_into_ui(remote)
        else:
            self.append_training_log(
                self.tr(
                    "Saved remote training metadata found. "
                    "Connect SSH and use Reattach Job if training is still running."
                )
            )
        self._refresh_reattach_button()

    def reattach_training_job(self):
        lst, _ = probe_local_job(load_local_job())
        if lst == "alive":
            if self.training_status == "training":
                QMessageBox.information(
                    self,
                    self.tr("Training"),
                    self.tr("Already attached to a running training session."),
                )
                return
            job = load_local_job() or {}
            self._apply_training_job_to_ui(job)
            self.train_mode = (
                "interpreter"
                if job.get("python")
                and job.get("python") != __import__("sys").executable
                else "local"
            )
            success, message = self.training_manager.attach_local_training()
            if not success:
                QMessageBox.warning(self, self.tr("Reattach"), message)
                self._refresh_reattach_button()
                return
            self.go_to_specific_tab(2)
            self.append_training_log(message)
            if self.tensorboard_url and hasattr(self, "open_tb_button"):
                self.open_tb_button.setVisible(True)
            return
        self.reattach_remote_job()

    def reattach_remote_job(self):
        if not self.ssh_client.connected:
            QMessageBox.warning(
                self,
                self.tr("SSH"),
                self.tr("Please connect to the remote server first."),
            )
            return
        if self.training_status == "training":
            QMessageBox.information(
                self,
                self.tr("Training"),
                self.tr("Already attached to a running training session."),
            )
            return

        job = load_remote_job() or {}
        self._apply_training_job_to_ui(job)
        self.train_mode = "remote"
        if hasattr(self, "mode_combo"):
            idx = self.mode_combo.findText("Remote")
            if idx >= 0:
                self.mode_combo.setCurrentIndex(idx)

        success, message = self.training_manager.attach_remote_training(
            self.ssh_client
        )
        if not success:
            QMessageBox.warning(self, self.tr("Reattach"), message)
            self._refresh_reattach_button()
            return

        self.go_to_specific_tab(2)
        self.append_training_log(message)
        if self.tensorboard_url and hasattr(self, "open_tb_button"):
            self.open_tb_button.setVisible(True)

    def init_dataset_root_panel(self, parent_layout):
        group = QGroupBox(self.tr("Detect Dataset (make_dataset layout)"))
        layout = QVBoxLayout(group)

        tip = QLabel(
            self.tr(
                "Select ROOT containing classes.txt and sub-versions "
                "(each with train/val/test/all.txt and images/)."
            )
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)

        self.dataset_remote_source_row = QWidget()
        src_layout = QHBoxLayout(self.dataset_remote_source_row)
        src_layout.setContentsMargins(0, 0, 0, 0)
        self.rb_dataset_upload = QRadioButton(
            self.tr("Upload local dataset to remote")
        )
        self.rb_dataset_remote = QRadioButton(
            self.tr("Use existing dataset on remote (no upload)")
        )
        self.rb_dataset_upload.setChecked(True)
        self._dataset_source_group = QButtonGroup(self)
        self._dataset_source_group.addButton(self.rb_dataset_upload)
        self._dataset_source_group.addButton(self.rb_dataset_remote)
        src_layout.addWidget(self.rb_dataset_upload)
        src_layout.addWidget(self.rb_dataset_remote)
        src_layout.addStretch()
        self._dataset_source_group.buttonToggled.connect(
            self._on_remote_dataset_source_changed
        )
        layout.addWidget(self.dataset_remote_source_row)

        root_row = QHBoxLayout()
        self.dataset_root_edit = CustomLineEdit()
        self.dataset_root_edit.setReadOnly(True)
        self.dataset_root_browse_btn = SecondaryButton(
            self.tr("Open Dataset Folder")
        )
        self.dataset_root_browse_btn.clicked.connect(self.browse_dataset_root)
        root_row.addWidget(self.dataset_root_edit)
        root_row.addWidget(self.dataset_root_browse_btn)
        layout.addLayout(root_row)

        layout.addWidget(QLabel(self.tr("Sub-dataset versions:")))
        self.subdir_list_widget = QWidget()
        self.subdir_list_layout = QVBoxLayout(self.subdir_list_widget)
        self.subdir_list_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.subdir_list_widget)

        self.dataset_status_label = QLabel("")
        self.dataset_status_label.setWordWrap(True)
        layout.addWidget(self.dataset_status_label)

        btn_row = QHBoxLayout()
        self.prepare_dataset_btn = PrimaryButton(
            self.tr("Generate Dataset Lists")
        )
        self.prepare_dataset_btn.clicked.connect(self.prepare_dataset_lists)
        btn_row.addWidget(self.prepare_dataset_btn)
        self.upload_pause_btn = SecondaryButton(self.tr("Pause Upload"))
        self.upload_pause_btn.clicked.connect(self.pause_dataset_upload)
        self.upload_pause_btn.setVisible(False)
        btn_row.addWidget(self.upload_pause_btn)
        self.upload_resume_btn = SecondaryButton(self.tr("Resume Upload"))
        self.upload_resume_btn.clicked.connect(self.resume_dataset_upload)
        self.upload_resume_btn.setVisible(False)
        self.upload_resume_btn.setEnabled(False)
        btn_row.addWidget(self.upload_resume_btn)
        self.upload_stop_btn = SecondaryButton(self.tr("Stop Upload"))
        self.upload_stop_btn.clicked.connect(self.stop_dataset_upload)
        self.upload_stop_btn.setVisible(False)
        btn_row.addWidget(self.upload_stop_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        parent_layout.addWidget(group)
        self._sync_remote_dataset_source_ui()

    def _remote_dataset_use_existing(self) -> bool:
        return (
            self.train_mode == "remote"
            and getattr(self, "remote_dataset_source", "upload") == "remote"
        )

    def _refresh_prepare_dataset_btn(self) -> None:
        if not hasattr(self, "prepare_dataset_btn"):
            return
        if self.train_mode != "remote":
            self.prepare_dataset_btn.setText(self.tr("Generate Dataset Lists"))
        elif self._remote_dataset_use_existing():
            self.prepare_dataset_btn.setText(self.tr("Generate on Server"))
        else:
            self.prepare_dataset_btn.setText(self.tr("Upload & Generate"))

    def _sync_remote_dataset_source_ui(self) -> None:
        remote = self.train_mode == "remote"
        if hasattr(self, "dataset_remote_source_row"):
            self.dataset_remote_source_row.setVisible(remote)
        if not remote:
            return
        existing = self._remote_dataset_use_existing()
        if hasattr(self, "dataset_root_browse_btn"):
            self.dataset_root_browse_btn.setText(
                self.tr("Select Remote ROOT…")
                if existing
                else self.tr("Open Dataset Folder")
            )
        if hasattr(self, "dataset_root_edit"):
            self.dataset_root_edit.setReadOnly(existing)

    def _on_remote_dataset_source_changed(self, *_args) -> None:
        self.remote_dataset_source = (
            "remote" if self.rb_dataset_remote.isChecked() else "upload"
        )
        self.data_yaml_path = ""
        if self.remote_dataset_source == "remote":
            self.dataset_root = ""
        else:
            self.remote_dataset_root = ""
            self.dataset_root_edit.clear()
        self.dataset_status_label.setText("")
        self._sync_remote_dataset_source_ui()
        self._refresh_prepare_dataset_btn()
        self.clear_cache()
        self.refresh_dataset_summary()

    def _set_upload_controls(self, active: bool, paused: bool = False) -> None:
        self.prepare_dataset_btn.setEnabled(not active)
        for btn in (
            self.upload_pause_btn,
            self.upload_resume_btn,
            self.upload_stop_btn,
        ):
            btn.setVisible(active)
        self.upload_pause_btn.setEnabled(active and not paused)
        self.upload_resume_btn.setEnabled(active and paused)
        self.upload_stop_btn.setEnabled(active)

    def pause_dataset_upload(self) -> None:
        if not self._dataset_upload_running:
            return
        self._upload_gate.pause()
        self._set_upload_controls(True, paused=True)
        self.dataset_status_label.setText(
            self.tr("Upload paused. Click Resume Upload to continue.")
        )

    def resume_dataset_upload(self) -> None:
        if not self._dataset_upload_running:
            return
        self._upload_gate.resume()
        self._set_upload_controls(True, paused=False)
        self.dataset_status_label.setText(self.tr("Uploading dataset..."))

    def stop_dataset_upload(self) -> None:
        if not self._dataset_upload_running:
            return
        self._upload_gate.stop()
        self.upload_pause_btn.setEnabled(False)
        self.upload_resume_btn.setEnabled(False)
        self.upload_stop_btn.setEnabled(False)
        self.dataset_status_label.setText(self.tr("Stopping upload..."))

    def _on_dataset_upload_progress(self, message: str) -> None:
        text = str(message or "").strip()
        if text and hasattr(self, "dataset_status_label"):
            self.dataset_status_label.setText(text)

    def _begin_dataset_upload(self, root: str, subdirs) -> None:
        self._dataset_upload_running = True
        self._upload_gate = UploadGate()
        self._set_upload_controls(True, paused=False)
        self.dataset_status_label.setText(self.tr("Uploading dataset..."))
        remote_base = self._remote_base_dir()
        remote_python = self._remote_python_for_exec() or ""
        gate = self._upload_gate
        client = self.ssh_client

        def _progress(message: str) -> None:
            self._upload_bridge.log.emit(str(message))

        def _progress_state(
            overall_pct: float,
            detail: str,
            subdir: str,
            sub_pct: float,
        ) -> None:
            if subdir and sub_pct >= 0:
                msg = self.tr(
                    "Overall {overall:.0f}% · {subdir} {subpct:.0f}% — {detail}"
                ).format(
                    overall=overall_pct,
                    subdir=subdir,
                    subpct=sub_pct,
                    detail=detail,
                )
            else:
                msg = self.tr("Overall {overall:.0f}% — {detail}").format(
                    overall=overall_pct,
                    detail=detail,
                )
            self._upload_bridge.log.emit(msg)

        def _run() -> None:
            try:
                remote_root, skipped = upload_dataset(
                    client,
                    root,
                    subdirs=subdirs,
                    progress_state=_progress_state,
                    remote_base=remote_base,
                    remote_python=remote_python,
                    control=gate,
                )
                if skipped:
                    _progress_state(
                        100.0,
                        self.tr("Remote dataset unchanged, skipped upload: %s")
                        % remote_root,
                        "",
                        -1.0,
                    )
                else:
                    _progress_state(
                        100.0,
                        self.tr("Merging lists on remote..."),
                        "",
                        -1.0,
                    )
                gate.checkpoint()
                ok2, info = remote_merge_lists(
                    client,
                    remote_root,
                    subdirs,
                    remote_python=remote_python or None,
                )
                self._upload_bridge.finished.emit(
                    "dataset_upload",
                    bool(ok2),
                    {
                        "remote": remote_root,
                        "skipped": skipped,
                        "info": info,
                        "stopped": False,
                    },
                )
            except UploadStopped:
                self._upload_bridge.finished.emit(
                    "dataset_upload",
                    False,
                    {"stopped": True},
                )
            except Exception as e:
                self._upload_bridge.finished.emit(
                    "dataset_upload",
                    False,
                    {"stopped": False, "error": str(e)},
                )

        threading.Thread(
            target=_run, daemon=True, name="xal-dataset-upload"
        ).start()

    def _begin_remote_merge_only(self, remote_root: str, subdirs) -> None:
        if self._dataset_upload_running:
            return
        self._dataset_upload_running = True
        self._set_upload_controls(False)
        self.dataset_status_label.setText(self.tr("Merging lists on remote..."))
        remote_python = self._remote_python_for_exec() or ""
        client = self.ssh_client

        def _run() -> None:
            try:
                ok, info = remote_merge_lists(
                    client,
                    remote_root,
                    subdirs,
                    remote_python=remote_python or None,
                )
                self._upload_bridge.finished.emit(
                    "dataset_upload",
                    bool(ok),
                    {
                        "remote": remote_root,
                        "skipped": True,
                        "info": info,
                        "stopped": False,
                        "remote_only": True,
                    },
                )
            except Exception as e:
                self._upload_bridge.finished.emit(
                    "dataset_upload",
                    False,
                    {"stopped": False, "error": str(e)},
                )

        threading.Thread(
            target=_run, daemon=True, name="xal-remote-merge"
        ).start()

    def _on_dataset_upload_finished(self, _task: str, ok: bool, payload) -> None:
        self._dataset_upload_running = False
        self._set_upload_controls(False)
        data = payload if isinstance(payload, dict) else {"error": str(payload)}
        if data.get("stopped"):
            self.dataset_status_label.setText(
                self.tr(
                    "Upload stopped. Click Generate Dataset Lists to upload again."
                )
            )
            return
        if data.get("error"):
            self.dataset_status_label.setText(str(data["error"]))
            QMessageBox.critical(
                self, self.tr("Upload Error"), str(data["error"])
            )
            return
        if not ok:
            info = str(data.get("info") or "")
            self.dataset_status_label.setText(info[:500])
            QMessageBox.critical(self, self.tr("Remote merge failed"), info)
            return
        remote_root = data.get("remote") or ""
        info = data.get("info") or ""
        self.remote_dataset_root = remote_root
        self.data_yaml_path = info
        self.dataset_status_label.setText(
            self.tr("Remote dataset ready:\n%s\n%s") % (remote_root, info)
        )
        self.clear_cache()
        self.refresh_dataset_summary()
        if data.get("remote_only"):
            done = self.tr(
                "Generated data.yaml on the server from the remote dataset "
                "(no upload)."
            )
        elif data.get("skipped"):
            done = self.tr(
                "Dataset unchanged, upload skipped. data.yaml is ready on the server."
            )
        else:
            done = self.tr(
                "Uploaded selected sub-datasets and generated data.yaml on the server."
            )
        QMessageBox.information(self, self.tr("Success"), done)

    def browse_dataset_root(self):
        if self._remote_dataset_use_existing():
            self.browse_remote_dataset_root()
            return
        dir_path = QFileDialog.getExistingDirectory(
            self, self.tr("Select Dataset Root Directory"), ""
        )
        if not dir_path:
            return
        self.dataset_root = dir_path
        self.dataset_root_edit.setText(dir_path)
        self._rebuild_subdir_checks(dir_path)
        self.data_yaml_path = ""
        self.remote_dataset_root = ""
        self.dataset_status_label.setText("")
        self.clear_cache()
        self.refresh_dataset_summary()

    def browse_remote_dataset_root(self) -> None:
        if not self._ensure_selected_server_ready():
            return
        home = getattr(self.ssh_client, "home", "") or ""
        default = resolve_remote_base(self._remote_base_dir(), home)
        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr("Remote Dataset ROOT"))
        dlg.setMinimumWidth(520)
        lay = QVBoxLayout(dlg)
        hint = QLabel(
            self.tr(
                "Absolute path on the server containing classes.txt and "
                "sub-versions (each with images/, labels/, and split lists)."
            )
        )
        hint.setWordWrap(True)
        lay.addWidget(hint)
        path_edit = CustomLineEdit()
        path_edit.setText(
            (self.remote_dataset_root or "").strip() or default
        )
        lay.addWidget(path_edit)
        btn_row = QHBoxLayout()
        ok = PrimaryButton(self.tr("OK"))
        cancel = SecondaryButton(self.tr("Cancel"))
        btn_row.addStretch()
        btn_row.addWidget(cancel)
        btn_row.addWidget(ok)
        lay.addLayout(btn_row)
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        remote_root = path_edit.text().strip().replace("\\", "/")
        if not remote_root:
            QMessageBox.warning(
                self, self.tr("Dataset"), self.tr("Please enter a remote path.")
            )
            return
        remote_python = self._remote_python_for_exec() or ""
        ok_probe, err, subs = probe_remote_dataset_subdirs(
            self.ssh_client,
            remote_root,
            remote_python=remote_python,
        )
        if not ok_probe:
            QMessageBox.warning(self, self.tr("Dataset"), err)
            return
        self.remote_dataset_root = remote_root
        self.dataset_root = ""
        self.dataset_root_edit.setText(remote_root)
        self._rebuild_subdir_checks_from_names(subs)
        self.data_yaml_path = ""
        self.dataset_status_label.setText(
            self.tr("Remote ROOT ready:\n%s") % remote_root
        )
        self.clear_cache()
        self.refresh_dataset_summary()

    def _rebuild_subdir_checks_from_names(self, names: list) -> None:
        while self.subdir_list_layout.count():
            item = self.subdir_list_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self.subdir_checkboxes = {}
        if not names:
            self.subdir_list_layout.addWidget(
                QLabel(self.tr("No sub-dataset folders found under ROOT."))
            )
            return
        for name in names:
            cb = CustomCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_subdir_selection_changed)
            self.subdir_checkboxes[name] = cb
            self.subdir_list_layout.addWidget(cb)

    def _rebuild_subdir_checks(self, root: str):
        self._rebuild_subdir_checks_from_names(list_candidate_subdirs(root))

    def _on_subdir_selection_changed(self, *_args):
        # ROOT/all.txt (if present) already covers all versions; otherwise
        # recompute from checked subdir all.txt files.
        self.clear_cache()
        self.refresh_dataset_summary()

    def _selected_subdirs(self):
        return [
            name
            for name, cb in self.subdir_checkboxes.items()
            if cb.isChecked()
        ]

    def _ensure_selected_server_ready(self) -> bool:
        """Require Selected-server dropdown match the live SSH connection."""
        if self.train_mode != "remote":
            return True
        self._sync_rows_to_store()
        profile = self._selected_profile()
        if not profile:
            QMessageBox.warning(
                self,
                self.tr("SSH"),
                self.tr("Please select a server from the dropdown."),
            )
            return False

        sel_host = (profile.get("host") or "").strip()
        # Prefer silent reconnect if session dropped but credentials known
        if not self.ssh_client.is_alive():
            try:
                self.ssh_client.connect(
                    sel_host,
                    int(profile.get("port") or 22),
                    profile.get("username") or "",
                    profile.get("password") or "",
                )
            except SSHError as e:
                QMessageBox.warning(
                    self,
                    self.tr("SSH"),
                    self.tr(
                        "SSH session expired.\n\n%s\n\n"
                        "Please Test Connection, then retry."
                    )
                    % str(e),
                )
                return False

        cur_host = (getattr(self.ssh_client, "host", "") or "").strip()
        if sel_host and cur_host and sel_host != cur_host:
            QMessageBox.warning(
                self,
                self.tr("SSH"),
                self.tr(
                    "Selected server (%s) is not the connected host (%s).\n"
                    "Choose it in “Selected server” and Test Connection again."
                )
                % (sel_host, cur_host),
            )
            return False
        self._ssh_profiles["active_id"] = profile.get("id") or ""
        self._sync_remote_runtime_to_profile()
        self._persist_ssh_profiles()
        if not self._ensure_remote_conda_ready():
            return False
        return True

    def prepare_dataset_lists(self):
        if self.selected_task_type != "Detect":
            QMessageBox.warning(
                self,
                self.tr("Task Type"),
                self.tr(
                    "Dataset list workflow currently supports Detect only."
                ),
            )
            return
        subdirs = self._selected_subdirs()
        if self.train_mode == "remote" and self._remote_dataset_use_existing():
            if self._dataset_upload_running:
                return
            if not self._ensure_selected_server_ready():
                return
            remote_root = (self.remote_dataset_root or "").strip()
            if not remote_root:
                QMessageBox.warning(
                    self,
                    self.tr("Dataset"),
                    self.tr("Please select a remote dataset ROOT first."),
                )
                return
            remote_python = self._remote_python_for_exec() or ""
            ok, err = validate_remote_dataset_root(
                self.ssh_client,
                remote_root,
                subdirs,
                remote_python=remote_python,
            )
            if not ok:
                QMessageBox.warning(self, self.tr("Dataset"), err)
                return
            self._save_ssh_settings()
            self._begin_remote_merge_only(remote_root, subdirs)
            return

        root = self.dataset_root
        ok, err = validate_dataset_root(root, subdirs)
        if not ok:
            QMessageBox.warning(self, self.tr("Dataset"), err)
            return

        if self.train_mode == "remote":
            if self._dataset_upload_running:
                return
            if not self._ensure_selected_server_ready():
                return
            self._save_ssh_settings()
            self._begin_dataset_upload(root, subdirs)
            return

        result = merge_dataset_lists(root, subdirs)
        if not result.get("ok"):
            QMessageBox.warning(
                self, self.tr("Dataset"), result.get("error", "Failed")
            )
            return
        self.data_yaml_path = result["yaml_path"]
        counts = result["counts"]
        self.dataset_status_label.setText(
            self.tr(
                "Generated locally: %s\ntrain/val/test/all = %d/%d/%d/%d"
            )
            % (
                self.data_yaml_path,
                counts["train"],
                counts["val"],
                counts["test"],
                counts["all"],
            )
        )
        try:
            self.names = load_classes(os.path.join(root, "classes.txt"))
        except Exception:
            self.names = result.get("classes") or []
        self.clear_cache()
        self.refresh_dataset_summary()
        QMessageBox.information(
            self,
            self.tr("Success"),
            self.tr("Dataset lists and data.yaml generated."),
        )

    def refresh_dataset_summary(self):
        """Refresh summary from training dataset files, not the main UI folder.

        Detect: ROOT/classes.txt + ROOT/all.txt (or selected subdir all.txt)
        and matching YOLO ``labels/*.txt``. Classify still uses legacy image_list.
        """
        if self.selected_task_type == "Detect":
            if self.train_mode == "remote" and self._remote_dataset_use_existing():
                if not self.remote_dataset_root:
                    self.summary_table.clear()
                    return
                self.summary_table.load_data(
                    [
                        ["Class", "ID", "Instances", "Images"],
                        [
                            self.tr(
                                "(Summary requires local ROOT; generate on "
                                "server when using remote dataset only.)"
                            ),
                            "",
                            "",
                            "",
                        ],
                    ]
                )
                return
            if not self.dataset_root:
                self.summary_table.clear()
                return
            table_data = get_detect_dataset_summary_table(
                self.dataset_root, self._selected_subdirs() or None
            )
            self.summary_table.load_data(table_data)
            return

        if not self.image_list:
            self.summary_table.clear()
            return

        if self.selected_task_type == "Classify":
            table_data = self._get_classification_table_data()
        else:
            table_data = self._get_detection_table_data()

        self.summary_table.load_data(table_data)

    def _get_classification_table_data(self):
        if self._classification_cache is None:
            self._classification_cache = self._compute_classification_data()
        return self._classification_cache

    def _get_detection_table_data(self):
        if self._detection_cache is None:
            self._detection_cache = self._compute_detection_data()
        return self._detection_cache

    def _compute_classification_data(self):
        headers = ["Label"] + self.supported_shape + ["Total"]

        # Get classification statistics
        classify_shapes = TASK_SHAPE_MAPPINGS.get("Classify", ["flags"])
        label_infos = get_label_infos(
            self.image_list, classify_shapes, self.output_dir
        )
        if not label_infos:
            return [headers]

        table_data = [headers]
        total_counts = [0] * len(self.supported_shape)
        total_images = 0

        for label, infos in sorted(label_infos.items()):
            # All shape columns are 0 for classification
            shape_counts = [0] * len(self.supported_shape)
            image_count = infos.get("_total", 0)
            total_images += image_count

            row = [label] + [str(c) for c in shape_counts] + [str(image_count)]
            table_data.append(row)

        total_row = (
            ["Total"] + [str(c) for c in total_counts] + [str(total_images)]
        )
        table_data.append(total_row)

        return table_data

    def _compute_detection_data(self):
        return get_statistics_table_data(
            self.image_list, self.supported_shape, self.output_dir
        )

    def clear_cache(self):
        self._classification_cache = None
        self._detection_cache = None

    def load_images(self):
        self.parent().open_folder_dialog()
        self.image_list = self.parent().image_list
        self.clear_cache()
        self.refresh_dataset_summary()

    def init_dataset_summary(self, parent_layout):
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.addWidget(QLabel(self.tr("Dataset Summary:")))

        self.summary_table = CustomTable()
        summary_layout.addWidget(self.summary_table)
        parent_layout.addWidget(summary_widget, 1)

    def proceed_to_config(self):
        if self.selected_task_type != "Detect":
            is_valid, error_message = validate_task_requirements(
                self.selected_task_type, self.image_list, self.output_dir
            )
            if not is_valid:
                QMessageBox.warning(
                    self, self.tr("Validation Error"), error_message
                )
                return
        else:
            if not self.data_yaml_path:
                QMessageBox.warning(
                    self,
                    self.tr("Dataset"),
                    self.tr(
                        "Please generate dataset lists (data.yaml) first."
                    ),
                )
                return
            if self.train_mode == "remote" and not self._ensure_selected_server_ready():
                return

        project = os.path.join(
            get_default_project_dir(),
            (self.selected_task_type or "detect").lower(),
        )
        if self.train_mode == "remote" and self.ssh_client.connected:
            project = runs_remote_dir(
                self.selected_task_type or "detect",
                self.ssh_client.home,
                self._remote_base_dir(),
            )
        self.config_widgets["project"].setText(project)
        self.config_widgets["project"].setReadOnly(self.project_readonly)
        if self.data_yaml_path:
            self.config_widgets["data"].setText(self.data_yaml_path)
            if self._is_local_filesystem_mode() and os.path.isfile(
                self.data_yaml_path
            ):
                is_valid, result = validate_data_file(self.data_yaml_path)
                if is_valid:
                    self.names = result

        if self.selected_task_type == "Detect":
            if hasattr(self, "dataset_ratio_label"):
                self.config_widgets["dataset_ratio"].setVisible(False)
                self.dataset_ratio_label.setVisible(False)

        self.go_to_specific_tab(1)

    def init_actions(self, parent_layout):
        actions_layout = QHBoxLayout()

        self.load_images_button = SecondaryButton(
            self.tr("Load Images (legacy)")
        )
        self.load_images_button.clicked.connect(self.load_images)
        actions_layout.addWidget(self.load_images_button)
        actions_layout.addStretch()

        self.next_button = PrimaryButton(self.tr("Next"))
        self.next_button.clicked.connect(self.proceed_to_config)
        actions_layout.addWidget(self.next_button)
        parent_layout.addLayout(actions_layout)

    def init_data_tab(self):
        layout = QVBoxLayout(self.data_tab)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        self.init_mode_and_remote(scroll_layout)
        self.init_task_configuration(scroll_layout)
        self.init_dataset_root_panel(scroll_layout)
        self.init_dataset_summary(scroll_layout)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        self.init_actions(layout)

    # Config Tab
    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Model File"),
            "",
            "Model Files (*.pt);;All Files (*)",
        )
        if file_path:
            self.config_widgets["model"].setText(file_path)

    def browse_data_file(self):
        if self.selected_task_type == "Classify":
            dir_path = QFileDialog.getExistingDirectory(
                self, self.tr("Select Classification Dataset Directory"), ""
            )
            if dir_path:
                self.config_widgets["data"].setText(dir_path)
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                self.tr("Select Data File"),
                "",
                "Text Files (*.yaml);;All Files (*)",
            )
            if file_path:
                is_valid, result = validate_data_file(file_path)
                if is_valid:
                    self.config_widgets["data"].setText(file_path)
                    self.names = result
                    logger.info(f"Data file loaded successfully: {file_path}")
                else:
                    QMessageBox.warning(
                        self, self.tr("Invalid Data File"), result
                    )
                    self.config_widgets["data"].clear()
                    self.names = []

    def browse_pose_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Pose Config File"),
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if file_path:
            self.config_widgets["pose_config"].setText(file_path)

    def setup_cuda_checkboxes(self, device_count):
        if not hasattr(self, "_cuda_layout") or not self._cuda_layout:
            if self.device_checkboxes.layout() is None:
                self._cuda_layout = QHBoxLayout(self.device_checkboxes)
            else:
                self._cuda_layout = self.device_checkboxes.layout()
            self._cuda_layout.setContentsMargins(0, 0, 0, 0)
            self._cuda_layout.setSpacing(5)
        else:
            while self._cuda_layout.count():
                child = self._cuda_layout.takeAt(0)
                if child.widget():
                    child.widget().setParent(None)

        for i in range(device_count):
            checkbox = CustomCheckBox(f"GPU {i}")
            checkbox.setMaximumHeight(20)
            checkbox.setChecked(True)  # Default check all GPUs
            self._cuda_layout.addWidget(checkbox)

    def on_device_changed(self, device_text):
        if device_text == "cuda":
            device_count = 0
            # Remote training: use probed remote GPU count (local torch may be CPU-only)
            if self.train_mode == "remote" and self.remote_gpu_result:
                gpus = getattr(self.remote_gpu_result, "gpus", None) or []
                if gpus:
                    device_count = len(gpus)
                elif self.remote_gpu_result.selected_index is not None:
                    device_count = int(self.remote_gpu_result.selected_index) + 1
            # Interpreter: App may be CPU-only — use probe of external env
            elif (
                self.train_mode == "interpreter"
                and self.interpreter_probe_info
            ):
                info = self.interpreter_probe_info
                if info.get("cuda"):
                    device_count = int(info.get("device_count") or 0)
            if device_count <= 0:
                try:
                    import torch

                    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
                        cuda_visible_devices_backup = os.environ.get(
                            "CUDA_VISIBLE_DEVICES"
                        )
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                        torch.cuda.empty_cache()
                        device_count = torch.cuda.device_count()
                        if cuda_visible_devices_backup != "-1":
                            os.environ["CUDA_VISIBLE_DEVICES"] = (
                                cuda_visible_devices_backup
                            )
                    else:
                        device_count = torch.cuda.device_count()
                except ImportError:
                    device_count = 0

            # Remote/interpreter cuda with unknown count: still show at least GPU 0
            if device_count <= 0 and self.train_mode in (
                "remote",
                "interpreter",
            ):
                device_count = 1

            if device_count > 0:
                self.setup_cuda_checkboxes(device_count)
                self.device_checkboxes.setVisible(True)
            else:
                self.device_checkboxes.setVisible(False)
        else:
            self.device_checkboxes.setVisible(False)

    def _resolve_train_device(self, device_value):
        """Map UI device selection to ultralytics device arg.

        Local CPU-only PyTorch previously fell back to ``cpu`` whenever no
        GPU checkbox existed, even if the user chose ``cuda`` for remote.
        """
        if device_value != "cuda":
            return device_value

        selected_gpus = []
        if hasattr(self, "_cuda_layout") and self._cuda_layout:
            for i in range(self._cuda_layout.count()):
                widget = self._cuda_layout.itemAt(i).widget()
                if (
                    widget
                    and hasattr(widget, "isChecked")
                    and widget.isChecked()
                ):
                    gpu_text = widget.text()
                    try:
                        selected_gpus.append(int(gpu_text.split()[-1]))
                    except (ValueError, IndexError):
                        selected_gpus.append(i)

        if selected_gpus:
            if self.train_mode == "interpreter":
                info = self.interpreter_probe_info or {}
                if not info.get("cuda"):
                    return "cpu"
            elif self.train_mode == "local":
                try:
                    import torch

                    if not torch.cuda.is_available():
                        return "cpu"
                except ImportError:
                    return "cpu"
            return selected_gpus

        # Remote / Interpreter: prefer probed GPU, else default cuda:0
        if self.train_mode == "remote":
            if (
                self.remote_gpu_result
                and self.remote_gpu_result.selected_index is not None
            ):
                return [int(self.remote_gpu_result.selected_index)]
            return [0]
        if self.train_mode == "interpreter":
            return [0]

        return "cpu"

    def _is_custom_training_backend(self) -> bool:
        return getattr(self, "training_backend", "ultralytics") == "custom"

    def _custom_probe_require_ultralytics(self) -> bool:
        return not self._is_custom_training_backend()

    def init_training_backend_selector(self, parent_layout):
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        label = QLabel(self.tr("Training backend:"))
        label.setStyleSheet("font-weight: bold;")
        row.addWidget(label)
        self.training_backend_combo = CustomComboBox()
        self.training_backend_combo.addItem(
            self.tr("Ultralytics wizard"), "ultralytics"
        )
        self.training_backend_combo.addItem(
            self.tr("Custom command"), "custom"
        )
        self.training_backend_combo.currentIndexChanged.connect(
            self._on_training_backend_changed
        )
        row.addWidget(self.training_backend_combo, 1)
        parent_layout.addWidget(row_widget)

    def init_custom_training_panel(self, parent_layout):
        self.custom_training_panel = QGroupBox(self.tr("Custom command"))
        custom_layout = QVBoxLayout(self.custom_training_panel)

        hint = QLabel(
            self.tr(
                "Basic Settings supply effective paths (after upload/mapping). "
                "Use placeholders {model}, {data}, {project}, {name}, {run_dir}, "
                "{device} in your command. Values in the command for those fields "
                "are ignored. Multi-line commands with a trailing backslash (\\) "
                "are merged into one line before execution (required on Windows)."
            )
        )
        hint.setWordWrap(True)
        custom_layout.addWidget(hint)

        custom_layout.addWidget(QLabel(self.tr("Training command:")))
        self.custom_command_edit = QTextEdit()
        self.custom_command_edit.setPlaceholderText(
            self.tr(
                'e.g. python train.py --weights "{model}" --data "{data}"'
            )
        )
        self.custom_command_edit.setMinimumHeight(100)
        custom_layout.addWidget(self.custom_command_edit)

        cwd_row = QHBoxLayout()
        cwd_row.addWidget(QLabel(self.tr("Execution directory:")))
        self.custom_exec_cwd_edit = CustomLineEdit()
        self.custom_exec_cwd_edit.setPlaceholderText(
            self.tr("Optional — leave empty for default (like Ultralytics)")
        )
        self._refresh_exec_cwd_hint()
        cwd_row.addWidget(self.custom_exec_cwd_edit, 1)
        self.custom_exec_cwd_browse_btn = SecondaryButton(self.tr("Browse"))
        self.custom_exec_cwd_browse_btn.clicked.connect(
            self.browse_custom_exec_cwd
        )
        cwd_row.addWidget(self.custom_exec_cwd_browse_btn)
        custom_layout.addLayout(cwd_row)

        self.custom_training_panel.setVisible(False)
        parent_layout.addWidget(self.custom_training_panel)

    def _on_training_backend_changed(self, _index: int = 0) -> None:
        data = self.training_backend_combo.currentData()
        self.training_backend = data if data else "ultralytics"
        custom = self._is_custom_training_backend()
        self.custom_training_panel.setVisible(custom)
        if hasattr(self, "train_settings_group"):
            self.train_settings_group.setVisible(not custom)

    def _refresh_exec_cwd_hint(self) -> None:
        if not hasattr(self, "custom_exec_cwd_edit"):
            return
        if self.train_mode == "remote":
            self.custom_exec_cwd_edit.setPlaceholderText(
                self.tr(
                    "Optional remote directory. Browse the server, or leave empty."
                )
            )
        else:
            self.custom_exec_cwd_edit.setPlaceholderText(
                self.tr("Optional — leave empty for default (like Ultralytics)")
            )

    def _remote_exec_browse_start(self) -> str:
        text = (self.custom_exec_cwd_edit.text() or "").strip().replace("\\", "/")
        if text.startswith("/") and not is_windows_local_path(text):
            return text
        home = (getattr(self.ssh_client, "home", "") or "").rstrip("/")
        base = self._remote_base_dir().replace("\\", "/")
        if base == "~":
            return home or "/"
        if base.startswith("~/"):
            return f"{home}/{base[2:]}".rstrip("/") or home or "/"
        if base.startswith("/"):
            return base.rstrip("/") or "/"
        return home or "/"

    def browse_custom_exec_cwd(self) -> None:
        if self.train_mode == "remote":
            if not self.ssh_client.connected:
                QMessageBox.warning(
                    self,
                    self.tr("Remote"),
                    self.tr("Connect to the remote server first."),
                )
                return
            dlg = RemoteDirBrowserDialog(
                self,
                ssh_client=self.ssh_client,
                start_path=self._remote_exec_browse_start(),
            )
            if dlg.exec() == QDialog.DialogCode.Accepted:
                chosen = (dlg.selected_path() or "").strip()
                if chosen and not is_windows_local_path(chosen):
                    self.custom_exec_cwd_edit.setText(chosen)
            return
        path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Execution directory"),
            self.custom_exec_cwd_edit.text().strip() or "",
        )
        if path:
            self.custom_exec_cwd_edit.setText(path)

    def init_basic_settings(self, parent_layout):
        group = QGroupBox(self.tr("Basic Settings"))
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

        self.config_widgets["project"] = CustomLineEdit()
        selected_task_type = (
            self.selected_task_type.lower()
            if self.selected_task_type
            else "detect"
        )
        text_project = os.path.join(
            get_default_project_dir(), selected_task_type
        )
        self.config_widgets["project"].setText(text_project)
        layout.addRow("Project:", self.config_widgets["project"])

        self.config_widgets["name"] = CustomLineEdit()
        self.config_widgets["name"].setText("exp")
        layout.addRow("Name:", self.config_widgets["name"])

        model_layout = QHBoxLayout()
        self.config_widgets["model"] = CustomLineEdit()
        model_browse_btn = SecondaryButton("Browse")
        model_browse_btn.clicked.connect(self.browse_model_file)
        model_layout.addWidget(self.config_widgets["model"])
        model_layout.addWidget(model_browse_btn)
        layout.addRow("Model:", model_layout)

        data_layout = QHBoxLayout()
        self.config_widgets["data"] = CustomLineEdit()
        data_browse_btn = SecondaryButton("Browse")
        data_browse_btn.clicked.connect(self.browse_data_file)
        data_layout.addWidget(self.config_widgets["data"])
        data_layout.addWidget(data_browse_btn)
        layout.addRow("Data:", data_layout)

        pose_config_layout = QHBoxLayout()
        self.config_widgets["pose_config"] = CustomLineEdit()
        pose_config_browse_btn = SecondaryButton("Browse")
        pose_config_browse_btn.clicked.connect(self.browse_pose_config_file)
        pose_config_layout.addWidget(self.config_widgets["pose_config"])
        pose_config_layout.addWidget(pose_config_browse_btn)

        self.pose_config_label = QLabel("Pose Config:")
        layout.addRow(self.pose_config_label, pose_config_layout)
        self.pose_config_layout = pose_config_layout

        self.pose_config_label.setVisible(False)
        self.config_widgets["pose_config"].setVisible(False)
        pose_config_browse_btn.setVisible(False)

        device_layout = QHBoxLayout()
        self.config_widgets["device"] = CustomComboBox()
        device_items = list(DEVICE_OPTIONS)
        # Remote servers often have CUDA even when local torch is CPU-only
        if "cuda" not in device_items:
            device_items = ["cuda"] + device_items
        self.config_widgets["device"].addItems(device_items)
        self.device_checkboxes = QWidget()
        self.device_checkboxes.setVisible(False)
        self.config_widgets["device"].currentTextChanged.connect(
            self.on_device_changed
        )
        device_layout.addWidget(self.config_widgets["device"])
        device_layout.addWidget(self.device_checkboxes)
        layout.addRow("Device:", device_layout)
        self.on_device_changed(self.config_widgets["device"].currentText())

        dataset_layout = QHBoxLayout()
        self.config_widgets["dataset_ratio"] = CustomSlider(
            Qt.Orientation.Horizontal
        )
        self.config_widgets["dataset_ratio"].setRange(5, 95)
        self.config_widgets["dataset_ratio"].setValue(80)
        self.dataset_ratio_label = QLabel("0.8")
        self.config_widgets["dataset_ratio"].valueChanged.connect(
            lambda v: self.dataset_ratio_label.setText(str(v / 100.0))
        )
        dataset_layout.addWidget(self.config_widgets["dataset_ratio"])
        dataset_layout.addWidget(self.dataset_ratio_label)
        layout.addRow("Dataset Ratio:", dataset_layout)

        parent_layout.addWidget(group)

    def toggle_advanced_settings(self):
        """Toggle the visibility of advanced settings"""
        if self.advanced_content_widget.isVisible():
            self.advanced_content_widget.setVisible(False)
            self.advanced_toggle_btn.setIcon(
                QIcon(new_icon("caret-down", "svg"))
            )
        else:
            self.advanced_content_widget.setVisible(True)
            self.advanced_toggle_btn.setIcon(
                QIcon(new_icon("caret-up", "svg"))
            )

    def init_train_settings(self, parent_layout):
        group = QGroupBox(self.tr("Train Settings"))
        self.train_settings_group = group
        layout = QVBoxLayout(group)

        # Basic settings
        basic_group = QGroupBox(self.tr("Basic"))
        basic_layout = QHBoxLayout(basic_group)
        basic_layout.addWidget(QLabel("Epochs:"))
        self.config_widgets["epochs"] = CustomSpinBox()
        self.config_widgets["epochs"].setRange(1, 10000)
        self.config_widgets["epochs"].setValue(
            DEFAULT_TRAINING_CONFIG["epochs"]
        )
        basic_layout.addWidget(self.config_widgets["epochs"])

        basic_layout.addWidget(QLabel("Batch:"))
        self.config_widgets["batch"] = CustomSpinBox()
        self.config_widgets["batch"].setRange(-1, 8192)
        self.config_widgets["batch"].setValue(DEFAULT_TRAINING_CONFIG["batch"])
        basic_layout.addWidget(self.config_widgets["batch"])

        basic_layout.addWidget(QLabel("Image Size:"))
        self.config_widgets["imgsz"] = CustomSpinBox()
        self.config_widgets["imgsz"].setRange(32, 8192)
        self.config_widgets["imgsz"].setValue(DEFAULT_TRAINING_CONFIG["imgsz"])
        basic_layout.addWidget(self.config_widgets["imgsz"])

        basic_layout.addWidget(QLabel("Workers:"))
        self.config_widgets["workers"] = CustomSpinBox()
        self.config_widgets["workers"].setRange(0, NUM_WORKERS)
        self.config_widgets["workers"].setValue(
            DEFAULT_TRAINING_CONFIG["workers"]
        )
        basic_layout.addWidget(self.config_widgets["workers"])

        basic_layout.addWidget(QLabel("Classes:"))
        self.config_widgets["classes"] = CustomLineEdit()
        self.config_widgets["classes"].setText(
            DEFAULT_TRAINING_CONFIG["classes"]
        )
        self.config_widgets["classes"].setPlaceholderText(
            self.tr("Class indices (e.g., 0,1,2) or leave empty for all")
        )
        basic_layout.addWidget(self.config_widgets["classes"])

        self.config_widgets["single_cls"] = CustomCheckBox("Single Class")
        self.config_widgets["single_cls"].setChecked(
            DEFAULT_TRAINING_CONFIG["single_cls"]
        )
        basic_layout.addWidget(self.config_widgets["single_cls"])

        basic_layout.addStretch()
        layout.addWidget(basic_group)

        # Advanced settings
        advanced_container = QWidget()
        advanced_container_layout = QVBoxLayout(advanced_container)
        advanced_container_layout.setContentsMargins(0, 0, 0, 0)

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)

        advanced_label = QLabel(self.tr("Advanced Settings"))
        advanced_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(advanced_label)

        # Collapse/Expand button
        self.advanced_toggle_btn = QPushButton()
        self.advanced_toggle_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.advanced_toggle_btn.setStyleSheet(get_advanced_toggle_btn_style())
        self.advanced_toggle_btn.setIcon(QIcon(new_icon("caret-down", "svg")))
        self.advanced_toggle_btn.clicked.connect(self.toggle_advanced_settings)
        header_layout.addWidget(self.advanced_toggle_btn)
        header_layout.addStretch()
        advanced_container_layout.addWidget(header_widget)

        self.advanced_content_widget = QWidget()
        self.advanced_content_widget.setVisible(False)
        advanced_layout = QVBoxLayout(self.advanced_content_widget)

        # 1. Training Strategy
        strategy_group = QGroupBox("Training Strategy")
        strat_layout = QHBoxLayout(strategy_group)
        strat_layout.addWidget(QLabel("Time (h):"))
        self.config_widgets["time"] = CustomDoubleSpinBox()
        self.config_widgets["time"].setValue(DEFAULT_TRAINING_CONFIG["time"])
        self.config_widgets["time"].setSpecialValueText("None")
        strat_layout.addWidget(self.config_widgets["time"])

        strat_layout.addWidget(QLabel("Patience:"))
        self.config_widgets["patience"] = CustomSpinBox()
        self.config_widgets["patience"].setRange(1, 10000)
        self.config_widgets["patience"].setValue(
            DEFAULT_TRAINING_CONFIG["patience"]
        )
        strat_layout.addWidget(self.config_widgets["patience"])

        strat_layout.addWidget(QLabel("Close Mosaic:"))
        self.config_widgets["close_mosaic"] = CustomSpinBox()
        self.config_widgets["close_mosaic"].setRange(0, 1000)
        self.config_widgets["close_mosaic"].setValue(
            DEFAULT_TRAINING_CONFIG["close_mosaic"]
        )
        strat_layout.addWidget(self.config_widgets["close_mosaic"])

        strat_layout.addWidget(QLabel("Optimizer:"))
        self.config_widgets["optimizer"] = CustomComboBox()
        self.config_widgets["optimizer"].addItems(OPTIMIZER_OPTIONS)
        strat_layout.addWidget(self.config_widgets["optimizer"])

        self.config_widgets["cos_lr"] = CustomCheckBox("Cosine LR")
        self.config_widgets["cos_lr"].setChecked(
            DEFAULT_TRAINING_CONFIG["cos_lr"]
        )
        strat_layout.addWidget(self.config_widgets["cos_lr"])
        self.config_widgets["amp"] = CustomCheckBox("AMP")
        self.config_widgets["amp"].setChecked(DEFAULT_TRAINING_CONFIG["amp"])
        strat_layout.addWidget(self.config_widgets["amp"])
        strat_layout.addWidget(QLabel(self.tr("Multi Scale:")))
        self.config_widgets["multi_scale"] = CustomDoubleSpinBox()
        self.config_widgets["multi_scale"].setDecimals(3)
        self.config_widgets["multi_scale"].setRange(0.0, 1.0)
        ms = DEFAULT_TRAINING_CONFIG["multi_scale"]
        self.config_widgets["multi_scale"].setValue(
            float(ms) if not isinstance(ms, bool) else (0.5 if ms else 0.0)
        )
        strat_layout.addWidget(self.config_widgets["multi_scale"])
        strat_layout.addStretch()
        advanced_layout.addWidget(strategy_group)

        strategy_extra = QGroupBox(self.tr("Model init and runtime"))
        strat2 = QHBoxLayout(strategy_extra)
        strat2.addWidget(QLabel(self.tr("Freeze:")))
        self.config_widgets["freeze"] = CustomLineEdit()
        self.config_widgets["freeze"].setPlaceholderText(
            self.tr("e.g. 10 or 0,1,2 — empty for none")
        )
        fv = DEFAULT_TRAINING_CONFIG.get("freeze", "")
        self.config_widgets["freeze"].setText(
            "" if fv is None else str(fv).strip()
        )
        strat2.addWidget(self.config_widgets["freeze"], 1)

        strat2.addWidget(QLabel(self.tr("Seed:")))
        self.config_widgets["seed"] = CustomSpinBox()
        self.config_widgets["seed"].setRange(0, 2_147_483_647)
        self.config_widgets["seed"].setValue(
            int(DEFAULT_TRAINING_CONFIG.get("seed", 0) or 0)
        )
        strat2.addWidget(self.config_widgets["seed"])

        strat2.addWidget(QLabel(self.tr("NBS:")))
        self.config_widgets["nbs"] = CustomSpinBox()
        self.config_widgets["nbs"].setRange(1, 8192)
        self.config_widgets["nbs"].setValue(
            int(DEFAULT_TRAINING_CONFIG.get("nbs", 64) or 64)
        )
        strat2.addWidget(self.config_widgets["nbs"])

        self.config_widgets["pretrained"] = CustomCheckBox(
            self.tr("Pretrained")
        )
        self.config_widgets["pretrained"].setChecked(
            bool(DEFAULT_TRAINING_CONFIG.get("pretrained", True))
        )
        strat2.addWidget(self.config_widgets["pretrained"])
        self.config_widgets["deterministic"] = CustomCheckBox(
            self.tr("Deterministic")
        )
        self.config_widgets["deterministic"].setChecked(
            bool(DEFAULT_TRAINING_CONFIG.get("deterministic", True))
        )
        strat2.addWidget(self.config_widgets["deterministic"])
        self.config_widgets["profile"] = CustomCheckBox(self.tr("Profile"))
        self.config_widgets["profile"].setChecked(
            bool(DEFAULT_TRAINING_CONFIG.get("profile", False))
        )
        strat2.addWidget(self.config_widgets["profile"])
        self.config_widgets["compile"] = CustomCheckBox(self.tr("Compile"))
        self.config_widgets["compile"].setChecked(
            bool(DEFAULT_TRAINING_CONFIG.get("compile", False))
        )
        strat2.addWidget(self.config_widgets["compile"])
        strat2.addStretch()
        advanced_layout.addWidget(strategy_extra)

        # 2. Learning Rate
        lr_group = QGroupBox("Learning Rate")
        lr_layout = QHBoxLayout(lr_group)
        lr_layout.addWidget(QLabel("LR0:"))
        self.config_widgets["lr0"] = CustomDoubleSpinBox()
        self.config_widgets["lr0"].setDecimals(6)
        self.config_widgets["lr0"].setValue(DEFAULT_TRAINING_CONFIG["lr0"])
        lr_layout.addWidget(self.config_widgets["lr0"])

        lr_layout.addWidget(QLabel("LRF:"))
        self.config_widgets["lrf"] = CustomDoubleSpinBox()
        self.config_widgets["lrf"].setDecimals(6)
        self.config_widgets["lrf"].setValue(DEFAULT_TRAINING_CONFIG["lrf"])
        lr_layout.addWidget(self.config_widgets["lrf"])

        lr_layout.addWidget(QLabel("Momentum:"))
        self.config_widgets["momentum"] = CustomDoubleSpinBox()
        self.config_widgets["momentum"].setDecimals(3)
        self.config_widgets["momentum"].setValue(
            DEFAULT_TRAINING_CONFIG["momentum"]
        )
        lr_layout.addWidget(self.config_widgets["momentum"])

        lr_layout.addWidget(QLabel("Weight Decay:"))
        self.config_widgets["weight_decay"] = CustomDoubleSpinBox()
        self.config_widgets["weight_decay"].setDecimals(6)
        self.config_widgets["weight_decay"].setValue(
            DEFAULT_TRAINING_CONFIG["weight_decay"]
        )
        lr_layout.addWidget(self.config_widgets["weight_decay"])
        lr_layout.addStretch()
        advanced_layout.addWidget(lr_group)

        # 3. Warmup Parameters
        warmup_group = QGroupBox("Warmup Parameters")
        warmup_layout = QHBoxLayout(warmup_group)
        warmup_layout.addWidget(QLabel("Warmup Epochs:"))
        self.config_widgets["warmup_epochs"] = CustomDoubleSpinBox()
        self.config_widgets["warmup_epochs"].setDecimals(1)
        self.config_widgets["warmup_epochs"].setValue(
            DEFAULT_TRAINING_CONFIG["warmup_epochs"]
        )
        warmup_layout.addWidget(self.config_widgets["warmup_epochs"])

        warmup_layout.addWidget(QLabel("Warmup Momentum:"))
        self.config_widgets["warmup_momentum"] = CustomDoubleSpinBox()
        self.config_widgets["warmup_momentum"].setDecimals(3)
        self.config_widgets["warmup_momentum"].setValue(
            DEFAULT_TRAINING_CONFIG["warmup_momentum"]
        )
        warmup_layout.addWidget(self.config_widgets["warmup_momentum"])

        warmup_layout.addWidget(QLabel("Warmup Bias LR:"))
        self.config_widgets["warmup_bias_lr"] = CustomDoubleSpinBox()
        self.config_widgets["warmup_bias_lr"].setDecimals(3)
        self.config_widgets["warmup_bias_lr"].setValue(
            DEFAULT_TRAINING_CONFIG["warmup_bias_lr"]
        )
        warmup_layout.addWidget(self.config_widgets["warmup_bias_lr"])
        warmup_layout.addStretch()
        advanced_layout.addWidget(warmup_group)

        # 4. Augmentation Settings
        augment_group = QGroupBox("Augmentation Settings")
        augment_layout = QVBoxLayout(augment_group)
        augment_params = [
            (
                "hsv_h",
                "HSV Hue:",
                DEFAULT_TRAINING_CONFIG["hsv_h"],
                0.0,
                1.0,
                3,
            ),
            (
                "hsv_s",
                "HSV Saturation:",
                DEFAULT_TRAINING_CONFIG["hsv_s"],
                0.0,
                1.0,
                3,
            ),
            (
                "hsv_v",
                "HSV Value:",
                DEFAULT_TRAINING_CONFIG["hsv_v"],
                0.0,
                1.0,
                3,
            ),
            (
                "degrees",
                "Rotation Degrees:",
                DEFAULT_TRAINING_CONFIG["degrees"],
                -180.0,
                180.0,
                1,
            ),
            (
                "translate",
                "Translate:",
                DEFAULT_TRAINING_CONFIG["translate"],
                0.0,
                1.0,
                3,
            ),
            ("scale", "Scale:", DEFAULT_TRAINING_CONFIG["scale"], 0.0, 2.0, 3),
            (
                "shear",
                "Shear:",
                DEFAULT_TRAINING_CONFIG["shear"],
                -45.0,
                45.0,
                1,
            ),
            (
                "perspective",
                "Perspective:",
                DEFAULT_TRAINING_CONFIG["perspective"],
                0.0,
                0.001,
                6,
            ),
            (
                "flipud",
                self.tr("Flip UD:"),
                DEFAULT_TRAINING_CONFIG["flipud"],
                0.0,
                1.0,
                3,
            ),
            (
                "fliplr",
                self.tr("Flip LR:"),
                DEFAULT_TRAINING_CONFIG["fliplr"],
                0.0,
                1.0,
                3,
            ),
            (
                "mosaic",
                "Mosaic:",
                DEFAULT_TRAINING_CONFIG["mosaic"],
                0.0,
                1.0,
                3,
            ),
            (
                "mixup",
                "MixUp:",
                DEFAULT_TRAINING_CONFIG["mixup"],
                0.0,
                1.0,
                3,
            ),
            (
                "cutmix",
                "CutMix:",
                DEFAULT_TRAINING_CONFIG["cutmix"],
                0.0,
                1.0,
                3,
            ),
            (
                "copy_paste",
                self.tr("Copy-Paste:"),
                DEFAULT_TRAINING_CONFIG["copy_paste"],
                0.0,
                1.0,
                3,
            ),
            (
                "bgr",
                "BGR:",
                DEFAULT_TRAINING_CONFIG["bgr"],
                0.0,
                1.0,
                3,
            ),
        ]

        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(10)
        grid_layout.setVerticalSpacing(5)
        for i, (
            param,
            label,
            default,
            min_val,
            max_val,
            decimals,
        ) in enumerate(augment_params):
            row = i // 4
            col = (i % 4) * 2

            label_widget = QLabel(label)
            label_widget.setMinimumWidth(80)
            grid_layout.addWidget(label_widget, row, col)

            widget = CustomDoubleSpinBox()
            widget.setRange(min_val, max_val)
            widget.setDecimals(decimals)
            widget.setValue(default)
            widget.setMinimumWidth(80)
            self.config_widgets[param] = widget
            grid_layout.addWidget(widget, row, col + 1)

        for col in range(8, 10):
            grid_layout.setColumnStretch(col, 1)
        augment_layout.addLayout(grid_layout)
        advanced_layout.addWidget(augment_group)

        # 5. Regularization
        reg_group = QGroupBox("Regularization")
        reg_layout = QHBoxLayout(reg_group)
        reg_layout.addWidget(QLabel("Dropout:"))
        self.config_widgets["dropout"] = CustomDoubleSpinBox()
        self.config_widgets["dropout"].setDecimals(3)
        self.config_widgets["dropout"].setValue(
            DEFAULT_TRAINING_CONFIG["dropout"]
        )
        reg_layout.addWidget(self.config_widgets["dropout"])

        reg_layout.addWidget(QLabel("Fraction:"))
        self.config_widgets["fraction"] = CustomDoubleSpinBox()
        self.config_widgets["fraction"].setDecimals(3)
        self.config_widgets["fraction"].setValue(
            DEFAULT_TRAINING_CONFIG["fraction"]
        )
        reg_layout.addWidget(self.config_widgets["fraction"])

        self.config_widgets["rect"] = CustomCheckBox("Rectangular")
        self.config_widgets["rect"].setChecked(DEFAULT_TRAINING_CONFIG["rect"])
        reg_layout.addWidget(self.config_widgets["rect"])
        reg_layout.addStretch()
        advanced_layout.addWidget(reg_group)

        # 6. Loss Weights
        loss_group = QGroupBox("Loss Weights")
        loss_layout = QHBoxLayout(loss_group)
        loss_layout.addWidget(QLabel("Box:"))
        self.config_widgets["box"] = CustomDoubleSpinBox()
        self.config_widgets["box"].setDecimals(2)
        self.config_widgets["box"].setValue(DEFAULT_TRAINING_CONFIG["box"])
        loss_layout.addWidget(self.config_widgets["box"])

        loss_layout.addWidget(QLabel("Cls:"))
        self.config_widgets["cls"] = CustomDoubleSpinBox()
        self.config_widgets["cls"].setDecimals(2)
        self.config_widgets["cls"].setValue(DEFAULT_TRAINING_CONFIG["cls"])
        loss_layout.addWidget(self.config_widgets["cls"])

        loss_layout.addWidget(QLabel("DFL:"))
        self.config_widgets["dfl"] = CustomDoubleSpinBox()
        self.config_widgets["dfl"].setDecimals(2)
        self.config_widgets["dfl"].setValue(DEFAULT_TRAINING_CONFIG["dfl"])
        loss_layout.addWidget(self.config_widgets["dfl"])

        loss_layout.addWidget(QLabel("Pose:"))
        self.config_widgets["pose"] = CustomDoubleSpinBox()
        self.config_widgets["pose"].setDecimals(2)
        self.config_widgets["pose"].setValue(DEFAULT_TRAINING_CONFIG["pose"])
        loss_layout.addWidget(self.config_widgets["pose"])

        loss_layout.addWidget(QLabel("Kobj:"))
        self.config_widgets["kobj"] = CustomDoubleSpinBox()
        self.config_widgets["kobj"].setDecimals(2)
        self.config_widgets["kobj"].setValue(DEFAULT_TRAINING_CONFIG["kobj"])
        loss_layout.addWidget(self.config_widgets["kobj"])
        loss_layout.addStretch()
        advanced_layout.addWidget(loss_group)

        # 7. Checkpoint and Validation
        ckpt_group = QGroupBox("Checkpoint and Validation")
        ckpt_layout = QHBoxLayout(ckpt_group)
        ckpt_layout.addWidget(QLabel("Save Period:"))
        self.config_widgets["save_period"] = CustomSpinBox()
        self.config_widgets["save_period"].setRange(-1, 1000)
        self.config_widgets["save_period"].setValue(
            DEFAULT_TRAINING_CONFIG["save_period"]
        )
        self.config_widgets["save_period"].setSpecialValueText("Disabled")
        ckpt_layout.addWidget(self.config_widgets["save_period"])

        self.config_widgets["val"] = CustomCheckBox("Validation")
        self.config_widgets["val"].setChecked(DEFAULT_TRAINING_CONFIG["val"])
        ckpt_layout.addWidget(self.config_widgets["val"])
        self.config_widgets["plots"] = CustomCheckBox("Plots")
        self.config_widgets["plots"].setChecked(
            DEFAULT_TRAINING_CONFIG["plots"]
        )
        ckpt_layout.addWidget(self.config_widgets["plots"])
        self.config_widgets["save"] = CustomCheckBox("Save")
        self.config_widgets["save"].setChecked(DEFAULT_TRAINING_CONFIG["save"])
        ckpt_layout.addWidget(self.config_widgets["save"])
        self.config_widgets["resume"] = CustomCheckBox("Resume")
        self.config_widgets["resume"].setChecked(
            DEFAULT_TRAINING_CONFIG["resume"]
        )
        ckpt_layout.addWidget(self.config_widgets["resume"])
        self.config_widgets["cache"] = CustomCheckBox("Cache")
        self.config_widgets["cache"].setChecked(
            DEFAULT_TRAINING_CONFIG["cache"]
        )
        ckpt_layout.addWidget(self.config_widgets["cache"])
        self.config_widgets["skip_empty_files"] = CustomCheckBox(
            "Skip Empty Files"
        )
        self.config_widgets["skip_empty_files"].setChecked(False)
        ckpt_layout.addWidget(self.config_widgets["skip_empty_files"])
        ckpt_layout.addStretch()
        advanced_layout.addWidget(ckpt_group)

        advanced_container_layout.addWidget(self.advanced_content_widget)
        layout.addWidget(advanced_container)
        parent_layout.addWidget(group)

    def load_config_to_ui(self, config):
        def set_widget_value(key, value):
            if key not in self.config_widgets:
                return

            widget = self.config_widgets[key]
            widget_type = type(widget).__name__

            try:
                if widget_type == "CustomLineEdit":
                    if key == "classes":
                        widget.setText(format_classes_display(value))
                    else:
                        widget.setText(str(value) if value is not None else "")
                elif widget_type in ["CustomSpinBox", "CustomDoubleSpinBox"]:
                    widget.setValue(value)
                elif widget_type == "CustomComboBox":
                    if isinstance(value, str):
                        index = widget.findText(value)
                        if index >= 0:
                            widget.setCurrentIndex(index)
                    else:
                        widget.setCurrentIndex(value)
                elif widget_type == "CustomCheckBox":
                    widget.setChecked(bool(value))
                elif widget_type == "CustomSlider":
                    widget.setValue(value)
            except Exception as e:
                logger.warning(f"Failed to set value for widget {key}: {e}")

        sections_to_process = [
            "basic",
            "train",
            "augment",
            "strategy",
            "learning_rate",
            "warmup",
            "regularization",
            "loss_weights",
            "checkpoint",
        ]
        for section in sections_to_process:
            if section in config:
                for key, value in config[section].items():
                    if key == "dataset_ratio":
                        if 0 <= value <= 1:
                            self.config_widgets[key].setValue(int(value * 100))
                            self.dataset_ratio_label.setText(str(value))
                        else:
                            self.config_widgets[key].setValue(int(value))
                            self.dataset_ratio_label.setText(
                                str(value / 100.0)
                            )
                    elif key == "device":
                        index = self.config_widgets[key].findText(str(value))
                        if index >= 0:
                            self.config_widgets[key].setCurrentIndex(index)
                            self.on_device_changed(str(value))
                    elif key == "optimizer":
                        index = self.config_widgets[key].findText(str(value))
                        if index >= 0:
                            self.config_widgets[key].setCurrentIndex(index)
                    elif key == "pose_config":
                        if value:
                            self.config_widgets[key].setText(value)
                    elif key == "skip_empty_files":
                        set_widget_value(key, value)
                    elif key == "freeze":
                        if value is None:
                            self.config_widgets[key].setText("")
                        elif isinstance(value, list):
                            self.config_widgets[key].setText(
                                ",".join(str(x) for x in value)
                            )
                        else:
                            self.config_widgets[key].setText(str(value))
                    elif key == "multi_scale" and isinstance(value, bool):
                        set_widget_value(key, 0.5 if value else 0.0)
                    else:
                        set_widget_value(key, value)

        for key, value in config.items():
            if key not in sections_to_process and key in self.config_widgets:
                set_widget_value(key, value)

        backend = config.get("training_backend") or "ultralytics"
        if hasattr(self, "training_backend_combo"):
            idx = self.training_backend_combo.findData(backend)
            if idx >= 0:
                self.training_backend_combo.setCurrentIndex(idx)
            self._on_training_backend_changed()
        else:
            self.training_backend = backend
        custom = config.get("custom") or {}
        if hasattr(self, "custom_command_edit"):
            self.custom_command_edit.setPlainText(custom.get("command") or "")
        if hasattr(self, "custom_exec_cwd_edit"):
            self.custom_exec_cwd_edit.setText(custom.get("exec_cwd") or "")

    def import_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Import Config"),
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if file_path:
            config = load_config_from_file(file_path)
            if config:
                self.load_config_to_ui(config)
                QMessageBox.information(
                    self,
                    self.tr("Success"),
                    self.tr("Config imported successfully"),
                )
            else:
                QMessageBox.warning(
                    self, self.tr("Error"), self.tr("Failed to import config")
                )

    def get_current_config(self):
        def get_widget_value(key):
            if key not in self.config_widgets:
                return None
            widget = self.config_widgets[key]
            widget_type = type(widget).__name__
            try:
                if widget_type == "CustomLineEdit":
                    return widget.text()
                elif widget_type in ["CustomSpinBox", "CustomDoubleSpinBox"]:
                    return widget.value()
                elif widget_type == "CustomComboBox":
                    return widget.currentText()
                elif widget_type == "CustomCheckBox":
                    return widget.isChecked()
                elif widget_type == "CustomSlider":
                    return widget.value()
            except Exception:
                return None
            return None

        config = {
            "basic": {
                "project": get_widget_value("project"),
                "name": get_widget_value("name"),
                "model": get_widget_value("model").strip('"'),
                "data": get_widget_value("data").strip('"'),
                "device": get_widget_value("device"),
                "dataset_ratio": (
                    get_widget_value("dataset_ratio") / 100.0
                    if get_widget_value("dataset_ratio") is not None
                    else 0.8
                ),
                "pose_config": get_widget_value("pose_config"),
            },
            "train": {
                "epochs": get_widget_value("epochs"),
                "batch": get_widget_value("batch"),
                "imgsz": get_widget_value("imgsz"),
                "workers": get_widget_value("workers"),
                "single_cls": get_widget_value("single_cls"),
                "classes": parse_string_to_digit_list(
                    get_widget_value("classes")
                ),
            },
            "strategy": {
                "time": get_widget_value("time"),
                "patience": get_widget_value("patience"),
                "close_mosaic": get_widget_value("close_mosaic"),
                "optimizer": get_widget_value("optimizer"),
                "cos_lr": get_widget_value("cos_lr"),
                "amp": get_widget_value("amp"),
                "multi_scale": get_widget_value("multi_scale"),
                "freeze": (get_widget_value("freeze") or "").strip(),
                "pretrained": get_widget_value("pretrained"),
                "seed": get_widget_value("seed"),
                "deterministic": get_widget_value("deterministic"),
                "nbs": get_widget_value("nbs"),
                "profile": get_widget_value("profile"),
                "compile": get_widget_value("compile"),
            },
            "learning_rate": {
                "lr0": get_widget_value("lr0"),
                "lrf": get_widget_value("lrf"),
                "momentum": get_widget_value("momentum"),
                "weight_decay": get_widget_value("weight_decay"),
            },
            "warmup": {
                "warmup_epochs": get_widget_value("warmup_epochs"),
                "warmup_momentum": get_widget_value("warmup_momentum"),
                "warmup_bias_lr": get_widget_value("warmup_bias_lr"),
            },
            "augment": {
                "hsv_h": get_widget_value("hsv_h"),
                "hsv_s": get_widget_value("hsv_s"),
                "hsv_v": get_widget_value("hsv_v"),
                "degrees": get_widget_value("degrees"),
                "translate": get_widget_value("translate"),
                "scale": get_widget_value("scale"),
                "shear": get_widget_value("shear"),
                "perspective": get_widget_value("perspective"),
                "flipud": get_widget_value("flipud"),
                "fliplr": get_widget_value("fliplr"),
                "mosaic": get_widget_value("mosaic"),
                "mixup": get_widget_value("mixup"),
                "cutmix": get_widget_value("cutmix"),
                "copy_paste": get_widget_value("copy_paste"),
                "bgr": get_widget_value("bgr"),
            },
            "regularization": {
                "dropout": get_widget_value("dropout"),
                "fraction": get_widget_value("fraction"),
                "rect": get_widget_value("rect"),
            },
            "loss_weights": {
                "box": get_widget_value("box"),
                "cls": get_widget_value("cls"),
                "dfl": get_widget_value("dfl"),
                "pose": get_widget_value("pose"),
                "kobj": get_widget_value("kobj"),
            },
            "checkpoint": {
                "save_period": get_widget_value("save_period"),
                "val": get_widget_value("val"),
                "plots": get_widget_value("plots"),
                "save": get_widget_value("save"),
                "resume": get_widget_value("resume"),
                "cache": get_widget_value("cache"),
                "skip_empty_files": get_widget_value("skip_empty_files"),
            },
            "training_backend": (
                self.training_backend_combo.currentData()
                if hasattr(self, "training_backend_combo")
                else self.training_backend
            ),
            "custom": {
                "command": (
                    self.custom_command_edit.toPlainText()
                    if hasattr(self, "custom_command_edit")
                    else ""
                ),
                "exec_cwd": (
                    self.custom_exec_cwd_edit.text()
                    if hasattr(self, "custom_exec_cwd_edit")
                    else ""
                ),
            },
        }

        return config

    def save_current_config(self):
        try:
            save_config(self.get_current_config())
            template = self.tr("Configuration saved successfully to %s")
            msg_test = template % get_settings_config_path()
            QMessageBox.information(self, self.tr("Success"), msg_test)
        except Exception as e:
            QMessageBox.warning(
                self, self.tr("Error"), f"Failed to save config: {str(e)}"
            )

    def start_training(self):
        if self.training_status == "training":
            QMessageBox.warning(
                self,
                self.tr("Training in Progress"),
                self.tr(
                    "Training is currently in progress. Please stop the training first if you need to reconfigure."
                ),
            )
            return

        if not self._ensure_remote_train_allowed():
            return
        if (
            self.train_mode == "interpreter"
            and not self._ensure_selected_interpreter_ready()
        ):
            return

        config = self.get_current_config()
        custom_backend = (config.get("training_backend") or "ultralytics") == "custom"
        if custom_backend:
            cmd = (config.get("custom") or {}).get("command", "").strip()
            if not cmd:
                QMessageBox.warning(
                    self,
                    self.tr("Custom training"),
                    self.tr("Please enter a training command."),
                )
                return
        is_valid, error_message = validate_basic_config(
            config,
            remote=(self.train_mode == "remote"),
            custom=custom_backend,
        )
        if is_valid == "directory_exists":
            project_dir = error_message
            potential_model_path = os.path.join(
                project_dir, "weights", "best.pt"
            )

            if os.path.exists(potential_model_path):
                reply = QMessageBox.question(
                    self,
                    self.tr("Existing Model Detected"),
                    self.tr(
                        "A trained model already exists at this location.\n\n"
                        "Do you want to:\n"
                        "Yes - Export the existing model directly\n"
                        "No - Continue to retrain (will overwrite)"
                    ),
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )

                if reply == QMessageBox.StandardButton.Yes:
                    self.current_project_path = project_dir
                    self.training_status = "completed"
                    save_config(config)
                    self.go_to_specific_tab(2)
                    self.update_training_status_display()
                    self.start_training_button.setVisible(False)
                    self._set_post_train_buttons(visible=True)
                    self.previous_button.setVisible(True)
                    self.update_training_images()
                    self.append_training_log(
                        f"Loaded existing model from: {potential_model_path}"
                    )
                    return

            reply = QMessageBox.question(
                self,
                self.tr("Directory Exists"),
                self.tr(
                    "Project directory already exists! Do you want to overwrite it?\nIf not, please manually modify the `Name` field value."
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                try:
                    shutil.rmtree(error_message)
                    self.append_training_log(
                        f"Removed existing directory: {error_message}"
                    )
                except Exception as e:
                    error_msg = f"Failed to remove directory: {str(e)}"
                    logger.error(error_msg)
                    return
            else:
                return
        elif not is_valid:
            QMessageBox.warning(
                self, self.tr("Validation Error"), error_message
            )
            self.append_training_log(f"Validation Error: {error_message}")
            return

        if not self.selected_task_type:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Please select a task type first"),
            )
            return

        if (
            not custom_backend
            and self.selected_task_type.lower() == "pose"
        ):
            pose_config = config["basic"].get("pose_config", "")
            if not pose_config or not os.path.exists(pose_config):
                QMessageBox.warning(
                    self,
                    self.tr("Error"),
                    self.tr(
                        "Please select a valid pose configuration file for pose detection tasks"
                    ),
                )
                return

        if self.training_status in ["completed", "error"]:
            reply = QMessageBox.question(
                self,
                self.tr("Reset Training"),
                self.tr(
                    "Training traces detected. Do you want to reset the training tab?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.reset_train_tab()
        elif self.training_status == "stop":
            self.start_training_button.setVisible(True)
            self.training_status == "idle"

        save_config(config)
        self.go_to_specific_tab(2)

    def init_config_buttons(self, parent_layout):
        button_layout = QHBoxLayout()

        import_btn = SecondaryButton(self.tr("Import Config"))
        import_btn.clicked.connect(self.import_config)
        button_layout.addWidget(import_btn)

        save_btn = SecondaryButton(self.tr("Save Config"))
        save_btn.clicked.connect(self.save_current_config)
        button_layout.addWidget(save_btn)
        button_layout.addStretch()

        previous_btn = SecondaryButton(self.tr("Previous"))
        previous_btn.clicked.connect(lambda: self.go_to_specific_tab(0))
        button_layout.addWidget(previous_btn)

        train_btn = PrimaryButton(self.tr("Next"))
        train_btn.clicked.connect(self.start_training)
        button_layout.addWidget(train_btn)

        parent_layout.addLayout(button_layout)

    def load_default_config(self):
        config = load_config()
        self.load_config_to_ui(config)

    def init_config_tab(self):
        layout = QVBoxLayout(self.config_tab)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        self.init_training_backend_selector(scroll_layout)
        self.init_basic_settings(scroll_layout)
        self.init_custom_training_panel(scroll_layout)
        self.init_train_settings(scroll_layout)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        self.init_config_buttons(layout)
        self.load_default_config()

    # Train tab
    def update_training_status_display(self):
        color = TRAINING_STATUS_COLORS.get(self.training_status, "#6c757d")
        text = self.tr(
            TRAINING_STATUS_TEXTS.get(self.training_status, "Unknown status")
        )
        self.status_label.setText(text)
        self.status_label.setStyleSheet(get_status_label_style(color))

    def update_training_progress(self):
        if not self.current_project_path:
            return

        # Remote runs write results.csv on the server; progress is driven by
        # training_progress events (log / remote CSV poll). Do not block UI on SSH.
        if self.train_mode == "remote":
            return

        if self.training_status == "training" and self.train_mode != "remote":
            log_path = getattr(self.training_manager, "_local_log_path", None) or ""
            if log_path and os.path.isfile(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        body = latest_log_session_text(f.read())
                    text = format_train_log_terminal(body)
                    if text.strip():
                        self._set_training_log_mirror(text)
                except OSError:
                    pass

        results_file = os.path.join(self.current_project_path, "results.csv")
        if os.path.exists(results_file):
            started = float(getattr(self, "_training_started_at", 0) or 0)
            stale_csv = False
            if self.training_status == "training" and started > 0:
                try:
                    stale_csv = os.path.getmtime(results_file) < started - 0.5
                except OSError:
                    stale_csv = True
            if not stale_csv:
                try:
                    with open(results_file, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                        if len(rows) > 1:  # Skip header
                            self._apply_epoch_progress(
                                len(rows) - 1, self.total_epochs
                            )
                except Exception as e:
                    logger.warning(f"Failed to read results.csv: {e}")

    def _on_training_log_scroll(self, value: int) -> None:
        if getattr(self, "_log_scroll_programmatic", False):
            return
        if not hasattr(self, "log_display"):
            return
        bar = self.log_display.verticalScrollBar()
        at_bottom = value >= bar.maximum() - 3
        self._log_follow_tail = at_bottom

    def _set_training_log_mirror(self, text: str) -> None:
        if not hasattr(self, "log_display"):
            return
        text = (text or "").strip()
        if not text:
            return
        follow = bool(getattr(self, "_log_follow_tail", True))
        bar = self.log_display.verticalScrollBar()
        scroll_frac = None
        if not follow and bar.maximum() > 0:
            scroll_frac = bar.value() / float(bar.maximum())
        self.log_display.setPlainText(text)
        self._log_scroll_programmatic = True
        try:
            bar = self.log_display.verticalScrollBar()
            if follow or scroll_frac is None:
                bar.setValue(bar.maximum())
                self._log_follow_tail = True
            else:
                new_max = max(bar.maximum(), 1)
                bar.setValue(int(round(scroll_frac * new_max)))
        finally:
            self._log_scroll_programmatic = False

    def _apply_epoch_progress(self, current_epochs: int, total_epochs: int):
        if total_epochs and total_epochs > 0:
            self.total_epochs = int(total_epochs)
        current_epochs = max(0, int(current_epochs or 0))
        if current_epochs < self.current_epochs and self.current_epochs > 0:
            # Avoid going backwards on partial CSV / duplicate mid-epoch lines
            return
        self.current_epochs = current_epochs
        if self.total_epochs <= 0:
            return
        progress = min(
            100, int((self.current_epochs / self.total_epochs) * 100)
        )
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(
            f"{self.current_epochs}/{self.total_epochs}"
        )

    def update_training_images(self):
        if not self.current_project_path:
            return

        # Remote: images are pushed via training_images events from SSH poll.
        if self.train_mode == "remote":
            cached = getattr(self, "_remote_preview_paths", None) or []
            if cached:
                self._show_training_image_paths(cached)
            return

        def find_images_by_pattern(patterns, max_count=3):
            found_files = []
            for pattern in patterns:
                matches = glob.glob(
                    os.path.join(self.current_project_path, pattern)
                )
                matches.sort()
                found_files.extend(matches)
                if len(found_files) >= max_count:
                    break
            return found_files[:max_count]

        if self.selected_task_type == "Classify":
            image_configs = [
                {"patterns": ["train_batch*.jpg"], "max_count": 3},
                {
                    "patterns": [
                        "val_batch0_labels.jpg",
                        "val_batch0_pred.jpg",
                        "results.png",
                    ],
                    "max_count": 3,
                },
            ]
        else:
            image_configs = [
                {"patterns": ["train_batch*.jpg"], "max_count": 3},
                {
                    "patterns": [
                        "*PR_curve.png",
                        "*F1_curve.png",
                        "results.png",
                    ],
                    "max_count": 3,
                },
            ]

        all_images = []
        for config in image_configs:
            all_images.extend(
                find_images_by_pattern(config["patterns"], config["max_count"])
            )
        self._show_training_image_paths(all_images)

    def _show_training_image_paths(self, all_images):
        for i, image_label in enumerate(self.image_labels):
            if i < len(all_images):
                image_path = all_images[i]
                try:
                    pixmap = QPixmap(image_path)
                    if not pixmap.isNull():
                        scaled_pixmap = pixmap.scaled(
                            150,
                            150,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        image_label.setPixmap(scaled_pixmap)
                        image_label.setText("")
                        image_label.setToolTip(os.path.basename(image_path))
                        self.image_paths[i] = image_path
                    else:
                        image_label.clear()
                        image_label.setText(self.tr("No image"))
                        image_label.setToolTip("")
                        self.image_paths[i] = None
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")
                    image_label.clear()
                    image_label.setText(self.tr("No image"))
                    image_label.setToolTip("")
                    self.image_paths[i] = None
            else:
                image_label.clear()
                image_label.setText(self.tr("No image"))
                image_label.setToolTip("")
                self.image_paths[i] = None

    def on_training_event(self, event_type, data):
        if event_type == "training_started":
            self.training_status = "training"
            self.total_epochs = data["total_epochs"]
            self.current_epochs = 0
            self._training_started_at = float(
                getattr(self.training_manager, "_local_run_started_at", 0) or 0
            )
            self.progress_bar.setValue(0)
            te = int(data.get("total_epochs") or self.total_epochs or 0)
            if data.get("custom") or self._is_custom_training_backend():
                if te > 0:
                    self.total_epochs = te
                    self.progress_bar.setRange(0, 100)
                    self.progress_bar.setFormat(f"0/{te}")
                else:
                    self.progress_bar.setRange(0, 100)
                    self.progress_bar.setValue(0)
                    self.progress_bar.setFormat(self.tr("Training…"))
            else:
                self.progress_bar.setRange(0, max(1, self.total_epochs or 1))
                self.progress_bar.setFormat(f"0/{self.total_epochs}")
            self._remote_preview_paths = []
            if hasattr(self, "log_display"):
                self.log_display.clear()
                self._log_follow_tail = True
                if data.get("reattached"):
                    log_path = getattr(
                        self.training_manager, "_local_log_path", None
                    ) or getattr(self.training_manager, "_remote_log_path", None)
                    if log_path and os.path.isfile(str(log_path)):
                        try:
                            with open(
                                log_path, "r", encoding="utf-8", errors="replace"
                            ) as _lf:
                                body = latest_log_session_text(_lf.read())
                            text = format_train_log_terminal(body)
                            if text.strip():
                                self._set_training_log_mirror(text)
                        except OSError:
                            pass
            if hasattr(self, "image_labels") and not data.get("reattached"):
                for i, image_label in enumerate(self.image_labels):
                    image_label.clear()
                    image_label.setText(self.tr("No image"))
                    image_label.setToolTip("")
                    self.image_paths[i] = None
            self.update_training_status_display()
            self.start_training_button.setVisible(False)
            self.stop_training_button.setVisible(True)
            self._set_post_train_buttons(visible=False)
            self.previous_button.setVisible(False)
            self.progress_timer.start(1000)
            self.image_timer.start(5000)
            tb = data.get("tensorboard_url") or getattr(
                self.training_manager, "tensorboard_url", ""
            )
            if tb:
                self.tensorboard_url = tb
                if hasattr(self, "open_tb_button"):
                    self.open_tb_button.setVisible(True)
        elif event_type == "training_completed":
            self.training_status = "completed"
            self.update_training_status_display()
            self.stop_training_button.setVisible(False)
            self.start_training_button.setVisible(False)
            self.previous_button.setVisible(True)
            self._set_post_train_buttons(visible=True)
            self.progress_timer.stop()
            self.image_timer.stop()
            self.update_training_progress()
            self.update_training_images()
            log_path = getattr(self.training_manager, "_local_log_path", None) or ""
            if log_path and os.path.isfile(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        text = format_train_log_terminal(
                            latest_log_session_text(f.read())
                        )
                    if text.strip():
                        self._set_training_log_mirror(text)
                except OSError:
                    pass
            tb = data.get("tensorboard_url") or getattr(
                self.training_manager, "tensorboard_url", ""
            )
            if tb:
                self.tensorboard_url = tb
                if hasattr(self, "open_tb_button"):
                    self.open_tb_button.setVisible(True)
            if hasattr(self, "ssh_reattach_btn"):
                self._refresh_reattach_button()
        elif event_type == "training_error":
            self.training_status = "error"
            self.update_training_status_display()
            self.start_training_button.setVisible(False)
            self.previous_button.setVisible(True)
            self.stop_training_button.setVisible(False)
            self._set_post_train_buttons(visible=False)
            self.progress_timer.stop()
            self.image_timer.stop()
            error_msg = data.get("error", "Unknown error occurred")
            self.append_training_log(f"ERROR: {error_msg}")
            if hasattr(self, "ssh_reattach_btn"):
                self._refresh_reattach_button()
        elif event_type == "training_detached":
            # Network lost: remote job continues; allow reattach
            self.training_status = "stop"
            self.update_training_status_display()
            self.start_training_button.setVisible(True)
            self.previous_button.setVisible(True)
            self.stop_training_button.setVisible(False)
            self._set_post_train_buttons(visible=False)
            self.progress_timer.stop()
            self.image_timer.stop()
            err = data.get("error", "")
            hint = data.get(
                "hint",
                "Remote job kept running. Reconnect SSH and click Reattach.",
            )
            self.append_training_log(f"DETACHED: {hint}")
            if err:
                self.append_training_log(f"DETACHED detail: {err}")
            if hasattr(self, "ssh_reattach_btn"):
                self.ssh_reattach_btn.setEnabled(True)
            if not getattr(self, "_application_closing", False):
                QMessageBox.warning(
                    self,
                    self.tr("Connection Lost"),
                    self.tr(
                        "Lost contact with the training monitor.\n"
                        "The training job is still running.\n\n"
                        "Re-open MyTrain and click Reattach Job to continue."
                    ),
                )
        elif event_type == "training_stopped":
            self.training_status = "stop"
            self.update_training_status_display()
            self.start_training_button.setVisible(False)
            self.previous_button.setVisible(True)
            self.stop_training_button.setVisible(False)
            self._set_post_train_buttons(visible=False)
            self.progress_timer.stop()
            self.image_timer.stop()
            self.append_training_log(self.tr("Training stopped by user"))
            if hasattr(self, "ssh_reattach_btn"):
                self._refresh_reattach_button()
        elif event_type == "training_log_snapshot":
            self._set_training_log_mirror(data.get("text", ""))
        elif event_type == "training_log":
            log_message = data.get("message", "")
            if self.training_status == "training":
                if log_message.startswith("TensorBoard:"):
                    self.tensorboard_url = log_message.split("TensorBoard:", 1)[
                        -1
                    ].strip()
                    if hasattr(self, "open_tb_button"):
                        self.open_tb_button.setVisible(True)
            elif log_message:
                self.append_training_log(log_message)
        elif event_type == "training_progress":
            cur = int(data.get("current_epochs", 0) or 0)
            self._apply_epoch_progress(
                cur,
                data.get("total_epochs", self.total_epochs),
            )
        elif event_type == "training_images":
            paths = [
                p
                for p in (data.get("paths") or [])
                if p and os.path.isfile(p)
            ]
            if paths:
                self._remote_preview_paths = paths
                self._show_training_image_paths(paths)

    def append_training_log(self, text):
        if not hasattr(self, "log_display"):
            return
        if self.training_status == "training":
            return
        text = format_train_log_terminal(text or "")
        if not text.strip():
            return
        self.log_display.append(text.strip())
        if getattr(self, "_log_follow_tail", True):
            self._log_scroll_programmatic = True
            try:
                bar = self.log_display.verticalScrollBar()
                bar.setValue(bar.maximum())
            finally:
                self._log_scroll_programmatic = False

    def init_training_status(self, parent_layout):
        status_group = QGroupBox(self.tr("Training Status"))
        status_layout = QVBoxLayout(status_group)

        self.status_label = QLabel(self.tr("Ready to train"))
        self.status_label.setStyleSheet(get_status_label_style())
        status_layout.addWidget(self.status_label)

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel(self.tr("Progress:")))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0/0")
        self.progress_bar.setStyleSheet(get_progress_bar_style())
        progress_layout.addWidget(self.progress_bar)
        status_layout.addLayout(progress_layout)
        parent_layout.addWidget(status_group)

    def clear_training_logs(self):
        if hasattr(self, "log_display"):
            reply = QMessageBox.question(
                self,
                self.tr("Clear Logs"),
                self.tr("Are you sure you want to clear all training logs?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.log_display.clear()
                self._log_follow_tail = True

    def copy_training_logs(self):
        if hasattr(self, "log_display"):
            text = self.log_display.toPlainText()
            if text:
                clipboard = QApplication.clipboard()
                clipboard.setText(text)

    def init_training_logs(self, parent_layout):
        logs_group = QGroupBox(self.tr("Training Logs"))
        logs_layout = QVBoxLayout(logs_group)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(250)
        self.log_display.setStyleSheet(get_log_display_style())
        self.log_display.verticalScrollBar().valueChanged.connect(
            self._on_training_log_scroll
        )
        logs_layout.addWidget(self.log_display)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.clear_logs_button = SecondaryButton(self.tr("Clear"))
        self.clear_logs_button.clicked.connect(self.clear_training_logs)
        button_layout.addWidget(self.clear_logs_button)

        self.copy_logs_button = SecondaryButton(self.tr("Copy"))
        self.copy_logs_button.clicked.connect(self.copy_training_logs)
        button_layout.addWidget(self.copy_logs_button)

        logs_layout.addLayout(button_layout)
        parent_layout.addWidget(logs_group)

    def init_training_images(self, parent_layout):
        images_group = QGroupBox(self.tr("Training Images"))
        images_layout = QVBoxLayout(images_group)
        images_layout.setContentsMargins(5, 5, 5, 5)

        self.image_labels = []
        self.image_paths = [None] * 6
        self.images_widget = QWidget()
        images_row_layout = QHBoxLayout(self.images_widget)
        images_row_layout.setSpacing(10)
        images_row_layout.setContentsMargins(0, 0, 0, 0)

        for i in range(6):
            image_label = QLabel()
            image_label.setMinimumSize(150, 150)
            image_label.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            image_label.setStyleSheet(get_image_label_style())
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setText(self.tr("No image"))
            image_label.setScaledContents(False)
            image_label.mousePressEvent = (
                lambda event, idx=i: self.on_image_clicked(idx)
            )
            self.image_labels.append(image_label)
            images_row_layout.addWidget(image_label, 1)

        images_layout.addWidget(self.images_widget, 1)
        parent_layout.addWidget(images_group, 1)

    def on_image_clicked(self, index):
        if self.image_paths[index]:
            self.open_image_file(self.image_paths[index])

    def open_image_file(self, image_path):
        try:
            is_wsl2 = False
            try:
                if (
                    hasattr(os, "uname")
                    and "microsoft" in os.uname().release.lower()
                ):
                    is_wsl2 = True
            except (AttributeError, OSError):
                pass

            if is_wsl2:  # WSL2
                windows_path = (
                    subprocess.check_output(["wslpath", "-w", image_path])
                    .decode()
                    .strip()
                )
                subprocess.run(
                    [
                        "powershell.exe",
                        "-c",
                        f'Start-Process "{windows_path}"',
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif os.name == "nt":  # Windows
                os.startfile(image_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", image_path])
            elif os.name == "posix":  # Linux
                subprocess.run(["xdg-open", image_path])
        except Exception as e:
            logger.warning(f"Failed to open image {image_path}: {e}")

    def open_training_directory(self):
        if self.current_project_path and os.path.exists(
            self.current_project_path
        ):
            try:
                is_wsl2 = False
                try:
                    if (
                        hasattr(os, "uname")
                        and "microsoft" in os.uname().release.lower()
                    ):
                        is_wsl2 = True
                except (AttributeError, OSError):
                    pass

                if is_wsl2:  # WSL2
                    wsl_path = self.current_project_path
                    windows_path = (
                        subprocess.check_output(["wslpath", "-w", wsl_path])
                        .decode()
                        .strip()
                    )
                    subprocess.run(["explorer.exe", windows_path])
                elif os.name == "nt":  # Windows
                    os.startfile(self.current_project_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", self.current_project_path])
                elif os.name == "posix":  # Linux
                    subprocess.run(["xdg-open", self.current_project_path])
            except Exception as e:
                self.append_training_log(f"Failed to open directory: {str(e)}")
                QMessageBox.information(
                    self,
                    self.tr("Info"),
                    f"Directory path: {self.current_project_path}",
                )
        else:
            QMessageBox.information(
                self,
                self.tr("Info"),
                self.tr("No training directory available"),
            )

    def stop_training(self):
        reply = QMessageBox.question(
            self,
            self.tr("Confirm Stop"),
            self.tr("Are you sure you want to stop the training?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            success = self.training_manager.stop_training()
            if success:
                self.append_training_log(self.tr("Stopping training..."))
            else:
                self.append_training_log(self.tr("Cancel to stop training"))

    def get_training_args(self, config):
        try:
            data_cfg = config["basic"]["data"]
            # Primary Detect path: use pre-built data.yaml (local or remote path)
            if self.selected_task_type == "Detect" and self.data_yaml_path:
                data_path = self.data_yaml_path
                self.append_training_log(f"Using dataset yaml: {data_path}")
            elif self.selected_task_type == "Classify" and os.path.isdir(
                data_cfg
            ):
                data_path = data_cfg
                self.append_training_log(
                    f"Using existing dataset: {data_path}"
                )
            elif (
                self.selected_task_type == "Detect"
                and isinstance(data_cfg, str)
                and data_cfg.endswith((".yaml", ".yml"))
                and (
                    self.train_mode == "remote"
                    or os.path.isfile(data_cfg)
                )
            ):
                data_path = data_cfg
                self.append_training_log(f"Using dataset yaml: {data_path}")
            else:
                # Legacy: build from open image_list
                temp_dir = create_yolo_dataset(
                    self.image_list,
                    self.selected_task_type,
                    config["basic"]["dataset_ratio"],
                    config["basic"]["data"],
                    self.output_dir,
                    config["basic"].get("pose_config"),
                    config["checkpoint"].get("skip_empty_files", False),
                )
                logger.info(f"Successfully created YOLO dataset at {temp_dir}")
                self.append_training_log(f"Created dataset: {temp_dir}")

                if self.selected_task_type == "Classify":
                    data_path = temp_dir
                else:
                    data_path = os.path.join(temp_dir, "data.yaml")

            device_value = self._resolve_train_device(config["basic"]["device"])
            self.append_training_log(f"Using device: {device_value}")

            model_path = config["basic"]["model"]
            project_path = config["basic"]["project"]
            if (
                self.train_mode == "remote"
                and getattr(self.ssh_client, "connected", False)
                and model_path
                and os.path.isfile(model_path)
            ):
                remote_project = str(project_path or "").replace("\\", "/")
                if remote_project:
                    self.ssh_client.mkdir_p(remote_project)
                    uploaded = upload_model_if_local(
                        self.ssh_client, model_path, remote_project
                    )
                    self.append_training_log(
                        self.tr("Uploaded local model to %s") % uploaded
                    )
                    model_path = uploaded

            train_args = {
                "data": data_path,
                "model": model_path,
                "project": project_path,
                "name": config["basic"]["name"],
                # Keep the exact Name from UI; Ultralytics would otherwise
                # increment to name-2 / name-3 when the save dir already exists.
                "exist_ok": True,
                "device": device_value,
                "_task": (self.selected_task_type or "detect").lower(),
            }

            # Add advanced parameters
            advanced_params = {}
            for section in [
                "train",
                "strategy",
                "learning_rate",
                "warmup",
                "augment",
                "regularization",
                "loss_weights",
                "checkpoint",
            ]:
                advanced_params.update(config.get(section, {}))
            # Skip UI defaults that match DEFAULT_TRAINING_CONFIG to keep the
            # train() kwargs lean. Exception: always forward core train knobs
            # (epochs/batch/imgsz/workers) — remote containers often ship a
            # different Ultralytics default.yaml (e.g. epochs=5 for demos).
            # Progress bar used to show UI 100 while the server used its own 5.
            _always_pass = {"epochs", "batch", "imgsz", "workers"}
            freeze_raw = advanced_params.get("freeze")
            freeze_parsed = parse_freeze_value(freeze_raw)
            if freeze_parsed is None:
                advanced_params.pop("freeze", None)
            else:
                advanced_params["freeze"] = freeze_parsed

            advanced_params = {
                key: value
                for key, value in advanced_params.items()
                if key in DEFAULT_TRAINING_CONFIG
                and (
                    key in _always_pass
                    or key == "freeze"
                    or value != DEFAULT_TRAINING_CONFIG[key]
                )
            }
            # skip_empty_files is UI-only
            advanced_params.pop("skip_empty_files", None)
            # time=0 means "not set" in the UI (shown as None); do not pass it
            # or Ultralytics may treat it as a time budget.
            if advanced_params.get("time") in (0, 0.0, None):
                advanced_params.pop("time", None)
            train_args.update(advanced_params)
            self.total_epochs = int(
                train_args.get("epochs")
                or (config.get("train") or {}).get("epochs")
                or 100
            )

            cmd_parts = ["yolo", self.selected_task_type.lower(), "train"]
            for key, value in train_args.items():
                if key.startswith("_"):
                    continue
                cmd_parts.append(f"{key}={value}")
            self.append_training_log(
                f"Training command: {' '.join(cmd_parts)}"
            )

            return train_args

        except Exception as e:
            self.append_training_log(
                f"Error preparing training args: {str(e)}"
            )
            raise

    def build_custom_training_spec(self, config):
        """Resolve paths and substitute placeholders in the custom command."""
        custom = config.get("custom") or {}
        template = (custom.get("command") or "").strip()
        if not template:
            raise ValueError(self.tr("Custom command is empty"))

        self.append_training_log(self.tr("Resolving training paths..."))
        train_args = self.get_training_args(config)
        remote = self.train_mode == "remote"
        model = train_args.get("model") or ""
        data = train_args.get("data") or ""
        project = train_args.get("project") or ""
        name = train_args.get("name") or "exp"
        if remote and self.ssh_client.connected:
            remote_project = project.replace("\\", "/")
            self.ssh_client.mkdir_p(remote_project)
            model = upload_model_if_local(
                self.ssh_client, model, remote_project
            )
        paths = build_path_context(
            model=model,
            data=data,
            project=project,
            name=name,
            device=train_args.get("device"),
        )
        if remote:
            for key in paths:
                paths[key] = posix_if_needed(paths[key], remote=True)
        resolved = substitute_custom_command(template, paths)
        py_exe = ""
        if self.train_mode == "interpreter":
            py_exe = self._selected_interpreter_python() or ""
            if py_exe:
                before = resolved
                resolved = bind_command_to_python(resolved, py_exe)
                if resolved != before:
                    self.append_training_log(
                        self.tr(
                            "Interpreter: using YOLO CLI from this env "
                            "(not system yolo on PATH): %s"
                        )
                        % py_exe
                    )
        elif remote:
            remote_py = self._remote_python_for_exec() or ""
            if remote_py:
                before = resolved
                resolved = bind_remote_command_to_python(resolved, remote_py)
                if resolved != before:
                    self.append_training_log(
                        self.tr(
                            "Remote: %s is first on PATH so 'yolo' resolves "
                            "inside this env."
                        )
                        % remote_py
                    )
        exec_cwd = (custom.get("exec_cwd") or "").strip()
        if remote and is_windows_local_path(exec_cwd):
            self.append_training_log(
                self.tr(
                    "Ignored local execution directory for remote training: %s"
                )
                % exec_cwd
            )
            exec_cwd = ""
        elif remote:
            exec_cwd = posix_if_needed(exec_cwd, remote=True)

        self.append_training_log(self.tr("Effective model: %s") % paths["model"])
        self.append_training_log(self.tr("Effective data: %s") % paths["data"])
        self.append_training_log(
            self.tr("Effective device (Basic Settings): %s") % paths["device"]
        )
        self.append_training_log(
            self.tr(
                "Basic fields (model, data, project, name, device) in the "
                "command are replaced by Config values."
            )
        )
        self.append_training_log(self.tr("Resolved command: %s") % resolved)

        epochs = (config.get("train") or {}).get("epochs")
        if epochs is None:
            epochs = parse_epochs_from_cli(resolved)
        try:
            epochs = max(0, int(epochs or 0))
        except (TypeError, ValueError):
            epochs = parse_epochs_from_cli(resolved)

        return {
            "resolved_command": resolved,
            "exec_cwd": exec_cwd,
            "project": paths["project"],
            "name": paths["name"],
            "run_dir": paths["run_dir"],
            "epochs": epochs,
        }

    def start_training_from_train_tab(self):
        if self.train_mode == "remote" and not self._ensure_selected_server_ready():
            return
        if (
            self.train_mode == "interpreter"
            and not self._ensure_selected_interpreter_ready()
        ):
            return
        if not self._ensure_remote_train_allowed():
            return

        if self.train_mode == "remote" and self.ssh_client.connected:
            self._apply_remote_project_dir()
        config = self.get_current_config()
        custom_backend = (config.get("training_backend") or "ultralytics") == "custom"
        project_path = config["basic"]["project"]
        name = config["basic"]["name"]
        self.current_project_path = self._join_run_dir(project_path, name)
        if self.train_mode == "remote":
            self._remote_run_dir_path = self.current_project_path

        model_path = (config.get("basic") or {}).get("model") or ""
        imgsz = (config.get("train") or {}).get("imgsz")
        if imgsz is None:
            imgsz = self._current_imgsz()
        self.export_meta = resolve_export_meta(
            model_path=model_path,
            imgsz=imgsz,
            num_classes=len(self.names or []),
            dataset_root=self.dataset_root
            or self.remote_dataset_root
            or "",
            subdirs=self._selected_subdirs(),
        )

        try:
            self.append_training_log(self.tr("Preparing training..."))
            py_exe = self._selected_interpreter_python()
            if py_exe:
                self.append_training_log(
                    self.tr("Training Python: %s") % py_exe
                )
            start_mode = (
                "remote" if self.train_mode == "remote" else "local"
            )
            remote_py = ""
            if self.train_mode == "remote":
                rp = self._remote_python_for_exec()
                if rp:
                    remote_py = rp
                    self.append_training_log(
                        self.tr("Remote Python: %s") % remote_py
                    )
            if custom_backend:
                spec = self.build_custom_training_spec(config)
                success, message = self.training_manager.start_custom_training(
                    spec,
                    mode=start_mode,
                    ssh_client=self.ssh_client
                    if self.train_mode == "remote"
                    else None,
                    python_executable=py_exe,
                    remote_python=remote_py,
                )
            else:
                train_args = self.get_training_args(config)
                success, message = self.training_manager.start_training(
                    train_args,
                    mode=start_mode,
                    ssh_client=self.ssh_client
                    if self.train_mode == "remote"
                    else None,
                    python_executable=py_exe,
                    remote_python=remote_py,
                )
            if not success:
                self.append_training_log(
                    f"Failed to start training: {message}"
                )
                QMessageBox.critical(self, self.tr("Training Error"), message)
                return
            tb = getattr(self.training_manager, "tensorboard_url", "")
            if tb:
                self.tensorboard_url = tb
                if hasattr(self, "open_tb_button"):
                    self.open_tb_button.setVisible(True)

        except Exception as e:
            error_msg = f"Failed to start training: {str(e)}"
            self.append_training_log(f"ERROR: {error_msg}")
            QMessageBox.critical(self, self.tr("Training Error"), error_msg)

    def open_tensorboard(self):
        url = self.tensorboard_url or getattr(
            self.training_manager, "tensorboard_url", ""
        )
        # Prefer remote host:port when in remote mode (published Docker port)
        if self.train_mode == "remote" and self.ssh_client.connected:
            host = getattr(self.ssh_client, "host", "") or ""
            if host:
                port = 6006
                job_port = None
                try:
                    from anylabeling.services.auto_training.mytrain.remote.job_state import (
                        load_remote_job,
                    )

                    job = load_remote_job() or {}
                    job_port = job.get("tensorboard_remote_port")
                except Exception:
                    job_port = None
                if job_port:
                    port = int(job_port)
                elif ":" in (url or ""):
                    try:
                        port = int(url.rstrip("/").rsplit(":", 1)[-1])
                    except ValueError:
                        port = 6006
                url = f"http://{host}:{port}"
        if not url:
            QMessageBox.information(
                self,
                self.tr("TensorBoard"),
                self.tr("TensorBoard URL is not available yet."),
            )
            return
        import webbrowser

        webbrowser.open(url)
        self.append_training_log(f"Opened TensorBoard: {url}")

    def init_training_actions(self, parent_layout):
        actions_layout = QHBoxLayout()

        self.open_dir_button = SecondaryButton(self.tr("Open Directory"))
        self.open_dir_button.clicked.connect(self.open_training_directory)
        actions_layout.addWidget(self.open_dir_button)

        self.open_tb_button = SecondaryButton(self.tr("Open TensorBoard"))
        self.open_tb_button.clicked.connect(self.open_tensorboard)
        self.open_tb_button.setVisible(False)
        actions_layout.addWidget(self.open_tb_button)
        actions_layout.addStretch()

        self.stop_training_button = SecondaryButton(self.tr("Stop Training"))
        self.stop_training_button.clicked.connect(self.stop_training)
        self.stop_training_button.setVisible(False)
        actions_layout.addWidget(self.stop_training_button)

        self.previous_button = SecondaryButton(self.tr("Previous"))
        self.previous_button.clicked.connect(
            lambda: self.go_to_specific_tab(1)
        )
        self.previous_button.setVisible(True)
        actions_layout.addWidget(self.previous_button)

        self.start_training_button = PrimaryButton(self.tr("Start Training"))
        self.start_training_button.clicked.connect(
            self.start_training_from_train_tab
        )
        actions_layout.addWidget(self.start_training_button)

        self.export_button = PrimaryButton(self.tr("Export ONNX"))
        self.export_button.clicked.connect(self.start_export)
        self.export_button.setVisible(False)
        actions_layout.addWidget(self.export_button)

        self.download_button = PrimaryButton(self.tr("Download Folder"))
        self.download_button.clicked.connect(self.start_download_run_folder)
        self.download_button.setVisible(False)
        actions_layout.addWidget(self.download_button)

        self.infer_button = PrimaryButton(self.tr("Infer"))
        self.infer_button.clicked.connect(self.start_predict)
        self.infer_button.setVisible(False)
        actions_layout.addWidget(self.infer_button)

        self.batch_infer_button = PrimaryButton(self.tr("Batch Infer"))
        self.batch_infer_button.clicked.connect(self.start_batch_predict)
        self.batch_infer_button.setVisible(False)
        actions_layout.addWidget(self.batch_infer_button)

        self.download_batch_button = SecondaryButton(
            self.tr("Download Batch Results")
        )
        self.download_batch_button.clicked.connect(
            self.start_download_batch_results
        )
        self.download_batch_button.setVisible(False)
        actions_layout.addWidget(self.download_batch_button)

        parent_layout.addLayout(actions_layout)

    def init_train_tab(self):
        layout = QVBoxLayout(self.train_tab)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        self.init_training_status(scroll_layout)
        self.init_training_logs(scroll_layout)
        self.init_training_images(scroll_layout)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        self.init_training_actions(layout)

    def on_export_event(self, event_type, data):
        if event_type == "export_started":
            self.append_training_log(self.tr("Export started..."))
            self.export_button.setEnabled(False)
        elif event_type == "export_completed":
            exported_path = data.get("exported_path", "")
            export_format = data.get("format", "onnx")
            self.append_training_log(
                self.tr(
                    f"Export completed successfully! File saved to: {exported_path}"
                )
            )
            QMessageBox.information(
                self,
                self.tr("Export Successful"),
                self.tr(
                    f"Model successfully exported to {export_format.upper()} format:\n{exported_path}"
                ),
            )
            self.export_button.setEnabled(True)
        elif event_type == "export_error":
            error_msg = data.get("error", "Unknown error occurred")
            self.append_training_log(f"ERROR: {error_msg}")
            QMessageBox.warning(self, self.tr("Export Error"), error_msg)
            self.export_button.setEnabled(True)
        elif event_type == "export_log":
            log_message = data.get("message", "")
            if log_message:
                self.append_training_log(log_message)

    def _join_run_dir(self, project: str, name: str) -> str:
        """Join project/name with OS-correct separators (POSIX when remote)."""
        project = (project or "").strip()
        name = (name or "exp").strip() or "exp"
        if self.train_mode == "remote" or project.replace("\\", "/").startswith(
            "/"
        ):
            return posix_join(project.replace("\\", "/"), name)
        return os.path.join(project, name)

    def _set_post_train_buttons(self, *, visible: bool) -> None:
        if hasattr(self, "export_button"):
            self.export_button.setVisible(visible)
        if hasattr(self, "infer_button"):
            self.infer_button.setVisible(visible)
        if hasattr(self, "batch_infer_button"):
            self.batch_infer_button.setVisible(visible)
        if hasattr(self, "download_button"):
            self.download_button.setVisible(
                visible and self.train_mode == "remote"
            )
        if hasattr(self, "download_batch_button"):
            # Shown after a successful batch infer (local or remote)
            if not visible:
                self.download_batch_button.setVisible(False)

    def _remote_run_dir(self) -> str:
        remote = getattr(self, "_remote_run_dir_path", None) or ""
        if not remote:
            remote = self.current_project_path or ""
        return remote.replace("\\", "/").rstrip("/")

    def _remote_best_weights_path(self) -> str:
        return posix_join(self._remote_run_dir(), "weights", "best.pt")

    def _download_remote_run_folder(self, *, force: bool = False):
        """
        Ask for a local parent folder and download the whole remote run dir.

        Returns (local_run_dir, error_message).
        Deprecated path: prefer start_download_run_folder (async). Kept for
        any sync callers; UI dialogs stay on the calling thread.
        """
        prepared = self._prepare_download_destination(force=force)
        if prepared is None:
            return None, ""
        remote_run, local_run, err = prepared
        if err:
            return None, err
        if not remote_run and local_run:
            return local_run, ""
        ok, payload = self._transfer_remote_run_folder(remote_run, local_run)
        if not ok:
            return None, str(payload)
        self.current_project_path = payload
        return payload, ""

    def _prepare_download_destination(self, *, force: bool = False):
        """UI-only: resolve remote/local paths. Returns (remote, local, err) or None if cancelled."""
        if not self.ssh_client or not self.ssh_client.connected:
            return None, None, "Remote SSH is not connected; reconnect to save."

        remote_run = self._remote_run_dir()
        if not remote_run or not remote_run.startswith("/"):
            return None, None, "No remote training directory available."

        local_existing = self.current_project_path or ""
        local_weights = os.path.join(local_existing, "weights", "best.pt")
        if (
            not force
            and local_existing
            and os.path.isfile(local_weights)
            and not local_existing.replace("\\", "/").startswith("/")
        ):
            reply = QMessageBox.question(
                self,
                self.tr("Local Copy Found"),
                self.tr(
                    f"A local training folder already exists:\n{local_existing}\n\n"
                    "Yes - Use this folder\n"
                    "No - Download again from remote"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                return None, local_existing, ""

        try:
            if not self.ssh_client.exists(remote_run):
                return None, None, f"Remote training folder not found: {remote_run}"
        except Exception as e:
            return None, None, f"Failed to check remote folder: {e}"

        run_name = os.path.basename(remote_run) or "exp"
        task = (self.selected_task_type or "detect").lower()
        default_parent = os.path.join(get_default_project_dir(), task)
        os.makedirs(default_parent, exist_ok=True)

        parent = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select folder to save the training run"),
            default_parent,
        )
        if not parent:
            return None

        local_run = os.path.join(parent, run_name)
        if os.path.exists(local_run):
            reply = QMessageBox.question(
                self,
                self.tr("Directory Exists"),
                self.tr(
                    f"Local folder already exists:\n{local_run}\n\n"
                    "Overwrite it?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return None
            try:
                shutil.rmtree(local_run)
            except Exception as e:
                return None, None, f"Failed to clear local folder: {e}"

        return remote_run, local_run, ""

    def _transfer_remote_run_folder(self, remote_run: str, local_run: str):
        """Download remote run folder (no UI). Returns (ok, local_path_or_error)."""
        self._bg_bridge.log.emit(
            f"Downloading remote run folder:\n  {remote_run}\n-> {local_run}"
        )
        try:
            file_count = {"n": 0}

            def _prog(_rp: str) -> None:
                file_count["n"] += 1
                if file_count["n"] == 1 or file_count["n"] % 20 == 0:
                    self._bg_bridge.log.emit(
                        f"  ↓ downloaded {file_count['n']} files..."
                    )

            self.ssh_client.download_directory(
                remote_run, local_run, progress=_prog
            )
            self._bg_bridge.log.emit(
                f"  ↓ done, {file_count['n']} files total"
            )
        except Exception as e:
            return False, f"Failed to download remote training folder: {e}"

        weights = os.path.join(local_run, "weights", "best.pt")
        if not os.path.isfile(weights):
            last_pt = os.path.join(local_run, "weights", "last.pt")
            if not os.path.isfile(last_pt):
                return (
                    False,
                    f"Download finished but weights missing under:\n"
                    f"{os.path.join(local_run, 'weights')}",
                )
        return True, local_run

    def _set_post_train_job_buttons_enabled(self, enabled: bool) -> None:
        for name in (
            "export_button",
            "infer_button",
            "batch_infer_button",
            "download_button",
            "download_batch_button",
        ):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(enabled)

    def _start_bg_task(self, task_name: str, worker_fn) -> bool:
        """Run ``worker_fn() -> (ok, payload)`` off the UI thread."""
        if self._bg_busy:
            QMessageBox.information(
                self,
                self.tr("Busy"),
                self.tr(
                    "Another job is still running "
                    "(export / infer / batch infer / download). "
                    "Please wait for it to finish."
                ),
            )
            return False
        self._bg_busy = True
        self._set_post_train_job_buttons_enabled(False)

        def _run():
            try:
                ok, payload = worker_fn()
                self._bg_bridge.finished.emit(task_name, bool(ok), payload)
            except Exception as e:
                self._bg_bridge.finished.emit(task_name, False, str(e))

        threading.Thread(
            target=_run, daemon=True, name=f"xal-{task_name}"
        ).start()
        return True

    def _on_bg_task_finished(self, task_name: str, ok: bool, payload) -> None:
        try:
            self._bg_busy = False
            self._set_post_train_job_buttons_enabled(True)
        except RuntimeError:
            return

        try:
            if task_name == "export":
                if not ok:
                    self.append_training_log(f"ERROR: {payload}")
                    QMessageBox.warning(
                        self, self.tr("Export Error"), self.tr(str(payload))
                    )
                    return
                self.append_training_log(f"ONNX saved on remote: {payload}")
                QMessageBox.information(
                    self,
                    self.tr("Export Successful"),
                    self.tr(
                        f"ONNX saved to remote weights/:\n{payload}\n\n"
                        f"Use “Download Folder” to pull the whole training run."
                    ),
                )
            elif task_name == "predict":
                if not ok:
                    self.append_training_log(f"ERROR: {payload}")
                    QMessageBox.warning(
                        self, self.tr("Infer Failed"), self.tr(str(payload))
                    )
                    return
                self.append_training_log(
                    f"Predict finished. YOLO labels: {payload}\n"
                    "Use Download Folder (remote) or Open Directory (local) "
                    "to get results."
                )
                QMessageBox.information(
                    self,
                    self.tr("Infer Complete"),
                    self.tr(
                        f"YOLO txt labels saved to:\n{payload}\n\n"
                        "Remote: use Download Folder to pull the whole "
                        "training run (includes predict/labels).\n"
                        "Local: use Open Directory."
                    ),
                )
            elif task_name == "batch_predict":
                if not ok:
                    self.append_training_log(f"ERROR: {payload}")
                    QMessageBox.warning(
                        self,
                        self.tr("Batch Infer Failed"),
                        self.tr(str(payload)),
                    )
                    return
                labels_dir = str(payload or "")
                self._last_batch_infer_labels = labels_dir
                if hasattr(self, "download_batch_button"):
                    self.download_batch_button.setVisible(True)
                self.append_training_log(
                    f"Batch predict finished. YOLO labels: {labels_dir}\n"
                    "Use “Download Batch Results” to save the labels folder."
                )
                QMessageBox.information(
                    self,
                    self.tr("Batch Infer Complete"),
                    self.tr(
                        f"YOLO txt labels saved to:\n{labels_dir}\n\n"
                        "Click “Download Batch Results” to copy/download "
                        "this labels folder."
                    ),
                )
            elif task_name == "download_batch":
                if not ok:
                    QMessageBox.warning(
                        self,
                        self.tr("Download Failed"),
                        self.tr(str(payload)),
                    )
                    return
                self.append_training_log(
                    f"Batch results saved to: {payload}"
                )
                QMessageBox.information(
                    self,
                    self.tr("Saved"),
                    self.tr(f"Batch labels folder saved to:\n{payload}"),
                )
            elif task_name == "free_infer":
                if hasattr(self, "free_infer_panel"):
                    self.free_infer_panel.set_busy(False)
                if not ok:
                    self.append_training_log(f"[Free Infer] ERROR: {payload}")
                    if hasattr(self, "free_infer_panel"):
                        self.free_infer_panel.set_status(str(payload))
                    QMessageBox.warning(
                        self,
                        self.tr("Free Inference Failed"),
                        self.tr(str(payload)),
                    )
                    return
                data = payload if isinstance(payload, dict) else {}
                job_dir = data.get("job_dir", "")
                source_dir = data.get("source_dir", "")
                labels_dir = data.get("labels_dir", "")
                kind = data.get("kind", "")
                if hasattr(self, "free_infer_panel"):
                    self.free_infer_panel.set_last_result(
                        job_dir, source_dir, labels_dir
                    )
                    self.free_infer_panel.set_status(
                        self.tr(
                            "Inference complete ({kind}). "
                            "Use Download Images + Labels to export."
                        ).format(kind=kind or "images")
                    )
                self.append_training_log(
                    f"[Free Infer] done labels={labels_dir} source={source_dir}"
                )
                QMessageBox.information(
                    self,
                    self.tr("Free Inference Complete"),
                    self.tr(
                        "Inference finished.\n\n"
                        "Labels:\n{labels}\n\n"
                        "Source images (frames for video):\n{source}\n\n"
                        "Click “Download Images + Labels” to save both "
                        "(original video is not included)."
                    ).format(labels=labels_dir, source=source_dir),
                )
            elif task_name == "free_infer_download":
                if hasattr(self, "free_infer_panel"):
                    self.free_infer_panel.set_busy(False)
                if not ok:
                    QMessageBox.warning(
                        self,
                        self.tr("Download Failed"),
                        self.tr(str(payload)),
                    )
                    return
                self.append_training_log(
                    f"[Free Infer] packaged: {payload}"
                )
                if hasattr(self, "free_infer_panel"):
                    self.free_infer_panel.set_status(str(payload))
                QMessageBox.information(
                    self,
                    self.tr("Saved"),
                    self.tr(str(payload)),
                )
            elif task_name == "download":
                if not ok:
                    QMessageBox.warning(
                        self, self.tr("Save Failed"), self.tr(str(payload))
                    )
                    return
                local_project = str(payload or "")
                if local_project:
                    self.current_project_path = local_project
                    self.append_training_log(
                        f"Remote training folder saved to: {local_project}"
                    )
                    QMessageBox.information(
                        self,
                        self.tr("Saved"),
                        self.tr(
                            f"Training folder saved to:\n{local_project}"
                        ),
                    )
        except RuntimeError:
            return

    def _export_meta_for_dialog(self) -> dict:
        """Always refresh project/subdir from Data/Config page."""
        cfg = self.get_current_config()
        model_path = (cfg.get("basic") or {}).get("model") or ""
        # Prefer cached stem from training start if model widget changed
        cached = dict(self.export_meta or {})
        imgsz = (cfg.get("train") or {}).get("imgsz")
        if imgsz is None:
            imgsz = self._current_imgsz()
        meta = resolve_export_meta(
            model_path=model_path or cached.get("model_stem", ""),
            imgsz=imgsz,
            num_classes=len(self.names or [])
            or int(cached.get("num_classes") or 0),
            dataset_root=self.dataset_root
            or self.remote_dataset_root
            or "",
            subdirs=self._selected_subdirs(),
        )
        if cached.get("model_stem") and (
            not model_path
            or os.path.basename(model_path).rsplit(".", 1)[0].lower()
            in ("best", "last")
        ):
            meta["model_stem"] = cached["model_stem"]
        meta["version"] = "v1"
        return meta

    def start_export(self):
        if not self.current_project_path and not self._remote_run_dir():
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("No training project available for export"),
            )
            return

        meta = self._export_meta_for_dialog()
        export_dialog = ExportFormatDialog(
            self,
            remote=(self.train_mode == "remote"),
            model_stem=meta.get("model_stem", "model"),
            imgsz=meta.get("imgsz", 640),
            num_classes=meta.get("num_classes", 0),
            project=meta.get("project", "project"),
            subdir=meta.get("subdir", "data"),
            version=meta.get("version", "v1"),
        )
        if export_dialog.exec() != QDialog.DialogCode.Accepted:
            return

        export_format = export_dialog.get_selected_format()
        weights_file = export_dialog.get_weights_file()
        onnx_name = export_dialog.get_onnx_filename()
        imgsz = int(meta.get("imgsz") or 640)

        if self.train_mode == "remote":
            if not self.ssh_client or not self.ssh_client.connected:
                QMessageBox.warning(
                    self,
                    self.tr("Export Error"),
                    self.tr("Remote SSH is not connected; reconnect to export."),
                )
                return
            remote_run = self._remote_run_dir()
            self.append_training_log(
                f"Remote ONNX export: {weights_file} → weights/{onnx_name}"
            )

            def _job():
                return remote_export_onnx(
                    self.ssh_client,
                    remote_run,
                    weights_file=weights_file,
                    output_name=onnx_name,
                    imgsz=imgsz,
                    remote_python=self._remote_python_for_exec(),
                )

            self._start_bg_task("export", _job)
            return

        # Local: export into run_dir/weights with optional rename
        local_project = self.current_project_path
        weights_path = os.path.join(local_project, "weights", weights_file)
        if not os.path.isfile(weights_path):
            QMessageBox.warning(
                self,
                self.tr("Model Not Found"),
                self.tr(f"Model weights not found at: {weights_path}"),
            )
            return

        output_name = onnx_name if export_format == "onnx" else ""
        py_exe = self._selected_interpreter_python()
        if py_exe:
            self.append_training_log(
                self.tr("Export Python: %s") % py_exe
            )
        success, message = self.export_manager.start_export(
            local_project,
            export_format,
            weights_file=weights_file,
            output_name=output_name,
            imgsz=imgsz,
            python_executable=py_exe,
        )
        if not success:
            QMessageBox.critical(self, self.tr("Export Error"), message)
            self.append_training_log(f"Failed to start export: {message}")

    def start_download_run_folder(self):
        """Download the full remote training run folder to a local path."""
        if self.train_mode != "remote":
            QMessageBox.information(
                self,
                self.tr("Download Folder"),
                self.tr(
                    "Local training results are already on disk. "
                    "Use Open Directory to browse them."
                ),
            )
            return
        if not self.ssh_client or not self.ssh_client.connected:
            QMessageBox.warning(
                self,
                self.tr("Download Failed"),
                self.tr("Remote SSH is not connected; reconnect to download."),
            )
            return
        if not self._remote_run_dir():
            QMessageBox.warning(
                self,
                self.tr("Download Failed"),
                self.tr("No remote training folder available."),
            )
            return

        prepared = self._prepare_download_destination(force=True)
        if prepared is None:
            return
        remote_run, local_run, err = prepared
        if err:
            QMessageBox.warning(self, self.tr("Save Failed"), self.tr(err))
            return
        if not remote_run and local_run:
            self.current_project_path = local_run
            QMessageBox.information(
                self,
                self.tr("Saved"),
                self.tr(f"Training folder saved to:\n{local_run}"),
            )
            return

        self.append_training_log(
            f"Queuing download:\n  {remote_run}\n-> {local_run}"
        )
        self._start_bg_task(
            "download",
            lambda: self._transfer_remote_run_folder(remote_run, local_run),
        )

    def _resolve_predict_data_yaml(self) -> str:
        data = (self.data_yaml_path or "").strip()
        if not data and "data" in self.config_widgets:
            data = (self.config_widgets["data"].text() or "").strip()
        if not data:
            job = load_remote_job() or {}
            data = (job.get("data") or "").strip()
        return data.replace("\\", "/") if data else ""

    def _local_weight_choices(self, run_dir: str) -> list:
        weights_dir = os.path.join(run_dir, "weights")
        choices = []
        for name in ("best.pt", "last.pt"):
            if os.path.isfile(os.path.join(weights_dir, name)):
                choices.append(name)
        if os.path.isdir(weights_dir):
            for name in sorted(os.listdir(weights_dir)):
                if name.lower().endswith(".onnx") and name not in choices:
                    choices.append(name)
        return choices or ["best.pt", "last.pt"]

    def _run_local_predict(
        self,
        run_dir: str,
        data_yaml: str,
        weights_file: str,
        imgsz: int,
        *,
        python_executable: str = "",
    ) -> tuple:
        """Run local YOLO predict with UTF-8-safe reading of all.txt.

        Must pass the ``all.txt`` path (not a Python list): Ultralytics converts
        path lists via ``autocast_list`` → PIL and then saves ``image0.txt``.
        Ultralytics reads list files with ``Path.read_text()`` using the locale
        encoding (GBK on Chinese Windows), so we temporarily force UTF-8.
        When ``python_executable`` is set (Interpreter mode), run in that env.
        """
        import builtins
        import sys
        import tempfile
        from pathlib import Path

        dataset_root = os.path.dirname(data_yaml)
        all_txt = os.path.join(dataset_root, "all.txt")
        model_path = os.path.join(run_dir, "weights", weights_file)
        if not os.path.isfile(model_path):
            return False, f"Model weights not found: {model_path}"
        if not os.path.isfile(all_txt):
            return False, f"all.txt not found: {all_txt}"

        # Validate all.txt is readable as UTF-8 before Ultralytics touches it.
        try:
            Path(all_txt).read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            return False, f"all.txt is not valid UTF-8: {e}"

        labels_dir = os.path.join(run_dir, "predict", "labels")
        all_txt_key = os.path.normcase(os.path.abspath(all_txt))
        py_exe = (python_executable or "").strip()
        if py_exe:
            if not os.path.isfile(py_exe):
                return False, f"Training Python not found: {py_exe}"
            # Import YOLO first, then patch only reads of all.txt.
            # Global Path.read_text monkeypatch before import breaks torch on
            # Python < 3.10 (newline kwarg) via importlib.metadata.
            script = f"""# -*- coding: utf-8 -*-
import io
import os
import sys
from pathlib import Path
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from ultralytics import YOLO
import builtins
_ALL_TXT = {all_txt_key!r}
_orig_open = builtins.open
_orig_read_text = Path.read_text
_HAS_NEWLINE = sys.version_info >= (3, 10)

def _is_all_txt(path):
    try:
        return os.path.normcase(os.path.abspath(str(path))) == _ALL_TXT
    except Exception:
        return False

def _open_utf8(file, mode="r", buffering=-1, encoding=None, errors=None,
               newline=None, closefd=True, opener=None):
    if "b" not in str(mode) and encoding is None and _is_all_txt(file):
        encoding = "utf-8"
        if errors is None:
            errors = "replace"
    return _orig_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

def _read_text_utf8(self, encoding=None, errors=None, newline=None):
    if encoding is None and _is_all_txt(self):
        encoding = "utf-8"
        if errors is None:
            errors = "replace"
    kwargs = {{}}
    if encoding is not None:
        kwargs["encoding"] = encoding
    if errors is not None:
        kwargs["errors"] = errors
    if newline is not None and _HAS_NEWLINE:
        kwargs["newline"] = newline
    return _orig_read_text(self, **kwargs)

builtins.open = _open_utf8
Path.read_text = _read_text_utf8
try:
    model = YOLO({model_path!r})
    model.predict(
        source={all_txt!r},
        save_txt=True,
        save_conf=False,
        rect=False,
        save=False,
        project={run_dir!r},
        name="predict",
        exist_ok=True,
        imgsz={int(imgsz)},
        verbose=True,
    )
finally:
    builtins.open = _orig_open
    Path.read_text = _orig_read_text
print("PREDICT_OK:" + {labels_dir!r}, flush=True)
"""
            fd, script_path = tempfile.mkstemp(
                suffix="_predict.py", prefix="ual_"
            )
            os.close(fd)
            try:
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(script)
                proc = subprocess.run(
                    [py_exe, script_path],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                out = (proc.stdout or "") + (proc.stderr or "")
                if proc.returncode != 0:
                    return False, out.strip() or f"predict exited {proc.returncode}"
                return True, labels_dir
            finally:
                try:
                    os.remove(script_path)
                except OSError:
                    pass

        try:
            from ultralytics import YOLO
        except Exception as e:
            return False, f"Failed to import ultralytics: {e}"

        _orig_open = builtins.open
        _orig_read_text = Path.read_text
        _has_newline = sys.version_info >= (3, 10)

        def _is_all_txt(path):
            try:
                return (
                    os.path.normcase(os.path.abspath(str(path))) == all_txt_key
                )
            except Exception:
                return False

        def _open_utf8(
            file,
            mode="r",
            buffering=-1,
            encoding=None,
            errors=None,
            newline=None,
            closefd=True,
            opener=None,
        ):
            if (
                "b" not in str(mode)
                and encoding is None
                and _is_all_txt(file)
            ):
                encoding = "utf-8"
                if errors is None:
                    errors = "replace"
            return _orig_open(
                file,
                mode,
                buffering,
                encoding,
                errors,
                newline,
                closefd,
                opener,
            )

        def _read_text_utf8(self, encoding=None, errors=None, newline=None):
            if encoding is None and _is_all_txt(self):
                encoding = "utf-8"
                if errors is None:
                    errors = "replace"
            kwargs = {}
            if encoding is not None:
                kwargs["encoding"] = encoding
            if errors is not None:
                kwargs["errors"] = errors
            if newline is not None and _has_newline:
                kwargs["newline"] = newline
            return _orig_read_text(self, **kwargs)

        try:
            builtins.open = _open_utf8
            Path.read_text = _read_text_utf8
            model = YOLO(model_path)
            model.predict(
                source=all_txt,
                save_txt=True,
                save_conf=False,
                rect=False,
                save=False,
                project=run_dir,
                name="predict",
                exist_ok=True,
                imgsz=int(imgsz),
                verbose=True,
            )
        except Exception as e:
            return False, str(e)
        finally:
            builtins.open = _orig_open
            Path.read_text = _orig_read_text
        return True, labels_dir

    def start_predict(self):
        run_dir = (
            self._remote_run_dir()
            if self.train_mode == "remote"
            else (self.current_project_path or "")
        )
        data_yaml = self._resolve_predict_data_yaml()
        if not run_dir:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("No training project available for inference."),
            )
            return
        if not data_yaml:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Dataset data.yaml path is unknown; cannot find all.txt."),
            )
            return

        imgsz = int((self.export_meta or {}).get("imgsz") or self._current_imgsz())
        if self.train_mode == "remote":
            dataset_root = os.path.dirname(data_yaml.replace("\\", "/"))
            all_txt = f"{dataset_root.rstrip('/')}/all.txt"
            labels_out = (
                f"{run_dir.rstrip('/').replace(chr(92), '/')}/predict/labels"
            )
        else:
            all_txt = os.path.join(os.path.dirname(data_yaml), "all.txt")
            labels_out = os.path.join(run_dir, "predict", "labels")

        if self.train_mode == "remote":
            if not self.ssh_client or not self.ssh_client.connected:
                QMessageBox.warning(
                    self,
                    self.tr("Error"),
                    self.tr("Remote SSH is not connected; reconnect to infer."),
                )
                return
            try:
                choices = list_remote_weight_choices(self.ssh_client, run_dir)
            except Exception:
                choices = ["best.pt", "last.pt"]
        else:
            choices = self._local_weight_choices(run_dir)

        dlg = RemotePredictDialog(
            self,
            weight_choices=choices,
            source_all_txt=all_txt,
            output_labels_dir=labels_out,
            imgsz=imgsz,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        weights_file = dlg.get_weights_file()
        self.append_training_log(
            f"Starting predict: model={weights_file}, source={all_txt}"
        )

        run_dir_arg = run_dir
        data_yaml_arg = data_yaml
        weights_arg = weights_file
        imgsz_arg = imgsz
        remote = self.train_mode == "remote"
        py_exe = self._selected_interpreter_python()

        def _job():
            if remote:
                return remote_predict_yolo(
                    self.ssh_client,
                    run_dir_arg,
                    data_yaml_arg,
                    weights_file=weights_arg,
                    imgsz=imgsz_arg,
                    remote_python=self._remote_python_for_exec(),
                )
            return self._run_local_predict(
                run_dir_arg,
                data_yaml_arg,
                weights_arg,
                imgsz_arg,
                python_executable=py_exe,
            )

        self._start_bg_task("predict", _job)

    def _batch_infer_labels_dir(self, run_dir: str) -> str:
        if self.train_mode == "remote" or str(run_dir).replace(
            "\\", "/"
        ).startswith("/"):
            return (
                f"{run_dir.rstrip('/').replace(chr(92), '/')}"
                f"/batch_infer/labels"
            )
        return os.path.join(run_dir, "batch_infer", "labels")

    def _run_local_batch_predict(
        self,
        run_dir: str,
        image_folder: str,
        weights_file: str,
        imgsz: int,
        *,
        python_executable: str = "",
    ) -> tuple:
        """Predict all images in a folder → ``{run_dir}/batch_infer/labels``."""
        import tempfile

        image_folder = os.path.abspath(image_folder or "")
        if not os.path.isdir(image_folder):
            return False, f"Image folder not found: {image_folder}"
        n_img = count_local_images(image_folder)
        if n_img <= 0:
            return False, f"No images found under: {image_folder}"

        model_path = weights_file
        if not os.path.isabs(model_path):
            model_path = os.path.join(run_dir, "weights", weights_file)
        if not os.path.isfile(model_path):
            return False, f"Model weights not found: {model_path}"

        labels_dir = os.path.join(run_dir, "batch_infer", "labels")
        # Clear previous batch results so stems don't mix runs
        batch_root = os.path.join(run_dir, "batch_infer")
        if os.path.isdir(batch_root):
            try:
                shutil.rmtree(batch_root, ignore_errors=True)
            except Exception:
                pass

        py_exe = (python_executable or "").strip()
        if py_exe:
            if not os.path.isfile(py_exe):
                return False, f"Training Python not found: {py_exe}"
            script = f"""# -*- coding: utf-8 -*-
import io
import os
import sys
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from ultralytics import YOLO
model = YOLO({model_path!r})
model.predict(
    source={image_folder!r},
    save_txt=True,
    save_conf=False,
    rect=False,
    save=False,
    project={run_dir!r},
    name="batch_infer",
    exist_ok=True,
    imgsz={int(imgsz)},
    verbose=True,
)
labels_dir = {labels_dir!r}
n = 0
if os.path.isdir(labels_dir):
    n = sum(1 for f in os.listdir(labels_dir) if f.lower().endswith(".txt"))
print("BATCH_PREDICT_OK:" + labels_dir + " (%d txt)" % n, flush=True)
"""
            fd, script_path = tempfile.mkstemp(
                suffix="_batch_predict.py", prefix="ual_"
            )
            os.close(fd)
            try:
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(script)
                proc = subprocess.run(
                    [py_exe, script_path],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                out = (proc.stdout or "") + (proc.stderr or "")
                if proc.returncode != 0:
                    return (
                        False,
                        out.strip()
                        or f"batch predict exited {proc.returncode}",
                    )
                return True, labels_dir
            finally:
                try:
                    os.remove(script_path)
                except OSError:
                    pass

        try:
            from ultralytics import YOLO
        except Exception as e:
            return False, f"Failed to import ultralytics: {e}"

        try:
            model = YOLO(model_path)
            model.predict(
                source=image_folder,
                save_txt=True,
                save_conf=False,
                rect=False,
                save=False,
                project=run_dir,
                name="batch_infer",
                exist_ok=True,
                imgsz=int(imgsz),
                verbose=True,
            )
        except Exception as e:
            return False, str(e)
        return True, labels_dir

    def start_batch_predict(self):
        run_dir = (
            self._remote_run_dir()
            if self.train_mode == "remote"
            else (self.current_project_path or "")
        )
        if not run_dir:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("No training project available for batch inference."),
            )
            return

        if self.train_mode == "remote":
            if not self.ssh_client or not self.ssh_client.connected:
                QMessageBox.warning(
                    self,
                    self.tr("Error"),
                    self.tr(
                        "Remote SSH is not connected; reconnect to batch infer."
                    ),
                )
                return
            try:
                choices = list_remote_weight_choices(self.ssh_client, run_dir)
            except Exception:
                choices = ["best.pt", "last.pt"]
        else:
            choices = self._local_weight_choices(run_dir)

        imgsz = int(
            (self.export_meta or {}).get("imgsz") or self._current_imgsz()
        )
        labels_out = self._batch_infer_labels_dir(run_dir)
        dlg = BatchInferDialog(
            self,
            weight_choices=choices,
            output_labels_dir=labels_out,
            imgsz=imgsz,
            remote=(self.train_mode == "remote"),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        image_folder = dlg.get_image_folder()
        weights_file = dlg.get_weights_file()
        n_img = count_local_images(image_folder)
        self.append_training_log(
            f"Starting batch predict: model={weights_file}, "
            f"folder={image_folder} ({n_img} images)"
        )

        run_dir_arg = run_dir
        folder_arg = image_folder
        weights_arg = weights_file
        imgsz_arg = imgsz
        remote = self.train_mode == "remote"
        py_exe = self._selected_interpreter_python()

        def _job():
            if remote:
                return remote_batch_predict_yolo(
                    self.ssh_client,
                    run_dir_arg,
                    folder_arg,
                    weights_file=weights_arg,
                    imgsz=imgsz_arg,
                    remote_python=self._remote_python_for_exec(),
                )
            return self._run_local_batch_predict(
                run_dir_arg,
                folder_arg,
                weights_arg,
                imgsz_arg,
                python_executable=py_exe,
            )

        self._start_bg_task("batch_predict", _job)

    def start_download_batch_results(self):
        """Copy/download the batch_infer/labels folder to a user-chosen path."""
        labels = (self._last_batch_infer_labels or "").strip()
        if not labels:
            # Fallback: derive from current run dir
            run_dir = (
                self._remote_run_dir()
                if self.train_mode == "remote"
                else (self.current_project_path or "")
            )
            if run_dir:
                labels = self._batch_infer_labels_dir(run_dir)

        if not labels:
            QMessageBox.information(
                self,
                self.tr("Download Batch Results"),
                self.tr(
                    "No batch inference results yet. "
                    "Run “Batch Infer” first."
                ),
            )
            return

        parent = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select folder to save batch labels"),
            get_default_project_dir(),
        )
        if not parent:
            return

        dest = os.path.join(parent, "batch_infer_labels")
        if os.path.exists(dest):
            reply = QMessageBox.question(
                self,
                self.tr("Directory Exists"),
                self.tr(
                    f"Folder already exists:\n{dest}\n\nOverwrite?"
                ),
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            try:
                shutil.rmtree(dest)
            except Exception as e:
                QMessageBox.warning(
                    self, self.tr("Download Failed"), str(e)
                )
                return

        remote = self.train_mode == "remote" and labels.replace(
            "\\", "/"
        ).startswith("/")
        labels_arg = labels
        dest_arg = dest

        if remote:
            if not self.ssh_client or not self.ssh_client.connected:
                QMessageBox.warning(
                    self,
                    self.tr("Download Failed"),
                    self.tr(
                        "Remote SSH is not connected; reconnect to download."
                    ),
                )
                return

            def _job():
                try:
                    self.ssh_client.download_directory(
                        labels_arg, dest_arg
                    )
                    return True, dest_arg
                except Exception as e:
                    return False, str(e)

            self.append_training_log(
                f"Downloading batch labels:\n  {labels_arg}\n-> {dest_arg}"
            )
            self._start_bg_task("download_batch", _job)
            return

        # Local / Interpreter: copy the labels folder
        if not os.path.isdir(labels_arg):
            QMessageBox.warning(
                self,
                self.tr("Download Failed"),
                self.tr(f"Labels folder not found:\n{labels_arg}"),
            )
            return

        def _job_local():
            try:
                shutil.copytree(labels_arg, dest_arg)
                return True, dest_arg
            except Exception as e:
                return False, str(e)

        self.append_training_log(
            f"Copying batch labels:\n  {labels_arg}\n-> {dest_arg}"
        )
        self._start_bg_task("download_batch", _job_local)

    def reset_train_tab(self):
        self.training_status = "idle"
        self.current_project_path = None
        self.current_epochs = 0
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0/0")
        self.update_training_status_display()

        if hasattr(self, "log_display"):
            self.log_display.clear()
            self._log_follow_tail = True

        for i, image_label in enumerate(self.image_labels):
            image_label.clear()
            image_label.setText(self.tr("No image"))
            image_label.setToolTip("")
            self.image_paths[i] = None

        self.previous_button.setVisible(True)
        self.start_training_button.setVisible(True)
        self._set_post_train_buttons(visible=False)
        self.stop_training_button.setVisible(False)
        self._last_batch_infer_labels = ""
        if hasattr(self, "download_batch_button"):
            self.download_batch_button.setVisible(False)
