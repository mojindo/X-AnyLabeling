"""Local / remote Ultralytics training process management."""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from io import StringIO
from typing import Dict, Optional, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

from .config import get_settings_config_path
from .custom_training import (
    bind_command_to_python,
    build_remote_custom_launch,
    build_ultralytics_train_cli,
    interpreter_path_env,
    wrap_local_shell_command,
)
from .logging_bridge import emit_train_log


class TrainingEventRedirector(QObject):
    """Thread-safe training event redirector"""

    training_event_signal = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__()

    def emit_training_event(self, event_type, data):
        self.training_event_signal.emit(event_type, data)


class TrainingLogRedirector(QObject):
    """Thread-safe training log redirector"""

    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.log_stream = StringIO()

    def write(self, text):
        if text.strip():
            self.log_signal.emit(text)

    def flush(self):
        pass


def _kill_process_tree(pid: Optional[int]) -> None:
    """Stop Ultralytics training and its worker children (Windows needs /T)."""
    if pid is None:
        return
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return
    if pid_int <= 0:
        return
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(
            ["taskkill", "/PID", str(pid_int), "/T", "/F"],
            check=False,
            capture_output=True,
            creationflags=flags,
        )
        return
    try:
        pgid = os.getpgid(pid_int)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid_int, signal.SIGTERM)
        except OSError:
            return
    time.sleep(0.4)
    try:
        pgid = os.getpgid(pid_int)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid_int, signal.SIGKILL)
        except OSError:
            pass


def _purge_local_run_dir(run_dir: str) -> None:
    """Remove a previous Ultralytics run folder so exist_ok does not continue."""
    rd = (run_dir or "").strip()
    if not rd:
        return
    for _ in range(4):
        if os.path.isdir(rd):
            shutil.rmtree(rd, ignore_errors=True)
        if not os.path.isdir(rd):
            break
        time.sleep(0.2)


class TrainingManager:
    def __init__(self):
        self.training_process = None
        self.is_training = False
        self.callbacks = []
        self.total_epochs = 100
        self.stop_event = threading.Event()
        self._mode = "local"  # local | remote
        self._ssh_client = None
        self._remote_pid = None
        self._remote_log_path = None
        self._tb_process = None
        self.tensorboard_url = ""
        self._log_offset = 0
        self._detach_without_kill = False
        self._local_log_path: Optional[str] = None
        self._local_project = ""
        self._local_name = ""
        self._local_run_dir = ""
        self._local_python_exe = ""
        self._local_pid: Optional[int] = None
        self._custom_job = False
        self._local_log_handle = None
        self._local_monitor_started_at = 0.0
        self._local_run_started_at = 0.0

    def notify_callbacks(self, event_type: str, data: dict):
        for callback in self.callbacks:
            try:
                callback(event_type, data)
            except Exception:
                pass

    def start_training(
        self,
        train_args: Dict,
        *,
        mode: str = "local",
        ssh_client=None,
        python_executable: str = "",
        remote_python: str = "",
    ) -> Tuple[bool, str]:
        if self.is_training:
            return False, "Training is already in progress"

        mode = (mode or "local").lower()
        if mode == "remote":
            return self._start_remote(
                train_args, ssh_client, remote_python=remote_python or ""
            )
        # local | interpreter — same filesystem path, optional external Python
        return self._start_local(
            train_args, python_executable=python_executable or ""
        )

    def _start_local(
        self, train_args: Dict, *, python_executable: str = ""
    ) -> Tuple[bool, str]:
        try:
            import sys

            args = dict(train_args)
            self._custom_job = False
            self._mode = "local"
            self.total_epochs = args.get("epochs", 100)
            self.stop_event.clear()
            self._detach_without_kill = False
            self.tensorboard_url = ""
            task = (args.pop("_task", None) or "detect").strip().lower()
            from anylabeling.services.auto_training.mytrain.remote.job_state import (
                load_local_job,
                probe_local_job,
            )

            existing = load_local_job()
            st, sm = probe_local_job(existing)
            if st == "alive":
                return (
                    False,
                    "A local training job is still running. Reattach to watch it.",
                )
            from anylabeling.services.auto_training.mytrain.remote.job_state import (
                clear_local_job,
            )

            if existing:
                clear_local_job()

            # exist_ok reuses project/name; drop leftover results/images so the
            # progress bar and previews are not stuck on the previous run.
            project_dir = args.get("project", "") or ""
            run_name = args.get("name", "exp")
            run_dir = os.path.join(project_dir, run_name) if project_dir else ""
            if run_dir and args.get("exist_ok", True):
                _purge_local_run_dir(run_dir)

            args["resume"] = False
            command = build_ultralytics_train_cli(args, task=task)
            if not command.strip():
                return False, "Empty training command"

            py_exe = (python_executable or "").strip() or sys.executable
            if not os.path.isfile(py_exe):
                return False, f"Training Python not found: {py_exe}"

            resolved = bind_command_to_python(command, py_exe)
            os.makedirs(project_dir or ".", exist_ok=True)

            self._local_log_path = os.path.join(project_dir, "train_run.log")
            self._close_local_log_handle()
            try:
                with open(self._local_log_path, "w", encoding="utf-8") as _lf:
                    _lf.write(
                        f"=== Training {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    )
                    _lf.write(f"Command: {command}\n")
            except OSError:
                pass
            self._local_project = project_dir
            self._local_name = run_name
            self._local_run_dir = run_dir or os.path.join(project_dir, run_name)
            self._local_python_exe = py_exe
            self._log_offset = 0
            self._local_run_started_at = time.time()

            log_handle = open(
                self._local_log_path, "a", encoding="utf-8", buffering=1
            )
            self._local_log_handle = log_handle
            popen_kwargs: Dict = {
                "stdout": log_handle,
                "stderr": subprocess.STDOUT,
            }
            using_external_py = bool(
                (python_executable or "").strip()
                and os.path.isfile((python_executable or "").strip())
            )
            if using_external_py:
                popen_kwargs["env"] = interpreter_path_env(py_exe)
                launcher = (
                    "import os, subprocess, sys; "
                    f"rc = subprocess.run({resolved!r}, shell=True, env=os.environ); "
                    "sys.exit(rc.returncode if rc.returncode is not None else 1)"
                )
                popen_cmd = [py_exe, "-c", launcher]
                if os.name != "nt":
                    popen_kwargs["preexec_fn"] = os.setsid
                    popen_kwargs["close_fds"] = True
                else:
                    popen_kwargs["creationflags"] = (
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        | getattr(
                            subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000
                        )
                    )
            elif os.name == "nt":
                popen_kwargs["shell"] = True
                popen_cmd = resolved
                popen_kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
                )
            else:
                popen_cmd = wrap_local_shell_command(resolved)
                popen_kwargs["preexec_fn"] = os.setsid
                popen_kwargs["close_fds"] = True

            self.training_process = subprocess.Popen(popen_cmd, **popen_kwargs)
            self._local_pid = self.training_process.pid
            self._persist_local_job(
                {
                    "status": "running",
                    "data": args.get("data", ""),
                    "model": args.get("model", ""),
                    "resolved_command": command[:2000],
                }
            )

            def save_settings_config():
                save_path = os.path.join(args["project"], args["name"])
                save_file = os.path.join(save_path, "settings.json")
                while not os.path.exists(save_path):
                    if self.stop_event.is_set():
                        return
                    time.sleep(1)
                try:
                    shutil.copy2(get_settings_config_path(), save_file)
                except Exception:
                    pass

            self._maybe_start_local_tb(project_dir)

            threading.Thread(
                target=self._monitor_local,
                kwargs={"from_attach": False, "script_path": ""},
                daemon=True,
            ).start()
            threading.Thread(target=save_settings_config, daemon=True).start()
            return True, "Training started successfully"

        except ImportError:
            return (
                False,
                "Ultralytics is not installed. Please install it with: pip install ultralytics",
            )
        except Exception as e:
            return False, f"Failed to start training: {str(e)}"

    def _maybe_start_local_tb(self, logdir: str) -> None:
        if self.tensorboard_url or self._custom_job:
            return
        try:
            from anylabeling.services.auto_training.mytrain.remote.remote_trainer import (
                start_local_tensorboard,
            )

            # Only ensure project root exists — never pre-create the run name
            # subdirectory (Ultralytics would treat it as occupied → name-2).
            os.makedirs(logdir, exist_ok=True)
            proc, port, url = start_local_tensorboard(
                logdir, python_executable=self._local_python_exe or ""
            )
            self._tb_process = proc
            self.tensorboard_url = url
            self.notify_callbacks(
                "training_log",
                {"message": f"TensorBoard: {url}"},
            )
        except Exception as e:
            emit_train_log(self, f"TensorBoard not started: {e}")

    def _close_local_log_handle(self) -> None:
        handle = self._local_log_handle
        if handle is None:
            return
        try:
            handle.flush()
            handle.close()
        except Exception:
            pass
        self._local_log_handle = None

    def _persist_local_job(self, extra: Optional[Dict] = None) -> None:
        from anylabeling.services.auto_training.mytrain.remote.job_state import (
            save_local_job,
        )

        pid = self._local_pid
        if self.training_process is not None:
            pid = self.training_process.pid
        state = {
            "mode": "local",
            "pid": pid,
            "log_path": self._local_log_path,
            "run_dir": self._local_run_dir,
            "project": self._local_project,
            "name": self._local_name,
            "epochs": self.total_epochs,
            "python": self._local_python_exe,
            "tensorboard_url": self.tensorboard_url,
            "log_offset": self._log_offset,
            "status": "running",
        }
        if extra:
            state.update(extra)
        save_local_job(state)

    def _emit_train_log_snapshot(self, log_path: str) -> None:
        from .custom_training import format_train_log_terminal, latest_log_session_text

        if not log_path or not os.path.isfile(log_path):
            return
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            return
        body = latest_log_session_text(content)
        text = format_train_log_terminal(body)
        if not text.strip():
            return
        self.notify_callbacks("training_log_snapshot", {"text": text})

    def _read_local_log_delta(self, log_path: str, byte_offset: int) -> int:
        if not log_path or not os.path.isfile(log_path):
            return max(0, int(byte_offset or 0))
        offset = max(0, int(byte_offset or 0))
        try:
            size = os.path.getsize(log_path)
        except OSError:
            return offset
        if size < offset:
            offset = 0
        chunk_limit = 512_000
        start_offset = offset
        with open(log_path, "rb") as f:
            while size > offset:
                f.seek(offset)
                data = f.read(chunk_limit)
                if not data:
                    break
                offset += len(data)
                if len(data) < chunk_limit:
                    break
                try:
                    size = os.path.getsize(log_path)
                except OSError:
                    break
        if offset != start_offset:
            self._log_offset = offset
            self._persist_local_job()
            self._emit_train_log_snapshot(log_path)
        return offset

    def _notify_local_epoch_from_csv(self) -> None:
        run_dir = self._local_run_dir or ""
        if not run_dir:
            return
        csv_path = os.path.join(run_dir, "results.csv")
        if not os.path.isfile(csv_path):
            return
        started = float(getattr(self, "_local_run_started_at", 0) or 0)
        if started > 0:
            try:
                if os.path.getmtime(csv_path) < started - 0.5:
                    return
            except OSError:
                return
        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(500_000)
        except OSError:
            return
        cur = self._count_epochs_from_results_csv(content)
        if cur <= 0:
            return
        self.notify_callbacks(
            "training_progress",
            {"current_epochs": cur, "total_epochs": self.total_epochs},
        )

    def _monitor_local(self, *, from_attach: bool = False, script_path: str = ""):
        from anylabeling.services.auto_training.mytrain.remote.job_state import (
            clear_local_job,
            is_local_pid_alive,
        )

        log_path = self._local_log_path or ""
        last_size = int(self._log_offset or 0)

        try:
            self.is_training = True
            self._local_monitor_started_at = time.time()
            self.notify_callbacks(
                "training_started",
                {
                    "total_epochs": self.total_epochs,
                    "python": self._local_python_exe,
                    "reattached": from_attach,
                    "custom": self._custom_job,
                },
            )
            if from_attach:
                emit_train_log(
                    self,
                    f"Reattached to local PID {self._local_pid}",
                )
            if log_path:
                self._emit_train_log_snapshot(log_path)

            while not self.stop_event.is_set():
                last_size = self._read_local_log_delta(log_path, last_size)
                self._notify_local_epoch_from_csv()

                proc = self.training_process
                if proc is not None:
                    if proc.poll() is not None:
                        break
                elif self._local_pid and not is_local_pid_alive(self._local_pid):
                    break
                time.sleep(1.0)

            if self.stop_event.is_set():
                if self._detach_without_kill:
                    self.is_training = False
                    self._persist_local_job({"status": "detached"})
                    self.notify_callbacks(
                        "training_detached",
                        {
                            "hint": (
                                "Training continues in the background. "
                                "Re-open MyTrain and click Reattach Job."
                            )
                        },
                    )
                    return
                proc = self.training_process
                root_pid = (
                    proc.pid if proc is not None and proc.poll() is None else None
                )
                if root_pid is None and self._local_pid:
                    from anylabeling.services.auto_training.mytrain.remote.job_state import (
                        is_local_pid_alive,
                    )

                    if is_local_pid_alive(self._local_pid):
                        root_pid = self._local_pid
                if root_pid is not None:
                    _kill_process_tree(root_pid)
                    if proc is not None:
                        try:
                            proc.wait(timeout=8)
                        except Exception:
                            pass
                self._close_local_log_handle()
                self.is_training = False
                self.training_process = None
                clear_local_job()
                self.notify_callbacks("training_stopped", {})
                if script_path:
                    try:
                        os.remove(script_path)
                    except OSError:
                        pass
                return

            self._read_local_log_delta(log_path, last_size)
            self._notify_local_epoch_from_csv()
            return_code = None
            if self.training_process is not None:
                return_code = self.training_process.poll()
            self.is_training = False
            self._close_local_log_handle()
            ran_for = time.time() - float(self._local_monitor_started_at or 0)
            log_bytes = 0
            if log_path and os.path.isfile(log_path):
                try:
                    log_bytes = os.path.getsize(log_path)
                except OSError:
                    log_bytes = 0
            suspect_quick = (
                self._custom_job
                and ran_for < 8.0
                and log_bytes < 180
                and return_code == 0
            )
            if return_code == 0 and not suspect_quick:
                clear_local_job()
                self._maybe_start_local_tb(self._local_project)
                self.notify_callbacks(
                    "training_completed",
                    {
                        "results": "Training completed successfully",
                        "tensorboard_url": self.tensorboard_url,
                    },
                )
            elif suspect_quick:
                clear_local_job()
                self.notify_callbacks(
                    "training_error",
                    {
                        "error": (
                            "Custom command exited immediately with no log output. "
                            "Check Execution directory, placeholders, and that "
                            "the command runs in a shell."
                        )
                    },
                )
            else:
                clear_local_job()
                self.notify_callbacks(
                    "training_error",
                    {
                        "error": f"Training process exited with code {return_code}"
                    },
                )
        except Exception as e:
            self.is_training = False
            self._persist_local_job({"status": "detached"})
            self.notify_callbacks(
                "training_detached",
                {
                    "error": str(e),
                    "hint": (
                        "Training may still be running. Re-open MyTrain "
                        "and click Reattach Job."
                    ),
                },
            )
        finally:
            if script_path:
                try:
                    os.remove(script_path)
                except OSError:
                    pass

    def attach_local_training(self) -> Tuple[bool, str]:
        if self.is_training:
            return False, "Training monitor is already attached"

        from anylabeling.services.auto_training.mytrain.remote.job_state import (
            clear_local_job,
            load_local_job,
            probe_local_job,
        )

        job = load_local_job()
        status, msg = probe_local_job(job)
        if status == "none":
            return False, "No saved local training job found"
        if status == "error":
            return False, msg
        if status == "finished":
            clear_local_job()
            return False, f"Local job already finished: {msg}"

        self._mode = "local"
        self.stop_event.clear()
        self._detach_without_kill = False
        self._local_pid = int(job.get("pid") or 0)
        self._local_log_path = job.get("log_path") or ""
        self._local_run_dir = job.get("run_dir") or ""
        self._local_project = job.get("project") or ""
        self._local_name = job.get("name") or ""
        self._local_python_exe = job.get("python") or ""
        self.total_epochs = int(job.get("epochs") or 100)
        self.tensorboard_url = job.get("tensorboard_url") or ""
        self._log_offset = int(job.get("log_offset") or 0)
        self.training_process = None
        self._custom_job = bool(job.get("custom"))

        self._persist_local_job({"status": "running"})
        threading.Thread(
            target=self._monitor_local,
            kwargs={"from_attach": True},
            daemon=True,
        ).start()
        return True, f"Reattached to local PID {self._local_pid}"

    def detach_for_app_exit(self) -> None:
        """Leave training subprocess / remote job running when the app exits."""
        from anylabeling.services.auto_training.mytrain.remote.job_state import (
            load_local_job,
            load_remote_job,
            probe_local_job,
            save_local_job,
            save_remote_job,
        )

        if self.is_training:
            self._detach_without_kill = True
            self.stop_event.set()
            if self._mode == "remote":
                self._persist_remote_job({"status": "detached"})
            elif self._mode == "local":
                self._persist_local_job({"status": "detached"})
            return

        local = load_local_job()
        if local and probe_local_job(local)[0] == "alive":
            local["status"] = "detached"
            save_local_job(local)
        remote = load_remote_job()
        if remote and remote.get("status") != "finished":
            remote["status"] = "detached"
            save_remote_job(remote)

    def _persist_remote_job(self, extra: Optional[Dict] = None) -> None:
        from anylabeling.services.auto_training.mytrain.remote.job_state import (
            save_remote_job,
        )

        client = self._ssh_client
        if client is None:
            return
        state = {
            "host": getattr(client, "host", ""),
            "port": getattr(client, "port", 22),
            "username": getattr(client, "username", ""),
            "pid": self._remote_pid,
            "log_path": self._remote_log_path,
            "run_dir": getattr(self, "_remote_run_dir", ""),
            "project": getattr(self, "_remote_project", ""),
            "name": getattr(self, "_remote_name", ""),
            "epochs": self.total_epochs,
            "tensorboard_url": self.tensorboard_url,
            "log_offset": self._log_offset,
            "status": "running",
        }
        if extra:
            state.update(extra)
        save_remote_job(state)

    @staticmethod
    def _count_epochs_from_results_csv(content: str) -> int:
        rows = [ln for ln in content.splitlines() if ln.strip()]
        if len(rows) <= 1:
            return 0
        return len(rows) - 1

    def _clear_local_remote_preview_cache(self, run_name: str) -> None:
        cache_root = os.path.join(
            tempfile.gettempdir(),
            "xanylabeling_remote_preview",
            run_name or "exp",
        )
        try:
            shutil.rmtree(cache_root, ignore_errors=True)
        except Exception:
            pass

    def _reset_remote_run_artifacts(
        self, ssh_client, run_dir: str, remote_log: str
    ) -> None:
        """Remove previous run outputs so a same-Name retrain starts clean."""
        run_name = getattr(self, "_remote_name", "") or "exp"
        self._clear_local_remote_preview_cache(run_name)
        try:
            if run_dir:
                ssh_client.exec(
                    f"rm -rf {shlex.quote(run_dir)} 2>/dev/null || true"
                )
            if remote_log:
                ssh_client.exec(
                    f": > {shlex.quote(remote_log)} 2>/dev/null || "
                    f"true > {shlex.quote(remote_log)} 2>/dev/null || true"
                )
        except Exception:
            pass
        self.notify_callbacks(
            "training_log",
            {
                "message": (
                    f"Cleared previous run outputs at {run_dir} "
                    "(same Name retrain)"
                )
            },
        )

    def _notify_epoch_from_remote_csv(self, ssh_client) -> None:
        run_dir = getattr(self, "_remote_run_dir", "") or ""
        if not run_dir:
            return
        csv_path = f"{run_dir.rstrip('/')}/results.csv"
        try:
            content = ssh_client.read_text(csv_path, max_bytes=500_000)
        except Exception:
            return
        cur = self._count_epochs_from_results_csv(content)
        if cur <= 0:
            return
        self.notify_callbacks(
            "training_progress",
            {
                "current_epochs": cur,
                "total_epochs": self.total_epochs,
            },
        )

    def _notify_remote_train_batch_images(self, ssh_client) -> None:
        """Download remote train_batch*.jpg into a local cache for the UI."""
        run_dir = getattr(self, "_remote_run_dir", "") or ""
        if not run_dir:
            return
        try:
            names = ssh_client.listdir(run_dir)
        except Exception:
            return
        batch_names = sorted(
            n
            for n in names
            if n.startswith("train_batch") and n.lower().endswith(".jpg")
        )[:6]
        if not batch_names:
            return

        cache_root = os.path.join(
            tempfile.gettempdir(),
            "xanylabeling_remote_preview",
            getattr(self, "_remote_name", "") or "exp",
        )
        os.makedirs(cache_root, exist_ok=True)
        local_paths = []
        for name in batch_names:
            remote = f"{run_dir.rstrip('/')}/{name}"
            local = os.path.join(cache_root, name)
            try:
                ssh_client.download_file(remote, local)
                if os.path.isfile(local):
                    local_paths.append(local)
            except Exception:
                continue
        if local_paths:
            self.notify_callbacks(
                "training_images",
                {"paths": local_paths},
            )

    def _emit_remote_log_delta(
        self, ssh_client, remote_log: str, byte_offset: int
    ) -> int:
        """Read new log bytes from ``byte_offset`` and emit training_log lines.

        Returns the updated byte offset. Resets to 0 if the remote file was
        truncated/rotated (size smaller than the stored offset).
        """
        if not remote_log:
            return int(byte_offset or 0)

        offset = max(0, int(byte_offset or 0))
        size = ssh_client.file_size(remote_log)
        if size < 0:
            return offset
        if size < offset:
            # Log rewritten for a new run — start over
            offset = 0

        # Pull in chunks so a huge backlog after reattach does not stall UI
        chunk_limit = 512_000
        start_offset = offset
        while size > offset:
            data = ssh_client.read_bytes_from(
                remote_log, offset=offset, max_bytes=chunk_limit
            )
            if not data:
                break
            offset += len(data)
            if len(data) < chunk_limit:
                break
            size = ssh_client.file_size(remote_log)
            if size < 0:
                break

        if offset != start_offset or offset != int(self._log_offset or 0):
            self._log_offset = offset
            self._persist_remote_job()
            self._emit_remote_train_log_snapshot(ssh_client, remote_log)
        return offset

    def _emit_remote_train_log_snapshot(self, ssh_client, remote_log: str) -> None:
        from .custom_training import format_train_log_terminal, latest_log_session_text

        if not remote_log:
            return
        size = ssh_client.file_size(remote_log)
        if size <= 0:
            return
        cap = 2_000_000
        data = ssh_client.read_bytes_from(
            remote_log, offset=0, max_bytes=min(size, cap)
        )
        if not data:
            return
        body = latest_log_session_text(data.decode("utf-8", errors="replace"))
        text = format_train_log_terminal(body)
        if text.strip():
            self.notify_callbacks("training_log_snapshot", {"text": text})

    def _monitor_remote(self, ssh_client, *, from_attach: bool = False):
        from anylabeling.services.auto_training.mytrain.remote.job_state import (
            clear_remote_job,
            is_remote_pid_alive,
        )
        from anylabeling.services.auto_training.mytrain.remote.remote_trainer import (
            fetch_remote_log_tail,
        )
        from anylabeling.services.auto_training.mytrain.remote.ssh_client import SSHError

        remote_log = self._remote_log_path
        last_size = int(self._log_offset or 0)

        try:
            self.is_training = True
            self.notify_callbacks(
                "training_started",
                {
                    "total_epochs": self.total_epochs,
                    "tensorboard_url": self.tensorboard_url,
                    "reattached": from_attach,
                },
            )
            if from_attach:
                self.notify_callbacks(
                    "training_log",
                    {
                        "message": self._tr_attach_msg(
                            f"Reattached to remote PID {self._remote_pid}"
                        )
                    },
                )
            if self.tensorboard_url:
                self.notify_callbacks(
                    "training_log",
                    {"message": f"TensorBoard: {self.tensorboard_url}"},
                )

            consecutive_net_errors = 0
            csv_poll_counter = 0
            while not self.stop_event.is_set():
                try:
                    alive_ok = is_remote_pid_alive(ssh_client, self._remote_pid)
                    consecutive_net_errors = 0
                except Exception as e:
                    consecutive_net_errors += 1
                    self.notify_callbacks(
                        "training_log",
                        {
                            "message": f"Network issue while polling remote "
                            f"({consecutive_net_errors}): {e}"
                        },
                    )
                    self._persist_remote_job({"status": "detached"})
                    if consecutive_net_errors >= 3:
                        self.is_training = False
                        self.notify_callbacks(
                            "training_detached",
                            {
                                "error": str(e),
                                "hint": "Remote job kept running. Reconnect SSH and click Reattach.",
                            },
                        )
                        return
                    time.sleep(2.0)
                    continue

                try:
                    # Incremental byte-offset read. Previously we always read
                    # from the start with a 2MB cap, so once train_run.log grew
                    # past that, UI progress (results.csv) kept updating but
                    # the log stream froze.
                    last_size = self._emit_remote_log_delta(
                        ssh_client, remote_log, last_size
                    )
                except (SSHError, OSError, IOError) as e:
                    consecutive_net_errors += 1
                    self.notify_callbacks(
                        "training_log",
                        {"message": f"Log stream interrupted: {e}"},
                    )
                    self._persist_remote_job({"status": "detached"})
                    if consecutive_net_errors >= 3:
                        self.is_training = False
                        self.notify_callbacks(
                            "training_detached",
                            {
                                "error": str(e),
                                "hint": "Remote job kept running. Reconnect SSH and click Reattach.",
                            },
                        )
                        return
                except Exception:
                    pass

                # Progress + preview images from remote run dir (not log text)
                csv_poll_counter += 1
                self._notify_epoch_from_remote_csv(ssh_client)
                if csv_poll_counter % 3 == 0:
                    self._notify_remote_train_batch_images(ssh_client)

                if not alive_ok:
                    break
                time.sleep(1.5)

            if self.stop_event.is_set():
                if self._detach_without_kill:
                    self.is_training = False
                    self._persist_remote_job({"status": "detached"})
                    self.notify_callbacks(
                        "training_detached",
                        {
                            "hint": (
                                "Remote training continues. "
                                "Re-open MyTrain, connect SSH, and Reattach Job."
                            )
                        },
                    )
                    return
                try:
                    ssh_client.exec(
                        f"kill {shlex.quote(str(self._remote_pid))} 2>/dev/null || true"
                    )
                except Exception:
                    pass
                self.is_training = False
                clear_remote_job()
                self.notify_callbacks("training_stopped", {})
                return

            try:
                self._emit_remote_log_delta(
                    ssh_client, remote_log, last_size
                )
            except Exception:
                pass

            try:
                _c2, o2, _ = ssh_client.exec(
                    f"tail -n 40 {shlex.quote(remote_log)} 2>/dev/null || true"
                )
                tail = o2 or ""
            except Exception as e:
                self.is_training = False
                self._persist_remote_job({"status": "detached"})
                self.notify_callbacks(
                    "training_detached",
                    {
                        "error": str(e),
                        "hint": "Could not confirm finish. Reconnect to check status.",
                    },
                )
                return

            self.is_training = False
            # Final progress / preview sync before completion callbacks
            try:
                self._notify_epoch_from_remote_csv(ssh_client)
                self._notify_remote_train_batch_images(ssh_client)
            except Exception:
                pass
            if "ENV_ERROR:" in tail or "TRAIN_ERROR:" in tail:
                extra = fetch_remote_log_tail(ssh_client, [remote_log], lines=120)
                clear_remote_job()
                self.notify_callbacks(
                    "training_error",
                    {"error": extra or tail},
                )
            elif "Traceback (most recent call last):" in tail and (
                "Error" in tail or "Exception" in tail
            ):
                extra = fetch_remote_log_tail(ssh_client, [remote_log], lines=120)
                clear_remote_job()
                self.notify_callbacks(
                    "training_error",
                    {"error": extra or "Remote training failed (see log)"},
                )
            else:
                clear_remote_job()
                self.notify_callbacks(
                    "training_completed",
                    {
                        "results": "Training completed successfully",
                        "tensorboard_url": self.tensorboard_url,
                    },
                )
        except Exception as e:
            self.is_training = False
            self._persist_remote_job({"status": "detached"})
            extra = ""
            try:
                extra = fetch_remote_log_tail(
                    ssh_client, [remote_log], lines=120
                )
            except Exception:
                pass
            self.notify_callbacks(
                "training_detached",
                {
                    "error": f"{e}\n{extra}".strip(),
                    "hint": "Remote job may still be running. Reconnect and click Reattach.",
                },
            )

    @staticmethod
    def _tr_attach_msg(msg: str) -> str:
        return msg

    def attach_remote_training(self, ssh_client) -> Tuple[bool, str]:
        """Reattach UI to a previously started remote job (nohup still running)."""
        if self.is_training:
            return False, "Training monitor is already attached"

        if ssh_client is None or not getattr(ssh_client, "connected", False):
            return False, "Remote SSH is not connected"

        from anylabeling.services.auto_training.mytrain.remote.job_state import (
            clear_remote_job,
            load_remote_job,
            probe_remote_job,
        )

        job = load_remote_job()
        status, msg = probe_remote_job(ssh_client, job)
        if status == "none":
            return False, "No saved remote training job found"
        if status == "mismatch":
            return False, msg
        if status == "error":
            return False, msg
        if status == "finished":
            clear_remote_job()
            return False, f"Remote job already finished: {msg}"

        self._mode = "remote"
        self._ssh_client = ssh_client
        self.stop_event.clear()
        self._remote_pid = str(job.get("pid") or "")
        self._remote_log_path = job.get("log_path") or ""
        self._remote_run_dir = job.get("run_dir") or ""
        self._remote_project = job.get("project") or ""
        self._remote_name = job.get("name") or ""
        self.total_epochs = int(job.get("epochs") or 100)
        self._custom_job = bool(job.get("custom"))
        self.tensorboard_url = job.get("tensorboard_url") or ""
        self._log_offset = int(job.get("log_offset") or 0)

        # Prefer host:port URL for Open TensorBoard (Docker published port)
        remote_tb_port = int(job.get("tensorboard_remote_port") or 6006)
        host = getattr(ssh_client, "host", "") or job.get("host") or ""
        if host:
            self.tensorboard_url = f"http://{host}:{remote_tb_port}"
        else:
            self.tensorboard_url = (
                job.get("tensorboard_url")
                or f"http://127.0.0.1:{remote_tb_port}"
            )

        self._persist_remote_job(
            {
                "status": "running",
                "tensorboard_url": self.tensorboard_url,
                "tensorboard_remote_port": remote_tb_port,
            }
        )
        threading.Thread(
            target=self._monitor_remote,
            args=(ssh_client,),
            kwargs={"from_attach": True},
            daemon=True,
        ).start()
        return True, f"Reattached to remote PID {self._remote_pid}"

    def _start_remote(
        self,
        train_args: Dict,
        ssh_client,
        *,
        remote_python: str = "",
    ) -> Tuple[bool, str]:
        if ssh_client is None or not getattr(ssh_client, "connected", False):
            return False, "Remote SSH is not connected"

        try:
            from anylabeling.services.auto_training.mytrain.remote.job_state import (
                load_remote_job,
                probe_remote_job,
            )
            from anylabeling.services.auto_training.mytrain.remote.remote_paths import (
                runs_remote_dir,
            )
            from anylabeling.services.auto_training.mytrain.remote.remote_trainer import (
                build_remote_train_script,
                fetch_remote_log_tail,
                start_remote_tensorboard,
                upload_model_if_local,
            )

            # Avoid starting a second job if one is still alive on this host
            existing = load_remote_job()
            st, sm = probe_remote_job(ssh_client, existing)
            if st == "alive":
                return (
                    False,
                    "A remote job is still running. Use Reattach instead of starting a new one.",
                )

            args = dict(train_args)
            self._custom_job = False
            self._mode = "remote"
            self._ssh_client = ssh_client
            self.total_epochs = args.get("epochs", 100)
            self.stop_event.clear()
            self._detach_without_kill = False
            self.tensorboard_url = ""
            self._log_offset = 0

            task = args.pop("_task", "detect")
            remote_project = args.get("project") or runs_remote_dir(
                task, ssh_client.home
            )
            args["project"] = remote_project
            ssh_client.mkdir_p(remote_project)

            model = upload_model_if_local(
                ssh_client, args.get("model", ""), remote_project
            )
            args["model"] = model

            script = build_remote_train_script(args)
            remote_script = f"{remote_project.rstrip('/')}/train_script.py"
            remote_log = f"{remote_project.rstrip('/')}/train_run.log"
            self._remote_log_path = remote_log
            self._remote_project = remote_project
            self._remote_name = args.get("name", "exp")

            run_dir = f"{remote_project.rstrip('/')}/{args.get('name', 'exp')}"
            self._remote_run_dir = run_dir
            # Same Name + exist_ok reuses the folder; wipe previous results.csv /
            # train_batch / weights so progress & previews don't show stale data.
            self._reset_remote_run_artifacts(ssh_client, run_dir, remote_log)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix="_train.py", delete=False, encoding="utf-8"
            ) as f:
                f.write(script)
                local_script = f.name
            try:
                ssh_client.upload_file(local_script, remote_script)
            finally:
                try:
                    os.unlink(local_script)
                except OSError:
                    pass

            # Do NOT mkdir run_dir before train — Ultralytics may bump name-2
            # when exist_ok is False; with exist_ok=True it recreates outputs.
            try:
                # Watch the project root so each exp (test1/...) shows as a run;
                # empty single-run dir before epoch 1 causes "no dashboards".
                port, url = start_remote_tensorboard(
                    ssh_client,
                    remote_project,
                    remote_python=remote_python or None,
                )
                self.tensorboard_url = url
                self._persist_remote_job(
                    {
                        "tensorboard_url": url,
                        "tensorboard_remote_port": port,
                        "tensorboard_logdir": remote_project,
                    }
                )
                self.notify_callbacks(
                    "training_log",
                    {"message": f"TensorBoard: {url}"},
                )
                self.notify_callbacks(
                    "training_log",
                    {
                        "message": (
                            f"TensorBoard logdir={remote_project}; "
                            "Ultralytics tensorboard logging is enabled; "
                            "refresh after epoch 1 and select the run on the left"
                        )
                    },
                )
            except Exception as e:
                self.notify_callbacks(
                    "training_log",
                    {"message": f"TensorBoard not started: {e}"},
                )
            from anylabeling.services.auto_training.mytrain.remote.remote_runtime import (
                format_nohup_train,
            )

            start_cmd = format_nohup_train(
                remote_python or None, remote_script, remote_log
            )
            code, out, err = ssh_client.exec(start_cmd)

            pid_line = (out or "").strip().splitlines()
            if code != 0 or not pid_line:
                extra = fetch_remote_log_tail(ssh_client, [remote_log])
                msg = f"Failed to start remote training: {err or out}\n{extra}"
                return False, msg

            self._remote_pid = pid_line[-1].strip()
            self._persist_remote_job(
                {
                    "data": args.get("data", ""),
                    "model": args.get("model", ""),
                    "status": "running",
                    "remote_python": (remote_python or "").strip(),
                }
            )

            threading.Thread(
                target=self._monitor_remote,
                args=(ssh_client,),
                kwargs={"from_attach": False},
                daemon=True,
            ).start()
            return True, "Remote training started successfully"

        except Exception as e:
            return False, f"Failed to start remote training: {e}"

    def start_custom_training(
        self,
        spec: Dict,
        *,
        mode: str = "local",
        ssh_client=None,
        python_executable: str = "",
        remote_python: str = "",
    ) -> Tuple[bool, str]:
        """Run a user-defined shell command (placeholders already resolved)."""
        if self.is_training:
            return False, "Training is already in progress"
        mode = (mode or "local").lower()
        if mode == "remote":
            return self._start_custom_remote(
                spec, ssh_client, remote_python=remote_python or ""
            )
        return self._start_custom_local(
            spec, python_executable=python_executable or ""
        )

    def _start_custom_local(
        self, spec: Dict, *, python_executable: str = ""
    ) -> Tuple[bool, str]:
        import sys

        from anylabeling.services.auto_training.mytrain.remote.job_state import (
            load_local_job,
            probe_local_job,
        )

        try:
            resolved = (spec.get("resolved_command") or "").strip()
            if not resolved:
                return False, "Custom command is empty"
            project = (spec.get("project") or "").strip() or os.path.abspath(
                os.path.curdir
            )
            name = (spec.get("name") or "exp").strip()
            run_dir = os.path.join(project, name)
            exec_cwd = (spec.get("exec_cwd") or "").strip() or None
            if exec_cwd and not os.path.isdir(exec_cwd):
                return False, f"Execution directory not found: {exec_cwd}"

            existing = load_local_job()
            st, _sm = probe_local_job(existing)
            if st == "alive":
                return (
                    False,
                    "A local training job is still running. Reattach to watch it.",
                )
            from anylabeling.services.auto_training.mytrain.remote.job_state import (
                clear_local_job,
            )

            if existing:
                clear_local_job()

            os.makedirs(project, exist_ok=True)
            _purge_local_run_dir(run_dir)

            log_path = os.path.join(project, "train_run.log")
            self._close_local_log_handle()
            try:
                with open(log_path, "w", encoding="utf-8") as header_f:
                    header_f.write(
                        f"=== Custom training {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    )
                    header_f.write(f"Command: {resolved}\n")
                    if exec_cwd:
                        header_f.write(f"CWD: {exec_cwd}\n")
            except OSError:
                pass
            self._mode = "local"
            self._custom_job = True
            from anylabeling.services.auto_training.mytrain.custom_training import (
                parse_epochs_from_cli,
            )

            self.total_epochs = int(spec.get("epochs") or 0) or parse_epochs_from_cli(
                resolved
            )
            self.stop_event.clear()
            self._detach_without_kill = False
            self.tensorboard_url = ""
            self._local_log_path = log_path
            self._local_project = project
            self._local_name = name
            self._local_run_dir = run_dir
            py_exe = (python_executable or "").strip() or sys.executable
            self._local_python_exe = py_exe
            self._log_offset = 0
            self._local_run_started_at = time.time()

            log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
            self._local_log_handle = log_handle
            popen_kwargs: Dict = {
                "stdout": log_handle,
                "stderr": subprocess.STDOUT,
                "cwd": exec_cwd,
            }
            using_external_py = bool(
                (python_executable or "").strip()
                and os.path.isfile((python_executable or "").strip())
            )
            if using_external_py:
                popen_kwargs["env"] = interpreter_path_env(py_exe)
                launcher = (
                    "import os, subprocess, sys; "
                    f"rc = subprocess.run({resolved!r}, shell=True, env=os.environ); "
                    "sys.exit(rc.returncode if rc.returncode is not None else 1)"
                )
                popen_cmd = [py_exe, "-c", launcher]
                popen_kwargs["stdout"] = log_handle
                popen_kwargs["stderr"] = subprocess.STDOUT
                if os.name != "nt":
                    popen_kwargs["preexec_fn"] = os.setsid
                    popen_kwargs["close_fds"] = True
                else:
                    popen_kwargs["creationflags"] = (
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        | getattr(
                            subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000
                        )
                    )
            elif os.name == "nt":
                popen_kwargs["shell"] = True
                popen_cmd = resolved
                popen_kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
                )
            else:
                popen_cmd = wrap_local_shell_command(resolved)
                popen_kwargs["preexec_fn"] = os.setsid
                popen_kwargs["close_fds"] = True

            self.training_process = subprocess.Popen(popen_cmd, **popen_kwargs)
            self._local_pid = self.training_process.pid
            self._persist_local_job(
                {
                    "status": "running",
                    "custom": True,
                    "resolved_command": resolved[:2000],
                }
            )
            threading.Thread(
                target=self._monitor_local,
                kwargs={"from_attach": False, "script_path": ""},
                daemon=True,
            ).start()
            return True, "Custom training started successfully"
        except Exception as e:
            self._custom_job = False
            return False, f"Failed to start custom training: {e}"

    def _start_custom_remote(
        self, spec: Dict, ssh_client, *, remote_python: str = ""
    ) -> Tuple[bool, str]:
        from anylabeling.services.auto_training.mytrain.remote.job_state import (
            load_remote_job,
            probe_remote_job,
        )
        from anylabeling.services.auto_training.mytrain.remote.remote_trainer import (
            fetch_remote_log_tail,
        )

        if ssh_client is None or not getattr(ssh_client, "connected", False):
            return False, "Remote SSH is not connected"
        try:
            resolved = (spec.get("resolved_command") or "").strip()
            if not resolved:
                return False, "Custom command is empty"
            project = (spec.get("project") or "").strip().replace("\\", "/")
            if not project:
                return False, "Project path is required"
            name = (spec.get("name") or "exp").strip()
            exec_cwd = (spec.get("exec_cwd") or "").strip().replace("\\", "/")
            if exec_cwd:
                _c, o, _e = ssh_client.exec(
                    f"test -d {shlex.quote(exec_cwd)} && echo ok"
                )
                if "ok" not in (o or ""):
                    return False, f"Remote execution directory not found: {exec_cwd}"

            existing = load_remote_job()
            st, _sm = probe_remote_job(ssh_client, existing)
            if st == "alive":
                return (
                    False,
                    "A remote job is still running. Use Reattach instead of starting a new one.",
                )

            ssh_client.mkdir_p(project)
            remote_log = f"{project.rstrip('/')}/train_run.log"
            run_dir = f"{project.rstrip('/')}/{name}"
            from anylabeling.services.auto_training.mytrain.custom_training import (
                parse_epochs_from_cli,
            )

            ssh_client.exec(f": > {shlex.quote(remote_log)}")

            self._mode = "remote"
            self._ssh_client = ssh_client
            self._custom_job = True
            self.total_epochs = int(spec.get("epochs") or 0) or parse_epochs_from_cli(
                resolved
            )
            self.stop_event.clear()
            self._detach_without_kill = False
            self.tensorboard_url = ""
            self._remote_log_path = remote_log
            self._remote_project = project
            self._remote_name = name
            self._remote_run_dir = run_dir
            self._log_offset = 0

            start_cmd = build_remote_custom_launch(
                resolved,
                remote_log=remote_log,
                exec_cwd=exec_cwd,
            )
            code, out, err = ssh_client.exec(start_cmd)
            pid_line = (out or "").strip().splitlines()
            if code != 0 or not pid_line:
                extra = fetch_remote_log_tail(ssh_client, [remote_log])
                return False, f"Failed to start remote custom training: {err or out}\n{extra}"

            self._remote_pid = pid_line[-1].strip()
            self._persist_remote_job(
                {
                    "status": "running",
                    "custom": True,
                    "resolved_command": resolved[:2000],
                    "remote_python": (remote_python or "").strip(),
                }
            )
            threading.Thread(
                target=self._monitor_remote,
                args=(ssh_client,),
                kwargs={"from_attach": False},
                daemon=True,
            ).start()
            return True, "Remote custom training started successfully"
        except Exception as e:
            self._custom_job = False
            return False, f"Failed to start remote custom training: {e}"

    def _local_root_pid(self) -> Optional[int]:
        proc = self.training_process
        if proc is not None and proc.poll() is None:
            return int(proc.pid)
        if self._local_pid:
            return int(self._local_pid)
        return None

    def stop_training(self, wait: bool = False) -> bool:
        if not self.is_training and self._mode not in ("remote", "local"):
            return False
        try:
            self._detach_without_kill = False
            self.stop_event.set()

            from anylabeling.services.auto_training.mytrain.remote.job_state import (
                clear_local_job,
                clear_remote_job,
                is_local_pid_alive,
                load_local_job,
                load_remote_job,
            )

            root_pid = self._local_root_pid()
            if root_pid and is_local_pid_alive(root_pid):
                _kill_process_tree(root_pid)
                self._close_local_log_handle()
                if self.training_process is not None:
                    try:
                        self.training_process.wait(timeout=8)
                    except Exception:
                        pass
                    self.training_process = None

            if not self.is_training:
                ljob = load_local_job()
                lpid = (ljob or {}).get("pid")
                if lpid and is_local_pid_alive(lpid):
                    _kill_process_tree(int(lpid))
                    clear_local_job()

            if self._ssh_client is not None and not self.is_training:
                job = load_remote_job()
                pid = (job or {}).get("pid") or self._remote_pid
                if pid:
                    try:
                        if os.name == "nt":
                            self._ssh_client.exec(
                                f"kill {shlex.quote(str(pid))} 2>/dev/null || true"
                            )
                        else:
                            self._ssh_client.exec(
                                f"kill -- -$(ps -o pgid= -p {shlex.quote(str(pid))} | tr -d ' ') "
                                f"2>/dev/null || kill {shlex.quote(str(pid))} 2>/dev/null || true"
                            )
                    except Exception:
                        pass
                clear_remote_job()
            return True
        except Exception:
            return False


_training_manager = TrainingManager()


def get_training_manager() -> TrainingManager:
    return _training_manager
