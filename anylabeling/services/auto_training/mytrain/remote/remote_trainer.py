"""Remote Detect training + TensorBoard helpers over SSH."""

from __future__ import annotations

import json
import os
import shlex
import tempfile
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from anylabeling.services.auto_training.mytrain.dataset_lists import (
    build_remote_merge_script,
)

from .remote_paths import dataset_remote_dir, runs_remote_dir
from .remote_runtime import (
    format_exec_python,
    http_probe_python,
    tensorboard_launcher_candidates,
)
from .ssh_client import SSHClient, SSHError


_UPLOAD_MANIFEST = ".xal_upload_manifest.json"


class UploadStopped(Exception):
    """User stopped a dataset upload. A later run must upload again."""


class UploadGate:
    """Pause blocks the next file; stop aborts and cannot be resumed."""

    def __init__(self) -> None:
        self._resume = threading.Event()
        self._resume.set()
        self._stop = threading.Event()

    def pause(self) -> None:
        if not self._stop.is_set():
            self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def checkpoint(self) -> None:
        while not self._resume.is_set():
            if self._stop.is_set():
                raise UploadStopped("Upload stopped")
            self._resume.wait(0.2)
        if self._stop.is_set():
            raise UploadStopped("Upload stopped")


def _local_upload_entries(
    local_root: str, subdirs: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    """Selected subdirs plus ROOT/classes.txt. Values are size and mtime_ns."""
    root = Path(local_root)
    entries: Dict[str, Dict[str, int]] = {}
    classes = root / "classes.txt"
    if classes.is_file():
        st = classes.stat()
        entries["classes.txt"] = {
            "size": int(st.st_size),
            "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        }
    for sd in subdirs:
        sub = root / sd
        if not sub.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(sub):
            dirnames[:] = [d for d in dirnames if d and not d.startswith(".")]
            for name in filenames:
                if not name or name.startswith("."):
                    continue
                full = Path(dirpath) / name
                if not full.is_file():
                    continue
                rel = full.relative_to(root).as_posix()
                st = full.stat()
                entries[rel] = {
                    "size": int(st.st_size),
                    "mtime_ns": int(
                        getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
                    ),
                }
    return entries


def _remote_selected_sizes(
    client: SSHClient,
    remote_root: str,
    subdirs: Sequence[str],
    *,
    remote_python: str = "",
) -> Optional[Dict[str, int]]:
    """Size of classes.txt and each selected subdir already on the server."""
    script = """# -*- coding: utf-8 -*-
import json, os, sys
root = sys.argv[1]
subdirs = json.loads(sys.argv[2])
out = {}
classes = os.path.join(root, "classes.txt")
if os.path.isfile(classes):
    out["classes.txt"] = os.path.getsize(classes)
for sd in subdirs:
    base = os.path.join(root, sd)
    if not os.path.isdir(base):
        continue
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d and not d.startswith(".")]
        for name in filenames:
            if not name or name.startswith("."):
                continue
            full = os.path.join(dirpath, name)
            if not os.path.isfile(full):
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            try:
                out[rel] = os.path.getsize(full)
            except OSError:
                pass
print(json.dumps(out))
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_xal_sizes.py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        local_script = f.name
    remote_script = f"{remote_root.rstrip('/')}/_xal_sizes.py"
    try:
        client.mkdir_p(remote_root)
        client.upload_file(local_script, remote_script)
        script_q = shlex.quote(remote_script)
        arg_s = (
            shlex.quote(remote_root)
            + " "
            + shlex.quote(json.dumps(list(subdirs)))
        )
        py = (remote_python or "").strip()
        if py:
            cmd = f"{shlex.quote(py)} {script_q} {arg_s}"
        else:
            cmd = (
                f"python3 {script_q} {arg_s} || python {script_q} {arg_s}"
            )
        code, out, err = client.exec(cmd, timeout=300)
        client.exec(f"rm -f {shlex.quote(remote_script)}")
        if code != 0:
            return None
        text = (out or "").strip().splitlines()
        if not text:
            return {}
        data = json.loads(text[-1])
        if not isinstance(data, dict):
            return None
        return {str(k): int(v) for k, v in data.items()}
    except Exception:
        return None
    finally:
        try:
            os.unlink(local_script)
        except OSError:
            pass


def _read_remote_manifest(
    client: SSHClient, remote_root: str
) -> Optional[dict]:
    path = f"{remote_root.rstrip('/')}/{_UPLOAD_MANIFEST}"
    try:
        raw = client.read_text(path, max_bytes=8_000_000)
    except Exception:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _payload_unchanged(
    local_entries: Dict[str, Dict[str, int]],
    remote_sizes: Optional[Dict[str, int]],
    manifest: Optional[dict],
) -> bool:
    if remote_sizes is None:
        return False
    local_sizes = {rel: meta["size"] for rel, meta in local_entries.items()}
    if local_sizes != remote_sizes:
        return False
    if not manifest:
        return True
    old = manifest.get("files") or {}
    if not isinstance(old, dict) or set(old) != set(local_entries):
        return False
    for rel, meta in local_entries.items():
        prev = old.get(rel) or {}
        if not isinstance(prev, dict):
            return False
        if int(prev.get("size", -1)) != int(meta["size"]):
            return False
        if int(prev.get("mtime_ns", -1)) != int(meta["mtime_ns"]):
            return False
    return True


def _write_remote_manifest(
    client: SSHClient,
    remote_root: str,
    subdirs: Sequence[str],
    entries: Dict[str, Dict[str, int]],
) -> None:
    payload = {
        "version": 1,
        "subdirs": list(subdirs),
        "files": entries,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_xal_manifest.json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(payload, f)
        local_path = f.name
    try:
        client.upload_file(
            local_path, f"{remote_root.rstrip('/')}/{_UPLOAD_MANIFEST}"
        )
    finally:
        try:
            os.unlink(local_path)
        except OSError:
            pass


def upload_dataset(
    client: SSHClient,
    local_root: str,
    *,
    subdirs: Sequence[str],
    progress: Optional[Callable[[str], None]] = None,
    remote_base: str = "",
    remote_python: str = "",
    control: Optional[UploadGate] = None,
) -> Tuple[str, bool]:
    """Upload selected subdirs and classes.txt.

    Returns ``(remote_path, skipped)``. ``skipped`` is True when the remote
    copy already matches and nothing was transferred.
    """
    remote = dataset_remote_dir(local_root, client.home, remote_base)
    entries = _local_upload_entries(local_root, subdirs)
    if "classes.txt" not in entries:
        raise SSHError(f"classes.txt not found under {local_root}")
    if not any(rel != "classes.txt" for rel in entries):
        raise SSHError("No files found in the selected sub-datasets")

    def _tick() -> None:
        if control is not None:
            control.checkpoint()

    _tick()
    client.mkdir_p(remote)
    remote_sizes = _remote_selected_sizes(
        client, remote, subdirs, remote_python=remote_python
    )
    manifest = _read_remote_manifest(client, remote)
    if _payload_unchanged(entries, remote_sizes, manifest):
        if not manifest:
            _write_remote_manifest(client, remote, subdirs, entries)
        if progress:
            progress(f"unchanged {remote}")
        return remote, True

    root = Path(local_root)
    for sd in subdirs:
        _tick()
        remote_sub = f"{remote.rstrip('/')}/{sd}"
        client.exec(f"rm -rf {shlex.quote(remote_sub)}")
        local_sub = root / sd
        if local_sub.is_dir():
            client.upload_directory(
                str(local_sub),
                remote_sub,
                progress=progress,
                control=control,
            )
    _tick()
    client.upload_file(str(root / "classes.txt"), f"{remote.rstrip('/')}/classes.txt")
    _tick()
    _write_remote_manifest(client, remote, subdirs, entries)
    return remote, False


def remote_merge_lists(
    client: SSHClient,
    remote_root: str,
    subdirs: Sequence[str],
    *,
    remote_python: Optional[str] = None,
) -> Tuple[bool, str]:
    """Run merge script on remote; return (ok, message_or_yaml_path)."""
    script = build_remote_merge_script(remote_root, list(subdirs))
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_merge_dataset.py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        local_script = f.name
    try:
        remote_script = f"{remote_root}/_xal_merge_dataset.py"
        client.upload_file(local_script, remote_script)
        code, out, err = client.exec(
            format_exec_python(remote_python, remote_script),
            timeout=600,
        )
        client.exec(f"rm -f {shlex.quote(remote_script)}")
        text = (out or "") + (err or "")
        if code != 0:
            return False, text.strip() or f"merge failed with code {code}"
        yaml_path = f"{remote_root.rstrip('/')}/data.yaml"
        return True, yaml_path
    finally:
        try:
            os.unlink(local_script)
        except OSError:
            pass


def upload_model_if_local(
    client: SSHClient, model_path: str, remote_project: str
) -> str:
    """If model_path is a local file, upload beside remote project; else return as-is."""
    if model_path and os.path.isfile(model_path):
        name = os.path.basename(model_path)
        remote_model = f"{remote_project.rstrip('/')}/{name}"
        client.upload_file(model_path, remote_model)
        return remote_model
    return model_path


def build_remote_train_script(train_args: Dict) -> str:
    args = dict(train_args)
    model = args.pop("model")
    return f"""# -*- coding: utf-8 -*-
import sys
import traceback

try:
    from ultralytics import YOLO, settings
except Exception as e:
    print("ENV_ERROR: failed to import ultralytics: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(2)

try:
    # Ultralytics >=8.3.111 disables TensorBoard by default
    settings.update({{"tensorboard": True}})
except Exception as e:
    print("WARN: could not enable tensorboard settings: %s" % e, flush=True)

try:
    model = YOLO({repr(model)})
    train_args = {repr(args)}
    train_args["verbose"] = True
    train_args["show"] = False
    results = model.train(**train_args)
except Exception as e:
    print("TRAIN_ERROR: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(1)
"""


def _resolve_remote_tensorboard_cmd(
    client: SSHClient, remote_python: Optional[str] = None
) -> str:
    """
    Find how to invoke TensorBoard inside the remote env/container.

    Docker YOLO images often have the Python package but no ``tensorboard``
    shell script on PATH.
    """
    candidates = tensorboard_launcher_candidates(remote_python)
    for launcher in candidates:
        # Run in sh so $? reflects the launcher, not echo.
        code, out, err = client.exec(
            f"sh -c {shlex.quote(launcher + ' --help >/dev/null 2>&1')}; echo __EC:$?",
            timeout=40,
        )
        text = ((out or "") + (err or "")).strip()
        if "__EC:0" in text:
            return launcher
    raise SSHError(
        "TensorBoard is not available in the remote container. "
        "Install it inside the container, e.g.: "
        "pip install tensorboard"
    )


def _remote_tb_ready(
    client: SSHClient,
    port: int,
    tb_log: str,
    remote_python: Optional[str] = None,
) -> Tuple[bool, str]:
    """Return (ready, detail). Prefer log / HTTP probe — ss/netstat often missing in Docker."""
    detail = ""
    try:
        detail = client.read_text(tb_log, max_bytes=4000)
    except Exception:
        detail = ""

    if detail:
        if (
            "No such file or directory" in detail
            or "No module named" in detail
            or "ModuleNotFoundError" in detail
        ):
            return False, detail
        # Success banner, e.g. "TensorBoard 2.21.0 at http://0.0.0.0:6006/"
        if "at http://" in detail.lower() or f":{port}/" in detail:
            return True, detail

    probe = (
        http_probe_python(remote_python, port)
        + f" || curl -sf -o /dev/null http://127.0.0.1:{port}/ && echo OK "
        f"|| wget -q -O /dev/null http://127.0.0.1:{port}/ && echo OK "
        f"|| true"
    )
    _c, out, _e = client.exec(probe, timeout=15)
    if "OK" in (out or ""):
        return True, detail or out

    # Last resort: ss/netstat if present
    _c2, o2, _ = client.exec(
        f"(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) "
        f"| grep -E ':{port}\\s' && echo READY || true"
    )
    if "READY" in (o2 or "") or f":{port}" in (o2 or ""):
        return True, detail or o2

    return False, detail


def start_remote_tensorboard(
    client: SSHClient,
    logdir: str,
    preferred_port: int = 6006,
    *,
    remote_python: Optional[str] = None,
) -> Tuple[int, str]:
    """
    Start tensorboard on remote.

    Returns (remote_port, url) where url is ``http://<ssh-host>:<port>/``
    so the UI "Open TensorBoard" button opens the published Docker port
    (e.g. http://192.168.112.217:6006/).
    """
    import time

    launcher = _resolve_remote_tensorboard_cmd(client, remote_python)

    # Ensure logdir exists (may still be empty before first epoch)
    client.mkdir_p(logdir)

    port = preferred_port
    for candidate in range(preferred_port, preferred_port + 20):
        ready_now, _ = _remote_tb_ready(
            client,
            candidate,
            f"/tmp/xal_tensorboard_{candidate}.log",
            remote_python=remote_python,
        )
        # If something already answers on this port, reuse it after restart
        _c, o, _ = client.exec(
            f"(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null || true) "
            f"| grep -E ':{candidate}\\s' || true"
        )
        if f":{candidate}" not in (o or "") and not ready_now:
            # Also try bind probe via python
            _c3, o3, _ = client.exec(
                f"python3 -c \"import socket;s=socket.socket();r=s.connect_ex(("
                f"'127.0.0.1',{candidate}));print(r);s.close()\" 2>/dev/null || echo 1"
            )
            try:
                if int((o3 or "1").strip().splitlines()[-1]) != 0:
                    port = candidate
                    break
            except Exception:
                port = candidate
                break

    logdir_q = shlex.quote(logdir)
    tb_log = f"/tmp/xal_tensorboard_{port}.log"
    client.exec("pkill -f 'tensorboard' 2>/dev/null || true")
    time.sleep(0.5)

    # reload_interval helps pick up events written after TB starts
    cmd = (
        f"nohup {launcher} --logdir {logdir_q} --host 0.0.0.0 --port {port} "
        f"--reload_interval 5 "
        f"> {shlex.quote(tb_log)} 2>&1 & echo $!"
    )
    code, out, err = client.exec(cmd)
    if code != 0:
        raise SSHError(f"Failed to start TensorBoard: {err or out}")

    ready = False
    last_detail = ""
    for _ in range(40):
        time.sleep(0.5)
        ready, last_detail = _remote_tb_ready(
            client, port, tb_log, remote_python=remote_python
        )
        if ready:
            break
        if last_detail and (
            "No such file or directory" in last_detail
            or "No module named" in last_detail
            or "ModuleNotFoundError" in last_detail
        ):
            raise SSHError(
                f"TensorBoard failed to start via `{launcher}`:\n{last_detail}\n"
                "Install in the container: pip install tensorboard"
            )

    if not ready:
        raise SSHError(
            f"TensorBoard did not become ready on port {port} "
            f"(launcher=`{launcher}`).\n{last_detail}\n"
            "If missing, install inside the container: pip install tensorboard"
        )

    # UI opens host IP directly (e.g. http://192.168.112.217:6006/)
    if getattr(client, "host", None):
        url = f"http://{client.host}:{port}"
    else:
        url = f"http://127.0.0.1:{port}"

    return port, url


def start_local_tensorboard(
    logdir: str,
    preferred_port: int = 6006,
    *,
    python_executable: str = "",
) -> Tuple[object, int, str]:
    """Start local tensorboard subprocess. Returns (Popen, port, url)."""
    import subprocess
    import socket
    import sys

    def _free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) != 0

    port = preferred_port
    for candidate in range(preferred_port, preferred_port + 20):
        if _free(candidate):
            port = candidate
            break

    tb_args = ["--logdir", logdir, "--host", "127.0.0.1", "--port", str(port)]
    py = (python_executable or "").strip() or sys.executable
    candidates = [
        ["tensorboard", *tb_args],
        [py, "-m", "tensorboard.main", *tb_args],
        [sys.executable, "-m", "tensorboard.main", *tb_args],
        [sys.executable, "-m", "tensorboard", *tb_args],
    ]
    last_err: Optional[Exception] = None
    for cmd in candidates:
        exe = cmd[0]
        if exe not in ("tensorboard",) and not os.path.isfile(exe):
            continue
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            url = f"http://127.0.0.1:{port}"
            return proc, port, url
        except (FileNotFoundError, OSError) as e:
            last_err = e
            continue
    raise FileNotFoundError(last_err or "tensorboard not found in PATH")


def fetch_remote_log_tail(
    client: SSHClient, remote_paths: List[str], lines: int = 200
) -> str:
    chunks: List[str] = []
    for path in remote_paths:
        if not path:
            continue
        code, out, err = client.exec(
            f"tail -n {int(lines)} {shlex.quote(path)} 2>/dev/null || true"
        )
        text = (out or "").strip()
        if text:
            chunks.append(f"===== {path} =====\n{text}")
    return "\n\n".join(chunks)


def build_remote_onnx_export_script(
    weights_pt: str,
    output_onnx: str,
    imgsz: int = 640,
) -> str:
    """Python script: export PT → ONNX and save as ``output_onnx``."""
    return f"""# -*- coding: utf-8 -*-
import os
import sys
import traceback

weights_pt = {repr(weights_pt)}
output_onnx = {repr(output_onnx)}
imgsz = {int(imgsz)}

try:
    from ultralytics import YOLO
except Exception as e:
    print("ENV_ERROR: failed to import ultralytics: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(2)

if not os.path.isfile(weights_pt):
    print("EXPORT_ERROR: weights not found: %s" % weights_pt, flush=True)
    sys.exit(1)

try:
    model = YOLO(weights_pt)
    result = model.export(format="onnx", imgsz=imgsz)
    src = result if isinstance(result, str) else str(result)
    if not src or not os.path.isfile(src):
        guess = os.path.splitext(weights_pt)[0] + ".onnx"
        src = guess if os.path.isfile(guess) else src
    if not src or not os.path.isfile(src):
        print("EXPORT_ERROR: onnx output not found (got %r)" % src, flush=True)
        sys.exit(1)
    out_dir = os.path.dirname(output_onnx) or "."
    os.makedirs(out_dir, exist_ok=True)
    if os.path.abspath(src) != os.path.abspath(output_onnx):
        if os.path.exists(output_onnx):
            os.remove(output_onnx)
        os.replace(src, output_onnx)
    print("EXPORT_OK: %s" % output_onnx, flush=True)
except Exception as e:
    print("EXPORT_ERROR: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(1)
"""


def remote_export_onnx(
    client: SSHClient,
    run_dir: str,
    *,
    weights_file: str = "best.pt",
    output_name: str,
    imgsz: int = 640,
    remote_python: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Export best/last.pt to ONNX on the remote host into
    ``{run_dir}/weights/<output_name>``. Does not download the file.

    Returns (ok, message_or_remote_onnx_path).
    """
    run_dir = (run_dir or "").rstrip("/").replace("\\", "/")
    weights_file = os.path.basename(weights_file or "best.pt")
    if weights_file not in ("best.pt", "last.pt"):
        weights_file = "best.pt"
    output_name = os.path.basename(output_name or "")
    if not output_name.lower().endswith(".onnx"):
        return False, "output_name must end with .onnx"

    weights_pt = f"{run_dir}/weights/{weights_file}"
    output_onnx = f"{run_dir}/weights/{output_name}"
    script = build_remote_onnx_export_script(weights_pt, output_onnx, imgsz)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_export_onnx.py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        local_script = f.name
    remote_script = f"{run_dir}/_xal_export_onnx.py"
    try:
        try:
            client.upload_file(local_script, remote_script)
        finally:
            try:
                os.unlink(local_script)
            except OSError:
                pass

        code, out, err = client.exec(
            format_exec_python(remote_python, remote_script),
            timeout=3600,
        )
        text = ((out or "") + "\n" + (err or "")).strip()
        if code != 0 or "EXPORT_OK:" not in text:
            return False, text or f"Remote ONNX export failed (code={code})"
        remote_path = output_onnx
        for line in text.splitlines():
            if line.startswith("EXPORT_OK:"):
                remote_path = (
                    line.split("EXPORT_OK:", 1)[-1].strip() or output_onnx
                )
        return True, remote_path
    except SSHError as e:
        return False, str(e)
    finally:
        try:
            client.exec(
                f"rm -f {shlex.quote(remote_script)} 2>/dev/null || true"
            )
        except Exception:
            pass


def build_remote_predict_script(
    model_path: str,
    source_all_txt: str,
    run_dir: str,
    imgsz: int = 640,
) -> str:
    """Predict all.txt → YOLO txt under ``{run_dir}/predict/labels`` (no images)."""
    return f"""# -*- coding: utf-8 -*-
import os
import sys
import traceback

model_path = {repr(model_path)}
source = {repr(source_all_txt)}
run_dir = {repr(run_dir)}
imgsz = {int(imgsz)}
labels_dir = os.path.join(run_dir, "predict", "labels")

try:
    from ultralytics import YOLO
except Exception as e:
    print("ENV_ERROR: failed to import ultralytics: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(2)

if not os.path.isfile(model_path):
    print("PREDICT_ERROR: model not found: %s" % model_path, flush=True)
    sys.exit(1)
if not os.path.isfile(source):
    print("PREDICT_ERROR: source all.txt not found: %s" % source, flush=True)
    sys.exit(1)

try:
    model = YOLO(model_path)
    model.predict(
        source=source,
        save_txt=True,
        save_conf=False,
        rect=False,
        save=False,
        project=run_dir,
        name="predict",
        exist_ok=True,
        imgsz=imgsz,
        verbose=True,
    )
    n = 0
    if os.path.isdir(labels_dir):
        n = sum(
            1
            for f in os.listdir(labels_dir)
            if f.lower().endswith(".txt")
        )
    print("PREDICT_OK: %s (%d txt)" % (labels_dir, n), flush=True)
except Exception as e:
    print("PREDICT_ERROR: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(1)
"""


def remote_predict_yolo(
    client: SSHClient,
    run_dir: str,
    data_yaml: str,
    *,
    weights_file: str = "best.pt",
    imgsz: int = 640,
    remote_python: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Run YOLO predict on remote ``all.txt``; write txt labels to
    ``{run_dir}/predict/labels`` without saving annotated images.

    Returns (ok, message_or_labels_dir).
    """
    run_dir = (run_dir or "").rstrip("/").replace("\\", "/")
    data_yaml = (data_yaml or "").replace("\\", "/")
    if not run_dir or not data_yaml:
        return False, "run_dir and data_yaml are required"

    dataset_root = os.path.dirname(data_yaml).replace("\\", "/")
    all_txt = f"{dataset_root.rstrip('/')}/all.txt"
    weights_name = os.path.basename(weights_file or "best.pt")
    model_path = f"{run_dir}/weights/{weights_name}"

    if not client.exists(all_txt):
        return False, f"all.txt not found: {all_txt}"
    if not client.exists(model_path):
        return False, f"Model weights not found: {model_path}"

    script = build_remote_predict_script(
        model_path, all_txt, run_dir, imgsz=imgsz
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_predict.py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        local_script = f.name
    remote_script = f"{run_dir}/_xal_predict.py"
    labels_dir = f"{run_dir}/predict/labels"
    try:
        try:
            client.upload_file(local_script, remote_script)
        finally:
            try:
                os.unlink(local_script)
            except OSError:
                pass

        code, out, err = client.exec(
            format_exec_python(remote_python, remote_script),
            timeout=7200,
        )
        text = ((out or "") + "\n" + (err or "")).strip()
        if code != 0 or "PREDICT_OK:" not in text:
            return False, text or f"Remote predict failed (code={code})"
        path = labels_dir
        for line in text.splitlines():
            if line.startswith("PREDICT_OK:"):
                rest = line.split("PREDICT_OK:", 1)[-1].strip()
                # "path (N txt)" → path
                if " (" in rest:
                    rest = rest.split(" (", 1)[0].strip()
                path = rest or labels_dir
        return True, path
    except SSHError as e:
        return False, str(e)
    finally:
        try:
            client.exec(
                f"rm -f {shlex.quote(remote_script)} 2>/dev/null || true"
            )
        except Exception:
            pass


def list_remote_weight_choices(client: SSHClient, run_dir: str) -> List[str]:
    """Return best.pt / last.pt / *.onnx that exist under run_dir/weights."""
    run_dir = (run_dir or "").rstrip("/").replace("\\", "/")
    weights_dir = f"{run_dir}/weights"
    choices: List[str] = []
    for name in ("best.pt", "last.pt"):
        try:
            if client.exists(f"{weights_dir}/{name}"):
                choices.append(name)
        except Exception:
            pass
    try:
        for name in sorted(client.listdir(weights_dir)):
            if name.lower().endswith(".onnx") and name not in choices:
                choices.append(name)
    except Exception:
        pass
    return choices or ["best.pt", "last.pt"]


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def count_local_images(folder: str) -> int:
    n = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                n += 1
    return n


def build_remote_batch_predict_script(
    model_path: str,
    source_dir: str,
    run_dir: str,
    imgsz: int = 640,
) -> str:
    """Predict a folder → YOLO txt under ``{run_dir}/batch_infer/labels``."""
    return f"""# -*- coding: utf-8 -*-
import os
import sys
import traceback

model_path = {repr(model_path)}
source = {repr(source_dir)}
run_dir = {repr(run_dir)}
imgsz = {int(imgsz)}
labels_dir = os.path.join(run_dir, "batch_infer", "labels")

try:
    from ultralytics import YOLO
except Exception as e:
    print("ENV_ERROR: failed to import ultralytics: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(2)

if not os.path.isfile(model_path):
    print("PREDICT_ERROR: model not found: %s" % model_path, flush=True)
    sys.exit(1)
if not os.path.isdir(source):
    print("PREDICT_ERROR: source folder not found: %s" % source, flush=True)
    sys.exit(1)

try:
    model = YOLO(model_path)
    model.predict(
        source=source,
        save_txt=True,
        save_conf=False,
        rect=False,
        save=False,
        project=run_dir,
        name="batch_infer",
        exist_ok=True,
        imgsz=imgsz,
        verbose=True,
    )
    n = 0
    if os.path.isdir(labels_dir):
        n = sum(
            1
            for f in os.listdir(labels_dir)
            if f.lower().endswith(".txt")
        )
    print("BATCH_PREDICT_OK: %s (%d txt)" % (labels_dir, n), flush=True)
except Exception as e:
    print("PREDICT_ERROR: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(1)
"""


def remote_batch_predict_yolo(
    client: SSHClient,
    run_dir: str,
    local_image_folder: str,
    *,
    weights_file: str = "best.pt",
    imgsz: int = 640,
    progress: Optional[Callable[[str], None]] = None,
    model_path: Optional[str] = None,
    local_weight_to_upload: Optional[str] = None,
    remote_python: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Upload a local image folder and run YOLO predict on the remote server.

    Images → ``{run_dir}/batch_infer/images``
    Labels → ``{run_dir}/batch_infer/labels`` (same stem as images)

    ``model_path`` may override ``{run_dir}/weights/{weights_file}``.
    ``local_weight_to_upload`` copies a local .pt/.onnx into remote weights/.

    Returns (ok, remote_labels_dir_or_error).
    """
    run_dir = (run_dir or "").rstrip("/").replace("\\", "/")
    local_image_folder = os.path.abspath(local_image_folder or "")
    if not run_dir:
        return False, "run_dir is required"
    if not local_image_folder or not os.path.isdir(local_image_folder):
        return False, f"Image folder not found: {local_image_folder}"

    n_img = count_local_images(local_image_folder)
    if n_img <= 0:
        return False, f"No images found under: {local_image_folder}"

    weights_name = os.path.basename(weights_file or "best.pt")
    if local_weight_to_upload:
        local_w = os.path.abspath(local_weight_to_upload)
        if not os.path.isfile(local_w):
            return False, f"Local weight not found: {local_w}"
        weights_name = os.path.basename(local_w)
        remote_weights_dir = f"{run_dir}/weights"
        model_path = f"{remote_weights_dir}/{weights_name}"
        try:
            client.exec(f"mkdir -p {shlex.quote(run_dir)}")
            client.exec(f"mkdir -p {shlex.quote(remote_weights_dir)}")
            if progress:
                progress(f"Uploading weight {weights_name}…")
            client.upload_file(local_w, model_path)
        except SSHError as e:
            return False, str(e)
    elif model_path:
        model_path = model_path.replace("\\", "/")
    else:
        model_path = f"{run_dir}/weights/{weights_name}"
    if not client.exists(model_path):
        return False, f"Model weights not found: {model_path}"

    remote_images = f"{run_dir}/batch_infer/images"
    remote_labels = f"{run_dir}/batch_infer/labels"
    try:
        if progress:
            progress(f"Clearing remote {remote_images}")
        client.exec(
            f"rm -rf {shlex.quote(run_dir + '/batch_infer')} 2>/dev/null || true"
        )
        if progress:
            progress(f"Uploading {n_img} image(s)…")
        client.upload_directory(
            local_image_folder, remote_images, progress=progress
        )

        script = build_remote_batch_predict_script(
            model_path, remote_images, run_dir, imgsz=imgsz
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_batch_predict.py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(script)
            local_script = f.name
        remote_script = f"{run_dir}/_xal_batch_predict.py"
        try:
            client.upload_file(local_script, remote_script)
        finally:
            try:
                os.unlink(local_script)
            except OSError:
                pass

        if progress:
            progress("Running remote batch predict…")
        code, out, err = client.exec(
            format_exec_python(remote_python, remote_script),
            timeout=7200,
        )
        text = ((out or "") + "\n" + (err or "")).strip()
        if code != 0 or "BATCH_PREDICT_OK:" not in text:
            return False, text or f"Remote batch predict failed (code={code})"
        path = remote_labels
        for line in text.splitlines():
            if line.startswith("BATCH_PREDICT_OK:"):
                rest = line.split("BATCH_PREDICT_OK:", 1)[-1].strip()
                if " (" in rest:
                    rest = rest.split(" (", 1)[0].strip()
                path = rest or remote_labels
        return True, path
    except SSHError as e:
        return False, str(e)
    finally:
        try:
            client.exec(
                f"rm -f {shlex.quote(run_dir + '/_xal_batch_predict.py')} "
                f"2>/dev/null || true"
            )
        except Exception:
            pass

