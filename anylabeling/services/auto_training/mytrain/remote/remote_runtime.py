"""Remote Python / Conda runtime selection over SSH (Scheme B: under Remote mode)."""

from __future__ import annotations

import json
import os
import posixpath
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .ssh_client import SSHClient, SSHError


@dataclass
class RemoteCondaEnv:
    name: str
    prefix: str
    python: str

    @property
    def display(self) -> str:
        return f"{self.name}  ({self.python})"


def default_shell_python_cmd(remote_script: str) -> str:
    """Legacy Remote: remote login shell ``python3`` with ``python`` fallback."""
    s = shlex.quote(remote_script)
    return f"python3 -u {s} || python -u {s}"


def conda_python_cmd(remote_python: str, remote_script: str) -> str:
    """Run a script with the chosen env's Python (absolute path on remote host)."""
    py = shlex.quote(remote_python.strip())
    s = shlex.quote(remote_script)
    return f"{py} -u {s}"


def format_exec_python(
    remote_python: Optional[str], remote_script: str
) -> str:
    if remote_python and str(remote_python).strip():
        return conda_python_cmd(remote_python.strip(), remote_script)
    return default_shell_python_cmd(remote_script)


def format_nohup_train(
    remote_python: Optional[str], remote_script: str, remote_log: str
) -> str:
    """Background training: redirect stdout/stderr to remote_log, print PID."""
    log_q = shlex.quote(remote_log)
    script_q = shlex.quote(remote_script)
    if remote_python and str(remote_python).strip():
        py_q = shlex.quote(remote_python.strip())
        return (
            f"nohup {py_q} {script_q} > {log_q} 2>&1 < /dev/null & echo $!"
        )
    return (
        f"nohup python3 {script_q} > {log_q} 2>&1 < /dev/null & echo $!; "
        f"test $? -eq 0 || nohup python {script_q} > {log_q} 2>&1 "
        f"< /dev/null & echo $!"
    )


def tensorboard_launcher_candidates(
    remote_python: Optional[str],
) -> List[str]:
    if remote_python and str(remote_python).strip():
        py = shlex.quote(remote_python.strip())
        return [
            f"{py} -m tensorboard.main",
            f"{py} -m tensorboard",
            "tensorboard",
        ]
    return [
        "tensorboard",
        "python3 -m tensorboard.main",
        "python -m tensorboard.main",
        "python3 -m tensorboard",
        "python -m tensorboard",
    ]


def http_probe_python(remote_python: Optional[str], port: int) -> str:
    """One-liner to probe TensorBoard HTTP from inside the remote host."""
    py = "python3"
    if remote_python and str(remote_python).strip():
        py = shlex.quote(remote_python.strip())
    return (
        f"{py} -c \"import urllib.request; urllib.request.urlopen("
        f"'http://127.0.0.1:{int(port)}/', timeout=2).read(32); print('OK')\" "
        f"2>/dev/null || true"
    )


_COMMON_CONDA_REL = (
    "miniconda3/bin/conda",
    "anaconda3/bin/conda",
    "mambaforge/bin/conda",
    "miniforge3/bin/conda",
    "opt/conda/bin/conda",
)


def resolve_remote_conda_executable(
    client: SSHClient, user_path: str = ""
) -> Tuple[str, str]:
    """
    Return (conda_exe_path, detail). Empty conda_exe means not found.
    """
    home = (client.home or "").rstrip("/")
    tried: List[str] = []
    candidates: List[str] = []
    raw = (user_path or "").strip()
    if raw:
        candidates.append(raw)
    candidates.append("conda")
    for rel in _COMMON_CONDA_REL:
        if home:
            candidates.append(posixpath.join(home, rel))
    candidates.append("/opt/conda/bin/conda")

    seen = set()
    for exe in candidates:
        exe = exe.strip()
        if not exe or exe in seen:
            continue
        seen.add(exe)
        tried.append(exe)
        q = shlex.quote(exe)
        code, out, err = client.exec(
            f"sh -c 'command -v {q} >/dev/null 2>&1 && echo {q} || "
            f"test -x {q} && echo {q} || true'",
            timeout=25,
        )
        line = (out or "").strip().splitlines()
        if line and line[-1].strip():
            found = line[-1].strip()
            return found, f"Using conda: {found}"
        code2, out2, _ = client.exec(
            f"{q} --version 2>/dev/null && echo OK", timeout=20
        )
        if code2 == 0 and "OK" in (out2 or ""):
            return exe, f"Using conda: {exe}"

    detail = "Conda not found. Set conda path or install Miniconda/Anaconda."
    if tried:
        detail += " Tried: " + ", ".join(tried[:6])
    return "", detail


def list_remote_conda_envs(
    client: SSHClient, conda_exe: str
) -> Tuple[List[RemoteCondaEnv], str]:
    if not conda_exe:
        return [], "Conda executable is empty"
    q = shlex.quote(conda_exe)
    code, out, err = client.exec(f"{q} env list --json", timeout=90)
    text = (out or "") + (err or "")
    if code != 0:
        code2, out2, err2 = client.exec(
            f"{q} info --envs --json 2>/dev/null", timeout=90
        )
        text2 = (out2 or "") + (err2 or "")
        if code2 != 0:
            return [], (text2 or text).strip()[:500] or "conda env list failed"
        out = out2
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        return [], f"Invalid conda JSON: {e}\n{(out or '')[:300]}"

    root = str(data.get("root_prefix") or "").rstrip("/")
    envs: List[RemoteCondaEnv] = []
    for item in data.get("envs") or []:
        if isinstance(item, str):
            prefix = item.rstrip("/")
            if root and prefix == root:
                name = "base"
            else:
                name = posixpath.basename(prefix) or "env"
        elif isinstance(item, dict):
            prefix = str(item.get("prefix") or "").rstrip("/")
            name = str(item.get("name") or posixpath.basename(prefix) or "env")
        else:
            continue
        if not prefix:
            continue
        py = posixpath.join(prefix, "bin", "python")
        envs.append(RemoteCondaEnv(name=name, prefix=prefix, python=py))

    if not envs and root:
        envs.append(
            RemoteCondaEnv(
                name="base",
                prefix=root,
                python=posixpath.join(root, "bin", "python"),
            )
        )

    # Dedupe by python path
    dedup: Dict[str, RemoteCondaEnv] = {}
    for e in envs:
        dedup[e.python] = e
    result = sorted(dedup.values(), key=lambda x: x.name.lower())
    if not result:
        return [], "No conda environments returned"
    return result, f"Found {len(result)} environment(s)"


def probe_remote_python(
    client: SSHClient, remote_python: str, timeout: float = 60.0
) -> Tuple[bool, str, Dict[str, Any]]:
    """Check ultralytics/torch in a remote Python (same idea as local interpreter probe)."""
    import tempfile
    import time

    py = (remote_python or "").strip()
    if not py:
        return False, "Remote Python path is empty", {}

    code = r"""import json
import sys
info = {"python": "", "ultralytics": False, "torch": "", "cuda": False, "device_count": 0, "error": ""}
try:
    info["python"] = sys.version.split()[0]
    try:
        import ultralytics
        info["ultralytics"] = True
        info["ultralytics_version"] = getattr(ultralytics, "__version__", "")
    except Exception as e:
        info["error"] = "ultralytics: %s" % e
        print(json.dumps(info))
        raise SystemExit(2)
    try:
        import torch
        info["torch"] = str(torch.__version__)
        info["cuda"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count()) if info["cuda"] else 0
    except Exception as e:
        info["error"] = "torch: %s" % e
        print(json.dumps(info))
        raise SystemExit(3)
    print(json.dumps(info))
except SystemExit:
    raise
except Exception as e:
    info["error"] = str(e)
    print(json.dumps(info))
    raise SystemExit(1)
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_xal_probe.py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        local_script = f.name
    home = (client.home or "/tmp").rstrip("/")
    remote_script = f"{home}/.xanylabeling_probe_{int(time.time())}.py"
    qpy = shlex.quote(py)
    qscript = shlex.quote(remote_script)
    try:
        client.upload_file(local_script, remote_script)
        code_out, out, err = client.exec(
            f"{qpy} {qscript}",
            timeout=int(timeout) + 10,
        )
    except Exception as e:
        return False, f"SSH probe failed: {e}", {}
    finally:
        try:
            os.unlink(local_script)
        except OSError:
            pass
        try:
            client.exec(f"rm -f {shlex.quote(remote_script)} 2>/dev/null || true")
        except Exception:
            pass

    lines = [ln for ln in (out or "").splitlines() if ln.strip()]
    payload: Dict[str, Any] = {}
    if lines:
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            payload = {}
    if code_out != 0:
        msg = (
            payload.get("error")
            or (err or "").strip()
            or (out or "").strip()
            or "probe failed"
        )
        return False, str(msg)[:400], payload

    if not payload.get("ultralytics"):
        return False, payload.get("error") or "ultralytics not importable", payload

    py_v = payload.get("python") or "?"
    ultra_v = payload.get("ultralytics_version") or "ok"
    torch_v = payload.get("torch") or "?"
    cuda = payload.get("cuda")
    n = payload.get("device_count") or 0
    msg = (
        f"Python {py_v}, ultralytics {ultra_v}, "
        f"torch {torch_v}, cuda={cuda}, GPUs={n}"
    )
    return True, msg, payload


def python_for_conda_env(env: RemoteCondaEnv) -> str:
    return env.python