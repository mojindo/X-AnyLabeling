"""Custom command training: placeholder substitution and path helpers."""

from __future__ import annotations

import os
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple, Union

_BASIC_CLI_KEYS = ("model", "data", "project", "name", "device", "exist_ok")

CUSTOM_PLACEHOLDERS = (
    "model",
    "data",
    "project",
    "name",
    "run_dir",
    "device",
)


def format_device_for_placeholder(device: Any) -> str:
    if device is None:
        return ""
    if isinstance(device, (list, tuple)):
        return ",".join(str(x) for x in device)
    return str(device)


def build_path_context(
    *,
    model: str,
    data: str,
    project: str,
    name: str,
    device: Any,
) -> Dict[str, str]:
    run_dir = os.path.join(project or "", name or "")
    return {
        "model": model or "",
        "data": data or "",
        "project": project or "",
        "name": name or "",
        "run_dir": run_dir,
        "device": format_device_for_placeholder(device),
    }


def normalize_custom_shell_command(command: str) -> str:
    """Merge shell line continuations (trailing ``\\``) into one line.

    Windows ``cmd`` does not treat ``\\`` as a line continuation; Ultralytics
    CLI then receives ``\\`` as a stray argument. Users often paste multi-line
    ``yolo … \\`` blocks from docs — fold those into a single command string.
    """
    text = (command or "").strip()
    if not text:
        return ""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    statements: List[str] = []
    buf: List[str] = []
    for raw in lines:
        line = raw.rstrip()
        if not line.strip() and not buf:
            continue
        if line.endswith("\\"):
            buf.append(line[:-1].rstrip())
            continue
        if buf:
            if line.strip():
                buf.append(line.strip())
            statements.append(" ".join(s for s in buf if s))
            buf = []
        elif line.strip():
            statements.append(line.strip())
    if buf:
        statements.append(" ".join(s for s in buf if s))
    if not statements:
        return ""
    if len(statements) == 1:
        return " ".join(statements[0].split())
    return " && ".join(" ".join(s.split()) for s in statements)


def _quote_cli_value(value: str) -> str:
    value = value or ""
    if not value:
        return value
    if any(ch.isspace() for ch in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def enforce_basic_cli_overrides(command: str, paths: Dict[str, str]) -> str:
    """Strip model/data/project/name/device from CLI text; append Basic Settings."""
    cmd = normalize_custom_shell_command(command)
    pattern = re.compile(
        r"(?<!\S)("
        + "|".join(_BASIC_CLI_KEYS)
        + r")=(?:[^\s\"']+|\"[^\"]*\"|'[^']*')",
        re.IGNORECASE,
    )
    stripped = pattern.sub(" ", cmd)
    stripped = " ".join(stripped.split())
    tail: List[str] = []
    for key in ("model", "data", "project", "name", "device"):
        val = (paths.get(key) or "").strip()
        if val:
            tail.append(f"{key}={_quote_cli_value(val)}")
    tail.append("exist_ok=True")
    merged = (stripped + " " + " ".join(tail)).strip()
    return merged


def substitute_custom_command(template: str, paths: Dict[str, str]) -> str:
    """Replace {model}, {data}, … with effective training paths."""
    out = template or ""
    for key in CUSTOM_PLACEHOLDERS:
        out = out.replace("{" + key + "}", paths.get(key, ""))
    out = normalize_custom_shell_command(out)
    return enforce_basic_cli_overrides(out, paths)


def posix_if_needed(path: str, *, remote: bool) -> str:
    if not path:
        return path
    if remote:
        return path.replace("\\", "/")
    return path


def resolve_env_yolo_cli(python_executable: str) -> str:
    """Return ``yolo``/``yolo.exe`` shipped in the same venv/conda as ``python_executable``."""
    py = os.path.abspath((python_executable or "").strip())
    if not py or not os.path.isfile(py):
        return ""
    base = os.path.dirname(py)
    for name in ("yolo.exe", "yolo-script.py", "yolo"):
        for folder in ("Scripts", "bin", ""):
            if folder:
                candidate = os.path.join(base, folder, name)
            else:
                candidate = os.path.join(base, name)
            if os.path.isfile(candidate):
                return candidate
    return ""


def _split_yolo_args(tail: str) -> List[str]:
    try:
        return shlex.split(tail, posix=(os.name != "nt"))
    except ValueError:
        return tail.split()


def bind_command_to_python(command: str, python_executable: str) -> str:
    """Run ``yolo …`` with the selected interpreter env (not global PATH ``yolo``)."""
    py = (python_executable or "").strip()
    cmd = normalize_custom_shell_command(command)
    if not py or not os.path.isfile(py):
        return cmd
    stripped = cmd.strip()
    lower = stripped.lower()
    if not (lower.startswith("yolo ") or lower == "yolo"):
        return cmd
    tail = stripped[4:].lstrip() if len(stripped) > 4 else ""
    yolo_cli = resolve_env_yolo_cli(py)
    if yolo_cli:
        yq = f'"{yolo_cli}"' if os.name == "nt" else shlex.quote(yolo_cli)
        return f"{yq} {tail}".strip()
    py_q = f'"{py}"' if os.name == "nt" else shlex.quote(py)
    argv = repr(["yolo"] + _split_yolo_args(tail))
    # ultralytics has no __main__; invoke CLI entrypoint explicitly
    return (
        f"{py_q} -c "
        f"\"import sys; from ultralytics.cfg import entrypoint; "
        f"sys.argv = {argv}; entrypoint()\""
    )


def interpreter_path_env(python_executable: str) -> Dict[str, str]:
    """PATH prefix so conda/venv Scripts resolve when shell still calls ``yolo``."""
    env = os.environ.copy()
    py = (python_executable or "").strip()
    if not py or not os.path.isfile(py):
        return env
    py_dir = os.path.dirname(os.path.abspath(py))
    extra = [py_dir]
    for scripts in (
        os.path.join(py_dir, "Scripts"),
        os.path.join(os.path.dirname(py_dir), "Scripts"),
    ):
        if os.path.isdir(scripts):
            extra.append(scripts)
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def wrap_local_shell_command(command: str) -> Union[str, List[str]]:
    """argv or shell string for Popen on local/interpreter runs."""
    command = (command or "").strip()
    if not command:
        raise ValueError("Empty command")
    if os.name == "nt":
        return command
    return ["bash", "-lc", command]


def _cli_train_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return str(value)


def build_ultralytics_train_cli(
    train_args: Dict, *, task: str = "detect"
) -> str:
    """Build a ``yolo <task> train key=val ...`` line (same as manual CLI)."""
    args = dict(train_args or {})
    task_name = (task or args.pop("_task", None) or "detect").strip().lower()
    parts = ["yolo", task_name, "train"]
    for key, value in args.items():
        if key.startswith("_"):
            continue
        if value is None:
            continue
        parts.append(f"{key}={_cli_train_value(value)}")
    return " ".join(parts)


def parse_epochs_from_cli(command: str) -> int:
    m = re.search(r"(?:^|\s)epochs=(\d+)", command or "", re.IGNORECASE)
    if m:
        try:
            return max(0, int(m.group(1)))
        except ValueError:
            pass
    return 0


_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_EPOCH_HEAD = re.compile(r"^\s*(\d+)/(\d+)\b")


def collapse_in_epoch_tqdm_lines(text: str) -> str:
    """Fold non-TTY tqdm rows: one live line per train/val bar within an epoch."""
    if not (text or "").strip():
        return text or ""
    lines = text.split("\n")
    out: List[str] = []
    last_key = None
    cur_epoch = 0

    def _is_ultralytics_train_row(line: str) -> Optional[Tuple[int, int]]:
        m = _EPOCH_HEAD.match(line)
        if not m:
            return None
        ep, total = int(m.group(1)), int(m.group(2))
        if ep < 1 or ep > total or total > 50_000:
            return None
        # Val batch rows look like "1/8" but never carry GPU mem or imgsz column.
        if re.search(r"\d+\.\d+G", line) or re.search(r"\b640:\s*", line):
            return ep, total
        return None

    def train_bar_key(line: str):
        pair = _is_ultralytics_train_row(line)
        if not pair:
            return None
        ep, total = pair
        return ("train", ep, total)

    def val_bar_key(line: str):
        if _is_ultralytics_train_row(line):
            return None
        if "%" not in line:
            return None
        has_bar = "|" in line or re.search(r"\d+/\d+\s*(\[|<)", line)
        if not has_bar:
            return None
        if re.search(r"\b640:\s*", line) or re.search(r"\d+\.\d+G", line):
            return None
        ep = cur_epoch or 0
        return ("val", ep)

    for line in lines:
        pair = _is_ultralytics_train_row(line)
        if pair:
            cur_epoch = pair[0]

        key = train_bar_key(line) or val_bar_key(line)
        if key is not None:
            if out and last_key == key:
                out[-1] = line
            else:
                out.append(line)
            last_key = key
        else:
            out.append(line)
            last_key = None
    return "\n".join(out)


def format_train_log_terminal(content: str) -> str:
    """Mirror terminal text from a captured train log (ANSI + \\r only)."""
    text = _ANSI_ESCAPE.sub("", content or "")
    if not text.strip():
        return ""
    lines = []
    for raw in text.split("\n"):
        if "\r" in raw:
            raw = raw.split("\r")[-1]
        lines.append(raw)
    return collapse_in_epoch_tqdm_lines("\n".join(lines))


def latest_log_session_text(content: str) -> str:
    """Keep only the last run block from an append-style train_run.log."""
    text = content or ""
    if not text.strip():
        return ""
    markers = ("=== Custom training", "=== Training")
    best = -1
    for marker in markers:
        idx = text.rfind(marker)
        if idx > best:
            best = idx
    if best >= 0:
        return text[best:]
    return text


def build_remote_custom_launch(
    command: str,
    *,
    remote_log: str,
    exec_cwd: str = "",
) -> str:
    """Single remote shell line: optional cd, then user command, via nohup."""
    command = (command or "").strip()
    if not command:
        raise ValueError("Empty command")
    inner_parts: List[str] = []
    cwd = (exec_cwd or "").strip()
    if cwd:
        inner_parts.append(f"cd {shlex.quote(cwd)}")
    inner_parts.append(command)
    inner = " && ".join(inner_parts)
    log_q = shlex.quote(remote_log.replace("\\", "/"))
    return f"nohup bash -lc {shlex.quote(inner)} > {log_q} 2>&1 & echo $!"