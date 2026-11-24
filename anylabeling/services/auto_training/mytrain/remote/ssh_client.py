"""SSH / SFTP client wrapper (password auth via paramiko)."""

from __future__ import annotations

import os
import select
import socket
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple


class SSHError(Exception):
    pass


class SSHClient:
    def __init__(self):
        self._client = None
        self._sftp = None
        self.host = ""
        self.port = 22
        self.username = ""
        self.password = ""
        self.home = ""
        self._tunnels: List[dict] = []

    @property
    def connected(self) -> bool:
        """True only when an SSH transport is still alive (not just object present)."""
        return self.is_alive()

    def is_alive(self) -> bool:
        if self._client is None:
            return False
        try:
            transport = self._client.get_transport()
            return bool(transport is not None and transport.is_active())
        except Exception:
            return False

    def connect(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        timeout: float = 20.0,
    ) -> None:
        try:
            import paramiko
        except ImportError as e:
            import sys

            raise SSHError(
                "paramiko is not installed in the Python running X-AnyLabeling "
                f"({sys.executable}). Install with: "
                f'"{sys.executable}" -m pip install "paramiko>=3.4"'
            ) from e

        self.disconnect()
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=host.strip(),
                port=int(port),
                username=username.strip(),
                password=password,
                timeout=timeout,
                banner_timeout=timeout,
                auth_timeout=timeout,
                allow_agent=False,
                look_for_keys=False,
            )
        except Exception as e:
            raise SSHError(f"SSH connect failed: {e}") from e

        self._client = client
        self.host = host.strip()
        self.port = int(port)
        self.username = username.strip()
        self.password = password or ""
        # Keep idle sessions from being dropped silently by middleboxes / servers
        try:
            transport = client.get_transport()
            if transport is not None:
                transport.set_keepalive(30)
        except Exception:
            pass

        code, out, err = self.exec("echo $HOME", timeout=10)
        if code != 0:
            self.disconnect()
            raise SSHError(f"Failed to resolve remote $HOME: {err or out}")
        self.home = (out or "").strip().splitlines()[-1].strip()
        if not self.home:
            self.disconnect()
            raise SSHError("Remote $HOME is empty")

    def ensure_connected(self, *, timeout: float = 20.0) -> None:
        """No-op if alive; otherwise reconnect with stored credentials."""
        if self.is_alive():
            return
        if not self.host or not self.username:
            raise SSHError(
                "SSH session dropped and no credentials cached; "
                "please Test Connection again."
            )
        self.connect(
            self.host,
            self.port or 22,
            self.username,
            self.password,
            timeout=timeout,
        )

    def disconnect(self) -> None:
        self.stop_all_tunnels()
        if self._sftp is not None:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        # Keep host/user/password so ensure_connected can retry

    def stop_all_tunnels(self) -> None:
        for t in list(self._tunnels):
            t["stop"].set()
            try:
                t["sock"].close()
            except Exception:
                pass
        self._tunnels.clear()

    def open_local_forward(
        self,
        remote_port: int,
        *,
        remote_host: str = "127.0.0.1",
        local_port: int = 0,
    ) -> int:
        """
        Forward local TCP port -> remote_host:remote_port via SSH.

        Used so TensorBoard works without publishing Docker ports to the LAN.
        Returns the bound local port (open http://127.0.0.1:<port>).
        """
        self._require()
        transport = self._client.get_transport()
        if transport is None or not transport.is_active():
            raise SSHError("SSH transport is not active")

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", int(local_port)))
        server.listen(32)
        bound_port = server.getsockname()[1]
        stop_event = threading.Event()

        def _pipe(src, dst):
            try:
                while not stop_event.is_set():
                    r, _, _ = select.select([src], [], [], 1.0)
                    if not r:
                        continue
                    data = src.recv(32768)
                    if not data:
                        break
                    dst.sendall(data)
            except Exception:
                pass
            finally:
                try:
                    src.close()
                except Exception:
                    pass
                try:
                    dst.close()
                except Exception:
                    pass

        def _accept_loop():
            while not stop_event.is_set():
                try:
                    server.settimeout(1.0)
                    try:
                        client_sock, peer = server.accept()
                    except socket.timeout:
                        continue
                    try:
                        channel = transport.open_channel(
                            "direct-tcpip",
                            (remote_host, int(remote_port)),
                            peer,
                        )
                    except Exception:
                        client_sock.close()
                        continue
                    threading.Thread(
                        target=_pipe, args=(client_sock, channel), daemon=True
                    ).start()
                    threading.Thread(
                        target=_pipe, args=(channel, client_sock), daemon=True
                    ).start()
                except Exception:
                    if stop_event.is_set():
                        break

        thread = threading.Thread(target=_accept_loop, daemon=True)
        thread.start()
        self._tunnels.append(
            {
                "sock": server,
                "stop": stop_event,
                "thread": thread,
                "local_port": bound_port,
                "remote_port": int(remote_port),
            }
        )
        return bound_port

    def _require(self):
        if not self.is_alive():
            # Attempt one silent reconnect before failing the caller
            try:
                self.ensure_connected()
            except SSHError:
                raise SSHError(
                    "SSH session is not active. Reconnect via Test Connection."
                )
        if self._client is None:
            raise SSHError("Not connected to remote server")

    def sftp(self):
        self._require()
        if self._sftp is not None:
            try:
                # Probe whether the cached SFTP handle is still usable
                self._sftp.listdir(".")
            except Exception:
                try:
                    self._sftp.close()
                except Exception:
                    pass
                self._sftp = None
        if self._sftp is None:
            self._sftp = self._client.open_sftp()
        return self._sftp

    def exec(
        self,
        command: str,
        *,
        timeout: Optional[float] = None,
        get_pty: bool = False,
    ) -> Tuple[int, str, str]:
        self._require()
        try:
            stdin, stdout, stderr = self._client.exec_command(
                command, timeout=timeout, get_pty=get_pty
            )
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            code = stdout.channel.recv_exit_status()
            return code, out, err
        except Exception as e:
            # One reconnect + retry for dropped mid-session connections
            msg = str(e).lower()
            if "not active" in msg or "session" in msg or "closed" in msg:
                try:
                    self.ensure_connected()
                    stdin, stdout, stderr = self._client.exec_command(
                        command, timeout=timeout, get_pty=get_pty
                    )
                    out = stdout.read().decode("utf-8", errors="replace")
                    err = stderr.read().decode("utf-8", errors="replace")
                    code = stdout.channel.recv_exit_status()
                    return code, out, err
                except Exception as e2:
                    raise SSHError(f"Remote exec failed: {e2}") from e2
            raise SSHError(f"Remote exec failed: {e}") from e

    def exec_streaming(
        self,
        command: str,
        on_line: Callable[[str], None],
        *,
        get_pty: bool = True,
    ) -> int:
        self._require()
        try:
            stdin, stdout, stderr = self._client.exec_command(
                command, get_pty=get_pty
            )
            channel = stdout.channel
            while True:
                if channel.recv_ready():
                    chunk = channel.recv(4096).decode(
                        "utf-8", errors="replace"
                    )
                    for line in chunk.splitlines():
                        if line.strip():
                            on_line(line.rstrip())
                if channel.exit_status_ready() and not channel.recv_ready():
                    break
            while channel.recv_ready():
                chunk = channel.recv(4096).decode("utf-8", errors="replace")
                for line in chunk.splitlines():
                    if line.strip():
                        on_line(line.rstrip())
            return channel.recv_exit_status()
        except Exception as e:
            raise SSHError(f"Remote streaming exec failed: {e}") from e

    def mkdir_p(self, remote_dir: str) -> None:
        sftp = self.sftp()
        parts = remote_dir.strip("/").split("/")
        cur = ""
        for part in parts:
            if not part:
                continue
            cur = (
                f"{cur}/{part}"
                if cur or remote_dir.startswith("/")
                else part
            )
            if remote_dir.startswith("/") and not cur.startswith("/"):
                cur = "/" + cur
            try:
                sftp.stat(cur)
            except IOError:
                sftp.mkdir(cur)

    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        *,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        tick: Optional[Callable[[], None]] = None,
    ) -> None:
        remote_dir = os.path.dirname(remote_path.replace("\\", "/"))
        if remote_dir:
            self.mkdir_p(remote_dir)

        def _cb(done: int, total: int) -> None:
            if tick is not None:
                tick()
            if progress_callback is not None:
                progress_callback(done, total)

        sftp = self.sftp()
        if progress_callback is not None or tick is not None:
            sftp.put(local_path, remote_path, callback=_cb)
        else:
            sftp.put(local_path, remote_path)

    def download_file(self, remote_path: str, local_path: str) -> None:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        self.sftp().get(remote_path, local_path)

    def download_directory(
        self,
        remote_dir: str,
        local_dir: str,
        progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Recursively download a remote directory tree via SFTP."""
        import stat as _stat

        remote_dir = remote_dir.rstrip("/").replace("\\", "/")
        local_dir = str(Path(local_dir))
        os.makedirs(local_dir, exist_ok=True)
        sftp = self.sftp()

        def _walk(rpath: str, lpath: str) -> None:
            try:
                entries = sftp.listdir_attr(rpath)
            except IOError as e:
                raise SSHError(f"Cannot list remote directory {rpath}: {e}") from e
            for attr in entries:
                name = attr.filename
                if name in (".", ".."):
                    continue
                rp = f"{rpath}/{name}"
                lp = os.path.join(lpath, name)
                if _stat.S_ISDIR(attr.st_mode):
                    os.makedirs(lp, exist_ok=True)
                    _walk(rp, lp)
                else:
                    if progress:
                        progress(rp)
                    os.makedirs(os.path.dirname(lp) or ".", exist_ok=True)
                    sftp.get(rp, lp)

        _walk(remote_dir, local_dir)

    def upload_directory(
        self,
        local_dir: str,
        remote_dir: str,
        progress: Optional[Callable[[str], None]] = None,
        control: Optional[object] = None,
    ) -> None:
        local_dir = str(Path(local_dir).resolve())
        self.mkdir_p(remote_dir)
        sftp = self.sftp()

        def _check() -> None:
            checkpoint = getattr(control, "checkpoint", None)
            if checkpoint is not None:
                checkpoint()

        for root, dirs, files in os.walk(local_dir):
            _check()
            rel = os.path.relpath(root, local_dir)
            if rel == ".":
                rdir = remote_dir
            else:
                rdir = f"{remote_dir}/{rel.replace(os.sep, '/')}"
            self.mkdir_p(rdir)
            for d in dirs:
                self.mkdir_p(f"{rdir}/{d}")
            for name in files:
                _check()
                lp = os.path.join(root, name)
                rp = f"{rdir}/{name}"
                if progress:
                    progress(rp)

                def _during_put(_done, _total, _check=_check):
                    _check()

                sftp.put(lp, rp, callback=_during_put)

    def read_text(self, remote_path: str, max_bytes: int = 200_000) -> str:
        sftp = self.sftp()
        with sftp.open(remote_path, "r") as f:
            data = f.read(max_bytes)
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return str(data)

    def file_size(self, remote_path: str) -> int:
        """Return remote file size in bytes, or -1 if missing."""
        try:
            return int(self.sftp().stat(remote_path).st_size)
        except IOError:
            return -1

    def read_bytes_from(
        self, remote_path: str, offset: int = 0, max_bytes: int = 512_000
    ) -> bytes:
        """Read up to ``max_bytes`` starting at byte ``offset`` (binary)."""
        offset = max(0, int(offset or 0))
        max_bytes = max(1, int(max_bytes or 1))
        sftp = self.sftp()
        with sftp.open(remote_path, "rb") as f:
            if offset:
                f.seek(offset)
            data = f.read(max_bytes)
        return data if isinstance(data, (bytes, bytearray)) else bytes(data or b"")

    def listdir(self, remote_path: str) -> List[str]:
        try:
            return list(self.sftp().listdir(remote_path))
        except IOError:
            return []

    def exists(self, remote_path: str) -> bool:
        try:
            self.sftp().stat(remote_path)
            return True
        except IOError:
            return False

    def is_dir(self, remote_path: str) -> bool:
        try:
            import stat as statmod

            mode = self.sftp().stat(remote_path).st_mode
            return statmod.S_ISDIR(mode)
        except IOError:
            return False
