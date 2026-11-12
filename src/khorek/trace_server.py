"""A tiny local web UI for browsing khorek run traces.

The server is deliberately dependency-free: Python's stdlib HTTP server plus
three static assets shipped in ``khorek/web``.  It binds to localhost by
default because traces contain full prompts, source code, and model output.

Traced workflows start the viewer in the background so a run can be watched
live.  The child process is detached and records its address in
``<trace-root>/server.pid``; ``khorek traces --stop`` uses that file to shut it
down after the workflow has exited.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from khorek.trace import TraceStore

_WEB_DIR = Path(__file__).resolve().parent / "web"
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}
SERVER_INFO_NAME = "server.pid"


@dataclass(frozen=True)
class ServerInfo:
    """The address and process of a running trace viewer."""

    pid: int
    host: str
    port: int
    url: str
    root: Path


def server_info_path(root: Path) -> Path:
    """Return the pid/address file for a trace root."""
    return Path(root) / SERVER_INFO_NAME


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_info_file(root: Path) -> ServerInfo | None:
    """Parse ``server.pid`` without checking whether the process is alive."""
    try:
        raw = server_info_path(root).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return ServerInfo(
            pid=int(data["pid"]),
            host=str(data["host"]),
            port=int(data["port"]),
            url=str(data["url"]),
            root=Path(data.get("root", root)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def read_server_info(root: Path) -> ServerInfo | None:
    """Return live viewer info for *root*, or ``None`` for a stale/missing file."""
    info = _read_info_file(root)
    if info is None or not _pid_alive(info.pid):
        return None
    return info


def _write_server_info(root: Path, info: ServerInfo) -> None:
    path = server_info_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "pid": info.pid,
                "host": info.host,
                "port": info.port,
                "url": info.url,
                "root": str(info.root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _remove_server_info(root: Path, pid: int | None = None) -> None:
    """Remove ``server.pid`` when it names *pid* (or unconditionally)."""
    path = server_info_path(root)
    if pid is not None:
        info = _read_info_file(root)
        if info is not None and info.pid != pid:
            return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _probe(url: str, root: Path, *, timeout: float = 1.0) -> bool:
    """Return ``True`` when *url* serves the trace root *root*."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/runs", timeout=timeout) as response:
            payload = json.loads(response.read())
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    try:
        served = Path(str(payload.get("root"))).resolve()
    except OSError:
        return False
    return served == Path(root).resolve()


def _looks_like_viewer(pid: int) -> bool | None:
    """Best-effort guard against signalling a recycled PID.

    ``None`` means the command line could not be inspected (for example ``ps``
    is missing or restricted), so the caller must fall back to probing the
    recorded URL.
    """
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    command = result.stdout.lower()
    return "khorek" in command and "traces" in command


def _spawn(
    root: Path,
    *,
    host: str,
    port: int,
    open_browser: bool,
) -> subprocess.Popen[bytes] | None:
    command = [
        sys.executable,
        "-m",
        "khorek",
        "traces",
        "--trace-dir",
        str(root),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if not open_browser:
        command.append("--no-open")
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        print(f"khorek: cannot start trace viewer: {exc}", file=sys.stderr)
        return None


def _wait_for_server(
    process: subprocess.Popen[bytes],
    root: Path,
    *,
    wait: float,
) -> ServerInfo | None:
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        info = read_server_info(root)
        if info is not None and info.pid == process.pid and _probe(info.url, root):
            return info
        if process.poll() is not None:
            return None
        time.sleep(0.05)
    return None


def start_viewer(
    root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    wait: float = 10.0,
) -> ServerInfo | None:
    """Start a detached trace viewer, reusing one that already serves *root*.

    Returns ``None`` when the viewer cannot be started; a diagnostic viewer
    must never fail a workflow.  If *port* is busy, one retry binds an
    ephemeral port, and the returned :class:`ServerInfo` carries the actual
    address.
    """
    root = Path(root).resolve()
    existing = read_server_info(root)
    if existing is not None and _probe(existing.url, root):
        return existing

    for attempt_port in (port, 0):
        process = _spawn(root, host=host, port=attempt_port, open_browser=open_browser)
        if process is None:
            return None
        info = _wait_for_server(process, root, wait=wait)
        if info is not None:
            return info
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
        if attempt_port == port:
            # The preferred port may be taken; retry with an ephemeral one.
            continue
        break
    print("khorek: trace viewer did not start; continuing without it", file=sys.stderr)
    return None


def stop_viewer(root: Path, *, wait: float = 5.0) -> ServerInfo | None:
    """Stop the viewer serving *root*; returns its info, or ``None`` if none."""
    root = Path(root).resolve()
    info = read_server_info(root)
    if info is None:
        _remove_server_info(root)
        return None
    looks = _looks_like_viewer(info.pid)
    if looks is not True and not _probe(info.url, root):
        print(
            f"khorek: refusing to stop pid {info.pid}: cannot verify it is "
            "a khorek traces process for this trace directory",
            file=sys.stderr,
        )
        return None
    try:
        os.kill(info.pid, signal.SIGTERM)
    except ProcessLookupError:
        _remove_server_info(root, info.pid)
        return None
    except OSError as exc:
        print(f"khorek: cannot stop trace viewer (pid {info.pid}): {exc}", file=sys.stderr)
        return None

    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if not _pid_alive(info.pid):
            _remove_server_info(root, info.pid)
            return info
        time.sleep(0.05)
    try:
        os.kill(info.pid, signal.SIGKILL)
    except OSError:
        pass
    _remove_server_info(root, info.pid)
    return info


def _read_asset(name: str) -> bytes:
    return (_WEB_DIR / name).read_bytes()


def _json_response(handler: BaseHTTPRequestHandler, payload: object, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    _json_response(handler, {"error": message}, status=status)


def make_handler(store: TraceStore) -> type[BaseHTTPRequestHandler]:
    class TraceHandler(BaseHTTPRequestHandler):
        server_version = "khorek-traces/1"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            # Keep the terminal quiet unless explicitly requested.
            if getattr(self.server, "verbose", False):
                super().log_message(format, *args)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            parts = urlsplit(self.path)
            path = parts.path
            query = parse_qs(parts.query)

            if path == "/":
                self._asset("index.html")
                return
            if path in {"/app.js", "/app.css"}:
                self._asset(path.lstrip("/"))
                return
            if path == "/api/runs":
                _json_response(
                    self,
                    {
                        "root": str(store.root),
                        "runs": [run.as_dict() for run in store.list_runs()],
                    },
                )
                return
            if path.startswith("/api/runs/"):
                self._run_api(path, query)
                return
            _error(self, HTTPStatus.NOT_FOUND, f"not found: {path}")

        def _asset(self, name: str) -> None:
            try:
                body = _read_asset(name)
            except OSError:
                _error(self, HTTPStatus.NOT_FOUND, f"missing asset: {name}")
                return
            content_type = _CONTENT_TYPES.get(Path(name).suffix, "application/octet-stream")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _run_api(self, path: str, query: dict[str, list[str]]) -> None:
            segments = [segment for segment in path.split("/") if segment]
            # ["api", "runs", "<id>", ...]
            if len(segments) < 3:
                _error(self, HTTPStatus.NOT_FOUND, "missing run id")
                return
            run_id = segments[2]
            if store.run_dir(run_id) is None:
                _error(self, HTTPStatus.NOT_FOUND, f"unknown run: {run_id}")
                return
            if len(segments) == 3:
                for run in store.list_runs():
                    if run.id == run_id:
                        _json_response(self, run.as_dict())
                        return
                _error(self, HTTPStatus.NOT_FOUND, f"unknown run: {run_id}")
                return
            if segments[3] == "events":
                if len(segments) >= 5 and segments[4].isdigit():
                    seq = int(segments[4])
                    for event in store.events(run_id):
                        if int(event.get("seq", 0)) == seq:
                            _json_response(self, event)
                            return
                    _error(self, HTTPStatus.NOT_FOUND, f"unknown event: {seq}")
                    return
                since_raw = query.get("since", ["0"])[0]
                try:
                    since = int(since_raw)
                except ValueError:
                    since = 0
                events = store.events_since(run_id, since) if since else store.events(run_id)
                _json_response(
                    self,
                    {"run_id": run_id, "since": since, "events": events},
                )
                return
            if segments[3] == "raw":
                directory = store.run_dir(run_id)
                assert directory is not None
                try:
                    body = (directory / "events.jsonl").read_bytes()
                except OSError:
                    _error(self, HTTPStatus.NOT_FOUND, "no events file")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{run_id}.events.jsonl"'
                )
                self.end_headers()
                self.wfile.write(body)
                return
            _error(self, HTTPStatus.NOT_FOUND, f"not found: {path}")

    return TraceHandler


class TraceServer(ThreadingHTTPServer):
    """Threading HTTP server carrying the trace store and a verbosity flag."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], store: TraceStore, *, verbose: bool = False):
        super().__init__(address, make_handler(store))
        self.store = store
        self.verbose = verbose


def serve(
    root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    verbose: bool = False,
) -> None:
    """Run the trace viewer until interrupted."""
    store = TraceStore(root)
    try:
        server = TraceServer((host, port), store, verbose=verbose)
    except OSError as exc:
        print(f"khorek: cannot start trace viewer on {host}:{port}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    actual_host = str(server.server_address[0])
    actual_port = int(server.server_address[1])
    url = f"http://{actual_host}:{actual_port}/"
    info = ServerInfo(
        pid=os.getpid(),
        host=actual_host,
        port=actual_port,
        url=url,
        root=Path(root),
    )
    try:
        _write_server_info(Path(root), info)
    except OSError as exc:
        print(f"khorek: cannot write trace viewer state: {exc}", file=sys.stderr)
    print(f"khorek traces: {url} ({root})", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except OSError:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _remove_server_info(Path(root), os.getpid())


__all__ = [
    "SERVER_INFO_NAME",
    "ServerInfo",
    "TraceServer",
    "make_handler",
    "read_server_info",
    "serve",
    "server_info_path",
    "start_viewer",
    "stop_viewer",
]
