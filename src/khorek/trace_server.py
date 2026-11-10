"""A tiny local web UI for browsing khorek run traces.

The server is deliberately dependency-free: Python's stdlib HTTP server plus
three static assets shipped in ``khorek/web``.  It binds to localhost by
default because traces contain full prompts, source code, and model output.
"""

from __future__ import annotations

import json
import sys
import webbrowser
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
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/"
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


__all__ = ["TraceServer", "make_handler", "serve"]
