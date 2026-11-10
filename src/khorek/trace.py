"""Append-only run traces and a tiny reader for the trace viewer.

Every workflow run can record exactly what the model saw and returned: the
full request payload (system/user messages, tool definitions, sampling
settings), the raw response (including reasoning/thinking and tool calls), the
tool results, repairs, and agent errors.  Traces are written as JSON Lines so a
crashed or timed-out run still leaves everything recorded up to that point.

Tracing is diagnostic, never part of the decision path: any I/O problem while
writing a trace is reported once and then ignored so it cannot fail a run.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from khorek.config import ObserveConfig

TRACE_ENV = "KHOREK_TRACE"
TRACE_DIR_ENV = "KHOREK_TRACE_DIR"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _json_default(value: object) -> str:
    if isinstance(value, (Path, bytes)):
        return str(value)
    if isinstance(value, Exception):
        return f"{type(value).__name__}: {value}"
    return repr(value)


def _jsonable(value: Any) -> Any:
    """Force *value* through JSON so the event line can never fail to encode."""
    try:
        json.dumps(value, ensure_ascii=False, default=_json_default)
    except (TypeError, ValueError):
        return _json_default(value)
    return value


def _parse_bool(raw: str) -> bool | None:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def trace_dir_for(db_path: Path) -> Path:
    """Default trace root for a database: ``<db>.traces`` next to the DB."""
    return db_path.with_name(db_path.name + ".traces")


def _enabled_from_env(default: bool) -> bool:
    raw = os.environ.get(TRACE_ENV)
    if raw is None:
        return default
    parsed = _parse_bool(raw)
    return default if parsed is None else parsed


def _directory_from_env(default: Path) -> Path:
    raw = os.environ.get(TRACE_DIR_ENV)
    if raw is None or not raw.strip():
        return default
    return Path(raw).expanduser().resolve()


def _thinking_of(message: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("thinking", "reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    return "".join(parts)


def message_summary(message: dict[str, Any]) -> dict[str, Any]:
    """A response message reduced to the parts worth showing in a timeline."""
    tool_calls = []
    raw_calls = message.get("tool_calls")
    if isinstance(raw_calls, list):
        for raw in raw_calls:
            if not isinstance(raw, dict):
                continue
            function = raw.get("function")
            if not isinstance(function, dict):
                continue
            tool_calls.append(
                {
                    "id": raw.get("id"),
                    "name": function.get("name"),
                    "arguments": function.get("arguments"),
                }
            )
    return {
        "content": message.get("content"),
        "thinking": _thinking_of(message),
        "tool_calls": tool_calls,
    }


class TraceRecorder:
    """Writes one run's trace to ``<trace_root>/<run_id>/``."""

    @classmethod
    def from_config(
        cls,
        observe: ObserveConfig,
        *,
        db_path: Path,
        run: dict[str, Any] | None = None,
        directory: str | Path | None = None,
        enabled: bool | None = None,
        keep_runs: int | None = None,
    ) -> TraceRecorder | None:
        """Build a recorder from ``[observe]`` config plus CLI/env overrides.

        Returns ``None`` when tracing is disabled or the trace directory cannot
        be created; a trace problem must never fail a real run.
        """
        if enabled is None:
            enabled = _enabled_from_env(observe.enabled)
        if not enabled:
            return None
        try:
            if directory is not None:
                resolved = Path(directory).expanduser().resolve()
            elif observe.directory.strip():
                resolved = _directory_from_env(Path(observe.directory).expanduser().resolve())
            else:
                resolved = _directory_from_env(trace_dir_for(db_path).resolve())
            return cls(
                resolved,
                run=run,
                keep_runs=observe.keep_runs if keep_runs is None else keep_runs,
            )
        except OSError as exc:
            print(f"khorek: trace disabled: {exc}", file=sys.stderr)
            return None

    def __init__(
        self,
        trace_root: Path,
        *,
        run: dict[str, Any] | None = None,
        keep_runs: int = 30,
    ) -> None:
        self.trace_root = trace_root
        self.keep_runs = keep_runs
        self._lock = threading.Lock()
        self._closed = False
        self._warned = False
        self._seq = 0

        trace_root.mkdir(parents=True, exist_ok=True)
        self._prune(keep=max(0, keep_runs - 1))
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}"
        self.run_id = run_id
        self.run_dir = trace_root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.meta_path = self.run_dir / "meta.json"
        self._started_monotonic = time.monotonic()
        self._meta: dict[str, Any] = {
            "id": run_id,
            "pid": os.getpid(),
            "started_at": _now(),
            "cwd": str(Path.cwd()),
            **(run or {}),
        }
        self._write_meta()
        try:
            self._events = self.events_path.open("a", encoding="utf-8")
        except OSError as exc:
            self._events = None
            self._warn(f"cannot open trace file {self.events_path}: {exc}")
        # A stable pointer to the newest run helps the CLI and humans.
        try:
            (trace_root / "latest").write_text(run_id + "\n", encoding="utf-8")
        except OSError:
            pass

    # -- lifecycle -----------------------------------------------------

    def _prune(self, *, keep: int) -> None:
        if keep <= 0:
            return
        try:
            runs = [
                path
                for path in self.trace_root.iterdir()
                if path.is_dir() and (path / "meta.json").is_file()
            ]
        except OSError:
            return
        runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for stale in runs[keep:]:
            shutil.rmtree(stale, ignore_errors=True)

    def _warn(self, text: str) -> None:
        if self._warned:
            return
        self._warned = True
        print(f"khorek: trace disabled: {text}", file=sys.stderr)

    def _write_meta(self) -> None:
        try:
            temporary = self.meta_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(self._meta, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
            temporary.replace(self.meta_path)
        except OSError as exc:
            self._warn(f"cannot write trace metadata: {exc}")

    def close(
        self,
        *,
        exit_code: int | None = None,
        status: str | None = None,
    ) -> None:
        if self._closed:
            return
        self.record(
            "run_end",
            exit_code=exit_code,
            status=status,
            duration_ms=round((time.monotonic() - self._started_monotonic) * 1000),
        )
        self._closed = True
        self._meta["ended_at"] = _now()
        self._meta["exit_code"] = exit_code
        self._meta["status"] = status
        self._meta["duration_ms"] = round((time.monotonic() - self._started_monotonic) * 1000)
        self._write_meta()
        with self._lock:
            handle = getattr(self, "_events", None)
            if handle is not None:
                try:
                    handle.flush()
                    handle.close()
                except OSError:
                    pass
            self._events = None

    # -- writing -------------------------------------------------------

    def record(self, kind: str, **fields: Any) -> dict[str, Any] | None:
        """Append one event; returns it (or ``None`` when tracing is closed)."""
        if self._closed:
            return None
        with self._lock:
            handle = getattr(self, "_events", None)
            if handle is None:
                return None
            self._seq += 1
            event = {
                "seq": self._seq,
                "ts": _now(),
                "kind": kind,
                **{key: _jsonable(value) for key, value in fields.items()},
            }
            try:
                handle.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")
                handle.flush()
            except OSError as exc:
                self._warn(f"cannot write trace event: {exc}")
                return None
            return event

    # -- conveniences --------------------------------------------------

    def model_call(
        self,
        *,
        agent: str,
        step: int,
        request: dict[str, Any],
        response: dict[str, Any] | str | None = None,
        message: dict[str, Any] | None = None,
        status: int | None = None,
        duration_ms: int | None = None,
        error: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        finish_reason: str | None = None,
    ) -> None:
        response_text = response if isinstance(response, str) else None
        response_object = response if isinstance(response, dict) else None
        summary = message_summary(message or {})
        self.record(
            "model_call",
            agent=agent,
            step=step,
            request=request,
            request_bytes=len(
                json.dumps(request, ensure_ascii=False, default=_json_default).encode("utf-8")
            ),
            response=response_object if response_object is not None else response_text,
            response_bytes=(
                len(response_text.encode("utf-8")) if response_text is not None else None
            ),
            message=message,
            content=summary["content"],
            thinking=summary["thinking"],
            tool_calls=summary["tool_calls"],
            status=status,
            finish_reason=finish_reason,
            usage=usage,
            duration_ms=duration_ms,
            error=error,
        )

    def tool_call(
        self,
        *,
        agent: str,
        step: int,
        call: dict[str, Any],
        result: Any = None,
        error: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self.record(
            "tool_call",
            agent=agent,
            step=step,
            call=call,
            result=result,
            error=error,
            duration_ms=duration_ms,
        )

    def agent_start(
        self,
        *,
        agent: str,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        max_tool_calls: int,
        force_tool_call: bool,
        text_is_final: bool,
        terminal_tool: str | None,
    ) -> None:
        self.record(
            "agent_start",
            agent=agent,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            system_prompt_bytes=len(system_prompt.encode("utf-8")),
            user_prompt_bytes=len(user_prompt.encode("utf-8")),
            tools=tools,
            max_tool_calls=max_tool_calls,
            force_tool_call=force_tool_call,
            text_is_final=text_is_final,
            terminal_tool=terminal_tool,
        )

    def agent_end(
        self,
        *,
        agent: str,
        status: str,
        text: str = "",
        data: Any = None,
        error: dict[str, Any] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self.record(
            "agent_end",
            agent=agent,
            status=status,
            text=text,
            data=data,
            error=error,
            duration_ms=duration_ms,
        )


@dataclass(frozen=True)
class RunInfo:
    """One run directory as shown by the trace list."""

    id: str
    path: Path
    meta: dict[str, Any]
    events: int
    errors: int
    updated_at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "meta": self.meta,
            "events": self.events,
            "errors": self.errors,
            "updated_at": self.updated_at,
        }


class TraceStore:
    """Read-only access to a trace root for the CLI and the web UI."""

    _RUN_ID_RE = r"^[A-Za-z0-9._-]+$"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._count_cache: dict[Path, tuple[tuple[int, int], int, int]] = {}

    def run_dir(self, run_id: str) -> Path | None:
        if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
            return None
        candidate = self.root / run_id
        try:
            if candidate.resolve().parent != self.root.resolve():
                return None
        except OSError:
            return None
        if not candidate.is_dir():
            return None
        return candidate

    def list_runs(self) -> list[RunInfo]:
        try:
            entries = [path for path in self.root.iterdir() if path.is_dir()]
        except OSError:
            return []
        runs: list[RunInfo] = []
        for entry in entries:
            meta = self._read_meta(entry)
            if meta is None:
                continue
            events, errors = self._count_events(entry)
            try:
                updated = (entry / "events.jsonl").stat().st_mtime
            except OSError:
                updated = entry.stat().st_mtime
            runs.append(
                RunInfo(
                    id=entry.name,
                    path=entry,
                    meta=meta,
                    events=events,
                    errors=errors,
                    updated_at=updated,
                )
            )
        runs.sort(key=lambda run: run.updated_at, reverse=True)
        return runs

    def _read_meta(self, directory: Path) -> dict[str, Any] | None:
        try:
            raw = (directory / "meta.json").read_text(encoding="utf-8")
            meta = json.loads(raw)
        except (OSError, ValueError):
            return None
        return meta if isinstance(meta, dict) else None

    def _count_events(self, directory: Path) -> tuple[int, int]:
        events_path = directory / "events.jsonl"
        try:
            stat = events_path.stat()
        except OSError:
            return 0, 0
        key = (stat.st_mtime_ns, stat.st_size)
        cached = self._count_cache.get(directory)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        count = 0
        errors = 0
        try:
            with events_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    count += 1
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("kind") == "agent_end" and event.get("status") == "error":
                        errors += 1
                    elif event.get("kind") == "run_end" and event.get("status") == "fatal":
                        errors += 1
        except OSError:
            return count, errors
        self._count_cache[directory] = (key, count, errors)
        return count, errors

    def events(self, run_id: str) -> list[dict[str, Any]]:
        directory = self.run_dir(run_id)
        if directory is None:
            return []
        events: list[dict[str, Any]] = []
        try:
            with (directory / "events.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict):
                        events.append(event)
        except OSError:
            return []
        return events

    def events_since(self, run_id: str, since: int) -> list[dict[str, Any]]:
        return [event for event in self.events(run_id) if int(event.get("seq", 0)) > since]


__all__ = [
    "TRACE_DIR_ENV",
    "TRACE_ENV",
    "RunInfo",
    "TraceRecorder",
    "TraceStore",
    "message_summary",
    "trace_dir_for",
]
