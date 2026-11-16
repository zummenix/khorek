"""Shared test harness: fake OpenAI-compatible model and tmp projects.

No test ever touches the network or ``~/.khorek``.  The fake model is an
``httpx.MockTransport`` injected through ``khorek.cli.main(..., transport=...)``;
scripted chat completions are consumed in order and every request is recorded
so tests can assert call counts and inspect tool results.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from khorek.cli import main
from khorek.db import Database
from khorek.files import sha256_file

Message = dict[str, Any]
ScriptItem = Message | Callable[[dict[str, Any]], Message]

PROJECT_FILE = "# a correct comment\nvalue = 1\n"

DEFAULT_LLM: dict[str, Any] = {
    "base_url": "http://model.test/v1",
    "model": "fake-model",
    "reasoning_effort": "none",
    "temperature": 0.0,
    "call_timeout": 10.0,
}

DEFAULT_DEFAULTS: dict[str, Any] = {
    "max_issues_per_file": 3,
    "max_file_bytes": 200_000,
    "max_prompt_bytes": 400_000,
    "max_tool_output_bytes": 16_000,
    "max_cycles": 25,
}

DEFAULT_AGENTS: dict[str, dict[str, Any]] = {
    "aspect": {"max_handoffs": 3, "max_tool_calls": 20},
    "explorer": {"max_tool_calls": 20},
    "fixer": {"max_attempts": 3, "max_tool_calls": 20},
    "verifier": {},
    "reviewer": {"reasoning_effort": "low", "max_handoffs": 2, "max_tool_calls": 20},
}

DEFAULT_ASPECT: dict[str, Any] = {
    "id": "comments",
    "prompt": "Find comments that are inconsistent with the code.",
    "files": ["src/"],
}


def tool_message(
    name: str,
    arguments: dict[str, Any] | str,
    *,
    call_id: str = "call_1",
) -> Message:
    """Build an assistant message containing one tool call."""
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": raw},
            }
        ],
    }


def text_message(text: str) -> Message:
    """Build a plain assistant text message."""
    return {"role": "assistant", "content": text}


def review(approved: bool, summary: str) -> Message:
    """Build a ``report_review`` tool call (the Reviewer's terminal tool)."""
    return tool_message("report_review", {"approved": approved, "summary": summary})


def revalidate(*verdicts: dict[str, Any]) -> Message:
    """Build a ``report_revalidation`` tool call with the given verdicts.

    Each verdict is a dict with ``issue_id`` and ``status`` and optional
    ``line_start``/``line_end``/``description``/``reason``.  An empty call
    (``revalidate()``) reports that no open issue survives or is closed.
    """
    return tool_message("report_revalidation", {"results": list(verdicts)})


def _completion(message: Message, model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
    }


class FakeModel:
    """Scripted OpenAI-compatible server over ``httpx.MockTransport``."""

    def __init__(
        self,
        script: Sequence[ScriptItem] | None = None,
        *,
        model: str = "fake-model",
    ) -> None:
        self.script: list[ScriptItem] = list(script or [])
        self.model = model
        self.chat_requests: list[dict[str, Any]] = []
        self.model_requests: int = 0
        self.transport = httpx.MockTransport(self._handle)
        self._next = 0

    def add(self, item: ScriptItem) -> None:
        self.script.append(item)

    def extend(self, items: Sequence[ScriptItem]) -> None:
        self.script.extend(items)

    @property
    def chat_calls(self) -> int:
        return len(self.chat_requests)

    def tool_results(self, request_index: int = -1) -> list[str]:
        """Tool-message contents visible to the model in one request."""
        messages = self.chat_requests[request_index]["messages"]
        return [
            str(message.get("content"))
            for message in messages
            if isinstance(message, dict) and message.get("role") == "tool"
        ]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path.endswith("/models"):
            self.model_requests += 1
            return httpx.Response(200, json={"object": "list", "data": [{"id": self.model}]})
        if request.method == "POST" and path.endswith("/chat/completions"):
            body = json.loads(request.content)
            self.chat_requests.append(body)
            if self._next >= len(self.script):
                raise AssertionError(f"unexpected model call #{self._next + 1}: script exhausted")
            item = self.script[self._next]
            self._next += 1
            message = item(body) if not isinstance(item, dict) else item
            return httpx.Response(200, json=_completion(message, self.model))
        return httpx.Response(404, text=f"unexpected request: {request.method} {path}")


@dataclass
class Project:
    """A tmp project root plus the tmp database path used for the run."""

    root: Path
    db_path: Path
    files: dict[str, str] = field(default_factory=dict)

    def write(self, relative: str, content: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.files[relative] = content
        return path

    def read(self, relative: str) -> str:
        return (self.root / relative).read_text(encoding="utf-8")

    def remove(self, relative: str) -> None:
        (self.root / relative).unlink()

    def exists(self, relative: str) -> bool:
        return (self.root / relative).exists()

    @property
    def trace_root(self) -> Path:
        return self.db_path.with_name(self.db_path.name + ".traces")


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML test value: {value!r}")


def write_config(
    project_root: Path,
    *,
    aspects: Sequence[dict[str, Any]] | None = None,
    llm: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
    agents: dict[str, dict[str, Any]] | None = None,
    verify: Sequence[dict[str, Any]] | None = None,
    observe: dict[str, Any] | None = None,
) -> Path:
    """Write a valid khorek.toml, merging test overrides into the defaults."""
    llm_values = {**DEFAULT_LLM, **(llm or {})}
    default_values = {**DEFAULT_DEFAULTS, **(defaults or {})}
    agent_values = {
        name: {**DEFAULT_AGENTS[name], **((agents or {}).get(name, {}))} for name in DEFAULT_AGENTS
    }
    aspect_values = list(aspects) if aspects is not None else [dict(DEFAULT_ASPECT)]

    lines: list[str] = ["[llm]"]
    lines.extend(f"{key} = {_toml_value(value)}" for key, value in llm_values.items())
    lines.extend(["", "[defaults]"])
    lines.extend(f"{key} = {_toml_value(value)}" for key, value in default_values.items())

    if observe is not None:
        lines.extend(["", "[observe]"])
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in observe.items())

    for agent_name, agent_config in agent_values.items():
        lines.extend(["", f"[agents.{agent_name}]"])
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in agent_config.items())

    for command in verify or []:
        lines.extend(["", "[[verify]]"])
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in command.items())

    for aspect in aspect_values:
        lines.extend(["", "[[aspects]]"])
        for key, value in aspect.items():
            lines.append(f"{key} = {_toml_value(value)}")

    path = project_root / "khorek.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_project(
    base: Path,
    *,
    files: dict[str, str] | None = None,
    aspects: Sequence[dict[str, Any]] | None = None,
    llm: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
    agents: dict[str, dict[str, Any]] | None = None,
    verify: Sequence[dict[str, Any]] | None = None,
    observe: dict[str, Any] | None = None,
) -> Project:
    """Create a tmp project with a config and files."""
    root = base
    root.mkdir(parents=True, exist_ok=True)
    project = Project(root=root, db_path=base.parent / f"{base.name}.db")
    file_values = files if files is not None else {"src/a.py": PROJECT_FILE}
    for relative, content in file_values.items():
        project.write(relative, content)
    write_config(
        root,
        aspects=aspects,
        llm=llm,
        defaults=defaults,
        agents=agents,
        verify=verify,
        observe=observe,
    )
    return project


def run_cli(fake: FakeModel, *arguments: str | Path) -> int:
    """Run the CLI in-process with the fake model transport."""
    return main([str(argument) for argument in arguments], transport=fake.transport)


@pytest.fixture(autouse=True)
def trace_viewer_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Path, dict[str, Any]]]:
    """Keep tests from starting a real trace viewer; records start attempts."""
    calls: list[tuple[Path, dict[str, Any]]] = []

    def fake_start(root: Path, **kwargs: Any) -> None:
        calls.append((Path(root), kwargs))
        return None

    monkeypatch.setattr("khorek.cli.start_viewer", fake_start)
    return calls


def explore_args(project: Project, *extra: str | Path) -> tuple[str | Path, ...]:
    """Standard ``explore`` arguments for a tmp project."""
    return (
        "explore",
        "-p",
        project.root,
        "--db",
        project.db_path,
        *extra,
    )


def fix_args(project: Project, *extra: str | Path) -> tuple[str | Path, ...]:
    """Standard ``fix`` arguments for a tmp project."""
    return (
        "fix",
        "-p",
        project.root,
        "--db",
        project.db_path,
        *extra,
    )


def explore_and_fix_args(project: Project, *extra: str | Path) -> tuple[str | Path, ...]:
    """Standard ``explore-and-fix`` arguments for a tmp project."""
    return (
        "explore-and-fix",
        "-p",
        project.root,
        "--db",
        project.db_path,
        *extra,
    )


def seed_issue(
    project: Project,
    *,
    aspect_id: str = "comments",
    path: str = "src/a.py",
    description: str = "the comment is inconsistent",
    line_start: int = 1,
    line_end: int = 1,
    severity: str = "high",
    retry_count: int = 0,
) -> int:
    """Insert one open issue.

    Issues carry no sha of their own: the pair's ``file_aspect_state`` row is
    the version token.  Pair a call with :func:`seed_state` (matching sha) to
    make the issue fresh, or with a drifted ``seed_state`` to make it stale.
    """
    with Database(project.db_path, project.root) as database:
        issue_id = database.insert_issue(
            aspect_id=aspect_id,
            path=path,
            description=description,
            line_start=line_start,
            line_end=line_end,
            severity=severity,
        )
        if retry_count:
            database.update_issue(issue_id, retry_count=retry_count)
    return issue_id


def seed_state(
    project: Project,
    *,
    aspect_id: str = "comments",
    path: str = "src/a.py",
    outcome: str = "issues",
    sha256: str | None = None,
) -> None:
    """Write one ``file_aspect_state`` row for a tmp project."""
    with Database(project.db_path, project.root) as database:
        database.upsert_state(
            aspect_id,
            path,
            sha256 if sha256 is not None else sha256_file(project.root / path),
            outcome,
        )


def set_created_at(project: Project, issue_id: int, created_at: str) -> None:
    """Override an issue's ``created_at`` to test queue ordering."""
    with Database(project.db_path, project.root) as database:
        with database.connection:
            database.connection.execute(
                "UPDATE issues SET created_at = ? WHERE id = ?",
                (created_at, issue_id),
            )


def python_command(code: str, *, timeout: float = 10.0) -> dict[str, object]:
    """A hermetic verify command running *code* with this interpreter."""
    return {
        "cmd": f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}",
        "timeout": timeout,
    }


@pytest.fixture
def project(tmp_path: Path) -> Project:
    """A minimal tmp project with ``src/a.py`` and the comments aspect."""
    return make_project(tmp_path / "project")


@contextmanager
def held_lock(lock_path: Path) -> Generator[subprocess.Popen[str], None, None]:
    """Hold an ``flock`` on *lock_path* from a subprocess until teardown."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "import fcntl, sys, time\n"
        "handle = open(sys.argv[1], 'a+')\n"
        "fcntl.flock(handle.fileno(), fcntl.LOCK_EX)\n"
        "handle.seek(0)\n"
        "handle.truncate()\n"
        "handle.write('pid=999999\\nstarted_at=from-test\\n')\n"
        "handle.flush()\n"
        "print('locked', flush=True)\n"
        "time.sleep(60)\n"
    )
    process: subprocess.Popen[str] = subprocess.Popen(
        [sys.executable, "-c", script, str(lock_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    stdout = process.stdout
    assert stdout is not None
    line = stdout.readline()
    if line.strip() != "locked":
        process.kill()
        process.wait(timeout=5)
        raise RuntimeError(f"lock helper failed to start: {line!r}")
    try:
        yield process
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
