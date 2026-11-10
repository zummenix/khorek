"""Run traces: recorder, store, HTTP viewer, and timeout capture."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path
from typing import Any

import httpx

from khorek.trace import TraceRecorder, TraceStore, trace_dir_for
from khorek.trace_server import TraceServer
from tests.conftest import (
    FakeModel,
    explore_args,
    make_project,
    run_cli,
    tool_message,
)


def _run_traced(project: Any) -> list[dict[str, Any]]:
    fake = FakeModel([tool_message("report_issues", {"issues": []})])
    assert run_cli(fake, *explore_args(project)) == 0
    store = TraceStore(project.trace_root)
    runs = store.list_runs()
    assert len(runs) == 1
    return store.events(runs[0].id)


def test_trace_records_request_response_tools_and_agent(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project", observe={"enabled": True})
    events = _run_traced(project)

    kinds = [event["kind"] for event in events]
    assert kinds[0] == "run_start"
    assert "agent_start" in kinds
    assert "model_call" in kinds
    assert "tool_call" in kinds
    assert kinds[-1] == "run_end"

    call = next(event for event in events if event["kind"] == "model_call")
    request = call["request"]
    assert call["agent"] == "aspect:comments:src/a.py"
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][1]["role"] == "user"
    assert request["tool_choice"] == "required"
    assert request["max_tokens"] == 8192
    assert {tool["function"]["name"] for tool in request["tools"]} == {
        "read_file",
        "grep",
        "list_files",
        "ask_explorer",
        "report_issues",
    }
    assert call["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == (
        "report_issues"
    )
    assert call["duration_ms"] is not None
    assert call["request_bytes"] > 0


def test_trace_captures_reasoning_field(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project", observe={"enabled": True})
    fake = FakeModel(
        [
            {
                "role": "assistant",
                "content": None,
                "reasoning": "I should inspect the comment first.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "report_issues",
                            "arguments": json.dumps({"issues": []}),
                        },
                    }
                ],
            }
        ]
    )
    assert run_cli(fake, *explore_args(project)) == 0
    call = next(
        event
        for event in TraceStore(project.trace_root).events(
            TraceStore(project.trace_root).list_runs()[0].id
        )
        if event["kind"] == "model_call"
    )
    assert call["thinking"] == "I should inspect the comment first."


def test_trace_disabled_by_default(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    fake = FakeModel([tool_message("report_issues", {"issues": []})])
    assert run_cli(fake, *explore_args(project)) == 0
    assert not project.trace_root.exists()


def test_trace_env_enables_and_no_trace_flag_disables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KHOREK_TRACE", "1")
    project = make_project(tmp_path / "project")
    fake = FakeModel([tool_message("report_issues", {"issues": []})])
    assert run_cli(fake, *explore_args(project)) == 0
    assert project.trace_root.exists()

    disabled = make_project(tmp_path / "disabled")
    fake2 = FakeModel([tool_message("report_issues", {"issues": []})])
    assert run_cli(fake2, *explore_args(disabled), "--no-trace") == 0
    assert not disabled.trace_root.exists()


def test_trace_records_timeout_with_the_request(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project", observe={"enabled": True})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        raise httpx.ReadTimeout("too slow")

    fake = FakeModel([])
    fake.transport = httpx.MockTransport(handler)
    assert run_cli(fake, *explore_args(project)) == 1

    store = TraceStore(project.trace_root)
    events = store.events(store.list_runs()[0].id)
    call = next(event for event in events if event["kind"] == "model_call")
    assert call["error"]["code"] == "timeout"
    assert call["request"]["messages"][0]["role"] == "system"
    assert call["duration_ms"] is not None
    assert events[-1]["exit_code"] == 1


def test_trace_prunes_old_runs(tmp_path: Path) -> None:
    root = tmp_path / "traces"
    for _ in range(4):
        recorder = TraceRecorder(root, run={"command": "explore"}, keep_runs=2)
        recorder.record("run_start")
        recorder.close(exit_code=0)
    assert len(TraceStore(root).list_runs()) == 2


def test_trace_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces")
    assert store.run_dir("../outside") is None
    assert store.run_dir("..") is None
    assert store.run_dir(".") is None


def test_trace_server_serves_ui_and_api(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path / "traces", run={"command": "explore", "model": "fake"})
    recorder.record("model_call", agent="aspect", step=0, request={"messages": []})
    recorder.close(exit_code=0, status="ok")

    store = TraceStore(tmp_path / "traces")
    run = store.list_runs()[0]
    server = TraceServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            assert "khorek traces" in response.read().decode("utf-8")
        with urllib.request.urlopen(base + "/app.js", timeout=5) as response:
            assert "renderTimeline" in response.read().decode("utf-8")
        with urllib.request.urlopen(base + "/api/runs", timeout=5) as response:
            payload = json.loads(response.read())
        assert payload["runs"][0]["id"] == run.id
        with urllib.request.urlopen(base + f"/api/runs/{run.id}/events", timeout=5) as response:
            events = json.loads(response.read())["events"]
        assert events[0]["kind"] == "run_start" or events[0]["kind"] == "model_call"
        with urllib.request.urlopen(base + f"/api/runs/{run.id}/raw", timeout=5) as response:
            assert b"model_call" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_trace_dir_helper() -> None:
    assert trace_dir_for(Path("/tmp/x.db")) == Path("/tmp/x.db.traces")


def test_traces_cli_uses_configured_directory(tmp_path: Path, monkeypatch) -> None:
    from khorek.cli import main

    trace_dir = tmp_path / "custom-traces"
    project = make_project(
        tmp_path / "project",
        observe={"enabled": True, "directory": str(trace_dir)},
    )
    fake = FakeModel([tool_message("report_issues", {"issues": []})])
    assert run_cli(fake, *explore_args(project)) == 0

    calls: list[tuple[Path, dict[str, Any]]] = []

    def fake_serve(root: Path, **kwargs: Any) -> None:
        calls.append((root, kwargs))

    monkeypatch.setattr("khorek.cli.serve_traces", fake_serve)
    assert main(["traces", "-p", str(project.root), "--no-open"]) == 0
    assert calls[0][0] == trace_dir.resolve()
    assert calls[0][1]["open_browser"] is False


def test_traced_run_prints_the_trace_path(tmp_path: Path, capsys) -> None:
    project = make_project(tmp_path / "project", observe={"enabled": True})
    fake = FakeModel([tool_message("report_issues", {"issues": []})])
    assert run_cli(fake, *explore_args(project)) == 0
    output = capsys.readouterr().out
    assert "trace: " in output
    assert str(project.trace_root) in output
