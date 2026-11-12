"""Fixer tools and agent prompt: exact edits, writes, escapes, context."""

from __future__ import annotations

from pathlib import Path

import pytest

from khorek.agents.fixer import (
    FIXER_TOOL_NAMES,
    FixerAgent,
    FixOutcome,
    edit_file_tool,
    write_file_tool,
)
from khorek.config import DefaultsConfig, FixerAgentConfig, LLMConfig
from khorek.content import format_file_lines
from khorek.db import Issue
from khorek.llm import LLMClient, Tool, ToolValidationError
from khorek.snapshots import Snapshotter
from tests.conftest import FakeModel, make_project, tool_message


def handler(tool: Tool):
    assert tool.handler is not None
    return tool.handler


def test_edit_file_exact_match_succeeds_and_snapshots(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "a.py"
    target.write_text("value = 1\nnext = 2\n", encoding="utf-8")
    snapshotter = Snapshotter(root)

    result = handler(edit_file_tool(root, snapshotter))(
        {"path": "a.py", "search": "value = 1", "replace": "value = 10"}
    )

    assert result["path"] == "a.py"
    assert result["replacements"] == 1
    assert target.read_text(encoding="utf-8") == "value = 10\nnext = 2\n"
    assert snapshotter.has_changes() is True

    snapshotter.restore()
    assert target.read_text(encoding="utf-8") == "value = 1\nnext = 2\n"


def test_edit_file_missing_search_is_a_structured_error(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.py").write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(ToolValidationError, match="was not found"):
        handler(edit_file_tool(root, Snapshotter(root)))(
            {"path": "a.py", "search": "value = 2", "replace": "value = 3"}
        )


def test_edit_file_ambiguous_search_is_a_structured_error(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.py").write_text("value = 1\nvalue = 1\n", encoding="utf-8")

    with pytest.raises(ToolValidationError, match="occurs 2 times"):
        handler(edit_file_tool(root, Snapshotter(root)))(
            {"path": "a.py", "search": "value = 1", "replace": "value = 2"}
        )


def test_edit_file_rejects_empty_search_and_missing_file(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tool = edit_file_tool(root, Snapshotter(root))

    with pytest.raises(ToolValidationError, match="must not be empty"):
        handler(tool)({"path": "a.py", "search": "", "replace": "x"})
    with pytest.raises(ToolValidationError, match="does not exist"):
        handler(tool)({"path": "a.py", "search": "x", "replace": "y"})


def test_write_file_creates_and_overwrites(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.py").write_text("old\n", encoding="utf-8")
    snapshotter = Snapshotter(root)
    tool = write_file_tool(root, snapshotter)

    created = handler(tool)({"path": "src/new.py", "content": "new = 1\n"})
    assert created == {"path": "src/new.py", "bytes_written": 8, "created": True}
    assert (root / "src" / "new.py").read_text(encoding="utf-8") == "new = 1\n"

    overwritten = handler(tool)({"path": "a.py", "content": "replaced\n"})
    assert overwritten["created"] is False
    assert (root / "a.py").read_text(encoding="utf-8") == "replaced\n"

    snapshotter.restore()
    assert not (root / "src" / "new.py").exists()
    assert (root / "a.py").read_text(encoding="utf-8") == "old\n"


def test_write_file_rejects_escaping_paths(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    tool = write_file_tool(root, Snapshotter(root))

    with pytest.raises(ToolValidationError, match="escapes the project root"):
        handler(tool)({"path": "../outside/x.py", "content": "x\n"})
    with pytest.raises(ToolValidationError, match="escapes the project root"):
        handler(tool)({"path": str(outside / "x.py"), "content": "x\n"})

    link = root / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ToolValidationError, match="escapes the project root"):
        handler(tool)({"path": "link/x.py", "content": "x\n"})
    assert not (outside / "x.py").exists()


def test_edit_file_rejects_escaping_paths(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.py").write_text("x = 1\n", encoding="utf-8")
    tool = edit_file_tool(root, Snapshotter(root))

    with pytest.raises(ToolValidationError, match="escapes the project root"):
        handler(tool)({"path": "../outside/a.py", "search": "x", "replace": "y"})
    with pytest.raises(ToolValidationError, match="escapes the project root"):
        handler(tool)({"path": str(outside / "a.py"), "search": "x", "replace": "y"})


def _issue(line_start: int = 2, line_end: int = 2) -> Issue:
    return Issue(
        id=7,
        aspect_id="comments",
        path="src/a.py",
        sha256="stored-sha",
        description="the comment is inconsistent",
        line_start=line_start,
        line_end=line_end,
        severity="high",
        retry_count=0,
        last_error=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


def _client(fake: FakeModel) -> LLMClient:
    return LLMClient(
        LLMConfig(
            base_url="http://model.test/v1",
            model="fake-model",
            reasoning_effort="none",
            temperature=0.0,
            call_timeout=10.0,
        ),
        transport=fake.transport,
    )


def test_fixer_prompt_has_issue_file_snippet_and_previous_failure(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project", files={"src/a.py": "one\ntwo\nthree\n"})
    fake = FakeModel([tool_message("report_fix_result", {"fixed": False, "summary": "cannot fix"})])
    client = _client(fake)
    try:
        agent = FixerAgent(
            client,
            project_root=project.root,
            config=FixerAgentConfig(),
            defaults=DefaultsConfig(),
            snapshotter=Snapshotter(project.root),
        )
        outcome = agent.fix(
            _issue(),
            "src/a.py",
            "one\ntwo\nthree\n",
            previous_failure="verification failed: expected TWO",
        )
    finally:
        client.close()

    assert outcome == FixOutcome(fixed=False, summary="cannot fix")
    request = fake.chat_requests[0]
    user_prompt = request["messages"][1]["content"]
    assert "the comment is inconsistent" in user_prompt
    assert "Current line count: 3" in user_prompt
    snippet = format_file_lines("src/a.py", "one\ntwo\nthree\n", offset=2, limit=1)
    assert snippet in user_prompt
    full = format_file_lines("src/a.py", "one\ntwo\nthree\n", offset=None, limit=None)
    assert full in user_prompt
    assert "verification failed: expected TWO" in user_prompt

    tool_names = {definition["function"]["name"] for definition in request["tools"]}
    assert tool_names == FIXER_TOOL_NAMES
    assert {"grep", "list_files", "ask_explorer"}.isdisjoint(tool_names)


def test_fixer_system_prompt_is_overridable(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    fake = FakeModel([tool_message("report_fix_result", {"fixed": False, "summary": "no"})])
    client = _client(fake)
    try:
        agent = FixerAgent(
            client,
            project_root=project.root,
            config=FixerAgentConfig(prompt="CUSTOM FIXER PROMPT"),
            defaults=DefaultsConfig(),
            snapshotter=Snapshotter(project.root),
        )
        agent.fix(_issue(1, 1), "src/a.py", "x = 1\n")
    finally:
        client.close()
    assert fake.chat_requests[0]["messages"][0]["content"] == "CUSTOM FIXER PROMPT"


def test_fixer_agent_edits_through_tool_calls(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project", files={"src/a.py": "value = 1\n"})
    fake = FakeModel(
        [
            tool_message(
                "edit_file", {"path": "src/a.py", "search": "value = 1", "replace": "value = 2"}
            ),
            tool_message("report_fix_result", {"fixed": True, "summary": "changed 1 to 2"}),
        ]
    )
    client = _client(fake)
    snapshotter = Snapshotter(project.root)
    try:
        agent = FixerAgent(
            client,
            project_root=project.root,
            config=FixerAgentConfig(),
            defaults=DefaultsConfig(),
            snapshotter=snapshotter,
        )
        outcome = agent.fix(_issue(1, 1), "src/a.py", "value = 1\n")
    finally:
        client.close()

    assert outcome.fixed is True
    assert project.read("src/a.py") == "value = 2\n"
    assert snapshotter.has_changes() is True
    results = fake.tool_results(1)
    assert any("replacements" in result for result in results)
