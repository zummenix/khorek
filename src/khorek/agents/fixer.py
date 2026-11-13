"""Fixer agent: apply an exact-match edit (or a file write) for one issue.

The Fixer's tool set is deliberately narrow (``read_file``, ``edit_file``,
``write_file``, terminal ``report_fix_result``).  Both mutating tools capture
the file in the attempt's :class:`~khorek.snapshots.Snapshotter` before their
first write, so a failed attempt can be rolled back completely.

A round that reports ``fixed: true`` without changing any snapshotted file is
*not* trusted: the workflow treats it as a failed fix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from khorek.agents.tools import read_file_tool
from khorek.config import DefaultsConfig, FixerAgentConfig
from khorek.content import format_file_lines
from khorek.db import Issue
from khorek.files import (
    PathEscapeError,
    read_text_file,
    resolve_for_write,
    resolve_within_root,
)
from khorek.llm import LLMClient, Tool, ToolValidationError
from khorek.prompts import default_prompt
from khorek.snapshots import Snapshotter

FIXER_TOOL_NAMES = frozenset({"read_file", "edit_file", "write_file", "report_fix_result"})


class FixReport(BaseModel):
    """Terminal ``report_fix_result`` payload."""

    model_config = ConfigDict(extra="forbid")

    fixed: bool
    summary: str


@dataclass(frozen=True)
class FixOutcome:
    """What the Fixer reported for one round."""

    fixed: bool
    summary: str


def _string_argument(arguments: dict[str, Any], name: str, *, default: str | None = None) -> str:
    value = arguments.get(name, default)
    if not isinstance(value, str):
        raise ToolValidationError(f"argument {name!r} must be a string")
    return value


def _resolve_existing(root: Path, raw: str) -> tuple[Path, str]:
    """Resolve an existing file inside the root, for editing."""
    root_real = Path(os.path.realpath(root))
    try:
        resolved = resolve_within_root(root, raw)
    except PathEscapeError as exc:
        raise ToolValidationError(str(exc)) from exc
    if not resolved.is_file():
        raise ToolValidationError(f"not a file: {raw}")
    return resolved, resolved.relative_to(root_real).as_posix()


def edit_file_tool(root: Path, snapshotter: Snapshotter) -> Tool:
    """Create the exact-match ``edit_file`` tool."""

    def handler(arguments: dict[str, Any]) -> dict[str, object]:
        raw = _string_argument(arguments, "path")
        search = _string_argument(arguments, "search")
        replace = _string_argument(arguments, "replace")
        if not search:
            raise ToolValidationError("search text must not be empty")
        if search == replace:
            raise ToolValidationError("search text and replace text are exactly the same")
        resolved, relative = _resolve_existing(root, raw)
        try:
            content = read_text_file(resolved)
        except OSError as exc:
            raise ToolValidationError(f"cannot read {raw}: {exc}") from exc

        occurrences = content.count(search)
        if occurrences == 0:
            raise ToolValidationError(
                f"search text was not found in {relative}; it must match the "
                "current file contents exactly, including whitespace"
            )
        if occurrences > 1:
            raise ToolValidationError(
                f"search text occurs {occurrences} times in {relative}; "
                "include more surrounding lines so it matches exactly once"
            )

        snapshotter.capture(resolved)
        updated = content.replace(search, replace, 1)
        try:
            resolved.write_bytes(updated.encode("utf-8"))
        except OSError as exc:
            raise ToolValidationError(f"cannot write {relative}: {exc}") from exc
        return {
            "path": relative,
            "replacements": 1,
            "line_count": max(1, len(updated.splitlines())),
        }

    return Tool(
        name="edit_file",
        description=(
            "Apply an exact-match SEARCH/REPLACE edit to one project file. "
            "'search' must occur exactly once in the current file and must be "
            "copied verbatim, including whitespace; otherwise the call is "
            "rejected with an error."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Project-relative path of the file to edit.",
                },
                "search": {
                    "type": "string",
                    "description": "Exact text to find; must occur exactly once.",
                },
                "replace": {
                    "type": "string",
                    "description": "Text to put in place of the search text.",
                },
            },
            "required": ["path", "search", "replace"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def write_file_tool(root: Path, snapshotter: Snapshotter) -> Tool:
    """Create the ``write_file`` tool (overwrites or creates a UTF-8 file)."""

    def handler(arguments: dict[str, Any]) -> dict[str, object]:
        raw = _string_argument(arguments, "path")
        content = _string_argument(arguments, "content")
        try:
            target = resolve_for_write(root, raw)
        except PathEscapeError as exc:
            raise ToolValidationError(str(exc)) from exc
        if target.is_dir():
            raise ToolValidationError(f"path is a directory: {raw}")
        created = not target.exists()
        snapshotter.capture(target)
        encoded = content.encode("utf-8")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(encoded)
        except OSError as exc:
            raise ToolValidationError(f"cannot write {raw}: {exc}") from exc
        root_real = Path(os.path.realpath(root))
        return {
            "path": target.relative_to(root_real).as_posix(),
            "bytes_written": len(encoded),
            "created": created,
        }

    return Tool(
        name="write_file",
        description=(
            "Overwrite or create a UTF-8 text file in the project. Use this only "
            "when replacing most of the file is simpler than an exact-match edit."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Project-relative path of the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Full new contents of the file.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def report_fix_result_tool() -> Tool:
    """The terminal ``report_fix_result`` tool."""
    return Tool(
        name="report_fix_result",
        description=(
            "Report the outcome for this issue. Call with fixed=true only after "
            "you actually changed the file; the harness verifies the change and "
            "runs the configured checks. This must be your final call."
        ),
        parameters=FixReport.model_json_schema(),
        handler=None,
    )


def _build_user_prompt(
    issue: Issue,
    path: str,
    content: str,
    previous_failure: str | None,
    previous_source: str | None = None,
) -> str:
    lines = content.splitlines()
    count = max(1, len(lines))
    start = min(max(1, issue.line_start), count)
    end = min(max(issue.line_end, start), count)

    parts = [
        f"Issue #{issue.id} (aspect: {issue.aspect_id}, severity: {issue.severity})",
        f"File: {path}",
        f"Current line count: {count}",
        "",
        "Issue description:",
        issue.description.strip(),
        "",
        f"Issue line range: {start}-{end}; snippet reconstructed from the current file:",
        format_file_lines(path, content, offset=start, limit=end - start + 1),
        "",
        "Current file contents:",
        format_file_lines(path, content, offset=None, limit=None),
        "",
    ]
    if previous_failure:
        label = (
            "Previous review rejection from the last attempt (verification passed):"
            if previous_source == "review"
            else "Previous verification failure from the last attempt:"
        )
        parts.extend(
            [
                label,
                "----- BEGIN FAILURE -----",
                previous_failure.strip(),
                "----- END FAILURE -----",
                "",
            ]
        )
    parts.append(
        "Fix this issue with edit_file (exact SEARCH/REPLACE) or write_file, then "
        "call report_fix_result. If you cannot fix it, report fixed=false with a "
        "short explanation."
    )
    return "\n".join(parts)


class FixerAgent:
    """Runs one bounded Fixer round for one issue."""

    def __init__(
        self,
        client: LLMClient,
        *,
        project_root: Path,
        config: FixerAgentConfig,
        defaults: DefaultsConfig,
        snapshotter: Snapshotter,
    ) -> None:
        self._client = client
        self._config = config
        self._snapshotter = snapshotter
        self._system_prompt = config.prompt.strip() or default_prompt("fixer")
        self._tools = [
            read_file_tool(project_root, defaults.max_file_bytes),
            edit_file_tool(project_root, snapshotter),
            write_file_tool(project_root, snapshotter),
            report_fix_result_tool(),
        ]

    def fix(
        self,
        issue: Issue,
        path: str,
        content: str,
        *,
        previous_failure: str | None = None,
        previous_source: str | None = None,
    ) -> FixOutcome:
        """Run one Fixer round and return its reported outcome."""
        result = self._client.run_agent(
            system_prompt=self._system_prompt,
            user_prompt=_build_user_prompt(issue, path, content, previous_failure, previous_source),
            tools=self._tools,
            max_tool_calls=self._config.max_tool_calls,
            force_tool_call=True,
            text_is_final=False,
            terminal_tool="report_fix_result",
            terminal_schema=FixReport,
            agent=f"fixer:#{issue.id}:{path}",
            reasoning_effort=self._config.reasoning_effort,
        )
        data = result.data
        if not isinstance(data, FixReport):
            raise TypeError("report_fix_result did not return a FixReport")
        return FixOutcome(fixed=data.fixed, summary=data.summary.strip())


__all__ = [
    "FIXER_TOOL_NAMES",
    "FixOutcome",
    "FixReport",
    "FixerAgent",
    "edit_file_tool",
    "report_fix_result_tool",
    "write_file_tool",
]
