"""Read-only file tools shared by the Aspect and Explorer agents."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from pathspec import PathSpec

from khorek.files import PathEscapeError, is_binary, read_text_file, resolve_within_root, walk_files
from khorek.llm import Tool, ToolValidationError

MAX_GREP_MATCHES = 200
MAX_LISTED_FILES = 2000


def _string_argument(arguments: dict[str, Any], name: str, *, default: str | None = None) -> str:
    value = arguments.get(name, default)
    if not isinstance(value, str):
        raise ToolValidationError(f"argument {name!r} must be a string")
    return value


def _optional_string_argument(arguments: dict[str, Any], name: str) -> str | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolValidationError(f"argument {name!r} must be a string when given")
    return value


def _resolve_file(root: Path, root_real: Path, raw: str) -> tuple[Path, str]:
    try:
        resolved = resolve_within_root(root, raw)
    except PathEscapeError as exc:
        raise ToolValidationError(str(exc)) from exc
    if not resolved.is_file():
        raise ToolValidationError(f"not a file: {raw}")
    try:
        relative = resolved.relative_to(root_real).as_posix()
    except ValueError as exc:
        raise ToolValidationError(f"path escapes the project root: {raw}") from exc
    return resolved, relative


def read_file_tool(root: Path, max_file_bytes: int) -> Tool:
    """Create the ``read_file`` tool."""
    root_real = Path(os.path.realpath(root))

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        raw = _string_argument(arguments, "path")
        resolved, relative = _resolve_file(root, root_real, raw)
        try:
            if resolved.stat().st_size > max_file_bytes:
                raise ToolValidationError(
                    f"file is larger than max_file_bytes={max_file_bytes}: {relative}"
                )
            content = read_text_file(resolved)
        except OSError as exc:
            raise ToolValidationError(f"cannot read {relative}: {exc}") from exc
        return {
            "path": relative,
            "line_count": max(1, len(content.splitlines())),
            "content": content,
        }

    return Tool(
        name="read_file",
        description=(
            "Read a UTF-8 text file from the project. The path must be relative to "
            "the project root. Returns the whole content and its line count."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Project-relative path of the file to read.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _iter_grep_candidates(
    root: Path, root_real: Path, target: str, max_file_bytes: int
) -> list[tuple[Path, str]]:
    try:
        start = resolve_within_root(root, target or ".")
    except PathEscapeError as exc:
        raise ToolValidationError(str(exc)) from exc
    if start.is_file():
        relative = start.relative_to(root_real).as_posix()
        return [(start, relative)]
    if not start.is_dir():
        raise ToolValidationError(f"not a file or directory: {target}")

    candidates: list[tuple[Path, str]] = []
    for relative_path in walk_files(root, target or "."):
        absolute = root_real / relative_path
        try:
            if is_binary(absolute) or absolute.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue
        candidates.append((absolute, relative_path.as_posix()))
    return candidates


def grep_tool(root: Path, max_file_bytes: int) -> Tool:
    """Create the ``grep`` tool (Python regular expressions, 1-based lines)."""
    root_real = Path(os.path.realpath(root))

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        pattern = _string_argument(arguments, "pattern")
        target = _string_argument(arguments, "path", default=".")
        glob = _optional_string_argument(arguments, "glob")
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolValidationError(f"invalid regular expression: {exc}") from exc

        glob_spec = PathSpec.from_lines("gitignore", [glob]) if glob else None
        matches: list[dict[str, Any]] = []
        truncated = False
        for absolute, relative in _iter_grep_candidates(root, root_real, target, max_file_bytes):
            if glob_spec is not None and not glob_spec.match_file(relative):
                continue
            try:
                content = read_text_file(absolute)
            except OSError:
                continue
            for line_number, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    if len(matches) >= MAX_GREP_MATCHES:
                        truncated = True
                        break
                    matches.append({"path": relative, "line": line_number, "text": line})
            if truncated:
                break
        return {"matches": matches, "truncated": truncated}

    return Tool(
        name="grep",
        description=(
            "Search project text files with a Python regular expression. Returns "
            "matches with 1-based line numbers. Optionally restrict to a path and a "
            "glob (gitignore-style, e.g. '*.py')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regular expression to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Project-relative file or directory (default '.').",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional gitignore-style glob filter, e.g. '*.py'.",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def list_files_tool(root: Path) -> Tool:
    """Create the ``list_files`` tool."""

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        target = _string_argument(arguments, "path", default=".")
        pattern = _optional_string_argument(arguments, "pattern")
        files = walk_files(root, target)
        if pattern:
            if "*" in pattern or "?" in pattern or "[" in pattern:
                files = [path for path in files if fnmatch.fnmatch(path.as_posix(), pattern)]
            else:
                files = [path for path in files if path.as_posix() == pattern]
        truncated = len(files) > MAX_LISTED_FILES
        listed = [path.as_posix() for path in files[:MAX_LISTED_FILES]]
        return {"files": listed, "count": len(listed), "truncated": truncated}

    return Tool(
        name="list_files",
        description=(
            "List project files under a path, relative to the project root, sorted. "
            "Optionally filter with a shell-style pattern (e.g. '*.py')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Project-relative directory (default '.').",
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional shell-style filename pattern.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=handler,
    )


def read_only_tools(root: Path, max_file_bytes: int) -> list[Tool]:
    """The shared read-only tool set used by both agents."""
    return [
        read_file_tool(root, max_file_bytes),
        grep_tool(root, max_file_bytes),
        list_files_tool(root),
    ]


__all__ = [
    "MAX_GREP_MATCHES",
    "MAX_LISTED_FILES",
    "grep_tool",
    "list_files_tool",
    "read_file_tool",
    "read_only_tools",
]
