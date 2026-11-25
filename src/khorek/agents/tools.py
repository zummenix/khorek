"""Read-only file tools shared by the Aspect and Explorer agents.

Every tool returns a dict and bounds its own serialized result to the
configured ``max_tool_output_bytes``: the loop no longer truncates centrally.

The budget covers the whole result, fixed metadata included.  When a budget is
set below the size of a result's fixed metadata (its "envelope": the path, line
numbers, and wrapper keys), no layout can honor it; the tool then returns the
smallest valid result — empty payload, ``truncated: true`` — and the envelope
is the floor.  The envelope is small for sane budgets; only a deliberately tiny
``max_tool_output_bytes`` (or a very long project-relative path echoed by
``read_file``) can reach it.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pathspec import PathSpec

from khorek.agents.arguments import (
    optional_int_argument,
    optional_string_argument,
    string_argument,
)
from khorek.content import format_file_lines
from khorek.files import PathEscapeError, is_binary, read_text_file, resolve_within_root, walk_files
from khorek.llm import Tool, ToolValidationError

MAX_GREP_MATCHES = 100
MAX_LISTED_FILES = 2000
MAX_FILE_LINES = 1000


def serialized_size(value: Any) -> int:
    """UTF-8 size of *value* exactly as the agent loop serializes tool results."""
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def count_that_fits(candidates: int, fits: Callable[[int], bool]) -> int:
    """Return the largest value in [0, candidates] for which *fits* is true.

    Return -1 when no value fits.
    """
    low, high, best = 0, candidates, -1
    while low <= high:
        middle = (low + high) // 2
        if fits(middle):
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def fit_result(
    result: dict[str, Any],
    fallbacks: Sequence[dict[str, Any]],
    max_bytes: int,
) -> dict[str, Any]:
    """Return the first of *result* and *fallbacks* whose serialized size fits.

    *result* is the full result; *fallbacks* are progressively smaller valid
    shapes of the same result, the last one being the floor.  Use this for
    tools whose only variable cost is the path: they cannot shrink a payload,
    so they drop optional fields instead.  The cap covers the whole JSON
    result, metadata included.
    """
    for candidate in (result, *fallbacks):
        if serialized_size(candidate) <= max_bytes:
            return candidate
    return fallbacks[-1]


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


def _read_file_result(
    relative: str,
    total: int,
    start: int,
    end: int,
    content: str,
    truncated: bool,
) -> dict[str, Any]:
    """Assemble the ``read_file`` result dict for one line window."""
    return {
        "path": relative,
        "line_start": start,
        "line_end": end,
        "total_lines": total,
        "truncated": truncated,
        "remaining": max(0, total - end),
        "content": content,
    }


def _fit_read_file_result(
    relative: str,
    lines: list[str],
    text: str,
    start: int,
    requested: int,
    total: int,
    max_bytes: int,
) -> dict[str, Any]:
    """The widest line window from *start* whose serialized result fits *max_bytes*.

    The cap applies to the whole JSON result the loop sends, not just the
    ``content`` string: metadata and JSON escaping count too.
    """

    def build_window(count: int) -> dict[str, Any]:
        end = start + count - 1
        content = format_file_lines(relative, text, offset=start, limit=count, header=False)
        return _read_file_result(relative, total, start, end, content, end < total)

    best_count = count_that_fits(
        requested,
        lambda count: (
            count == 0 or (count > 0 and serialized_size(build_window(count)) <= max_bytes)
        ),
    )
    if best_count >= 1:
        return build_window(best_count)

    # A single numbered line is already over the cap: show as much of it as
    # fits.  No tool can read further into a line, so the tail is truncated.
    line = lines[start - 1]
    prefix = f"{start:>{len(str(start))}} | "
    best = _read_file_result(relative, total, start, start, "", True)
    if serialized_size(best) > max_bytes:
        return best  # fixed metadata alone exceeds the cap; nothing more to drop

    def build_line(characters: int) -> dict[str, Any]:
        content = prefix + line[:characters]
        return _read_file_result(relative, total, start, start, content, True)

    best_characters = count_that_fits(
        len(line), lambda characters: serialized_size(build_line(characters)) <= max_bytes
    )
    if best_characters >= 0:
        return build_line(best_characters)
    return best


def read_file_tool(root: Path, max_file_bytes: int, max_tool_output_bytes: int) -> Tool:
    """Create the self-truncating ``read_file`` tool."""
    root_real = Path(os.path.realpath(root))

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        raw = string_argument(arguments, "path")
        offset = optional_int_argument(arguments, "offset")
        limit = optional_int_argument(arguments, "limit")
        resolved, relative = _resolve_file(root, root_real, raw)
        try:
            if resolved.stat().st_size > max_file_bytes:
                raise ToolValidationError(
                    f"file is larger than max_file_bytes={max_file_bytes}: {relative}"
                )
            text = read_text_file(resolved)
        except OSError as exc:
            raise ToolValidationError(f"cannot read {relative}: {exc}") from exc

        lines = text.splitlines() or [""]
        total = len(lines)
        start = 1 if offset is None or offset <= 0 else offset
        start = min(start, total)
        requested = total - start + 1
        if limit is not None and limit > 0:
            requested = min(requested, limit)
        requested = min(requested, MAX_FILE_LINES)

        return _fit_read_file_result(
            relative, lines, text, start, requested, total, max_tool_output_bytes
        )

    return Tool(
        name="read_file",
        description=(
            "Read a UTF-8 text file from the project. The path must be relative to "
            "the project root. Reads lines *offset* through *offset+limit-1* "
            "(1-based); both are optional and default to the whole file. The "
            "result includes the line range, the number of lines shown, how many "
            "lines remain unread, and the numbered lines themselves; when "
            "truncated is true, use a later offset to keep reading."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Project-relative path of the file to read.",
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based first line to read (default 1).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: all "
                    "lines from offset).",
                },
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


def _grep_result(matches: list[dict[str, Any]], truncated: bool) -> dict[str, Any]:
    return {"matches": matches, "truncated": truncated}


def _truncated_grep_match(
    match: dict[str, Any], kept: list[dict[str, Any]], max_bytes: int
) -> dict[str, Any] | None:
    """The boundary match with its text cut to fit, or ``None`` if it cannot."""
    base = {"path": match["path"], "line": match["line"]}
    text = match["text"]

    def fits(characters: int) -> bool:
        candidate = {**base, "text": text[:characters]}
        return serialized_size(_grep_result([*kept, candidate], True)) <= max_bytes

    best_characters = count_that_fits(len(text), fits)
    if best_characters < 0:
        return None
    return {**base, "text": text[:best_characters]}


def _fit_grep_result(
    matches: list[dict[str, Any]], count_truncated: bool, max_bytes: int
) -> dict[str, Any]:
    """Trim *matches* until the serialized result fits *max_bytes*.

    Whole matches are kept first; if the next match's line alone does not fit,
    its ``text`` is cut short instead.  The cap applies to the whole JSON
    result the loop sends, metadata and JSON escaping included.
    """
    if serialized_size(_grep_result(matches, count_truncated)) <= max_bytes:
        return _grep_result(matches, count_truncated)

    best = count_that_fits(
        len(matches),
        lambda count: serialized_size(_grep_result(matches[:count], True)) <= max_bytes,
    )
    if best < 0:
        best = 0

    if best < len(matches):
        boundary = _truncated_grep_match(matches[best], matches[:best], max_bytes)
        if boundary is not None:
            return _grep_result([*matches[:best], boundary], True)
    return _grep_result(matches[:best], True)


def grep_tool(root: Path, max_file_bytes: int, max_tool_output_bytes: int) -> Tool:
    """Create the ``grep`` tool (Python regular expressions, 1-based lines)."""
    root_real = Path(os.path.realpath(root))

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        pattern = string_argument(arguments, "pattern")
        target = string_argument(arguments, "path", default=".")
        glob = optional_string_argument(arguments, "glob")
        requested = optional_int_argument(arguments, "limit")
        match_limit = MAX_GREP_MATCHES
        if requested is not None and requested > 0:
            match_limit = min(requested, MAX_GREP_MATCHES)
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
                    if len(matches) >= match_limit:
                        truncated = True
                        break
                    matches.append({"path": relative, "line": line_number, "text": line})
            if truncated:
                break
        return _fit_grep_result(matches, truncated, max_tool_output_bytes)

    return Tool(
        name="grep",
        description=(
            "Search project text files with a Python regular expression. Returns "
            "matches with 1-based line numbers. Optionally restrict to a path and a "
            "glob (gitignore-style, e.g. '*.py'). At most *limit* matches are "
            f"returned (default and hard cap {MAX_GREP_MATCHES}); long match lines "
            "and matches beyond the harness output budget are dropped or cut "
            "short, and truncated is set."
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
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Maximum matches to return (default and hard cap {MAX_GREP_MATCHES})."
                    ),
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _list_files_result(files: list[str], truncated: bool) -> dict[str, Any]:
    return {"files": files, "count": len(files), "truncated": truncated}


def _fit_list_files_result(
    files: list[str], count_truncated: bool, max_bytes: int
) -> dict[str, Any]:
    """Keep as many whole paths as fit *max_bytes*; drop the rest from the tail.

    A path is never cut short: a partial path is misleading, so a boundary
    path that does not fit is dropped instead.  The cap applies to the whole
    JSON result, metadata included.
    """
    if serialized_size(_list_files_result(files, count_truncated)) <= max_bytes:
        return _list_files_result(files, count_truncated)

    best = count_that_fits(
        len(files),
        lambda count: serialized_size(_list_files_result(files[:count], True)) <= max_bytes,
    )
    if best < 0:
        best = 0
    return _list_files_result(files[:best], True)


def list_files_tool(root: Path, max_tool_output_bytes: int) -> Tool:
    """Create the self-truncating ``list_files`` tool."""

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        target = string_argument(arguments, "path", default=".")
        pattern = optional_string_argument(arguments, "pattern")
        files = walk_files(root, target)
        if pattern:
            if "*" in pattern or "?" in pattern or "[" in pattern:
                files = [path for path in files if fnmatch.fnmatch(path.as_posix(), pattern)]
            else:
                files = [path for path in files if path.as_posix() == pattern]
        truncated = len(files) > MAX_LISTED_FILES
        listed = [path.as_posix() for path in files[:MAX_LISTED_FILES]]
        return _fit_list_files_result(listed, truncated, max_tool_output_bytes)

    return Tool(
        name="list_files",
        description=(
            "List project files under a path, relative to the project root, sorted. "
            "Optionally filter with a shell-style pattern (e.g. '*.py'). The "
            "list is capped by the harness output budget; truncated is set when "
            "files were dropped."
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


def read_only_tools(root: Path, max_file_bytes: int, max_tool_output_bytes: int) -> list[Tool]:
    """The shared read-only tool set used by both agents."""
    return [
        read_file_tool(root, max_file_bytes, max_tool_output_bytes),
        grep_tool(root, max_file_bytes, max_tool_output_bytes),
        list_files_tool(root, max_tool_output_bytes),
    ]


__all__ = [
    "MAX_FILE_LINES",
    "MAX_GREP_MATCHES",
    "MAX_LISTED_FILES",
    "count_that_fits",
    "fit_result",
    "grep_tool",
    "list_files_tool",
    "read_file_tool",
    "read_only_tools",
    "serialized_size",
]
