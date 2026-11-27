"""Formatting helpers for presenting file contents to an LLM."""

from __future__ import annotations


def line_count(text: str) -> int:
    """Return the number of lines in *text*, with a minimum of one."""
    return max(1, len(text.splitlines()))


def indent_block(text: str, width: int = 4) -> list[str]:
    """Indent each non-blank line after stripping its surrounding whitespace."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        lines.append(f"{' ' * width}{stripped}" if stripped else "")
    return lines


def format_content_lines(
    content: str,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    """Render a slice of *content* as numbered lines, with no framing.

    *offset* is the 1-based first line to show (or ``None`` for the first
    line); *limit* is the maximum number of lines to show (or ``None`` for
    every line from *offset*).  Each line is prefixed with a right-aligned
    ``N`` padded to the width of the largest line number in the window,
    followed by `` | `` so the separator stays on one column.

    An empty file renders as a single numbered line with an empty body.
    """
    lines = content.splitlines()
    if not lines:
        lines = [""]

    total = len(lines)
    start = 1 if offset is None or offset <= 0 else offset
    if limit is None or limit <= 0:
        end = total
    else:
        end = min(start + limit - 1, total)

    if start > total:
        start = total
        end = total

    excerpt = lines[start - 1 : end]

    # Pad every number to the width of the last one in the window, so the
    # `` | `` separator lands on the same column on every line.
    numbered = [
        f"{index:>{len(str(end))}} | {line}" for index, line in enumerate(excerpt, start=start)
    ]
    return "\n".join(numbered) + "\n"


def format_file_lines(relative_file_path: str, content: str) -> str:
    """Render the whole of *content* wrapped in a dashed-rule header.

    The header is ``File: <path> (Lines 1-<total> of <total>)`` followed by a
    dashed rule, then the numbered lines (see :func:`format_content_lines`),
    then a closing rule.  *relative_file_path* is reported as given; callers
    must supply paths relative to the current working directory.  An empty
    file renders as a single numbered line with an empty body.
    """
    total = line_count(content)
    body = format_content_lines(content)
    title = f"File: {relative_file_path} (Lines 1-{total} of {total})"
    separator = "-" * len(title)
    return f"{title}\n{separator}\n{body}{separator}\n"


__all__ = ["format_file_lines", "format_content_lines", "indent_block", "line_count"]
