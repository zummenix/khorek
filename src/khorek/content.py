"""Formatting helpers for presenting file contents to an LLM.

The one public helper, :func:`format_file_lines`, renders a slice of a file's
text as a numbered excerpt with a dashed rule around it and a footer when the
file is cut short.  It is the presentation layer for excerpts; the file
reading itself stays in :mod:`khorek.files`.
"""

from __future__ import annotations


def format_file_lines(
    relative_file_path: str,
    content: str,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    """Render a slice of *content* as a numbered, prefixed excerpt.

    *offset* is the 1-based first line to show (or ``None`` for the first
    line); *limit* is the maximum number of lines to show (or ``None`` for
    every line from *offset*).  Each line is prefixed with a right-aligned
    ``N`` padded to the width of the largest line number in the excerpt,
    followed by `` | `` so the separator stays on one column.

    The excerpt is wrapped in a dashed rule, and a footer is appended when
    the file is cut short (or when *offset* lands past the end of the file).
    *relative_file_path* is reported as given; callers must supply paths
    relative to the current working directory.  An empty file renders as a
    single blank line.
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
    shown = len(excerpt)
    remaining = total - shown

    header = f"File: {relative_file_path} (Lines {start}-{end} of {total})"
    separator = "-" * len(header)

    # Pad every number to the width of the last one in the window, so the
    # `` | `` separator lands on the same column on every line.
    parts = [header, separator]
    for index, line in enumerate(excerpt, start=start):
        parts.append(f"{index:>{len(str(end))}} | {line}")
    parts.append(separator)

    if remaining:
        parts.append(
            f"[Truncated: {remaining} more line{'s' if remaining != 1 else ''} remaining. "
            f"Use offset={start + shown} to read further.]"
        )

    return "\n".join(parts) + "\n"


__all__ = ["format_file_lines"]
