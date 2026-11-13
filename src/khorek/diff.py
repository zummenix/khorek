"""Unified diff of a fix attempt's changed files, for the Reviewer.

The Reviewer sees the full change set the Fixer produced: every file the Fixer
modified or created, as a standard unified diff.  The diff is computed from the
per-issue :class:`~khorek.snapshots.Snapshotter`, which captured the issue's
file before the Fixer ran and every extra file the Fixer's mutating tools
touched.  Hunk headers (``@@ -a,b +c,d @@``) carry the line anchors; line
numbers refer to the post-fix file.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path

from khorek.files import read_text_file
from khorek.snapshots import Snapshotter


def format_file_diff(original: str, current: str, path: str, *, context: int = 3) -> str:
    """Return the unified diff of *original* vs *current* for *path*.

    The ``a/`` and ``b/`` prefixes label the old and new sides respectively.
    The diff is ``""`` when the two texts are identical.
    """
    lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=context,
        )
    )
    if not lines:
        return ""
    return "".join(lines)


def format_changes(project_root: Path, snapshotter: Snapshotter) -> str:
    """Return the concatenated unified diff of every changed captured file.

    Each changed file contributes one per-file diff (an identical file yields
    ``""`` and is omitted); the surviving diffs are joined with a blank line.
    Returns ``""`` when nothing changed.
    """
    root_real = Path(os.path.realpath(project_root))
    diffs: list[str] = []
    for change in snapshotter.changes():
        relative = change.path.relative_to(root_real).as_posix()
        current = read_text_file(change.path)
        original = (
            "" if change.original is None else change.original.decode("utf-8", errors="replace")
        )
        diff = format_file_diff(original, current, relative)
        if diff:
            diffs.append(diff)
    return "\n".join(diffs)


__all__ = ["format_changes", "format_file_diff"]
