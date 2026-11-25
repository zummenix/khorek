"""Issue triage: staleness, prune, dismiss, delete, and list/show rendering.

Plain database and filesystem operations: no LLM calls, no LangGraph.  The
CLI layer owns locking, argument parsing, and output; the functions here take
an open :class:`~khorek.db.Database` and project root.

Staleness: an issue is *stale* when the file it was recorded on no longer
matches its pair's (aspect, path) state row — the single version token the
explore wrote (the file is missing, or its content changed out-of-band).
A missing state row counts as fresh: an ``issues delete`` survivor is a
re-explore, not a prune.  Dismissing a stale issue is meaningful: stamping
the pair clean with the *current* sha says "I reviewed this file as it is
now".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from khorek.content import indent_block
from khorek.db import Database, Issue
from khorek.files import sha256_file

# Staleness classifications.
FRESH = "fresh"
CHANGED = "changed"
MISSING = "missing"

# Human status labels for ``issues show``.
STATUS_FRESH = "fresh"
STATUS_CHANGED = "stale (file changed since recorded)"
STATUS_MISSING = "stale (file missing)"


@dataclass(frozen=True)
class IssueOutcome:
    """The result of processing one issue id in a dismiss/delete batch.

    ``found`` is False for an unknown id.  ``action`` is the suffix the
    reporter prints after the ``→``: ``"pair marked clean"``,
    ``"pair removed (file not found)"``, ``"state cleared"``, or ``""`` (no
    suffix, e.g. siblings remain after a dismiss).
    """

    issue_id: int
    found: bool
    aspect_id: str = ""
    path: str = ""
    lines: str = ""
    action: str = ""


def _lines(issue: Issue) -> str:
    """``<line>`` or ``<start>-<end>`` for the issue's line span."""
    if issue.line_start == issue.line_end:
        return str(issue.line_start)
    return f"{issue.line_start}-{issue.line_end}"


def _pair_state_sha(database: Database, issue: Issue) -> str | None:
    """The pair's state-row sha256, or None when the row is absent."""
    row = database.get_state(issue.aspect_id, issue.path)
    return row.sha256 if row is not None else None


def _classify(database: Database, project_root: Path, issue: Issue) -> str:
    """Classify one issue as FRESH / CHANGED / MISSING.

    Staleness is a property of the (aspect, path) pair: MISSING when the file
    is gone, FRESH when its current sha matches the pair's state row (or the
    row is absent — a re-explore, not a prune), else CHANGED.
    """
    current = _current_sha(project_root, issue)
    if current is None:
        return MISSING
    recorded = _pair_state_sha(database, issue)
    if recorded is None or recorded == current:
        return FRESH
    return CHANGED


def issue_status(database: Database, project_root: Path, issue: Issue) -> str:
    """Human status label for one issue (the ``status:`` field of show)."""
    status = _classify(database, project_root, issue)
    if status == FRESH:
        return STATUS_FRESH
    if status == CHANGED:
        return STATUS_CHANGED
    return STATUS_MISSING


def staleness(database: Database, project_root: Path) -> dict[int, str]:
    """Map every open issue id to FRESH / CHANGED / MISSING.

    Staleness is a property of the (aspect, path) pair: each distinct path is
    hashed at most once and each distinct pair's state row is read at most
    once.  A missing state row counts as FRESH (a re-explore, not a prune).
    """
    hashes: dict[str, str | None] = {}
    state_shas: dict[tuple[str, str], str | None] = {}
    result: dict[int, str] = {}
    for issue in database.list_issues():
        if issue.path not in hashes:
            hashes[issue.path] = _current_sha(project_root, issue)
        current = hashes[issue.path]
        if current is None:
            result[issue.id] = MISSING
            continue
        key = (issue.aspect_id, issue.path)
        if key not in state_shas:
            state_shas[key] = _pair_state_sha(database, issue)
        recorded = state_shas[key]
        result[issue.id] = FRESH if recorded is None or recorded == current else CHANGED
    return result


def prune_stale(database: Database, project_root: Path) -> int:
    """Delete all stale issue rows (file changed or missing); return the count."""
    statuses = staleness(database, project_root)
    stale_ids = [issue_id for issue_id, status in statuses.items() if status != FRESH]
    return database.delete_issues(stale_ids)


def _current_sha(project_root: Path, issue: Issue) -> str | None:
    """The file's current sha256, or None when it is missing or unreadable."""
    absolute = project_root / issue.path
    if not absolute.is_file():
        return None
    try:
        return sha256_file(absolute)
    except OSError:
        return None


def dismiss(database: Database, project_root: Path, ids: Sequence[int]) -> list[IssueOutcome]:
    """Dismiss issue ids as not relevant, processing them in order.

    Per id: delete the issue row; if the file no longer exists, also delete
    the pair's state row; otherwise, when no open issues remain on the
    (aspect, path) pair, stamp it clean with the file's *current* sha.  While
    siblings remain the state row is left untouched.
    """
    outcomes: list[IssueOutcome] = []
    for issue_id in ids:
        issue = database.get_issue(issue_id)
        if issue is None:
            outcomes.append(IssueOutcome(issue_id=issue_id, found=False))
            continue
        database.delete_issue(issue_id)
        current_sha = _current_sha(project_root, issue)
        if current_sha is None:
            database.delete_state(issue.aspect_id, issue.path)
            action = "pair removed (file not found)"
        elif not database.open_issues_for(issue.aspect_id, issue.path):
            database.upsert_state(issue.aspect_id, issue.path, current_sha, "clean")
            action = "pair marked clean"
        else:
            action = ""
        outcomes.append(
            IssueOutcome(
                issue_id=issue_id,
                found=True,
                aspect_id=issue.aspect_id,
                path=issue.path,
                lines=_lines(issue),
                action=action,
            )
        )
    return outcomes


def delete(database: Database, project_root: Path, ids: Sequence[int]) -> list[IssueOutcome]:
    """Delete issue ids completely, processing them in order.

    The pair's state row is removed unconditionally (even while sibling
    issues remain), so the next explore re-examines the file from scratch.
    """
    del project_root
    outcomes: list[IssueOutcome] = []
    for issue_id in ids:
        issue = database.get_issue(issue_id)
        if issue is None:
            outcomes.append(IssueOutcome(issue_id=issue_id, found=False))
            continue
        database.delete_issue(issue_id)
        database.delete_state(issue.aspect_id, issue.path)
        outcomes.append(
            IssueOutcome(
                issue_id=issue_id,
                found=True,
                aspect_id=issue.aspect_id,
                path=issue.path,
                lines=_lines(issue),
                action="state cleared",
            )
        )
    return outcomes


# -- rendering -------------------------------------------------------------


def _pad(value: str, width: int) -> str:
    """Right-pad *value* to a column of *width* plus a >= 2-space gap.

    *width* is the max field width of the column, so every field's successor
    starts in the same column (dynamic table width, like ``ps``).
    """
    return value.ljust(width + 2)


def render_list(issues: Sequence[Issue], *, stale: Mapping[int, str] | None) -> str:
    """Render the issues list block (rows plus footer), or ``no open issues``.

    Rows are rendered in the given order (callers pass issues already sorted
    by id, as returned by :meth:`~khorek.db.Database.list_issues`) and
    aligned in dynamic-width columns (like ``ps``):
    ``<id> | <created date>  <aspect>  <severity>  <path>:<lines>``.  When
    *stale* is given (stale-showing mode), rows whose status is not FRESH are
    suffixed with ``[stale]`` and the footer counts them.
    """
    if not issues:
        return "no open issues"

    def is_stale(issue: Issue) -> bool:
        return stale is not None and stale.get(issue.id, FRESH) != FRESH

    stale_count = sum(1 for issue in issues if is_stale(issue))
    date_width = max(len(issue.created_at[:10]) for issue in issues)
    aspect_width = max(len(issue.aspect_id) for issue in issues)
    severity_width = max(len(issue.severity) for issue in issues)

    blocks: list[str] = []
    for issue in issues:
        header = (
            f"{issue.id} | "
            f"{_pad(issue.created_at[:10], date_width)}"
            f"{_pad(issue.aspect_id, aspect_width)}"
            f"{_pad(issue.severity, severity_width)}"
            f"{issue.path}:{_lines(issue)}"
        )
        if is_stale(issue):
            header += "  [stale]"
        lines = [header]
        if issue.retry_count > 0:
            lines.append(f"    Retried: {issue.retry_count}")
        lines.extend(indent_block(issue.description))
        blocks.append("\n".join(lines))

    footer = f"{len(issues)} open issue" + ("" if len(issues) == 1 else "s")
    if stale_count:
        footer += f" ({stale_count} stale)"
    return "\n\n".join(blocks) + "\n\n" + footer


def render_show(issue: Issue, *, status: str) -> str:
    """Render the full detail block for one issue."""
    lines = [f"#{issue.id}  {issue.aspect_id}  {issue.path}:{_lines(issue)}"]
    fields = [
        ("severity", issue.severity),
        ("status", status),
        ("retries", str(issue.retry_count)),
        ("created", issue.created_at),
        ("updated", issue.updated_at),
    ]
    width = max(len(label) + 1 for label, _ in fields)
    for label, value in fields:
        lines.append(f"{(label + ':').ljust(width)}  {value}")
    if issue.last_error:
        lines.append("last_error:")
        lines.extend(indent_block(issue.last_error))
    lines.append("")
    lines.append(issue.description)
    return "\n".join(lines)


__all__ = [
    "CHANGED",
    "FRESH",
    "MISSING",
    "STATUS_CHANGED",
    "STATUS_FRESH",
    "STATUS_MISSING",
    "IssueOutcome",
    "delete",
    "dismiss",
    "issue_status",
    "prune_stale",
    "render_list",
    "render_show",
    "staleness",
]
