"""Stdout-only progress reporting and run summaries."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

from khorek.content import indent_block
from khorek.db import Issue


@dataclass
class ExploreStats:
    """Counters for one explore run."""

    considered: int = 0
    skipped: int = 0
    explored: int = 0
    errored: int = 0
    issues_created: int = 0
    truncated: int = 0

    @property
    def outcome(self) -> str:
        if self.errored:
            return "errored"
        if self.issues_created or self.truncated:
            return "issues"
        return "clean"


@dataclass
class FixStats:
    """Counters for one fix run.

    ``revalidated`` counts sibling issue rows refreshed by a post-fix
    revalidation (a ``valid`` verdict that moved or re-described the row);
    ``resolved`` counts sibling rows a ``gone`` verdict closed.  Neither counts
    against the ``--limit`` attempt budget: the limit counts *attempted* issues.
    """

    attempted: int = 0
    fixed: int = 0
    revalidated: int = 0
    resolved: int = 0
    failed: int = 0
    rejected: int = 0
    skipped_stale: int = 0


@dataclass
class CombinedStats:
    """Counters for one explore-and-fix run (one unified end-of-run line)."""

    fixed: int = 0
    revalidated: int = 0
    resolved: int = 0
    failed: int = 0
    rejected: int = 0
    skipped_stale: int = 0
    remaining: int = 0
    created: int = 0
    pruned: int = 0
    cycles: int = 0


class Reporter:
    """Minimal deterministic stdout reporter (no logging, no timestamps)."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def message(self, text: str) -> None:
        print(text, file=self._stream, flush=True)

    def pair_skipped(self, aspect_id: str, path: str, reason: str) -> None:
        self.message(f"explore {aspect_id} {path}: skipped ({reason})")

    def pair_explored(self, aspect_id: str, path: str, created: int, outcome: str) -> None:
        self.message(f"explore {aspect_id} {path}: {created} issue(s) recorded ({outcome})")

    def pair_errored(self, aspect_id: str, path: str, code: str, error: str) -> None:
        self.message(f"explore {aspect_id} {path}: error [{code}]: {error}")

    def summary(self, stats: ExploreStats, *, pruned: int, open_issues: int) -> None:
        self.message(
            "explore summary: "
            f"{stats.considered} pairs "
            f"({stats.skipped} skipped, {stats.explored} explored, "
            f"{stats.errored} errored), "
            f"{stats.issues_created} issues created, "
            f"{pruned} pruned, "
            f"{open_issues} open issues; "
            f"outcome: {stats.outcome}"
        )

    # -- fix -----------------------------------------------------------

    def fix_attempted(self, issue_id: int, aspect_id: str, path: str, severity: str) -> None:
        self.message(f"fix #{issue_id} {aspect_id} {path}: attempting (severity {severity})")

    def fix_finished(self, issue_id: int, aspect_id: str, path: str) -> None:
        self.message(f"fix #{issue_id} {aspect_id} {path}: fixed")

    def fix_revalidated(
        self, issue_id: int, aspect_id: str, path: str, line_start: int, line_end: int
    ) -> None:
        """Report a sibling issue whose row a revalidation refreshed in place."""
        lines = str(line_start) if line_start == line_end else f"{line_start}-{line_end}"
        self.message(f"fix #{issue_id} {aspect_id} {path}: revalidated (lines {lines})")

    def fix_resolved(self, issue_id: int, aspect_id: str, path: str, reason: str) -> None:
        """Report a sibling issue a revalidation closed as ``gone``."""
        self.message(f"fix #{issue_id} {aspect_id} {path}: closed by revalidation ({reason})")

    def fix_report(self, issue: Issue, fix_summary: str, *, review_summary: str = "") -> None:
        """Print the per-issue problem/fix block after a successful fix.

        Each field is stripped first; an empty field omits its label and
        content, and when all fields are empty nothing is printed (the bare
        ``fix_finished`` line stands on its own).
        """
        lines: list[str] = []
        problem = issue.description.strip()
        if problem:
            lines.append("  problem:")
            lines.extend(indent_block(problem))
        summary = fix_summary.strip()
        if summary:
            lines.append("  fix:")
            lines.extend(indent_block(summary))
        review = review_summary.strip()
        if review:
            lines.append("  review:")
            lines.extend(indent_block(review))
        if lines:
            self.message("\n".join(lines))

    def fix_failed(self, issue_id: int, aspect_id: str, path: str, error: str) -> None:
        self.message(f"fix #{issue_id} {aspect_id} {path}: failed: {error}")

    def fix_skipped_stale(self, issue_id: int, aspect_id: str, path: str) -> None:
        self.message(f"fix #{issue_id} {aspect_id} {path}: skipped (stale hash)")

    def fix_summary(self, stats: FixStats, *, remaining: int) -> None:
        self.message(
            "fix summary: "
            f"{stats.attempted} attempted, "
            f"{stats.fixed} fixed, "
            f"{stats.revalidated} revalidated, "
            f"{stats.resolved} resolved, "
            f"{stats.failed} failed, "
            f"{stats.rejected} rejected, "
            f"{stats.skipped_stale} skipped-stale, "
            f"{remaining} open issues"
        )

    # -- explore-and-fix -----------------------------------------------

    def combined_summary(self, stats: CombinedStats) -> None:
        self.message(
            "explore-and-fix summary: "
            f"{stats.fixed} fixed, "
            f"{stats.revalidated} revalidated, "
            f"{stats.resolved} resolved, "
            f"{stats.failed} failed, "
            f"{stats.rejected} rejected, "
            f"{stats.skipped_stale} skipped-stale, "
            f"{stats.created} created, "
            f"{stats.remaining} remaining, "
            f"{stats.pruned} pruned, "
            f"{stats.cycles} cycle(s)"
        )


__all__ = ["CombinedStats", "ExploreStats", "FixStats", "Reporter"]
