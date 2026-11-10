"""Stdout-only progress reporting and run summaries."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO


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


__all__ = ["ExploreStats", "Reporter"]
