"""Explore workflow: aspect x file pairs with skip rules and validation.

Implemented as a LangGraph ``StateGraph`` with plain function nodes and a
conditional edge that loops back to the pair processor.  No checkpointer: the
SQLite database is the source of truth.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field, replace
from typing import Any, cast

from langgraph.graph import END, StateGraph

from khorek.agents.aspect import AspectAgent, IssueReport
from khorek.agents.explorer import ExplorerAgent
from khorek.config import AspectConfig
from khorek.context import AppContext
from khorek.files import enumerate_files, read_text_file, sha256_file
from khorek.llm import AgentError
from khorek.report import ExploreStats

_SEVERITIES = frozenset({"low", "medium", "high"})


@dataclass(frozen=True)
class Pair:
    """One (aspect, project-relative file) unit of work."""

    aspect: AspectConfig
    path: str


@dataclass(frozen=True)
class ValidatedIssue:
    """A reported issue after deterministic repair, ready to store."""

    description: str
    line_start: int
    line_end: int
    severity: str


@dataclass
class ExploreState:
    """State flowing through the explore graph."""

    pairs: list[Pair] = field(default_factory=list)
    index: int = 0
    stats: ExploreStats = field(default_factory=ExploreStats)
    stop: bool = False


def line_count(content: str) -> int:
    """Line count used for clamping (never less than 1)."""
    return max(1, len(content.splitlines()))


def validate_issues(
    report: IssueReport, *, lines: int, cap: int
) -> tuple[list[ValidatedIssue], bool]:
    """Deterministically repair a report and trim it to *cap*.

    Clamping happens before trimming (so a repaired out-of-range issue that
    survives the cap is stored); empty descriptions are dropped.  Returns the
    kept issues and whether the cap truncated the list.
    """
    n = max(1, lines)
    validated: list[ValidatedIssue] = []
    for reported in report.issues:
        description = reported.description.strip()
        if not description:
            continue
        line_start = min(max(1, reported.line_start), n)
        line_end = min(max(reported.line_end, line_start), n)
        severity = reported.severity if reported.severity in _SEVERITIES else "low"
        validated.append(
            ValidatedIssue(
                description=description,
                line_start=line_start,
                line_end=line_end,
                severity=severity,
            )
        )
    truncated = len(validated) > cap
    return validated[:cap], truncated


class _Explorer:
    """Per-run agents and graph construction."""

    def __init__(self, app: AppContext) -> None:
        self.app = app
        self.explorer_agent = ExplorerAgent(
            app.llm,
            project_root=app.project_root,
            config=app.config.agents.explorer,
            defaults=app.config.defaults,
        )
        self.aspect_agent = AspectAgent(
            app.llm,
            project_root=app.project_root,
            config=app.config.agents.aspect,
            defaults=app.config.defaults,
            explorer=self.explorer_agent,
        )

    def enumerate_pairs(self, aspect_ids: Collection[str] | None = None) -> list[Pair]:
        """Enumerate (aspect, file) pairs, optionally restricted to *aspect_ids*.

        An empty or ``None`` filter explores every configured aspect.
        """
        wanted = set(aspect_ids) if aspect_ids else None
        pairs: list[Pair] = []
        for aspect in self.app.config.aspects:
            if wanted is not None and aspect.id not in wanted:
                continue
            files = enumerate_files(
                self.app.project_root,
                aspect.files,
                max_bytes=self.app.config.defaults.max_file_bytes,
            )
            pairs.extend(Pair(aspect=aspect, path=path.as_posix()) for path in files)
        return pairs

    def process_pair(self, pair: Pair, *, remaining_budget: int | None = None) -> ExploreStats:
        """Process one pair and return its stats delta.

        ``remaining_budget`` is the number of issues the run may still create
        (``None`` means unlimited).  It participates in the effective cap so a
        pair can never overshoot ``--limit``.
        """
        app = self.app
        stats = ExploreStats()
        aspect = pair.aspect
        relative = pair.path
        absolute = app.project_root / relative

        try:
            current_sha = sha256_file(absolute)
        except OSError as exc:
            app.reporter.pair_errored(aspect.id, relative, "read", str(exc))
            return replace(stats, errored=1)

        # The state row is the pair's version token.  A matching row means the
        # pair was explored at exactly the current content, so its open issues
        # (if any) are still valid and the pair is skipped.  An absent or
        # drifted row means the pair is re-explored and its old issues replaced.
        state = app.database.get_state(aspect.id, relative)
        if state is not None and state.sha256 == current_sha:
            app.reporter.pair_skipped(aspect.id, relative, state.outcome)
            return replace(stats, skipped=1)

        # Stale or never-explored pair: replace the state and any open issues
        # with a fresh exploration.
        app.database.delete_state(aspect.id, relative)
        app.database.delete_issues_for(aspect.id, relative)

        try:
            content = read_text_file(absolute)
        except OSError as exc:
            app.reporter.pair_errored(aspect.id, relative, "read", str(exc))
            return replace(stats, errored=1)

        try:
            report = self.aspect_agent.review(aspect=aspect, path=relative, content=content)
        except AgentError as exc:
            app.reporter.pair_errored(aspect.id, relative, exc.code, str(exc))
            return replace(stats, errored=1)

        cap = (
            aspect.max_issues_per_file
            if aspect.max_issues_per_file is not None
            else app.config.defaults.max_issues_per_file
        )
        if remaining_budget is not None:
            cap = min(cap, remaining_budget)
        kept, was_truncated = validate_issues(report, lines=line_count(content), cap=cap)
        for issue in kept:
            app.database.insert_issue(
                aspect_id=aspect.id,
                path=relative,
                description=issue.description,
                line_start=issue.line_start,
                line_end=issue.line_end,
                severity=issue.severity,
            )

        if was_truncated:
            outcome = "truncated"
        elif kept:
            outcome = "issues"
        else:
            outcome = "clean"
        app.database.upsert_state(aspect.id, relative, current_sha, outcome)
        app.reporter.pair_explored(aspect.id, relative, len(kept), outcome)

        return replace(
            stats,
            explored=1,
            issues_created=len(kept),
            truncated=1 if was_truncated else 0,
        )

    def build_graph(self, pairs: list[Pair], limit: int) -> StateGraph[ExploreState]:
        explore = self

        def enumerate_node(state: ExploreState) -> dict[str, Any]:
            del state
            return {
                "pairs": pairs,
                "index": 0,
                "stats": ExploreStats(considered=len(pairs)),
                "stop": False,
            }

        def process_node(state: ExploreState) -> dict[str, Any]:
            index = state.index
            stats = state.stats
            if index >= len(state.pairs):
                return {"stop": True}
            if limit > 0 and stats.issues_created >= limit:
                return {"stop": True}
            remaining = None if limit <= 0 else limit - stats.issues_created
            delta = explore.process_pair(state.pairs[index], remaining_budget=remaining)
            combined = ExploreStats(
                considered=stats.considered,
                skipped=stats.skipped + delta.skipped,
                explored=stats.explored + delta.explored,
                errored=stats.errored + delta.errored,
                issues_created=stats.issues_created + delta.issues_created,
                truncated=stats.truncated + delta.truncated,
            )
            return {
                "index": index + 1,
                "stats": combined,
                "stop": limit > 0 and combined.issues_created >= limit,
            }

        def route(state: ExploreState) -> str:
            if state.stop or state.index >= len(state.pairs):
                return END
            return "process"

        graph: StateGraph[ExploreState] = StateGraph(ExploreState)
        graph.add_node("enumerate", enumerate_node)
        graph.add_node("process", process_node)
        graph.set_entry_point("enumerate")
        graph.add_edge("enumerate", "process")
        graph.add_conditional_edges("process", route, {"process": "process", END: END})
        return graph


def run_explore(
    app: AppContext,
    limit: int = 0,
    *,
    aspect_ids: Collection[str] | None = None,
) -> ExploreStats:
    """Run the explore workflow and return its counters.

    ``aspect_ids`` restricts the run to the named aspects (empty/``None``
    explores every configured aspect).
    """
    explore = _Explorer(app)
    pairs = explore.enumerate_pairs(aspect_ids=aspect_ids)
    graph = explore.build_graph(pairs, limit)
    compiled = graph.compile()
    final = cast(dict[str, Any], compiled.invoke(ExploreState()))
    return cast(ExploreStats, final["stats"])


__all__ = [
    "ExploreState",
    "ExploreStats",
    "Pair",
    "ValidatedIssue",
    "line_count",
    "run_explore",
    "validate_issues",
]
