"""Fix workflow: dequeue open issues, fix, verify, and roll back on failure.

Implemented as a LangGraph ``StateGraph`` with plain function nodes and
conditional edges (``select -> fix -> verify -> success|fail -> select``).  No
checkpointer: the SQLite database is the source of truth.

Queues and round semantics follow the plan exactly:

* the queue is ordered by ``retry_count ASC, severity DESC, created_at ASC``
  and computed once per run;
* a ``fixed: false`` report or a Fixer agent error ends the issue's attempts
  immediately; only a *verification* failure consumes one of the remaining
  rounds and feeds the Verifier summary into the next round;
* a ``fixed: true`` report that leaves every snapshotted file byte-identical is
  treated as a failed fix;
* success deletes the issue, advances the stored hash of sibling open issues on
  the same path (so a second issue in the file is fixable), and marks the pair
  clean only when no open issue remains for (aspect, path);
* failure restores every snapshot, bumps ``retry_count``, records
  ``last_error``, keeps the issue open, and continues with the next issue.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

from langgraph.graph import END, StateGraph

from khorek.agents.fixer import FixerAgent
from khorek.agents.verifier import VerifierAgent
from khorek.context import AppContext
from khorek.db import Database, Issue
from khorek.files import PathEscapeError, read_text_file, sha256_file
from khorek.llm import AgentError
from khorek.report import FixStats
from khorek.snapshots import Snapshotter
from khorek.verify import Failure, run_commands
from khorek.workflows.explore import line_count

MAX_LAST_ERROR_CHARS = 2000
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
_UNKNOWN_SEVERITY_RANK = len(_SEVERITY_RANK)


def build_queue(database: Database) -> list[Issue]:
    """Return open issues in fix order (computed once per run)."""
    return sorted(
        database.list_issues(),
        key=lambda issue: (
            issue.retry_count,
            _SEVERITY_RANK.get(issue.severity, _UNKNOWN_SEVERITY_RANK),
            issue.created_at,
            issue.id,
        ),
    )


def trim_error(text: str, limit: int = MAX_LAST_ERROR_CHARS) -> str:
    """Trim whitespace and bound an error/summary string."""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return "..." + stripped[-limit:]


def fallback_failure_summary(failures: Sequence[Failure]) -> str:
    """Raw failure text used when the Verifier itself fails."""
    if not failures:
        return "verification failed"
    first = failures[0]
    status = "timed out" if first.exit_code is None else f"exited with code {first.exit_code}"
    detail = f"verification command failed: {first.cmd} ({status})"
    if first.output_tail:
        detail = f"{detail}\n{first.output_tail}"
    return trim_error(detail)


@dataclass
class FixState:
    """State flowing through the fix graph (one issue at a time)."""

    queue: list[Issue] = field(default_factory=list)
    index: int = 0
    stats: FixStats = field(default_factory=FixStats)
    stop: bool = False
    current: Issue | None = None
    snapshotter: Snapshotter | None = None
    pre_fix_sha: str = ""
    content: str = ""
    new_sha: str = ""
    lines_after: int = 1
    rounds: int = 0
    previous_failure: str | None = None
    verified: bool = False
    fix_fixed: bool = False
    fix_summary: str = ""
    fix_error: str | None = None
    last_error: str | None = None


class _FixWorkflow:
    """Per-run agents, queue, and graph construction."""

    def __init__(self, app: AppContext, limit: int) -> None:
        self.app = app
        self.limit = limit
        self.queue = build_queue(app.database)
        self.max_attempts = app.config.agents.fixer.max_attempts
        self.verifier = VerifierAgent(app.llm, config=app.config.agents.verifier)
        self.recursion_limit = 100 + len(self.queue) * (2 * self.max_attempts + 4)

    # -- helpers -------------------------------------------------------

    @property
    def _database(self) -> Database:
        return self.app.database

    def _under_limit(self, state: FixState) -> bool:
        return self.limit <= 0 or state.stats.attempted < self.limit

    def _reset_issue_fields(self, **updates: Any) -> dict[str, Any]:
        reset: dict[str, Any] = {
            "current": None,
            "snapshotter": None,
            "pre_fix_sha": "",
            "content": "",
            "new_sha": "",
            "lines_after": 1,
            "rounds": 0,
            "previous_failure": None,
            "verified": False,
            "fix_fixed": False,
            "fix_summary": "",
            "fix_error": None,
            "last_error": None,
        }
        reset.update(updates)
        return reset

    # -- nodes ---------------------------------------------------------

    def _prepare(self, state: FixState) -> dict[str, Any]:
        del state
        return {
            "queue": self.queue,
            "index": 0,
            "stats": FixStats(),
            "stop": False,
            **self._reset_issue_fields(),
        }

    def _select(self, state: FixState) -> dict[str, Any]:
        if state.stop or not self._under_limit(state) or state.index >= len(state.queue):
            return {"stop": True}

        queued = state.queue[state.index]
        next_index = state.index + 1
        issue = self._database.get_issue(queued.id)
        if issue is None:
            return {"index": next_index}

        absolute = self.app.project_root / issue.path
        try:
            current_sha = sha256_file(absolute)
        except OSError:
            current_sha = None

        # Fix staleness: the file changed outside khorek; drop the issue.
        if current_sha is None or issue.sha256 != current_sha:
            self._database.delete_issue(issue.id)
            self.app.reporter.fix_skipped_stale(issue.id, issue.aspect_id, issue.path)
            return {
                "index": state.index + 1,
                "stats": replace(
                    state.stats,
                    attempted=state.stats.attempted + 1,
                    skipped_stale=state.stats.skipped_stale + 1,
                ),
            }

        snapshotter = Snapshotter(self.app.project_root)
        stats = replace(state.stats, attempted=state.stats.attempted + 1)
        try:
            snapshotter.capture(absolute)
            content = read_text_file(absolute)
        except (OSError, PathEscapeError) as exc:
            self.app.reporter.fix_attempted(issue.id, issue.aspect_id, issue.path, issue.severity)
            return {
                "index": next_index,
                "stats": stats,
                "current": issue,
                "snapshotter": snapshotter,
                "last_error": f"cannot prepare {issue.path}: {exc}",
            }

        self.app.reporter.fix_attempted(issue.id, issue.aspect_id, issue.path, issue.severity)
        return {
            "index": next_index,
            "stats": stats,
            "current": issue,
            "snapshotter": snapshotter,
            "pre_fix_sha": issue.sha256,
            "content": content,
            "rounds": 0,
            "previous_failure": None,
            "verified": False,
            "fix_fixed": False,
            "fix_summary": "",
            "fix_error": None,
            "last_error": None,
        }

    def _fix(self, state: FixState) -> dict[str, Any]:
        issue = state.current
        snapshotter = state.snapshotter
        assert issue is not None and snapshotter is not None

        absolute = self.app.project_root / issue.path
        try:
            content = read_text_file(absolute)
        except OSError:
            content = state.content

        fixer = FixerAgent(
            self.app.llm,
            project_root=self.app.project_root,
            config=self.app.config.agents.fixer,
            defaults=self.app.config.defaults,
            snapshotter=snapshotter,
        )
        try:
            outcome = fixer.fix(
                issue,
                issue.path,
                content,
                previous_failure=state.previous_failure,
            )
        except AgentError as exc:
            return {
                "fix_fixed": False,
                "fix_error": f"fixer [{exc.code}]: {exc}",
            }

        if not outcome.fixed:
            return {
                "fix_fixed": False,
                "fix_summary": outcome.summary or "fixer reported no fix",
            }
        if not snapshotter.has_changes():
            return {
                "fix_fixed": False,
                "fix_summary": outcome.summary,
                "fix_error": "fixer reported success but made no file changes",
            }
        return {
            "fix_fixed": True,
            "fix_summary": outcome.summary,
            "content": content,
        }

    def _verify(self, state: FixState) -> dict[str, Any]:
        issue = state.current
        assert issue is not None
        result = run_commands(self.app.project_root, self.app.config.verify)
        if result.passed:
            absolute = self.app.project_root / issue.path
            try:
                content = read_text_file(absolute)
                new_sha = sha256_file(absolute)
            except OSError as exc:
                return {
                    "verified": False,
                    "last_error": f"cannot read fixed file {issue.path}: {exc}",
                    "rounds": self.max_attempts,
                }
            return {
                "verified": True,
                "new_sha": new_sha,
                "lines_after": line_count(content),
                "last_error": None,
            }

        failures = result.failures
        try:
            summary = self.verifier.summarise(failures)
        except AgentError as exc:
            summary = fallback_failure_summary(failures)
            self.app.reporter.message(
                f"fix #{issue.id} {issue.aspect_id} {issue.path}: "
                f"verifier error [{exc.code}]: {exc}; using raw failure output"
            )
        if not summary.strip():
            summary = fallback_failure_summary(failures)
        summary = trim_error(summary)

        rounds = state.rounds + 1
        status = "timeout" if failures and failures[0].exit_code is None else "failed"
        self.app.reporter.message(
            f"fix #{issue.id} {issue.aspect_id} {issue.path}: "
            f"verification {status} "
            f"(round {rounds}/{self.max_attempts})"
        )
        return {
            "verified": False,
            "rounds": rounds,
            "previous_failure": summary,
            "last_error": summary,
        }

    def _success(self, state: FixState) -> dict[str, Any]:
        issue = state.current
        assert issue is not None
        database = self._database

        self._database.delete_issue(issue.id)
        count = max(1, state.lines_after)
        for sibling in self._database.open_issues_for_path(issue.path):
            if sibling.sha256 != state.pre_fix_sha:
                continue
            line_start = min(max(1, sibling.line_start), count)
            line_end = min(max(sibling.line_end, line_start), count)
            database.update_issue(
                sibling.id,
                sha256=state.new_sha,
                line_start=line_start,
                line_end=line_end,
            )

        if not database.open_issues_for(issue.aspect_id, issue.path):
            row = database.get_state(issue.aspect_id, issue.path)
            if row is not None:
                if row.outcome == "truncated":
                    database.delete_state(issue.aspect_id, issue.path)
                else:
                    database.upsert_state(issue.aspect_id, issue.path, state.new_sha, "clean")

        self.app.reporter.fix_finished(issue.id, issue.aspect_id, issue.path)
        return {
            "stats": replace(state.stats, fixed=state.stats.fixed + 1),
            **self._reset_issue_fields(),
        }

    def _fail(self, state: FixState) -> dict[str, Any]:
        issue = state.current
        assert issue is not None
        error = trim_error(state.fix_error or state.last_error or state.fix_summary or "fix failed")
        if state.snapshotter is not None:
            try:
                state.snapshotter.restore()
            except OSError as exc:
                self.app.reporter.message(
                    f"fix #{issue.id} {issue.aspect_id} {issue.path}: restore failed: {exc}"
                )
                error = trim_error(f"{error}\nrestore failed: {exc}")

        self._database.update_issue(
            issue.id,
            retry_count=issue.retry_count + 1,
            last_error=error,
        )
        self.app.reporter.fix_failed(issue.id, issue.aspect_id, issue.path, error)
        return {
            "stats": replace(state.stats, failed=state.stats.failed + 1),
            **self._reset_issue_fields(),
        }

    # -- routing -------------------------------------------------------

    def _route_select(self, state: FixState) -> str:
        if state.stop:
            return END
        if state.current is not None:
            return "fix" if state.snapshotter is not None else "fail"
        return "select"

    def _route_fix(self, state: FixState) -> str:
        return "verify" if state.fix_fixed else "fail"

    def _route_verify(self, state: FixState) -> str:
        if state.verified:
            return "success"
        if state.rounds >= self.max_attempts:
            return "fail"
        return "fix"

    # -- graph ---------------------------------------------------------

    def build_graph(self) -> StateGraph[FixState]:
        graph: StateGraph[FixState] = StateGraph(FixState)
        graph.add_node("prepare", self._prepare)
        graph.add_node("select", self._select)
        graph.add_node("fix", self._fix)
        graph.add_node("verify", self._verify)
        graph.add_node("success", self._success)
        graph.add_node("fail", self._fail)
        graph.set_entry_point("prepare")
        graph.add_edge("prepare", "select")
        graph.add_conditional_edges(
            "select",
            self._route_select,
            {"select": "select", "fix": "fix", "fail": "fail", END: END},
        )
        graph.add_conditional_edges(
            "fix",
            self._route_fix,
            {"verify": "verify", "fail": "fail"},
        )
        graph.add_conditional_edges(
            "verify",
            self._route_verify,
            {"success": "success", "fix": "fix", "fail": "fail"},
        )
        graph.add_edge("success", "select")
        graph.add_edge("fail", "select")
        return graph


def run_fix(app: AppContext, limit: int = 0) -> FixStats:
    """Run the fix workflow and return its counters."""
    workflow = _FixWorkflow(app, limit)
    compiled = workflow.build_graph().compile()
    final = cast(
        dict[str, Any],
        compiled.invoke(FixState(), {"recursion_limit": workflow.recursion_limit}),
    )
    stats = cast(FixStats, final["stats"])
    app.reporter.fix_summary(stats, remaining=app.database.count_issues())
    return stats


__all__ = [
    "MAX_LAST_ERROR_CHARS",
    "FixState",
    "build_queue",
    "fallback_failure_summary",
    "run_fix",
    "trim_error",
]
