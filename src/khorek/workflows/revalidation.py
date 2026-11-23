"""Post-fix revalidation of open issues against updated file contents."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from khorek.agents.aspect import AspectAgent, RevalidationReport
from khorek.config import AspectConfig
from khorek.context import AppContext
from khorek.db import Issue
from khorek.files import PathEscapeError, read_text_file
from khorek.llm import AgentError


@dataclass(frozen=True)
class RevalidationResult:
    """Counts of open issue rows refreshed or closed during revalidation."""

    refreshed: int = 0
    resolved: int = 0


class Revalidator:
    """Re-judge open issues of advanced aspects after a successful fix."""

    def __init__(self, app: AppContext, aspects: Mapping[str, AspectConfig]) -> None:
        self.app = app
        self._aspect_agent = AspectAgent(
            app.llm,
            project_root=app.project_root,
            config=app.config.agents.aspect,
            defaults=app.config.defaults,
        )
        self._aspects = dict(aspects)

    def revalidate(self, path: str, advanced: Collection[str]) -> RevalidationResult:
        """Re-validate the open issues of *advanced* aspects on *path*.

        Reads the post-fix contents and, for every advanced aspect that still has
        open issues, runs one revalidation call and applies its verdicts in
        place.  A read failure or an LLM error leaves the affected rows untouched
        and the fix still stands.
        """
        if not advanced:
            return RevalidationResult()
        try:
            content = read_text_file(self.app.project_root / path)
        except (OSError, PathEscapeError) as exc:
            self.app.reporter.message(
                f"khorek: cannot revalidate {path}: {exc}; sibling issues left unchanged"
            )
            return RevalidationResult()
        revalidated = 0
        resolved = 0
        for aspect_id in sorted(advanced):
            issues = self.app.database.open_issues_for(aspect_id, path)
            if not issues:
                continue
            aspect = self._aspects.get(aspect_id)
            if aspect is None:
                # No prompt for this aspect: cannot revalidate; leave as-is.
                continue
            report = self._run_one(aspect, path, content, issues)
            if report is None:
                continue
            revalidated_delta, resolved_delta = self._apply(
                aspect_id, path, content, issues, report
            )
            revalidated += revalidated_delta
            resolved += resolved_delta
        return RevalidationResult(refreshed=revalidated, resolved=resolved)

    def _run_one(
        self,
        aspect: AspectConfig,
        path: str,
        content: str,
        issues: list[Issue],
    ) -> RevalidationReport | None:
        """One revalidation call for *aspect*; ``None`` on an LLM error."""
        try:
            return self._aspect_agent.revalidate(
                aspect=aspect, path=path, content=content, issues=issues
            )
        except AgentError as exc:
            self.app.reporter.message(
                f"khorek: revalidation for {aspect.id} {path} failed "
                f"({exc.code}); its issues are left unchanged"
            )
            return None

    def _apply(
        self,
        aspect_id: str,
        path: str,
        content: str,
        issues: list[Issue],
        report: RevalidationReport,
    ) -> tuple[int, int]:
        """Apply a revalidation report in place; return ``(refreshed, closed)``.

        Conservative by construction (silence is never loss): a ``valid`` verdict
        refreshes the row in place with lines clamped to the current file and an
        optional description refresh — severity and ``retry_count`` untouched — a
        ``gone`` verdict with a concrete reason deletes the row, and ``uncertain``,
        a missing verdict, an unknown id, or a verdict that cannot be acted on
        leaves the row unchanged.  A verdict for id *N* touches only row *N*.
        """
        database = self.app.database
        reporter = self.app.reporter
        line_count = max(1, len(content.splitlines()))
        open_by_id = {issue.id: issue for issue in issues}
        revalidated = 0
        resolved = 0
        for verdict in report.results:
            issue = open_by_id.get(verdict.issue_id)
            if issue is None:
                continue  # unknown id or already closed earlier this round
            if verdict.status == "gone":
                reason = (verdict.reason or "").strip()
                if not reason:
                    continue  # gone without a reason: never loss, leave open
                database.delete_issue(issue.id)
                reporter.fix_resolved(issue.id, aspect_id, path, reason)
                resolved += 1
                open_by_id.pop(issue.id, None)
            elif verdict.status == "valid":
                start = issue.line_start
                end = issue.line_end
                fields: dict[str, object] = {}
                if verdict.line_start is not None and verdict.line_end is not None:
                    start = min(max(1, verdict.line_start), line_count)
                    end = min(max(start, verdict.line_end), line_count)
                    fields["line_start"] = start
                    fields["line_end"] = end
                description = (verdict.description or "").strip()
                if description:
                    fields["description"] = description
                if not fields:
                    continue  # a valid verdict with nothing to refresh
                database.update_issue(issue.id, **fields)
                reporter.fix_revalidated(issue.id, aspect_id, path, start, end)
                revalidated += 1
            # uncertain or an unrecognised status: leave the row unchanged.
        return revalidated, resolved


__all__ = ["RevalidationResult", "Revalidator"]
