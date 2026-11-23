"""Unit tests for post-fix issue revalidation."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any, cast

from khorek.agents.aspect import RevalidationReport, RevalidationStatus, RevalidationVerdict
from khorek.config import load_config
from khorek.context import AppContext
from khorek.db import Database
from khorek.llm import AgentError
from khorek.report import Reporter
from khorek.workflows.revalidation import RevalidationResult, Revalidator
from tests.conftest import Project, make_project, seed_issue


class FakeAspectAgent:
    """Return a scripted report or raise without making an LLM call."""

    def __init__(self, result: RevalidationReport | Exception) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def revalidate(self, **kwargs: Any) -> RevalidationReport:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make_revalidator(project: Project, database: Database, reporter: Reporter) -> Revalidator:
    config = load_config(project.root)
    app = AppContext(
        project_root=project.root,
        config=config,
        database=database,
        llm=cast(Any, None),
        reporter=reporter,
    )
    return Revalidator(app, {aspect.id: aspect for aspect in config.aspects})


def verdict(
    issue_id: int,
    status: RevalidationStatus,
    **fields: Any,
) -> RevalidationVerdict:
    return RevalidationVerdict(issue_id=issue_id, status=status, **fields)


def test_valid_refreshes_row_and_clamps_lines_to_file(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project", files={"src/a.py": "one\ntwo\n"})
    issue_id = seed_issue(project, line_start=1, line_end=1, retry_count=2)
    with Database(project.db_path, project.root) as database:
        revalidator = make_revalidator(project, database, Reporter(StringIO()))
        fake = FakeAspectAgent(
            RevalidationReport(
                results=[
                    verdict(
                        issue_id,
                        "valid",
                        line_start=100,
                        line_end=101,
                        description="  refreshed description  ",
                    )
                ]
            )
        )
        cast(Any, revalidator)._aspect_agent = fake

        result = revalidator.revalidate("src/a.py", {"comments"})

        issue = database.get_issue(issue_id)
    assert result == RevalidationResult(refreshed=1, resolved=0)
    assert issue is not None
    assert (issue.line_start, issue.line_end) == (2, 2)
    assert issue.description == "refreshed description"
    assert issue.retry_count == 2
    assert len(fake.calls) == 1


def test_gone_with_reason_deletes_row(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    issue_id = seed_issue(project)
    with Database(project.db_path, project.root) as database:
        revalidator = make_revalidator(project, database, Reporter(StringIO()))
        fake = FakeAspectAgent(
            RevalidationReport(results=[verdict(issue_id, "gone", reason="the fix corrected it")])
        )
        cast(Any, revalidator)._aspect_agent = fake

        result = revalidator.revalidate("src/a.py", {"comments"})

        issue = database.get_issue(issue_id)
    assert result == RevalidationResult(refreshed=0, resolved=1)
    assert issue is None


def test_conservative_verdicts_leave_rows_open(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    no_reason = seed_issue(project, description="gone without reason")
    uncertain = seed_issue(project, description="uncertain")
    missing = seed_issue(project, description="missing verdict")
    unknown_target = seed_issue(project, description="unknown id target")
    with Database(project.db_path, project.root) as database:
        revalidator = make_revalidator(project, database, Reporter(StringIO()))
        fake = FakeAspectAgent(
            RevalidationReport(
                results=[
                    verdict(no_reason, "gone", reason="  "),
                    verdict(uncertain, "uncertain"),
                    verdict(99999, "gone", reason="unknown id"),
                ]
            )
        )
        cast(Any, revalidator)._aspect_agent = fake

        result = revalidator.revalidate("src/a.py", {"comments"})

        remaining_ids = {issue.id for issue in database.open_issues_for("comments", "src/a.py")}
    assert result == RevalidationResult()
    assert remaining_ids == {no_reason, uncertain, missing, unknown_target}


def test_unreadable_file_returns_zero_counts_without_agent_call(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    seed_issue(project)
    project.remove("src/a.py")
    with Database(project.db_path, project.root) as database:
        output = StringIO()
        revalidator = make_revalidator(project, database, Reporter(output))
        fake = FakeAspectAgent(RevalidationReport(results=[]))
        cast(Any, revalidator)._aspect_agent = fake

        result = revalidator.revalidate("src/a.py", {"comments"})
    assert result == RevalidationResult()
    assert fake.calls == []
    assert "cannot revalidate src/a.py" in output.getvalue()


def test_agent_error_returns_zero_counts_and_leaves_issue_open(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    issue_id = seed_issue(project)
    with Database(project.db_path, project.root) as database:
        output = StringIO()
        revalidator = make_revalidator(project, database, Reporter(output))
        fake = FakeAspectAgent(AgentError("loop", "failed"))
        cast(Any, revalidator)._aspect_agent = fake

        result = revalidator.revalidate("src/a.py", {"comments"})

        issue = database.get_issue(issue_id)
    assert result == RevalidationResult()
    assert issue is not None
    assert len(fake.calls) == 1
    assert "revalidation for comments src/a.py failed (loop)" in output.getvalue()


def test_no_open_issues_for_aspect_skips_agent_call(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    with Database(project.db_path, project.root) as database:
        revalidator = make_revalidator(project, database, Reporter(StringIO()))
        fake = FakeAspectAgent(RevalidationReport(results=[]))
        cast(Any, revalidator)._aspect_agent = fake

        result = revalidator.revalidate("src/a.py", {"comments"})

    assert result == RevalidationResult()
    assert fake.calls == []
