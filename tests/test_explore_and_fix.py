"""Explore-and-fix workflow: drain/explore loop, budgets, guard, exit codes."""

from __future__ import annotations

from pathlib import Path

from khorek.db import Database
from khorek.workflows.fix import build_queue
from tests.conftest import (
    FakeModel,
    Project,
    explore_and_fix_args,
    make_project,
    revalidate,
    review,
    run_cli,
    seed_issue,
    seed_state,
    tool_message,
)

TRUE_VERIFY = [{"cmd": "true", "timeout": 10}]


def report(*issues: dict[str, object]) -> dict[str, object]:
    return tool_message("report_issues", {"issues": list(issues)})


def issue(
    description: str = "the comment is inconsistent",
    line_start: int = 1,
    line_end: int = 1,
    severity: str = "high",
) -> dict[str, object]:
    return {
        "description": description,
        "line_start": line_start,
        "line_end": line_end,
        "severity": severity,
    }


def edit(search: str, replace: str, path: str = "src/a.py") -> dict[str, object]:
    return tool_message("edit_file", {"path": path, "search": search, "replace": replace})


def fix_report(summary: str = "applied the fix", *, fixed: bool = True) -> dict[str, object]:
    return tool_message("report_fix_result", {"fixed": fixed, "summary": summary})


def issues(project: Project):
    with Database(project.db_path, project.root) as database:
        return database.list_issues()


def state(project: Project, path: str = "src/a.py"):
    with Database(project.db_path, project.root) as database:
        return database.get_state("comments", path)


def tool_names(request: dict[str, object]) -> set[str]:
    tools = request.get("tools")
    assert isinstance(tools, list)
    return {str(tool["function"]["name"]) for tool in tools}


def test_queue_is_drained_before_exploring(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n", "src/b.py": "b = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    issue_id = seed_issue(project)
    # src/b.py has no state row yet, so explore really calls the Aspect agent
    # for it after the queued issue is fixed (and reviewed).
    fake = FakeModel(
        [
            edit("value = 1", "value = 2"),
            fix_report("fixed it"),
            review(True, "good"),
            report(),
        ]
    )

    assert run_cli(fake, *explore_and_fix_args(project)) == 0

    # Fixer first, Aspect agent last: the queue is consumed before exploring.
    assert fake.chat_calls == 4
    assert "edit_file" in tool_names(fake.chat_requests[0])
    assert "report_issues" not in tool_names(fake.chat_requests[0])
    assert "report_issues" in tool_names(fake.chat_requests[3])
    assert "edit_file" not in tool_names(fake.chat_requests[3])
    assert "Issue #" in fake.chat_requests[0]["messages"][1]["content"]

    assert project.read("src/a.py") == "value = 2\n"
    assert issues(project) == []
    row = state(project)
    assert row is not None
    assert row.outcome == "clean"
    output = capsys.readouterr().out
    assert f"fix #{issue_id}" in output
    assert "explore comments src/b.py" in output
    assert "\nfix summary:" not in "\n" + output
    assert "explore-and-fix summary:" in output


def test_aspect_filter_restricts_both_drain_and_explore(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n", "src/b.py": "b = 1\n"},
        aspects=[
            {
                "id": "comments",
                "prompt": "Find comments that are inconsistent with the code.",
                "files": ["src/"],
            },
            {"id": "style", "prompt": "Find style issues.", "files": ["src/"]},
        ],
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues", path="src/a.py")
    comments_id = seed_issue(project, aspect_id="comments", path="src/a.py")
    style_id = seed_issue(project, aspect_id="style", path="src/b.py", description="style issue")
    fake = FakeModel(
        [
            edit("value = 1", "value = 2"),
            fix_report("fixed it"),
            review(True, "good"),
            report(),
        ]
    )

    assert run_cli(fake, *explore_and_fix_args(project, "--aspect", "comments")) == 0

    # One fix block for the comments issue, then one explore call for the
    # remaining comments pair: the style aspect is neither drained nor explored.
    assert fake.chat_calls == 4
    output = capsys.readouterr().out
    assert f"fix #{comments_id}" in output
    assert "explore comments src/b.py" in output
    assert "explore style" not in output
    assert "0 remaining" in output
    with Database(project.db_path, project.root) as database:
        remaining = {issue.id for issue in database.list_issues()}
    assert comments_id not in remaining
    assert style_id in remaining


def test_fix_block_streams_and_unified_summary_stays_last(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n", "src/b.py": "b = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    issue_id = seed_issue(project)
    fake = FakeModel(
        [edit("value = 1", "value = 2"), fix_report("fixed it"), review(True, "good"), report()]
    )

    assert run_cli(fake, *explore_and_fix_args(project)) == 0

    output = capsys.readouterr().out
    assert (
        f"fix #{issue_id} comments src/a.py: fixed\n"
        "  problem:\n"
        "    the comment is inconsistent\n"
        "  fix:\n"
        "    fixed it\n"
    ) in output
    last = [line for line in output.splitlines() if line]
    assert last and last[-1].startswith("explore-and-fix summary:")


def test_failed_issue_is_not_reattempted_and_remains_open(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    issue_id = seed_issue(project)
    fake = FakeModel([fix_report("cannot fix this", fixed=False)])

    assert run_cli(fake, *explore_and_fix_args(project)) == 1

    # Exactly one Fixer call: the failed issue never re-enters this run even
    # though the loop had budget left and the issue is still open.
    assert fake.chat_calls == 1
    stored = issues(project)
    assert [entry.id for entry in stored] == [issue_id]
    assert stored[0].retry_count == 1
    assert stored[0].last_error == "cannot fix this"
    output = capsys.readouterr().out
    assert "0 fixed, 0 revalidated, 0 resolved, 1 failed" in output
    assert "1 remaining" in output


def test_limit_caps_created_issues_independently(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "a = 1\n", "src/b.py": "b = 1\n"},
        verify=TRUE_VERIFY,
    )
    fake = FakeModel(
        [
            report(issue("a issue")),
            edit("a = 1", "a = 2"),
            fix_report("fixed a"),
            review(True, "good"),
        ]
    )

    assert run_cli(fake, *explore_and_fix_args(project, "--limit", "1")) == 0

    # One explore call only: the creation budget stops exploration after the
    # first pair, and the second file is never touched.
    assert fake.chat_calls == 4
    assert "report_issues" in tool_names(fake.chat_requests[0])
    assert "edit_file" in tool_names(fake.chat_requests[1])
    assert state(project, "src/b.py") is None
    assert issues(project) == []
    output = capsys.readouterr().out
    assert "1 created" in output
    assert "0 remaining" in output


def test_limit_caps_attempted_issues_independently(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "a = 1\n", "src/b.py": "b = 1\n"},
        verify=TRUE_VERIFY,
    )
    first = seed_issue(project, path="src/a.py", description="a is wrong")
    second = seed_issue(project, path="src/b.py", description="b is wrong")
    fake = FakeModel([edit("a = 1", "a = 2"), fix_report("fixed a"), review(True, "good")])

    assert run_cli(fake, *explore_and_fix_args(project, "--limit", "1")) == 1

    # The attempt budget is spent, so the run stops before exploring at all.
    assert fake.chat_calls == 3
    assert all("report_issues" not in tool_names(call) for call in fake.chat_requests)
    stored = issues(project)
    assert [entry.id for entry in stored] == [second]
    assert second != first
    output = capsys.readouterr().out
    assert "1 fixed" in output
    assert "1 remaining" in output


def test_revalidation_is_free_of_the_attempt_budget(tmp_path: Path) -> None:
    """In a combined run, a revalidation call does not consume the `--limit`."""
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\nbeta = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    first = seed_issue(project, description="alpha is wrong", line_start=1, line_end=1)
    second = seed_issue(project, description="beta is wrong", line_start=2, line_end=2)
    fake = FakeModel(
        [
            edit("alpha = 1", "alpha = 2"),
            fix_report("fixed alpha"),
            review(True, "good"),
            # the sibling is re-validated, free of the attempt budget
            revalidate({"issue_id": second, "status": "uncertain"}),
        ]
    )

    assert run_cli(fake, *explore_and_fix_args(project, "--limit", "1")) == 1

    # 3 calls to fix #1 + 1 revalidation call; the explore phase is skipped
    # because the attempt budget is spent, so the sibling is left re-validated.
    assert fake.chat_calls == 4
    assert any("report_revalidation" in tool_names(call) for call in fake.chat_requests)
    stored = issues(project)
    assert [entry.id for entry in stored] == [second]
    assert second != first


def test_loop_terminates_when_explore_creates_nothing(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    fake = FakeModel([report()])

    assert run_cli(fake, *explore_and_fix_args(project)) == 0

    assert fake.chat_calls == 1
    output = capsys.readouterr().out
    assert "0 created" in output
    assert "0 cycle(s)" in output
    assert "0 remaining" in output


def test_max_cycles_guard_stops_a_model_that_keeps_reporting(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        # Cap 1 makes the pair "truncated", so a successful fix deletes the
        # state row and the next explore revisits the pair: a model that keeps
        # re-reporting issues would loop without the guard.
        defaults={"max_issues_per_file": 1, "max_cycles": 2},
        verify=TRUE_VERIFY,
    )
    fake = FakeModel(
        [
            report(issue("first wrong comment"), issue("second wrong comment")),
            edit("value = 1", "value = 2"),
            fix_report("fixed once"),
            review(True, "good"),
            report(issue("first wrong comment"), issue("second wrong comment")),
        ]
    )

    assert run_cli(fake, *explore_and_fix_args(project)) == 1

    assert fake.chat_calls == 5
    output = capsys.readouterr().out
    assert "max_cycles=2" in output
    assert "2 cycle(s)" in output
    assert "1 remaining" in output
    assert len(issues(project)) == 1


def test_exit_code_zero_when_nothing_remains(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "# correct\nvalue = 1\n"},
        verify=TRUE_VERIFY,
    )
    fake = FakeModel([report()])

    assert run_cli(fake, *explore_and_fix_args(project)) == 0
    assert fake.chat_calls == 1
    assert issues(project) == []


def test_exit_code_one_when_an_issue_remains(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project)
    fake = FakeModel([fix_report(fixed=False, summary="still broken")])

    assert run_cli(fake, *explore_and_fix_args(project)) == 1
    assert len(issues(project)) == 1


def test_combined_summary_reports_unified_counters_and_pruned(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    # An aspect removed from the config is pruned before the workflow starts.
    with Database(project.db_path, project.root) as database:
        database.upsert_state("retired", "src/a.py", "sha", "issues")
        database.insert_issue(
            aspect_id="retired",
            path="src/a.py",
            description="old issue",
            line_start=1,
            line_end=1,
            severity="low",
        )
    fake = FakeModel([report()])

    assert run_cli(fake, *explore_and_fix_args(project)) == 0

    output = capsys.readouterr().out
    assert (
        "explore-and-fix summary: 0 fixed, 0 revalidated, 0 resolved, "
        "0 failed, 0 rejected, 0 skipped-stale, "
        "0 created, 0 remaining, 2 pruned, 0 cycle(s)"
    ) in output
    assert fake.chat_calls == 1


def test_build_queue_can_exclude_already_attempted_issues(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "a = 1\n", "src/b.py": "b = 1\n"},
    )
    first = seed_issue(project, path="src/a.py")
    second = seed_issue(project, path="src/b.py")

    with Database(project.db_path, project.root) as database:
        assert [entry.id for entry in build_queue(database)] == [first, second]
        assert [entry.id for entry in build_queue(database, exclude_ids={first})] == [second]
        assert build_queue(database, exclude_ids={first, second}) == []
