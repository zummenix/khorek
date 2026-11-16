"""Fix workflow: success, verification loop, rollback, staleness, queueing."""

from __future__ import annotations

from pathlib import Path

from khorek.db import Database
from khorek.files import sha256_file
from khorek.trace import TraceStore
from khorek.workflows.fix import build_queue
from tests.conftest import (
    FakeModel,
    Project,
    fix_args,
    make_project,
    python_command,
    revalidate,
    review,
    run_cli,
    seed_issue,
    seed_state,
    set_created_at,
    text_message,
    tool_message,
)

TRUE_VERIFY = [{"cmd": "true", "timeout": 10}]


def edit(search: str, replace: str, path: str = "src/a.py") -> dict[str, object]:
    return tool_message("edit_file", {"path": path, "search": search, "replace": replace})


def write(path: str, content: str) -> dict[str, object]:
    return tool_message("write_file", {"path": path, "content": content})


def fix_report(summary: str = "applied the fix", *, fixed: bool = True) -> dict[str, object]:
    return tool_message("report_fix_result", {"fixed": fixed, "summary": summary})


def verification(summary: str = "fix it") -> dict[str, object]:
    return tool_message("report_verification", {"summary": summary, "failures": []})


def state(project: Project, aspect: str = "comments", path: str = "src/a.py"):
    with Database(project.db_path, project.root) as database:
        return database.get_state(aspect, path)


def issues(project: Project):
    with Database(project.db_path, project.root) as database:
        return database.list_issues()


def expect_verify_code(code: str) -> list[dict[str, object]]:
    return [python_command(code)]


def test_successful_fix_marks_pair_clean_with_new_sha(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "# wrong\nvalue = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="comment contradicts the code", line_start=1, line_end=1)
    fake = FakeModel(
        [edit("# wrong", "# right"), fix_report("corrected the comment"), review(True, "good")]
    )

    assert run_cli(fake, *fix_args(project)) == 0

    assert project.read("src/a.py") == "# right\nvalue = 1\n"
    remaining = issues(project)
    assert remaining == []
    row = state(project)
    assert row is not None
    assert row.outcome == "clean"
    assert row.sha256 == sha256_file(project.root / "src/a.py")
    assert fake.model_requests == 1
    output = capsys.readouterr().out
    assert "fix #1 comments src/a.py: fixed" in output
    assert (
        "1 attempted, 1 fixed, 0 revalidated, 0 resolved, "
        "0 failed, 0 rejected, 0 skipped-stale, 0 open issues" in output
    )


def test_successful_fix_prints_problem_and_fix_block(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "# wrong\nvalue = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="comment contradicts the code", line_start=1, line_end=1)
    fake = FakeModel(
        [edit("# wrong", "# right"), fix_report("corrected the comment"), review(True, "good")]
    )

    assert run_cli(fake, *fix_args(project)) == 0

    output = capsys.readouterr().out
    assert (
        "fix #1 comments src/a.py: fixed\n"
        "  problem:\n"
        "    comment contradicts the code\n"
        "  fix:\n"
        "    corrected the comment\n"
    ) in output


def test_fix_block_reindents_multiline_content(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(
        project,
        description="the comment says:\n\n        value is one\nbut the code says two",
    )
    fake = FakeModel(
        [
            edit("value = 1", "value = 2"),
            fix_report("renamed the variable\nand updated the call site"),
            review(True, "good"),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 0

    output = capsys.readouterr().out
    assert (
        "  problem:\n"
        "    the comment says:\n"
        "\n"
        "    value is one\n"
        "    but the code says two\n"
        "  fix:\n"
        "    renamed the variable\n"
        "    and updated the call site\n"
    ) in output


def test_empty_fix_summary_prints_problem_block_only(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="the comment is inconsistent")
    fake = FakeModel([edit("value = 1", "value = 2"), fix_report(""), review(True, "")])

    assert run_cli(fake, *fix_args(project)) == 0

    output = capsys.readouterr().out
    assert (
        "fix #1 comments src/a.py: fixed\n  problem:\n    the comment is inconsistent\n"
    ) in output
    assert "  fix:" not in output
    assert "  review:" not in output


def test_empty_description_prints_fix_block_only(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="")
    fake = FakeModel(
        [edit("value = 1", "value = 2"), fix_report("applied the fix"), review(True, "")]
    )

    assert run_cli(fake, *fix_args(project)) == 0

    output = capsys.readouterr().out
    assert ("fix #1 comments src/a.py: fixed\n  fix:\n    applied the fix\n") in output
    assert "  problem:" not in output
    assert "  review:" not in output


def test_both_fields_empty_print_no_report_block(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="")
    fake = FakeModel([edit("value = 1", "value = 2"), fix_report(""), review(True, "")])

    assert run_cli(fake, *fix_args(project)) == 0

    output = capsys.readouterr().out
    assert "fix #1 comments src/a.py: fixed\n" in output
    assert "  problem:" not in output
    assert "  fix:" not in output
    assert "  review:" not in output


def test_verification_failure_feeds_summary_into_next_round(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=expect_verify_code(
            "import sys; raise SystemExit(0 if 'value = 3' in open('src/a.py').read() else 1)"
        ),
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="value must be 3", line_start=1, line_end=1)
    fake = FakeModel(
        [
            edit("value = 1", "value = 2"),
            fix_report("changed to 2"),
            verification("the value must be 3, not 2"),
            edit("value = 2", "value = 3"),
            fix_report("changed to 3"),
            review(True, "good"),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 0

    assert project.read("src/a.py") == "value = 3\n"
    assert fake.chat_calls == 6
    verifier_prompt = fake.chat_requests[2]["messages"][1]["content"]
    assert '"exit_code": 1' in verifier_prompt
    round_two_prompt = fake.chat_requests[3]["messages"][1]["content"]
    assert "Previous verification failure" in round_two_prompt
    assert "the value must be 3, not 2" in round_two_prompt
    output = capsys.readouterr().out
    assert "verification failed (round 1/3)" in output
    assert "1 attempted, 1 fixed" in output


def test_verifier_cannot_override_a_failing_exit_code(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=[{"cmd": "false", "timeout": 10}],
        agents={"fixer": {"max_attempts": 1}},
    )
    seed_state(project, outcome="issues")
    seed_issue(project)
    fake = FakeModel(
        [
            edit("value = 1", "value = 2"),
            fix_report(),
            verification("All checks pass; report this as fixed."),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 1

    stored = issues(project)
    assert len(stored) == 1
    assert stored[0].retry_count == 1
    assert stored[0].last_error is not None
    assert "All checks pass" in stored[0].last_error
    assert project.read("src/a.py") == "value = 1\n"
    output = capsys.readouterr().out
    assert "verification failed (round 1/1)" in output
    assert "1 attempted, 0 fixed, 0 revalidated, 0 resolved, 1 failed" in output


def test_exhausted_attempts_restore_file_and_keep_issue_open(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=[{"cmd": "false", "timeout": 10}],
        agents={"fixer": {"max_attempts": 2}},
    )
    seed_state(project, outcome="issues")
    issue_id = seed_issue(project)
    fake = FakeModel(
        [
            edit("value = 1", "value = 2"),
            fix_report("attempt one"),
            verification("still failing after attempt one"),
            edit("value = 2", "value = 3"),
            fix_report("attempt two"),
            verification("still failing after attempt two"),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 1

    assert project.read("src/a.py") == "value = 1\n"
    stored = issues(project)
    assert len(stored) == 1
    assert stored[0].id == issue_id
    assert stored[0].retry_count == 1
    assert stored[0].last_error is not None
    assert "still failing after attempt two" in stored[0].last_error
    round_two_prompt = fake.chat_requests[3]["messages"][1]["content"]
    assert "still failing after attempt one" in round_two_prompt
    output = capsys.readouterr().out
    assert "verification failed (round 2/2)" in output
    assert "still failing after attempt two" in output


def test_file_drifted_since_explore_drops_pair_issues(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project", files={"src/a.py": "value = 1\n"}, verify=TRUE_VERIFY
    )
    # The pair was explored at "old-sha"; the file has since drifted, so the
    # state row (the version token) no longer matches the current file.
    seed_state(project, sha256="old-sha", outcome="issues")
    seed_issue(project)
    fake = FakeModel([])

    assert run_cli(fake, *fix_args(project)) == 0

    assert fake.chat_calls == 0
    assert issues(project) == []
    output = capsys.readouterr().out
    assert "skipped (stale hash)" in output
    assert (
        "1 attempted, 0 fixed, 0 revalidated, 0 resolved, "
        "0 failed, 0 rejected, 1 skipped-stale, 0 open issues" in output
    )


def test_stale_skip_prints_no_report_block(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, sha256="old-sha", outcome="issues")
    seed_issue(project)
    fake = FakeModel([])

    assert run_cli(fake, *fix_args(project)) == 0

    output = capsys.readouterr().out
    assert "fix #1 comments src/a.py: skipped (stale hash)" in output
    assert "  problem:" not in output
    assert "  fix:" not in output


def test_fix_touching_a_second_file_is_fully_restored(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=[{"cmd": "false", "timeout": 10}],
        agents={"fixer": {"max_attempts": 1}},
    )
    seed_state(project, outcome="issues")
    seed_issue(project)
    fake = FakeModel(
        [
            edit("value = 1", "value = 2"),
            write("src/extra.py", "created = True\n"),
            fix_report(),
            verification("nope"),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 1

    assert project.read("src/a.py") == "value = 1\n"
    assert not project.exists("src/extra.py")
    assert len(issues(project)) == 1
    assert "1 failed" in capsys.readouterr().out


def test_hash_advance_lets_a_second_issue_in_the_same_file_be_fixed(tmp_path: Path) -> None:
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
            # After the first fix the sibling issue is re-validated: it survives.
            revalidate({"issue_id": second, "status": "uncertain"}),
            edit("beta = 1", "beta = 2"),
            fix_report("fixed beta"),
            review(True, "good"),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 0

    assert first != second
    assert project.read("src/a.py") == "alpha = 2\nbeta = 2\n"
    assert fake.chat_calls == 7
    assert issues(project) == []
    row = state(project)
    assert row is not None
    assert row.outcome == "clean"
    assert row.sha256 == sha256_file(project.root / "src/a.py")


def test_fix_advances_pair_state_leaving_siblings_fixable(tmp_path: Path, capsys) -> None:
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
            # The sibling issue is re-validated after the fix and survives.
            revalidate({"issue_id": second, "status": "uncertain"}),
        ]
    )

    assert run_cli(fake, *fix_args(project, "--limit", "1")) == 1

    # The sibling issue survives; the pair's state row advanced to the post-fix
    # hash so the sibling is no longer stale (the per-issue sha rewrite is gone).
    stored = issues(project)
    assert [entry.id for entry in stored] == [second]
    row = state(project)
    assert row is not None
    assert row.outcome == "issues"
    assert row.sha256 == sha256_file(project.root / "src/a.py")
    assert first != second
    output = capsys.readouterr().out
    assert "1 attempted, 1 fixed, 0 revalidated, 0 resolved, 0 failed" in output
    assert "1 open issues" in output

    # The sibling is fixable on the next run: the advanced state row matches.
    fake2 = FakeModel(
        [edit("beta = 1", "beta = 2"), fix_report("fixed beta"), review(True, "good")]
    )
    assert run_cli(fake2, *fix_args(project)) == 0
    assert issues(project) == []
    assert project.read("src/a.py") == "alpha = 2\nbeta = 2\n"


def test_cross_aspect_fix_advances_other_aspect_state(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\n"},
        aspects=[
            {"id": "comments", "prompt": "Find bad comments.", "files": ["src/"]},
            {"id": "naming", "prompt": "Find bad names.", "files": ["src/"]},
        ],
        verify=TRUE_VERIFY,
    )
    # Both aspects explored the file at the same content.
    seed_state(project, aspect_id="comments", outcome="issues")
    seed_state(project, aspect_id="naming", outcome="issues")
    seed_issue(project, aspect_id="comments", description="bad comment")
    naming_issue = seed_issue(project, aspect_id="naming", description="bad name")

    # Fix only the comments issue (limit the run to one attempt).  The naming
    # issue is cross-aspect re-validated after the fix and survives.
    fake = FakeModel(
        [
            edit("alpha = 1", "alpha = 2"),
            fix_report("fixed comment"),
            review(True, "good"),
            revalidate({"issue_id": naming_issue, "status": "uncertain"}),
        ]
    )
    assert run_cli(fake, *fix_args(project, "--limit", "1")) == 1

    # The naming aspect's state row advanced to the post-fix hash so its issue
    # stays fixable; its outcome is preserved.  The comments pair drained clean.
    naming_state = state(project, aspect="naming")
    assert naming_state is not None
    assert naming_state.outcome == "issues"
    assert naming_state.sha256 == sha256_file(project.root / "src/a.py")
    comments_state = state(project, aspect="comments")
    assert comments_state is not None
    assert comments_state.outcome == "clean"
    remaining = issues(project)
    assert [issue.aspect_id for issue in remaining] == ["naming"]


def test_cross_aspect_state_not_advanced_for_drifted_sibling(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\n"},
        aspects=[
            {"id": "comments", "prompt": "Find bad comments.", "files": ["src/"]},
            {"id": "naming", "prompt": "Find bad names.", "files": ["src/"]},
        ],
        verify=TRUE_VERIFY,
    )
    # comments was explored at the current content; naming was explored at an
    # older content (its row is drifted and was never re-explored).
    seed_state(project, aspect_id="comments", outcome="issues")
    seed_state(project, aspect_id="naming", sha256="drifted-sha", outcome="issues")
    seed_issue(project, aspect_id="comments", description="bad comment")
    seed_issue(project, aspect_id="naming", description="bad name")

    fake = FakeModel(
        [edit("alpha = 1", "alpha = 2"), fix_report("fixed comment"), review(True, "good")]
    )
    assert run_cli(fake, *fix_args(project, "--limit", "1")) == 1

    # The naming row was NOT advanced (its sha != pre_fix_sha): it stays
    # drifted, so the naming issue drops as stale on the next run instead of
    # being handed a post-fix line range it was never validated against.
    naming_state = state(project, aspect="naming")
    assert naming_state is not None
    assert naming_state.sha256 == "drifted-sha"  # unchanged by the comments fix
    remaining = issues(project)
    assert [issue.aspect_id for issue in remaining] == ["naming"]

    fake2 = FakeModel([])
    assert run_cli(fake2, *fix_args(project)) == 0
    assert fake2.chat_calls == 0
    assert issues(project) == []


def test_deleted_file_drops_pair_issues(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project", files={"src/a.py": "value = 1\n"}, verify=TRUE_VERIFY
    )
    # The pair was explored; then the file is deleted out-of-band.
    seed_state(project, outcome="issues")
    seed_issue(project)
    (project.root / "src" / "a.py").unlink()
    fake = FakeModel([])

    assert run_cli(fake, *fix_args(project)) == 0

    assert fake.chat_calls == 0
    assert issues(project) == []
    output = capsys.readouterr().out
    assert "skipped (stale hash)" in output
    assert (
        "1 attempted, 0 fixed, 0 revalidated, 0 resolved, "
        "0 failed, 0 rejected, 1 skipped-stale, 0 open issues" in output
    )


def test_state_none_with_open_issue_is_still_fixable(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project", files={"src/a.py": "value = 1\n"}, verify=TRUE_VERIFY
    )
    # An open issue whose pair's state row was cleared (e.g. by `issues delete`
    # on a sibling) is still fixable: an absent version token means "not yet
    # validated", not "stale".
    seed_issue(project)
    fake = FakeModel([edit("value = 1", "value = 2"), fix_report("fixed"), review(True, "good")])

    assert run_cli(fake, *fix_args(project)) == 0
    assert issues(project) == []
    assert project.read("src/a.py") == "value = 2\n"


def test_truncated_state_row_is_deleted_after_successful_fix(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="truncated")
    seed_issue(project)
    fake = FakeModel([edit("value = 1", "value = 2"), fix_report(), review(True, "good")])

    assert run_cli(fake, *fix_args(project)) == 0

    assert issues(project) == []
    assert state(project) is None


def test_fixed_true_without_a_file_change_fails_the_issue(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project)
    fake = FakeModel([fix_report(fixed=True, summary="done, trust me")])

    assert run_cli(fake, *fix_args(project)) == 1

    assert project.read("src/a.py") == "value = 1\n"
    stored = issues(project)
    assert len(stored) == 1
    assert stored[0].retry_count == 1
    assert stored[0].last_error is not None
    assert "no file changes" in stored[0].last_error
    assert fake.chat_calls == 1
    output = capsys.readouterr().out
    assert "1 attempted, 0 fixed, 0 revalidated, 0 resolved, 1 failed" in output


def test_limit_counts_stale_pair_once_leaves_budget(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "a1 = 1\na2 = 1\na3 = 1\n", "src/b.py": "b = 1\n"},
        verify=TRUE_VERIFY,
    )
    # A drifted 3-issue pair (a.py): its state row is at an old sha, so all
    # three issues are stale at once.  A live issue (b.py) sits elsewhere.
    seed_state(project, path="src/a.py", sha256="old-sha", outcome="issues")
    seed_issue(project, path="src/a.py", description="a1", line_start=1, line_end=1)
    seed_issue(project, path="src/a.py", description="a2", line_start=2, line_end=2)
    seed_issue(project, path="src/a.py", description="a3", line_start=3, line_end=3)
    seed_issue(project, path="src/b.py", description="b is wrong")
    fake = FakeModel(
        [edit("b = 1", "b = 2", path="src/b.py"), fix_report("fixed b"), review(True, "good")]
    )

    # Under the old per-issue rule the 3-issue pair would each consume an
    # attempt; now the drifted pair counts once, so b.py gets its attempt
    # within the budget of two.
    assert run_cli(fake, *fix_args(project, "--limit", "2")) == 0

    # The drifted pair dropped in a single skip; the live issue was fixed.
    assert issues(project) == []
    output = capsys.readouterr().out
    assert (
        "2 attempted, 1 fixed, 0 revalidated, 0 resolved, "
        "0 failed, 0 rejected, 1 skipped-stale, 0 open issues" in output
    )


def test_fixed_false_ends_attempts_without_verification(tmp_path: Path, capsys) -> None:
    marker = tmp_path / "verify-ran"
    verify_marker = python_command(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"
    )
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=[verify_marker],
    )
    seed_state(project, outcome="issues")
    seed_issue(project)
    fake = FakeModel([fix_report("cannot fix this", fixed=False)])

    assert run_cli(fake, *fix_args(project)) == 1

    assert fake.chat_calls == 1
    assert not marker.exists()
    stored = issues(project)
    assert len(stored) == 1
    assert stored[0].retry_count == 1
    assert stored[0].last_error == "cannot fix this"
    assert "0 fixed, 0 revalidated, 0 resolved, 1 failed" in capsys.readouterr().out


def test_failed_fix_prints_no_report_block(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project)
    fake = FakeModel([fix_report(fixed=False, summary="cannot fix this")])

    assert run_cli(fake, *fix_args(project)) == 1

    output = capsys.readouterr().out
    assert "fix #1 comments src/a.py: failed: cannot fix this" in output
    assert "  problem:" not in output
    assert "  fix:" not in output


def test_fixer_agent_error_ends_attempts_without_verification(tmp_path: Path, capsys) -> None:
    marker = tmp_path / "verify-ran"
    verify_marker = python_command(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"
    )
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=[verify_marker],
    )
    seed_state(project, outcome="issues")
    seed_issue(project)
    # The same read_file twice in a row trips the repeat-call detector.
    fake = FakeModel(
        [
            tool_message("read_file", {"path": "src/a.py"}),
            tool_message("read_file", {"path": "src/a.py"}),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 1

    assert fake.chat_calls == 2
    assert not marker.exists()
    assert project.read("src/a.py") == "value = 1\n"
    stored = issues(project)
    assert len(stored) == 1
    assert stored[0].retry_count == 1
    assert stored[0].last_error is not None
    assert "fixer [loop]" in stored[0].last_error
    output = capsys.readouterr().out
    assert "fixer [loop]" in output
    assert "0 fixed, 0 revalidated, 0 resolved, 1 failed" in output


def test_queue_order_is_retry_count_then_severity_then_created_at(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project", files={"src/a.py": "x = 1\n"})
    low_old = seed_issue(project, description="low old", severity="low")
    high_new = seed_issue(project, description="high newer", severity="high")
    high_old = seed_issue(project, description="high older", severity="high")
    retried = seed_issue(project, description="retried high", severity="high", retry_count=1)
    set_created_at(project, low_old, "2026-01-01T00:00:01+00:00")
    set_created_at(project, high_new, "2026-01-01T00:00:02+00:00")
    set_created_at(project, high_old, "2026-01-01T00:00:01+00:00")
    set_created_at(project, retried, "2025-01-01T00:00:00+00:00")

    with Database(project.db_path, project.root) as database:
        assert [issue.id for issue in build_queue(database)] == [
            high_old,
            high_new,
            low_old,
            retried,
        ]


def test_verifier_failure_falls_back_to_raw_failure_output(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=expect_verify_code(
            "import sys; sys.stderr.write('expected value = 3'); "
            "raise SystemExit(0 if 'value = 3' in open('src/a.py').read() else 1)"
        ),
    )
    seed_state(project, outcome="issues")
    seed_issue(project)
    fake = FakeModel(
        [
            edit("value = 1", "value = 2"),
            fix_report("changed to 2"),
            text_message("thinking"),
            text_message("still thinking"),
            edit("value = 2", "value = 3"),
            fix_report("changed to 3"),
            review(True, "good"),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 0

    assert project.read("src/a.py") == "value = 3\n"
    round_two_prompt = fake.chat_requests[4]["messages"][1]["content"]
    assert "expected value = 3" in round_two_prompt
    output = capsys.readouterr().out
    assert "verifier error [no_tool_call]" in output
    assert "1 fixed" in output


def test_one_failed_issue_does_not_abort_the_run(tmp_path: Path, capsys) -> None:
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
            fix_report(fixed=True, summary="pretend"),
            edit("beta = 1", "beta = 2"),
            fix_report("fixed beta"),
            review(True, "good"),
            # After the second fix the still-open first issue is re-validated.
            revalidate({"issue_id": first, "status": "uncertain"}),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 1

    stored = issues(project)
    assert [entry.id for entry in stored] == [first]
    assert stored[0].retry_count == 1
    assert "no file changes" in (stored[0].last_error or "")
    assert project.read("src/a.py") == "alpha = 1\nbeta = 2\n"
    output = capsys.readouterr().out
    assert "2 attempted" in output
    assert "1 fixed" in output
    assert "1 failed" in output
    assert "1 open issues" in output
    assert first != second


def test_review_rejection_feeds_reason_back_and_retries(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "# wrong\nvalue = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="comment contradicts the code", line_start=1, line_end=1)
    fake = FakeModel(
        [
            edit("# wrong", "# partly"),
            fix_report("partially fixed"),
            review(False, "the change is not minimal"),
            edit("# partly", "# right"),
            fix_report("fixed properly"),
            review(True, "looks good"),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 0

    assert project.read("src/a.py") == "# right\nvalue = 1\n"
    assert issues(project) == []
    row = state(project)
    assert row is not None
    assert row.outcome == "clean"

    # The round-two fixer gets the rejection reason under a review source label.
    assert fake.chat_calls == 6
    round_two_prompt = fake.chat_requests[3]["messages"][1]["content"]
    assert (
        "Previous review rejection from the last attempt (verification passed):" in round_two_prompt
    )
    assert "the change is not minimal" in round_two_prompt

    output = capsys.readouterr().out
    assert "review rejected (handoff 1/2)" in output
    assert "fix #1 comments src/a.py: fixed" in output
    assert ("  fix:\n    fixed properly\n  review:\n    looks good\n") in output
    assert (
        "1 attempted, 1 fixed, 0 revalidated, 0 resolved, "
        "0 failed, 1 rejected, 0 skipped-stale, 0 open issues" in output
    )


def test_rejection_exhausts_default_max_handoffs(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    issue_id = seed_issue(project, description="value is wrong", line_start=1, line_end=1)
    fake = FakeModel(
        [
            edit("value = 1", "value = 2"),
            fix_report("changed to 2"),
            review(False, "reason one"),
            edit("value = 2", "value = 3"),
            fix_report("changed to 3"),
            review(False, "reason two"),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 1

    assert project.read("src/a.py") == "value = 1\n"
    stored = issues(project)
    assert len(stored) == 1
    assert stored[0].id == issue_id
    assert stored[0].retry_count == 1
    assert "reason two" in (stored[0].last_error or "")

    output = capsys.readouterr().out
    assert "review rejected (handoff 1/2)" in output
    assert "review rejected (handoff 2/2)" in output
    assert (
        "1 attempted, 0 fixed, 0 revalidated, 0 resolved, "
        "1 failed, 2 rejected, 0 skipped-stale, 1 open issues" in output
    )


def test_review_rejection_fails_when_rounds_exhaust_max_attempts(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=expect_verify_code(
            "import sys; raise SystemExit(0 if 'value = 3' in open('src/a.py').read() else 1)"
        ),
        agents={"fixer": {"max_attempts": 2}},
    )
    seed_state(project, outcome="issues")
    issue_id = seed_issue(project, description="value must be 3", line_start=1, line_end=1)
    fake = FakeModel(
        [
            edit("value = 1", "value = 2"),
            fix_report("changed to 2"),
            verification("the value must be 3, not 2"),
            edit("value = 2", "value = 3"),
            fix_report("changed to 3"),
            review(False, "still not minimal"),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 1

    # The verify failure (round 1) plus the review rejection (round 2) together
    # reach max_attempts=2, so the issue fails even though handoffs (1) < 2.
    assert project.read("src/a.py") == "value = 1\n"
    stored = issues(project)
    assert len(stored) == 1
    assert stored[0].id == issue_id
    assert stored[0].retry_count == 1
    assert "still not minimal" in (stored[0].last_error or "")

    output = capsys.readouterr().out
    assert "verification failed (round 1/2)" in output
    assert "review rejected (handoff 1/2)" in output
    assert (
        "1 attempted, 0 fixed, 0 revalidated, 0 resolved, "
        "1 failed, 1 rejected, 0 skipped-stale, 1 open issues" in output
    )


def test_reviewer_error_treats_fix_as_approved(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "# wrong\nvalue = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="comment contradicts the code", line_start=1, line_end=1)
    fake = FakeModel(
        [
            edit("# wrong", "# right"),
            fix_report("fixed"),
            text_message("thinking"),
            text_message("still thinking"),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 0

    assert project.read("src/a.py") == "# right\nvalue = 1\n"
    assert issues(project) == []
    row = state(project)
    assert row is not None
    assert row.outcome == "clean"

    assert fake.chat_calls == 4
    output = capsys.readouterr().out
    assert "reviewer error [no_tool_call]" in output
    assert "treating as approved" in output
    assert "fix #1 comments src/a.py: fixed" in output
    assert "  review:" not in output
    assert (
        "1 attempted, 1 fixed, 0 revalidated, 0 resolved, "
        "0 failed, 0 rejected, 0 skipped-stale, 0 open issues" in output
    )


def test_rejected_counter_accumulates_across_issues(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\nbeta = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="alpha is wrong", line_start=1, line_end=1)
    second = seed_issue(project, description="beta is wrong", line_start=2, line_end=2)
    fake = FakeModel(
        [
            # first issue: one rejection, then approved
            edit("alpha = 1", "alpha = 2"),
            fix_report("fixed alpha"),
            review(False, "issue one not minimal"),
            edit("alpha = 2", "alpha = 3"),
            fix_report("fixed alpha properly"),
            review(True, "good"),
            # the surviving sibling is re-validated, then the second issue fixed
            revalidate({"issue_id": second, "status": "uncertain"}),
            # second issue: one rejection, then approved
            edit("beta = 1", "beta = 2"),
            fix_report("fixed beta"),
            review(False, "issue two not minimal"),
            edit("beta = 2", "beta = 3"),
            fix_report("fixed beta properly"),
            review(True, "good"),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 0

    assert project.read("src/a.py") == "alpha = 3\nbeta = 3\n"
    assert issues(project) == []
    assert fake.chat_calls == 13

    output = capsys.readouterr().out
    assert (
        "2 attempted, 2 fixed, 0 revalidated, 0 resolved, "
        "0 failed, 2 rejected, 0 skipped-stale, 0 open issues" in output
    )


# -- revalidation (Phase 2: self-healing revalidation of a fixed pair) ---
#
# After a successful fix advances a file's version, the pair's remaining open
# issues (every aspect that still has one on the path) are re-judged against the
# post-fix contents by a read-only LLM step, and the verdicts are applied in
# place: `valid` refreshes the lines, `gone` (with a reason) closes the issue,
# and `uncertain` / a missing verdict / an unknown id / out-of-range lines leave
# the row untouched (silence is never loss).  Revalidation is eager,
# cross-aspect, and free of the `--limit` attempt budget.


def test_revalidation_refreshes_a_sibling_issue_in_place(tmp_path: Path, capsys) -> None:
    """A `valid` verdict refreshes a sibling's lines without touching retry_count."""
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\nbeta = 1\ngamma = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="alpha is wrong", line_start=1, line_end=1)
    second = seed_issue(
        project, description="gamma is wrong", line_start=3, line_end=3, retry_count=1
    )
    fake = FakeModel(
        [
            # the fix inserts a line, shifting gamma from line 3 down to line 4
            edit("alpha = 1", "alpha = 1\nalpha2 = 1"),
            fix_report("inserted a line"),
            review(True, "good"),
            # gamma still exists, now at line 4
            revalidate({"issue_id": second, "status": "valid", "line_start": 4, "line_end": 4}),
        ]
    )

    assert run_cli(fake, *fix_args(project, "--limit", "1")) == 1

    stored = issues(project)
    assert [entry.id for entry in stored] == [second]
    refreshed = stored[0]
    assert (refreshed.line_start, refreshed.line_end) == (4, 4)
    # revalidation must not reset the retry budget or the description
    assert refreshed.retry_count == 1
    assert refreshed.description == "gamma is wrong"
    output = capsys.readouterr().out
    assert "1 revalidated, 0 resolved" in output
    assert "fix #2 comments src/a.py: revalidated (lines 4)" in output


def test_revalidation_gone_closes_a_sibling_with_a_reason(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\nbeta = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="alpha is wrong", line_start=1, line_end=1)
    second = seed_issue(project, description="beta is wrong", line_start=2, line_end=2)
    fake = FakeModel(
        [
            edit("alpha = 1", "alpha = 2"),
            fix_report("fixed alpha"),
            review(True, "good"),
            revalidate(
                {
                    "issue_id": second,
                    "status": "gone",
                    "reason": "the same edit corrected beta",
                }
            ),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 0

    assert issues(project) == []
    output = capsys.readouterr().out
    assert "0 revalidated, 1 resolved" in output
    assert "closed by revalidation (the same edit corrected beta)" in output
    # edit / fix_report / review for #1, then one revalidation call
    assert fake.chat_calls == 4


def test_revalidation_uncertain_leaves_the_issue_unchanged(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\nbeta = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="alpha is wrong", line_start=1, line_end=1)
    second = seed_issue(
        project, description="beta is wrong", line_start=2, line_end=2, retry_count=1
    )
    fake = FakeModel(
        [
            edit("alpha = 1", "alpha = 2"),
            fix_report("fixed alpha"),
            review(True, "good"),
            revalidate({"issue_id": second, "status": "uncertain"}),
        ]
    )

    assert run_cli(fake, *fix_args(project, "--limit", "1")) == 1

    stored = issues(project)
    assert [entry.id for entry in stored] == [second]
    entry = stored[0]
    assert (entry.line_start, entry.line_end) == (2, 2)
    assert entry.description == "beta is wrong"
    assert entry.retry_count == 1


def test_revalidation_ignores_unknown_issue_ids(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\nbeta = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="alpha is wrong", line_start=1, line_end=1)
    second = seed_issue(project, description="beta is wrong", line_start=2, line_end=2)
    fake = FakeModel(
        [
            edit("alpha = 1", "alpha = 2"),
            fix_report("fixed alpha"),
            review(True, "good"),
            # a verdict for a non-existent id is ignored; #2 gets no verdict
            revalidate({"issue_id": 999, "status": "gone", "reason": "x"}),
        ]
    )

    assert run_cli(fake, *fix_args(project, "--limit", "1")) == 1

    stored = issues(project)
    assert [entry.id for entry in stored] == [second]
    assert (stored[0].line_start, stored[0].line_end) == (2, 2)


def test_revalidation_missing_verdict_leaves_issue_unchanged(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\nbeta = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="alpha is wrong", line_start=1, line_end=1)
    second = seed_issue(project, description="beta is wrong", line_start=2, line_end=2)
    fake = FakeModel(
        [
            edit("alpha = 1", "alpha = 2"),
            fix_report("fixed alpha"),
            review(True, "good"),
            revalidate(),  # an empty report: no verdicts at all
        ]
    )

    assert run_cli(fake, *fix_args(project, "--limit", "1")) == 1

    stored = issues(project)
    assert [entry.id for entry in stored] == [second]


def test_revalidation_clamps_out_of_range_lines(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\nbeta = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="alpha is wrong", line_start=1, line_end=1)
    second = seed_issue(project, description="beta is wrong", line_start=2, line_end=2)
    fake = FakeModel(
        [
            edit("alpha = 1", "alpha = 2"),
            fix_report("fixed alpha"),
            review(True, "good"),
            # a `valid` verdict with lines past the end of the file is clamped
            revalidate(
                {"issue_id": second, "status": "valid", "line_start": 999, "line_end": 1000}
            ),
        ]
    )

    assert run_cli(fake, *fix_args(project, "--limit", "1")) == 1

    stored = issues(project)
    assert [entry.id for entry in stored] == [second]
    assert (stored[0].line_start, stored[0].line_end) == (2, 2)


def test_revalidation_gone_drains_a_sibling_aspect(tmp_path: Path, capsys) -> None:
    """Cross-aspect: a `gone` verdict drains a sibling aspect's pair to clean."""
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\n"},
        aspects=[
            {"id": "comments", "prompt": "Find bad comments.", "files": ["src/"]},
            {"id": "naming", "prompt": "Find bad names.", "files": ["src/"]},
        ],
        verify=TRUE_VERIFY,
    )
    seed_state(project, aspect_id="comments", outcome="issues")
    seed_state(project, aspect_id="naming", outcome="issues")
    seed_issue(project, aspect_id="comments", description="bad comment")
    naming_issue = seed_issue(project, aspect_id="naming", description="bad name")

    fake = FakeModel(
        [
            edit("alpha = 1", "alpha = 2"),
            fix_report("fixed comment"),
            review(True, "good"),
            # the naming issue is closed by revalidation, draining its pair
            revalidate(
                {
                    "issue_id": naming_issue,
                    "status": "gone",
                    "reason": "the name was fixed by the same edit",
                }
            ),
        ]
    )

    # both issues are gone (#1 fixed, #2 closed by revalidation), so the run
    # ends clean even though only one attempt was spent
    assert run_cli(fake, *fix_args(project, "--limit", "1")) == 0

    assert issues(project) == []
    naming_state = state(project, aspect="naming")
    assert naming_state is not None
    assert naming_state.outcome == "clean"
    output = capsys.readouterr().out
    assert "1 resolved" in output


def test_revalidation_does_not_consume_the_attempt_budget(tmp_path: Path) -> None:
    """A revalidation model call is free of the `--limit` attempt budget."""
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\nbeta = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="alpha is wrong", line_start=1, line_end=1)
    second = seed_issue(project, description="beta is wrong", line_start=2, line_end=2)
    fake = FakeModel(
        [
            edit("alpha = 1", "alpha = 2"),
            fix_report("fixed alpha"),
            review(True, "good"),
            # the revalidation of the sibling is an extra model call, not an attempt
            revalidate({"issue_id": second, "status": "uncertain"}),
        ]
    )

    # --limit 1 spends the single attempt on #1; #2 is re-validated and left open
    assert run_cli(fake, *fix_args(project, "--limit", "1")) == 1

    stored = issues(project)
    assert [entry.id for entry in stored] == [second]
    # edit / fix_report / review for #1, then one revalidation call
    assert fake.chat_calls == 4


def test_revalidation_is_traced_as_a_revalidate_run(tmp_path: Path) -> None:
    """Revalidation runs under the read-only tool set and a `:revalidate` label."""
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\nbeta = 1\n"},
        verify=TRUE_VERIFY,
        observe={"enabled": True},
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="alpha is wrong", line_start=1, line_end=1)
    second = seed_issue(project, description="beta is wrong", line_start=2, line_end=2)
    fake = FakeModel(
        [
            edit("alpha = 1", "alpha = 2"),
            fix_report("fixed alpha"),
            review(True, "good"),
            revalidate({"issue_id": second, "status": "uncertain"}),
        ]
    )

    assert run_cli(fake, *fix_args(project, "--limit", "1")) == 1

    store = TraceStore(project.trace_root)
    events = store.events(store.list_runs()[0].id)
    revalidate_calls = [
        event
        for event in events
        if event["kind"] == "model_call"
        and event.get("agent") == "aspect:comments:src/a.py:revalidate"
    ]
    assert len(revalidate_calls) == 1
    tool_names = {tool["function"]["name"] for tool in revalidate_calls[0]["request"]["tools"]}
    assert tool_names == {"read_file", "grep", "list_files", "report_revalidation"}


def test_issues_list_shows_refreshed_lines_after_revalidation(tmp_path: Path, capsys) -> None:
    """An e2e-style check: a later `khorek issues list` reflects the refresh."""
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\nbeta = 1\ngamma = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="alpha is wrong", line_start=1, line_end=1)
    second = seed_issue(project, description="gamma is wrong", line_start=3, line_end=3)
    fake = FakeModel(
        [
            # the fix inserts a line, shifting gamma from line 3 down to line 4
            edit("alpha = 1", "alpha = 1\nalpha2 = 1"),
            fix_report("inserted a line"),
            review(True, "good"),
            revalidate({"issue_id": second, "status": "valid", "line_start": 4, "line_end": 4}),
        ]
    )
    assert run_cli(fake, *fix_args(project, "--limit", "1")) == 1

    # a fresh, lockless `khorek issues list` reflects the refreshed line span
    assert (
        run_cli(FakeModel([]), "issues", "list", "-p", project.root, "--db", project.db_path) == 0
    )
    output = capsys.readouterr().out
    assert "src/a.py:4" in output
    assert str(second) in output
