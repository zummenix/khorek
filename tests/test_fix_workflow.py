"""Fix workflow: success, verification loop, rollback, staleness, queueing."""

from __future__ import annotations

from pathlib import Path

from khorek.db import Database
from khorek.files import sha256_file
from khorek.workflows.fix import build_queue
from tests.conftest import (
    FakeModel,
    Project,
    fix_args,
    make_project,
    python_command,
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
    fake = FakeModel([edit("# wrong", "# right"), fix_report("corrected the comment")])

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
    assert "1 attempted, 1 fixed, 0 failed, 0 skipped-stale, 0 open issues" in output


def test_successful_fix_prints_problem_and_fix_block(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "# wrong\nvalue = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="comment contradicts the code", line_start=1, line_end=1)
    fake = FakeModel([edit("# wrong", "# right"), fix_report("corrected the comment")])

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
    fake = FakeModel([edit("value = 1", "value = 2"), fix_report("")])

    assert run_cli(fake, *fix_args(project)) == 0

    output = capsys.readouterr().out
    assert (
        "fix #1 comments src/a.py: fixed\n  problem:\n    the comment is inconsistent\n"
    ) in output
    assert "  fix:" not in output


def test_empty_description_prints_fix_block_only(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="")
    fake = FakeModel([edit("value = 1", "value = 2"), fix_report("applied the fix")])

    assert run_cli(fake, *fix_args(project)) == 0

    output = capsys.readouterr().out
    assert ("fix #1 comments src/a.py: fixed\n  fix:\n    applied the fix\n") in output
    assert "  problem:" not in output


def test_both_fields_empty_print_no_report_block(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    seed_issue(project, description="")
    fake = FakeModel([edit("value = 1", "value = 2"), fix_report("")])

    assert run_cli(fake, *fix_args(project)) == 0

    output = capsys.readouterr().out
    assert "fix #1 comments src/a.py: fixed\n" in output
    assert "  problem:" not in output
    assert "  fix:" not in output


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
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 0

    assert project.read("src/a.py") == "value = 3\n"
    assert fake.chat_calls == 5
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
    assert "1 attempted, 0 fixed, 1 failed" in output


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


def test_stale_hash_issue_is_deleted_and_counted(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project", files={"src/a.py": "value = 1\n"}, verify=TRUE_VERIFY
    )
    seed_issue(project, sha256="stale-hash")
    fake = FakeModel([])

    assert run_cli(fake, *fix_args(project)) == 0

    assert fake.chat_calls == 0
    assert issues(project) == []
    output = capsys.readouterr().out
    assert "skipped (stale hash)" in output
    assert "1 attempted, 0 fixed, 0 failed, 1 skipped-stale, 0 open issues" in output


def test_stale_skip_prints_no_report_block(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_issue(project, sha256="stale-hash")
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
            edit("beta = 1", "beta = 2"),
            fix_report("fixed beta"),
        ]
    )

    assert run_cli(fake, *fix_args(project)) == 0

    assert first != second
    assert project.read("src/a.py") == "alpha = 2\nbeta = 2\n"
    assert fake.chat_calls == 4
    assert issues(project) == []
    row = state(project)
    assert row is not None
    assert row.outcome == "clean"
    assert row.sha256 == sha256_file(project.root / "src/a.py")


def test_fix_advances_sibling_hash_but_leaves_state_while_issues_remain(
    tmp_path: Path, capsys
) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "alpha = 1\nbeta = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="issues")
    old_sha = sha256_file(project.root / "src/a.py")
    first = seed_issue(project, description="alpha is wrong", line_start=1, line_end=1)
    second = seed_issue(project, description="beta is wrong", line_start=2, line_end=2)
    fake = FakeModel([edit("alpha = 1", "alpha = 2"), fix_report("fixed alpha")])

    assert run_cli(fake, *fix_args(project, "--limit", "1")) == 1

    stored = issues(project)
    assert [entry.id for entry in stored] == [second]
    new_sha = sha256_file(project.root / "src/a.py")
    assert stored[0].sha256 == new_sha
    assert stored[0].sha256 != old_sha
    row = state(project)
    assert row is not None
    assert row.outcome == "issues"
    assert row.sha256 == old_sha
    assert first != second
    output = capsys.readouterr().out
    assert "1 attempted, 1 fixed, 0 failed" in output
    assert "1 open issues" in output


def test_truncated_state_row_is_deleted_after_successful_fix(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "value = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_state(project, outcome="truncated")
    seed_issue(project)
    fake = FakeModel([edit("value = 1", "value = 2"), fix_report()])

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
    assert "1 attempted, 0 fixed, 1 failed" in output


def test_limit_counts_stale_skips(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "a = 1\n", "src/b.py": "b = 1\n"},
        verify=TRUE_VERIFY,
    )
    seed_issue(project, path="src/a.py", sha256="stale-hash")
    seed_issue(project, path="src/b.py", description="b is wrong")
    fake = FakeModel([])

    assert run_cli(fake, *fix_args(project, "--limit", "1")) == 1

    assert fake.chat_calls == 0
    stored = issues(project)
    assert [entry.path for entry in stored] == ["src/b.py"]
    output = capsys.readouterr().out
    assert "1 attempted, 0 fixed, 0 failed, 1 skipped-stale, 1 open issues" in output


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
    assert "0 fixed, 1 failed" in capsys.readouterr().out


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
    assert "0 fixed, 1 failed" in output


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
