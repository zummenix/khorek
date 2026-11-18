"""CLI wiring: preamble, exit codes, --db override, locks, reset, and issues subcommand."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from khorek.cli import main
from khorek.db import Database, default_db_path
from khorek.lock import lock_path_for
from tests.conftest import (
    FakeModel,
    Project,
    explore_and_fix_args,
    explore_args,
    fix_args,
    held_lock,
    make_project,
    run_cli,
    seed_issue,
    seed_state,
    set_created_at,
    tool_message,
    write_config,
)


def issues_args(project: Project, *extra: str | Path) -> tuple[str | Path, ...]:
    """Standard ``issues`` arguments for a tmp project."""
    return ("issues", "-p", project.root, "--db", project.db_path, *extra)


def report() -> dict[str, object]:
    return tool_message("report_issues", {"issues": []})


def test_no_arguments_prints_help_and_succeeds(capsys) -> None:
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "usage: khorek" in captured.out
    assert "explore" in captured.out


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_explore_and_fix_requires_a_verify_command(tmp_path: Path, capsys) -> None:
    project = make_project(tmp_path / "project")
    fake = FakeModel([])

    assert run_cli(fake, *explore_and_fix_args(project)) == 2

    assert fake.model_requests == 0
    assert not project.db_path.exists()
    captured = capsys.readouterr()
    assert "explore-and-fix requires" in captured.err
    assert "[[verify]]" in captured.err


def test_explore_and_fix_runs_the_combined_workflow(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        verify=[{"cmd": "true", "timeout": 10}],
    )
    fake = FakeModel([report()])

    assert run_cli(fake, *explore_and_fix_args(project)) == 0

    assert fake.model_requests == 1
    assert fake.chat_calls == 1
    output = capsys.readouterr().out
    assert "explore-and-fix summary:" in output
    assert "explore summary:" not in output
    assert "\nfix summary:" not in "\n" + output


def test_explore_and_fix_held_lock_exits_2(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "# correct\nvalue = 1\n"},
        verify=[{"cmd": "true", "timeout": 10}],
    )
    fake = FakeModel([report()])
    assert run_cli(fake, *explore_and_fix_args(project)) == 0
    before = project.db_path.read_bytes()

    with held_lock(lock_path_for(project.db_path)):
        assert run_cli(fake, *explore_and_fix_args(project)) == 2

    assert project.db_path.read_bytes() == before
    assert "locked by process 999999" in capsys.readouterr().err
    assert fake.chat_calls == 1


def test_fix_requires_a_verify_command(tmp_path: Path, capsys) -> None:
    project = make_project(tmp_path / "project")
    fake = FakeModel([])

    assert run_cli(fake, *fix_args(project)) == 2

    assert fake.model_requests == 0
    assert not project.db_path.exists()
    assert "[[verify]]" in capsys.readouterr().err


def test_negative_limit_is_fatal_before_any_work() -> None:
    fake = FakeModel([])
    assert run_cli(fake, "explore", "--limit", "-1") == 2
    assert fake.model_requests == 0


def test_missing_project_path_is_fatal(tmp_path: Path) -> None:
    fake = FakeModel([])
    assert (
        run_cli(
            fake,
            "explore",
            "-p",
            tmp_path / "missing",
            "--db",
            tmp_path / "x.db",
        )
        == 2
    )
    assert fake.model_requests == 0


def test_missing_config_is_fatal(tmp_path: Path, capsys) -> None:
    project = make_project(tmp_path / "project")
    (project.root / "khorek.toml").unlink()
    fake = FakeModel([])
    assert run_cli(fake, *explore_args(project)) == 2
    assert fake.model_requests == 0
    assert "config file not found" in capsys.readouterr().err


def test_invalid_config_is_fatal(tmp_path: Path, capsys) -> None:
    project = make_project(tmp_path / "project")
    (project.root / "khorek.toml").write_text("[llm]\n", encoding="utf-8")
    fake = FakeModel([])
    assert run_cli(fake, *explore_args(project)) == 2
    assert fake.model_requests == 0
    assert "invalid config" in capsys.readouterr().err


def test_model_unavailable_is_fatal_and_leaves_no_db(tmp_path: Path, capsys) -> None:
    project = make_project(tmp_path / "project")
    db_path = tmp_path / "khorek.db"

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    code = main(
        ["explore", "-p", str(project.root), "--db", str(db_path)],
        transport=httpx.MockTransport(handler),
    )
    assert code == 2
    assert not db_path.exists()
    assert "unreachable" in capsys.readouterr().err


def test_explicit_db_writes_to_tmp_path_not_home(
    project, tmp_path: Path, monkeypatch, capsys
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    fake = FakeModel([report()])

    nested = tmp_path / "nested" / "khorek.db"
    assert run_cli(fake, "explore", "-p", project.root, "--db", nested) == 0
    assert nested.is_file()
    assert not (home / ".khorek").exists()
    assert "explore summary" in capsys.readouterr().out


def test_default_db_path_uses_patched_home(project, tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    fake = FakeModel([report()])

    assert run_cli(fake, "explore", "-p", project.root) == 0
    expected = default_db_path(project.root, home=home)
    assert expected.is_file()
    with Database(expected, project.root) as db:
        assert db.count_issues() == 0


def test_reusing_db_for_another_project_exits_2(tmp_path: Path, capsys) -> None:
    one = make_project(tmp_path / "one")
    two = make_project(tmp_path / "two")
    shared = tmp_path / "shared.db"
    fake = FakeModel([report()])

    assert run_cli(fake, "explore", "-p", one.root, "--db", shared) == 0
    assert fake.chat_calls == 1

    assert run_cli(fake, "explore", "-p", two.root, "--db", shared) == 2
    assert fake.chat_calls == 1
    assert "belongs to project" in capsys.readouterr().err


def test_held_lock_exits_2_without_touching_db(project, capsys) -> None:
    fake = FakeModel([report()])
    assert run_cli(fake, *explore_args(project)) == 0
    before = project.db_path.read_bytes()

    with held_lock(lock_path_for(project.db_path)):
        assert run_cli(fake, *explore_args(project)) == 2

    assert project.db_path.read_bytes() == before
    captured = capsys.readouterr()
    assert "locked by process 999999" in captured.err
    assert fake.chat_calls == 1


def test_reset_deletes_db_wal_shm_and_lock(project, capsys) -> None:
    fake = FakeModel([report()])
    assert run_cli(fake, *explore_args(project)) == 0

    wal = project.db_path.with_name(project.db_path.name + "-wal")
    shm = project.db_path.with_name(project.db_path.name + "-shm")
    lock = lock_path_for(project.db_path)
    wal.write_bytes(b"wal")
    shm.write_bytes(b"shm")
    assert lock.exists()

    assert run_cli(fake, "reset", "-p", project.root, "--db", project.db_path) == 0
    assert not project.db_path.exists()
    assert not wal.exists()
    assert not shm.exists()
    assert not lock.exists()
    assert str(project.db_path.resolve()) in capsys.readouterr().out


def test_reset_of_missing_db_succeeds(tmp_path: Path, capsys) -> None:
    project = make_project(tmp_path / "project", files={})
    db_path = tmp_path / "never.db"
    fake = FakeModel([])
    assert run_cli(fake, "reset", "-p", project.root, "--db", db_path) == 0
    assert not db_path.exists()
    assert str(db_path.resolve()) in capsys.readouterr().out


def test_reset_refuses_when_locked(project, capsys) -> None:
    fake = FakeModel([report()])
    assert run_cli(fake, *explore_args(project)) == 0
    assert project.db_path.exists()

    with held_lock(lock_path_for(project.db_path)):
        assert run_cli(fake, "reset", "-p", project.root, "--db", project.db_path) == 2

    assert project.db_path.exists()
    assert "locked by process 999999" in capsys.readouterr().err


def test_config_override_selects_alternate_file_set(project, tmp_path: Path) -> None:
    project.write("other/x.py", "x = 1\n")
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    alt = write_config(
        config_dir,
        aspects=[
            {
                "id": "comments",
                "prompt": "Find inconsistent comments.",
                "files": ["other/"],
            }
        ],
    )
    fake = FakeModel([report()])
    assert (
        run_cli(
            fake,
            "explore",
            "-p",
            project.root,
            "--db",
            project.db_path,
            "--config",
            alt,
        )
        == 0
    )
    with Database(project.db_path, project.root) as db:
        assert db.get_state("comments", "other/x.py") is not None
        assert db.get_state("comments", "src/a.py") is None


# -- issues ----------------------------------------------------------------


def test_issues_help_lists_subcommands_without_llm_flags(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["issues", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for name in ("list", "show", "dismiss", "delete"):
        assert name in out
    assert "--db" in out
    assert "--limit" not in out
    assert "--trace" not in out
    assert "--config" not in out


def test_top_level_help_lists_issues(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    assert "issues" in capsys.readouterr().out


def test_issues_list_renders_seeded_db(project, capsys) -> None:
    project.write("src/b.py", "b = 1\n")
    first = seed_issue(
        project,
        path="src/a.py",
        line_start=234,
        line_end=238,
        severity="high",
        description='Comment says "O(n)"',
    )
    second = seed_issue(
        project,
        path="src/b.py",
        line_start=88,
        line_end=88,
        severity="low",
        retry_count=2,
        description="docstring mentions removed flag --verbose2",
    )
    set_created_at(project, first, "2026-11-12T10:15:30+00:00")
    set_created_at(project, second, "2026-11-13T09:00:00+00:00")

    fake = FakeModel([])
    assert run_cli(fake, *issues_args(project, "list")) == 0
    out = capsys.readouterr().out
    assert f"{first} | 2026-11-12  comments  high  src/a.py:234-238" in out
    assert f"{second} | 2026-11-13  comments  low   src/b.py:88" in out
    assert "    Retried: 2" in out
    assert '    Comment says "O(n)"' in out
    assert "    docstring mentions removed flag --verbose2" in out
    assert "2 open issues" in out
    assert "[stale]" not in out
    # No model activity.
    assert fake.model_requests == 0
    assert fake.chat_calls == 0


def test_issues_list_filters(project, capsys) -> None:
    project.write("src/b.py", "b = 1\n")
    project.write("lib/c.py", "c = 1\n")
    seed_issue(project, path="src/a.py", severity="high", description="A-high")
    seed_issue(
        project, path="src/a.py", line_start=2, line_end=2, severity="low", description="A-low"
    )
    seed_issue(
        project, path="src/b.py", aspect_id="dead-code", severity="high", description="B-high"
    )
    seed_issue(
        project, path="lib/c.py", aspect_id="dead-code", severity="medium", description="C-medium"
    )
    fake = FakeModel([])

    def listed(*extra: str) -> str:
        assert run_cli(fake, *issues_args(project, "list", *extra)) == 0
        return capsys.readouterr().out

    out = listed("--severity", "high")
    assert "A-high" in out and "B-high" in out
    assert "A-low" not in out and "C-medium" not in out

    out = listed("--severity", "low", "--severity", "high")
    assert "A-high" in out and "A-low" in out and "B-high" in out
    assert "C-medium" not in out

    out = listed("--aspect", "dead-code")
    assert "B-high" in out and "C-medium" in out
    assert "A-high" not in out and "A-low" not in out

    out = listed("--path", "src/")
    assert "A-high" in out and "A-low" in out and "B-high" in out
    assert "C-medium" not in out

    out = listed("--path", "src/a.py")
    assert "A-high" in out and "A-low" in out
    assert "B-high" not in out and "C-medium" not in out

    out = listed("--aspect", "comments", "--severity", "high")
    assert "A-high" in out
    assert "A-low" not in out and "B-high" not in out and "C-medium" not in out

    out = listed("--aspect", "comments", "--aspect", "dead-code")
    assert "A-high" in out and "B-high" in out and "C-medium" in out
    assert "4 open issues" in out


def test_issues_list_prunes_stale(project, capsys) -> None:
    project.write("src/b.py", "b = 1\n")
    fresh = seed_issue(project, path="src/a.py", description="fresh one")
    stale = seed_issue(project, path="src/b.py", description="stale one")
    seed_state(project, path="src/a.py", outcome="issues")
    seed_state(project, path="src/b.py", outcome="issues")
    project.write("src/b.py", "b = 2\n")  # change after recording -> stale

    fake = FakeModel([])
    assert run_cli(fake, *issues_args(project, "list")) == 0
    out = capsys.readouterr().out
    assert "pruned 1 stale issue(s) (file changed or missing)" in out
    assert "fresh one" in out
    assert "stale one" not in out
    with Database(project.db_path, project.root, readonly=True) as db:
        assert [issue.id for issue in db.list_issues()] == [fresh]
        assert stale not in [issue.id for issue in db.list_issues()]


def test_issues_list_stale_flag_no_prune(project, capsys) -> None:
    project.write("src/b.py", "b = 1\n")
    fresh = seed_issue(project, path="src/a.py", description="fresh one")
    stale = seed_issue(project, path="src/b.py", description="stale one")
    seed_state(project, path="src/a.py", outcome="issues")
    seed_state(project, path="src/b.py", outcome="issues")
    project.write("src/b.py", "b = 2\n")
    before = project.db_path.read_bytes()

    fake = FakeModel([])
    assert run_cli(fake, *issues_args(project, "--stale")) == 0
    out = capsys.readouterr().out
    assert "1 stale issue(s) shown, not pruned" in out
    assert "run in progress" not in out
    assert "[stale]" in out
    assert "fresh one" in out and "stale one" in out
    assert out.rstrip().endswith("2 open issues (1 stale)")
    assert project.db_path.read_bytes() == before  # nothing pruned
    assert {fresh, stale} == {issue.id for issue in _all_ids(project)}


def test_issues_list_held_lock_labels_stale(project, capsys) -> None:
    project.write("src/b.py", "b = 1\n")
    seed_issue(project, path="src/a.py", description="fresh one")
    seed_issue(project, path="src/b.py", description="stale one")
    seed_state(project, path="src/a.py", outcome="issues")
    seed_state(project, path="src/b.py", outcome="issues")
    project.write("src/b.py", "b = 2\n")
    before = project.db_path.read_bytes()

    with held_lock(lock_path_for(project.db_path)):
        fake = FakeModel([])
        assert run_cli(fake, *issues_args(project, "list")) == 0
        out = capsys.readouterr().out
        assert "run in progress (pid 999999): 1 stale issue(s) shown, not pruned" in out
        assert "[stale]" in out
        assert "fresh one" in out and "stale one" in out

    assert project.db_path.read_bytes() == before  # still no prune


def test_issues_show_fresh_with_error(project, capsys) -> None:
    issue_id = seed_issue(
        project,
        path="src/a.py",
        line_start=234,
        line_end=238,
        severity="high",
        description="the description",
        retry_count=2,
    )
    with Database(project.db_path, project.root) as db:
        db.update_issue(issue_id, last_error="verification failed: uv run pytest (exited 1)")

    fake = FakeModel([])
    assert run_cli(fake, *issues_args(project, "show", str(issue_id))) == 0
    out = capsys.readouterr().out
    assert f"#{issue_id}  comments  src/a.py:234-238" in out
    assert "severity:  high" in out
    assert "status:    fresh" in out
    assert "retries:   2" in out
    assert "last_error:" in out
    assert "    verification failed: uv run pytest (exited 1)" in out
    assert "the description" in out


def test_issues_show_stale_and_single_line(project, capsys) -> None:
    project.write("src/b.py", "b = 1\n")
    issue_id = seed_issue(
        project, path="src/b.py", line_start=4, line_end=4, severity="low", description="stale desc"
    )
    seed_state(project, path="src/b.py", outcome="issues")
    project.write("src/b.py", "b = 2\n")  # now stale

    fake = FakeModel([])
    assert run_cli(fake, *issues_args(project, "show", str(issue_id))) == 0
    out = capsys.readouterr().out
    assert f"#{issue_id}  comments  src/b.py:4" in out
    assert "status:    stale (file changed since recorded)" in out
    assert "retries:   0" in out
    assert "last_error:" not in out


def test_issues_show_unknown_id(project, capsys) -> None:
    seed_issue(project)
    fake = FakeModel([])
    assert run_cli(fake, *issues_args(project, "show", "999")) == 1
    assert "unknown issue id: 999" in capsys.readouterr().err


def test_issues_dismiss_marks_pair_clean(project, capsys) -> None:
    issue_id = seed_issue(project, path="src/a.py", line_start=1, line_end=1)
    seed_state(project, path="src/a.py", outcome="issues")
    fake = FakeModel([])
    assert run_cli(fake, *issues_args(project, "dismiss", str(issue_id))) == 0
    out = capsys.readouterr().out
    assert f"dismissed #{issue_id}: comments src/a.py:1 \u2192 pair marked clean" in out
    with Database(project.db_path, project.root, readonly=True) as db:
        assert db.get_issue(issue_id) is None
        state = db.get_state("comments", "src/a.py")
        assert state is not None and state.outcome == "clean"


def test_issues_dismiss_unknown_id_exit_1(project, capsys) -> None:
    known = seed_issue(project, path="src/a.py", description="known")
    seed_state(project, path="src/a.py", outcome="issues")
    fake = FakeModel([])
    assert run_cli(fake, *issues_args(project, "dismiss", "999", str(known))) == 1
    captured = capsys.readouterr()
    assert "unknown issue id: 999" in captured.err
    assert f"dismissed #{known}:" in captured.out  # the known id was still processed
    with Database(project.db_path, project.root, readonly=True) as db:
        assert db.get_issue(known) is None


def test_issues_delete_clears_state(project, capsys) -> None:
    issue_id = seed_issue(project, path="src/a.py")
    seed_state(project, path="src/a.py", outcome="issues")
    fake = FakeModel([])
    assert run_cli(fake, *issues_args(project, "delete", str(issue_id))) == 0
    out = capsys.readouterr().out
    assert f"deleted #{issue_id}: comments src/a.py:1 \u2192 state cleared" in out
    with Database(project.db_path, project.root, readonly=True) as db:
        assert db.get_issue(issue_id) is None
        assert db.get_state("comments", "src/a.py") is None


def test_issues_dismiss_held_lock_exit_2(project, capsys) -> None:
    issue_id = seed_issue(project, path="src/a.py")
    seed_state(project, path="src/a.py", outcome="issues")
    before = project.db_path.read_bytes()
    with held_lock(lock_path_for(project.db_path)):
        fake = FakeModel([])
        assert run_cli(fake, *issues_args(project, "dismiss", str(issue_id))) == 2
        assert "locked by process 999999" in capsys.readouterr().err
    assert project.db_path.read_bytes() == before  # untouched


def test_issues_delete_held_lock_exit_2(project, capsys) -> None:
    issue_id = seed_issue(project, path="src/a.py")
    seed_state(project, path="src/a.py", outcome="issues")
    before = project.db_path.read_bytes()
    with held_lock(lock_path_for(project.db_path)):
        fake = FakeModel([])
        assert run_cli(fake, *issues_args(project, "delete", str(issue_id))) == 2
        assert "locked by process 999999" in capsys.readouterr().err
    assert project.db_path.read_bytes() == before


def test_issues_commands_without_config(tmp_path: Path, capsys) -> None:
    root = tmp_path / "bare"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_text("a = 1\n")
    db = tmp_path / "bare.db"
    with Database(db, root) as database:
        issue_id = database.insert_issue(
            aspect_id="comments",
            path="src/a.py",
            description="no config here",
            line_start=1,
            line_end=1,
            severity="high",
        )
    assert not (root / "khorek.toml").exists()

    fake = FakeModel([])
    assert run_cli(fake, "issues", "list", "-p", root, "--db", db) == 0
    assert "no config here" in capsys.readouterr().out

    assert run_cli(fake, "issues", "dismiss", str(issue_id), "-p", root, "--db", db) == 0
    with Database(db, root, readonly=True) as database:
        assert database.list_issues() == []


def test_issues_list_missing_db_prints_none(project, capsys) -> None:
    assert not project.db_path.exists()
    fake = FakeModel([])
    assert run_cli(fake, *issues_args(project, "list")) == 0
    assert capsys.readouterr().out.strip() == "no open issues"
    assert not project.db_path.exists()  # not created as a side effect


def test_issues_show_dismiss_delete_missing_db(project, capsys) -> None:
    fake = FakeModel([])
    for sub in ("show 1", "dismiss 1", "delete 1"):
        assert run_cli(fake, *issues_args(project, *sub.split())) == 1
        assert "no database at" in capsys.readouterr().err
    assert not project.db_path.exists()


def _all_ids(project: Project) -> list:
    with Database(project.db_path, project.root, readonly=True) as db:
        return list(db.list_issues())
