"""CLI wiring: preamble, exit codes, --db override, locks, and reset."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from khorek.cli import main
from khorek.db import Database, default_db_path
from khorek.lock import lock_path_for
from tests.conftest import (
    FakeModel,
    explore_and_fix_args,
    explore_args,
    fix_args,
    held_lock,
    make_project,
    run_cli,
    tool_message,
    write_config,
)


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
