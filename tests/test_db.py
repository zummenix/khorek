"""Database schema, slugging, meta guard, queries, and drift pruning."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from khorek.db import (
    SCHEMA_VERSION,
    Database,
    DatabaseError,
    default_db_path,
)


def test_default_db_path_slug_and_patched_home(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    project = tmp_path / "project"
    project.mkdir()

    first = default_db_path(project)
    second = default_db_path(project)
    assert first == second
    assert first.parent == home / ".khorek"
    assert re.search(r"-[0-9a-f]{8}\.db$", first.name)
    assert "project" in first.name


def test_default_db_path_explicit_home(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    path = default_db_path(project, home=home)
    assert path.parent == home / ".khorek"


def test_default_db_path_distinguishes_roots(tmp_path: Path) -> None:
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    assert default_db_path(one).name != default_db_path(two).name


def test_fresh_database_schema_and_meta(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    db = Database(tmp_path / "khorek.db", project)
    try:
        with sqlite3.connect(tmp_path / "khorek.db") as raw:
            meta = dict(raw.execute("SELECT key, value FROM meta").fetchall())
            tables = {
                row[0]
                for row in raw.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
        assert meta == {
            "project_root": str(project.resolve()),
            "schema_version": SCHEMA_VERSION,
        }
        assert {"meta", "file_aspect_state", "issues"} <= tables
    finally:
        db.close()


def test_project_root_mismatch_is_fatal(tmp_path: Path) -> None:
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    path = tmp_path / "khorek.db"
    with Database(path, one):
        pass
    before = path.read_bytes()
    with pytest.raises(DatabaseError, match="belongs to project"):
        Database(path, two)
    assert path.read_bytes() == before


def test_schema_version_mismatch_is_fatal(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    path = tmp_path / "khorek.db"
    with Database(path, project):
        pass
    with sqlite3.connect(path) as raw:
        raw.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
        raw.commit()
    with pytest.raises(DatabaseError, match="schema version"):
        Database(path, project)


def test_state_crud(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with Database(tmp_path / "khorek.db", project) as db:
        assert db.get_state("comments", "src/a.py") is None
        db.upsert_state("comments", "src/a.py", "abc", "clean")
        row = db.get_state("comments", "src/a.py")
        assert row is not None
        assert (row.sha256, row.outcome) == ("abc", "clean")
        assert [state.path for state in db.list_state()] == ["src/a.py"]

        db.upsert_state("comments", "src/a.py", "def", "issues")
        row = db.get_state("comments", "src/a.py")
        assert row is not None
        assert (row.sha256, row.outcome) == ("def", "issues")
        assert db.delete_state("comments", "src/a.py") is True
        assert db.get_state("comments", "src/a.py") is None
        assert db.delete_state("comments", "src/a.py") is False


def test_invalid_state_outcome_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with Database(tmp_path / "khorek.db", project) as db:
        with pytest.raises(ValueError, match="invalid state outcome"):
            db.upsert_state("comments", "src/a.py", "abc", "nope")


def test_issue_crud_counts_and_lookups(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with Database(tmp_path / "khorek.db", project) as db:
        first = db.insert_issue(
            aspect_id="comments",
            path="src/a.py",
            sha256="sha1",
            description="bad comment",
            line_start=1,
            line_end=2,
            severity="high",
        )
        second = db.insert_issue(
            aspect_id="dead-code",
            path="src/a.py",
            sha256="sha1",
            description="unused import",
            line_start=3,
            line_end=3,
            severity="low",
        )
        db.insert_issue(
            aspect_id="comments",
            path="src/b.py",
            sha256="sha2",
            description="other",
            line_start=1,
            line_end=1,
            severity="medium",
        )

        assert [issue.id for issue in db.list_issues()] == [first, second, 3]
        assert [issue.id for issue in db.open_issues_for("comments", "src/a.py")] == [first]
        assert [issue.id for issue in db.open_issues_for_path("src/a.py")] == [
            first,
            second,
        ]
        assert db.count_issues() == 3
        assert db.count_issues(aspect_id="comments") == 2
        assert db.count_issues(path="src/a.py") == 2

        assert db.update_issue(first, severity="medium", retry_count=2, last_error="boom")
        updated = db.get_issue(first)
        assert updated is not None
        assert (updated.severity, updated.retry_count, updated.last_error) == (
            "medium",
            2,
            "boom",
        )
        assert db.delete_issue(second) is True
        assert db.delete_issue(second) is False
        assert db.delete_issues_for("comments", "src/b.py") == 1
        assert db.count_issues() == 1


def test_update_issue_rejects_unknown_fields(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with Database(tmp_path / "khorek.db", project) as db:
        issue_id = db.insert_issue(
            aspect_id="comments",
            path="src/a.py",
            sha256="sha",
            description="x",
            line_start=1,
            line_end=1,
            severity="low",
        )
        with pytest.raises(ValueError, match="unknown issue fields"):
            db.update_issue(issue_id, drop_table="issues")


def test_timestamps_are_utc_iso8601(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with Database(tmp_path / "khorek.db", project) as db:
        db.upsert_state("comments", "src/a.py", "abc", "clean")
        row = db.get_state("comments", "src/a.py")
        assert row is not None
        parsed = datetime.fromisoformat(row.updated_at)
        assert parsed.tzinfo is not None
        offset = parsed.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 0


def test_prune_drift_removes_only_absent_aspects(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with Database(tmp_path / "khorek.db", project) as db:
        db.upsert_state("kept", "src/a.py", "sha1", "issues")
        db.upsert_state("gone", "src/a.py", "sha1", "clean")
        db.insert_issue(
            aspect_id="kept",
            path="src/a.py",
            sha256="sha1",
            description="kept",
            line_start=1,
            line_end=1,
            severity="low",
        )
        db.insert_issue(
            aspect_id="gone",
            path="src/a.py",
            sha256="sha1",
            description="gone",
            line_start=1,
            line_end=1,
            severity="low",
        )
        db.insert_issue(
            aspect_id="gone",
            path="src/b.py",
            sha256="sha2",
            description="gone too",
            line_start=1,
            line_end=1,
            severity="low",
        )

        assert db.prune_drift(["kept"]) == 3
        assert [state.aspect_id for state in db.list_state()] == ["kept"]
        assert [issue.aspect_id for issue in db.list_issues()] == ["kept"]


def test_prune_drift_with_empty_config_removes_everything(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with Database(tmp_path / "khorek.db", project) as db:
        db.upsert_state("gone", "src/a.py", "sha1", "clean")
        db.insert_issue(
            aspect_id="gone",
            path="src/a.py",
            sha256="sha1",
            description="gone",
            line_start=1,
            line_end=1,
            severity="low",
        )
        assert db.prune_drift([]) == 2
        assert db.list_state() == []
        assert db.list_issues() == []


def test_foreign_database_is_not_mutated_before_fatal(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    path = tmp_path / "foreign.db"
    with sqlite3.connect(path) as raw:
        raw.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        raw.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("project_root", str(other.resolve())),
                ("schema_version", SCHEMA_VERSION),
            ],
        )
        raw.commit()

    before = path.read_bytes()
    with pytest.raises(DatabaseError, match="belongs to project"):
        Database(path, project)

    assert path.read_bytes() == before
    with sqlite3.connect(path) as raw:
        tables = {
            str(row[0])
            for row in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
    assert "issues" not in tables
    assert "file_aspect_state" not in tables


def test_incomplete_meta_is_fatal_without_writes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    path = tmp_path / "incomplete.db"
    with sqlite3.connect(path) as raw:
        raw.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        raw.execute(
            "INSERT INTO meta (key, value) VALUES ('project_root', ?)",
            (str(project.resolve()),),
        )
        raw.commit()

    before = path.read_bytes()
    with pytest.raises(DatabaseError, match="incomplete meta"):
        Database(path, project)
    assert path.read_bytes() == before
    with sqlite3.connect(path) as raw:
        tables = {
            str(row[0])
            for row in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
    assert tables == {"meta"}
