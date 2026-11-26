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
    Issue,
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


def _create_v1_db(path: Path, project: Path) -> int:
    """Create a schema-v1 database (issues.sha256 present) with one issue.

    Returns the seeded issue id so a later migration can prove the
    AUTOINCREMENT sequence is preserved.
    """
    now = "2024-01-01T00:00:00+00:00"
    with sqlite3.connect(path) as raw:
        raw.executescript(
            """
            CREATE TABLE meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE file_aspect_state (
              aspect_id TEXT NOT NULL,
              path      TEXT NOT NULL,
              sha256    TEXT NOT NULL,
              outcome   TEXT NOT NULL CHECK (outcome IN ('clean', 'issues', 'truncated')),
              updated_at TEXT NOT NULL,
              PRIMARY KEY (aspect_id, path)
            );
            CREATE TABLE issues (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
              aspect_id   TEXT NOT NULL,
              path        TEXT NOT NULL,
              sha256      TEXT NOT NULL,
              description TEXT NOT NULL,
              line_start  INTEGER NOT NULL,
              line_end    INTEGER NOT NULL,
              severity    TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high')),
              retry_count INTEGER NOT NULL DEFAULT 0,
              last_error  TEXT,
              created_at  TEXT NOT NULL,
              updated_at  TEXT NOT NULL
            );
            CREATE INDEX idx_issues_aspect_path ON issues (aspect_id, path);
            CREATE INDEX idx_issues_path ON issues (path);
            CREATE INDEX idx_state_path ON file_aspect_state (path);
            """
        )
        raw.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [("project_root", str(project.resolve())), ("schema_version", "1")],
        )
        cursor = raw.execute(
            "INSERT INTO issues "
            "(aspect_id, path, sha256, description, line_start, line_end, severity, "
            "retry_count, last_error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)",
            ("comments", "src/a.py", "old-sha", "stale v1 issue", 1, 1, "high", now, now),
        )
        raw.commit()
        rowid = cursor.lastrowid
        assert rowid is not None
        return rowid


def test_v1_database_migrates_on_write_open(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    path = tmp_path / "khorek.db"
    seeded = _create_v1_db(path, project)
    assert seeded == 1

    with Database(path, project) as db:
        # The schema stamp advanced and the per-issue column is gone.
        with sqlite3.connect(path) as raw:
            meta = dict(raw.execute("SELECT key, value FROM meta").fetchall())
            columns = {row[1] for row in raw.execute("PRAGMA table_info(issues)")}
        assert meta["schema_version"] == SCHEMA_VERSION
        assert "sha256" not in columns
        assert db.get_issue(seeded) is not None
        fresh = db.insert_issue(
            aspect_id="comments",
            path="src/a.py",
            description="new issue",
            line_start=2,
            line_end=2,
            severity="low",
        )
        assert fresh == seeded + 1


def test_v1_database_readonly_open_is_allowed_and_unmigrated(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    path = tmp_path / "khorek.db"
    _create_v1_db(path, project)

    # A readonly open of a known older schema is allowed and leaves it unmigrated.
    with Database(path, project, readonly=True) as ro:
        assert ro.readonly is True
        assert ro.count_issues() == 1
        assert ro.list_issues()[0].description == "stale v1 issue"

    with sqlite3.connect(path) as raw:
        meta = dict(raw.execute("SELECT key, value FROM meta").fetchall())
        columns = {row[1] for row in raw.execute("PRAGMA table_info(issues)")}
    assert meta["schema_version"] == "1"
    assert "sha256" in columns


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
            description="bad comment",
            line_start=1,
            line_end=2,
            severity="high",
        )
        second = db.insert_issue(
            aspect_id="dead-code",
            path="src/a.py",
            description="unused import",
            line_start=3,
            line_end=3,
            severity="low",
        )
        db.insert_issue(
            aspect_id="comments",
            path="src/b.py",
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


def _seed_filter_fixtures(db: Database) -> dict[str, int]:
    """Insert a known set of issues covering aspects, severities, and paths."""
    specs = {
        "a": ("comments", "src/khorek/db.py", "high"),
        "b": ("comments", "src/khorek/cli.py", "low"),
        "c": ("dead-code", "src/khorek/agents/explorer.py", "medium"),
        "d": ("dead-code", "src/other.py", "high"),
        # Controls for LIKE-wildcard escaping in path prefixes.
        "pct": ("comments", "src/50%done.py", "low"),
        "pct_control": ("comments", "src/50xdone.py", "low"),
        "und": ("comments", "src/a_b.py", "medium"),
        "und_control": ("comments", "src/aXb.py", "medium"),
    }
    ids: dict[str, int] = {}
    for name, (aspect, path, severity) in specs.items():
        ids[name] = db.insert_issue(
            aspect_id=aspect,
            path=path,
            description=f"issue {name}",
            line_start=1,
            line_end=1,
            severity=severity,
        )
    return ids


def _ids(issues: list[Issue]) -> list[int]:
    return [issue.id for issue in issues]


def test_list_issues_severity_and_aspect_filters(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with Database(tmp_path / "khorek.db", project) as db:
        ids = _seed_filter_fixtures(db)

        # OR within a single flag.
        assert _ids(db.list_issues(severities=["high"])) == sorted(
            [ids["a"], ids["d"]], key=lambda i: i
        )
        assert set(_ids(db.list_issues(severities=["high", "low"]))) == {
            ids["a"],
            ids["b"],
            ids["d"],
            ids["pct"],
            ids["pct_control"],
        }
        assert set(_ids(db.list_issues(aspect_ids=["dead-code"]))) == {
            ids["c"],
            ids["d"],
        }
        assert set(_ids(db.list_issues(aspect_ids=["comments", "dead-code"]))) == set(ids.values())

        # AND across flags.
        assert _ids(db.list_issues(aspect_ids=["comments"], severities=["high"])) == [ids["a"]]
        assert _ids(db.list_issues(aspect_ids=["dead-code"], severities=["high"])) == [ids["d"]]

        # count_issues mirrors the filters.
        assert db.count_issues(severities=["high", "low"]) == 5
        assert db.count_issues(aspect_ids=["dead-code"], severities=["high"]) == 1
        assert db.count_issues(aspect_ids=["nope"]) == 0


def test_list_issues_issue_ids_filter(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with Database(tmp_path / "khorek.db", project) as db:
        ids = _seed_filter_fixtures(db)

        # OR within the flag.
        assert set(_ids(db.list_issues(issue_ids=[ids["a"], ids["c"]]))) == {
            ids["a"],
            ids["c"],
        }
        # Duplicates and unsorted values collapse to the same set.
        assert _ids(db.list_issues(issue_ids=[ids["b"], ids["b"], ids["a"]])) == [
            ids["a"],
            ids["b"],
        ]

        # AND across flags.
        assert _ids(db.list_issues(issue_ids=[ids["a"], ids["c"]], aspect_ids=["dead-code"])) == [
            ids["c"]
        ]
        assert _ids(db.list_issues(issue_ids=[ids["a"]], severities=["high"])) == [ids["a"]]
        assert _ids(db.list_issues(issue_ids=[ids["a"]], path_prefixes=["src/other.py"])) == []

        # count_issues mirrors the filters.
        assert db.count_issues(issue_ids=[ids["a"], ids["c"]]) == 2
        assert db.count_issues(issue_ids=[ids["a"], ids["c"]], aspect_ids=["dead-code"]) == 1

        # An empty filter is no constraint; an unknown id matches nothing.
        assert db.count_issues(issue_ids=[]) == len(ids)
        assert db.list_issues(issue_ids=[999999]) == []
        assert db.count_issues(issue_ids=[999999]) == 0


def test_list_issues_path_prefix_filters(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with Database(tmp_path / "khorek.db", project) as db:
        ids = _seed_filter_fixtures(db)

        # Prefix match (directory-like), OR within the flag.
        assert set(_ids(db.list_issues(path_prefixes=["src/khorek/"]))) == {
            ids["a"],
            ids["b"],
            ids["c"],
        }
        assert _ids(db.list_issues(path_prefixes=["src/khorek/agents"])) == [ids["c"]]
        assert set(_ids(db.list_issues(path_prefixes=["src/khorek/", "src/other.py"]))) == {
            ids["a"],
            ids["b"],
            ids["c"],
            ids["d"],
        }

        # '%' in a prefix is literal, not a LIKE wildcard.
        assert _ids(db.list_issues(path_prefixes=["src/50%"])) == [ids["pct"]]
        # '_' in a prefix is literal, not a LIKE wildcard.
        assert _ids(db.list_issues(path_prefixes=["src/a_b"])) == [ids["und"]]

        # AND with another flag.
        assert _ids(db.list_issues(path_prefixes=["src/khorek/"], severities=["medium"])) == [
            ids["c"]
        ]


def test_list_issues_invalid_severity_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with Database(tmp_path / "khorek.db", project) as db:
        db.insert_issue(
            aspect_id="comments",
            path="src/a.py",
            description="x",
            line_start=1,
            line_end=1,
            severity="high",
        )
        with pytest.raises(ValueError, match="invalid severity"):
            db.list_issues(severities=["bogus"])
        with pytest.raises(ValueError, match="invalid severity"):
            db.count_issues(severities=["bogus"])


def test_delete_issues_batch(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with Database(tmp_path / "khorek.db", project) as db:
        ids = _seed_filter_fixtures(db)
        assert db.count_issues() == len(ids)

        assert db.delete_issues([ids["a"], ids["d"]]) == 2
        remaining = set(_ids(db.list_issues()))
        assert remaining == set(ids.values()) - {ids["a"], ids["d"]}

        # Unknown ids are ignored; duplicates collapse.
        assert db.delete_issues([999999]) == 0
        assert db.delete_issues([ids["b"], ids["b"]]) == 1
        assert db.delete_issues([]) == 0
        assert db.count_issues() == len(ids) - 3


def test_readonly_database_rejects_missing_file(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = tmp_path / "nested" / "khorek.db"
    with pytest.raises(DatabaseError, match="does not exist"):
        Database(target, project, readonly=True)
    assert not target.parent.exists()  # no parent directory created


def test_readonly_database_reads_and_validates(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    path = tmp_path / "khorek.db"
    with Database(path, project) as db:
        issue_id = db.insert_issue(
            aspect_id="comments",
            path="src/a.py",
            description="x",
            line_start=1,
            line_end=1,
            severity="high",
        )
        db.upsert_state("comments", "src/a.py", "sha", "issues")

    with Database(path, project, readonly=True) as ro:
        assert ro.readonly is True
        assert ro.get_issue(issue_id) is not None
        assert ro.list_issues(severities=["high"]) == [ro.get_issue(issue_id)]
        assert ro.count_issues() == 1
        assert ro.get_state("comments", "src/a.py") is not None

    # A wrong project root is still rejected, leaving the file byte-identical.
    other = tmp_path / "other"
    other.mkdir()
    before = path.read_bytes()
    with pytest.raises(DatabaseError, match="belongs to project"):
        Database(path, other, readonly=True)
    assert path.read_bytes() == before

    # A schema-version mismatch is still rejected.
    with sqlite3.connect(path) as raw:
        raw.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
        raw.commit()
    with pytest.raises(DatabaseError, match="schema version"):
        Database(path, project, readonly=True)


def test_readonly_database_creates_no_ddl(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    # A database with valid meta but no khorek tables: readonly must not create them.
    path = tmp_path / "bare.db"
    with sqlite3.connect(path) as raw:
        raw.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        raw.execute(
            "INSERT INTO meta (key, value) VALUES ('project_root', ?)",
            (str(project.resolve()),),
        )
        raw.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        raw.commit()

    with Database(path, project, readonly=True) as ro:
        assert ro.readonly is True
    with sqlite3.connect(path) as raw:
        tables = {
            row[0] for row in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert tables == {"meta"}


def test_update_issue_rejects_unknown_fields(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with Database(tmp_path / "khorek.db", project) as db:
        issue_id = db.insert_issue(
            aspect_id="comments",
            path="src/a.py",
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
            description="kept",
            line_start=1,
            line_end=1,
            severity="low",
        )
        db.insert_issue(
            aspect_id="gone",
            path="src/a.py",
            description="gone",
            line_start=1,
            line_end=1,
            severity="low",
        )
        db.insert_issue(
            aspect_id="gone",
            path="src/b.py",
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
