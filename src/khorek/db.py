"""SQLite persistence for khorek.

The database is the source of truth: per-(aspect, file) state rows and open
issues.  Paths are stored relative to the project root; the resolved root and
the schema version live in ``meta`` so a database can never be silently reused
for another project.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import urllib.parse
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- One row per explored (aspect, file) pair. Written after every explore.
CREATE TABLE IF NOT EXISTS file_aspect_state (
  aspect_id TEXT NOT NULL,
  path      TEXT NOT NULL,
  sha256    TEXT NOT NULL,
  outcome   TEXT NOT NULL CHECK (outcome IN ('clean', 'issues', 'truncated')),
  updated_at TEXT NOT NULL,
  PRIMARY KEY (aspect_id, path)
);

-- A row exists iff the issue is open.
CREATE TABLE IF NOT EXISTS issues (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  aspect_id   TEXT NOT NULL,
  path        TEXT NOT NULL,
  sha256      TEXT NOT NULL,          -- hash of the file when the issue was found
  description TEXT NOT NULL,          -- free form, what is wrong
  line_start  INTEGER NOT NULL,       -- 1-based line number (clamped upstream)
  line_end    INTEGER NOT NULL,
  severity    TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high')),
  retry_count INTEGER NOT NULL DEFAULT 0,
  last_error  TEXT,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_issues_aspect_path ON issues (aspect_id, path);
CREATE INDEX IF NOT EXISTS idx_issues_path ON issues (path);
CREATE INDEX IF NOT EXISTS idx_state_path ON file_aspect_state (path);
"""

_STATE_OUTCOMES = frozenset({"clean", "issues", "truncated"})
_SEVERITIES = frozenset({"low", "medium", "high"})

_SLUG_READABLE_LIMIT = 120


class DatabaseError(Exception):
    """Fatal database problem (wrong project root, schema mismatch, I/O)."""


@dataclass(frozen=True)
class StateRow:
    """One ``file_aspect_state`` row."""

    aspect_id: str
    path: str
    sha256: str
    outcome: str
    updated_at: str


@dataclass(frozen=True)
class Issue:
    """One open issue row."""

    id: int
    aspect_id: str
    path: str
    sha256: str
    description: str
    line_start: int
    line_end: int
    severity: str
    retry_count: int
    last_error: str | None
    created_at: str
    updated_at: str


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _canonical(path: Path) -> Path:
    """Canonical absolute path (symlinks resolved, path need not exist)."""
    return Path(os.path.realpath(path))


def _slug_readable(real_path: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", real_path.strip("/"))
    readable = readable.strip("-._")
    if not readable:
        readable = "project"
    if len(readable) > _SLUG_READABLE_LIMIT:
        readable = readable[-_SLUG_READABLE_LIMIT:].lstrip("-._") or "project"
    return readable


def default_db_path(project_root: Path, home: Path | None = None) -> Path:
    """Return ``~/.khorek/<readable-path>-<8 hex>.db`` for *project_root*."""
    real = os.path.realpath(project_root)
    digest = hashlib.sha256(real.encode("utf-8")).hexdigest()[:8]
    base = home if home is not None else Path.home()
    return base / ".khorek" / f"{_slug_readable(real)}-{digest}.db"


class Database:
    """An open khorek database with schema applied and meta validated.

    ``readonly=True`` opens an existing database for reading only: no parent
    directory, no WAL pragma, no DDL, no meta stamp.  This is required for the
    lockless read paths (``issues --stale``, ``issues show``, and a list whose
    run lock is held): the write-mode preamble would take a brief write lock
    and stall or fail against a concurrent writer.
    """

    def __init__(self, path: Path, project_root: Path, *, readonly: bool = False) -> None:
        self.path = Path(path)
        self.project_root = _canonical(project_root)
        self.readonly = readonly
        self._connection: sqlite3.Connection | None = None
        try:
            if readonly:
                if not self.path.is_file():
                    raise DatabaseError(f"database does not exist: {self.path}")
                uri = f"file:{urllib.parse.quote(str(self.path.resolve()), safe='/')}?mode=ro"
                connection = sqlite3.connect(uri, uri=True)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(str(self.path))
            connection.row_factory = sqlite3.Row
            self._connection = connection
            # A meta table means this is an existing database.  Validate it
            # before changing the journal mode or creating any khorek table:
            # a root/schema mismatch must leave the file byte-identical.
            existing_meta = self._meta_table_exists(connection)
            if existing_meta:
                self._validate_meta(connection)
            if not readonly:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=NORMAL")
                connection.executescript(_SCHEMA)
                if not existing_meta:
                    self._stamp_meta(connection)
                connection.commit()
        except DatabaseError:
            self.close()
            raise
        except (OSError, sqlite3.Error) as exc:
            self.close()
            raise DatabaseError(f"cannot open database {self.path}: {exc}") from exc

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise DatabaseError("database is closed")
        return self._connection

    def _meta_table_exists(self, connection: sqlite3.Connection) -> bool:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
        ).fetchone()
        return row is not None

    def _read_meta(self, connection: sqlite3.Connection) -> dict[str, str]:
        rows = connection.execute("SELECT key, value FROM meta").fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def _stamp_meta(self, connection: sqlite3.Connection) -> None:
        connection.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("project_root", str(self.project_root)),
                ("schema_version", SCHEMA_VERSION),
            ],
        )

    def _validate_meta(self, connection: sqlite3.Connection) -> None:
        existing = self._read_meta(connection)
        missing = {"project_root", "schema_version"} - existing.keys()
        if missing:
            raise DatabaseError(
                f"database {self.path} has incomplete meta (missing {', '.join(sorted(missing))})"
            )

        stored_root = existing["project_root"]
        stored_canonical = _canonical(Path(stored_root))
        if stored_canonical != self.project_root:
            raise DatabaseError(
                f"database {self.path} belongs to project {stored_root}, not {self.project_root}"
            )
        stored_version = existing["schema_version"]
        if stored_version != SCHEMA_VERSION:
            raise DatabaseError(
                f"database {self.path} uses schema version {stored_version}, "
                f"expected {SCHEMA_VERSION}"
            )

    # -- lifecycle -----------------------------------------------------

    def close(self) -> None:
        """Close the database; safe to call more than once."""
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.commit()
            finally:
                connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    # -- state ---------------------------------------------------------

    def get_state(self, aspect_id: str, path: str) -> StateRow | None:
        row = self.connection.execute(
            "SELECT aspect_id, path, sha256, outcome, updated_at "
            "FROM file_aspect_state WHERE aspect_id = ? AND path = ?",
            (aspect_id, path),
        ).fetchone()
        if row is None:
            return None
        return StateRow(
            aspect_id=str(row["aspect_id"]),
            path=str(row["path"]),
            sha256=str(row["sha256"]),
            outcome=str(row["outcome"]),
            updated_at=str(row["updated_at"]),
        )

    def upsert_state(self, aspect_id: str, path: str, sha256: str, outcome: str) -> None:
        if outcome not in _STATE_OUTCOMES:
            raise ValueError(f"invalid state outcome: {outcome!r}")
        with self.connection:
            self.connection.execute(
                "INSERT INTO file_aspect_state (aspect_id, path, sha256, outcome, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (aspect_id, path) DO UPDATE SET "
                "sha256 = excluded.sha256, outcome = excluded.outcome, "
                "updated_at = excluded.updated_at",
                (aspect_id, path, sha256, outcome, utc_now()),
            )

    def delete_state(self, aspect_id: str, path: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM file_aspect_state WHERE aspect_id = ? AND path = ?",
                (aspect_id, path),
            )
        return cursor.rowcount > 0

    def list_state(self) -> list[StateRow]:
        rows = self.connection.execute(
            "SELECT aspect_id, path, sha256, outcome, updated_at "
            "FROM file_aspect_state ORDER BY aspect_id, path"
        ).fetchall()
        return [
            StateRow(
                aspect_id=str(row["aspect_id"]),
                path=str(row["path"]),
                sha256=str(row["sha256"]),
                outcome=str(row["outcome"]),
                updated_at=str(row["updated_at"]),
            )
            for row in rows
        ]

    # -- issues --------------------------------------------------------

    def insert_issue(
        self,
        *,
        aspect_id: str,
        path: str,
        sha256: str,
        description: str,
        line_start: int,
        line_end: int,
        severity: str,
    ) -> int:
        if severity not in _SEVERITIES:
            raise ValueError(f"invalid severity: {severity!r}")
        now = utc_now()
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO issues "
                "(aspect_id, path, sha256, description, line_start, line_end, severity, "
                "retry_count, last_error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)",
                (
                    aspect_id,
                    path,
                    sha256,
                    description,
                    line_start,
                    line_end,
                    severity,
                    now,
                    now,
                ),
            )
        issue_id = cursor.lastrowid
        if issue_id is None:
            raise DatabaseError("failed to insert issue")
        return int(issue_id)

    def delete_issue(self, issue_id: int) -> bool:
        with self.connection:
            cursor = self.connection.execute("DELETE FROM issues WHERE id = ?", (issue_id,))
        return cursor.rowcount > 0

    def delete_issues(self, ids: Sequence[int]) -> int:
        """Delete the given issue ids in one transaction; return how many were removed."""
        unique = sorted({int(issue_id) for issue_id in ids})
        if not unique:
            return 0
        placeholders = ", ".join("?" for _ in unique)
        with self.connection:
            cursor = self.connection.execute(
                f"DELETE FROM issues WHERE id IN ({placeholders})", unique
            )
        return cursor.rowcount

    def delete_issues_for(self, aspect_id: str, path: str) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM issues WHERE aspect_id = ? AND path = ?", (aspect_id, path)
            )
        return cursor.rowcount

    def delete_issues_for_path(self, path: str) -> int:
        with self.connection:
            cursor = self.connection.execute("DELETE FROM issues WHERE path = ?", (path,))
        return cursor.rowcount

    def list_issues(
        self,
        *,
        aspect_id: str | None = None,
        path: str | None = None,
        aspect_ids: Collection[str] | None = None,
        severities: Collection[str] | None = None,
        path_prefixes: Collection[str] | None = None,
    ) -> list[Issue]:
        """Return open issues sorted by id.

        ``aspect_id``/``path`` are exact matches (used by the workflows);
        ``aspect_ids``/``severities``/``path_prefixes`` are repeatable filters
        where values within one argument are OR-ed and different arguments
        are AND-ed.  A prefix is matched literally (``%``/``_`` escaped).
        """
        where, parameters = _issue_filter(
            aspect_id=aspect_id,
            path=path,
            aspect_ids=aspect_ids,
            severities=severities,
            path_prefixes=path_prefixes,
        )
        rows = self.connection.execute(
            f"SELECT * FROM issues{where} ORDER BY id",
            parameters,
        ).fetchall()
        return [_issue_from_row(row) for row in rows]

    def open_issues_for(self, aspect_id: str, path: str) -> list[Issue]:
        return self.list_issues(aspect_id=aspect_id, path=path)

    def open_issues_for_path(self, path: str) -> list[Issue]:
        return self.list_issues(path=path)

    def get_issue(self, issue_id: int) -> Issue | None:
        row = self.connection.execute("SELECT * FROM issues WHERE id = ?", (issue_id,)).fetchone()
        return None if row is None else _issue_from_row(row)

    def count_issues(
        self,
        *,
        aspect_id: str | None = None,
        path: str | None = None,
        aspect_ids: Collection[str] | None = None,
        severities: Collection[str] | None = None,
        path_prefixes: Collection[str] | None = None,
    ) -> int:
        where, parameters = _issue_filter(
            aspect_id=aspect_id,
            path=path,
            aspect_ids=aspect_ids,
            severities=severities,
            path_prefixes=path_prefixes,
        )
        row = self.connection.execute(
            f"SELECT COUNT(*) AS total FROM issues{where}", parameters
        ).fetchone()
        return int(row["total"]) if row is not None else 0

    def update_issue(self, issue_id: int, **fields: Any) -> bool:
        """Update selected issue columns and bump ``updated_at``.

        Unknown column names are rejected so callers cannot smuggle SQL.
        """
        allowed = {
            "aspect_id",
            "path",
            "sha256",
            "description",
            "line_start",
            "line_end",
            "severity",
            "retry_count",
            "last_error",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown issue fields: {', '.join(sorted(unknown))}")
        if not fields:
            return self.get_issue(issue_id) is not None
        if "severity" in fields and fields["severity"] not in _SEVERITIES:
            raise ValueError(f"invalid severity: {fields['severity']!r}")
        assignments = ", ".join(f"{name} = ?" for name in fields)
        parameters = [*fields.values(), utc_now(), issue_id]
        with self.connection:
            cursor = self.connection.execute(
                f"UPDATE issues SET {assignments}, updated_at = ? WHERE id = ?",
                parameters,
            )
        return cursor.rowcount > 0

    # -- drift ---------------------------------------------------------

    def prune_drift(self, aspect_ids: Collection[str]) -> int:
        """Delete issues and state rows whose aspect is not in *aspect_ids*."""
        ids = sorted(set(aspect_ids))
        if ids:
            placeholders = ", ".join("?" for _ in ids)
            issue_sql = f"DELETE FROM issues WHERE aspect_id NOT IN ({placeholders})"
            state_sql = f"DELETE FROM file_aspect_state WHERE aspect_id NOT IN ({placeholders})"
        else:
            issue_sql = "DELETE FROM issues"
            state_sql = "DELETE FROM file_aspect_state"
        with self.connection:
            issues_removed = self.connection.execute(issue_sql, ids).rowcount
            state_removed = self.connection.execute(state_sql, ids).rowcount
        return int(issues_removed + state_removed)


def _like_escape(value: str) -> str:
    """Escape LIKE wildcards so *value* matches literally as a prefix."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _issue_filter(
    *,
    aspect_id: str | None = None,
    path: str | None = None,
    aspect_ids: Collection[str] | None = None,
    severities: Collection[str] | None = None,
    path_prefixes: Collection[str] | None = None,
) -> tuple[str, list[str]]:
    """Build the WHERE clause (and parameters) for issue queries.

    Exact ``aspect_id``/``path`` match; ``aspect_ids``/``severities`` are
    OR-ed within the argument and AND-ed across arguments; ``path_prefixes``
    are literal prefix matches (``%``/``_`` escaped), OR-ed within the
    argument.
    """
    clauses: list[str] = []
    parameters: list[str] = []
    if aspect_id is not None:
        clauses.append("aspect_id = ?")
        parameters.append(aspect_id)
    if path is not None:
        clauses.append("path = ?")
        parameters.append(path)
    if aspect_ids:
        placeholders = ", ".join("?" for _ in aspect_ids)
        clauses.append(f"aspect_id IN ({placeholders})")
        parameters.extend(aspect_ids)
    if severities:
        unknown = [severity for severity in severities if severity not in _SEVERITIES]
        if unknown:
            raise ValueError(f"invalid severity: {unknown[0]!r}")
        placeholders = ", ".join("?" for _ in severities)
        clauses.append(f"severity IN ({placeholders})")
        parameters.extend(severities)
    if path_prefixes:
        prefix_clauses = []
        for prefix in path_prefixes:
            prefix_clauses.append("path LIKE ? ESCAPE '\\'")
            parameters.append(_like_escape(prefix) + "%")
        if len(prefix_clauses) == 1:
            clauses.append(prefix_clauses[0])
        else:
            clauses.append("(" + " OR ".join(prefix_clauses) + ")")
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, parameters


def _issue_from_row(row: sqlite3.Row) -> Issue:
    return Issue(
        id=int(row["id"]),
        aspect_id=str(row["aspect_id"]),
        path=str(row["path"]),
        sha256=str(row["sha256"]),
        description=str(row["description"]),
        line_start=int(row["line_start"]),
        line_end=int(row["line_end"]),
        severity=str(row["severity"]),
        retry_count=int(row["retry_count"]),
        last_error=None if row["last_error"] is None else str(row["last_error"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


__all__ = [
    "SCHEMA_VERSION",
    "Database",
    "DatabaseError",
    "Issue",
    "StateRow",
    "default_db_path",
    "utc_now",
]
