"""Issue triage: staleness, prune, dismiss, delete, and rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

import khorek.issues as issues_mod
from khorek.db import Database, Issue
from khorek.files import sha256_file
from khorek.issues import (
    CHANGED,
    FRESH,
    MISSING,
    STATUS_CHANGED,
    STATUS_FRESH,
    STATUS_MISSING,
    IssueOutcome,
    delete,
    dismiss,
    issue_status,
    prune_stale,
    render_list,
    render_show,
    staleness,
)

CONTENT_A = "# a correct comment\nvalue = 1\n"
CONTENT_B = "x = 1\n"


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A project root plus two source files (a.py, b.py)."""
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    a = root / "src" / "a.py"
    a.write_text(CONTENT_A)
    b = root / "src" / "b.py"
    b.write_text(CONTENT_B)
    return root, a, b


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "khorek.db"


def _seed(
    root: Path,
    db: Database,
    path: str,
    *,
    aspect_id: str = "comments",
    description: str = "an issue",
    line_start: int = 1,
    line_end: int = 1,
    severity: str = "high",
) -> int:
    return db.insert_issue(
        aspect_id=aspect_id,
        path=path,
        description=description,
        line_start=line_start,
        line_end=line_end,
        severity=severity,
    )


# -- staleness -------------------------------------------------------------


def test_staleness_fresh_changed_missing(project, db_path) -> None:
    root, a, b = project
    with Database(db_path, root) as db:
        fresh = _seed(root, db, "src/a.py")
        sibling = _seed(root, db, "src/a.py")  # sibling on same path
        changed = _seed(root, db, "src/b.py")
        missing = _seed(root, db, "src/gone.py")
        db.upsert_state("comments", "src/a.py", sha256_file(a), "issues")
        db.upsert_state("comments", "src/b.py", sha256_file(b), "issues")

    # Make b.py changed; gone.py was never created so it is already missing.
    b.write_text("x = 2\n")

    with Database(db_path, root, readonly=True) as db:
        statuses = staleness(db, root)
    assert statuses[fresh] == FRESH
    assert statuses[sibling] == FRESH  # same path, same hash
    assert statuses[changed] == CHANGED
    assert statuses[missing] == MISSING


def test_staleness_hashes_each_distinct_path_once(project, db_path, monkeypatch) -> None:
    root, a, b = project
    with Database(db_path, root) as db:
        _seed(root, db, "src/a.py")
        _seed(root, db, "src/a.py")
        _seed(root, db, "src/b.py")
        _seed(root, db, "src/b.py")
        _seed(root, db, "src/gone.py")

    calls: list[str] = []
    real = sha256_file

    def counting(path: Path) -> str:
        calls.append(str(path))
        return real(path)

    monkeypatch.setattr(issues_mod, "sha256_file", counting)
    with Database(db_path, root, readonly=True) as db:
        staleness(db, root)
    # The two existing distinct paths are each hashed once; the missing file is
    # detected via is_file() and never hashed.
    assert len(calls) == 2
    assert {Path(call).name for call in calls} == {"a.py", "b.py"}


def test_issue_status_labels(project, db_path) -> None:
    root, a, b = project
    with Database(db_path, root) as db:
        fresh = _seed(root, db, "src/a.py")
        changed = _seed(root, db, "src/b.py")
        missing = _seed(root, db, "src/gone.py")
        db.upsert_state("comments", "src/a.py", sha256_file(a), "issues")
        db.upsert_state("comments", "src/b.py", sha256_file(b), "issues")
    b.write_text("changed\n")
    with Database(db_path, root, readonly=True) as db:
        for issue_id, expected in (
            (fresh, STATUS_FRESH),
            (changed, STATUS_CHANGED),
            (missing, STATUS_MISSING),
        ):
            issue = db.get_issue(issue_id)
            assert issue is not None
            assert issue_status(db, root, issue) == expected


# -- prune -----------------------------------------------------------------


def test_prune_stale_deletes_only_stale(project, db_path) -> None:
    root, a, b = project
    with Database(db_path, root) as db:
        fresh = _seed(root, db, "src/a.py")
        changed = _seed(root, db, "src/b.py")
        missing = _seed(root, db, "src/gone.py")
        db.upsert_state("comments", "src/a.py", sha256_file(a), "issues")
        db.upsert_state("comments", "src/b.py", sha256_file(b), "issues")
    b.write_text("x = 2\n")

    with Database(db_path, root) as db:
        pruned = prune_stale(db, root)
    assert pruned == 2
    with Database(db_path, root, readonly=True) as db:
        remaining = [issue.id for issue in db.list_issues()]
        assert remaining == [fresh]
        assert changed not in remaining and missing not in remaining


def test_prune_stale_none(project, db_path) -> None:
    root, a, b = project
    with Database(db_path, root) as db:
        _seed(root, db, "src/a.py")
        _seed(root, db, "src/b.py")
        db.upsert_state("comments", "src/a.py", sha256_file(a), "issues")
        db.upsert_state("comments", "src/b.py", sha256_file(b), "issues")
    with Database(db_path, root) as db:
        assert prune_stale(db, root) == 0


# -- dismiss ---------------------------------------------------------------


def test_dismiss_drained_pair_marked_clean(project, db_path) -> None:
    root, a, b = project
    with Database(db_path, root) as db:
        issue = _seed(root, db, "src/a.py")
        db.upsert_state("comments", "src/a.py", sha256_file(a), "issues")

    with Database(db_path, root) as db:
        outcomes = dismiss(db, root, [issue])
    assert [o.found for o in outcomes] == [True]
    assert outcomes[0].action == "pair marked clean"
    assert (outcomes[0].aspect_id, outcomes[0].path, outcomes[0].lines) == (
        "comments",
        "src/a.py",
        "1",
    )
    with Database(db_path, root, readonly=True) as db:
        assert db.get_issue(issue) is None
        state = db.get_state("comments", "src/a.py")
        assert state is not None
        assert (state.sha256, state.outcome) == (sha256_file(a), "clean")


def test_dismiss_stale_pair_uses_current_sha(project, db_path) -> None:
    root, a, b = project
    with Database(db_path, root) as db:
        issue = _seed(root, db, "src/a.py")
        db.upsert_state("comments", "src/a.py", sha256_file(a), "issues")
    # File changed after the issue was recorded.
    a.write_text("# now different\n")

    with Database(db_path, root) as db:
        outcomes = dismiss(db, root, [issue])
    assert outcomes[0].action == "pair marked clean"
    with Database(db_path, root, readonly=True) as db:
        state = db.get_state("comments", "src/a.py")
        assert state is not None
        # Stamped with the *current* sha, not the recorded one.
        assert state.sha256 == sha256_file(a)
        assert state.outcome == "clean"


def test_dismiss_siblings_leaves_state(project, db_path) -> None:
    root, a, b = project
    with Database(db_path, root) as db:
        first = _seed(root, db, "src/a.py", line_start=1, line_end=1)
        _seed(root, db, "src/a.py", line_start=2, line_end=3)
        db.upsert_state("comments", "src/a.py", sha256_file(a), "issues")

    with Database(db_path, root) as db:
        outcomes = dismiss(db, root, [first])
    assert outcomes[0].action == ""
    with Database(db_path, root, readonly=True) as db:
        state = db.get_state("comments", "src/a.py")
        assert state is not None
        assert (state.sha256, state.outcome) == (sha256_file(a), "issues")
        assert len(db.open_issues_for("comments", "src/a.py")) == 1


def test_dismiss_multiple_ids_same_pair(project, db_path) -> None:
    root, a, b = project
    with Database(db_path, root) as db:
        first = _seed(root, db, "src/a.py", line_start=1, line_end=1)
        second = _seed(root, db, "src/a.py", line_start=2, line_end=2)
        db.upsert_state("comments", "src/a.py", sha256_file(a), "issues")

    with Database(db_path, root) as db:
        outcomes = dismiss(db, root, [first, second])
    # The pair is evaluated per id: only the last one drains it.
    assert [o.action for o in outcomes] == ["", "pair marked clean"]
    with Database(db_path, root, readonly=True) as db:
        assert db.get_state("comments", "src/a.py") is not None


def test_dismiss_missing_file_removes_state(project, db_path) -> None:
    root, a, b = project
    with Database(db_path, root) as db:
        issue = _seed(root, db, "src/gone.py")
        db.upsert_state("comments", "src/gone.py", "deadbeef", "issues")

    with Database(db_path, root) as db:
        outcomes = dismiss(db, root, [issue])
    assert outcomes[0].action == "pair removed (file not found)"
    with Database(db_path, root, readonly=True) as db:
        assert db.get_issue(issue) is None
        assert db.get_state("comments", "src/gone.py") is None


def test_dismiss_unknown_id(project, db_path) -> None:
    root, a, b = project
    with Database(db_path, root) as db:
        outcomes = dismiss(db, root, [4242])
    assert outcomes == [IssueOutcome(issue_id=4242, found=False)]


# -- delete ----------------------------------------------------------------


def test_delete_removes_state_even_with_siblings(project, db_path) -> None:
    root, a, b = project
    with Database(db_path, root) as db:
        first = _seed(root, db, "src/a.py", line_start=1, line_end=1)
        _seed(root, db, "src/a.py", line_start=2, line_end=2)
        db.upsert_state("comments", "src/a.py", sha256_file(a), "issues")

    with Database(db_path, root) as db:
        outcomes = delete(db, root, [first])
    assert outcomes[0].action == "state cleared"
    with Database(db_path, root, readonly=True) as db:
        assert db.get_issue(first) is None
        assert db.get_state("comments", "src/a.py") is None
        # The sibling issue is untouched.
        assert len(db.open_issues_for("comments", "src/a.py")) == 1


def test_delete_missing_file(project, db_path) -> None:
    root, a, b = project
    with Database(db_path, root) as db:
        issue = _seed(root, db, "src/gone.py")
        db.upsert_state("comments", "src/gone.py", "x", "issues")
    with Database(db_path, root) as db:
        outcomes = delete(db, root, [issue])
    assert outcomes[0].action == "state cleared"
    with Database(db_path, root, readonly=True) as db:
        assert db.get_issue(issue) is None
        assert db.get_state("comments", "src/gone.py") is None


def test_delete_unknown_id(project, db_path) -> None:
    root, a, b = project
    with Database(db_path, root) as db:
        outcomes = delete(db, root, [7])
    assert outcomes == [IssueOutcome(issue_id=7, found=False)]


# -- render_list -----------------------------------------------------------


def _issue(
    id: int,
    aspect: str = "comments",
    path: str = "src/a.py",
    line_start: int = 1,
    line_end: int = 1,
    severity: str = "high",
    description: str = "an issue",
    retry_count: int = 0,
    created_at: str = "2026-11-12T10:15:30+00:00",
    updated_at: str = "2026-11-12T10:15:30+00:00",
    last_error: str | None = None,
):
    return Issue(
        id=id,
        aspect_id=aspect,
        path=path,
        description=description,
        line_start=line_start,
        line_end=line_end,
        severity=severity,
        retry_count=retry_count,
        last_error=last_error,
        created_at=created_at,
        updated_at=updated_at,
    )


def test_render_list_empty() -> None:
    assert render_list([], stale=None) == "no open issues"
    assert render_list([], stale={}) == "no open issues"


def test_render_list_alignment_and_layout() -> None:
    issues = [
        _issue(10, path="src/khorek/db.py", line_start=234, line_end=238, severity="high"),
        _issue(
            12,
            path="src/khorek/cli.py",
            line_start=88,
            line_end=88,
            severity="low",
            retry_count=2,
            created_at="2026-11-13T09:00:00+00:00",
        ),
        _issue(
            15,
            aspect="dead-code",
            path="src/khorek/lock.py",
            line_start=101,
            line_end=115,
            severity="medium",
            created_at="2026-11-13T11:00:00+00:00",
        ),
    ]
    stale = {15: CHANGED}
    rendered = render_list(issues, stale=stale)
    lines = rendered.splitlines()
    # Columns align: the path column starts at the same index on every header row.
    headers = [line for line in lines if " | " in line]
    path_starts = [line.index("src/") for line in headers]
    assert len(set(path_starts)) == 1
    assert headers[0] == "10 | 2026-11-12  comments   high    src/khorek/db.py:234-238"
    assert headers[1] == "12 | 2026-11-13  comments   low     src/khorek/cli.py:88"
    assert headers[2] == "15 | 2026-11-13  dead-code  medium  src/khorek/lock.py:101-115  [stale]"
    # Retried line appears only above the description when retry_count > 0.
    assert "    Retried: 2" in lines
    # Blank line between issues, then the footer.
    assert rendered.endswith("3 open issues (1 stale)")


def test_render_list_single_line_and_plural_footer() -> None:
    rendered = render_list([_issue(1)], stale=None)
    assert "src/a.py:1\n" in rendered
    assert rendered.endswith("1 open issue")

    rendered = render_list([_issue(1), _issue(2)], stale=None)
    assert rendered.endswith("2 open issues")


def test_render_list_stale_label_and_footer_count() -> None:
    issues = [_issue(1), _issue(2)]
    rendered = render_list(issues, stale={1: MISSING, 2: FRESH})
    assert "  [stale]" in rendered
    assert rendered.endswith("2 open issues (1 stale)")
    # Without stale info, no label and no stale count.
    rendered = render_list(issues, stale=None)
    assert "[stale]" not in rendered
    assert rendered.endswith("2 open issues")


def test_render_list_multiline_description_indented() -> None:
    issue = _issue(1, description="first line\n\nthird line")
    rendered = render_list([issue], stale=None)
    lines = rendered.splitlines()
    assert "    first line" in lines
    assert "" in lines  # internal blank line stays blank
    assert "    third line" in lines


# -- render_show -----------------------------------------------------------


def test_render_show_fresh() -> None:
    issue = _issue(
        10,
        path="src/khorek/db.py",
        line_start=234,
        line_end=238,
        description="Comment says the loop is O(n)",
        created_at="2026-11-12T10:15:30+00:00",
        updated_at="2026-11-12T10:15:30+00:00",
    )
    rendered = render_show(issue, status=STATUS_FRESH)
    assert rendered.splitlines() == [
        "#10  comments  src/khorek/db.py:234-238",
        "severity:  high",
        "status:    fresh",
        "retries:   0",
        "created:   2026-11-12T10:15:30+00:00",
        "updated:   2026-11-12T10:15:30+00:00",
        "",
        "Comment says the loop is O(n)",
    ]


def test_render_show_stale_with_last_error() -> None:
    issue = _issue(
        10,
        path="src/khorek/db.py",
        line_start=234,
        line_end=238,
        severity="high",
        description="Comment says the loop is O(n)",
        retry_count=2,
        last_error="verification failed: uv run pytest -q (exited with code 1)",
        created_at="2026-11-12T10:15:30+00:00",
        updated_at="2026-11-12T14:02:11+00:00",
    )
    rendered = render_show(issue, status=STATUS_CHANGED)
    assert rendered.splitlines() == [
        "#10  comments  src/khorek/db.py:234-238",
        "severity:  high",
        "status:    stale (file changed since recorded)",
        "retries:   2",
        "created:   2026-11-12T10:15:30+00:00",
        "updated:   2026-11-12T14:02:11+00:00",
        "last_error:",
        "    verification failed: uv run pytest -q (exited with code 1)",
        "",
        "Comment says the loop is O(n)",
    ]


def test_render_show_missing_status_and_single_line() -> None:
    issue = _issue(7, path="src/gone.py", line_start=4, line_end=4, severity="low")
    rendered = render_show(issue, status=STATUS_MISSING)
    lines = rendered.splitlines()
    assert lines[0] == "#7  comments  src/gone.py:4"
    assert "status:    stale (file missing)" in lines
    assert "last_error:" not in lines
