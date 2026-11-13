"""Unified diff of a fix attempt's changed files."""

from __future__ import annotations

from pathlib import Path

from khorek.diff import format_changes, format_file_diff
from khorek.snapshots import Snapshotter


def test_file_diff_shows_modification_anchors_and_lines() -> None:
    diff = format_file_diff("value = 1\n", "value = 2\n", "src/a.py")
    assert "--- a/src/a.py" in diff
    assert "+++ b/src/a.py" in diff
    assert "@@ -1 +1 @@" in diff
    assert "-value = 1" in diff
    assert "+value = 2" in diff


def test_file_diff_of_a_created_file() -> None:
    diff = format_file_diff("", "gamma = 1\n", "src/c.py")
    assert "--- a/src/c.py" in diff
    assert "+++ b/src/c.py" in diff
    assert "@@ -0,0 +1 @@" in diff
    assert "+gamma = 1" in diff


def test_file_diff_keeps_three_lines_of_context() -> None:
    diff = format_file_diff("l1\nl2\nl3\nl4\n", "l1\nl2 changed\nl3\nl4\n", "src/x.py")
    assert "@@ -1,4 +1,4 @@" in diff
    assert " l1" in diff
    assert " l4" in diff


def test_file_diff_is_empty_when_identical() -> None:
    assert format_file_diff("same\n", "same\n", "src/a.py") == ""


def test_format_changes_concatenates_changed_files(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    a = root / "src" / "a.py"
    b = root / "src" / "b.py"
    c = root / "src" / "c.py"
    a.write_text("alpha = 1\n", encoding="utf-8")
    b.write_text("beta = 1\n", encoding="utf-8")

    snapshotter = Snapshotter(root)
    snapshotter.capture(a)
    snapshotter.capture(b)
    snapshotter.capture(c)  # created file: does not exist yet
    a.write_text("alpha = 2\n", encoding="utf-8")
    c.write_text("gamma = 1\n", encoding="utf-8")  # b stays unchanged

    diff = format_changes(root, snapshotter)
    assert "a/src/a.py" in diff
    assert "-alpha = 1" in diff
    assert "+alpha = 2" in diff
    assert "b/src/c.py" in diff
    assert "+gamma = 1" in diff
    assert "src/b.py" not in diff  # an unchanged captured file is omitted


def test_format_changes_is_empty_when_nothing_changed(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    a = root / "src" / "a.py"
    a.write_text("alpha = 1\n", encoding="utf-8")

    snapshotter = Snapshotter(root)
    snapshotter.capture(a)
    assert format_changes(root, snapshotter) == ""
