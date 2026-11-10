"""Snapshotter: capture once, restore bytes, delete created files."""

from __future__ import annotations

from pathlib import Path

import pytest

from khorek.files import PathEscapeError
from khorek.snapshots import Snapshotter


def test_restore_writes_original_bytes_and_has_changes(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "src" / "a.py"
    target.parent.mkdir()
    target.write_bytes(b"original = 1\n")

    snapshotter = Snapshotter(root)
    snapshotter.capture(target)
    assert snapshotter.has_changes() is False

    target.write_bytes(b"changed = 2\n")
    assert snapshotter.has_changes() is True

    snapshotter.restore()
    assert target.read_bytes() == b"original = 1\n"
    assert snapshotter.has_changes() is False


def test_created_file_is_deleted_on_restore(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    created = root / "src" / "new.py"

    snapshotter = Snapshotter(root)
    snapshotter.capture(created)
    assert snapshotter.has_changes() is False

    created.parent.mkdir(parents=True)
    created.write_text("created\n", encoding="utf-8")
    assert snapshotter.has_changes() is True

    snapshotter.restore()
    assert not created.exists()
    assert snapshotter.has_changes() is False


def test_restore_recreates_missing_parent_directories(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "nested" / "deep" / "a.py"
    target.parent.mkdir(parents=True)
    target.write_text("original\n", encoding="utf-8")

    snapshotter = Snapshotter(root)
    snapshotter.capture(target)
    target.unlink()
    target.parent.rmdir()
    (root / "nested").rmdir()
    assert not target.exists()

    snapshotter.restore()
    assert target.read_text(encoding="utf-8") == "original\n"


def test_capture_once_keeps_the_first_original(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "a.py"
    target.write_text("first\n", encoding="utf-8")

    snapshotter = Snapshotter(root)
    snapshotter.capture(target)
    target.write_text("second\n", encoding="utf-8")
    snapshotter.capture(target)
    target.write_text("third\n", encoding="utf-8")

    snapshotter.restore()
    assert target.read_text(encoding="utf-8") == "first\n"


def test_relative_paths_resolve_against_the_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "a.py"
    target.write_text("original\n", encoding="utf-8")

    snapshotter = Snapshotter(root)
    snapshotter.capture(Path("a.py"))
    target.write_text("changed\n", encoding="utf-8")
    snapshotter.restore()
    assert target.read_text(encoding="utf-8") == "original\n"


def test_paths_outside_the_root_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "a.py"
    target.write_text("outside\n", encoding="utf-8")

    snapshotter = Snapshotter(root)
    with pytest.raises(PathEscapeError, match="escapes the project root"):
        snapshotter.capture(target)


def test_symlinked_parent_escaping_the_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.py").write_text("outside\n", encoding="utf-8")
    link = root / "link"
    link.symlink_to(outside, target_is_directory=True)

    snapshotter = Snapshotter(root)
    with pytest.raises(PathEscapeError, match="escapes the project root"):
        snapshotter.capture(link / "a.py")
