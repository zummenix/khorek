"""File enumeration: globs, dirs, ignores, binaries, sizes, symlinks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from khorek.files import (
    PathEscapeError,
    enumerate_files,
    is_binary,
    read_text_file,
    resolve_within_root,
    sha256_file,
    walk_files,
)


def _write(root: Path, relative: str, content: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def test_explicit_file_glob_and_directory_are_deduped_and_sorted(tmp_path: Path) -> None:
    _write(tmp_path, "src/a.py", "a")
    _write(tmp_path, "src/b.py", "b")
    _write(tmp_path, "src/nested/c.py", "c")
    _write(tmp_path, "README.md", "readme")

    result = enumerate_files(
        tmp_path,
        ["src/b.py", "src/*.py", "src/", "README.md"],
        max_bytes=100,
    )
    assert [path.as_posix() for path in result] == [
        "README.md",
        "src/a.py",
        "src/b.py",
        "src/nested/c.py",
    ]


def test_recursive_glob_matches_nested_files(tmp_path: Path) -> None:
    _write(tmp_path, "src/a.py", "a")
    _write(tmp_path, "src/nested/b.py", "b")
    _write(tmp_path, "src/nested/c.txt", "c")

    result = enumerate_files(tmp_path, ["src/**/*.py"], max_bytes=100)
    assert [path.as_posix() for path in result] == ["src/a.py", "src/nested/b.py"]


def test_root_gitignore_applies(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "ignored.py\nbuild/\n")
    _write(tmp_path, "kept.py", "kept")
    _write(tmp_path, "ignored.py", "ignored")
    _write(tmp_path, "build/generated.py", "generated")

    result = enumerate_files(tmp_path, ["**/*.py"], max_bytes=100)
    assert [path.as_posix() for path in result] == ["kept.py"]


def test_fixed_tool_directories_skipped(tmp_path: Path) -> None:
    for directory in (".git", ".venv", "node_modules", "target", ".uv-cache", ".uv-python"):
        _write(tmp_path, f"{directory}/x.py", "x")
    _write(tmp_path, "src/a.py", "a")

    result = enumerate_files(tmp_path, ["."], max_bytes=100)
    assert [path.as_posix() for path in result] == ["src/a.py"]


def test_binary_and_oversized_files_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "binary.py", b"text\x00more")
    _write(tmp_path, "big.py", "x" * 50)
    _write(tmp_path, "small.py", "x" * 10)

    result = enumerate_files(tmp_path, ["."], max_bytes=20)
    assert [path.as_posix() for path in result] == ["small.py"]


def test_extra_ignores_apply(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "a")
    _write(tmp_path, "b.txt", "b")

    result = enumerate_files(tmp_path, ["."], max_bytes=100, extra_ignores=["*.txt"])
    assert [path.as_posix() for path in result] == ["a.py"]


def test_missing_pattern_is_ignored(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "a")
    result = enumerate_files(tmp_path, ["does/not/exist"], max_bytes=100)
    assert result == []


def test_patterns_escaping_root_are_skipped(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("outside", encoding="utf-8")
    root = tmp_path / "root"
    _write(root, "a.py", "a")

    result = enumerate_files(root, ["../outside.py"], max_bytes=100)
    assert result == []


def test_symlink_escaping_root_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, "a.py", "a")
    outside = tmp_path / "outside.py"
    outside.write_text("outside", encoding="utf-8")
    os.symlink(outside, root / "link.py")

    result = enumerate_files(root, ["."], max_bytes=100)
    assert [path.as_posix() for path in result] == ["a.py"]


def test_symlink_cycle_terminates(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, "src/a.py", "a")
    os.symlink(root / "src", root / "src" / "loop")

    result = enumerate_files(root, ["src/"], max_bytes=100)
    assert [path.as_posix() for path in result] == ["src/a.py"]
    assert [path.as_posix() for path in walk_files(root, "src")] == ["src/a.py"]


def test_sha_and_text_helpers(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.py", "hello\n")
    assert sha256_file(path) == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    assert read_text_file(path) == "hello\n"
    assert is_binary(path) is False


def test_binary_helper(tmp_path: Path) -> None:
    path = _write(tmp_path, "blob", b"\x00\x01")
    assert is_binary(path) is True


def test_resolve_within_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, "src/a.py", "a")
    assert resolve_within_root(root, "src/a.py") == (root / "src/a.py").resolve()
    with pytest.raises(PathEscapeError):
        resolve_within_root(root, "../outside")
    with pytest.raises(PathEscapeError):
        resolve_within_root(root, "")


def test_explicit_symlink_escaping_root_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, "a.py", "a")
    outside = tmp_path / "outside.py"
    outside.write_text("outside", encoding="utf-8")
    os.symlink(outside, root / "link.py")

    result = enumerate_files(root, ["link.py"], max_bytes=100)
    assert result == []
