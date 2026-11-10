"""File enumeration, hashing, and safe reads for khorek.

Enumeration follows the plan: ``files`` entries may be globs, directories, or
explicit paths; root ``.gitignore`` patterns are honoured with ``pathspec``;
the usual build/tooling directories are always skipped; binary and oversized
files are dropped; results are sorted, deduplicated, project-root-relative
paths.  Symlinks may not escape the project root and directory walks are
cycle-safe.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pathspec import PathSpec

# Directories that are never enumerated, regardless of .gitignore.
DEFAULT_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "node_modules",
        "target",
        ".uv-cache",
        ".uv-python",
        "__pycache__",
    }
)

_CHUNK_SIZE = 1024 * 1024
_BINARY_PROBE_BYTES = 8192


class PathEscapeError(ValueError):
    """Raised when a requested path resolves outside the project root."""


@dataclass(frozen=True)
class _IgnoreRules:
    """Combined root .gitignore and extra ignore patterns."""

    specs: tuple[PathSpec, ...]

    def ignored(self, relative: str) -> bool:
        return any(spec.match_file(relative) for spec in self.specs)

    def ignored_dir(self, relative: str) -> bool:
        return self.ignored(relative.rstrip("/") + "/")


def _load_ignore_rules(root: Path, extra_ignores: Iterable[str]) -> _IgnoreRules:
    specs: list[PathSpec] = []
    gitignore = root / ".gitignore"
    if gitignore.is_file():
        try:
            text = gitignore.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if text:
            specs.append(PathSpec.from_lines("gitignore", text.splitlines()))
    extra = list(extra_ignores)
    if extra:
        specs.append(PathSpec.from_lines("gitignore", extra))
    return _IgnoreRules(tuple(specs))


def _is_inside(path: Path, root_real: Path) -> bool:
    try:
        path.relative_to(root_real)
    except ValueError:
        return False
    return True


def _relative_posix(path: Path, root_real: Path) -> str | None:
    try:
        return path.relative_to(root_real).as_posix()
    except ValueError:
        return None


def _follow_symlink(path: Path, root_real: Path) -> Path | None:
    """Resolve a symlink, returning ``None`` if it is broken or escapes root."""
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if not _is_inside(resolved, root_real):
        return None
    return resolved


def _walk_dir(
    directory: Path,
    root_real: Path,
    rules: _IgnoreRules,
    visited: set[tuple[int, int]],
) -> Iterator[tuple[Path, str]]:
    """Yield ``(absolute_path, relative_posix)`` for files under *directory*."""
    try:
        stat = directory.stat()
    except OSError:
        return
    key = (stat.st_dev, stat.st_ino)
    if key in visited:
        return
    visited.add(key)

    try:
        entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
    except OSError:
        return

    for entry in entries:
        resolved = entry
        if entry.is_symlink():
            followed = _follow_symlink(entry, root_real)
            if followed is None:
                continue
            resolved = followed

        relative = _relative_posix(resolved, root_real)
        if relative is None or relative == "":
            continue

        if resolved.is_dir():
            if resolved.name in DEFAULT_IGNORED_DIRS:
                continue
            if rules.ignored_dir(relative):
                continue
            yield from _walk_dir(resolved, root_real, rules, visited)
        elif resolved.is_file():
            if any(part in DEFAULT_IGNORED_DIRS for part in Path(relative).parts):
                continue
            if rules.ignored(relative):
                continue
            yield resolved, relative


def walk_files(root: Path, start: str | Path = ".") -> list[Path]:
    """List project files under *start*, relative to *root*, sorted.

    This is the tool-facing listing primitive: it applies ignore rules and the
    fixed directory skips but leaves binary/size filtering to the caller.
    """
    root_real = Path(os.path.realpath(root))
    rules = _load_ignore_rules(root_real, ())
    raw_start = Path(start).expanduser()
    if raw_start.is_absolute():
        start_path = raw_start.resolve()
    else:
        start_path = (root_real / raw_start).resolve()
    if not _is_inside(start_path, root_real):
        return []
    if start_path.is_file():
        relative = _relative_posix(start_path, root_real)
        return [Path(relative)] if relative is not None else []

    visited: set[tuple[int, int]] = set()
    found: dict[str, Path] = {}
    for _absolute, relative in _walk_dir(start_path, root_real, rules, visited):
        found.setdefault(relative, Path(relative))
    return sorted(found.values(), key=lambda path: path.as_posix())


def _normalize_pattern(root_real: Path, raw: str | Path) -> tuple[str, Path] | None:
    """Normalize a configured pattern to root-relative form.

    Returns ``(pattern, root / pattern)`` or ``None`` when the pattern escapes
    the project root.
    """
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        try:
            resolved = Path(os.path.realpath(candidate))
        except OSError:
            return None
        if not _is_inside(resolved, root_real):
            return None
        candidate = resolved.relative_to(root_real)
    normalized = Path(os.path.normpath(candidate.as_posix()))
    if normalized.is_absolute() or ".." in normalized.parts:
        return None
    pattern = normalized.as_posix()
    if pattern == "":
        pattern = "."
    return pattern, root_real / normalized


def _is_binary_path(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            probe = handle.read(_BINARY_PROBE_BYTES)
    except OSError:
        return True
    return b"\x00" in probe


def _is_oversized(path: Path, max_bytes: int) -> bool:
    try:
        return path.stat().st_size > max_bytes
    except OSError:
        return True


def enumerate_files(
    root: Path,
    patterns: Sequence[str | Path],
    *,
    max_bytes: int,
    extra_ignores: Iterable[str] = (),
) -> list[Path]:
    """Enumerate matching project files as sorted, relative paths."""
    root_real = Path(os.path.realpath(root))
    rules = _load_ignore_rules(root_real, extra_ignores)
    candidates: dict[str, Path] = {}
    glob_patterns: list[str] = []

    def add(path: Path, relative: str) -> None:
        if any(part in DEFAULT_IGNORED_DIRS for part in Path(relative).parts):
            return
        if rules.ignored(relative):
            return
        if _is_binary_path(path) or _is_oversized(path, max_bytes):
            return
        candidates.setdefault(relative, Path(relative))

    for raw in patterns:
        normalized = _normalize_pattern(root_real, raw)
        if normalized is None:
            continue
        pattern, target = normalized
        if target.is_dir():
            visited: set[tuple[int, int]] = set()
            for absolute, relative in _walk_dir(target, root_real, rules, visited):
                add(absolute, relative)
        elif target.is_file():
            if target.is_symlink():
                followed = _follow_symlink(target, root_real)
                if followed is None:
                    continue
                target = followed
            relative = _relative_posix(target, root_real)
            if relative is not None:
                add(target, relative)
        else:
            glob_patterns.append(pattern)

    if glob_patterns:
        spec = PathSpec.from_lines("gitignore", glob_patterns)
        visited = set()
        for absolute, relative in _walk_dir(root_real, root_real, rules, visited):
            if spec.match_file(relative):
                add(absolute, relative)

    return [candidates[key] for key in sorted(candidates)]


def resolve_within_root(root: Path, raw: str) -> Path:
    """Resolve *raw* under *root*, rejecting paths that escape it."""
    if not raw or not raw.strip():
        raise PathEscapeError("path must not be empty")
    root_real = Path(os.path.realpath(root))
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root_real / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise PathEscapeError(f"path does not exist: {raw}") from exc
    if not _is_inside(resolved, root_real):
        raise PathEscapeError(f"path escapes the project root: {raw}")
    return resolved


def resolve_for_write(root: Path, raw: str) -> Path:
    """Resolve a write target inside *root*, allowing files that do not exist.

    Unlike :func:`resolve_within_root`, the final component may be missing
    (``write_file`` creates files).  Escapes via absolute paths, ``..``, or a
    symlinked parent pointing outside the project root are rejected; a symlink
    inside the root is followed only when its target stays inside.
    """
    if not raw or not raw.strip():
        raise PathEscapeError("path must not be empty")
    root_real = Path(os.path.realpath(root))
    candidate = Path(raw)
    if candidate.is_absolute():
        target = Path(os.path.realpath(candidate))
        if not _is_inside(target, root_real):
            raise PathEscapeError(f"path escapes the project root: {raw}")
        return target

    normalized = Path(os.path.normpath(candidate.as_posix()))
    if ".." in normalized.parts:
        raise PathEscapeError(f"path escapes the project root: {raw}")
    target = Path(os.path.realpath(root_real / normalized))
    if not _is_inside(target, root_real):
        raise PathEscapeError(f"path escapes the project root: {raw}")
    return target


def is_binary(path: Path) -> bool:
    """Return ``True`` when *path* looks binary (NUL byte in the first 8 KiB)."""
    return _is_binary_path(path)


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def read_text_file(path: Path) -> str:
    """Read a text file as UTF-8, replacing undecodable bytes."""
    return path.read_bytes().decode("utf-8", errors="replace")


__all__ = [
    "DEFAULT_IGNORED_DIRS",
    "PathEscapeError",
    "enumerate_files",
    "is_binary",
    "read_text_file",
    "resolve_for_write",
    "resolve_within_root",
    "sha256_file",
    "walk_files",
]
