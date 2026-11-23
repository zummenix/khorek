"""In-memory snapshots of the files one fix attempt may touch.

The workflow creates one :class:`Snapshotter` per issue attempt, captures the
issue's file before the Fixer runs, and the Fixer's mutating tools capture any
additional file before their first write.  A failed attempt calls
:meth:`Snapshotter.restore`, which writes the original bytes back and deletes
paths the Fixer created.

Only paths inside the project root may be captured; a path whose real location
escapes the root is rejected with :class:`PathEscapeError` and is never read or
written.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from khorek.files import PathEscapeError


@dataclass(frozen=True)
class Snapshot:
    """The original state of one captured path."""

    path: Path
    existed: bool
    original: bytes | None


FileChange = Snapshot  # keep the public alias


class Snapshotter:
    """Captures and restores originals for a single issue attempt."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(os.path.realpath(project_root))
        self._snapshots: list[Snapshot] = []
        self._by_path: dict[Path, Snapshot] = {}

    def _canonical(self, path: Path) -> Path:
        """Resolve *path* to a real absolute path inside the project root."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        resolved = Path(os.path.realpath(candidate))
        try:
            resolved.relative_to(self.project_root)
        except ValueError as exc:
            raise PathEscapeError(f"path escapes the project root: {path}") from exc
        return resolved

    def capture(self, path: Path) -> None:
        """Record the original bytes (or absence) of *path*, once per path."""
        canonical = self._canonical(path)
        if canonical in self._by_path:
            return
        if canonical.exists():
            snapshot = Snapshot(path=canonical, existed=True, original=canonical.read_bytes())
        else:
            snapshot = Snapshot(path=canonical, existed=False, original=None)
        self._by_path[canonical] = snapshot
        self._snapshots.append(snapshot)

    def restore(self) -> None:
        """Write originals back and delete created paths, in reverse order."""
        for snapshot in reversed(self._snapshots):
            if snapshot.existed:
                assert snapshot.original is not None
                snapshot.path.parent.mkdir(parents=True, exist_ok=True)
                snapshot.path.write_bytes(snapshot.original)
            else:
                try:
                    snapshot.path.unlink()
                except FileNotFoundError:
                    continue

    def has_changes(self) -> bool:
        """Return ``True`` when any captured path differs from its original."""
        return bool(self.changes())

    def changes(self) -> list[Snapshot]:
        """Return the captured paths whose current state differs from the original.

        A captured path is *changed* when it existed and its bytes now differ
        (an unreadable path counts as changed), or when it did not exist and the
        path now exists or is a symlink.
        """
        return [s for s in self._snapshots if self._is_changed(s)]

    @staticmethod
    def _is_changed(snapshot: Snapshot) -> bool:
        if snapshot.existed:
            assert snapshot.original is not None
            try:
                current = snapshot.path.read_bytes()
            except OSError:
                return True
            return current != snapshot.original
        return snapshot.path.exists() or snapshot.path.is_symlink()


__all__ = ["FileChange", "Snapshotter"]
