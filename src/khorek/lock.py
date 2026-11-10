"""One-run-per-database locking via POSIX ``fcntl.flock``.

The kernel releases the lock when the process dies, so a crashed run never
leaves a stale lock.  The lock file carries the holder's PID and start time for
a clear diagnostic when a second run is refused.
"""

from __future__ import annotations

import errno
import fcntl
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO


class LockError(Exception):
    """Fatal problem acquiring or inspecting the run lock."""


class LockHeld(LockError):
    """Another live process already holds the database lock."""

    def __init__(self, pid: int | None, started_at: str | None, lock_path: Path) -> None:
        self.pid = pid
        self.started_at = started_at
        self.lock_path = lock_path
        holder = "another process"
        if pid is not None:
            holder = f"process {pid}"
            if started_at:
                holder += f" (started {started_at})"
        super().__init__(f"database is locked by {holder}: {lock_path}")


def lock_path_for(db_path: Path) -> Path:
    """Return ``<db>.lock`` for a database path."""
    return db_path.with_name(db_path.name + ".lock")


@dataclass
class RunLock:
    """A held lock; release it with :meth:`release` or as a context manager."""

    path: Path
    _handle: TextIO
    pid: int
    started_at: str

    def unlink_file(self) -> None:
        """Unlink the lock file while the lock is still held (used by reset)."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            raise LockError(f"cannot remove lock file {self.path}: {exc}") from exc

    def release(self) -> None:
        """Release the flock and close the lock file."""
        handle = self._handle
        if handle.closed:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            # Closing releases the lock anyway.
            pass
        finally:
            handle.close()

    def __enter__(self) -> RunLock:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.release()


def _read_payload(handle: TextIO) -> tuple[int | None, str | None]:
    pid: int | None = None
    started_at: str | None = None
    try:
        handle.seek(0)
        for line in handle.read().splitlines():
            key, separator, value = line.partition("=")
            if not separator:
                continue
            if key.strip() == "pid":
                try:
                    pid = int(value.strip())
                except ValueError:
                    pid = None
            elif key.strip() == "started_at":
                started_at = value.strip() or None
    except (OSError, ValueError):
        pass
    return pid, started_at


@contextmanager
def acquire(db_path: Path) -> Generator[RunLock, None, None]:
    """Take a non-blocking exclusive lock on ``<db>.lock``.

    Raises :class:`LockHeld` if another process holds it, or :class:`LockError`
    if the lock file cannot be created.
    """
    path = lock_path_for(db_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+", encoding="utf-8")
    except OSError as exc:
        raise LockError(f"cannot open lock file {path}: {exc}") from exc

    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                pid, started_at = _read_payload(handle)
                handle.close()
                raise LockHeld(pid, started_at, path) from None
            handle.close()
            raise LockError(f"cannot lock {path}: {exc}") from exc

        pid = os.getpid()
        started_at = datetime.now(UTC).isoformat(timespec="seconds")
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={pid}\nstarted_at={started_at}\n")
            handle.flush()
            os.fsync(handle.fileno())
        except OSError as exc:
            handle.close()
            raise LockError(f"cannot write lock file {path}: {exc}") from exc

        run_lock = RunLock(path=path, _handle=handle, pid=pid, started_at=started_at)
        try:
            yield run_lock
        finally:
            run_lock.release()
    except BaseException:
        if not handle.closed:
            handle.close()
        raise


__all__ = ["LockError", "LockHeld", "RunLock", "acquire", "lock_path_for"]
