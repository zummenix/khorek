"""Run-lock acquisition, payload, release, and reset unlink behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from khorek.lock import LockHeld, acquire, lock_path_for
from tests.conftest import held_lock


def test_acquire_writes_payload_and_releases(tmp_path: Path) -> None:
    db_path = tmp_path / "khorek.db"
    lock_path = lock_path_for(db_path)
    assert lock_path == tmp_path / "khorek.db.lock"

    with acquire(db_path) as run_lock:
        assert lock_path.is_file()
        payload = lock_path.read_text(encoding="utf-8")
        assert f"pid={run_lock.pid}" in payload
        assert f"started_at={run_lock.started_at}" in payload
        assert run_lock.started_at
        with pytest.raises(LockHeld) as exc_info:
            with acquire(db_path):
                pass
        assert exc_info.value.pid == run_lock.pid

    # The lock file may remain, but the flock is released.
    with acquire(db_path):
        pass


def test_held_lock_reports_holder_from_subprocess(tmp_path: Path) -> None:
    db_path = tmp_path / "khorek.db"
    lock_path = lock_path_for(db_path)
    with held_lock(lock_path):
        with pytest.raises(LockHeld) as exc_info:
            with acquire(db_path):
                pass
    assert exc_info.value.pid == 999999
    assert exc_info.value.started_at == "from-test"
    assert "locked by process 999999" in str(exc_info.value)


def test_unlink_file_while_held(tmp_path: Path) -> None:
    db_path = tmp_path / "khorek.db"
    lock_path = lock_path_for(db_path)
    with acquire(db_path) as run_lock:
        run_lock.unlink_file()
        assert not lock_path.exists()
    assert not lock_path.exists()
