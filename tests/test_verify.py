"""Verification runner: exit codes, first-failure stop, tails, timeouts."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from khorek.config import VerifyCommand
from khorek.verify import FAILURE_OUTPUT_TAIL_CHARS, output_tail, run_commands


def python(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def command(cmd: str, timeout: float = 10.0) -> VerifyCommand:
    return VerifyCommand(cmd=cmd, timeout=timeout)


def test_all_commands_pass(tmp_path: Path) -> None:
    result = run_commands(tmp_path, [command("true"), command("true")])
    assert result.passed is True
    assert result.failures == ()
    assert result.failure is None


def test_first_failure_stops_the_run(tmp_path: Path) -> None:
    marker = tmp_path / "second-ran"
    second = python(f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')")
    result = run_commands(tmp_path, [command("false"), command(second)])

    assert result.passed is False
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.cmd == "false"
    assert failure.exit_code == 1
    assert failure.timed_out is False
    assert not marker.exists()


def test_output_tail_keeps_the_last_characters(tmp_path: Path) -> None:
    code = "import sys; sys.stdout.write('a' * 6000); sys.stderr.write('END-MARK'); sys.exit(3)"
    result = run_commands(tmp_path, [command(python(code))])

    assert result.passed is False
    failure = result.failures[0]
    assert failure.exit_code == 3
    assert failure.output_tail.endswith("END-MARK")
    assert len(failure.output_tail) <= FAILURE_OUTPUT_TAIL_CHARS


def test_timeout_is_a_failure_without_an_exit_code(tmp_path: Path) -> None:
    result = run_commands(tmp_path, [command(python("import time; time.sleep(5)"), timeout=0.05)])

    assert result.passed is False
    failure = result.failures[0]
    assert failure.exit_code is None
    assert failure.timed_out is True
    assert "timed out" in failure.output_tail


def test_unstartable_command_is_reported_cleanly(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    result = run_commands(missing, [command("true")])

    assert result.passed is False
    failure = result.failures[0]
    assert failure.exit_code is None
    assert "could not be started" in failure.output_tail


def test_output_tail_short_text_is_stripped_and_long_text_is_cut() -> None:
    assert output_tail("  hello  ") == "hello"
    long_text = "x" * (FAILURE_OUTPUT_TAIL_CHARS + 100)
    tail = output_tail(long_text)
    assert len(tail) == FAILURE_OUTPUT_TAIL_CHARS
    assert tail == long_text[-FAILURE_OUTPUT_TAIL_CHARS:]
