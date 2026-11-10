"""Deterministic verification command runner.

Pass/fail is never model-decided: the configured commands run in order in the
project root and the exit code decides.  The run stops at the first failure and
the captured output tail is handed to the Verifier agent for summarising.

Commands come from the config (``[[verify]]``) only; the model can never choose
or alter them.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from khorek.config import VerifyCommand

FAILURE_OUTPUT_TAIL_CHARS = 4000


@dataclass(frozen=True)
class Failure:
    """One failed verification command."""

    cmd: str
    exit_code: int | None
    output_tail: str
    timed_out: bool = False


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of running the configured commands in order."""

    passed: bool
    failures: tuple[Failure, ...] = ()

    @property
    def failure(self) -> Failure | None:
        """The first (and only) failure, if the run failed."""
        return self.failures[0] if self.failures else None


def output_tail(text: str, limit: int = FAILURE_OUTPUT_TAIL_CHARS) -> str:
    """Keep roughly the last *limit* characters of *text*, trimmed."""
    if len(text) <= limit:
        return text.strip()
    return text[-limit:].lstrip("\n").strip()


def _combine(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    parts: list[str] = []
    for stream in (stdout, stderr):
        if stream is None:
            continue
        if isinstance(stream, bytes):
            parts.append(stream.decode("utf-8", errors="replace"))
        else:
            parts.append(stream)
    return "".join(parts)


def run_commands(project_root: Path, commands: Sequence[VerifyCommand]) -> VerifyResult:
    """Run *commands* in order, stopping at the first failure."""
    for command in commands:
        try:
            completed = subprocess.run(
                command.cmd,
                shell=True,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=command.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            partial = _combine(exc.stdout, exc.stderr)
            note = f"command timed out after {command.timeout:g}s"
            tail = output_tail(f"{partial}\n{note}" if partial else note)
            return VerifyResult(
                passed=False,
                failures=(
                    Failure(cmd=command.cmd, exit_code=None, output_tail=tail, timed_out=True),
                ),
            )
        except OSError as exc:
            note = f"command could not be started: {exc}"
            return VerifyResult(
                passed=False,
                failures=(Failure(cmd=command.cmd, exit_code=None, output_tail=note),),
            )

        if completed.returncode != 0:
            tail = output_tail(_combine(completed.stdout, completed.stderr))
            return VerifyResult(
                passed=False,
                failures=(
                    Failure(
                        cmd=command.cmd,
                        exit_code=completed.returncode,
                        output_tail=tail,
                    ),
                ),
            )
    return VerifyResult(passed=True)


__all__ = [
    "FAILURE_OUTPUT_TAIL_CHARS",
    "Failure",
    "VerifyResult",
    "output_tail",
    "run_commands",
]
