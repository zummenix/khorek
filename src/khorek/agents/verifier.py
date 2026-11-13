"""Verifier agent: turn failing command output into a Fixer-friendly summary.

The Verifier never decides pass/fail.  The harness runs the configured commands
and derives ``passed`` from the exit codes; the Verifier only distils the first
failure's output into an actionable summary for the Fixer's next round.  Its
structured ``report_verification`` payload is therefore advisory data with no
authority over the workflow result.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from khorek.config import VerifierAgentConfig
from khorek.llm import LLMClient, Tool
from khorek.prompts import default_prompt
from khorek.verify import Failure

VERIFIER_TOOL_NAMES = frozenset({"report_verification"})
_VERIFIER_MAX_TURNS = 2


class VerifierFailure(BaseModel):
    """One failure entry echoed back by the Verifier."""

    model_config = ConfigDict(extra="forbid")

    cmd: str
    exit_code: int | None = None
    output_tail: str


class VerificationReport(BaseModel):
    """Terminal ``report_verification`` payload."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    failures: list[VerifierFailure] = []


def report_verification_tool() -> Tool:
    """The terminal ``report_verification`` tool (no other tools exist)."""
    return Tool(
        name="report_verification",
        description=(
            "Report a short, actionable summary of the first failing verification "
            "command for the fixer agent. Copy the failure entries you were given; "
            "do not invent commands or results. This must be your final call."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "Concise summary for the fixer: what failed, where, and what to "
                        "change next."
                    ),
                },
                "failures": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string"},
                            "exit_code": {
                                "type": ["integer", "null"],
                                "description": "Exit code, or null for a timeout.",
                            },
                            "output_tail": {"type": "string"},
                        },
                        "required": ["cmd", "exit_code", "output_tail"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["summary", "failures"],
            "additionalProperties": False,
        },
        handler=None,
    )


def _user_prompt(failures: Sequence[Failure]) -> str:
    payload = [
        {
            "cmd": failure.cmd,
            "exit_code": failure.exit_code,
            "output_tail": failure.output_tail,
        }
        for failure in failures
    ]
    return (
        "The configured verification command failed, so the fix did not pass.\n"
        "Write a concise, actionable summary for the fixer agent that will try "
        "again. The exit code is authoritative: a non-zero exit or timeout is a "
        "failure regardless of what the output says.\n\n"
        "Failure data (JSON):\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
        "Finish by calling report_verification."
    )


class VerifierAgent:
    """Runs the Verifier's single terminal-tool call."""

    def __init__(self, client: LLMClient, *, config: VerifierAgentConfig) -> None:
        self._client = client
        self._config = config
        self._system_prompt = default_prompt("verifier")
        self._tools = [report_verification_tool()]

    def summarise(self, failures: Sequence[Failure]) -> str:
        """Return the Verifier's summary for *failures*."""
        result = self._client.run_agent(
            system_prompt=self._system_prompt,
            user_prompt=_user_prompt(failures),
            tools=self._tools,
            max_tool_calls=_VERIFIER_MAX_TURNS,
            force_tool_call=True,
            text_is_final=False,
            terminal_tool="report_verification",
            terminal_schema=VerificationReport,
            agent="verifier",
            reasoning_effort=self._config.reasoning_effort,
        )
        data = result.data
        if not isinstance(data, VerificationReport):
            raise TypeError("report_verification did not return a VerificationReport")
        return data.summary.strip()


__all__ = [
    "VERIFIER_TOOL_NAMES",
    "VerificationReport",
    "VerifierAgent",
    "VerifierFailure",
    "report_verification_tool",
]
