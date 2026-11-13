"""Reviewer agent: a model-based quality gate on a verified fix.

The deterministic verification gate (the configured ``[[verify]]`` commands)
already passed, so the Reviewer's job is to judge whether the diff *correctly
and minimally resolves the stated issue* — not whether the tests pass.  It
reads the unified diff of every file the Fixer touched and, if it rejects, its
reason is handed back to the Fixer as the next instruction.  The Reviewer is a
read-only, bounded tool-calling loop (``read_file``/``grep``/``list_files``
plus its terminal ``report_review``); it never writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from khorek.agents.tools import read_only_tools
from khorek.config import DefaultsConfig, ReviewerAgentConfig
from khorek.db import Issue
from khorek.llm import LLMClient, Tool
from khorek.prompts import default_prompt

REVIEWER_TOOL_NAMES = frozenset({"read_file", "grep", "list_files", "report_review"})


class ReviewReport(BaseModel):
    """Terminal ``report_review`` payload."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    summary: str


@dataclass(frozen=True)
class ReviewOutcome:
    """What the Reviewer reported for one attempt."""

    approved: bool
    summary: str


def report_review_tool() -> Tool:
    """The terminal ``report_review`` tool."""
    return Tool(
        name="report_review",
        description=(
            "Report your verdict on the fixer's change. Call with approved=true "
            "only when the diff correctly and minimally resolves the stated "
            "issue; with approved=false when it does not actually address the "
            "issue, is not minimal, or introduces a clear defect. `summary` is "
            "required in both cases: a short confirmation on approval, or a "
            "concrete, actionable reason on rejection that is handed straight "
            "back to the fixer. This must be your final call."
        ),
        parameters=ReviewReport.model_json_schema(),
        handler=None,
    )


def _user_prompt(issue: Issue, diff: str, verify_status: list[str]) -> str:
    verify_lines = "\n".join(f"- {cmd} : passed" for cmd in verify_status)
    return (
        f"Issue #{issue.id} (aspect: {issue.aspect_id}, severity: {issue.severity})\n"
        f"File: {issue.path}\n"
        "\n"
        "Issue description:\n"
        f"{issue.description.strip()}\n"
        "\n"
        "Verification status (all configured commands passed):\n"
        f"{verify_lines}\n"
        "\n"
        "Diff of the fixer's changes (line numbers refer to the post-fix file):\n"
        f"{diff}\n"
        "\n"
        "Review this change. Call report_review(approved, summary)."
    )


class ReviewerAgent:
    """Runs one bounded, read-only Reviewer round for one verified fix."""

    def __init__(
        self,
        client: LLMClient,
        *,
        project_root: Path,
        config: ReviewerAgentConfig,
        defaults: DefaultsConfig,
    ) -> None:
        self._client = client
        self._config = config
        self._system_prompt = config.prompt.strip() or default_prompt("reviewer")
        self._tools = [
            *read_only_tools(project_root, defaults.max_file_bytes),
            report_review_tool(),
        ]

    def review(self, issue: Issue, diff: str, verify_status: list[str]) -> ReviewOutcome:
        """Run one Reviewer round and return its verdict."""
        result = self._client.run_agent(
            system_prompt=self._system_prompt,
            user_prompt=_user_prompt(issue, diff, verify_status),
            tools=self._tools,
            max_tool_calls=self._config.max_tool_calls,
            force_tool_call=True,
            text_is_final=False,
            terminal_tool="report_review",
            terminal_schema=ReviewReport,
            agent=f"reviewer:#{issue.id}:{issue.path}",
            reasoning_effort=self._config.reasoning_effort,
        )
        data = result.data
        if not isinstance(data, ReviewReport):
            raise TypeError("report_review did not return a ReviewReport")
        return ReviewOutcome(approved=data.approved, summary=data.summary.strip())


__all__ = [
    "REVIEWER_TOOL_NAMES",
    "ReviewOutcome",
    "ReviewReport",
    "ReviewerAgent",
    "report_review_tool",
]
