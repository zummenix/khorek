"""Aspect agent: review one file and report structured issues."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from khorek.agents.explorer import ExplorerAgent, ask_explorer_tool
from khorek.agents.schema import terminal_tool
from khorek.agents.tools import read_only_tools
from khorek.config import AspectAgentConfig, AspectConfig, DefaultsConfig
from khorek.content import format_file_lines
from khorek.db import Issue
from khorek.llm import LLMClient, Tool
from khorek.prompts import default_prompt

Severity = Literal["low", "medium", "high"]
_VALID_SEVERITIES = frozenset({"low", "medium", "high"})


class ReportedIssue(BaseModel):
    """One issue as reported by the model."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(description="What is wrong, concretely.")
    line_start: int = Field(description="1-based first line of the issue.")
    line_end: int = Field(description="1-based last line of the issue.")
    severity: Severity = Field(description="Impact: low, medium, or high.")

    @field_validator("severity", mode="before")
    @classmethod
    def _repair_severity(cls, value: object) -> object:
        # Invalid severities are deterministically repaired to "low" rather
        # than rejecting the whole report.
        if isinstance(value, str) and value in _VALID_SEVERITIES:
            return value
        return "low"


class IssueReport(BaseModel):
    """Terminal ``report_issues`` payload, in the model's impact order."""

    model_config = ConfigDict(extra="forbid")

    issues: list[ReportedIssue]


RevalidationStatus = Literal["valid", "gone", "uncertain"]


class RevalidationVerdict(BaseModel):
    """One verdict on a single open issue, keyed by its database id.

    The id is the ``issues`` table primary key, so a verdict maps straight to a
    row: an unknown id, a missing verdict, or a verdict that cannot be acted on
    leaves that issue untouched (silence is never loss).
    """

    model_config = ConfigDict(extra="forbid")

    issue_id: int = Field(description="The database id of the issue this verdict is for.")
    status: RevalidationStatus = Field(
        description="valid (survives), gone (resolved by the fix), or uncertain."
    )
    line_start: int | None = Field(
        default=None,
        description="Refreshed 1-based first line; required when status is valid.",
    )
    line_end: int | None = Field(
        default=None,
        description="Refreshed 1-based last line; required when status is valid.",
    )
    description: str | None = Field(
        default=None,
        description="Optional refreshed description, for valid when the wording changed.",
    )
    reason: str | None = Field(
        default=None,
        description="Required when status is gone: where it went or why it is resolved.",
    )


class RevalidationReport(BaseModel):
    """Terminal ``report_revalidation`` payload, one verdict per open issue."""

    model_config = ConfigDict(extra="forbid")

    results: list[RevalidationVerdict]


def report_issues_tool() -> Tool:
    """Create the terminal ``report_issues`` tool definition."""
    return terminal_tool(
        "report_issues",
        (
            "Report the issues found for this aspect and file, most important "
            "first. An empty list means the file is clean for this aspect. "
            "This must be your final call."
        ),
        IssueReport,
    )


def report_revalidation_tool() -> Tool:
    """Create the terminal ``report_revalidation`` tool definition."""
    return terminal_tool(
        "report_revalidation",
        (
            "Report a verdict for each open issue you were given, keyed by its "
            "database id. Use status 'valid' (with refreshed line_start/line_end) "
            "when the issue survives, 'gone' (with a concrete reason) when the "
            "fix resolved it, or 'uncertain' when you cannot tell. This must be "
            "your final call."
        ),
        RevalidationReport,
    )


class AspectAgent:
    """Runs one aspect over one file in a bounded tool-calling loop."""

    def __init__(
        self,
        client: LLMClient,
        *,
        project_root: Path,
        config: AspectAgentConfig,
        defaults: DefaultsConfig,
        explorer: ExplorerAgent | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._explorer = explorer
        self._system_prompt = default_prompt("aspect")
        self._revalidate_prompt = default_prompt("aspect_revalidate")
        self._read_tools = read_only_tools(
            project_root, defaults.max_file_bytes, defaults.max_tool_output_bytes
        )
        self._max_tool_output_bytes = defaults.max_tool_output_bytes

    def review(self, *, aspect: AspectConfig, path: str, content: str) -> IssueReport:
        """Review one file for *aspect* and return the validated report."""
        tools: list[Tool] = list(self._read_tools)
        # Revalidation never hands off to the explorer; only a full review does.
        if self._explorer is not None:
            tools.append(
                ask_explorer_tool(
                    self._explorer, self._config.max_handoffs, self._max_tool_output_bytes
                )
            )
        tools.append(report_issues_tool())
        user_prompt = (
            f"Aspect: {aspect.id}\n"
            f"Task: {aspect.prompt}\n\n"
            f"{format_file_lines(path, content, offset=None, limit=None)}\n\n"
            "Report your findings for this aspect with report_issues. If there is "
            "nothing to report, call it with an empty list for the `issues` argument."
        )
        return self._client.run_agent(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            max_tool_calls=self._config.max_tool_calls,
            force_tool_call=True,
            text_is_final=False,
            terminal_tool="report_issues",
            terminal_schema=IssueReport,
            agent=f"aspect:{aspect.id}:{path}",
            reasoning_effort=self._config.reasoning_effort,
        ).data_of(IssueReport)

    def revalidate(
        self,
        *,
        aspect: AspectConfig,
        path: str,
        content: str,
        issues: Sequence[Issue],
    ) -> RevalidationReport:
        """Re-judge the open *issues* of *aspect* against the current *content*.

        Runs one bounded, read-only agent loop with the ``report_revalidation``
        terminal tool and returns the validated report.  The model is given each
        open issue labelled by its database id, and each verdict maps back to that
        id.  The loop is traced as ``aspect:<id>:<path>:revalidate``.
        """
        tools = [
            *self._read_tools,
            report_revalidation_tool(),
        ]
        user_prompt = _revalidate_user_prompt(aspect, path, content, issues)
        return self._client.run_agent(
            system_prompt=self._revalidate_prompt,
            user_prompt=user_prompt,
            tools=tools,
            max_tool_calls=self._config.max_tool_calls,
            force_tool_call=True,
            text_is_final=False,
            terminal_tool="report_revalidation",
            terminal_schema=RevalidationReport,
            agent=f"aspect:{aspect.id}:{path}:revalidate",
            reasoning_effort=self._config.reasoning_effort,
        ).data_of(RevalidationReport)


def _revalidate_user_prompt(
    aspect: AspectConfig,
    path: str,
    content: str,
    issues: Sequence[Issue],
) -> str:
    """Build the revalidation user prompt: task, current file, and open issues."""
    count = max(1, len(content.splitlines()))
    issue_lines = [
        f"#{issue.id}: lines {issue.line_start}-{issue.line_end}: {issue.description.strip()}"
        for issue in issues
    ]
    return (
        f"Aspect: {aspect.id}\n"
        f"Task: {aspect.prompt}\n\n"
        f"File: {path}\n"
        f"Current line count: {count}\n\n"
        "Open issues for this aspect and file (labelled by database id):\n"
        + "\n".join(issue_lines)
        + "\n\n"
        "Current file contents:\n"
        f"{format_file_lines(path, content, offset=None, limit=None)}\n\n"
        "For every issue above, report a verdict with report_revalidation, keyed "
        "by the issue's id. A 'gone' verdict needs a concrete reason; an issue you "
        "are unsure about should be 'uncertain', never 'gone'."
    )


__all__ = [
    "AspectAgent",
    "IssueReport",
    "ReportedIssue",
    "RevalidationReport",
    "RevalidationStatus",
    "RevalidationVerdict",
    "Severity",
    "report_issues_tool",
    "report_revalidation_tool",
]
