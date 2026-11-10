"""Aspect agent: review one file and report structured issues."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from khorek.agents.explorer import ExplorerAgent, ask_explorer_tool
from khorek.agents.tools import read_only_tools
from khorek.config import AspectAgentConfig, AspectConfig, DefaultsConfig
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


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    """Inline JSON-schema ``$ref``s.

    Pydantic nests named models under ``$defs`` and refers to them with
    ``$ref``.  Small local models (the prototype target) often mishandle that:
    with the nested schema, ``mimo-v2.6`` called ``report_issues`` with ``{}``.
    Inlining keeps the exact field/enum constraints in a flat schema the model
    can follow.
    """
    if isinstance(node, dict):
        if "$ref" in node:
            reference = str(node["$ref"])
            name = reference.rsplit("/", 1)[-1]
            resolved = dict(defs.get(name, {}))
            resolved.update({key: value for key, value in node.items() if key != "$ref"})
            return _inline_refs(resolved, defs)
        return {key: _inline_refs(value, defs) for key, value in node.items() if key != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(item, defs) for item in node]
    return node


def report_issues_parameters() -> dict[str, Any]:
    """Flat JSON schema for the terminal ``report_issues`` tool."""
    schema = IssueReport.model_json_schema()
    definitions = schema.get("$defs", {})
    if not isinstance(definitions, dict):
        definitions = {}
    flattened = _inline_refs(schema, definitions)
    assert isinstance(flattened, dict)
    return flattened


def report_issues_tool() -> Tool:
    """Create the terminal ``report_issues`` tool definition."""
    return Tool(
        name="report_issues",
        description=(
            "Report the issues found for this aspect and file, most important "
            "first. An empty list means the file is clean for this aspect. "
            "This must be your final call."
        ),
        parameters=report_issues_parameters(),
        handler=None,
    )


def _line_count(content: str) -> int:
    return max(1, len(content.splitlines()))


class AspectAgent:
    """Runs one aspect over one file in a bounded tool-calling loop."""

    def __init__(
        self,
        client: LLMClient,
        *,
        project_root: Path,
        config: AspectAgentConfig,
        defaults: DefaultsConfig,
        explorer: ExplorerAgent,
    ) -> None:
        self._client = client
        self._config = config
        self._explorer = explorer
        self._system_prompt = config.prompt.strip() or default_prompt("aspect")
        self._read_tools = read_only_tools(project_root, defaults.max_file_bytes)

    def review(self, *, aspect: AspectConfig, path: str, content: str) -> IssueReport:
        """Review one file for *aspect* and return the validated report."""
        tools = [
            *self._read_tools,
            ask_explorer_tool(self._explorer, self._config.max_handoffs),
            report_issues_tool(),
        ]
        user_prompt = (
            f"Aspect: {aspect.id}\n"
            f"Task: {aspect.prompt}\n\n"
            f"File under review: {path}\n"
            f"Line count: {_line_count(content)}\n\n"
            f"----- BEGIN FILE {path} -----\n"
            f"{content}\n"
            f"----- END FILE {path} -----\n\n"
            "Report your findings for this aspect with report_issues. If there is "
            "nothing to report, call it with an empty list."
        )
        result = self._client.run_agent(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            max_tool_calls=self._config.max_tool_calls,
            force_tool_call=True,
            text_is_final=False,
            terminal_tool="report_issues",
            terminal_schema=IssueReport,
        )
        data = result.data
        if not isinstance(data, IssueReport):
            raise TypeError("report_issues did not return an IssueReport")
        return data


__all__ = [
    "AspectAgent",
    "IssueReport",
    "ReportedIssue",
    "Severity",
    "report_issues_parameters",
    "report_issues_tool",
]
