"""Explorer sub-agent: read-only investigation with a free-text answer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from khorek.agents.tools import fit_result, read_only_tools, serialized_size
from khorek.config import DefaultsConfig, ExplorerAgentConfig
from khorek.llm import LLMClient, Tool, ToolValidationError
from khorek.prompts import default_prompt


class ExplorerAgent:
    """Bounded read-only agent used through the Aspect agent's hand-off tool."""

    def __init__(
        self,
        client: LLMClient,
        *,
        project_root: Path,
        config: ExplorerAgentConfig,
        defaults: DefaultsConfig,
    ) -> None:
        self._client = client
        self._config = config
        self._tools = read_only_tools(
            project_root, defaults.max_file_bytes, defaults.max_tool_output_bytes
        )
        self._system_prompt = default_prompt("explorer")

    def ask(self, question: str) -> str:
        """Ask the explorer a question and return its plain-text answer."""
        if not question.strip():
            raise ToolValidationError("ask_explorer requires a non-empty question")
        result = self._client.run_agent(
            system_prompt=self._system_prompt,
            user_prompt=question,
            tools=self._tools,
            max_tool_calls=self._config.max_tool_calls,
            force_tool_call=False,
            text_is_final=True,
            agent="explorer",
            reasoning_effort=self._config.reasoning_effort,
        )
        return result.text


def _fit_answer(answer: str, max_bytes: int) -> dict[str, Any]:
    """Fit the whole ``ask_explorer`` result, JSON wrapper included, to *max_bytes*.

    The answer is cut on a character boundary.  When the fixed wrapper alone
    exceeds the cap, the answer is empty and the wrapper is the floor.
    """

    def build(text: str, truncated: bool) -> dict[str, Any]:
        return {"answer": text, "truncated": truncated}

    if serialized_size(build(answer, False)) <= max_bytes:
        return build(answer, False)

    low, high, best = 0, len(answer), 0
    while low <= high:
        middle = (low + high) // 2
        if serialized_size(build(answer[:middle], True)) <= max_bytes:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return build(answer[:best], True)


def _handoff_limit_result(max_handoffs: int, max_bytes: int) -> dict[str, Any]:
    """The hand-off-limit error, shrunk to fit *max_bytes*.

    The full message is preferred; when it does not fit, the actionable hint
    and then the count are dropped, and ``truncated`` marks the cut.
    """
    return fit_result(
        {
            "error": (
                f"hand-off limit reached ({max_handoffs} per aspect run); "
                "use read_file, grep, or list_files instead"
            ),
            "truncated": False,
        },
        [
            {
                "error": f"hand-off limit reached ({max_handoffs} per aspect run)",
                "truncated": True,
            },
            {"error": "hand-off limit reached", "truncated": True},
            {"truncated": True},
        ],
        max_bytes,
    )


def ask_explorer_tool(
    explorer: ExplorerAgent, max_handoffs: int, max_tool_output_bytes: int
) -> Tool:
    """Create the ``ask_explorer`` tool with a per-aspect-run hand-off budget."""
    handoffs = 0

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        nonlocal handoffs
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ToolValidationError("argument 'question' must be a non-empty string")
        if handoffs >= max_handoffs:
            return _handoff_limit_result(max_handoffs, max_tool_output_bytes)
        handoffs += 1
        return _fit_answer(explorer.ask(question), max_tool_output_bytes)

    return Tool(
        name="ask_explorer",
        description=(
            "Ask the read-only explorer sub-agent a focused question about the "
            "project (for example where a function is defined or used). "
            f"Limited to {max_handoffs} hand-offs per aspect run."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "A focused question about the project.",
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        handler=handler,
    )


__all__ = ["ExplorerAgent", "ask_explorer_tool"]
