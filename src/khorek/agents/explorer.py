"""Explorer sub-agent: read-only investigation with a free-text answer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from khorek.agents.tools import read_only_tools
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
        self._tools = read_only_tools(project_root, defaults.max_file_bytes)
        self._system_prompt = config.prompt.strip() or default_prompt("explorer")

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


def ask_explorer_tool(explorer: ExplorerAgent, max_handoffs: int) -> Tool:
    """Create the ``ask_explorer`` tool with a per-review hand-off budget."""
    handoffs = 0

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        nonlocal handoffs
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ToolValidationError("argument 'question' must be a non-empty string")
        if handoffs >= max_handoffs:
            return {
                "error": (
                    f"hand-off limit reached ({max_handoffs} per aspect run); "
                    "use read_file, grep, or list_files instead"
                )
            }
        handoffs += 1
        return {"answer": explorer.ask(question)}

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
