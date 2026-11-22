"""LLM client: preflight, tool loop, budgets, repair, nudge, truncation."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from khorek.agents.tools import read_only_tools
from khorek.config import LLMConfig
from khorek.llm import AgentError, LLMClient, ModelUnavailable, Tool, ToolValidationError
from tests.conftest import FakeModel, text_message, tool_message


class Report(BaseModel):
    value: str


def _config() -> LLMConfig:
    return LLMConfig(
        base_url="http://model.test/v1",
        model="fake-model",
        reasoning_effort="none",
        temperature=0.0,
        call_timeout=5.0,
    )


def _client(
    fake: FakeModel,
    *,
    max_prompt_bytes: int = 400_000,
) -> LLMClient:
    return LLMClient(
        _config(),
        transport=fake.transport,
        max_prompt_bytes=max_prompt_bytes,
    )


def _echo(arguments: dict[str, Any]) -> dict[str, Any]:
    text = arguments.get("text")
    if not isinstance(text, str):
        raise ToolValidationError("text must be a string")
    return {"echo": text}


def _echo_tool() -> Tool:
    return Tool(
        name="echo",
        description="Echo text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        handler=_echo,
    )


def _report_tool() -> Tool:
    return Tool(
        name="report",
        description="Finish.",
        parameters=Report.model_json_schema(),
        handler=None,
    )


def _run(
    client: LLMClient,
    *,
    force: bool = True,
    text_final: bool = False,
    max_calls: int = 10,
    tools: list[Tool] | None = None,
    terminal: str | None = "report",
) -> Any:
    tool_list = tools if tools is not None else [_echo_tool(), _report_tool()]
    return client.run_agent(
        system_prompt="system",
        user_prompt="user",
        tools=tool_list,
        max_tool_calls=max_calls,
        force_tool_call=force,
        text_is_final=text_final,
        terminal_tool=terminal,
        terminal_schema=Report if terminal else None,
    )


def test_preflight_connection_error_is_fatal() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = LLMClient(_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable, match="unreachable"):
        client.preflight()


def test_preflight_timeout_is_fatal() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    client = LLMClient(_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable, match="timed out"):
        client.preflight()


def test_preflight_non_2xx_only_warns() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(404))
    client = LLMClient(_config(), transport=transport)
    with pytest.warns(RuntimeWarning, match="preflight"):
        client.preflight()


def test_loop_executes_tools_then_returns_terminal_result() -> None:
    fake = FakeModel(
        [
            tool_message("echo", {"text": "hi"}),
            tool_message("report", {"value": "done"}),
        ]
    )
    result = _run(_client(fake))
    assert isinstance(result.data, Report)
    assert result.data.value == "done"
    assert fake.chat_calls == 2
    assert "hi" in fake.tool_results()[0]
    # The required terminal tool is forced via tool_choice.
    assert fake.chat_requests[0]["tool_choice"] == "required"


def test_repeat_call_twice_in_a_row_aborts_loop() -> None:
    fake = FakeModel(
        [
            tool_message("echo", {"text": "same"}),
            tool_message("echo", {"text": "same"}),
        ]
    )
    with pytest.raises(AgentError) as exc_info:
        _run(_client(fake))
    assert exc_info.value.code == "loop"
    assert fake.chat_calls == 2


def test_repeat_call_three_times_overall_aborts_loop() -> None:
    fake = FakeModel(
        [
            tool_message("echo", {"text": "a"}),
            tool_message("echo", {"text": "b"}),
            tool_message("echo", {"text": "a"}),
            tool_message("echo", {"text": "a"}),
        ]
    )
    with pytest.raises(AgentError) as exc_info:
        _run(_client(fake))
    assert exc_info.value.code == "loop"
    assert fake.chat_calls == 4


def test_tool_call_budget_aborts() -> None:
    fake = FakeModel(
        [
            tool_message("echo", {"text": "one"}),
            tool_message("echo", {"text": "two"}),
        ]
    )
    with pytest.raises(AgentError) as exc_info:
        _run(_client(fake), max_calls=1)
    assert exc_info.value.code == "budget"
    assert fake.chat_calls == 2


def test_text_reply_gets_nudge_then_aborts_and_keeps_text() -> None:
    fake = FakeModel([text_message("first text"), text_message("second text")])
    with pytest.raises(AgentError) as exc_info:
        _run(_client(fake))
    error = exc_info.value
    assert error.code == "no_tool_call"
    assert error.last_text == "second text"
    assert "second text" in str(error)
    assert fake.chat_calls == 2
    reminder = fake.chat_requests[1]["messages"][-1]
    assert reminder["role"] == "user"
    assert "REMINDER" in reminder["content"]


def test_prompt_budget_aborts_before_sending() -> None:
    fake = FakeModel([tool_message("report", {"value": "never"})])
    with pytest.raises(AgentError) as exc_info:
        _run(_client(fake, max_prompt_bytes=10))
    assert exc_info.value.code == "prompt_budget"
    assert fake.chat_calls == 0


def test_malformed_arguments_get_one_repair() -> None:
    fake = FakeModel(
        [
            tool_message("echo", "{not json"),
            tool_message("report", {"value": "recovered"}),
        ]
    )
    result = _run(_client(fake))
    assert isinstance(result.data, Report)
    assert result.data.value == "recovered"
    assert "invalid_arguments" in fake.tool_results()[0]


def test_second_malformed_call_aborts() -> None:
    fake = FakeModel(
        [
            tool_message("echo", "{not json"),
            tool_message("echo", "{still not json"),
        ]
    )
    with pytest.raises(AgentError) as exc_info:
        _run(_client(fake))
    assert exc_info.value.code == "schema"
    assert fake.chat_calls == 2


def test_handler_validation_error_gets_one_repair() -> None:
    fake = FakeModel(
        [
            tool_message("echo", {"wrong": "shape"}),
            tool_message("report", {"value": "ok"}),
        ]
    )
    result = _run(_client(fake))
    assert isinstance(result.data, Report)
    assert "text must be a string" in fake.tool_results()[0]


def test_non_dict_tool_result_is_a_schema_error() -> None:
    def raw(_arguments: dict[str, Any]) -> Any:
        return "plain text"

    raw_tool = Tool(
        name="raw",
        description="Raw.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=raw,
    )
    fake = FakeModel([tool_message("raw", {})])
    with pytest.raises(AgentError) as exc_info:
        _client(fake).run_agent(
            system_prompt="system",
            user_prompt="user",
            tools=[raw_tool],
            max_tool_calls=5,
            force_tool_call=False,
            text_is_final=True,
        )
    assert exc_info.value.code == "schema"
    assert "must return a dict" in str(exc_info.value)
    assert "str" in str(exc_info.value)
    assert fake.chat_calls == 1


def test_text_final_agent_can_answer_without_tools() -> None:
    fake = FakeModel([text_message("just an answer")])
    result = _run(_client(fake), force=False, text_final=True, tools=[_echo_tool()], terminal=None)
    assert result.text == "just an answer"
    assert fake.chat_calls == 1


def test_http_error_is_structured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    client = LLMClient(_config(), transport=transport)
    with pytest.raises(AgentError) as exc_info:
        client.run_agent(
            system_prompt="system",
            user_prompt="user",
            tools=[_echo_tool()],
            max_tool_calls=2,
            force_tool_call=False,
            text_is_final=True,
        )
    assert exc_info.value.code == "http"
    assert "500" in str(exc_info.value)


def test_timeout_is_structured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        raise httpx.ReadTimeout("too slow")

    client = LLMClient(_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(AgentError) as exc_info:
        client.run_agent(
            system_prompt="system",
            user_prompt="user",
            tools=[_echo_tool()],
            max_tool_calls=2,
            force_tool_call=False,
            text_is_final=True,
        )
    assert exc_info.value.code == "timeout"


def test_payload_caps_tokens_and_disables_thinking() -> None:
    fake = FakeModel([tool_message("report", {"value": "ok"})])
    config = _config().model_copy(update={"max_tokens": 123})
    client = LLMClient(config, transport=fake.transport)
    _run(client)
    payload = fake.chat_requests[0]
    assert payload["max_tokens"] == 123
    assert payload["reasoning_effort"] == "none"
    assert "chat_template_kwargs" not in payload


def test_payload_passes_explicit_reasoning_effort() -> None:
    fake = FakeModel([tool_message("report", {"value": "ok"})])
    config = _config().model_copy(update={"reasoning_effort": "low"})
    client = LLMClient(config, transport=fake.transport)
    _run(client)
    assert fake.chat_requests[0]["reasoning_effort"] == "low"


def test_payload_omits_max_tokens_when_zero() -> None:
    fake = FakeModel([tool_message("report", {"value": "ok"})])
    config = _config().model_copy(update={"max_tokens": 0})
    client = LLMClient(config, transport=fake.transport)
    _run(client)
    assert "max_tokens" not in fake.chat_requests[0]


def test_unknown_tool_repair_names_available_tools_and_no_shell() -> None:
    fake = FakeModel(
        [
            tool_message("bash", {"command": "ls"}),
            tool_message("report", {"value": "recovered"}),
        ]
    )
    result = _run(_client(fake))
    assert isinstance(result.data, Report)
    assert result.data.value == "recovered"
    repair = json.loads(fake.tool_results(1)[0])
    assert repair["error"] == "invalid_arguments"
    assert repair["available_tools"] == ["echo", "report"]
    assert "no shell" in repair["message"]
    assert "bash" in repair["message"]


def test_run_agent_reasoning_effort_override_wins_over_config() -> None:
    fake = FakeModel([tool_message("report", {"value": "ok"})])
    config = _config().model_copy(update={"reasoning_effort": "medium"})
    client = LLMClient(config, transport=fake.transport)
    client.run_agent(
        system_prompt="system",
        user_prompt="user",
        tools=[_echo_tool(), _report_tool()],
        max_tool_calls=10,
        terminal_tool="report",
        terminal_schema=Report,
        reasoning_effort="none",
    )
    assert fake.chat_requests[0]["reasoning_effort"] == "none"


def test_run_agent_inherits_global_reasoning_effort() -> None:
    fake = FakeModel([tool_message("report", {"value": "ok"})])
    config = _config().model_copy(update={"reasoning_effort": "low"})
    client = LLMClient(config, transport=fake.transport)
    client.run_agent(
        system_prompt="system",
        user_prompt="user",
        tools=[_echo_tool(), _report_tool()],
        max_tool_calls=10,
        terminal_tool="report",
        terminal_schema=Report,
    )
    assert fake.chat_requests[0]["reasoning_effort"] == "low"


def test_run_agent_omits_reasoning_effort_when_unset() -> None:
    fake = FakeModel([tool_message("report", {"value": "ok"})])
    config = _config().model_copy(update={"reasoning_effort": None})
    client = LLMClient(config, transport=fake.transport)
    client.run_agent(
        system_prompt="system",
        user_prompt="user",
        tools=[_echo_tool(), _report_tool()],
        max_tool_calls=10,
        terminal_tool="report",
        terminal_schema=Report,
    )
    assert "reasoning_effort" not in fake.chat_requests[0]


def test_finish_reason_length_without_tool_call_aborts_immediately() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": None},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    client = LLMClient(_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(AgentError) as exc_info:
        client.run_agent(
            system_prompt="system",
            user_prompt="user",
            tools=[_echo_tool(), _report_tool()],
            max_tool_calls=10,
            terminal_tool="report",
            terminal_schema=Report,
        )
    assert exc_info.value.code == "budget"
    assert "thinking loop" in str(exc_info.value)


# -- budget note ---------------------------------------------


def _collect_notes(fake: FakeModel) -> list[str]:
    """Every budget ``note`` that appeared in any request's tool messages."""
    notes: list[str] = []
    for request in fake.chat_requests:
        for message in request["messages"]:
            if message.get("role") != "tool":
                continue
            try:
                parsed = json.loads(message.get("content"))
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("note"), str):
                notes.append(parsed["note"])
    return notes


def test_non_terminal_tool_result_carries_budget_note() -> None:
    fake = FakeModel(
        [
            tool_message("echo", {"text": "hi"}),
            tool_message("report", {"value": "done"}),
        ]
    )
    result = _run(_client(fake), max_calls=20)
    assert isinstance(result.data, Report)
    # The echo result is visible to the model in the second (terminal) request,
    # stamped with the remaining-budget note.
    payload = json.loads(fake.tool_results(1)[0])
    assert payload["echo"] == "hi"
    assert payload["note"] == "[tool calls 1/20 – 19 left]"


def test_budget_note_excludes_repairs_and_text_nudges() -> None:
    fake = FakeModel(
        [
            tool_message("echo", "{not json"),  # malformed -> repair, does not count
            text_message("thinking..."),  # text reply -> nudge, does not count
            tool_message("echo", {"text": "ok"}),  # first real tool -> 1/20
            tool_message("report", {"value": "done"}),
        ]
    )
    result = _run(_client(fake), max_calls=20)
    assert isinstance(result.data, Report)
    assert _collect_notes(fake) == ["[tool calls 1/20 – 19 left]"]


def test_terminal_report_turn_gets_no_note() -> None:
    fake = FakeModel([tool_message("report", {"value": "done"})])
    result = _run(_client(fake), max_calls=20)
    assert isinstance(result.data, Report)
    # The terminal tool returns before any note is stamped, so no tool message
    # ever carries one.
    assert _collect_notes(fake) == []


def test_parallel_tool_calls_each_get_a_note() -> None:
    parallel = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "echo", "arguments": json.dumps({"text": "a"})},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "echo", "arguments": json.dumps({"text": "b"})},
            },
        ],
    }
    fake = FakeModel([parallel, tool_message("report", {"value": "done"})])
    result = _run(_client(fake), max_calls=20)
    assert isinstance(result.data, Report)
    results = [json.loads(content) for content in fake.tool_results(1)]
    assert [result["echo"] for result in results] == ["a", "b"]
    assert [result["note"] for result in results] == [
        "[tool calls 1/20 – 19 left]",
        "[tool calls 2/20 – 18 left]",
    ]


def test_budget_note_shows_zero_left_before_abort() -> None:
    fake = FakeModel(
        [
            tool_message("echo", {"text": "one"}),
            tool_message("echo", {"text": "two"}),
        ]
    )
    with pytest.raises(AgentError) as exc_info:
        _run(_client(fake), max_calls=1)
    assert exc_info.value.code == "budget"
    # Inform-only: the last allowed call is stamped with 0 left before the abort.
    assert _collect_notes(fake) == ["[tool calls 1/1 – 0 left]"]


def test_text_final_agent_stamps_notes_on_its_tool_calls(project) -> None:
    # The explorer is a text-final agent that runs read-only tools; its results
    # get notes just like any other agent's.
    fake = FakeModel(
        [
            tool_message("read_file", {"path": "src/a.py"}),
            text_message("the value is defined on line 2"),
        ]
    )
    client = LLMClient(_config(), transport=fake.transport, max_prompt_bytes=400_000)
    client.run_agent(
        system_prompt="system",
        user_prompt="where is the value?",
        tools=read_only_tools(project.root, 200_000, 16_000),
        max_tool_calls=5,
        force_tool_call=False,
        text_is_final=True,
        agent="explorer",
    )
    assert _collect_notes(fake) == ["[tool calls 1/5 – 4 left]"]
