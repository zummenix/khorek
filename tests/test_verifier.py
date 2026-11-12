"""Verifier agent: terminal schema, failure data pass-through, no pass/fail."""

from __future__ import annotations

import pytest

from khorek.agents.verifier import VerifierAgent
from khorek.config import LLMConfig, VerifierAgentConfig
from khorek.llm import AgentError, LLMClient
from khorek.verify import Failure
from tests.conftest import FakeModel, text_message, tool_message


def _client(fake: FakeModel) -> LLMClient:
    return LLMClient(
        LLMConfig(
            base_url="http://model.test/v1",
            model="fake-model",
            reasoning_effort="none",
            temperature=0.0,
            call_timeout=10.0,
        ),
        transport=fake.transport,
    )


def _failure() -> Failure:
    return Failure(cmd="false", exit_code=1, output_tail="boom: expected value 3")


def test_summarise_passes_failure_data_and_returns_summary() -> None:
    fake = FakeModel(
        [
            tool_message(
                "report_verification",
                {
                    "summary": "set value to 3 at src/a.py:2",
                    "failures": [
                        {"cmd": "false", "exit_code": 1, "output_tail": "boom: expected value 3"}
                    ],
                },
            )
        ]
    )
    client = _client(fake)
    try:
        agent = VerifierAgent(client, config=VerifierAgentConfig())
        summary = agent.summarise([_failure()])
    finally:
        client.close()

    assert summary == "set value to 3 at src/a.py:2"
    request = fake.chat_requests[0]
    user_prompt = request["messages"][1]["content"]
    assert '"cmd": "false"' in user_prompt
    assert '"exit_code": 1' in user_prompt
    assert "boom: expected value 3" in user_prompt
    tool_names = [definition["function"]["name"] for definition in request["tools"]]
    assert tool_names == ["report_verification"]


def test_empty_terminal_payload_gets_one_repair() -> None:
    fake = FakeModel(
        [
            tool_message("report_verification", {}),
            tool_message(
                "report_verification",
                {"summary": "retry with the corrected text", "failures": []},
            ),
        ]
    )
    client = _client(fake)
    try:
        agent = VerifierAgent(client, config=VerifierAgentConfig())
        summary = agent.summarise([_failure()])
    finally:
        client.close()

    assert summary == "retry with the corrected text"
    assert "invalid_arguments" in fake.tool_results(1)[0]


def test_text_replies_abort_the_verifier() -> None:
    fake = FakeModel([text_message("first answer"), text_message("second answer")])
    client = _client(fake)
    try:
        agent = VerifierAgent(client, config=VerifierAgentConfig())
        with pytest.raises(AgentError) as exc_info:
            agent.summarise([_failure()])
    finally:
        client.close()

    assert exc_info.value.code == "no_tool_call"
    assert fake.chat_calls == 2
