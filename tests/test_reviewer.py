"""Reviewer agent: terminal schema, prompt contents, tool set, no write tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from khorek.agents.reviewer import REVIEWER_TOOL_NAMES, ReviewerAgent, ReviewOutcome
from khorek.config import DefaultsConfig, LLMConfig, ReviewerAgentConfig
from khorek.db import Issue
from khorek.llm import AgentError, LLMClient
from tests.conftest import FakeModel, review, text_message, tool_message

DIFF = (
    "--- a/src/a.py\n"
    "+++ b/src/a.py\n"
    "@@ -1,2 +1,2 @@\n"
    "-# the value is one\n"
    "+# the value is two\n"
    " value = 2\n"
)


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


def _issue() -> Issue:
    return Issue(
        id=7,
        aspect_id="comments",
        path="src/a.py",
        description="the comment is inconsistent with the code",
        line_start=1,
        line_end=1,
        severity="high",
        retry_count=0,
        last_error=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


def _agent(client: LLMClient, root: Path) -> ReviewerAgent:
    return ReviewerAgent(
        client,
        project_root=root,
        config=ReviewerAgentConfig(),
        defaults=DefaultsConfig(),
    )


def test_review_returns_approved_outcome_with_stripped_summary(tmp_path: Path) -> None:
    fake = FakeModel([review(True, "  looks good  ")])
    client = _client(fake)
    try:
        agent = _agent(client, tmp_path)
        outcome = agent.review(_issue(), DIFF, ["pytest -q"])
    finally:
        client.close()

    assert isinstance(outcome, ReviewOutcome)
    assert outcome.approved is True
    assert outcome.summary == "looks good"


def test_review_rejection_returns_the_reason(tmp_path: Path) -> None:
    fake = FakeModel([review(False, "the change is not minimal")])
    client = _client(fake)
    try:
        agent = _agent(client, tmp_path)
        outcome = agent.review(_issue(), DIFF, ["pytest -q"])
    finally:
        client.close()

    assert isinstance(outcome, ReviewOutcome)
    assert outcome.approved is False
    assert outcome.summary == "the change is not minimal"


def test_empty_review_payload_gets_one_repair(tmp_path: Path) -> None:
    fake = FakeModel(
        [
            tool_message("report_review", {}),
            review(True, "retry with the corrected text"),
        ]
    )
    client = _client(fake)
    try:
        agent = _agent(client, tmp_path)
        outcome = agent.review(_issue(), DIFF, ["pytest -q"])
    finally:
        client.close()

    assert outcome.approved is True
    assert outcome.summary == "retry with the corrected text"
    assert "invalid_arguments" in fake.tool_results(1)[0]


def test_review_prompt_contains_issue_diff_and_verify_status(tmp_path: Path) -> None:
    fake = FakeModel([review(False, "not minimal")])
    client = _client(fake)
    try:
        agent = _agent(client, tmp_path)
        agent.review(_issue(), DIFF, ["pytest -q", "ruff format --check ."])
    finally:
        client.close()

    user_prompt = fake.chat_requests[0]["messages"][1]["content"]
    assert "Issue #7" in user_prompt
    assert "File: src/a.py" in user_prompt
    assert "the comment is inconsistent with the code" in user_prompt
    assert "all configured commands passed" in user_prompt
    assert "- pytest -q : passed" in user_prompt
    assert "- ruff format --check . : passed" in user_prompt
    assert DIFF in user_prompt


def test_review_tool_set_is_read_only(tmp_path: Path) -> None:
    fake = FakeModel([review(True, "ok")])
    client = _client(fake)
    try:
        agent = _agent(client, tmp_path)
        agent.review(_issue(), DIFF, ["pytest -q"])
    finally:
        client.close()

    tool_names = [definition["function"]["name"] for definition in fake.chat_requests[0]["tools"]]
    assert set(tool_names) == REVIEWER_TOOL_NAMES
    assert "write_file" not in tool_names
    assert "edit_file" not in tool_names


def test_text_replies_abort_the_reviewer(tmp_path: Path) -> None:
    fake = FakeModel([text_message("first answer"), text_message("second answer")])
    client = _client(fake)
    try:
        agent = _agent(client, tmp_path)
        with pytest.raises(AgentError) as exc_info:
            agent.review(_issue(), DIFF, ["pytest -q"])
    finally:
        client.close()

    assert exc_info.value.code == "no_tool_call"
    assert fake.chat_calls == 2
