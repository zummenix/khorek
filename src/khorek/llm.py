"""OpenAI-compatible LLM client and the bounded tool-calling agent loop.

Every agent is the same loop: send messages, receive tool calls, run them,
repeat.  The loop enforces these safety rails:

* forced tool calls (``tool_choice: "required"``) for structured agents;
* per-call timeouts and a hard prompt-size budget;
* repeat-call detection (twice in a row, or a third occurrence overall);
* one repair attempt for malformed tool arguments;
* a nudge (then abort) when a tool-required agent replies with text;
* tool handlers that return a dict and bound their own output.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from khorek.config import DEFAULT_MAX_PROMPT_BYTES, LLMConfig
from khorek.trace import TraceRecorder

AgentErrorCode = Literal[
    "loop",
    "budget",
    "prompt_budget",
    "timeout",
    "http",
    "schema",
    "no_tool_call",
]

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


class ModelUnavailable(Exception):
    """Fatal preamble failure: no model endpoint reachable."""


class ToolValidationError(ValueError):
    """A tool call's arguments were invalid; the model gets one repair."""


class AgentError(Exception):
    """A bounded-agent failure with a machine-readable code."""

    def __init__(self, code: AgentErrorCode, message: str, *, last_text: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.last_text = last_text


@dataclass(frozen=True)
class Tool:
    """A tool exposed to the model.

    ``handler`` is ``None`` only for a terminal tool whose arguments are
    validated against ``terminal_schema`` in :meth:`LLMClient.run_agent`.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler | None = None


M = TypeVar("M", bound=BaseModel)


@dataclass
class AgentResult:
    """Final result of an agent loop."""

    text: str = ""
    data: BaseModel | None = None

    def data_of(self, model: type[M]) -> M:
        """Return terminal data if it has the expected model type."""
        if not isinstance(self.data, model):
            raise TypeError(f"agent returned {type(self.data).__name__}, expected {model.__name__}")
        return self.data


@dataclass(frozen=True)
class _ToolCall:
    call_id: str
    name: str
    arguments: Any


@dataclass
class _AgentState:
    steps: int = 0
    requests: int = 0
    tool_calls_used: int = 0
    repairs: int = 0
    last_signature: str | None = None
    signature_counts: dict[str, int] = field(default_factory=dict)
    consecutive_text: int = 0


@dataclass(frozen=True)
class _ToolOutcome:
    data: BaseModel


class LLMClient:
    """HTTP client plus the generic bounded agent loop."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
        tracer: TraceRecorder | None = None,
    ) -> None:
        self.config = config
        self.max_prompt_bytes = max_prompt_bytes
        self.tracer = tracer
        self._base_url = config.base_url.rstrip("/")
        self._client = httpx.Client(transport=transport)

    # -- HTTP ----------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def preflight(self) -> None:
        """Probe ``GET {base_url}/models``; connection failure is fatal."""
        url = f"{self._base_url}/models"
        try:
            response = self._client.get(url, timeout=5.0)
        except httpx.TimeoutException as exc:
            raise ModelUnavailable(f"model preflight timed out: {url}") from exc
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"model endpoint unreachable ({url}): {exc}") from exc
        if not response.is_success:
            warnings.warn(
                f"khorek: model preflight returned HTTP {response.status_code} from "
                f"{url}; continuing anyway",
                RuntimeWarning,
                stacklevel=2,
            )

    # -- agent loop ----------------------------------------------------

    def run_agent(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: Sequence[Tool],
        max_tool_calls: int,
        force_tool_call: bool = True,
        text_is_final: bool = False,
        terminal_tool: str | None = None,
        terminal_schema: type[BaseModel] | None = None,
        agent: str = "agent",
        reasoning_effort: str | None = None,
    ) -> AgentResult:
        """Run a bounded tool-calling loop and return its final result."""
        by_name = {tool.name: tool for tool in tools}
        if len(by_name) != len(tools):
            raise ValueError("duplicate tool names")
        if terminal_tool is not None and terminal_schema is None:
            raise ValueError("terminal_tool requires terminal_schema")
        if terminal_tool is not None and terminal_tool not in by_name:
            raise ValueError(f"terminal tool {terminal_tool!r} is not in tools")

        tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tracer = self.tracer
        started = time.monotonic()
        available_tools = list(by_name)

        if tracer is not None:
            tracer.agent_start(
                agent=agent,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tool_definitions,
                max_tool_calls=max_tool_calls,
                force_tool_call=force_tool_call,
                text_is_final=text_is_final,
                terminal_tool=terminal_tool,
            )

        state = _AgentState()

        def finish(result: AgentResult) -> AgentResult:
            if tracer is not None:
                tracer.agent_end(
                    agent=agent,
                    status="ok",
                    text=result.text,
                    data=result.data.model_dump() if result.data is not None else None,
                    duration_ms=_elapsed_ms(started),
                )
            return result

        try:
            while True:
                self._check_prompt_budget(messages)
                message, finish_reason = self._request(
                    messages,
                    tool_definitions,
                    force_tool_call,
                    agent=agent,
                    step=state.requests,
                    reasoning_effort=reasoning_effort,
                )
                state.requests += 1
                text = _message_text(message)
                calls = _extract_tool_calls(message)
                messages.append(_assistant_message(message, calls))

                if not calls:
                    if not self._handle_text_reply(
                        text,
                        finish_reason,
                        text_is_final=text_is_final,
                        messages=messages,
                        state=state,
                        tracer=tracer,
                        max_tool_calls=max_tool_calls,
                        terminal_tool=terminal_tool,
                        agent=agent,
                    ):
                        return finish(AgentResult(text=text))
                    continue

                state.consecutive_text = 0
                for call in calls:
                    signature = f"{call.name}:{_canonical_arguments(call.arguments)}"
                    state.signature_counts[signature] = state.signature_counts.get(signature, 0) + 1
                    if signature == state.last_signature or state.signature_counts[signature] >= 3:
                        raise AgentError(
                            "loop",
                            f"repeated tool call aborted the agent: {call.name}",
                        )
                    state.last_signature = signature

                    outcome = self._invoke_tool(
                        messages,
                        call,
                        by_name.get(call.name),
                        state=state,
                        tracer=tracer,
                        available_tools=available_tools,
                        terminal_tool=terminal_tool,
                        terminal_schema=terminal_schema,
                        max_tool_calls=max_tool_calls,
                        agent=agent,
                    )
                    if outcome is None:
                        continue
                    if isinstance(outcome, _ToolOutcome):
                        return finish(AgentResult(data=outcome.data))
                    content, state.tool_calls_used = outcome
                    messages.append(
                        {"role": "tool", "tool_call_id": call.call_id, "content": content}
                    )
        except AgentError as exc:
            if tracer is not None:
                tracer.agent_end(
                    agent=agent,
                    status="error",
                    error={
                        "code": exc.code,
                        "message": str(exc),
                        "last_text": exc.last_text,
                    },
                    duration_ms=_elapsed_ms(started),
                )
            raise

    def _handle_text_reply(
        self,
        text: str,
        finish_reason: str | None,
        *,
        text_is_final: bool,
        messages: list[dict[str, Any]],
        state: _AgentState,
        tracer: TraceRecorder | None,
        max_tool_calls: int,
        terminal_tool: str | None,
        agent: str,
    ) -> bool:
        """True when a nudge was appended; False when text is the final result."""
        if text_is_final:
            return False
        if finish_reason == "length":
            raise AgentError(
                "budget",
                "model hit max_tokens="
                f"{self.config.max_tokens or 'the model default'} before calling "
                "a tool; this usually means a thinking loop. Raise max_tokens or "
                'set reasoning_effort = "none" for this agent.',
            )
        state.consecutive_text += 1
        state.steps += 1
        if tracer is not None:
            tracer.record(
                "text_reply",
                agent=agent,
                step=state.requests,
                consecutive=state.consecutive_text,
                text=text,
            )
        if state.steps > max_tool_calls:
            raise AgentError(
                "budget",
                f"agent exceeded max_tool_calls={max_tool_calls} without calling a tool",
            )
        if state.consecutive_text >= 2:
            raise AgentError(
                "no_tool_call",
                "model replied with text twice instead of calling "
                f"{terminal_tool or 'a tool'}; last text: {text!r}",
                last_text=text,
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    "REMINDER from the khorek harness: you must respond with a "
                    f"tool call to `{terminal_tool or 'a tool'}`. Do not answer with "
                    "plain text."
                ),
            }
        )
        return True

    def _invoke_tool(
        self,
        messages: list[dict[str, Any]],
        call: _ToolCall,
        tool: Tool | None,
        *,
        state: _AgentState,
        tracer: TraceRecorder | None,
        available_tools: list[str],
        terminal_tool: str | None,
        terminal_schema: type[BaseModel] | None,
        max_tool_calls: int,
        agent: str,
    ) -> tuple[str, int] | _ToolOutcome | None:
        """Run one call; None signals repair, and _ToolOutcome signals terminal data."""
        try:
            arguments = _parse_arguments(call.arguments)
        except AgentError as exc:
            state.repairs = self._repair(
                messages, call, state.repairs, str(exc), available_tools, agent=agent
            )
            return None

        if tool is None:
            state.repairs = self._repair(
                messages,
                call,
                state.repairs,
                (
                    f"unknown tool: {call.name!r}; this harness has no shell and no "
                    f"other tools. Available tools: {', '.join(available_tools)}"
                ),
                available_tools,
                agent=agent,
            )
            return None

        if terminal_tool is not None and call.name == terminal_tool:
            assert terminal_schema is not None
            try:
                data = terminal_schema.model_validate(arguments)
            except ValidationError as exc:
                state.repairs = self._repair(
                    messages,
                    call,
                    state.repairs,
                    _format_validation(exc),
                    available_tools,
                    agent=agent,
                )
                return None
            if tracer is not None:
                tracer.tool_call(
                    agent=agent,
                    step=state.requests,
                    call=_call_dict(call),
                    result=data.model_dump(),
                )
            return _ToolOutcome(data=data)

        if tool.handler is None:
            state.repairs = self._repair(
                messages,
                call,
                state.repairs,
                f"tool {call.name} has no handler",
                available_tools,
                agent=agent,
            )
            return None

        if state.steps >= max_tool_calls:
            raise AgentError(
                "budget",
                f"agent exceeded max_tool_calls={max_tool_calls}",
            )
        state.steps += 1

        tool_started = time.monotonic()
        try:
            result = tool.handler(arguments)
        except ToolValidationError as exc:
            self._trace_tool_error(tracer, agent, state.requests, call, str(exc), tool_started)
            state.repairs = self._repair(
                messages, call, state.repairs, str(exc), available_tools, agent=agent
            )
            return None
        except AgentError:
            raise
        except Exception as exc:  # noqa: BLE001 - tool failures are agent failures
            self._trace_tool_error(tracer, agent, state.requests, call, str(exc), tool_started)
            raise AgentError("schema", f"tool {call.name} failed: {exc}") from exc

        if not isinstance(result, dict):
            raise AgentError(
                "schema",
                f"tool {call.name} returned {type(result).__name__}; "
                "every tool handler must return a dict",
            )
        state.tool_calls_used += 1
        result = {
            **result,
            "note": (
                f"[tool calls {state.tool_calls_used}/{max_tool_calls} "
                f"– {max_tool_calls - state.tool_calls_used} left]"
            ),
        }
        try:
            content = json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise AgentError("schema", f"tool {call.name} returned non-JSON data: {exc}") from exc
        if tracer is not None:
            tracer.tool_call(
                agent=agent,
                step=state.requests,
                call=_call_dict(call),
                result=_json_or_text(result),
                duration_ms=_elapsed_ms(tool_started),
            )
        return content, state.tool_calls_used

    @staticmethod
    def _trace_tool_error(
        tracer: TraceRecorder | None,
        agent: str,
        step: int,
        call: _ToolCall,
        error: str,
        started: float,
    ) -> None:
        if tracer is None:
            return
        tracer.tool_call(
            agent=agent,
            step=step,
            call=_call_dict(call),
            error=error,
            duration_ms=_elapsed_ms(started),
        )

    def _repair(
        self,
        messages: list[dict[str, Any]],
        call: _ToolCall,
        repairs_so_far: int,
        error: str,
        available_tools: list[str],
        *,
        agent: str,
    ) -> int:
        repairs = repairs_so_far + 1
        if repairs > 1:
            raise AgentError(
                "schema",
                f"second malformed tool call aborted the agent: {call.name}: {error}",
            )
        content = json.dumps(
            {
                "error": "invalid_arguments",
                "message": error,
                "available_tools": available_tools,
            },
            ensure_ascii=False,
        )
        messages.append({"role": "tool", "tool_call_id": call.call_id, "content": content})
        if self.tracer is not None:
            self.tracer.record(
                "repair",
                agent=agent,
                step=call.call_id,
                tool=call.name,
                error=error,
                content=content,
            )
        return repairs

    def _request(
        self,
        messages: list[dict[str, Any]],
        tool_definitions: list[dict[str, Any]],
        force_tool_call: bool,
        *,
        agent: str,
        step: int,
        reasoning_effort: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "tools": tool_definitions,
            "tool_choice": "required" if force_tool_call else "auto",
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens:
            # Ollama maps this to num_predict.  Without a cap a thinking model
            # can generate until the call timeout with nothing to show for it.
            payload["max_tokens"] = self.config.max_tokens
        # Per-agent value wins over the [llm] default; None omits the field
        # entirely so the model keeps its own default, "none" switches
        # thinking off (Ollama's documented switch for /v1 chat).
        effective = (
            reasoning_effort if reasoning_effort is not None else self.config.reasoning_effort
        )
        if effective:
            payload["reasoning_effort"] = effective

        tracer = self.tracer

        def record_error(
            code: str,
            message: str,
            *,
            status: int | None = None,
            duration_ms: int | None = None,
            response: dict[str, Any] | str | None = None,
        ) -> None:
            if tracer is None:
                return
            tracer.model_call(
                agent=agent,
                step=step,
                request=payload,
                response=response,
                status=status,
                duration_ms=duration_ms,
                error={"code": code, "message": message},
            )

        url = f"{self._base_url}/chat/completions"
        started = time.monotonic()
        try:
            response = self._client.post(url, json=payload, timeout=self.config.call_timeout)
        except httpx.TimeoutException as exc:
            record_error(
                "timeout", f"model call timed out: {url}", duration_ms=_elapsed_ms(started)
            )
            raise AgentError("timeout", f"model call timed out: {url}") from exc
        except httpx.HTTPError as exc:
            record_error(
                "http", f"model call failed ({url}): {exc}", duration_ms=_elapsed_ms(started)
            )
            raise AgentError("http", f"model call failed ({url}): {exc}") from exc

        duration_ms = _elapsed_ms(started)
        raw_text = response.text
        if not response.is_success:
            body = raw_text[:500]
            record_error(
                "http",
                f"model returned HTTP {response.status_code}: {body}",
                response=raw_text,
                status=response.status_code,
                duration_ms=duration_ms,
            )
            raise AgentError("http", f"model returned HTTP {response.status_code}: {body}")
        try:
            data = response.json()
        except ValueError as exc:
            record_error(
                "http",
                "model returned non-JSON response",
                response=raw_text,
                status=response.status_code,
                duration_ms=duration_ms,
            )
            raise AgentError("http", "model returned non-JSON response") from exc
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            record_error(
                "http",
                "model response has no choices",
                response=raw_text,
                status=response.status_code,
                duration_ms=duration_ms,
            )
            raise AgentError("http", "model response has no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            record_error(
                "http",
                "model response has no message",
                response=raw_text,
                status=response.status_code,
                duration_ms=duration_ms,
            )
            raise AgentError("http", "model response has no message")
        if tracer is not None:
            usage = data.get("usage") if isinstance(data, dict) else None
            tracer.model_call(
                agent=agent,
                step=step,
                request=payload,
                response=data if isinstance(data, dict) else raw_text,
                message=message,
                status=response.status_code,
                duration_ms=duration_ms,
                usage=usage if isinstance(usage, dict) else None,
                finish_reason=_finish_reason(data),
            )
        return message, _finish_reason(data)

    def _check_prompt_budget(self, messages: list[dict[str, Any]]) -> None:
        size = len(json.dumps(messages, ensure_ascii=False, default=str).encode("utf-8"))
        if size > self.max_prompt_bytes:
            raise AgentError(
                "prompt_budget",
                f"prompt is {size} bytes, over max_prompt_bytes={self.max_prompt_bytes}",
            )


# -- message helpers ---------------------------------------------------


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def _call_dict(call: _ToolCall) -> dict[str, Any]:
    return {"id": call.call_id, "name": call.name, "arguments": call.arguments}


def _json_or_text(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def _finish_reason(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    reason = choices[0].get("finish_reason")
    return reason if isinstance(reason, str) else None


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def _extract_tool_calls(message: dict[str, Any]) -> list[_ToolCall]:
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    calls: list[_ToolCall] = []
    for index, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            continue
        function = raw.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        call_id = raw.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{index}"
        calls.append(_ToolCall(call_id=call_id, name=name, arguments=function.get("arguments")))
    return calls


def _assistant_message(message: dict[str, Any], calls: list[_ToolCall]) -> dict[str, Any]:
    normalized_calls = [
        {
            "id": call.call_id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": call.arguments
                if isinstance(call.arguments, str)
                else json.dumps(call.arguments),
            },
        }
        for call in calls
    ]
    content = message.get("content")
    normalized: dict[str, Any] = {
        "role": "assistant",
        "content": content if isinstance(content, str) else None,
    }
    if normalized_calls:
        normalized["tool_calls"] = normalized_calls
    return normalized


def _canonical_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return arguments
        return json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False)


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        raise AgentError("schema", "tool arguments are not a JSON object")
    try:
        parsed = json.loads(arguments)
    except ValueError as exc:
        raise AgentError("schema", f"tool arguments are not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AgentError("schema", "tool arguments must be a JSON object")
    return parsed


def _format_validation(exc: ValidationError) -> str:
    details = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        details.append(f"{location}: {error['msg']}" if location else error["msg"])
    return "; ".join(details)


__all__ = [
    "AgentError",
    "AgentErrorCode",
    "AgentResult",
    "LLMClient",
    "ModelUnavailable",
    "Tool",
    "ToolValidationError",
]
