"""OpenAI-compatible LLM client and the bounded tool-calling agent loop.

Every agent is the same loop: send messages, receive tool calls, run them,
repeat.  The loop enforces these safety rails:

* forced tool calls (``tool_choice: "required"``) for structured agents;
* per-call timeouts and a hard prompt-size budget;
* repeat-call detection (twice in a row, or three times overall);
* one repair attempt for malformed tool arguments;
* a nudge (then abort) when a tool-required agent replies with text;
* tool-result truncation with a ``truncated`` marker.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError

from khorek.config import DEFAULT_MAX_PROMPT_BYTES, DEFAULT_MAX_TOOL_OUTPUT_BYTES, LLMConfig
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

ToolHandler = Callable[[dict[str, Any]], Any]


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


@dataclass
class AgentResult:
    """Final result of an agent loop."""

    text: str = ""
    data: BaseModel | None = None


@dataclass(frozen=True)
class _ToolCall:
    call_id: str
    name: str
    arguments: Any


class LLMClient:
    """HTTP client plus the generic bounded agent loop."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
        max_tool_output_bytes: int = DEFAULT_MAX_TOOL_OUTPUT_BYTES,
        tracer: TraceRecorder | None = None,
    ) -> None:
        self.config = config
        self.max_prompt_bytes = max_prompt_bytes
        self.max_tool_output_bytes = max_tool_output_bytes
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
        think: bool | None = None,
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

        steps = 0
        requests = 0
        repairs = 0
        last_signature: str | None = None
        signature_counts: dict[str, int] = {}
        consecutive_text = 0

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
                    step=requests,
                    think=think,
                )
                requests += 1
                text = _message_text(message)
                calls = _extract_tool_calls(message)
                messages.append(_assistant_message(message, calls))

                if not calls:
                    if text_is_final:
                        return finish(AgentResult(text=text))
                    if finish_reason == "length":
                        raise AgentError(
                            "budget",
                            "model hit max_tokens="
                            f"{self.config.max_tokens or 'the model default'} before calling "
                            "a tool; this usually means a thinking loop. Raise max_tokens or "
                            "set think = false for this agent.",
                        )
                    consecutive_text += 1
                    steps += 1
                    if tracer is not None:
                        tracer.record(
                            "text_reply",
                            agent=agent,
                            step=requests,
                            consecutive=consecutive_text,
                            text=text,
                        )
                    if steps > max_tool_calls:
                        raise AgentError(
                            "budget",
                            "agent exceeded max_tool_calls="
                            f"{max_tool_calls} without calling a tool",
                        )
                    if consecutive_text >= 2:
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
                    continue

                consecutive_text = 0
                for call in calls:
                    signature = f"{call.name}:{_canonical_arguments(call.arguments)}"
                    signature_counts[signature] = signature_counts.get(signature, 0) + 1
                    if signature == last_signature or signature_counts[signature] >= 3:
                        raise AgentError(
                            "loop",
                            f"repeated tool call aborted the agent: {call.name}",
                        )
                    last_signature = signature

                    try:
                        arguments = _parse_arguments(call.arguments)
                    except AgentError as exc:
                        repairs = self._repair(
                            messages, call, repairs, str(exc), available_tools, agent=agent
                        )
                        continue

                    tool = by_name.get(call.name)
                    if tool is None:
                        repairs = self._repair(
                            messages,
                            call,
                            repairs,
                            (
                                f"unknown tool: {call.name!r}; this harness has no shell and no "
                                f"other tools. Available tools: {', '.join(available_tools)}"
                            ),
                            available_tools,
                            agent=agent,
                        )
                        continue

                    if terminal_tool is not None and call.name == terminal_tool:
                        assert terminal_schema is not None
                        try:
                            data = terminal_schema.model_validate(arguments)
                        except ValidationError as exc:
                            repairs = self._repair(
                                messages,
                                call,
                                repairs,
                                _format_validation(exc),
                                available_tools,
                                agent=agent,
                            )
                            continue
                        if tracer is not None:
                            tracer.tool_call(
                                agent=agent,
                                step=requests,
                                call=_call_dict(call),
                                result=data.model_dump(),
                            )
                        return finish(AgentResult(data=data))

                    if tool.handler is None:
                        repairs = self._repair(
                            messages,
                            call,
                            repairs,
                            f"tool {call.name} has no handler",
                            available_tools,
                            agent=agent,
                        )
                        continue

                    if steps >= max_tool_calls:
                        raise AgentError(
                            "budget",
                            f"agent exceeded max_tool_calls={max_tool_calls}",
                        )
                    steps += 1

                    tool_started = time.monotonic()
                    try:
                        result = tool.handler(arguments)
                    except ToolValidationError as exc:
                        self._trace_tool_error(
                            tracer, agent, requests, call, str(exc), tool_started
                        )
                        repairs = self._repair(
                            messages, call, repairs, str(exc), available_tools, agent=agent
                        )
                        continue
                    except AgentError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - tool failures are agent failures
                        self._trace_tool_error(
                            tracer, agent, requests, call, str(exc), tool_started
                        )
                        raise AgentError("schema", f"tool {call.name} failed: {exc}") from exc

                    try:
                        content = _truncate_json(result, self.max_tool_output_bytes)
                    except (TypeError, ValueError) as exc:
                        raise AgentError(
                            "schema", f"tool {call.name} returned non-JSON data: {exc}"
                        ) from exc
                    if tracer is not None:
                        tracer.tool_call(
                            agent=agent,
                            step=requests,
                            call=_call_dict(call),
                            result=_json_or_text(result),
                            duration_ms=_elapsed_ms(tool_started),
                        )
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
        think: bool | None = None,
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
        effective_think = self.config.think if think is None else think
        if not effective_think:
            # Ollama's /v1 chat endpoint uses reasoning_effort (not
            # chat_template_kwargs) to switch thinking off.
            payload["reasoning_effort"] = "none"
        elif self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort

        url = f"{self._base_url}/chat/completions"
        tracer = self.tracer
        started = time.monotonic()
        try:
            response = self._client.post(url, json=payload, timeout=self.config.call_timeout)
        except httpx.TimeoutException as exc:
            if tracer is not None:
                tracer.model_call(
                    agent=agent,
                    step=step,
                    request=payload,
                    duration_ms=_elapsed_ms(started),
                    error={"code": "timeout", "message": f"model call timed out: {url}"},
                )
            raise AgentError("timeout", f"model call timed out: {url}") from exc
        except httpx.HTTPError as exc:
            if tracer is not None:
                tracer.model_call(
                    agent=agent,
                    step=step,
                    request=payload,
                    duration_ms=_elapsed_ms(started),
                    error={"code": "http", "message": f"model call failed ({url}): {exc}"},
                )
            raise AgentError("http", f"model call failed ({url}): {exc}") from exc

        duration_ms = _elapsed_ms(started)
        raw_text = response.text
        if not response.is_success:
            body = raw_text[:500]
            if tracer is not None:
                tracer.model_call(
                    agent=agent,
                    step=step,
                    request=payload,
                    response=raw_text,
                    status=response.status_code,
                    duration_ms=duration_ms,
                    error={
                        "code": "http",
                        "message": f"model returned HTTP {response.status_code}: {body}",
                    },
                )
            raise AgentError("http", f"model returned HTTP {response.status_code}: {body}")
        try:
            data = response.json()
        except ValueError as exc:
            if tracer is not None:
                tracer.model_call(
                    agent=agent,
                    step=step,
                    request=payload,
                    response=raw_text,
                    status=response.status_code,
                    duration_ms=duration_ms,
                    error={"code": "http", "message": "model returned non-JSON response"},
                )
            raise AgentError("http", "model returned non-JSON response") from exc
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            if tracer is not None:
                tracer.model_call(
                    agent=agent,
                    step=step,
                    request=payload,
                    response=raw_text,
                    status=response.status_code,
                    duration_ms=duration_ms,
                    error={"code": "http", "message": "model response has no choices"},
                )
            raise AgentError("http", "model response has no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            if tracer is not None:
                tracer.model_call(
                    agent=agent,
                    step=step,
                    request=payload,
                    response=raw_text,
                    status=response.status_code,
                    duration_ms=duration_ms,
                    error={"code": "http", "message": "model response has no message"},
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


# -- tool result truncation -------------------------------------------


def _truncate_json(value: Any, max_bytes: int) -> str:
    """Serialize a tool result, truncating with a ``truncated`` marker."""
    try:
        serialized = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"not JSON serializable: {exc}") from exc
    if len(serialized.encode("utf-8")) <= max_bytes:
        return serialized

    if isinstance(value, dict) and isinstance(value.get("content"), str):
        payload = {key: item for key, item in value.items() if key not in {"content", "truncated"}}
        payload["truncated"] = True
        return _fit_text(payload, "content", value["content"], max_bytes)

    wrapper: dict[str, Any] = {
        "truncated": True,
        "original_bytes": len(serialized.encode("utf-8")),
        "preview": serialized,
    }
    fitted = _fit_text(wrapper, "preview", serialized, max_bytes)
    if len(fitted.encode("utf-8")) > max_bytes:
        minimal = json.dumps({"truncated": True})
        if len(minimal.encode("utf-8")) <= max_bytes:
            return minimal
    return fitted


def _fit_text(payload: dict[str, Any], key: str, text: str, max_bytes: int) -> str:
    low = 0
    high = len(text)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        payload[key] = text[:middle]
        serialized = json.dumps(payload, ensure_ascii=False)
        if len(serialized.encode("utf-8")) <= max_bytes:
            best = payload[key]
            low = middle + 1
        else:
            high = middle - 1
    payload[key] = best
    return json.dumps(payload, ensure_ascii=False)


__all__ = [
    "AgentError",
    "AgentErrorCode",
    "AgentResult",
    "LLMClient",
    "ModelUnavailable",
    "Tool",
    "ToolValidationError",
]
