"""The ``khorek init`` interactive setup wizard."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from khorek.config import (
    DEFAULT_MAX_ISSUES_PER_FILE,
    DEFAULT_VERIFY_TIMEOUT,
    AgentsConfig,
    DefaultsConfig,
    LLMConfig,
    ObserveConfig,
)

Answers = Callable[[str], str]

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
ALL_FILES_SCOPE = "*"


class InitError(Exception):
    """Raised when the interactive wizard cannot produce a valid config."""


@dataclass(frozen=True)
class AspectDraft:
    """One opt-in aspect collected during the wizard."""

    id: str
    prompt: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class InitAnswers:
    """Validated answers collected by the interactive wizard."""

    base_url: str
    model: str
    verify_cmd: str
    aspect: AspectDraft | None


@dataclass(frozen=True)
class Section:
    """One model-backed TOML section and its rendering exceptions."""

    header: str
    config: BaseModel
    omitted_fields: frozenset[str] = frozenset()
    commented_fields: frozenset[str] = frozenset()


ASPECT_LIBRARY: tuple[AspectDraft, ...] = (
    AspectDraft(
        id="comments",
        prompt="Find comments that are inconsistent with the code.",
        files=("src/",),
    ),
)


def _default_answers(question: str) -> str:
    return input(question)


def ask(question: str, *, answers: Answers, default: str | None = None) -> str:
    """Ask *question* through the answers seam and return the stripped answer.

    An empty answer selects *default* (the wizard's skip path); with no *default*
    the empty answer is returned for the caller to validate.
    """
    value = answers(question).strip()
    if value == "" and default is not None:
        return default
    return value


def _collect_aspect(answers: Answers) -> AspectDraft | None:
    answer = ask("Add a code-review aspect? [y/N]:", answers=answers).lower()
    if answer not in ("y", "yes"):
        return None
    draft = ASPECT_LIBRARY[0]
    print(f"Aspect prompt: {draft.prompt}")
    files = ask(
        f"Files scope for aspect {draft.id!r} (comma-separated paths or globs, empty for all):",
        answers=answers,
        default=ALL_FILES_SCOPE,
    )
    scope = tuple(part.strip() for part in files.split(",") if part.strip()) or (ALL_FILES_SCOPE,)
    return AspectDraft(id=draft.id, prompt=draft.prompt, files=scope)


def collect_answers(answers: Answers | None = None) -> InitAnswers:
    """Run the interactive prompts and validate the answers.

    Raises :class:`InitError` when a required answer is missing; a ``KeyboardInterrupt``
    propagates so the caller can abort without dumping anything.
    """
    provider: Answers = answers if answers is not None else _default_answers
    base_url = ask(
        f"Base URL of the OpenAI-compatible endpoint [default: {DEFAULT_BASE_URL}]:",
        answers=provider,
        default=DEFAULT_BASE_URL,
    )
    model = ask("Model name: ", answers=provider)
    if not model:
        raise InitError("model is required: the dumped khorek.toml must set [llm] model")
    verify_cmd = ask(
        "Verify command (a non-mutating check run after each fix, e.g. 'ruff format --check .'):",
        answers=provider,
    )
    if not verify_cmd:
        raise InitError("at least one [[verify]] command is required in the dumped khorek.toml")
    aspect = _collect_aspect(answers=provider)
    return InitAnswers(base_url=base_url, model=model, verify_cmd=verify_cmd, aspect=aspect)


def _toml_string(value: str) -> str:
    """Render *value* as a single-line TOML basic string."""
    pieces = ['"']
    for char in value:
        if char == "\\":
            pieces.append("\\\\")
        elif char == '"':
            pieces.append('\\"')
        elif char == "\n":
            pieces.append("\\n")
        elif char == "\r":
            pieces.append("\\r")
        elif char == "\t":
            pieces.append("\\t")
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            pieces.append(f"\\u{ord(char):04X}")
        else:
            pieces.append(char)
    pieces.append('"')
    return "".join(pieces)


def _toml_value(value: object) -> str:
    """Render *value* as a TOML scalar literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    raise TypeError(f"unsupported TOML value: {value!r}")


def _commented(key: str, default: object) -> str:
    """Render an optional key commented out; the built-in default is shown when one exists."""
    if default is None:
        return f"# {key}"
    return f"# {key} = {_toml_value(default)}"


def render_init_config(answers: InitAnswers) -> str:
    """Render the canonical ``khorek.toml`` text for the collected answers."""
    agents = AgentsConfig()
    sections = (
        Section(
            "[llm]",
            LLMConfig(base_url=answers.base_url, model=answers.model),
            commented_fields=frozenset(
                {"temperature", "call_timeout", "max_tokens", "reasoning_effort"}
            ),
        ),
        Section("[defaults]", DefaultsConfig()),
        Section("[observe]", ObserveConfig()),
        Section(
            "[agents.aspect]",
            agents.aspect,
            omitted_fields=frozenset({"reasoning_effort"}),
        ),
        Section(
            "[agents.explorer]",
            agents.explorer,
            omitted_fields=frozenset({"reasoning_effort"}),
        ),
        Section(
            "[agents.fixer]",
            agents.fixer,
            omitted_fields=frozenset({"reasoning_effort"}),
        ),
        Section(
            "[agents.reviewer]",
            agents.reviewer,
            omitted_fields=frozenset({"reasoning_effort"}),
        ),
    )

    lines: list[str] = []
    for section in sections:
        if lines:
            lines.append("")
        lines.append(section.header)
        for key in type(section.config).model_fields:
            if key in section.omitted_fields:
                continue
            value = getattr(section.config, key)
            if key in section.commented_fields:
                lines.append(_commented(key, value))
            else:
                lines.append(f"{key} = {_toml_value(value)}")

    lines.append("")
    lines.append("[[verify]]")
    lines.append(f"cmd = {_toml_string(answers.verify_cmd)}")
    lines.append(_commented("timeout", DEFAULT_VERIFY_TIMEOUT))

    if answers.aspect is not None:
        lines.append("")
        lines.append("[[aspects]]")
        lines.append(f"id = {_toml_string(answers.aspect.id)}")
        lines.append(f"prompt = {_toml_string(answers.aspect.prompt)}")
        files_value = ", ".join(_toml_string(scope) for scope in answers.aspect.files)
        lines.append(f"files = [{files_value}]")
        lines.append(_commented("max_issues_per_file", DEFAULT_MAX_ISSUES_PER_FILE))

    return "\n".join(lines) + "\n"


def resolve_config_name(name: str) -> str:
    """Return the config file name for *name*, appending ``.toml`` when missing."""
    if name.endswith(".toml"):
        return name
    return f"{name}.toml"


def init_config_path(destination: Path, config_name: str) -> Path:
    """Return the dumped config path inside *destination*."""
    return destination / resolve_config_name(config_name)


__all__ = [
    "ALL_FILES_SCOPE",
    "Answers",
    "ASPECT_LIBRARY",
    "AspectDraft",
    "InitAnswers",
    "InitError",
    "ask",
    "collect_answers",
    "init_config_path",
    "render_init_config",
    "resolve_config_name",
]
