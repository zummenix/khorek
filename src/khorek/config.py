"""Configuration models and TOML loading for khorek.

The configuration file is ``<project>/khorek.toml`` unless ``--config`` points
somewhere else.  Unknown keys are errors so typos never silently fall back to
defaults.  ``KHOREK_LLM_*`` environment variables override the ``[llm]``
section and win over the file.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_MAX_ISSUES_PER_FILE = 3
DEFAULT_MAX_FILE_BYTES = 200_000
DEFAULT_MAX_PROMPT_BYTES = 400_000
DEFAULT_MAX_TOOL_OUTPUT_BYTES = 16_000
DEFAULT_MAX_CYCLES = 25
DEFAULT_MAX_HANDOFFS = 3
DEFAULT_REVIEW_MAX_HANDOFFS = 2
DEFAULT_MAX_TOOL_CALLS = 20
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TEMPERATURE = 0.2
DEFAULT_CALL_TIMEOUT = 600.0
DEFAULT_MAX_TOKENS = 8192
DEFAULT_KEEP_TRACES = 30

# Environment overrides for the [llm] section; the environment wins.
ENV_BASE_URL = "KHOREK_LLM_BASE_URL"
ENV_MODEL = "KHOREK_LLM_MODEL"
ENV_TEMPERATURE = "KHOREK_LLM_TEMPERATURE"
ENV_CALL_TIMEOUT = "KHOREK_LLM_CALL_TIMEOUT"
ENV_MAX_TOKENS = "KHOREK_LLM_MAX_TOKENS"
ENV_REASONING_EFFORT = "KHOREK_LLM_REASONING_EFFORT"


class ConfigError(Exception):
    """Raised when the configuration is missing, malformed, or invalid."""


class _StrictModel(BaseModel):
    """Base model that rejects unknown keys everywhere."""

    model_config = ConfigDict(extra="forbid")


class LLMConfig(_StrictModel):
    """OpenAI-compatible model endpoint settings."""

    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    call_timeout: float = Field(default=DEFAULT_CALL_TIMEOUT, gt=0.0)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=0)
    # Reasoning level passed to the model ("none", "low", "medium", "high", ...).
    # Values are backend-specific (Ollama and OpenAI differ slightly); we pass
    # them through unchecked so any value the endpoint accepts works.
    # ``None`` omits the field entirely, leaving the model's own default.
    reasoning_effort: str | None = None


class ObserveConfig(_StrictModel):
    """Opt-in traces of every model call, for debugging the harness."""

    enabled: bool = False
    directory: str = ""
    keep_runs: int = Field(default=DEFAULT_KEEP_TRACES, ge=0)


class DefaultsConfig(_StrictModel):
    """Harness-wide limits and budgets."""

    max_issues_per_file: int = Field(default=DEFAULT_MAX_ISSUES_PER_FILE, gt=0)
    max_file_bytes: int = Field(default=DEFAULT_MAX_FILE_BYTES, gt=0)
    max_prompt_bytes: int = Field(default=DEFAULT_MAX_PROMPT_BYTES, gt=0)
    max_tool_output_bytes: int = Field(default=DEFAULT_MAX_TOOL_OUTPUT_BYTES, gt=0)
    max_cycles: int = Field(default=DEFAULT_MAX_CYCLES, gt=0)


class AspectAgentConfig(_StrictModel):
    """Aspect agent settings (agent budgets)."""

    reasoning_effort: str | None = None  # None = inherit [llm].reasoning_effort
    max_handoffs: int = Field(default=DEFAULT_MAX_HANDOFFS, ge=0)
    max_tool_calls: int = Field(default=DEFAULT_MAX_TOOL_CALLS, gt=0)


class ExplorerAgentConfig(_StrictModel):
    """Explorer sub-agent settings."""

    reasoning_effort: str | None = None  # None = inherit [llm].reasoning_effort
    max_tool_calls: int = Field(default=DEFAULT_MAX_TOOL_CALLS, gt=0)


class FixerAgentConfig(_StrictModel):
    """Fixer settings (max rounds per issue and tool budget per round)."""

    reasoning_effort: str | None = None  # None = inherit [llm].reasoning_effort
    max_attempts: int = Field(default=DEFAULT_MAX_ATTEMPTS, gt=0)
    max_tool_calls: int = Field(default=DEFAULT_MAX_TOOL_CALLS, gt=0)


class VerifierAgentConfig(_StrictModel):
    """Verifier settings (used by the fix workflow)."""

    reasoning_effort: str | None = None  # None = inherit [llm].reasoning_effort


class ReviewerAgentConfig(_StrictModel):
    """Reviewer settings (used by the fix workflow)."""

    reasoning_effort: str | None = None  # None = inherit [llm].reasoning_effort
    max_handoffs: int = Field(default=DEFAULT_REVIEW_MAX_HANDOFFS, ge=0)
    max_tool_calls: int = Field(default=DEFAULT_MAX_TOOL_CALLS, gt=0)


class AgentsConfig(_StrictModel):
    """Per-agent configuration sections."""

    aspect: AspectAgentConfig = Field(default_factory=AspectAgentConfig)
    explorer: ExplorerAgentConfig = Field(default_factory=ExplorerAgentConfig)
    fixer: FixerAgentConfig = Field(default_factory=FixerAgentConfig)
    verifier: VerifierAgentConfig = Field(default_factory=VerifierAgentConfig)
    reviewer: ReviewerAgentConfig = Field(default_factory=ReviewerAgentConfig)


class VerifyCommand(_StrictModel):
    """One non-mutating validation command run after a fix."""

    cmd: str = Field(min_length=1)
    timeout: float = Field(default=120.0, gt=0.0)


class AspectConfig(_StrictModel):
    """One aspect: a prompt plus the files it applies to."""

    id: str
    prompt: str
    files: list[str] = Field(min_length=1)
    max_issues_per_file: int | None = Field(default=None, gt=0)


class Config(_StrictModel):
    """The whole ``khorek.toml`` file."""

    llm: LLMConfig
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    observe: ObserveConfig = Field(default_factory=ObserveConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    verify: list[VerifyCommand] = Field(default_factory=list)
    aspects: list[AspectConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_aspects(self) -> Config:
        seen: set[str] = set()
        for aspect in self.aspects:
            if not aspect.id.strip():
                raise ValueError("aspect id must not be empty")
            if aspect.id in seen:
                raise ValueError(f"duplicate aspect id: {aspect.id!r}")
            seen.add(aspect.id)
            if not aspect.prompt.strip():
                raise ValueError(f"aspect {aspect.id!r} must have a non-empty prompt")
            if (
                aspect.max_issues_per_file is not None
                and aspect.max_issues_per_file > self.defaults.max_issues_per_file
            ):
                raise ValueError(
                    f"aspect {aspect.id!r}: max_issues_per_file "
                    f"({aspect.max_issues_per_file}) may not exceed the default "
                    f"({self.defaults.max_issues_per_file})"
                )
        return self


def _env_overrides() -> dict[str, object]:
    """Collect and parse ``KHOREK_LLM_*`` overrides from the environment."""
    overrides: dict[str, object] = {}

    base_url = os.environ.get(ENV_BASE_URL)
    if base_url is not None:
        overrides["base_url"] = base_url
    model = os.environ.get(ENV_MODEL)
    if model is not None:
        overrides["model"] = model
    temperature = os.environ.get(ENV_TEMPERATURE)
    if temperature is not None:
        try:
            overrides["temperature"] = float(temperature)
        except ValueError as exc:
            raise ConfigError(f"{ENV_TEMPERATURE} must be a number, got {temperature!r}") from exc
    call_timeout = os.environ.get(ENV_CALL_TIMEOUT)
    if call_timeout is not None:
        try:
            overrides["call_timeout"] = float(call_timeout)
        except ValueError as exc:
            raise ConfigError(f"{ENV_CALL_TIMEOUT} must be a number, got {call_timeout!r}") from exc
    max_tokens = os.environ.get(ENV_MAX_TOKENS)
    if max_tokens is not None:
        try:
            overrides["max_tokens"] = int(max_tokens)
        except ValueError as exc:
            raise ConfigError(f"{ENV_MAX_TOKENS} must be an integer, got {max_tokens!r}") from exc
    reasoning_effort = os.environ.get(ENV_REASONING_EFFORT)
    if reasoning_effort is not None:
        overrides["reasoning_effort"] = reasoning_effort.strip() or None
    return overrides


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    details = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        message = error["msg"]
        details.append(f"{location}: {message}" if location else message)
    return f"invalid config {path}: " + "; ".join(details)


def _apply_env_overrides(config: Config, path: Path) -> Config:
    overrides = _env_overrides()
    if not overrides:
        return config
    merged = {**config.llm.model_dump(), **overrides}
    try:
        llm = LLMConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc
    return config.model_copy(update={"llm": llm})


def resolve_config_path(project_root: Path, config_path: Path | None = None) -> Path:
    """Return the config file to load (``<project>/khorek.toml`` by default)."""
    if config_path is None:
        return project_root / "khorek.toml"
    return config_path


def load_config(project_root: Path, config_path: Path | None = None) -> Config:
    """Load, validate, and env-override the project configuration."""
    path = resolve_config_path(project_root, config_path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc

    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"config file {path} is not valid UTF-8: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"cannot parse config file {path}: {exc}") from exc

    try:
        config = Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc

    return _apply_env_overrides(config, path)


__all__ = [
    "AspectAgentConfig",
    "AspectConfig",
    "AgentsConfig",
    "Config",
    "ConfigError",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_REVIEW_MAX_HANDOFFS",
    "DefaultsConfig",
    "ExplorerAgentConfig",
    "FixerAgentConfig",
    "LLMConfig",
    "ObserveConfig",
    "ReviewerAgentConfig",
    "VerifyCommand",
    "VerifierAgentConfig",
    "load_config",
    "resolve_config_path",
]
