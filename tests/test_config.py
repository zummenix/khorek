"""Config loading, defaults, validation, and environment overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from khorek.config import ConfigError, load_config
from tests.conftest import make_project, write_config


def test_minimal_config_applies_documented_defaults(project) -> None:
    config = load_config(project.root)
    assert config.llm.base_url == "http://model.test/v1"
    assert config.llm.model == "fake-model"
    assert config.defaults.max_issues_per_file == 3
    assert config.defaults.max_prompt_bytes == 400_000
    assert config.agents.aspect.max_handoffs == 3
    assert config.agents.aspect.max_tool_calls == 20
    assert config.agents.explorer.max_tool_calls == 20
    assert config.verify == []
    assert [aspect.id for aspect in config.aspects] == ["comments"]


def test_reviewer_defaults(project) -> None:
    config = load_config(project.root)
    assert config.agents.reviewer.prompt == ""
    assert config.agents.reviewer.reasoning_effort == "low"
    assert config.agents.reviewer.max_handoffs == 2
    assert config.agents.reviewer.max_tool_calls == 20


def test_config_path_override(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    other = tmp_path / "elsewhere.toml"
    other.write_text(
        (project.root / "khorek.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = load_config(project.root, other)
    assert config.llm.model == "fake-model"


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path)


def test_malformed_toml_raises(tmp_path: Path) -> None:
    (tmp_path / "khorek.toml").write_text("[llm\nbase_url = 1", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot parse"):
        load_config(tmp_path)


def test_unknown_top_level_key_rejected(project) -> None:
    text = (project.root / "khorek.toml").read_text(encoding="utf-8")
    (project.root / "khorek.toml").write_text(text + "\nbogus = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="bogus"):
        load_config(project.root)


def test_unknown_nested_key_rejected(project) -> None:
    text = (project.root / "khorek.toml").read_text(encoding="utf-8")
    (project.root / "khorek.toml").write_text(
        text.replace("max_handoffs = 3", "max_handoffs = 3\nnope = true"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="nope"):
        load_config(project.root)


@pytest.mark.parametrize(
    ("aspects", "message"),
    [
        (
            [
                {"id": "", "prompt": "x", "files": ["src/"]},
            ],
            "must not be empty",
        ),
        (
            [
                {"id": "a", "prompt": "x", "files": ["src/"]},
                {"id": "a", "prompt": "y", "files": ["src/"]},
            ],
            "duplicate aspect id",
        ),
        (
            [
                {"id": "a", "prompt": "   ", "files": ["src/"]},
            ],
            "non-empty prompt",
        ),
        (
            [
                {"id": "a", "prompt": "x", "files": []},
            ],
            "files",
        ),
    ],
)
def test_invalid_aspects_rejected(
    tmp_path: Path, aspects: list[dict[str, object]], message: str
) -> None:
    project = make_project(tmp_path / "project", aspects=aspects)
    with pytest.raises(ConfigError, match=message):
        load_config(project.root)


def test_aspect_cap_above_default_rejected(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        defaults={"max_issues_per_file": 2},
        aspects=[
            {
                "id": "comments",
                "prompt": "x",
                "files": ["src/"],
                "max_issues_per_file": 3,
            }
        ],
    )
    with pytest.raises(ConfigError, match="may not exceed"):
        load_config(project.root)


def test_aspect_cap_may_lower_default(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        defaults={"max_issues_per_file": 3},
        aspects=[
            {
                "id": "comments",
                "prompt": "x",
                "files": ["src/"],
                "max_issues_per_file": 1,
            }
        ],
    )
    config = load_config(project.root)
    assert config.aspects[0].max_issues_per_file == 1


def test_empty_verify_list_allowed(project) -> None:
    assert load_config(project.root).verify == []


def test_verify_commands_load(project) -> None:
    write_config(
        project.root,
        verify=[{"cmd": "true", "timeout": 5}, {"cmd": "false"}],
    )
    config = load_config(project.root)
    assert [command.cmd for command in config.verify] == ["true", "false"]
    assert config.verify[1].timeout == 120.0


def test_missing_llm_section_rejected(tmp_path: Path) -> None:
    (tmp_path / "khorek.toml").write_text("[[aspects]]\nid = 'a'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="llm"):
        load_config(tmp_path)


def test_env_overrides_win_over_file(project, monkeypatch) -> None:
    monkeypatch.setenv("KHOREK_LLM_BASE_URL", "http://env.test/v1")
    monkeypatch.setenv("KHOREK_LLM_MODEL", "env-model")
    monkeypatch.setenv("KHOREK_LLM_TEMPERATURE", "0.75")
    monkeypatch.setenv("KHOREK_LLM_CALL_TIMEOUT", "42")
    config = load_config(project.root)
    assert config.llm.base_url == "http://env.test/v1"
    assert config.llm.model == "env-model"
    assert config.llm.temperature == 0.75
    assert config.llm.call_timeout == 42.0


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("KHOREK_LLM_TEMPERATURE", "hot", "must be a number"),
        ("KHOREK_LLM_CALL_TIMEOUT", "soon", "must be a number"),
    ],
)
def test_invalid_env_overrides_raise(
    project, monkeypatch, name: str, value: str, message: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match=message):
        load_config(project.root)


def test_env_temperature_out_of_range_raises(project, monkeypatch) -> None:
    monkeypatch.setenv("KHOREK_LLM_TEMPERATURE", "9")
    with pytest.raises(ConfigError, match="temperature"):
        load_config(project.root)


def test_fixer_defaults_and_overrides(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    config = load_config(project.root)
    assert config.agents.fixer.max_attempts == 3
    assert config.agents.fixer.max_tool_calls == 20

    write_config(
        project.root,
        agents={"fixer": {"max_attempts": 2, "max_tool_calls": 5}},
    )
    config = load_config(project.root)
    assert config.agents.fixer.max_attempts == 2
    assert config.agents.fixer.max_tool_calls == 5


def test_per_agent_reasoning_effort_override_inherits_otherwise(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        llm={"reasoning_effort": "medium"},
        agents={"aspect": {"prompt": "", "reasoning_effort": "none"}},
    )
    config = load_config(project.root)
    assert config.llm.reasoning_effort == "medium"
    assert config.agents.aspect.reasoning_effort == "none"
    assert config.agents.explorer.reasoning_effort is None
    assert config.agents.fixer.reasoning_effort is None


def test_llm_max_tokens_and_reasoning_effort_env(tmp_path: Path, monkeypatch) -> None:
    project = make_project(tmp_path / "project")
    monkeypatch.setenv("KHOREK_LLM_MAX_TOKENS", "1234")
    monkeypatch.setenv("KHOREK_LLM_REASONING_EFFORT", "low")
    config = load_config(project.root)
    assert config.llm.max_tokens == 1234
    assert config.llm.reasoning_effort == "low"


def test_observe_defaults_to_disabled(project) -> None:
    config = load_config(project.root)
    assert config.observe.enabled is False
    assert config.observe.directory == ""
    assert config.observe.keep_runs == 30
