"""CLI wiring for the ``khorek init`` setup wizard."""

from __future__ import annotations

from pathlib import Path

from khorek.cli import build_parser, main
from khorek.config import AspectConfig, VerifyCommand, load_config
from khorek.init import Answers, InitAnswers, render_init_config, resolve_config_name
from tests.conftest import FakeModel


class ScriptedAnswers:
    """A prompt seam returning scripted answers in order and recording questions."""

    def __init__(self, *answers: str) -> None:
        self._queue = list(answers)
        self.asked: list[str] = []

    def __call__(self, question: str) -> str:
        self.asked.append(question)
        if not self._queue:
            raise AssertionError("scripted init answers exhausted")
        return self._queue.pop(0)


BASE_URL_QUESTION = (
    "Base URL of the OpenAI-compatible endpoint [default: http://127.0.0.1:11434/v1]:"
)
MODEL_QUESTION = "Model name: "
VERIFY_QUESTION = (
    "Verify command (a non-mutating check run after each fix, e.g. 'ruff format --check .'):"
)
ASPECT_QUESTION = "Add a code-review aspect? [y/N]:"
FILES_QUESTION = (
    "Files scope for aspect 'comments' (comma-separated paths or globs, empty for all):"
)


def _run_init(tmp_path: Path, answers: Answers, *extra: str) -> int:
    fake = FakeModel([])
    return main(["init", "-p", str(tmp_path), *extra], transport=fake.transport, answers=answers)


def test_init_happy_path_writes_canonical_config(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()
    answers = ScriptedAnswers(
        "http://localhost:11434/v1",
        "qwen3.8:27b-mxfp8",
        "ruff format --check .",
        "y",
        "",
    )

    assert _run_init(root, answers) == 0

    expected = """[llm]
base_url = "http://localhost:11434/v1"
model = "qwen3.8:27b-mxfp8"
# temperature = 0.2
# call_timeout = 600.0
# max_tokens = 8192
# reasoning_effort

[defaults]
max_issues_per_file = 3
max_file_bytes = 200000
max_prompt_bytes = 400000
max_tool_output_bytes = 16000
max_cycles = 25

[observe]
enabled = false
directory = ""
keep_runs = 30

[agents.aspect]
max_handoffs = 3
max_tool_calls = 20

[agents.explorer]
max_tool_calls = 20

[agents.fixer]
max_attempts = 3
max_tool_calls = 20

[agents.reviewer]
max_handoffs = 2
max_tool_calls = 20

[[verify]]
cmd = "ruff format --check ."
# timeout = 120.0

[[aspects]]
id = "comments"
prompt = "Find comments that are inconsistent with the code."
files = ["*"]
# max_issues_per_file = 3
"""
    assert (root / "khorek.toml").read_text(encoding="utf-8") == expected
    assert "khorek: wrote" in capsys.readouterr().out

    config = load_config(root)
    assert config.llm.base_url == "http://localhost:11434/v1"
    assert config.llm.model == "qwen3.8:27b-mxfp8"
    assert config.defaults.max_issues_per_file == 3
    assert config.defaults.max_file_bytes == 200_000
    assert config.defaults.max_prompt_bytes == 400_000
    assert config.defaults.max_tool_output_bytes == 16_000
    assert config.defaults.max_cycles == 25
    assert config.observe.enabled is False
    assert config.observe.directory == ""
    assert config.observe.keep_runs == 30
    assert config.agents.aspect.max_handoffs == 3
    assert config.agents.aspect.max_tool_calls == 20
    assert config.agents.explorer.max_tool_calls == 20
    assert config.agents.fixer.max_attempts == 3
    assert config.agents.fixer.max_tool_calls == 20
    assert config.agents.reviewer.max_handoffs == 2
    assert config.agents.reviewer.max_tool_calls == 20
    assert config.verify == [VerifyCommand(cmd="ruff format --check .", timeout=120.0)]
    assert config.aspects == [
        AspectConfig(
            id="comments", prompt="Find comments that are inconsistent with the code.", files=["*"]
        ),
    ]


def test_rendered_init_config_round_trips(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("KHOREK_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("KHOREK_LLM_MODEL", raising=False)
    destination = tmp_path / "project"
    destination.mkdir()
    answers = InitAnswers(
        base_url="http://localhost:11434/v1",
        model="qwen3.8:27b-mxfp8",
        verify_cmd="ruff format --check .",
        aspect=None,
    )
    (destination / "khorek.toml").write_text(render_init_config(answers), encoding="utf-8")

    config = load_config(destination)

    assert config.llm.base_url == answers.base_url
    assert config.llm.model == answers.model
    assert config.verify[0].cmd == answers.verify_cmd


def test_init_asks_the_expected_prompts_in_order(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    answers = ScriptedAnswers(
        "http://localhost:11434/v1", "qwen3.8:27b-mxfp8", "ruff format --check .", "n"
    )

    assert _run_init(root, answers) == 0
    assert answers.asked == [BASE_URL_QUESTION, MODEL_QUESTION, VERIFY_QUESTION, ASPECT_QUESTION]


def test_init_refuses_to_overwrite_existing_config(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()
    original = '[llm]\nmodel = "m"\n'
    (root / "khorek.toml").write_text(original, encoding="utf-8")

    assert _run_init(root, ScriptedAnswers("a", "b", "c", "n")) == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert (root / "khorek.toml").read_text(encoding="utf-8") == original


def test_init_abort_via_keyboard_interrupt_writes_nothing(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()

    def abort(question: str) -> str:
        raise KeyboardInterrupt

    assert _run_init(root, abort) == 1
    assert not (root / "khorek.toml").exists()
    assert "init aborted" in capsys.readouterr().err


def test_init_abort_via_eof_writes_nothing(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()

    def abort(question: str) -> str:
        raise EOFError

    assert _run_init(root, abort) == 1
    assert not (root / "khorek.toml").exists()
    assert "init aborted" in capsys.readouterr().err


def test_init_empty_model_is_fatal_and_writes_nothing(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()

    assert (
        _run_init(
            root, ScriptedAnswers("http://localhost:11434/v1", "", "ruff format --check .", "n")
        )
        == 2
    )
    assert not (root / "khorek.toml").exists()
    assert "model is required" in capsys.readouterr().err


def test_init_empty_verify_command_is_fatal_and_writes_nothing(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()

    assert _run_init(root, ScriptedAnswers("http://localhost:11434/v1", "m", "", "n")) == 2
    assert not (root / "khorek.toml").exists()
    assert "at least one" in capsys.readouterr().err


def test_init_opt_out_writes_no_aspects(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    answers = ScriptedAnswers(
        "http://localhost:11434/v1", "qwen3.8:27b-mxfp8", "ruff format --check .", "n"
    )

    assert _run_init(root, answers) == 0
    assert "[[aspects]]" not in (root / "khorek.toml").read_text(encoding="utf-8")
    assert load_config(root).aspects == []


def test_init_custom_files_scope_is_written(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    answers = ScriptedAnswers(
        "http://localhost:11434/v1",
        "qwen3.8:27b-mxfp8",
        "ruff format --check .",
        "y",
        "src/, lib/*.py",
    )

    assert _run_init(root, answers) == 0
    assert 'files = ["src/", "lib/*.py"]' in (root / "khorek.toml").read_text(encoding="utf-8")


def test_init_escapes_toml_special_characters(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    answers = ScriptedAnswers("", 'q"wen\\3', "true", "n")

    assert _run_init(root, answers) == 0
    assert 'model = "q\\"wen\\\\3"' in (root / "khorek.toml").read_text(encoding="utf-8")


def test_init_rejects_config_name_with_path_separator(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()
    assert _run_init(root, ScriptedAnswers("a", "b", "c", "n"), "--config-name", "sub/khorek") == 2
    assert "invalid --config-name" in capsys.readouterr().err
    assert list(root.iterdir()) == []


def test_init_writes_into_nested_destination(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "deep"
    answers = ScriptedAnswers(
        "http://localhost:11434/v1", "qwen3.8:27b-mxfp8", "ruff format --check .", "n"
    )

    assert _run_init(root, answers) == 0
    assert (root / "khorek.toml").exists()


def test_init_path_pointing_at_regular_file_is_fatal(tmp_path: Path, capsys) -> None:
    file_path = tmp_path / "file.toml"
    file_path.write_text("", encoding="utf-8")
    assert _run_init(file_path, ScriptedAnswers("a", "b", "c", "n")) == 2
    assert "not a directory" in capsys.readouterr().err


def test_init_config_name_suffix_resolution() -> None:
    assert resolve_config_name("khorek") == "khorek.toml"
    assert resolve_config_name("khorek.local") == "khorek.local.toml"
    assert resolve_config_name("khorek.toml") == "khorek.toml"


def test_init_parser_defaults_and_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["init"])
    assert args.path == "."
    assert args.config_name == "khorek"
    args = parser.parse_args(["init", "-p", "/tmp/project", "--config-name", "khorek.local"])
    assert args.path == "/tmp/project"
    assert args.config_name == "khorek.local"
