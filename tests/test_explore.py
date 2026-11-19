"""Explore workflow: skip rules, validation, errors, limits, and drift."""

from __future__ import annotations

import json
from pathlib import Path

from khorek.agents.aspect import IssueReport, ReportedIssue, report_issues_parameters
from khorek.config import load_config
from khorek.context import AppContext
from khorek.db import Database
from khorek.files import sha256_file
from khorek.llm import LLMClient
from khorek.report import Reporter
from khorek.workflows.explore import line_count, run_explore, validate_issues
from tests.conftest import (
    FakeModel,
    Project,
    explore_args,
    make_project,
    run_cli,
    seed_issue,
    text_message,
    tool_message,
)


def report(*issues: dict[str, object]) -> dict[str, object]:
    return tool_message("report_issues", {"issues": list(issues)})


def issue(
    description: str = "the comment is inconsistent",
    line_start: int = 1,
    line_end: int = 1,
    severity: str = "high",
) -> dict[str, object]:
    return {
        "description": description,
        "line_start": line_start,
        "line_end": line_end,
        "severity": severity,
    }


def state(project: Project, aspect: str = "comments", path: str = "src/a.py"):
    with Database(project.db_path, project.root) as db:
        return db.get_state(aspect, path)


def issues(project: Project):
    with Database(project.db_path, project.root) as db:
        return db.list_issues()


def test_clean_first_run_then_skip_on_clean(project, capsys) -> None:
    fake = FakeModel([report()])
    assert run_cli(fake, *explore_args(project)) == 0
    row = state(project)
    assert row is not None
    assert row.outcome == "clean"
    assert fake.chat_calls == 1

    capsys.readouterr()
    assert run_cli(fake, *explore_args(project)) == 0
    assert fake.chat_calls == 1
    output = capsys.readouterr().out
    assert "skipped (clean)" in output


def test_skip_on_open_issues_does_not_duplicate(project) -> None:
    fake = FakeModel([report(issue("bad comment", 2, 2, "high"))])
    assert run_cli(fake, *explore_args(project)) == 0
    assert len(issues(project)) == 1

    assert run_cli(fake, *explore_args(project)) == 0
    assert fake.chat_calls == 1
    assert len(issues(project)) == 1


def test_re_explores_when_state_row_is_absent(project) -> None:
    fake = FakeModel([report(issue("bad comment", 2, 2, "high"))])
    assert run_cli(fake, *explore_args(project)) == 0
    assert len(issues(project)) == 1

    # Deleting the state row (the pair's version token) makes the pair re-explore
    # even though an open issue exists: an absent token means "not yet validated".
    with Database(project.db_path, project.root) as db:
        db.delete_state("comments", "src/a.py")

    fake.add(report(issue("bad comment", 2, 2, "high")))
    assert run_cli(fake, *explore_args(project)) == 0
    assert fake.chat_calls == 2
    assert len(issues(project)) == 1


def test_hash_change_re_explores_and_replaces_old_issue(project) -> None:
    fake = FakeModel([report(issue("bad comment", 2, 2, "high"))])
    assert run_cli(fake, *explore_args(project)) == 0
    assert len(issues(project)) == 1

    project.write("src/a.py", "# changed\nvalue = 2\n")
    fake.add(report())
    assert run_cli(fake, *explore_args(project)) == 0
    assert fake.chat_calls == 2

    remaining = issues(project)
    assert remaining == []
    row = state(project)
    assert row is not None
    assert row.outcome == "clean"
    assert row.sha256 == sha256_file(project.root / "src/a.py")


def test_stale_state_and_issues_are_deleted_before_exploring(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    with Database(project.db_path, project.root) as db:
        db.upsert_state("comments", "src/a.py", "stale-hash", "issues")
        db.insert_issue(
            aspect_id="comments",
            path="src/a.py",
            description="stale issue",
            line_start=1,
            line_end=1,
            severity="low",
        )

    fake = FakeModel([report()])
    assert run_cli(fake, *explore_args(project)) == 0
    assert fake.chat_calls == 1

    assert issues(project) == []
    row = state(project)
    assert row is not None
    assert row.sha256 == sha256_file(project.root / "src/a.py")
    assert row.outcome == "clean"


def test_cap_truncation_produces_truncated_outcome(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "one\ntwo\nthree\n"},
        defaults={"max_issues_per_file": 1},
    )
    fake = FakeModel([report(issue("first", 1, 1, "high"), issue("second", 2, 2, "high"))])
    assert run_cli(fake, *explore_args(project)) == 0
    assert fake.chat_calls == 1

    row = state(project)
    assert row is not None
    assert row.outcome == "truncated"
    stored = issues(project)
    assert [entry.description for entry in stored] == ["first"]
    assert "truncated" in capsys.readouterr().out


def test_aspect_cap_override_truncates(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "one\ntwo\n"},
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
    fake = FakeModel([report(issue("first", 1, 1), issue("second", 2, 2))])
    assert run_cli(fake, *explore_args(project)) == 0
    row = state(project)
    assert row is not None
    assert row.outcome == "truncated"
    assert len(issues(project)) == 1
    assert "1 issues created" in capsys.readouterr().out


def test_line_range_and_severity_repaired(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project", files={"src/a.py": "one\ntwo\n"})
    fake = FakeModel([report(issue("  broken comment  ", -4, 99, "critical"))])
    assert run_cli(fake, *explore_args(project)) == 0

    stored = issues(project)
    assert len(stored) == 1
    assert stored[0].description == "broken comment"
    assert (stored[0].line_start, stored[0].line_end) == (1, 2)
    assert stored[0].severity == "low"


def test_empty_descriptions_are_dropped(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    fake = FakeModel([report(issue("   ", 1, 1, "low"), issue("real issue", 1, 1, "medium"))])
    assert run_cli(fake, *explore_args(project)) == 0

    stored = issues(project)
    assert [entry.description for entry in stored] == ["real issue"]
    row = state(project)
    assert row is not None
    assert row.outcome == "issues"


def test_validate_issues_clamps_drops_and_trims() -> None:
    report_model = IssueReport(
        issues=[
            ReportedIssue(
                description="   ",
                line_start=1,
                line_end=1,
                severity="low",
            ),
            ReportedIssue(
                description="  out of range  ",
                line_start=-5,
                line_end=999,
                severity="high",
            ),
            ReportedIssue(
                description="over the cap",
                line_start=2,
                line_end=2,
                severity="medium",
            ),
        ]
    )
    kept, truncated = validate_issues(report_model, lines=line_count("one\ntwo\n"), cap=1)
    assert truncated is True
    assert len(kept) == 1
    assert kept[0].description == "out of range"
    assert (kept[0].line_start, kept[0].line_end) == (1, 2)
    assert kept[0].severity == "high"


def test_agent_loop_abort_writes_nothing_and_exits_1(project, capsys) -> None:
    fake = FakeModel(
        [
            tool_message("read_file", {"path": "src/a.py"}),
            tool_message("read_file", {"path": "src/a.py"}),
        ]
    )
    assert run_cli(fake, *explore_args(project)) == 1
    assert fake.chat_calls == 2
    assert state(project) is None
    assert issues(project) == []
    output = capsys.readouterr().out
    assert "error [loop]" in output
    assert "outcome: errored" in output


def test_no_tool_call_nudge_then_abort(project, capsys) -> None:
    fake = FakeModel([text_message("first answer"), text_message("second answer")])
    assert run_cli(fake, *explore_args(project)) == 1
    assert fake.chat_calls == 2
    assert state(project) is None
    output = capsys.readouterr().out
    assert "error [no_tool_call]" in output
    assert "second answer" in output
    # The nudge really was appended to the conversation.
    reminder = fake.chat_requests[1]["messages"][-1]
    assert reminder["role"] == "user"
    assert "REMINDER" in reminder["content"]


def test_prompt_budget_abort_writes_nothing(tmp_path: Path, capsys) -> None:
    project = make_project(tmp_path / "project", defaults={"max_prompt_bytes": 64})
    fake = FakeModel([])
    assert run_cli(fake, *explore_args(project)) == 1
    assert fake.chat_calls == 0
    assert state(project) is None
    output = capsys.readouterr().out
    assert "error [prompt_budget]" in output


def test_prune_drift_reports_and_removes_old_aspect(project, capsys) -> None:
    with Database(project.db_path, project.root) as db:
        db.upsert_state("retired", "src/a.py", "sha", "issues")
        db.insert_issue(
            aspect_id="retired",
            path="src/a.py",
            description="old",
            line_start=1,
            line_end=1,
            severity="low",
        )

    fake = FakeModel([report()])
    assert run_cli(fake, *explore_args(project)) == 0
    with Database(project.db_path, project.root) as db:
        assert db.list_issues() == []
        assert [row.aspect_id for row in db.list_state()] == ["comments"]
    assert "2 pruned" in capsys.readouterr().out


def _two_aspects() -> list[dict[str, object]]:
    return [
        {
            "id": "comments",
            "prompt": "Find comments that are inconsistent with the code.",
            "files": ["src/"],
        },
        {"id": "style", "prompt": "Find style issues.", "files": ["src/"]},
    ]


def test_run_explore_aspect_filter_skips_other_aspects(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "# a correct comment\nvalue = 1\n"},
        aspects=_two_aspects(),
    )
    fake = FakeModel([report()])
    config = load_config(project.root)
    with Database(project.db_path, project.root) as database:
        llm = LLMClient(config.llm, transport=fake.transport)
        try:
            app = AppContext(
                project_root=project.root,
                config=config,
                database=database,
                llm=llm,
                reporter=Reporter(),
            )
            stats = run_explore(app, aspect_ids=["comments"])
        finally:
            llm.close()

        assert stats.considered == 1
        assert fake.chat_calls == 1
        # The second aspect was never enumerated or explored.
        assert [row.aspect_id for row in database.list_state()] == ["comments"]


def test_explore_aspect_filter_counts_only_that_aspect_in_summary(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "# a correct comment\nvalue = 1\n"},
        aspects=_two_aspects(),
    )
    seed_issue(project, aspect_id="style", path="src/b.py", description="style issue")
    fake = FakeModel([report(issue("bad comment", 1, 1, "high"))])

    assert run_cli(fake, *explore_args(project, "--aspect", "comments")) == 0

    with Database(project.db_path, project.root) as database:
        assert database.count_issues(aspect_ids=["comments"]) == 1
        assert database.count_issues(aspect_ids=["style"]) == 1
        assert database.get_state("style", "src/a.py") is None
    assert "1 open issues" in capsys.readouterr().out


def test_limit_stops_after_pair(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "one\n", "src/b.py": "two\n"},
    )
    fake = FakeModel([report(issue("a issue", 1, 1)), report(issue("b issue", 1, 1))])
    assert run_cli(fake, *explore_args(project, "--limit", "1")) == 0
    assert fake.chat_calls == 1

    with Database(project.db_path, project.root) as db:
        assert db.count_issues() == 1
        assert db.get_state("comments", "src/a.py") is not None
        assert db.get_state("comments", "src/b.py") is None
    output = capsys.readouterr().out
    assert "2 pairs" in output
    assert "1 explored" in output


def test_explorer_handoff_counts_as_separate_agent_run(project) -> None:
    fake = FakeModel(
        [
            tool_message("ask_explorer", {"question": "where is value used?"}),
            text_message("value is used in src/a.py line 2"),
            report(),
        ]
    )
    assert run_cli(fake, *explore_args(project)) == 0
    assert fake.chat_calls == 3
    assert fake.chat_requests[1]["tool_choice"] == "auto"
    assert "value is used in src/a.py line 2" in " ".join(fake.tool_results(2))
    row = state(project)
    assert row is not None
    assert row.outcome == "clean"


def test_one_pair_error_does_not_abort_the_run(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "one\n", "src/b.py": "two\n"},
    )
    fake = FakeModel(
        [
            tool_message("read_file", {"path": "src/a.py"}),
            tool_message("read_file", {"path": "src/a.py"}),
            report(),
        ]
    )
    assert run_cli(fake, *explore_args(project)) == 1
    assert fake.chat_calls == 3
    with Database(project.db_path, project.root) as db:
        assert db.get_state("comments", "src/a.py") is None
        second = db.get_state("comments", "src/b.py")
        assert second is not None
        assert second.outcome == "clean"
    output = capsys.readouterr().out
    assert "1 errored" in output
    assert "1 explored" in output


def test_limit_is_cumulative_across_pairs(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={
            "src/a.py": "one\n",
            "src/b.py": "two\n",
            "src/c.py": "three\n",
        },
    )
    fake = FakeModel(
        [
            report(issue("a issue", 1, 1)),
            report(issue("b issue", 1, 1)),
            report(issue("c issue", 1, 1)),
        ]
    )
    assert run_cli(fake, *explore_args(project, "--limit", "2")) == 0
    assert fake.chat_calls == 2
    with Database(project.db_path, project.root) as db:
        assert db.count_issues() == 2
        assert db.get_state("comments", "src/a.py") is not None
        assert db.get_state("comments", "src/b.py") is not None
        assert db.get_state("comments", "src/c.py") is None
    assert "2 issues created" in capsys.readouterr().out


def test_report_issues_schema_is_flat_and_requires_all_fields() -> None:
    schema = report_issues_parameters()
    rendered = json.dumps(schema)
    assert "$defs" not in schema
    assert "$ref" not in rendered
    assert schema["required"] == ["issues"]
    item_schema = schema["properties"]["issues"]["items"]
    assert set(item_schema["required"]) == {
        "description",
        "line_start",
        "line_end",
        "severity",
    }
    assert item_schema["properties"]["severity"]["enum"] == ["low", "medium", "high"]


def test_empty_terminal_arguments_get_repaired(project) -> None:
    # Observed with the real Ollama model against a nested Pydantic schema: it
    # called report_issues with "{}". The flattened schema plus required fields
    # turns that into one repair round instead of a bogus clean result.
    fake = FakeModel(
        [
            tool_message("report_issues", {}),
            report(issue("comment contradicts the code", 1, 1, "high")),
        ]
    )
    assert run_cli(fake, *explore_args(project)) == 0
    assert fake.chat_calls == 2
    stored = issues(project)
    assert len(stored) == 1
    assert "invalid_arguments" in fake.tool_results(1)[0]
    row = state(project)
    assert row is not None
    assert row.outcome == "issues"


def test_default_aspect_prompt_explains_one_based_lines(project) -> None:
    fake = FakeModel([report()])
    assert run_cli(fake, *explore_args(project)) == 0
    system_prompt = fake.chat_requests[0]["messages"][0]["content"]
    assert "1-based" in system_prompt
    assert "report_issues" in system_prompt


def test_limit_caps_issues_created_within_one_pair(tmp_path: Path, capsys) -> None:
    project = make_project(
        tmp_path / "project",
        files={"src/a.py": "one\ntwo\nthree\n"},
    )
    fake = FakeModel(
        [
            report(
                issue("first", 1, 1, "high"),
                issue("second", 2, 2, "high"),
                issue("third", 3, 3, "high"),
            )
        ]
    )
    assert run_cli(fake, *explore_args(project, "--limit", "1")) == 0
    assert fake.chat_calls == 1

    with Database(project.db_path, project.root) as db:
        stored = db.list_issues()
        assert [entry.description for entry in stored] == ["first"]
        row = db.get_state("comments", "src/a.py")
        assert row is not None
        assert row.outcome == "truncated"
    output = capsys.readouterr().out
    assert "1 issues created" in output
    assert "(truncated)" in output
