"""Tool handlers: dict results, self-truncation, and output-size caps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from khorek.agents.explorer import ExplorerAgent, _fit_answer, ask_explorer_tool
from khorek.agents.fixer import edit_file_tool, write_file_tool
from khorek.agents.tools import (
    MAX_FILE_LINES,
    MAX_GREP_MATCHES,
    _fit_grep_result,
    _fit_list_files_result,
    _fit_read_file_result,
    _grep_result,
    _list_files_result,
    _read_file_result,
    _truncated_grep_match,
    count_that_fits,
    grep_tool,
    list_files_tool,
    read_file_tool,
    read_only_tools,
)
from khorek.content import format_content_lines
from khorek.llm import Tool, ToolValidationError
from khorek.snapshots import Snapshotter


class _FakeExplorer:
    """Stand-in for ExplorerAgent: the tool only calls ``ask``."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.questions: list[str] = []

    def ask(self, question: str) -> str:
        self.questions.append(question)
        return self.answer


def _handler(tool: Tool) -> Any:
    assert tool.handler is not None
    return tool.handler


def _size(result: dict[str, Any]) -> int:
    """Size of a tool result exactly as the agent loop serializes it."""
    return len(json.dumps(result, ensure_ascii=False).encode("utf-8"))


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src/a.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    return root


def test_every_tool_handler_returns_a_dict_with_truncated(tmp_path: Path) -> None:
    root = _project(tmp_path)
    snapshotter = Snapshotter(root)
    explorer = cast(ExplorerAgent, _FakeExplorer("an answer"))
    cases: list[tuple[Tool, dict[str, Any]]] = [
        (read_file_tool(root, 200_000, 16_000), {"path": "src/a.py"}),
        (grep_tool(root, 200_000, 16_000), {"pattern": "two"}),
        (list_files_tool(root, 16_000), {}),
        (
            edit_file_tool(root, snapshotter, 16_000),
            {"path": "src/a.py", "search": "two", "replace": "TWO"},
        ),
        (
            write_file_tool(root, snapshotter, 16_000),
            {"path": "src/b.py", "content": "x\n"},
        ),
        (ask_explorer_tool(explorer, 3, 16_000), {"question": "where is two?"}),
    ]

    for tool, arguments in cases:
        result = _handler(tool)(arguments)
        assert isinstance(result, dict), tool.name
        assert isinstance(result.get("truncated"), bool), tool.name

    assert {tool.name for tool, _ in cases} == {
        "read_file",
        "grep",
        "list_files",
        "edit_file",
        "write_file",
        "ask_explorer",
    }


def test_read_only_tools_threads_the_output_cap(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "src/many.py").write_text(
        "".join(f"line {number:03d} {'y' * 40}\n" for number in range(1, 41)),
        encoding="utf-8",
    )
    tools = {tool.name: tool for tool in read_only_tools(root, 200_000, 400)}

    read = _handler(tools["read_file"])({"path": "src/many.py"})
    assert _size(read) <= 400
    assert read["truncated"] is True
    assert read["remaining"] > 0
    assert read["line_end"] < read["total_lines"]

    found = _handler(tools["grep"])({"pattern": "line"})
    assert _size(found) <= 400
    assert found["truncated"] is True
    assert 0 < len(found["matches"]) < 40

    listing = _handler(tools["list_files"])({})
    assert _size(listing) <= 400
    assert listing["count"] == len(listing["files"])


def test_read_file_small_file_is_not_truncated(tmp_path: Path) -> None:
    root = _project(tmp_path)

    result = _handler(read_file_tool(root, 200_000, 16_000))({"path": "src/a.py"})

    assert result == {
        "path": "src/a.py",
        "line_start": 1,
        "line_end": 3,
        "total_lines": 3,
        "truncated": False,
        "remaining": 0,
        "content": "1 | one\n2 | two\n3 | three\n",
    }


def test_read_file_honours_offset_and_limit(tmp_path: Path) -> None:
    root = _project(tmp_path)

    result = _handler(read_file_tool(root, 200_000, 16_000))(
        {"path": "src/a.py", "offset": 2, "limit": 1}
    )

    assert result["line_start"] == 2
    assert result["line_end"] == 2
    assert result["remaining"] == 1
    assert result["truncated"] is True
    assert result["content"] == "2 | two\n"


def test_read_file_caps_at_max_file_lines(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    lines = "".join(f"line {number}\n" for number in range(1, MAX_FILE_LINES + 2))
    (root / "big.py").write_text(lines, encoding="utf-8")

    result = _handler(read_file_tool(root, 200_000, 10_000_000))({"path": "big.py"})

    assert result["total_lines"] == MAX_FILE_LINES + 1
    assert result["line_start"] == 1
    assert result["line_end"] == MAX_FILE_LINES
    assert result["remaining"] == 1
    assert result["truncated"] is True
    assert len(result["content"].splitlines()) == MAX_FILE_LINES


def test_read_file_serialized_result_fits_the_cap(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    lines = "".join(f"line {number:03d}\n" for number in range(1, 101))
    (root / "big.py").write_text(lines, encoding="utf-8")

    result = _handler(read_file_tool(root, 200_000, 400))({"path": "big.py"})

    assert _size(result) <= 400
    assert result["truncated"] is True
    assert result["remaining"] > 0
    assert result["line_end"] < result["total_lines"]
    assert len(result["content"].splitlines()) == result["line_end"]


def test_read_file_serialized_cap_counts_json_escaping(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    # Quotes and backslashes are escaped by JSON, so their serialized form is
    # wider than the raw line.  A raw-byte cap would overshoot here.
    (root / "escaped.py").write_text('"' * 500 + "\n" + "\\" * 500 + "\n", encoding="utf-8")

    result = _handler(read_file_tool(root, 200_000, 300))({"path": "escaped.py"})

    assert _size(result) <= 300
    assert result["truncated"] is True
    assert result["line_end"] == 1
    assert result["content"]
    # The visible content is only a fraction of the raw line: every quote in
    # the serialized result costs two characters.
    assert result["content"].count('"') < 200


def test_read_file_serialized_cap_counts_metadata(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    # The path and the surrounding metadata alone eat most of this cap.
    long_name = "nested/" + "a" * 120 + ".py"
    (root / "nested").mkdir()
    (root / long_name).write_text("x\n" * 50, encoding="utf-8")

    result = _handler(read_file_tool(root, 200_000, 260))({"path": long_name})

    assert _size(result) <= 260
    assert result["truncated"] is True


def test_read_file_truncates_an_overlong_single_line(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "long.py").write_text("x" * 500 + "\n", encoding="utf-8")

    result = _handler(read_file_tool(root, 200_000, 250))({"path": "long.py"})

    assert result["line_start"] == 1
    assert result["line_end"] == 1
    assert result["truncated"] is True
    assert _size(result) <= 250
    assert result["content"].startswith("1 | x")
    assert result["content"].count("x") < 500


def test_read_file_refuses_files_over_max_file_bytes(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "big.py").write_text("x" * 300, encoding="utf-8")

    with pytest.raises(ToolValidationError, match="max_file_bytes"):
        _handler(read_file_tool(root, 200, 16_000))({"path": "big.py"})


def test_grep_serialized_result_fits_the_cap(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "src/many.py").write_text(
        "".join(f"hit {number:03d} {'z' * 200}\n" for number in range(1, 21)),
        encoding="utf-8",
    )

    result = _handler(grep_tool(root, 200_000, 400))({"pattern": "hit"})

    assert _size(result) <= 400
    assert result["truncated"] is True
    assert 0 < len(result["matches"]) < 20


def test_grep_truncates_a_single_overlong_match_line(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "src/long.py").write_text("hit " + "z" * 5000 + "\n", encoding="utf-8")

    result = _handler(grep_tool(root, 200_000, 300))({"pattern": "hit"})

    assert _size(result) <= 300
    assert result["truncated"] is True
    assert len(result["matches"]) == 1
    match = result["matches"][0]
    assert match["path"] == "src/long.py"
    assert match["line"] == 1
    assert match["text"].startswith("hit z")
    assert len(match["text"]) < 5000


def test_grep_limit_argument_caps_matches(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "src/many.py").write_text("hit\nhit\nhit\nhit\nhit\n", encoding="utf-8")
    tool = grep_tool(root, 200_000, 16_000)

    limited = _handler(tool)({"pattern": "hit", "limit": 2})
    assert len(limited["matches"]) == 2
    assert limited["truncated"] is True

    complete = _handler(tool)({"pattern": "hit", "limit": 100})
    assert len(complete["matches"]) == 5
    assert complete["truncated"] is False


def test_grep_default_match_cap_is_100(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "src/many.py").write_text("hit\n" * 101, encoding="utf-8")

    result = _handler(grep_tool(root, 200_000, 16_000))({"pattern": "hit"})

    assert MAX_GREP_MATCHES == 100
    assert len(result["matches"]) == 100
    assert result["truncated"] is True


def test_grep_reports_not_truncated_when_everything_fits(tmp_path: Path) -> None:
    root = _project(tmp_path)

    result = _handler(grep_tool(root, 200_000, 16_000))({"pattern": "two"})

    assert len(result["matches"]) == 1
    assert result["truncated"] is False


def test_list_files_serialized_result_fits_the_cap(tmp_path: Path) -> None:
    root = _project(tmp_path)
    for number in range(20):
        name = f"file_{number:02d}_{'n' * 30}.py"
        (root / "src" / name).write_text("x\n", encoding="utf-8")
    total = len(list((root / "src").iterdir()))

    result = _handler(list_files_tool(root, 300))({})

    assert _size(result) <= 300
    assert result["truncated"] is True
    assert 0 < result["count"] < total
    assert result["count"] == len(result["files"])


def test_list_files_drops_a_boundary_path_rather_than_cutting_it(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / ("p" * 120 + ".py")).write_text("x\n", encoding="utf-8")

    result = _handler(list_files_tool(root, 150))({})

    assert _size(result) <= 150
    assert result["files"] == []
    assert result["count"] == 0
    assert result["truncated"] is True


def test_list_files_reports_truncated_when_files_exceed_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    (root / "src/b.py").write_text("b\n", encoding="utf-8")
    (root / "src/c.py").write_text("c\n", encoding="utf-8")
    monkeypatch.setattr("khorek.agents.tools.MAX_LISTED_FILES", 2)

    truncated = _handler(list_files_tool(root, 16_000))({})
    assert truncated["count"] == 2
    assert truncated["truncated"] is True

    complete = _handler(list_files_tool(root, 16_000))({"path": "src/a.py"})
    assert complete["truncated"] is False


def test_mutating_tools_report_not_truncated(tmp_path: Path) -> None:
    root = _project(tmp_path)
    snapshotter = Snapshotter(root)

    edited = _handler(edit_file_tool(root, snapshotter, 16_000))(
        {"path": "src/a.py", "search": "two", "replace": "TWO"}
    )
    written = _handler(write_file_tool(root, snapshotter, 16_000))(
        {"path": "src/b.py", "content": "x\n"}
    )

    assert edited["truncated"] is False
    assert written["truncated"] is False
    assert _size(edited) <= 16_000
    assert _size(written) <= 16_000


def test_edit_file_serialized_result_fits_the_cap(tmp_path: Path) -> None:
    root = _project(tmp_path)

    result = _handler(edit_file_tool(root, Snapshotter(root), 60))(
        {"path": "src/a.py", "search": "two", "replace": "TWO"}
    )

    assert _size(result) <= 60
    assert result == {"path": "src/a.py", "truncated": True}
    assert (root / "src/a.py").read_text(encoding="utf-8") == "one\nTWO\nthree\n"


def test_write_file_serialized_result_fits_the_cap(tmp_path: Path) -> None:
    root = _project(tmp_path)

    result = _handler(write_file_tool(root, Snapshotter(root), 60))(
        {"path": "src/b.py", "content": "x\n"}
    )

    assert _size(result) <= 60
    assert result == {"path": "src/b.py", "truncated": True}
    assert (root / "src/b.py").read_text(encoding="utf-8") == "x\n"


def test_ask_explorer_caps_the_serialized_result() -> None:
    explorer = _FakeExplorer("a" * 100)
    tool = ask_explorer_tool(cast(ExplorerAgent, explorer), 3, 40)

    result = _handler(tool)({"question": "why?"})

    assert _size(result) <= 40
    assert result["truncated"] is True
    assert 0 < len(result["answer"]) < 100
    assert result["answer"] == "a" * len(result["answer"])
    assert explorer.questions == ["why?"]


def test_ask_explorer_serialized_cap_counts_json_escaping() -> None:
    explorer = _FakeExplorer('"' * 100)
    tool = ask_explorer_tool(cast(ExplorerAgent, explorer), 3, 80)

    result = _handler(tool)({"question": "why?"})

    assert _size(result) <= 80
    assert result["truncated"] is True
    # Every quote in the serialized answer costs two characters, so far fewer
    # than 100 survive.
    assert result["answer"].count('"') < 100


def test_ask_explorer_leaves_a_short_answer_alone() -> None:
    explorer = _FakeExplorer("short answer")
    tool = ask_explorer_tool(cast(ExplorerAgent, explorer), 3, 200)

    result = _handler(tool)({"question": "why?"})

    assert result["answer"] == "short answer"
    assert result["truncated"] is False


def test_ask_explorer_handoff_limit_is_a_not_truncated_error() -> None:
    explorer = _FakeExplorer("never used")
    tool = ask_explorer_tool(cast(ExplorerAgent, explorer), 0, 200)

    result = _handler(tool)({"question": "why?"})

    assert "hand-off limit reached" in result["error"]
    assert result["truncated"] is False
    assert explorer.questions == []


def test_ask_explorer_handoff_limit_fits_the_cap() -> None:
    explorer = cast(ExplorerAgent, _FakeExplorer("never used"))

    def exhausted(max_bytes: int) -> dict[str, Any]:
        handler = _handler(ask_explorer_tool(explorer, 3, max_bytes))
        for _ in range(3):
            handler({"question": "why?"})
        return handler({"question": "why?"})

    short = exhausted(80)
    assert _size(short) <= 80
    assert short["error"] == "hand-off limit reached (3 per aspect run)"
    assert short["truncated"] is True

    minimal = exhausted(60)
    assert _size(minimal) <= 60
    assert minimal["error"] == "hand-off limit reached"
    assert minimal["truncated"] is True


def test_caps_below_the_result_envelope_return_the_minimal_result(tmp_path: Path) -> None:
    """Documented exception: a cap below the result envelope returns the minimal result."""
    root = _project(tmp_path)
    explorer = cast(ExplorerAgent, _FakeExplorer("an answer"))

    read = _handler(read_file_tool(root, 200_000, 1))({"path": "src/a.py"})
    assert read == {
        "path": "src/a.py",
        "line_start": 1,
        "line_end": 1,
        "total_lines": 3,
        "truncated": True,
        "remaining": 2,
        "content": "",
    }
    assert _size(read) > 1

    found = _handler(grep_tool(root, 200_000, 1))({"pattern": "o"})
    assert found == {"matches": [], "truncated": True}
    assert _size(found) > 1

    listing = _handler(list_files_tool(root, 1))({})
    assert listing == {"files": [], "count": 0, "truncated": True}
    assert _size(listing) > 1

    asked = _handler(ask_explorer_tool(explorer, 3, 1))({"question": "why?"})
    assert asked == {"answer": "", "truncated": True}
    assert _size(asked) > 1

    edited = _handler(edit_file_tool(root, Snapshotter(root), 1))(
        {"path": "src/a.py", "search": "two", "replace": "TWO"}
    )
    assert edited == {"truncated": True}
    assert _size(edited) > 1

    written = _handler(write_file_tool(root, Snapshotter(root), 1))(
        {"path": "src/b.py", "content": "x\n"}
    )
    assert written == {"truncated": True}
    assert _size(written) > 1

    limited = _handler(ask_explorer_tool(explorer, 0, 1))({"question": "why?"})
    assert limited == {"truncated": True}
    assert _size(limited) > 1


def test_count_that_fits_returns_candidates_when_all_fit() -> None:
    assert count_that_fits(5, lambda _count: True) == 5


def test_count_that_fits_returns_minus_one_when_none_fit() -> None:
    assert count_that_fits(5, lambda _count: False) == -1


def test_count_that_fits_returns_exact_fitting_boundary() -> None:
    assert count_that_fits(10, lambda count: count <= 4) == 4


def test_count_that_fits_handles_zero_candidates() -> None:
    assert count_that_fits(0, lambda count: count == 0) == 0


def test_read_file_line_window_uses_largest_fitting_count() -> None:
    relative = "src/long.py"
    text = "alpha\nbravo\ncharlie\ndelta\necho\n"
    lines = [str(line) for line in text.splitlines()]
    start, requested, total = 2, 4, 5

    def build(count: int) -> dict[str, Any]:
        end = start + count - 1
        content = format_content_lines(text, offset=start, limit=count)
        return _read_file_result(relative, total, start, end, content, end < total)

    cap = _size(build(2))
    result = _fit_read_file_result(relative, lines, text, start, requested, total, cap)
    chosen = result["line_end"] - start + 1

    assert chosen == 2
    assert _size(build(chosen)) <= cap
    assert chosen < requested
    assert _size(build(chosen + 1)) > cap


def test_read_file_single_line_fallback_uses_largest_fitting_count() -> None:
    relative = "src/long.py"
    line = "x" * 100
    prefix = "1 | "

    def build(characters: int) -> dict[str, Any]:
        return _read_file_result(relative, 1, 1, 1, prefix + line[:characters], True)

    cap = _size(build(8))
    result = _fit_read_file_result(relative, [line], f"{line}\n", 1, 1, 1, cap)
    chosen = len(result["content"]) - len(prefix)

    assert chosen == 8
    assert _size(build(chosen)) <= cap
    assert chosen < len(line)
    assert _size(build(chosen + 1)) > cap


def test_truncated_grep_match_uses_largest_fitting_count() -> None:
    kept = [{"path": "src/a.py", "line": 1, "text": "kept"}]
    match: dict[str, Any] = {"path": "src/b.py", "line": 2, "text": "z" * 100}

    def build(characters: int) -> dict[str, Any]:
        candidate = {
            "path": match["path"],
            "line": match["line"],
            "text": match["text"][:characters],
        }
        return _grep_result([*kept, candidate], True)

    cap = _size(build(8))
    result = _truncated_grep_match(match, kept, cap)

    assert result is not None
    chosen = len(result["text"])
    assert chosen == 8
    assert _size(build(chosen)) <= cap
    assert chosen < len(match["text"])
    assert _size(build(chosen + 1)) > cap


def test_grep_match_count_uses_largest_fitting_count() -> None:
    matches = [
        {"path": f"src/{index}.py", "line": index, "text": "hit " + "z" * 20}
        for index in range(1, 5)
    ]

    def build(count: int) -> dict[str, Any]:
        return _grep_result(matches[:count], True)

    cap = _size(build(2))
    result = _fit_grep_result(matches, True, cap)
    chosen = len(result["matches"])

    assert result["matches"] == matches[:chosen]
    assert chosen == 2
    assert _size(build(chosen)) <= cap
    assert chosen < len(matches)
    assert _size(build(chosen + 1)) > cap


def test_list_file_count_uses_largest_fitting_count() -> None:
    files = [f"src/{index:02d}-{'x' * 20}.py" for index in range(5)]

    def build(count: int) -> dict[str, Any]:
        return _list_files_result(files[:count], True)

    cap = _size(build(2))
    result = _fit_list_files_result(files, False, cap)
    chosen = result["count"]

    assert result["files"] == files[:chosen]
    assert chosen == 2
    assert _size(build(chosen)) <= cap
    assert chosen < len(files)
    assert _size(build(chosen + 1)) > cap


def test_explorer_answer_uses_largest_fitting_count() -> None:
    answer = "answer-" * 20

    def build(count: int) -> dict[str, Any]:
        return {"answer": answer[:count], "truncated": True}

    cap = _size(build(8))
    result = _fit_answer(answer, cap)
    chosen = len(result["answer"])

    assert chosen == 8
    assert _size(build(chosen)) <= cap
    assert chosen < len(answer)
    assert _size(build(chosen + 1)) > cap
