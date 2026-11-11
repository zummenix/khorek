"""Formatting helpers for presenting file contents to an LLM."""

from __future__ import annotations

from pathlib import Path

from khorek.content import format_file_lines


def _write(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _rule(header: str) -> str:
    """Build the dashed rule that wraps an excerpt."""
    return "-" * len(header)


def _expected(path: str, start: int, end: int, lines: list[str], total: int) -> str:
    """Build the expected excerpt for *path*, lines *start*..*end*."""
    header = f"File: {path} (Lines {start}-{end} of {total})"
    body = "\n".join(
        f"{index:>{len(str(end))}} | {line}" for index, line in enumerate(lines, start=start)
    )
    return f"{header}\n{_rule(header)}\n{body}\n{_rule(header)}\n"


def test_full_file_with_no_offset_or_limit(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "line one\nline two\nline three\n")

    result = format_file_lines(
        relative_file_path="a.py", content="line one\nline two\nline three\n"
    )

    expected = _expected("a.py", 1, 3, ["line one", "line two", "line three"], 3)
    assert result == expected


def test_offset_and_limit(tmp_path: Path) -> None:
    content = "one\ntwo\nthree\nfour\nfive\nsix\nseven\neighteen\n"
    _write(tmp_path, "a.py", content)

    result = format_file_lines(relative_file_path="a.py", content=content, offset=3, limit=3)

    expected = _expected("a.py", 3, 5, ["three", "four", "five"], 8)
    expected += "[Truncated: 5 more lines remaining. Use offset=6 to read further.]\n"
    assert result == expected


def test_line_numbers_are_right_aligned_to_the_widest_number_in_window(tmp_path: Path) -> None:
    content = "\n".join(str(i) for i in range(1, 11)) + "\n"
    _write(tmp_path, "a.py", content)

    result = format_file_lines("a.py", content, 3, 8)

    # The window ends on line 10, so every number is right-aligned to a width
    # of 2 and the " | " separator lands on the same column.
    body = result.splitlines()[2 : 2 + 8]
    for index, line in enumerate(body, start=3):
        assert line.startswith(f"{index:>{len(str(10))}} | ")
        assert " | " in line


def test_many_digit_line_numbers_widen_the_padding(tmp_path: Path) -> None:
    lines = "\n".join(str(i) for i in range(1, 1001)) + "\n"
    _write(tmp_path, "big.py", lines)

    # A window that ends on line 1000 pads to a width of 4.
    result = format_file_lines("big.py", lines, 991, 10)

    assert " 991 | 991" in result
    assert " 992 | 992" in result
    assert " 999 | 999" in result
    assert "1000 | 1000" in result
    # The " | " separator must land on the same column for every line.
    for line in result.splitlines()[2:12]:
        assert line.index(" | ") == 4


def test_truncated_footer_counts_remaining_lines(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "one\ntwo\nthree\nfour\nfive\n")

    result = format_file_lines("a.py", "one\ntwo\nthree\nfour\nfive\n", 2, 2)

    assert "(Lines 2-3 of 5)" in result
    assert "[Truncated: 3 more lines remaining. Use offset=4 to read further.]" in result


def test_singular_remaining_line_uses_singular_word(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "one\ntwo\nthree\n")

    result = format_file_lines("a.py", "one\ntwo\nthree\n", 1, 1)

    assert "[Truncated: 2 more lines remaining. Use offset=2 to read further.]" in result
    assert "[Truncated: 1 more line remaining." not in result


def test_offset_past_end_of_file_clamps_to_last_line(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "one\ntwo\nthree\n")

    result = format_file_lines("a.py", "one\ntwo\nthree\n", 10)

    expected = _expected("a.py", 3, 3, ["three"], 3)
    expected += "[Truncated: 2 more lines remaining. Use offset=4 to read further.]\n"
    assert result == expected


def test_limit_zero_shows_every_line_from_offset(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "one\ntwo\nthree\n")

    result = format_file_lines("a.py", "one\ntwo\nthree\n", None, 0)

    assert "(Lines 1-3 of 3)" in result
    assert "[Truncated:" not in result


def test_empty_file_renders_as_a_blank_line(tmp_path: Path) -> None:
    _write(tmp_path, "empty.py", "")

    result = format_file_lines(relative_file_path="empty.py", content="")

    expected = _expected("empty.py", 1, 1, [""], 1)
    assert result == expected


def test_absolute_path_is_reported_as_given(tmp_path: Path) -> None:
    """Absolute paths are not normalized; they are reported as supplied."""
    root = tmp_path / "root"
    _write(root, "src/a.py", "one\ntwo\n")
    absolute = str(root / "src/a.py")
    result = format_file_lines(relative_file_path=absolute, content="one\ntwo\n")
    assert f"File: {absolute} (Lines 1-2 of 2)" in result


def test_relative_path_is_reported_as_given(tmp_path: Path) -> None:
    _write(tmp_path, "src/a.py", "one\ntwo\n")
    result = format_file_lines(relative_file_path="src/a.py", content="one\ntwo\n")
    assert "File: src/a.py (Lines 1-2 of 2)" in result


def test_no_trailing_newline_is_treated_as_a_line(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "one\ntwo")

    result = format_file_lines("a.py", "one\ntwo")

    assert "(Lines 1-2 of 2)" in result
    assert "2 | two" in result


def test_non_ascii_content_is_preserved(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "café\n日本語\n")

    result = format_file_lines("a.py", "café\n日本語\n")

    assert "1 | café" in result
    assert "2 | 日本語" in result


def test_limit_smaller_than_requested_range(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "one\ntwo\nthree\nfour\n")

    result = format_file_lines("a.py", "one\ntwo\nthree\nfour\n", 2, 1)

    expected = _expected("a.py", 2, 2, ["two"], 4)
    expected += "[Truncated: 3 more lines remaining. Use offset=3 to read further.]\n"
    assert result == expected
