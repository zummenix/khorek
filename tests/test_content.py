"""Tests for khorek.content formatting helpers."""

from __future__ import annotations

from khorek.content import (
    format_content_lines,
    format_file_lines,
    indent_block,
    line_count,
)


def _rule(header: str) -> str:
    """Build the dashed rule that wraps an excerpt."""
    return "-" * len(header)


def _expected_full(path: str, lines: list[str], total: int) -> str:
    """Build the expected full-file excerpt for *path*."""
    header = f"File: {path} (Lines 1-{total} of {total})"
    body = "\n".join(
        f"{index:>{len(str(total))}} | {line}" for index, line in enumerate(lines, start=1)
    )
    return f"{header}\n{_rule(header)}\n{body}\n{_rule(header)}\n"


def test_full_file_with_no_offset_or_limit() -> None:
    result = format_file_lines(
        relative_file_path="a.py", content="line one\nline two\nline three\n"
    )

    expected = _expected_full("a.py", ["line one", "line two", "line three"], 3)
    assert result == expected


def test_absolute_path_is_reported_as_given() -> None:
    """Absolute paths are not normalized; they are reported as supplied."""
    absolute = "/some/root/src/a.py"
    result = format_file_lines(relative_file_path=absolute, content="one\ntwo\n")
    assert f"File: {absolute} (Lines 1-2 of 2)" in result


def test_relative_path_is_reported_as_given() -> None:
    result = format_file_lines(relative_file_path="src/a.py", content="one\ntwo\n")
    assert "File: src/a.py (Lines 1-2 of 2)" in result


def test_no_trailing_newline_is_treated_as_a_line() -> None:
    result = format_file_lines("a.py", "one\ntwo")

    assert "(Lines 1-2 of 2)" in result
    assert "2 | two" in result


def test_non_ascii_content_is_preserved() -> None:
    result = format_file_lines("a.py", "café\n日本語\n")

    assert "1 | café" in result
    assert "2 | 日本語" in result


def test_empty_file_renders_as_a_blank_line() -> None:
    result = format_file_lines(relative_file_path="empty.py", content="")

    expected = _expected_full("empty.py", [""], 1)
    assert result == expected


def test_content_lines_offset_and_limit() -> None:
    content = "one\ntwo\nthree\nfour\nfive\nsix\nseven\neighteen\n"

    result = format_content_lines(content, offset=3, limit=3)

    assert result == "3 | three\n4 | four\n5 | five\n"


def test_content_lines_are_right_aligned_to_the_widest_number_in_window() -> None:
    content = "\n".join(str(i) for i in range(1, 11)) + "\n"

    result = format_content_lines(content, offset=3, limit=8)

    # The window ends on line 10, so every number is right-aligned to a width
    # of 2 and the " | " separator lands on the same column.
    body = result.splitlines()
    for index, line in enumerate(body, start=3):
        assert line.startswith(f"{index:>{len(str(10))}} | ")
        assert " | " in line


def test_many_digit_line_numbers_widen_the_padding() -> None:
    lines = "\n".join(str(i) for i in range(1, 1001)) + "\n"

    # A window that ends on line 1000 pads to a width of 4.
    result = format_content_lines(lines, offset=991, limit=10)

    assert " 991 | 991" in result
    assert " 992 | 992" in result
    assert " 999 | 999" in result
    assert "1000 | 1000" in result
    # The " | " separator must land on the same column for every line.
    for line in result.splitlines():
        assert line.index(" | ") == 4


def test_content_lines_offset_past_end_of_file_clamps_to_last_line() -> None:
    result = format_content_lines("one\ntwo\nthree\n", offset=10)

    assert result == "3 | three\n"


def test_content_lines_limit_zero_shows_every_line_from_offset() -> None:
    result = format_content_lines("one\ntwo\nthree\n", offset=None, limit=0)

    assert result == "1 | one\n2 | two\n3 | three\n"


def test_content_lines_limit_smaller_than_requested_range() -> None:
    result = format_content_lines("one\ntwo\nthree\nfour\n", offset=2, limit=1)

    assert result == "2 | two\n"


def test_content_lines_empty_file_renders_a_blank_numbered_line() -> None:
    assert format_content_lines("") == "1 | \n"


def test_line_count_empty_text_is_one_line() -> None:
    assert line_count("") == 1


def test_line_count_trailing_newline_does_not_add_a_line() -> None:
    assert line_count("one\ntwo\n") == 2


def test_indent_block_keeps_blank_lines_unindented() -> None:
    assert indent_block("first\n\nthird") == ["    first", "", "    third"]


def test_indent_block_strips_leading_whitespace() -> None:
    assert indent_block("  first\n\tsecond") == ["    first", "    second"]
