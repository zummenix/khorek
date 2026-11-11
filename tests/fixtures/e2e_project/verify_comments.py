#!/usr/bin/env python3
"""Deterministic comment/code consistency checker for the khorek e2e fixture.

Run from the fixture root (``python3 verify_comments.py``).  For every function
that adds or subtracts, the comments attached to it (the contiguous comment
block above the ``def``, comments inside the body, and a docstring) must state
the same direction as the operator actually used.  Wording is free: a set of
add/increment/plus/... and subtract/decrement/minus/... words plus the bare
``+`` / ``-`` symbols are recognised, and the first direction token found in
the comment is the one that counts.

Exit codes: 0 when every arithmetic function is consistent, 1 otherwise (the
message names each contradiction so the Fixer can act on it).
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

SOURCE_PATH = Path(__file__).with_name("math_utils.py")

ADDITION = "addition"
SUBTRACTION = "subtraction"
_OPERATOR = {ADDITION: "+", SUBTRACTION: "-"}

_ADDITION_WORDS = frozenset(
    {
        "add",
        "adds",
        "added",
        "adding",
        "addition",
        "additions",
        "increment",
        "increments",
        "incremented",
        "incrementing",
        "increase",
        "increases",
        "increased",
        "increasing",
        "plus",
        "more",
        "greater",
        "larger",
        "up",
        "sum",
        "sums",
        "summed",
        "summing",
        "total",
    }
)
_SUBTRACTION_WORDS = frozenset(
    {
        "subtract",
        "subtracts",
        "subtracted",
        "subtracting",
        "subtraction",
        "subtractions",
        "decrement",
        "decrements",
        "decremented",
        "decrementing",
        "minus",
        "less",
        "fewer",
        "reduce",
        "reduces",
        "reduced",
        "reducing",
        "reduction",
        "decrease",
        "decreases",
        "decreased",
        "decreasing",
        "smaller",
        "lower",
        "lowers",
        "lowered",
        "lowering",
        "below",
        "down",
        "drop",
        "drops",
        "dropped",
        "dropping",
        "difference",
        "differences",
    }
)

# Words and operator symbols, in source order; the first direction wins.
_DIRECTION_TOKEN = re.compile(r"[a-z]+|[+-]")


def direction_of_word(word: str) -> str | None:
    """Classify one comment word as addition or subtraction."""
    if word in _ADDITION_WORDS:
        return ADDITION
    if word in _SUBTRACTION_WORDS:
        return SUBTRACTION
    return None


def direction_tokens(text: str) -> list[str]:
    """Directions stated by *text*, in the order they appear."""
    found: list[str] = []
    for match in _DIRECTION_TOKEN.finditer(text.lower()):
        token = match.group(0)
        if token == "+":
            found.append(ADDITION)
        elif token == "-":
            found.append(SUBTRACTION)
        else:
            direction = direction_of_word(token)
            if direction is not None:
                found.append(direction)
    return found


def statement_directions(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Directions of the ``+``/``-`` operators used inside *function*."""
    directions: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                directions.add(ADDITION)
            elif isinstance(node.op, ast.Sub):
                directions.add(SUBTRACTION)
    return directions


def comments_by_line(source: str) -> dict[int, str]:
    """Map 1-based line numbers to the text of ``#`` comment tokens."""
    comments: dict[int, str] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            comments[token.start[0]] = token.string
    return comments


def attached_comment_text(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    comments: dict[int, str],
) -> str:
    """Comment/docstring text that documents *function*."""
    fragments: list[str] = []

    # Contiguous comment block above the def (blank lines are skipped so a
    # comment separated by one blank line still counts).
    line = function.lineno - 1
    while line >= 1:
        if line in comments:
            fragments.append(comments[line])
            line -= 1
            continue
        if lines[line - 1].strip() == "":
            line -= 1
            continue
        break

    # Comments inside the function body (including trailing inline comments).
    end = function.end_lineno or function.lineno
    for number in range(function.lineno, end + 1):
        if number in comments:
            fragments.append(comments[number])

    # A docstring replacing the comment must be just as acceptable.
    if function.body:
        first = function.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                fragments.append(first.value.value)

    return "\n".join(fragments)


def _describe(direction: str) -> str:
    return f"{direction} ({_OPERATOR[direction]})"


def _snippet(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def check() -> list[str]:
    """Return one problem string per inconsistent function (empty if clean)."""
    try:
        source = SOURCE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"cannot read {SOURCE_PATH.name}: {exc}"]

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{SOURCE_PATH.name} is not valid Python: {exc}"]

    lines = source.splitlines()
    comments = comments_by_line(source)
    problems: list[str] = []
    checked = 0

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        code = statement_directions(node)
        if not code:
            continue
        checked += 1
        name = node.name
        if len(code) > 1:
            problems.append(
                f"{name} (line {node.lineno}): the code mixes '+' and '-', "
                "so the comment cannot be checked"
            )
            continue

        code_direction = next(iter(code))
        text = attached_comment_text(node, lines, comments)
        stated = direction_tokens(text)
        if not stated:
            problems.append(
                f"{name} (line {node.lineno}): no comment says whether it "
                f"{'adds' if code_direction == ADDITION else 'subtracts'} "
                f"({_OPERATOR[code_direction]}); comment: {_snippet(text) or '<none>'!r}"
            )
            continue
        if stated[0] != code_direction:
            problems.append(
                f"{name} (line {node.lineno}): the comment says "
                f"{_describe(stated[0])} but the code uses "
                f"{_describe(code_direction)}; comment: {_snippet(text)!r}"
            )

    if checked == 0:
        problems.append("no arithmetic function with '+' or '-' was found")
    return problems


def main() -> int:
    problems = check()
    if problems:
        print("comment/code consistency check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("comment/code consistency check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
