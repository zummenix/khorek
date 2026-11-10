"""The Phase 4 e2e fixture's deterministic verifier behaves as advertised."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "e2e_project"
MATH_UTILS = FIXTURE / "math_utils.py"

CORRECT_COMMENTS = [
    ("# Adds b to a.", "# Subtracts b from a."),
    ("# Returns a plus b.", "# Returns a minus b."),
    ("# Increments a by b.", "# Decrements a by b."),
    ("# Returns the sum of a and b.", "# Returns the difference of a and b."),
    ("# Returns a + b.", "# Returns a - b."),
]


def copy_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "e2e_project"
    shutil.copytree(FIXTURE, root)
    return root


def run_checker(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "verify_comments.py"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )


def rewrite(root: Path, *, add_comment: str, sub_comment: str) -> None:
    text = (root / "math_utils.py").read_text(encoding="utf-8")
    text = text.replace(
        "# Subtracts b from a.\ndef add(a, b):",
        f"{add_comment}\ndef add(a, b):",
    )
    text = text.replace(
        "# Adds b to a.\ndef subtract(a, b):",
        f"{sub_comment}\ndef subtract(a, b):",
    )
    (root / "math_utils.py").write_text(text, encoding="utf-8")


def test_committed_fixture_fails_the_checker(tmp_path: Path) -> None:
    result = run_checker(copy_fixture(tmp_path))

    assert result.returncode == 1
    assert "FAILED" in result.stderr
    assert "add (line" in result.stderr
    assert "subtract (line" in result.stderr


@pytest.mark.parametrize(("add_comment", "sub_comment"), CORRECT_COMMENTS)
def test_checker_accepts_reworded_correct_comments(
    tmp_path: Path, add_comment: str, sub_comment: str
) -> None:
    root = copy_fixture(tmp_path)
    rewrite(root, add_comment=add_comment, sub_comment=sub_comment)

    result = run_checker(root)

    assert result.returncode == 0, result.stderr
    assert "passed" in result.stdout


def test_checker_still_fails_when_one_comment_is_unfixed(tmp_path: Path) -> None:
    root = copy_fixture(tmp_path)
    rewrite(root, add_comment="# Adds b to a.", sub_comment="# Adds b to a.")

    result = run_checker(root)

    assert result.returncode == 1
    assert "subtract (line" in result.stderr


def test_checker_accepts_docstrings_instead_of_comments(tmp_path: Path) -> None:
    root = copy_fixture(tmp_path)
    text = (root / "math_utils.py").read_text(encoding="utf-8")
    text = text.replace(
        "# Subtracts b from a.\ndef add(a, b):\n    return a + b",
        'def add(a, b):\n    """Adds b to a."""\n    return a + b',
    )
    text = text.replace(
        "# Adds b to a.\ndef subtract(a, b):\n    return a - b",
        'def subtract(a, b):\n    """Subtracts b from a."""\n    return a - b',
    )
    (root / "math_utils.py").write_text(text, encoding="utf-8")

    result = run_checker(root)

    assert result.returncode == 0, result.stderr


def test_checker_rejects_missing_comments(tmp_path: Path) -> None:
    root = copy_fixture(tmp_path)
    text = (root / "math_utils.py").read_text(encoding="utf-8")
    text = text.replace("# Subtracts b from a.\n", "").replace("# Adds b to a.\n", "")
    (root / "math_utils.py").write_text(text, encoding="utf-8")

    result = run_checker(root)

    assert result.returncode == 1
    assert "no comment" in result.stderr
