"""Tests for the khorek CLI scaffold."""

import pytest

from khorek.cli import main


def test_no_arguments_prints_help_and_succeeds(capsys) -> None:
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "usage: khorek" in captured.out
    assert "explore" in captured.out


def test_unimplemented_command_returns_2(capsys) -> None:
    assert main(["explore"]) == 2
    captured = capsys.readouterr()
    assert "not implemented yet" in captured.err


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
