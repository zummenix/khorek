"""Shared parsing helpers for agent tool arguments."""

from __future__ import annotations

from typing import Any

from khorek.llm import ToolValidationError


def string_argument(arguments: dict[str, Any], name: str, *, default: str | None = None) -> str:
    value = arguments.get(name, default)
    if not isinstance(value, str):
        raise ToolValidationError(f"argument {name!r} must be a string")
    return value


def optional_string_argument(arguments: dict[str, Any], name: str) -> str | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolValidationError(f"argument {name!r} must be a string when given")
    return value


def optional_int_argument(arguments: dict[str, Any], name: str) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolValidationError(f"argument {name!r} must be an integer when given")
    return value


__all__ = ["optional_int_argument", "optional_string_argument", "string_argument"]
