"""Default system prompts bundled with khorek."""

from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).parent


def default_prompt(name: str) -> str:
    """Load a bundled prompt file (``aspect``, ``explorer``, ``fixer``,
    ``verifier``, ``reviewer``)."""
    path = _PROMPT_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


__all__ = ["default_prompt"]
