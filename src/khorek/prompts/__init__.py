"""Default system prompts bundled with khorek.

Each agent's config section may override these with a non-empty ``prompt``.
"""

from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).parent


def default_prompt(name: str) -> str:
    """Load a bundled prompt file (``aspect``, ``explorer``, ``fixer``, ``verifier``)."""
    path = _PROMPT_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


__all__ = ["default_prompt"]
