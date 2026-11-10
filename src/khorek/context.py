"""Shared run context passed to workflows.

Phases 3-4 build on the same seam: a resolved project root, validated config,
open database, configured LLM client, and reporter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from khorek.config import Config
from khorek.db import Database
from khorek.llm import LLMClient
from khorek.report import Reporter
from khorek.trace import TraceRecorder


@dataclass
class AppContext:
    """Everything a workflow needs, assembled by the CLI preamble."""

    project_root: Path
    config: Config
    database: Database
    llm: LLMClient
    reporter: Reporter
    tracer: TraceRecorder | None = None


__all__ = ["AppContext"]
