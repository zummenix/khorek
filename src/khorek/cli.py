"""Argparse-based command-line interface for khorek."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from khorek.config import Config, ConfigError, load_config
from khorek.context import AppContext
from khorek.db import Database, DatabaseError, default_db_path
from khorek.llm import LLMClient, ModelUnavailable
from khorek.lock import LockError, LockHeld, acquire
from khorek.report import Reporter
from khorek.workflows.explore import run_explore

# Exit codes are fixed by the prototype plan.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_FATAL = 2

_NOT_IMPLEMENTED = "khorek {command}: not implemented yet"


def _add_workflow_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by all workflow commands."""
    parser.add_argument(
        "-p",
        "--path",
        default=".",
        help="project root to operate on (default: current directory)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="maximum number of issues to process (0 = unlimited)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to khorek.toml (default: <project>/khorek.toml)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="database file to use (default: ~/.khorek/<slug>.db)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="khorek",
        description="Scout a codebase for issues and fix them.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    explore = subparsers.add_parser(
        "explore",
        help="explore files and record issues",
        description="Explore files and record issues in the database.",
    )
    _add_workflow_arguments(explore)

    fix = subparsers.add_parser(
        "fix",
        help="fix open issues",
        description="Fix open issues one at a time, verifying each fix.",
    )
    _add_workflow_arguments(fix)

    explore_and_fix = subparsers.add_parser(
        "explore-and-fix",
        help="fix open issues, then explore and fix again",
        description="Fix open issues, explore for new ones, and fix those too.",
    )
    _add_workflow_arguments(explore_and_fix)

    reset = subparsers.add_parser(
        "reset",
        help="delete the database",
        description="Delete the database and its lock file.",
    )
    reset.add_argument(
        "-p",
        "--path",
        default=".",
        help="project root whose database should be removed (default: current directory)",
    )
    reset.add_argument(
        "--db",
        default=None,
        help="database file to delete (default: ~/.khorek/<slug>.db)",
    )

    return parser


def _fail(message: str) -> int:
    print(f"khorek: {message}", file=sys.stderr)
    return EXIT_FATAL


def _resolve_project(raw: str) -> Path | None:
    try:
        path = Path(raw).expanduser().resolve()
    except OSError as exc:
        print(f"khorek: cannot resolve project path {raw!r}: {exc}", file=sys.stderr)
        return None
    if not path.exists():
        print(f"khorek: project path does not exist: {path}", file=sys.stderr)
        return None
    if not path.is_dir():
        print(f"khorek: project path is not a directory: {path}", file=sys.stderr)
        return None
    return path


def _resolve_db_path(project_root: Path, db_argument: str | None) -> Path:
    if db_argument is not None:
        return Path(db_argument).expanduser().resolve()
    return default_db_path(project_root)


def _resolve_config_path(config_argument: str | None) -> Path | None:
    if config_argument is None:
        return None
    return Path(config_argument).expanduser().resolve()


def _load_config_or_report(project_root: Path, args: argparse.Namespace) -> Config | None:
    config_path = _resolve_config_path(getattr(args, "config", None))
    try:
        return load_config(project_root, config_path)
    except ConfigError as exc:
        print(f"khorek: {exc}", file=sys.stderr)
        return None


def _run_explore(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> int:
    project_root = _resolve_project(args.path)
    if project_root is None:
        return EXIT_FATAL

    config = _load_config_or_report(project_root, args)
    if config is None:
        return EXIT_FATAL

    llm = LLMClient(
        config.llm,
        transport=transport,
        max_prompt_bytes=config.defaults.max_prompt_bytes,
        max_tool_output_bytes=config.defaults.max_tool_output_bytes,
    )
    try:
        try:
            llm.preflight()
        except ModelUnavailable as exc:
            return _fail(str(exc))

        db_path = _resolve_db_path(project_root, args.db)
        try:
            with acquire(db_path):
                try:
                    with Database(db_path, project_root) as database:
                        reporter = Reporter()
                        pruned = database.prune_drift([aspect.id for aspect in config.aspects])
                        app = AppContext(
                            project_root=project_root,
                            config=config,
                            database=database,
                            llm=llm,
                            reporter=reporter,
                        )
                        stats = run_explore(app, limit=args.limit)
                        reporter.summary(
                            stats,
                            pruned=pruned,
                            open_issues=database.count_issues(),
                        )
                        return EXIT_ERROR if stats.errored else EXIT_OK
                except DatabaseError as exc:
                    return _fail(str(exc))
        except LockHeld as exc:
            return _fail(str(exc))
        except LockError as exc:
            return _fail(str(exc))
    finally:
        llm.close()


def _run_reset(args: argparse.Namespace) -> int:
    project_root = _resolve_project(args.path)
    if project_root is None:
        return EXIT_FATAL

    db_path = _resolve_db_path(project_root, args.db)
    targets = [
        db_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    ]
    try:
        with acquire(db_path) as run_lock:
            for target in targets:
                try:
                    target.unlink(missing_ok=True)
                except OSError as exc:
                    return _fail(f"cannot remove {target}: {exc}")
            run_lock.unlink_file()
    except LockHeld as exc:
        return _fail(str(exc))
    except LockError as exc:
        return _fail(str(exc))

    print(db_path)
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
) -> int:
    """Run the CLI and return the process exit code.

    ``transport`` is the in-process test seam: tests inject an
    ``httpx.MockTransport`` and never touch a real model.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_OK

    limit = getattr(args, "limit", 0)
    if isinstance(limit, int) and limit < 0:
        print("khorek: --limit must be >= 0 (0 = unlimited)", file=sys.stderr)
        return EXIT_FATAL

    if args.command == "explore":
        return _run_explore(args, transport)
    if args.command == "reset":
        return _run_reset(args)

    print(_NOT_IMPLEMENTED.format(command=args.command), file=sys.stderr)
    return EXIT_FATAL


if __name__ == "__main__":
    raise SystemExit(main())
