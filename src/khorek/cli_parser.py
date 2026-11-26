"""Argparse-based command-line parser for khorek.

Kept separate from the execution side so the parser can evolve independently of the
run logic; the execution side consumes it as a black box via ``build_parser()``.
"""

from __future__ import annotations

import argparse


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
    fix.add_argument(
        "--issue-id",
        dest="issue_ids",
        action="append",
        type=int,
        metavar="ID",
        help="only this issue id (repeatable, OR between values)",
    )

    explore_and_fix = subparsers.add_parser(
        "explore-and-fix",
        help="fix open issues, then explore and fix again",
        description="Fix open issues, explore for new ones, and fix those too.",
    )
    _add_workflow_arguments(explore_and_fix)

    traces = subparsers.add_parser(
        "traces",
        help="browse recorded model-call traces in a browser",
        description="Serve a local web UI over the run traces recorded by khorek.",
    )
    traces.add_argument(
        "--project-root",
        default=".",
        help="project root whose traces should be shown (default: current directory)",
    )
    traces.add_argument(
        "--db",
        default=None,
        help="database whose traces should be shown (default: ~/.khorek/<slug>.db)",
    )
    traces.add_argument(
        "--trace-dir",
        default=None,
        help="trace directory to serve (overrides the default <db>.traces)",
    )
    traces.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    traces.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    traces.add_argument(
        "--no-open",
        action="store_true",
        help="do not open a browser window",
    )
    traces.add_argument("--verbose", action="store_true", help="log HTTP requests")
    traces.add_argument(
        "--stop",
        action="store_true",
        help="stop a trace server left running by a workflow",
    )

    reset = subparsers.add_parser(
        "reset",
        help="delete the database",
        description="Delete the database and its lock file.",
    )
    reset.add_argument(
        "--project-root",
        default=".",
        help="project root whose database should be removed (default: current directory)",
    )
    reset.add_argument(
        "--db",
        default=None,
        help="database file to delete (default: ~/.khorek/<slug>.db)",
    )

    init = subparsers.add_parser(
        "init",
        help="write a canonical khorek.toml interactively",
        description="Interactively gather the basic settings and dump a canonical "
        "khorek.toml into the destination directory.",
    )
    init.add_argument(
        "--project-root",
        default=".",
        help="destination directory for the dumped config (default: current directory)",
    )
    init.add_argument(
        "--config-name",
        default="khorek",
        help="name of the dumped config file; a .toml suffix is added if missing (default: khorek)",
    )

    _build_issues_parser(subparsers)

    return parser


def _add_workflow_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by all workflow commands."""
    parser.add_argument(
        "--project-root",
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
        "--aspect",
        dest="aspects",
        action="append",
        metavar="ID",
        help="only this aspect id (repeatable, OR between values)",
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
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="record full model-call traces (default: [observe].enabled)",
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="directory for run traces (default: <db>.traces)",
    )


def _add_issues_root_arguments(parser: argparse.ArgumentParser, *, suppressed: bool) -> None:
    """Add the project-root and --db arguments shared by the issues commands.

    ``suppressed`` marks a subparser whose values would otherwise clobber
    arguments already parsed by the parent ``issues`` parser (argparse
    applies subparser defaults over parent values).
    """
    parser.add_argument(
        "--project-root",
        default=argparse.SUPPRESS if suppressed else ".",
        help="project root to operate on (default: current directory)",
    )
    parser.add_argument(
        "--db",
        default=argparse.SUPPRESS if suppressed else None,
        help="database file to use (default: ~/.khorek/<slug>.db)",
    )


def _add_list_arguments(parser: argparse.ArgumentParser, *, suppressed: bool) -> None:
    """Add the repeatable list filters (OR within a flag, AND across flags)."""
    default = argparse.SUPPRESS if suppressed else []
    parser.add_argument(
        "--aspect",
        dest="aspects",
        action="append",
        default=default,
        metavar="ID",
        help="only issues from this aspect id (repeatable, OR between values)",
    )
    parser.add_argument(
        "--severity",
        dest="severities",
        action="append",
        default=default,
        choices=("low", "medium", "high"),
        metavar="SEVERITY",
        help="only issues with this severity (repeatable, OR between values)",
    )
    parser.add_argument(
        "-p",
        "--path",
        dest="path_prefixes",
        action="append",
        default=default,
        metavar="PREFIX",
        help="only issues under this path prefix (repeatable, OR between values)",
    )
    parser.add_argument(
        "--stale",
        action="store_true",
        default=argparse.SUPPRESS if suppressed else False,
        help="show stale issues, labeled, without pruning (lockless read)",
    )


def _build_issues_parser(subparsers: argparse._SubParsersAction) -> None:
    """Build the ``issues`` subcommand group (list is the default).

    The bare ``issues`` parser carries the list filters so ``khorek issues
    --path src/`` works without the explicit ``list`` subcommand.  The project
    root is ``--project-root`` on every issues command; ``-p``/``--path`` are
    the path-prefix filter (``list`` and the bare ``issues`` parser).
    """
    issues = subparsers.add_parser(
        "issues",
        help="list, show, dismiss, or delete recorded issues",
        description="Triage the recorded issues of a project (no LLM calls).",
    )
    _add_issues_root_arguments(issues, suppressed=False)
    _add_list_arguments(issues, suppressed=False)

    issue_sub = issues.add_subparsers(dest="issues_command", metavar="SUBCOMMAND")

    list_parser = issue_sub.add_parser(
        "list",
        help="list open issues (the default subcommand)",
        description="List open issues; stale ones are pruned unless the run "
        "lock is held or --stale is given.",
    )
    _add_issues_root_arguments(list_parser, suppressed=True)
    _add_list_arguments(list_parser, suppressed=True)

    show_parser = issue_sub.add_parser(
        "show",
        help="show one issue in full detail",
        description="Show one issue in full detail (lockless read; never prunes).",
    )
    _add_issues_root_arguments(show_parser, suppressed=True)
    show_parser.add_argument("id", type=int, metavar="ID", help="issue id to show")

    dismiss_parser = issue_sub.add_parser(
        "dismiss",
        help="dismiss issues as not relevant",
        description="Dismiss issue ids as not relevant; a drained pair is "
        "marked clean so the next explore skips it.",
    )
    _add_issues_root_arguments(dismiss_parser, suppressed=True)
    dismiss_parser.add_argument(
        "ids",
        nargs="+",
        type=int,
        metavar="ID",
        help="issue ids to dismiss",
    )

    delete_parser = issue_sub.add_parser(
        "delete",
        help="delete issues and their pair state",
        description="Delete issue ids completely so the next explore "
        "re-examines the file from scratch.",
    )
    _add_issues_root_arguments(delete_parser, suppressed=True)
    delete_parser.add_argument(
        "ids",
        nargs="+",
        type=int,
        metavar="ID",
        help="issue ids to delete",
    )
