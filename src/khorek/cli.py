"""Argparse-based command-line interface for khorek."""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import httpx

from khorek.config import Config, ConfigError, load_config
from khorek.context import AppContext
from khorek.db import Database, DatabaseError, default_db_path
from khorek.init import Answers, InitError, collect_answers, init_config_path, render_init_config
from khorek.issues import (
    IssueOutcome,
    dismiss,
    issue_status,
    prune_stale,
    render_list,
    render_show,
    staleness,
)
from khorek.issues import delete as delete_issues
from khorek.llm import LLMClient, ModelUnavailable
from khorek.lock import LockError, LockHeld, acquire
from khorek.report import Reporter
from khorek.trace import TraceRecorder, trace_dir_for
from khorek.trace_server import ServerInfo, start_viewer, stop_viewer
from khorek.trace_server import serve as serve_traces
from khorek.workflows.explore import run_explore
from khorek.workflows.explore_and_fix import run_explore_and_fix
from khorek.workflows.fix import run_fix

# Exit codes are fixed: 0 for success, 1 for a workflow error, 2 for a fatal error.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_FATAL = 2

# A workflow body, executed with the open DB and lock: (app, limit, pruned) -> exit code.
WorkflowBody = Callable[[AppContext, int, int], int]


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
        "-p",
        "--path",
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

    init = subparsers.add_parser(
        "init",
        help="write a canonical khorek.toml interactively",
        description="Interactively gather the basic settings and dump a canonical "
        "khorek.toml into the destination directory.",
    )
    init.add_argument(
        "-p",
        "--path",
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


def _add_issues_root_arguments(
    parser: argparse.ArgumentParser, *, path_long: bool, suppressed: bool
) -> None:
    """Add the project-root and --db arguments shared by the issues commands.

    ``suppressed`` marks a subparser whose values would otherwise clobber
    arguments already parsed by the parent ``issues`` parser (argparse
    applies subparser defaults over parent values).
    """
    parser.add_argument(
        *("-p", "--path") if path_long else ("-p",),
        dest="path",
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
    --path src/`` works without the explicit ``list`` subcommand.  On ``list``
    the long form ``--path`` is the prefix *filter*; the project root is
    ``-p`` there.  On ``show``/``dismiss``/``delete`` the long form
    ``--path`` is the project root.
    """
    issues = subparsers.add_parser(
        "issues",
        help="list, show, dismiss, or delete recorded issues",
        description="Triage the recorded issues of a project (no LLM calls).",
    )
    _add_issues_root_arguments(issues, path_long=False, suppressed=False)
    _add_list_arguments(issues, suppressed=False)

    issue_sub = issues.add_subparsers(dest="issues_command", metavar="SUBCOMMAND")

    list_parser = issue_sub.add_parser(
        "list",
        help="list open issues (the default subcommand)",
        description="List open issues; stale ones are pruned unless the run "
        "lock is held or --stale is given.",
    )
    _add_issues_root_arguments(list_parser, path_long=False, suppressed=True)
    _add_list_arguments(list_parser, suppressed=True)

    show_parser = issue_sub.add_parser(
        "show",
        help="show one issue in full detail",
        description="Show one issue in full detail (lockless read; never prunes).",
    )
    _add_issues_root_arguments(show_parser, path_long=True, suppressed=True)
    show_parser.add_argument("id", type=int, metavar="ID", help="issue id to show")

    dismiss_parser = issue_sub.add_parser(
        "dismiss",
        help="dismiss issues as not relevant",
        description="Dismiss issue ids as not relevant; a drained pair is "
        "marked clean so the next explore skips it.",
    )
    _add_issues_root_arguments(dismiss_parser, path_long=True, suppressed=True)
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
    _add_issues_root_arguments(delete_parser, path_long=True, suppressed=True)
    delete_parser.add_argument(
        "ids",
        nargs="+",
        type=int,
        metavar="ID",
        help="issue ids to delete",
    )


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


@dataclass
class RunSetup:
    project_root: Path
    config: Config
    db_path: Path
    llm: LLMClient
    tracer: TraceRecorder | None = None


def _prepare_run(
    args: argparse.Namespace,
    *,
    command: str,
    transport: httpx.BaseTransport | None,
    require_verify: bool,
) -> RunSetup | int:
    """Resolve and validate workflow inputs, then prepare model tracing."""
    project_root = _resolve_project(args.path)
    if project_root is None:
        return EXIT_FATAL

    config = _load_config_or_report(project_root, args)
    if config is None:
        return EXIT_FATAL
    if require_verify and not config.verify:
        print(
            f"khorek: {command} requires at least one [[verify]] command in the config",
            file=sys.stderr,
        )
        return EXIT_FATAL

    requested_aspects = getattr(args, "aspects", None) or []
    if requested_aspects:
        known = {aspect.id for aspect in config.aspects}
        for aspect_id in requested_aspects:
            if aspect_id not in known:
                known_list = ", ".join(aspect.id for aspect in config.aspects)
                print(
                    f"khorek: unknown aspect: {aspect_id} (known: {known_list})",
                    file=sys.stderr,
                )
                return EXIT_FATAL

    db_path = _resolve_db_path(project_root, args.db)
    trace_enabled = getattr(args, "trace", None)
    if getattr(args, "trace_dir", None) is not None:
        trace_enabled = True
    tracer = TraceRecorder.from_config(
        config.observe,
        db_path=db_path,
        run={
            "command": command,
            "project_root": str(project_root),
            "model": config.llm.model,
            "db_path": str(db_path),
            "argv": list(sys.argv[1:]),
        },
        directory=getattr(args, "trace_dir", None),
        enabled=trace_enabled,
    )
    if tracer is not None:
        tracer.record(
            "run_start",
            command=command,
            project_root=str(project_root),
            model=config.llm.model,
            db_path=str(db_path),
            config={
                "temperature": config.llm.temperature,
                "max_tokens": config.llm.max_tokens,
                "reasoning_effort": config.llm.reasoning_effort,
                "reasoning_effort_overrides": {
                    "aspect": config.agents.aspect.reasoning_effort,
                    "explorer": config.agents.explorer.reasoning_effort,
                    "fixer": config.agents.fixer.reasoning_effort,
                    "verifier": config.agents.verifier.reasoning_effort,
                },
                "max_prompt_bytes": config.defaults.max_prompt_bytes,
                "max_tool_output_bytes": config.defaults.max_tool_output_bytes,
                "max_tool_calls": {
                    "aspect": config.agents.aspect.max_tool_calls,
                    "explorer": config.agents.explorer.max_tool_calls,
                    "fixer": config.agents.fixer.max_tool_calls,
                },
            },
        )

    llm = LLMClient(
        config.llm,
        transport=transport,
        max_prompt_bytes=config.defaults.max_prompt_bytes,
        tracer=tracer,
    )
    return RunSetup(
        project_root=project_root,
        config=config,
        db_path=db_path,
        llm=llm,
        tracer=tracer,
    )


def _teardown(setup: RunSetup, viewer: ServerInfo | None, exit_code: int) -> None:
    """Close run resources and print trace-viewer diagnostics."""
    setup.llm.close()
    if setup.tracer is not None:
        setup.tracer.close(
            exit_code=exit_code,
            status="fatal" if exit_code == EXIT_FATAL else ("issues" if exit_code else "ok"),
        )
        print(f"trace: {setup.tracer.run_dir}", flush=True)
    if viewer is not None:
        print(
            f"khorek traces: still running at {viewer.url} (pid {viewer.pid})",
            flush=True,
        )
        print(
            f"khorek traces: stop it with `khorek traces --stop --trace-dir "
            f"{shlex.quote(str(viewer.root))}` (or kill {viewer.pid})",
            flush=True,
        )


def _run_workflow(
    args: argparse.Namespace,
    transport: httpx.BaseTransport | None,
    *,
    command: str,
    require_verify: bool,
    body: WorkflowBody,
) -> int:
    """Run a workflow body with the shared preamble and teardown.

    The model probe and trace recorder are set up before the lock so a fatal
    run still leaves a record of why it failed.
    """
    prepared = _prepare_run(
        args,
        command=command,
        transport=transport,
        require_verify=require_verify,
    )
    if isinstance(prepared, int):
        return prepared

    setup = prepared
    exit_code = EXIT_FATAL
    viewer = None
    try:
        try:
            setup.llm.preflight()
        except ModelUnavailable as exc:
            return _fail(str(exc))

        try:
            with acquire(setup.db_path):
                try:
                    with Database(setup.db_path, setup.project_root) as database:
                        reporter = Reporter()
                        pruned = database.prune_drift(
                            [aspect.id for aspect in setup.config.aspects]
                        )
                        app = AppContext(
                            project_root=setup.project_root,
                            config=setup.config,
                            database=database,
                            llm=setup.llm,
                            reporter=reporter,
                            tracer=setup.tracer,
                        )
                        if setup.tracer is not None:
                            viewer = start_viewer(setup.tracer.trace_root)
                            if viewer is not None:
                                print(
                                    f"khorek traces: {viewer.url} ({viewer.root})",
                                    flush=True,
                                )
                        exit_code = body(app, args.limit, pruned)
                        return exit_code
                except DatabaseError as exc:
                    return _fail(str(exc))
        except LockHeld as exc:
            return _fail(str(exc))
        except LockError as exc:
            return _fail(str(exc))
    finally:
        _teardown(setup, viewer, exit_code)


def _filter_label(aspect_ids: Sequence[str] | None, issue_ids: Sequence[int] | None) -> str:
    """Render the active ``--aspect``/``--issue-id`` filters for a message."""
    parts = [f"--aspect {aspect_id}" for aspect_id in aspect_ids or []]
    parts.extend(f"--issue-id {issue_id}" for issue_id in issue_ids or [])
    return " ".join(parts)


def _run_explore(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> int:
    aspect_ids = getattr(args, "aspects", None) or None

    def body(app: AppContext, limit: int, pruned: int) -> int:
        stats = run_explore(app, limit=limit, aspect_ids=aspect_ids)
        app.reporter.summary(
            stats,
            pruned=pruned,
            open_issues=app.database.count_issues(aspect_ids=aspect_ids),
        )
        return EXIT_ERROR if stats.errored else EXIT_OK

    return _run_workflow(
        args,
        transport,
        command="explore",
        require_verify=False,
        body=body,
    )


def _run_fix(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> int:
    aspect_ids = getattr(args, "aspects", None) or None
    issue_ids = getattr(args, "issue_ids", None) or None

    def body(app: AppContext, limit: int, pruned: int) -> int:
        if pruned:
            app.reporter.message(
                f"fix: pruned {pruned} issue/state row(s) from aspects no longer in the config"
            )
        if issue_ids:
            requested = list(dict.fromkeys(issue_ids))
            found = {issue.id for issue in app.database.list_issues(issue_ids=requested)}
            missing = [issue_id for issue_id in requested if issue_id not in found]
            if missing:
                for issue_id in missing:
                    print(f"khorek: unknown issue id: {issue_id}", file=sys.stderr)
                return EXIT_FATAL
        if (aspect_ids or issue_ids) and (
            app.database.count_issues(aspect_ids=aspect_ids, issue_ids=issue_ids) == 0
        ):
            print(f"khorek: no issues match the filters ({_filter_label(aspect_ids, issue_ids)})")
            return EXIT_OK
        stats = run_fix(app, limit=limit, aspect_ids=aspect_ids, issue_ids=issue_ids)
        remaining = app.database.count_issues(aspect_ids=aspect_ids, issue_ids=issue_ids)
        return EXIT_ERROR if stats.failed or remaining else EXIT_OK

    return _run_workflow(
        args,
        transport,
        command="fix",
        require_verify=True,
        body=body,
    )


def _run_explore_and_fix(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> int:
    aspect_ids = getattr(args, "aspects", None) or None

    def body(app: AppContext, limit: int, pruned: int) -> int:
        stats = run_explore_and_fix(app, limit=limit, pruned=pruned, aspect_ids=aspect_ids)
        return EXIT_ERROR if stats.remaining else EXIT_OK

    return _run_workflow(
        args,
        transport,
        command="explore-and-fix",
        require_verify=True,
        body=body,
    )


def _resolve_trace_root(project_root: Path, args: argparse.Namespace) -> Path:
    """Resolve the trace directory the viewer command should use."""
    if args.trace_dir is not None:
        return Path(args.trace_dir).expanduser().resolve()
    try:
        config = load_config(project_root)
    except ConfigError:
        config = None
    if config is not None and config.observe.directory.strip():
        return Path(config.observe.directory).expanduser().resolve()
    return trace_dir_for(_resolve_db_path(project_root, args.db))


def _run_traces(args: argparse.Namespace) -> int:
    project_root = _resolve_project(args.path)
    if project_root is None:
        return EXIT_FATAL
    root = _resolve_trace_root(project_root, args)
    if args.stop:
        info = stop_viewer(root)
        if info is None:
            print(f"khorek: no trace viewer running for {root}")
        else:
            print(f"khorek traces: stopped {info.url} (pid {info.pid})")
        return EXIT_OK
    if not root.exists():
        print(
            f"khorek: no traces at {root}; run a workflow with --trace or "
            "set [observe] enabled = true",
            file=sys.stderr,
        )
        return EXIT_FATAL
    try:
        serve_traces(
            root,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
            verbose=args.verbose,
        )
    except SystemExit:
        return EXIT_FATAL
    return EXIT_OK


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


def _run_init(args: argparse.Namespace, answers: Answers | None = None) -> int:
    """Run the interactive ``khorek init`` wizard and dump a canonical config."""
    destination = Path(args.path).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        return _fail(f"not a directory: {destination}")
    if not args.config_name or Path(args.config_name).name != args.config_name:
        return _fail(f"invalid --config-name: {args.config_name!r}")
    target = init_config_path(destination, args.config_name)
    try:
        if target.exists():
            return _fail(
                f"refusing to overwrite existing config: {target}"
                "(use --config-name for a fresh file)"
            )
    except OSError as exc:
        return _fail(f"cannot check existing config: {target}: {exc}")

    try:
        collected = collect_answers(answers=answers)
    except (EOFError, KeyboardInterrupt):
        print("khorek: init aborted", file=sys.stderr)
        return EXIT_ERROR
    except InitError as exc:
        return _fail(str(exc))

    try:
        destination.mkdir(parents=True, exist_ok=True)
        target.write_text(render_init_config(collected), encoding="utf-8")
    except OSError as exc:
        return _fail(f"cannot write {target}: {exc}")

    print(f"khorek: wrote {target}")
    return EXIT_OK


def _issue_filters(
    args: argparse.Namespace,
) -> tuple[Sequence[str] | None, Sequence[str] | None, Sequence[str] | None]:
    """The repeatable list filters from the parsed arguments.

    Returns ``(aspect_ids, severities, path_prefixes)``; an empty filter list
    is normalized to ``None`` so the database applies no constraint.
    """
    aspects = getattr(args, "aspects", None)
    severities = getattr(args, "severities", None)
    prefixes = getattr(args, "path_prefixes", None)
    return aspects or None, severities or None, prefixes or None


def _list_readonly(
    db_path: Path,
    project_root: Path,
    filters: tuple[Sequence[str] | None, Sequence[str] | None, Sequence[str] | None],
    *,
    note_prefix: str | None,
) -> int:
    """Print the issue list from a read-only connection, stale rows labeled.

    ``note_prefix`` (e.g. ``"run in progress (pid 42): "``) is prepended to
    the ``K stale issue(s) shown, not pruned`` note, which is printed only
    when at least one displayed row is stale.
    """
    aspect_ids, severities, path_prefixes = filters
    try:
        with Database(db_path, project_root, readonly=True) as database:
            issues = database.list_issues(
                aspect_ids=aspect_ids,
                severities=severities,
                path_prefixes=path_prefixes,
            )
            stale_map = staleness(database, project_root)
            rendered = render_list(issues, stale=stale_map)
    except DatabaseError as exc:
        return _fail(str(exc))
    shown_stale = sum(1 for issue in issues if stale_map.get(issue.id) != "fresh")
    if shown_stale:
        note = f"{shown_stale} stale issue(s) shown, not pruned"
        if note_prefix:
            note = f"{note_prefix}{note}"
        print(note)
        print()
    print(rendered)
    return EXIT_OK


def _run_issues_list(args: argparse.Namespace, project_root: Path, db_path: Path) -> int:
    filters = _issue_filters(args)
    if getattr(args, "stale", False):
        return _list_readonly(db_path, project_root, filters, note_prefix=None)

    aspect_ids, severities, path_prefixes = filters
    try:
        with acquire(db_path):
            with Database(db_path, project_root) as database:
                pruned = prune_stale(database, project_root)
                issues = database.list_issues(
                    aspect_ids=aspect_ids,
                    severities=severities,
                    path_prefixes=path_prefixes,
                )
                rendered = render_list(issues, stale=None)
    except LockHeld as exc:
        return _list_readonly(
            db_path, project_root, filters, note_prefix=f"run in progress (pid {exc.pid}): "
        )
    except LockError as exc:
        return _fail(str(exc))
    except DatabaseError as exc:
        return _fail(str(exc))

    if pruned:
        print(f"pruned {pruned} stale issue(s) (file changed or missing)")
        print()
    print(rendered)
    return EXIT_OK


def _run_issues_show(args: argparse.Namespace, project_root: Path, db_path: Path) -> int:
    try:
        with Database(db_path, project_root, readonly=True) as database:
            issue = database.get_issue(args.id)
            if issue is None:
                print(f"khorek: unknown issue id: {args.id}", file=sys.stderr)
                return EXIT_ERROR
            status = issue_status(database, project_root, issue)
    except DatabaseError as exc:
        return _fail(str(exc))
    print(render_show(issue, status=status))
    return EXIT_OK


def _report_outcomes(verb: str, outcomes: Sequence[IssueOutcome]) -> int:
    """Print one line per outcome; exit 1 when any id was unknown."""
    exit_code = EXIT_OK
    for outcome in outcomes:
        if not outcome.found:
            print(f"khorek: unknown issue id: {outcome.issue_id}", file=sys.stderr)
            exit_code = EXIT_ERROR
            continue
        suffix = f" \u2192 {outcome.action}" if outcome.action else ""
        location = f"{outcome.aspect_id} {outcome.path}:{outcome.lines}"
        print(f"{verb} #{outcome.issue_id}: {location}{suffix}")
    return exit_code


def _run_issues_mutation(
    args: argparse.Namespace,
    project_root: Path,
    db_path: Path,
    *,
    verb: str,
    operation: Callable[[Database, Path, Sequence[int]], list[IssueOutcome]],
) -> int:
    """Shared body for ``dismiss``/``delete``: lock, run, report per-id lines."""
    try:
        with acquire(db_path):
            with Database(db_path, project_root) as database:
                outcomes = operation(database, project_root, args.ids)
    except LockHeld as exc:
        return _fail(str(exc))
    except LockError as exc:
        return _fail(str(exc))
    except DatabaseError as exc:
        return _fail(str(exc))
    return _report_outcomes(verb, outcomes)


_HANDLERS = {
    "list": _run_issues_list,
    "show": _run_issues_show,
    "dismiss": partial(_run_issues_mutation, verb="dismissed", operation=dismiss),
    "delete": partial(_run_issues_mutation, verb="deleted", operation=delete_issues),
}


def _run_issues(args: argparse.Namespace) -> int:
    """Shared preamble for the ``khorek issues`` commands.

    No config load, no LLM client, no tracer: the issue commands operate on
    the database alone.  Exit codes: 0 for list (any result), for fully
    processed batches, and for show; 1 for a missing database (show,
    dismiss, delete) or unknown ids; 2 for a held lock (dismiss, delete) or
    an unreadable database.
    """
    subcommand = getattr(args, "issues_command", None) or "list"
    project_root = _resolve_project(getattr(args, "path", "."))
    if project_root is None:
        return EXIT_FATAL
    db_path = _resolve_db_path(project_root, getattr(args, "db", None))
    if not db_path.exists():
        if subcommand == "list":
            print("no open issues")
            return EXIT_OK
        print(f"khorek: no database at {db_path} (run khorek explore first)", file=sys.stderr)
        return EXIT_ERROR
    handler = _HANDLERS.get(subcommand)
    if handler is None:
        return _fail(f"unknown issues subcommand: {subcommand}")
    return handler(args, project_root, db_path)


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    answers: Answers | None = None,
) -> int:
    """Run the CLI and return the process exit code.

    ``transport`` is the in-process test seam: tests inject an
    ``httpx.MockTransport`` and never touch a real model.  ``answers`` is the
    prompt seam for the interactive commands; ``input()`` is the default.
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

    if args.command == "init":
        return _run_init(args, answers)

    if args.command == "explore":
        return _run_explore(args, transport)
    if args.command == "fix":
        return _run_fix(args, transport)
    if args.command == "explore-and-fix":
        return _run_explore_and_fix(args, transport)
    if args.command == "traces":
        return _run_traces(args)
    if args.command == "reset":
        return _run_reset(args)
    if args.command == "issues":
        return _run_issues(args)

    print(f"khorek: unknown command: {args.command}", file=sys.stderr)
    return EXIT_FATAL


if __name__ == "__main__":
    raise SystemExit(main())
