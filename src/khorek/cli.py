"""Argparse-based command-line interface for khorek."""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx

from khorek.config import Config, ConfigError, load_config
from khorek.context import AppContext
from khorek.db import Database, DatabaseError, default_db_path
from khorek.llm import LLMClient, ModelUnavailable
from khorek.lock import LockError, LockHeld, acquire
from khorek.report import Reporter
from khorek.trace import TraceRecorder, trace_dir_for
from khorek.trace_server import serve as serve_traces
from khorek.trace_server import start_viewer, stop_viewer
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


def _run_workflow(
    args: argparse.Namespace,
    transport: httpx.BaseTransport | None,
    *,
    command: str,
    require_verify: bool,
    body: WorkflowBody,
) -> int:
    """Shared preamble/teardown for every workflow command.

    Resolve the project, load and validate config, optionally require a
    ``[[verify]]`` command, take the lock, open and validate the DB, prune
    config drift, run *body*, then release everything.  The model probe and the
    trace recorder are set up before the lock so a fatal run still leaves a
    record of why it failed.
    """
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
        max_tool_output_bytes=config.defaults.max_tool_output_bytes,
        tracer=tracer,
    )
    exit_code = EXIT_FATAL
    viewer = None
    try:
        try:
            llm.preflight()
        except ModelUnavailable as exc:
            return _fail(str(exc))

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
                            tracer=tracer,
                        )
                        if tracer is not None:
                            viewer = start_viewer(tracer.trace_root)
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
        llm.close()
        if tracer is not None:
            tracer.close(
                exit_code=exit_code,
                status="fatal" if exit_code == EXIT_FATAL else ("issues" if exit_code else "ok"),
            )
            print(f"trace: {tracer.run_dir}", flush=True)
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


def _run_explore(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> int:
    def body(app: AppContext, limit: int, pruned: int) -> int:
        stats = run_explore(app, limit=limit)
        app.reporter.summary(
            stats,
            pruned=pruned,
            open_issues=app.database.count_issues(),
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
    def body(app: AppContext, limit: int, pruned: int) -> int:
        if pruned:
            app.reporter.message(
                f"fix: pruned {pruned} issue/state row(s) from aspects no longer in the config"
            )
        stats = run_fix(app, limit=limit)
        remaining = app.database.count_issues()
        return EXIT_ERROR if stats.failed or remaining else EXIT_OK

    return _run_workflow(
        args,
        transport,
        command="fix",
        require_verify=True,
        body=body,
    )


def _run_explore_and_fix(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> int:
    def body(app: AppContext, limit: int, pruned: int) -> int:
        stats = run_explore_and_fix(app, limit=limit, pruned=pruned)
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
    if args.command == "fix":
        return _run_fix(args, transport)
    if args.command == "explore-and-fix":
        return _run_explore_and_fix(args, transport)
    if args.command == "traces":
        return _run_traces(args)
    if args.command == "reset":
        return _run_reset(args)

    print(f"khorek: unknown command: {args.command}", file=sys.stderr)
    return EXIT_FATAL


if __name__ == "__main__":
    raise SystemExit(main())
