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

from khorek.cli_parser import build_parser
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


@dataclass
class RunSetup:
    project_root: Path
    config: Config
    db_path: Path
    llm: LLMClient
    tracer: TraceRecorder | None = None


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


def _prepare_run(
    args: argparse.Namespace,
    *,
    command: str,
    transport: httpx.BaseTransport | None,
    require_verify: bool,
) -> RunSetup | int:
    """Resolve and validate workflow inputs, then prepare model tracing."""
    project_root = _resolve_project(args.project_root)
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
    project_root = _resolve_project(args.project_root)
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
    project_root = _resolve_project(args.project_root)
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
    destination = Path(args.project_root).expanduser().resolve()
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
    project_root = _resolve_project(getattr(args, "project_root", "."))
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


if __name__ == "__main__":
    raise SystemExit(main())
