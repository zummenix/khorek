# khorek

A tool to scout a codebase to find issues and fix them. Oriented for local
agentic LLMs running overnight (small contexts, avoid thinking loops).

## Idea

We often need to maintain healthy state of a project but sometimes do not have
time or resources for it. The solution is periodically launch autonomous system
that will scout the project for issues and fixes them.

`khorek` runs a set of configurable *aspects* (e.g. "comments consistency")
over a project, records the issues it finds in SQLite, then fixes them one at a
time with a verifier in the loop so nothing lands unvalidated.

## Scope of the prototype

One end-to-end vertical slice against a real local model:

```
aspects x files -> explore -> issues in SQLite -> fix -> verify -> clean state
```

Minimal CLI surface, sequential execution, single project at a time. The
acceptance target is: with `mimo-v2.6:9b-mxfp8` served by Ollama, khorek finds
a deliberately inconsistent comment, fixes it, verifies the project, and marks
the file clean.

## Resolved design decisions

| Area | Decision |
| --- | --- |
| LLM access | OpenAI-compatible HTTP (`/v1/chat/completions`) via `httpx`; `base_url`, `model`, `think`, `temperature` from config, env vars override |
| Structured output | Forced tool call (`tool_choice=required`) whose arguments are validated by a Pydantic model; one repair retry on schema violation |
| Explorer | Separate sub-agent, invoked as a tool by the Aspect agent; max 3 hand-offs per aspect run, max 20 tool calls each — both configurable |
| Loop safety | Per-agent tool budget, timeout per call, and repeat-call detection (identical tool+args aborts the agent) |
| Mutation | Exact-match `SEARCH/REPLACE` edits plus `write_file`; a mismatch is returned to the agent as a structured error |
| Validation | Explicit command list in config; Verifier runs them, captures exit codes and output tails. The LLM never chooses commands |
| Storage | Central `~/.khorek/<slug>.db`; slug = readable path + 8 hex chars of `sha256(realpath)`. `--db PATH` overrides the location outright (used by e2e tests with a temp file) |
| Config drift | Issues and state rows whose `aspect_id` is no longer in the config are pruned at the start of every workflow run |
| Concurrency | One run per database: non-blocking `flock` on `<db>.lock`, released automatically on process death. A second run exits 2 immediately |
| Staleness | Whole-file sha256 per (aspect, file); mismatch invalidates issues and clean state |
| Failure | Restore snapshots of every touched file, keep the issue open, `retry_count += 1`, continue the run, report at the end |
| Git | Not used. No requirement of a clean tree, no commits |
| Orchestration | LangGraph `StateGraph`, plain function nodes, no checkpointer (the DB is the source of truth) |
| Config | TOML at `<project>/khorek.toml` |
| Observability | stdout only for the prototype (progress + end-of-run report) |

## Config file

`khorek.toml` lives at the project root (individual per project).
`--config` overrides its location. Unknown keys are errors.

```toml
[llm]
base_url = "http://127.0.0.1:11434/v1"
model = "mimo-v2.6:9b-mxfp8"
think = true
temperature = 0.2
call_timeout = 600        # seconds per model call

[defaults]
max_issues_per_file = 3   # per aspect, per file
max_file_bytes = 200000

[agents.aspect]
prompt = ""               # empty = built-in default prompt
max_handoffs = 3
max_tool_calls = 20

[agents.explorer]
prompt = ""
max_tool_calls = 20

[agents.fixer]
prompt = ""
max_attempts = 3

[agents.verifier]
prompt = ""

# Run in order, first failure is reported back to the Fixer.
[[verify]]
cmd = "uv run ruff format --check ."
timeout = 120

[[verify]]
cmd = "uv run pytest -q"
timeout = 600

[[aspects]]
id = "comments"
prompt = "Look at the comments and find those that are out of sync or contain inconsistencies."
files = ["src/**/*.py"]
max_issues_per_file = 2   # optional override

[[aspects]]
id = "dead-code"
prompt = "Find functions, variables and imports that are never used."
files = ["src/"]
```

`files` entries may be globs, directories (expanded recursively), or explicit
paths, resolved relative to the project root. Enumeration drops anything
ignored by `.gitignore` (parsed with `pathspec`, no git binary needed), skips
binary files, files over `max_file_bytes`, and `.git/`, `.venv/`,
`node_modules/`, `target/`. Order is sorted so runs are reproducible.

Aspect config may tighten `max_issues_per_file` for that aspect only. The cap
applies to the issues recorded from one aspect over one file.

## Data model

SQLite at `~/.khorek/<slug>.db`, WAL mode, paths stored relative to the project
root. `--db PATH` bypasses slugging entirely and uses that file (the parent
directory is created if missing). The project root and schema version live in
`meta`.

```sql
CREATE TABLE meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- One row per explored (aspect, file) pair. Written after every explore.
CREATE TABLE file_aspect_state (
  aspect_id TEXT NOT NULL,
  path      TEXT NOT NULL,
  sha256    TEXT NOT NULL,
  outcome   TEXT NOT NULL CHECK (outcome IN ('clean', 'issues', 'truncated')),
  updated_at TEXT NOT NULL,
  PRIMARY KEY (aspect_id, path)
);

-- A row exists iff the issue is open.
CREATE TABLE issues (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  aspect_id   TEXT NOT NULL,
  path        TEXT NOT NULL,
  sha256      TEXT NOT NULL,          -- hash of the file when the issue was found
  description TEXT NOT NULL,          -- free form, what is wrong
  line_start  INTEGER NOT NULL,       -- 1-based, validated to be within the file
  line_end    INTEGER NOT NULL,
  severity    TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high')),
  retry_count INTEGER NOT NULL DEFAULT 0,
  last_error  TEXT,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
```

`outcome = 'truncated'` means the Aspect agent reported more issues than the
cap; only the first N (in the model's own impact order) were stored, and after
those are fixed the pair is deliberately **not** marked clean so the next
explore revisits it.

### Explore skip rules

For each (aspect, file) with the file's current sha256:

- state row with matching sha -> skip (any outcome)
- no row, or sha mismatch -> delete the stale row and its issues, then explore
- issues exist with a matching sha -> skip (already open)

### Fix staleness

Before fixing an issue, compare the issue's sha256 to the file's current hash.
Mismatch means the file changed under us: delete the issue, count it as skipped,
and move on.

### Config drift

Aspects get renamed and deleted. At the start of every workflow run (not
`reset`), the set of `aspect_id`s in the loaded config is compared with the
database, and every issue and `file_aspect_state` row whose aspect is absent is
deleted. The number pruned is reported. Aspect ids are therefore stable
identifiers, not labels: renaming an id retires its issue history.

## Run locking

One run per database at a time. Before opening the DB, khorek takes a
non-blocking exclusive `flock` on `<db>.lock` (`fcntl.flock` on POSIX) and
writes its PID and start time into the file. If the lock is held, khorek exits
`2` with the holder's PID and start time, without touching the database.

The kernel releases `flock` when the process dies, so a crashed overnight run
never leaves a stale lock behind. The lock is keyed by the database, not the
project, so different projects (and `--db` temp runs) execute in parallel.
Waiting for a lock is deliberately out of scope for the prototype.

## LLM layer

- `httpx.Client` posting to `{base_url}/chat/completions`; `think` is passed
  through as `chat_template_kwargs` when supported and ignored otherwise.
- Every agent is a bounded loop: send messages -> receive tool calls -> execute
  -> repeat, up to `max_tool_calls`. Tool results are JSON strings.
- Repeat-call detection: hash of `(tool_name, arguments)`; the same call twice
  in a row (or three times overall) aborts the agent with a loop error.
- All tool paths are resolved against the project root and rejected if they
  escape it.
- **Ollama context note**: `/v1` does not accept `num_ctx`, so the server must
  be started with `OLLAMA_CONTEXT_LENGTH` large enough for `max_file_bytes`.
  khorek warns when an assembled prompt exceeds the configured byte budget.

## Agents

| Agent | Tools | Structured result |
| --- | --- | --- |
| Aspect | `read_file`, `grep`, `list_files`, `ask_explorer` | `report_issues(issues=[{description, line_start, line_end, severity}])`, asked in impact order |
| Explorer | `read_file`, `grep`, `list_files` | free text answer |
| Fixer | `read_file`, `edit_file`, `write_file` | `report_fix_result(fixed: bool, summary: str)` |
| Verifier | none | `report_verification(summary, failures=[{cmd, exit_code, output_tail}])` |

Pass/fail is never model-decided: the harness runs the configured commands in
the project root, sets `passed` from the exit codes, and the Verifier LLM only
distils the failure output into an actionable summary for the Fixer.

The Fixer receives the whole file plus the snippet reconstructed from the
issue's line range. Line ranges are validated only for being within the file,
never for content.

## Run lifecycle

Every workflow command does the same preamble and teardown:

1. Resolve the project root and config; load and validate config.
2. Resolve the database path (slug or `--db`) and acquire `<db>.lock`; exit `2`
   if held.
3. Open the DB, apply the schema, compare with the config and prune drifted
   aspects.
4. Run the workflow.
5. Print the end-of-run report and release the lock.

## Workflows

### Explore (`khorek explore`)

For each aspect, for each enumerated file (sorted):

1. Apply skip rules. On re-explore, delete stale issues/state first.
2. Read the file, snapshot nothing (read-only flow).
3. Run the Aspect agent with the aspect prompt, the file path, and the file
   contents. It may call the Explorer to check usages elsewhere.
4. Validate reported issues: clamp/repair out-of-range lines, drop empty
   descriptions, trim to `max_issues_per_file`.
5. Insert issues, then write the state row:
   `clean` if none reported, `truncated` if the cap cut the list, `issues` otherwise.

`--limit N` caps the number of issues *created* in the run; the run stops
cleanly once reached.

### Fix (`khorek fix`)

Queue: open issues ordered by `retry_count ASC, severity DESC, created_at ASC`
— issues that already failed go to the back.

For one issue:

1. Verify the sha256; delete and skip if stale.
2. Snapshot the original bytes of the issue's file (in memory). Any additional
   file the Fixer writes is snapshotted before its first write.
3. Run the Fixer (max `max_attempts` rounds). Each round: fix, `report_fix_result`;
   if `fixed`, the harness runs the configured commands in the project root and
   the Verifier summarises any failure.
4. Verifier success -> delete the issue, then finalize the pair: if no issues
   remain and the state outcome is `issues`/`clean`, write `clean` with the new
   hash; if the outcome was `truncated`, delete the state row so the next
   explore revisits.
5. Exhausted attempts or `fixed: false` -> restore every snapshot, set
   `retry_count += 1` and `last_error`, leave the issue open, continue with the
   next issue. The run never aborts on a single failure.
6. End of run: printed report of fixed / failed / skipped-stale / remaining.

`--limit N` caps the number of issues *attempted* in the run.

### Explore-and-fix (`khorek explore-and-fix`)

Fix first, then explore, then fix again:

```
attempted = set()
loop:
  drain queue (each issue attempted at most once per run)
  if --limit reached: stop
  created = explore(remaining budget)
  if created == 0: stop
```

Fixed issues are the only reason to loop again; failed ones are left for the
next run, which keeps the loop finite. `--limit N` applies independently to the
issues attempted and the issues created: stop when either budget is spent.

### Reset (`khorek reset`)

Deletes the database (the `--db` path when given) and its lock file, and prints
the path. It takes the lock first, so it refuses to delete a database that is in
use. No confirmation prompt in the prototype.

## CLI

```
khorek explore        [-p PATH] [--limit N] [--config PATH] [--db PATH]
khorek fix            [-p PATH] [--limit N] [--config PATH] [--db PATH]
khorek explore-and-fix[-p PATH] [--limit N] [--config PATH] [--db PATH]
khorek reset          [-p PATH] [--db PATH]
```

`-p` defaults to the current directory and is resolved to a real path before
slugging. `--limit 0` (default) means unlimited. `--db PATH` replaces the
derived `~/.khorek/<slug>.db`; it also relocates the lock file to `PATH.lock`.

Exit codes: `0` run completed with no failures, `1` run completed but at least
one issue failed or needs attention, `2` fatal error (bad config, no model,
unreadable project, database already locked).

## Project structure

```
khorek/
├── Justfile
├── pyproject.toml            # uv, ruff, ty, pytest config
├── khorek.toml               # dogfood config for this repo (dummy dir first, then src/)
├── src/khorek/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                # argparse, exit codes
│   ├── config.py             # pydantic models, TOML load, defaults merge
│   ├── db.py                 # schema, slug, queries, drift pruning
│   ├── lock.py               # flock on <db>.lock, PID/start-time payload
│   ├── files.py              # enumeration, ignores, hashing, binary check
│   ├── llm.py                # httpx client, forced tool calls, budgets
│   ├── snapshots.py          # in-memory originals + restore
│   ├── verify.py             # subprocess runner, output tails
│   ├── report.py             # stdout progress + end-of-run summary
│   ├── agents/
│   │   ├── aspect.py
│   │   ├── explorer.py
│   │   ├── fixer.py
│   │   └── verifier.py
│   ├── prompts/              # default system prompts
│   └── workflows/
│       ├── explore.py        # LangGraph StateGraph
│       └── fix.py            # LangGraph StateGraph
└── tests/
    ├── conftest.py           # httpx.MockTransport fake model, tmp project, --db tmp file
    ├── fixtures/dummy_project/
    └── test_*.py
```

## Toolchain

Python 3.12+, `uv`, `ruff` (lint + format), `ty` (type check), `pytest`.
Runtime deps: `langgraph`, `httpx`, `pydantic`, `pathspec`.

```just
set dotenv-load

# Default recipe: list available commands.
_default:
    @just --list

# Run ruff lint.
lint:
    ruff check .

# Verify formatting (no changes).
format-check:
    ruff format --check .

# Apply formatting.
format:
    ruff format .

# Type check.
typecheck:
    ty check .

# Run the test suite (no real LLM calls).
test:
    pytest -q

# Run format check, lint, type check, and tests.
check: format-check lint typecheck test
```

`just check` is hermetic: tests inject a fake model through
`httpx.MockTransport`, so no network and no model server are needed. `ty` is
preview-stage; if it cannot handle pydantic/langgraph generics we pin `pyright`
and note the reason in this document.

## Phases

Each phase ends with `just check` green. Phases 2-4 additionally have a manual
real-LLM acceptance run (the DoD); that run is never part of `just check`.

### Phase 1 — Setup

Scaffold the package, `pyproject.toml`, `.python-version`, `Justfile`, ruff/ty/pytest
config, and a placeholder CLI entry point.

**DoD**: `just check` passes (format, lint, typecheck, one trivial test);
`khorek --help` works via `uv run`.

### Phase 2 — CLI, DB, and explore

Config loading and validation, DB slug/schema/queries, run locking, aspect
drift pruning, `--db` override, file enumeration, LLM client with forced tool
calls and budgets, Aspect + Explorer agents, the explore LangGraph,
`khorek explore` and `khorek reset`.

**DoD (deterministic tests)**: fake-model tests covering skip-on-clean,
skip-on-open-issues, re-explore after a hash change, stale issue deletion, cap
truncation producing `truncated`, line-range repair, agent loop-abort, pruning
of issues and state for an aspect removed from the config, `--db` writing to a
tmp file instead of `~/.khorek`, and a second run against a held lock exiting
`2` without touching the DB.

**DoD (real LLM)**: add a `comments` aspect to this repo's `khorek.toml`
scoped to a dummy directory containing one deliberately inconsistent comment;
run `khorek explore -p . --db "$TMPDIR/khorek-e2e.db" --limit 5`; confirm the
issue row appears in that DB with a plausible line range; run `khorek reset --db "$TMPDIR/khorek-e2e.db"`.

### Phase 3 — Fix and verify

Snapshot/restore, `edit_file`/`write_file` tools, Fixer and Verifier agents,
the fix LangGraph with the `fix -> verify` loop, stale-issue skipping, retry
bookkeeping, and the stdout report.

**DoD (deterministic tests)**: fake-model tests for a successful fix marking the
pair clean, a verification failure feeding errors back to the Fixer, exhausted
attempts restoring the file and incrementing `retry_count`, stale-hash issue
deletion, and a fix touching a second file being restored too.

**DoD (real LLM)**: with the inconsistent comment from phase 2 recorded, run
`khorek fix -p . --db "$TMPDIR/khorek-e2e.db"`; the comment is corrected, the
configured verify commands pass, the issue disappears, and the pair is marked
clean.

### Phase 4 — Explore-and-fix and reporting

Combined workflow with the drain/explore loop, `--limit` accounting, the
no-progress guard, and a unified end-of-run report.

**DoD (deterministic tests)**: fake-model tests proving the queue is consumed
before exploring, failures do not re-enter the same run, `--limit` is respected
across phases, and the loop terminates when explore finds nothing.

**DoD (real LLM)**: on a fixture project with two inconsistent comments, one
`khorek explore-and-fix -p <fixture>` run fixes both, reports them, and leaves
no open issues.

## Deferred (not in the prototype)

Git integration (clean-tree requirement, per-fix commits), on-disk run
transcripts and snapshots, parallel exploration, LangGraph checkpointing,
unified-diff edits, severity weighting in the report, config schema migration,
multiple aspects fixed in a single file pass, waiting for a held lock instead of
failing fast, and cross-platform locking (the prototype targets POSIX `flock`).
