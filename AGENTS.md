# AGENTS.md

## Project Overview

**khorek** is a tool that scouts a codebase for issues and fixes them using
local agentic LLMs. It runs configurable *aspects* (e.g. "comments consistency")
over a project, records findings in SQLite, then fixes them one at a time with a
verifier in the loop. After a successful fix moves a file's version, the pair's
remaining open issues on that path are **re-validated** against the new contents
(a read-only, cross-aspect LLM step): `valid` refreshes the issue's lines,
`gone` (with a reason) closes it, and `uncertain`/a missing verdict leaves it
untouched (silence is never loss).

- **Language**: Python >= 3.12
- **Main libraries**: httpx, langgraph, pathspec, pydantic
- **Purpose**: Autonomous codebase maintenance — periodically scout for issues
and fix them

## Structure

| Directory | Description |
|---|---|
| `src/khorek/` | Main source code — core modules (cli, config, agents, llm, etc.) |
| `tests/` | Test suite — unit and integration tests for all components |
| `docs/` | Documentation (tracing, configuration, etc.) |
| `tests/fixtures/` | E2E test fixtures and sample projects |
| `src/khorek/web/` | Web UI assets (app.css, app.js, index.html) |

## Testing and verification

- **Framework**: pytest
- **Strategy**:
  - Unit tests for individual modules in `tests/`
  - E2E tests using fixture projects in `tests/fixtures/`
  - Tests verify aspect detection, issue recording, fixer workflow, and verifier loops
  - Real LLM calls are avoided in unit tests; fixtures provide deterministic data

See [`Justfile`](Justfile) for more commands.
