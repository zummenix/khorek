# Configuration

khorek is configured through a single TOML file, `khorek.toml` at the project
root (override with `--config <path>` on any workflow command).  `khorek init`
writes a canonical file interactively; existing configs are never overwritten.

Unknown keys are rejected everywhere — a typo in a section or a misspelled
key fails fast instead of silently falling back to a default.  The file is
validated on every load, so a broken config never reaches a run.

```toml
[llm]
base_url = "http://127.0.0.1:11434/v1"   # required
model = "inkling-2-mini:26b-a5b-nvfp4"    # required
```

The repository's own [`khorek.toml`](../khorek.toml) is a complete, working
example.

## Environment overrides

`[llm]` values can be overridden from the environment; the environment always
wins over the file:

| Variable | Overrides | Parsing |
| --- | --- | --- |
| `KHOREK_LLM_BASE_URL` | `llm.base_url` | string |
| `KHOREK_LLM_MODEL` | `llm.model` | string |
| `KHOREK_LLM_TEMPERATURE` | `llm.temperature` | number |
| `KHOREK_LLM_CALL_TIMEOUT` | `llm.call_timeout` | number |
| `KHOREK_LLM_MAX_TOKENS` | `llm.max_tokens` | integer |
| `KHOREK_LLM_REASONING_EFFORT` | `llm.reasoning_effort` | string; empty means `None` |

A malformed override (e.g. `KHOREK_LLM_TEMPERATURE=hot`) fails the load with a
`ConfigError` naming the variable.

`[observe]` is not env-overridden at load time, but it can be flipped per
invocation with `KHOREK_TRACE=1|0`, `KHOREK_TRACE_DIR=<path>`, or the `--trace` /
`--no-trace` / `--trace-dir` flags (the `--no-trace` flag always wins).  See
[tracing.md](tracing.md).

## [llm]

OpenAI-compatible model endpoint settings.  Both `base_url` and `model` are
required; everything else has a default.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `base_url` | string | *(required)* | Base URL of the OpenAI-compatible endpoint, e.g. `http://127.0.0.1:11434/v1`.  The harness appends `/models` (preflight) and `/chat/completions`. |
| `model` | string | *(required)* | Model name sent as `model` in every request. |
| `temperature` | float | `0.2` | Sampling temperature, `0.0`–`2.0`. |
| `call_timeout` | float | `600.0` | Per-call HTTP timeout in seconds (`> 0`).  A timed-out call raises a `timeout` agent error. |
| `max_tokens` | int | `8192` | Hard output cap in tokens (`>= 0`).  Ollama maps this to `num_predict`.  A model that hits the cap stops with `finish_reason=length`; when it then replies with text instead of a tool call, the harness aborts the agent with a `budget` error instead of paying for a second identical call.  `0` omits the field, leaving the model's own default. |
| `reasoning_effort` | string \| omit | *(unset)* | Reasoning level passed to the model (`none`, `low`, `medium`, `high`, ...).  Values are backend-specific and passed through unchecked; the valid set is whatever the endpoint accepts (Ollama: `none|low|medium|high`, OpenAI: `minimal|low|medium|high`).  Omitting it leaves the field out of the request entirely, so the model keeps its own default; `"none"` switches thinking off on Ollama's `/v1/chat/completions`.  Per-agent `reasoning_effort` values override this. |

## [defaults]

Harness-wide limits and budgets.  Every agent and workflow reads these unless a
more specific setting applies.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `max_issues_per_file` | int | `3` | Cap on stored issues per (aspect, file) pair (`> 0`).  The explore workflow clamps reported line numbers, drops empty descriptions, repairs invalid severities to `low`, and trims the list to this cap.  An aspect's own `max_issues_per_file` may lower but never raise this. |
| `max_file_bytes` | int | `200000` | Files larger than this many bytes are never enumerated and are refused by the tools (`read_file`, `grep`) (`> 0`).  Binary files (NUL byte in the first 8 KiB) are always dropped regardless of size. |
| `max_prompt_bytes` | int | `400000` | Hard budget on the serialized messages checked before every model call (`> 0`).  Over budget, the agent aborts with a `prompt_budget` error. |
| `max_tool_output_bytes` | int | `16000` | Per-tool budget for the serialized result (`> 0`).  Each tool trims its own output to this budget; the fixed result metadata (paths, line numbers, wrapper keys) is a floor — a budget below it cannot be honored, so the tool returns the smallest valid result (empty payload, `truncated=true`). |
| `max_cycles` | int | `25` | No-progress guard for the `explore-and-fix` workflow (`> 0`): when this many explore→fix cycles elapse without converging (explore creating zero issues), the run stops with a warning.  The remaining issues stay open for the next run. |

## [observe]

Opt-in traces of every model call, for debugging the harness.  Tracing is
diagnostic only: a trace I/O failure is reported once and ignored, never fatal.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Record model calls.  Overridden per invocation by `--trace` / `--no-trace` and `KHOREK_TRACE`. |
| `directory` | string | `""` | Trace root directory.  Empty means `<db>.traces` next to the database (e.g. `~/.khorek/<project-slug>.db.traces/`).  `KHOREK_TRACE_DIR` overrides this; the `--trace-dir` flag wins over both. |
| `keep_runs` | int | `30` | Keep the newest N runs (`>= 0`).  Pruning happens when a new run starts; `0` keeps everything. |

See [tracing.md](tracing.md) for what the traces contain and how to view them.

## [agents.aspect]

The aspect agent reviews one file per (aspect, file) pair and reports structured
issues.  After a fix moves a file, a read-only revalidation run re-judges the
pair's remaining open issues against the new contents; it reuses `max_tool_calls`
and `reasoning_effort` from this section but has no `ask_explorer` hand-off, so
`max_handoffs` does not apply there.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `reasoning_effort` | string \| omit | *(inherit `[llm]`)* | Per-agent override of the `[llm]` reasoning level.  Omit to inherit. |
| `max_handoffs` | int | `3` | Maximum `ask_explorer` hand-offs per aspect run (`>= 0`).  Past the limit the tool returns an error instead of handing off.  `0` disables the explorer hand-off entirely. |
| `max_tool_calls` | int | `20` | Maximum tool calls in one review loop (`> 0`).  The loop also aborts on a repeated call or a model that keeps replying with text. |

## [agents.explorer]

The read-only sub-agent behind `ask_explorer`.  It has no terminal tool of its
own; it answers with plain text.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `reasoning_effort` | string \| omit | *(inherit `[llm]`)* | Per-agent override of the `[llm]` reasoning level.  Omit to inherit. |
| `max_tool_calls` | int | `20` | Maximum tool calls in one exploration (`> 0`). |

## [agents.fixer]

The fixer agent rewrites one file to address one issue, in a bounded loop with a
verifier in the loop.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `reasoning_effort` | string \| omit | *(inherit `[llm]`)* | Per-agent override of the `[llm]` reasoning level.  Omit to inherit. |
| `max_attempts` | int | `3` | Maximum fix rounds per issue (`> 0`).  A round is one fix attempt (fix → verify → review); a verification failure or a reviewer rejection consumes a round.  The loop exits after this many rounds without an accepted fix. |
| `max_tool_calls` | int | `20` | Maximum tool calls per fix round (`> 0`). |

## [agents.verifier]

The verifier summarizes failing verification output for the fixer.  Pass/fail
is never model-decided — the configured `[[verify]]` commands decide it; the
verifier only distils their output into an actionable summary.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `reasoning_effort` | string \| omit | *(inherit `[llm]`)* | Per-agent override of the `[llm]` reasoning level.  Omit to inherit.  No budgets are configurable; the loop is fixed at 2 turns. |

## [agents.reviewer]

The reviewer is a model-based quality gate on a verified fix: it reads the diff
and either approves or rejects it with a reason handed back to the fixer.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `reasoning_effort` | string \| omit | *(inherit `[llm]`)* | Per-agent override of the `[llm]` reasoning level.  Omit to inherit. |
| `max_handoffs` | int | `2` | Maximum review→fix hand-offs per issue (`>= 0`).  Past the limit, the issue is marked failed. |
| `max_tool_calls` | int | `20` | Maximum tool calls in one review loop (`> 0`). |

## [[verify]]

Deterministic, non-mutating commands run after each fix, in order, from the
project root.  The run stops at the first failure; the exit code (or timeout)
is authoritative and never model-decided.  The captured output tail is handed
to the Verifier agent for summarising.  Commands come from the config only —
the model can never choose or alter them.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `cmd` | string | *(required)* | Command line run with a shell (`shell=True`), e.g. `"uv run pytest -q"`. |
| `timeout` | float | `120.0` | Per-command timeout in seconds (`> 0`).  A timeout is a failure with no exit code. |

## [[aspects]]

One aspect is a prompt plus the files it applies to.  Aspects run during
`explore` (and again during revalidation after a fix).  An aspect is matched
against files by its `files` scope; see [`files.py`](../src/khorek/files.py)
for the enumeration rules.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `id` | string | *(required)* | Unique, non-empty identifier.  Reported issues are keyed by it.  Duplicates are rejected at load. |
| `prompt` | string | *(required)* | The task prompt given to the aspect agent.  Must be non-empty. |
| `files` | list of string | *(required)* | Scope of files the aspect runs over: globs, directories, or explicit paths (e.g. `["src/**/*.py", "tests/*.py"]`).  Globs use gitignore-style patterns.  Entries may be absolute paths inside the project.  Root `.gitignore` is honored, build/tooling directories are always skipped, and binary or oversized files are dropped.  Results are sorted, deduplicated, and relative to the project root. |
| `max_issues_per_file` | int \| omit | *(inherit `[defaults]`)* | Per-aspect cap on stored issues per file (`> 0`).  May lower but never exceed `defaults.max_issues_per_file` — an aspect above the default is rejected at load. |

## Full example

```toml
[llm]
base_url = "http://127.0.0.1:11434/v1"
model = "inkling-2-mini:26b-a5b-nvfp4"
temperature = 0.2
call_timeout = 3600
max_tokens = 16000

[defaults]
max_issues_per_file = 3
max_file_bytes = 200000
max_prompt_bytes = 400000
max_tool_output_bytes = 16000
max_cycles = 25

[observe]
enabled = true
keep_runs = 50

[agents.aspect]
reasoning_effort = "low"
max_handoffs = 3
max_tool_calls = 20

[agents.fixer]
reasoning_effort = "low"
max_attempts = 5
max_tool_calls = 20

[agents.reviewer]
reasoning_effort = "low"
max_handoffs = 2
max_tool_calls = 20

[[verify]]
cmd = "uv run pytest -q"
timeout = 600

[[aspects]]
id = "comments"
prompt = "Find comments that are inconsistent with the code."
files = ["src/**/*.py"]
max_issues_per_file = 2
```

See [`khorek.toml`](../khorek.toml) for the project's own, fully commented
configuration.
