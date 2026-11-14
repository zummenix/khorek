# khorek

A tool to scout a codebase for issues and fix them, designed for local
agentic LLMs running overnight.

## Name

khorek (хорёк in Russian) – a ferret is a small, long-bodied, and playful
carnivorous mammal belonging to the weasel family (Mustelidae) and descended
from the European polecat.

## Idea

We often need to maintain a healthy state of a project, but sometimes do not
have time or resources for it. The solution is to periodically launch an
autonomous system that will scout the project for issues and fix them.

`khorek` runs a set of configurable *aspects* (e.g. "comments consistency") over
a project, records the issues it finds in SQLite, then fixes them one at a time
with a verifier in the loop so nothing lands unvalidated.

## How it works

`khorek` runs configurable **aspects** – each is a prompt plus a file scope
(e.g. "comments consistency" over `src/**/*.py`) – against the project. Findings
are recorded in a local SQLite database. Each issue is then fixed individually
by a fixer agent, and a verifier loop re-checks the fix before it's accepted.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and a local OpenAI-compatible LLM
server (e.g. [Ollama](https://ollama.com)) with a model.

```bash
git clone https://github.com/zummenix/khorek.git
cd khorek
uv tool install .
```

**Init**

```bash
khorek init      # interactively setup the `khorek.toml`
```

It asks for the LLM base URL (skippable, defaults to a local Ollama server), the
model name (required), a verify command (required, e.g. `ruff format --check .`),
and whether to add the `comments` aspect. Existing configs are never overwritten.

**Explore**

```bash
khorek explore   # scout: record issues in the database
```

**View**

```bash
khorek issues    # view issues recorded after `khorek explore`
```

**Fix**

```bash
khorek fix       # fix open issues one at a time, verifying each
```

See `khorek --help` for more commands

## Configuration

Aspects, agents, limits and verifier commands. Take a look at
[`khorek.toml`](khorek.toml).

## Traces

`khorek` can record every model call and serve a local viewer:

```bash
khorek explore --trace
```

See [docs/tracing.md](docs/tracing.md) for details.

## Dogfooding

khorek uses itself to maintain its own codebase: the aspects it runs against
itself target the same kinds of issues it looks for in yours.

## License

[MIT](LICENSE)
