set dotenv-load

# Default recipe: list available commands.
_default:
    @just --list

# Run ruff lint.
lint:
    uv run ruff check .

# Verify formatting (no changes).
format-check:
    uv run ruff format --check .

# Apply formatting.
format:
    uv run ruff format .

# Type check.
typecheck:
    uv run ty check .

# Run the test suite (no real LLM calls).
test:
    uv run pytest -q

# Run format check, lint, type check, and tests.
check: format-check lint typecheck test
