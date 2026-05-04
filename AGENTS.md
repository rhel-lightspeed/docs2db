# AGENTS.md

Knowledge base for AI agents working on this repository.

## Project Overview

docs2db is a CLI tool and Python library that builds RAG databases from documents.

**Pipeline:** ingest (Docling) → chunk (LLM context) → embed (Granite 30M) → load (PostgreSQL/pgvector) → dump (portable SQL)

**Philosophy:** Files on disk are the source of truth. The database is a derived production artifact. All stages are idempotent — they check timestamps and hashes, skipping already-completed work.

**Related repositories:**

- [docs2db-api](https://github.com/rhel-lightspeed/docs2db-api) — serves RAG queries from the database docs2db builds
- [docs2db-mcp-server](https://github.com/rhel-lightspeed/docs2db-mcp-server) — MCP interface for AI assistants

**Author:** Ellis Low (elow@redhat.com)
**License:** Apache-2.0
**Python:** 3.12 (strict — no other versions)
**Package manager:** uv
**CLI framework:** Typer

## Development Environment

```bash
# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Run tests (requires PostgreSQL — see "Testing" section)
make db-up-test
make test

# Run linting/formatting/type checks
uv run pre-commit run --all-files

# CI-safe test run (excludes tests needing Podman/external services)
make test-ci

# Stop test database
make db-down-test
```

## Code Architecture

```
src/docs2db/
├── docs2db.py      # CLI interface (Typer app, 783 lines) — all commands defined here
├── database.py     # PostgreSQL + pgvector operations (2216 lines, largest file)
├── chunks.py       # Contextual chunking with LLM (Ollama/OpenAI/WatsonX)
├── ingest.py       # Document ingestion via Docling
├── embed.py        # Embedding generation orchestration (multiprocessing)
├── embeddings.py   # Embedding model configs (Granite, E5, Slate, etc.)
├── multiproc.py    # Multiprocessing batch processor utility
├── db_lifecycle.py # Database start/stop/destroy via Podman/Docker compose
├── audit.py        # Content directory auditing
├── config.py       # Pydantic settings
├── exceptions.py   # Custom exception hierarchy
├── const.py        # Constants (schema version, etc.)
└── utils.py        # Utilities
```

## Key Patterns and Conventions

- **Logging:** structlog (`structlog.get_logger()`). Do NOT use stdlib `logging`.
- **Log formatting:** Avoid f-strings in structlog calls. Use `%s` style for lazy evaluation.
- **Config/settings:** Pydantic (`pydantic-settings`).
- **PostgreSQL driver:** psycopg v3 (async). NOT psycopg2.
- **Idempotency:** All pipeline stages check timestamps/hashes and skip completed work.
- **File-based workflow:** The content directory (`docs2db_content/`) mirrors source document hierarchy. Files are source of truth; database is rebuilt from them.
- **PyTorch:** CPU-only by design. `pyproject.toml` uses the `pytorch-cpu` index. GPU is optional.
- **Multiprocessing:** CPU defaults to 1 worker (avoids PyTorch fork deadlocks). GPU gets 2 workers.

## Testing

- **Framework:** pytest with pytest-asyncio, pytest-cov, pytest-httpx, pytest-randomly
- **ALL tests are integration tests.** Every test requires PostgreSQL running. No unit tests exist yet.
- **Test database:** Port 5433 (NOT 5432), user `test_user`, password `test_password`, database `test_docs2db`
- **Start test DB:** `make db-up-test` (uses `--profile test` in `postgres-compose.yml`)
- **Stop test DB:** `make db-down-test`
- **Test config:** `tests/test_config.py` → `get_test_db_config()` reads env vars or uses defaults
- **conftest.py:** Auto-scoped fixture creates and drops a fresh test database per test function
- **Coverage:** Configured in `pyproject.toml` (term-missing, html, xml, branch)
- **Markers:** `no_ci` — tests requiring external services (Podman, LLM providers, etc.)
- **CI test command:** `make test-ci` (runs `pytest -m "not no_ci"`)

## Pre-commit Hooks

These run on every commit and in CI:

- **ruff** — linting with auto-fix
- **ruff-format** — code formatting
- **pyright** — type checking (src/docs2db/ only)
- **gitleaks** — secret detection
- **check-toml** — TOML validation
- **end-of-file-fixer** — ensures files end with newline
- **trailing-whitespace** — removes trailing spaces

Run manually: `uv run pre-commit run --all-files`

## Gotchas

- ALL tests need PostgreSQL running (`make db-up-test` first)
- Test DB runs on port 5433 (NOT 5432) with separate credentials
- `make test-ci` excludes `no_ci` marked tests
- database.py is 2216 lines — the largest and most complex module
- torch is CPU-only by design (pyproject.toml uses pytorch-cpu index)
- Embedding model loading takes 30-40s on first load — cold starts are slow
- `postgres-compose.yml` has two profiles: `prod` (port 5434) and `test` (port 5433)
- f-strings in structlog calls are discouraged (use `%s` style for lazy evaluation)

## Branch Protection

- **Org-level ruleset:** "Minimum required Branch Protection" (rhel-lightspeed org)
- Requires 1 approving review from code owner (`@rhel-lightspeed/developers`)
- Last pusher cannot approve their own PR
- Cannot be bypassed
