# Contributing to Docs2DB

Thank you for your interest in contributing to Docs2DB! This guide will help you set up your development environment and understand our development workflow.

## Development Setup

### Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- Podman or Docker - For running PostgreSQL
- **LLM Provider** - For contextual chunking (choose one):
  - [Ollama](https://ollama.ai/) - Local LLM (recommended for development)
  - OpenAI API key - External API
  - WatsonX API credentials - External API

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rhel-lightspeed/docs2db
   cd docs2db
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Install pre-commit hooks:**
   ```bash
   uv run pre-commit install
   ```

## Development Workflow

### Running Tests

Docs2DB uses pytest for automated testing. Tests require a PostgreSQL database.

**Start the test database:**
```bash
make db-up-test
```

This starts a PostgreSQL container (`test-db` profile in `postgres-compose.yml`) on port 5433 specifically for testing.

**Run the test suite:**
```bash
make test
```

**Run specific tests:**
```bash
uv run pytest tests/test_chunks.py
uv run pytest tests/test_embeddings.py::test_specific_function
```

**Stop the test database:**
```bash
make db-down-test
```

### Code Quality

**Install pre-commit hooks** to run automatically on every commit:
```bash
uv run pre-commit install
```

After installation, the hooks run automatically on `git commit` and will:
- Auto-fix formatting and style issues
- Block the commit if fixes are needed
- Leave fixed files in your working directory for review
- Require you to `git add` and commit again

**Run all checks and formatters:**
```bash
# Check all files
uv run pre-commit run --all-files
```

This runs:
- **ruff** - Linting with auto-fixes
- **ruff-format** - Code formatting
- **pyright** - Type checking
- **gitleaks** - Secret detection
- **check-toml** - TOML file validation
- **end-of-file-fixer** - Ensures files end with newline
- **trailing-whitespace** - Removes trailing spaces

**⚠️ Note:** Pre-commit hooks will **automatically modify your code** to fix formatting, import order, trailing whitespace, and other style issues. Always review the changes before committing.

### Database Management

**Development database** (main work):
```bash
uv run docs2db db-start    # Start PostgreSQL on port 5432
uv run docs2db db-stop     # Stop container (data persists)
uv run docs2db db-destroy  # Delete database and volumes
uv run docs2db db-status   # Check connection and stats
```

**Test database** (isolated for tests):
```bash
make db-up-test      # Start test PostgreSQL on port 5433
make db-down-test    # Stop test container
make db-destroy-test # Delete test database and volumes
```

Note: The main database and test database are completely separate and run on different ports.

### Manual Testing

**Quick pipeline** (runs all stages automatically):
```bash
uv run docs2db pipeline tests/fixtures/
```

This starts the database, ingests, chunks, embeds, loads, creates a dump, and stops the database.

**Run each stage individually** (for testing specific components):
```bash
# Ingest sample files
# Creates docs2db_content/ with Docling JSON files
uv run docs2db ingest tests/fixtures/

# Chunk with context (requires Ollama/OpenAI/WatsonX)
# Creates <name>.chunks.json files alongside each source file
uv run docs2db chunk

# Or skip context generation for faster testing
uv run docs2db chunk --skip-context

# Generate embeddings
# Creates <name>.gran.json files alongside each chunks file
uv run docs2db embed

# Load into database
uv run docs2db load

# Check database status
uv run docs2db db-status

# Create a dump
uv run docs2db db-dump
```

**Note:**
- Contextual chunking requires an LLM provider. If using Ollama (default), ensure it's running locally. Use `--skip-context` to bypass LLM requirements during testing.
- The default content directory is `docs2db_content/`. It includes a README explaining its purpose and recommending version control.

**Test the RAG demo client:**
```bash
uv run python scripts/rag_demo_client.py --query "your test query"
uv run python scripts/rag_demo_client.py --interactive
```

## Project Structure

```
docs2db/
├── src/docs2db/           # Main package code
│   ├── docs2db.py         # CLI interface (Typer)
│   ├── ingest.py          # Document ingestion (Docling)
│   ├── chunks.py          # Contextual chunking with LLM
│   ├── embed.py           # Embedding generation orchestration
│   ├── embeddings.py      # Embedding model configurations
│   ├── database.py        # PostgreSQL + pgvector operations
│   ├── db_lifecycle.py    # Database lifecycle (start/stop/destroy)
│   ├── config.py          # Pydantic settings
│   ├── multiproc.py       # Multiprocessing utilities
│   ├── audit.py           # Content directory auditing
│   ├── exceptions.py      # Custom exceptions
│   ├── const.py           # Constants
│   └── utils.py           # Utilities
├── tests/                 # Test suite
│   ├── fixtures/          # Test data (PDFs, DOCX, CSV, etc.)
│   ├── test_*.py          # Test files
│   └── conftest.py        # Pytest configuration
├── scripts/               # Helper scripts
│   └── rag_demo_client.py # RAG query demo
├── docs/                  # Additional documentation
│   ├── DESIGN.md
│   ├── INTEGRATION.md
│   ├── LLM_PROVIDERS.md
│   └── METADATA.md
├── postgres-compose.yml   # Database services (Docker Compose)
├── pyproject.toml         # Project config + dependencies
├── Makefile               # Development tasks
├── README.md              # User documentation
└── CONTRIBUTING.md        # This file
```

## Code Style

- **Python version:** 3.12
- **Formatter:** Ruff (enforced by pre-commit)
- **Type hints:** Required for public APIs
- **Imports:** Sorted by Ruff (isort rules)
- **Docstrings:** Use for public functions and classes

## Making Changes

### Branching

Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

### Commit Messages

Write clear, descriptive commit messages:
```
Add contextual chunking support

- Implement LLM-based context generation
- Add OpenAI and WatsonX providers
- Include map-reduce for large documents
```

If AI tools assist with your changes, please credit the specific model:
```
Refactor database connection logic

- Simplify connection pooling
- Add retry logic for transient failures

Co-authored-by: Claude 4.5 Sonnet
```

### Testing Your Changes

1. Add tests for new functionality
2. Run the test suite: `make test`
3. Run pre-commit checks: `uv run pre-commit run --all-files`
4. Test manually with `uv run docs2db ...`

### Submitting Changes

1. Ensure all tests pass: `make test`
2. Ensure pre-commit checks pass: `uv run pre-commit run --all-files`
3. Push your branch
4. Open a pull request with a clear description

## Common Development Tasks

### Adding a New Embedding Model

1. Add config to `EMBEDDING_CONFIGS` in `embeddings.py`
2. Test with `uv run docs2db embed --model your-model`
3. Update documentation

### Adding a New LLM Provider

1. Create provider class in `chunks.py` (inherit from `LLMProvider`)
2. Update `ContextualChunker.__init__()` to handle new provider
3. Add CLI option in `docs2db.py`
4. Update README with usage examples

### Debugging Database Issues

```bash
# Check connection
uv run docs2db db-status

# View logs
uv run docs2db db-logs
uv run docs2db db-logs --follow  # Stream in real-time

# Inspect database directly
podman exec -it docs2db-db psql -U postgres -d ragdb

# Clean slate
uv run docs2db db-destroy
uv run docs2db db-start
```

## Optional Dependencies

Some features require optional dependencies:

**WatsonX support:**
```bash
uv sync --group watsonx
```

## Need Help?

- Check existing issues and discussions
- Review the README for usage examples
- Look at test files for code examples
- Ask questions in pull requests or issues

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
