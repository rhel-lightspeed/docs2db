.PHONY: test test-ci lint typecheck db-up db-down db-logs db-status db-drop db-dump db-restore db-up-test db-down-test cleanup-workers docs2db list all

# Default source directory for docs2db target
SOURCE ?= tests/fixtures/input

test:
	uv run pytest

test-ci:
	uv run pytest -m "not no_ci"

lint:
	uv run ruff check --fix --exclude content --exclude external_sources
	uv run ruff format --diff --exclude content --exclude external_sources || true
	uv run ruff format --exclude content --exclude external_sources
	uv run pyright src/docs2db

typecheck:
	uv run pyright src/docs2db

db-up:
	podman compose -f postgres-compose.yml --profile prod --profile tools up -d

db-down:
	podman compose -f postgres-compose.yml --profile prod --profile tools down

db-logs:
	podman compose -f postgres-compose.yml logs -f db

db-status:
	uv run docs2db db-status

db-drop:
	$(MAKE) db-down
	podman volume rm docs2db_pgdata || true

db-dump:
	uv run docs2db db-dump

db-restore:
	uv run docs2db db-restore $(FILE)

cleanup-workers:
	uv run docs2db cleanup-workers

docs2db:
	@echo "Starting Docs2DB..."
	@echo "Starting database..."
	$(MAKE) db-up
	@echo "Ingesting documents from $(SOURCE)..."
	uv run docs2db ingest $(SOURCE)
	@echo "Chunking documents..."
	uv run docs2db chunk
	@echo "Generating embeddings..."
	uv run docs2db embed
	@echo "Loading into database..."
	uv run docs2db load
	@echo "Creating database dump..."
	uv run docs2db db-dump
	@echo "Stopping database..."
	$(MAKE) db-down
	@echo "Docs2DB complete, ragdb_dump.sql created."

# Test database targets (using profiles)
db-up-test:
	podman compose -f postgres-compose.yml --profile test up -d

db-down-test:
	podman compose -f postgres-compose.yml --profile test down

db-drop-test:
	$(MAKE) db-down-test
	podman volume rm docs2db_test_pgdata || true

# Run tests with proper test database
test-with-db: db-up-test
	@echo "Waiting for test database to be ready..."
	@sleep 5
	uv run pytest tests/
	$(MAKE) db-down-test

list:
	@echo "Available targets:"
	@echo "  docs2db      - Complete docs2db pipeline (db-up -> ingest -> chunk -> embed -> load -> db-dump -> db-down)"
	@echo "                 Usage: make docs2db [SOURCE=/path/to/documents] (default: tests/fixtures/input)"
	@echo "  test         - Run all tests"
	@echo "  test-ci      - Run CI tests (excluding no_ci marked tests)"
	@echo "  lint         - Run code linting and formatting"
	@echo "  typecheck    - Run type checking"
	@echo "  db-up        - Start production database"
	@echo "  db-down      - Stop production database"
	@echo "  db-status    - Check database status"
	@echo "  db-dump      - Create database dump"
	@echo "  db-restore   - Restore database from dump file"
	@echo "                 Usage: make db-restore FILE=/path/to/dump.sql"
	@echo "  db-drop      - Stop database and remove data"
	@echo "  cleanup-workers - Clean up any orphaned worker processes"

all: lint test typecheck
