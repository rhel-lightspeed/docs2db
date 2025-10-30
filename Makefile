.PHONY: test test-ci db-up-test db-down-test db-destroy-test test-with-db list

test:
	uv run pytest

test-ci:
	uv run pytest -m "not no_ci"

# Test database targets (separate from production database)
# Uses --profile test, different port (5433), and separate volumes
db-up-test:
	podman compose -f postgres-compose.yml --profile test up -d

db-down-test:
	podman compose -f postgres-compose.yml --profile test down

db-destroy-test:
	$(MAKE) db-down-test
	podman volume rm docs2db_test_pgdata || true

list:
	@echo "Available targets:"
	@echo ""
	@echo "Testing:"
	@echo "  test         - Run all tests"
	@echo "  test-ci      - Run CI tests (excluding no_ci marked tests)"
	@echo ""
	@echo "Test Database (port 5433):"
	@echo "  db-up-test      - Start test database"
	@echo "  db-down-test    - Stop test database"
	@echo "  db-destroy-test - Stop test database and remove volumes"
	@echo ""
	@echo "Linting & Formatting:"
	@echo "  pre-commit run --all-files  - Run all checks (ruff, pyright, etc.)"
	@echo "  pre-commit run              - Run checks on staged files only"
	@echo ""
	@echo "CLI Commands (use these for development):"
	@echo "  docs2db pipeline <path>     - Complete workflow"
	@echo "  docs2db db-start            - Start database"
	@echo "  docs2db db-stop             - Stop database"
	@echo "  docs2db --help              - See all available commands"
