"""Test configuration utilities for database testing."""

import os
from typing import Dict, Optional


def get_test_db_config() -> Dict[str, str]:
    """Get test database configuration from environment variables or defaults.

    This allows tests to use completely separate credentials from production.
    Set these environment variables to override defaults:
    - TEST_DB_HOST (default: localhost)
    - TEST_DB_PORT (default: 5433)  # Different port from production
    - TEST_DB_NAME (default: test_docs2db)  # Different database name
    - TEST_DB_USER (default: test_user)  # Different user
    - TEST_DB_PASSWORD (default: test_password)  # Different password
    """
    return {
        "host": os.getenv("TEST_DB_HOST", "localhost"),
        "port": os.getenv("TEST_DB_PORT", "5433"),
        "database": os.getenv("TEST_DB_NAME", "test_docs2db"),
        "user": os.getenv("TEST_DB_USER", "test_user"),
        "password": os.getenv("TEST_DB_PASSWORD", "test_password"),
    }


def should_skip_postgres_tests() -> bool:
    """Check if PostgreSQL tests should be skipped.

    Set TEST_SKIP_POSTGRES=1 to skip all PostgreSQL-dependent tests.
    This is useful for CI environments without PostgreSQL available.
    """
    return os.getenv("TEST_SKIP_POSTGRES", "0").lower() in ("1", "true", "yes")
