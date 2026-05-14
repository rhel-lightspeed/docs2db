"""Shared pytest fixtures for database tests."""

import psycopg
import pytest

from psycopg.sql import Identifier
from psycopg.sql import SQL

from tests.test_config import get_test_db_config
from tests.test_config import should_skip_postgres_tests


@pytest.fixture(scope="function", autouse=True)
def setup_clean_database():
    """Set up a clean database before running database tests.

    This fixture automatically runs for all test classes that need a database.
    It creates a fresh database before tests run and cleans up after.
    """
    if should_skip_postgres_tests():
        yield
        return

    config = get_test_db_config()

    # Connect to postgres database to drop/create our test database
    admin_conn_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/postgres"

    with psycopg.Connection.connect(admin_conn_string, autocommit=True) as conn:
        # Drop test database if it exists
        conn.execute(SQL("DROP DATABASE IF EXISTS {}").format(Identifier(config["database"])))
        # Create fresh test database
        conn.execute(SQL("CREATE DATABASE {}").format(Identifier(config["database"])))

    yield  # This is where the tests run

    # Cleanup after all tests
    with psycopg.Connection.connect(admin_conn_string, autocommit=True) as conn:
        conn.execute(SQL("DROP DATABASE IF EXISTS {}").format(Identifier(config["database"])))
