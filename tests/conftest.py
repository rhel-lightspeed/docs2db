"""Shared pytest fixtures for database tests."""

import psycopg
import pytest
import pytest_asyncio
from psycopg.sql import SQL, Identifier

from tests.test_config import get_test_db_config


@pytest_asyncio.fixture(scope="function", autouse=True)
async def setup_clean_database():
    """Set up a clean database before running database tests.

    This fixture automatically runs for all test classes that need a database.
    It creates a fresh database before tests run and cleans up after.
    """
    config = get_test_db_config()

    # Connect to postgres database to drop/create our test database
    admin_conn_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/postgres"

    async with await psycopg.AsyncConnection.connect(
        admin_conn_string, autocommit=True
    ) as conn:
        # Drop test database if it exists
        await conn.execute(
            SQL("DROP DATABASE IF EXISTS {}").format(Identifier(config["database"]))
        )
        # Create fresh test database
        await conn.execute(
            SQL("CREATE DATABASE {}").format(Identifier(config["database"]))
        )

    yield  # This is where the tests run

    # Cleanup after all tests
    async with await psycopg.AsyncConnection.connect(
        admin_conn_string, autocommit=True
    ) as conn:
        await conn.execute(
            SQL("DROP DATABASE IF EXISTS {}").format(Identifier(config["database"]))
        )
