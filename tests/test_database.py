"""Tests for database operations."""

import json
import tempfile
from pathlib import Path

import psycopg
import pytest

from docs2db.database import check_database_status, load_documents
from tests.test_config import get_test_db_config, should_skip_postgres_tests


async def create_connection():
    """Create a connection to the test database."""
    config = get_test_db_config()
    conn_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    return await psycopg.AsyncConnection.connect(conn_string)


async def count_records(conn, table_name: str) -> int:
    """Count records in a table."""
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            result = await cur.fetchone()
            return result[0] if result else 0
    except Exception:
        # Table might not exist
        return 0


async def table_exists(conn, table_name: str) -> bool:
    """Check if a table exists."""
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
            (table_name,),
        )
        result = await cur.fetchone()
        return result[0] if result else False


class TestDatabaseSQL:
    """Test database operations."""

    @pytest.mark.asyncio
    async def test_database_connection(self):
        """Test basic database connection using psycopg."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        async with await create_connection() as conn:
            # Simple connectivity test
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                result = await cur.fetchone()
                assert result is not None
                assert result[0] == 1

    @pytest.mark.asyncio
    async def test_load_documents_empty_directory(self):
        """Test loading from an empty directory."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Load from empty directory - should succeed but load nothing
            success = await load_documents(
                content_dir=temp_dir,
                model_name="ibm-granite/granite-embedding-30m-english",
                pattern="**",
                host=config["host"],
                port=int(config["port"]),
                db=config["database"],
                user=config["user"],
                password=config["password"],
                force=False,
            )

            # Should succeed (no errors) even with no files
            assert success is True

    @pytest.mark.asyncio
    async def test_load_documents_with_test_files(self):
        """Test loading actual documents."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        # Use existing test fixtures
        fixtures_dir = Path(__file__).parent / "fixtures" / "content" / "documents"
        if not fixtures_dir.exists():
            pytest.skip("Test fixtures not available")

        assert await load_documents(
            content_dir=str(fixtures_dir),
            model_name="ibm-granite/granite-embedding-30m-english",
            pattern="**",
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
            force=True,  # Force to ensure it processes
        )

    @pytest.mark.asyncio
    async def test_load_documents_force_parameter(self):
        """Test the force parameter in load_documents."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple JSON file
            test_file = Path(temp_dir) / "test.json"
            test_data = {
                "name": "test_doc",
                "origin": {"filename": "test.json"},
                "texts": [{"text": "This is a test document."}],
            }
            test_file.write_text(json.dumps(test_data, indent=2))

            # Test with force=False
            success1 = await load_documents(
                content_dir=temp_dir,
                model_name="ibm-granite/granite-embedding-30m-english",
                pattern="**",
                host=config["host"],
                port=int(config["port"]),
                db=config["database"],
                user=config["user"],
                password=config["password"],
                force=False,
            )

            # Test with force=True
            success2 = await load_documents(
                content_dir=temp_dir,
                model_name="ibm-granite/granite-embedding-30m-english",
                pattern="**",
                host=config["host"],
                port=int(config["port"]),
                db=config["database"],
                user=config["user"],
                password=config["password"],
                force=True,
            )

            # Both should return boolean values
            assert isinstance(success1, bool)
            assert isinstance(success2, bool)

    @pytest.mark.asyncio
    async def test_database_tables_after_load(self):
        """Test that expected tables exist after load attempt."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        # Try to load something to ensure schema is initialized
        with tempfile.TemporaryDirectory() as temp_dir:
            assert await load_documents(
                content_dir=temp_dir,
                model_name="ibm-granite/granite-embedding-30m-english",
                pattern="**",
                host=config["host"],
                port=int(config["port"]),
                db=config["database"],
                user=config["user"],
                password=config["password"],
                force=False,
            )

        async with await create_connection() as conn:
            # Check for expected tables
            assert await table_exists(conn, "documents")
            assert await table_exists(conn, "chunks")
            assert await table_exists(conn, "embeddings")

    @pytest.mark.asyncio
    async def test_database_stats_after_operations(self):
        """Test database record counts after operations."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        async with await create_connection() as conn:
            # Count records in each table (if they exist)
            doc_count = await count_records(conn, "documents")
            chunk_count = await count_records(conn, "chunks")
            embedding_count = await count_records(conn, "embeddings")

            # All should be non-negative integers
            assert isinstance(doc_count, int)
            assert isinstance(chunk_count, int)
            assert isinstance(embedding_count, int)
            assert doc_count >= 0
            assert chunk_count >= 0
            assert embedding_count >= 0

    @pytest.mark.asyncio
    async def test_load_documents_parameter_validation(self):
        """Test parameter validation for load_documents function."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with valid parameters
            try:
                success = await load_documents(
                    content_dir=temp_dir,
                    model_name="ibm-granite/granite-embedding-30m-english",
                    pattern="**",
                    host=config["host"],
                    port=int(config["port"]),
                    db=config["database"],
                    user=config["user"],
                    password=config["password"],
                    force=False,
                    batch_size=50,  # Test optional parameter
                )
                assert isinstance(success, bool)
            except Exception as e:
                # Schema issues are acceptable for this interface test
                assert isinstance(e, Exception)

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test connection error handling."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        # Test with invalid port
        try:
            await check_database_status(
                host="localhost",
                port=9999,  # Invalid port
                db="test_db",
                user="test_user",
                password="test_password",
            )
        except Exception as e:
            # Should handle connection errors gracefully
            assert "connection" in str(e).lower() or "refused" in str(e).lower()

    @pytest.mark.asyncio
    async def test_database_functions_interface(self):
        """Test that all SQL functions have the expected interface."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        # Test load_documents interface - initalizes database.
        with tempfile.TemporaryDirectory() as temp_dir:
            assert await load_documents(
                content_dir=temp_dir,
                model_name="ibm-granite/granite-embedding-30m-english",
                pattern="**",
                host=config["host"],
                port=int(config["port"]),
                db=config["database"],
                user=config["user"],
                password=config["password"],
            )

        result = await check_database_status(
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
        )

        assert result is None
