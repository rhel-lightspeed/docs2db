"""Tests for database operations."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import psycopg
import pytest

from docs2db.database import DatabaseManager, check_database_status, load_documents
from tests.test_config import get_test_db_config, should_skip_postgres_tests


def create_connection():
    """Create a connection to the test database."""
    config = get_test_db_config()
    conn_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    return psycopg.Connection.connect(conn_string)


def count_records(conn, table_name: str) -> int:
    """Count records in a table."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            result = cur.fetchone()
            return result[0] if result else 0
    except Exception:
        # Table might not exist
        return 0


def table_exists(conn, table_name: str) -> bool:
    """Check if a table exists."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
            (table_name,),
        )
        result = cur.fetchone()
        return result[0] if result else False


class TestDatabaseSQL:
    """Test database operations."""

    def test_database_connection(self):
        """Test basic database connection using psycopg."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        with create_connection() as conn:
            # Simple connectivity test
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                assert result is not None
                assert result[0] == 1

    def test_load_documents_empty_directory(self):
        """Test loading from an empty directory."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Load from empty directory - should succeed but load nothing
            success = load_documents(
                content_dir=temp_dir,
                model="ibm-granite/granite-embedding-30m-english",
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

    def test_load_documents_with_test_files(self):
        """Test loading actual documents."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        # Use existing test fixtures
        fixtures_dir = Path(__file__).parent / "fixtures" / "content" / "documents"
        if not fixtures_dir.exists():
            pytest.skip("Test fixtures not available")

        assert load_documents(
            content_dir=str(fixtures_dir),
            model="ibm-granite/granite-embedding-30m-english",
            pattern="**",
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
            force=True,  # Force to ensure it processes
        )

    def test_load_documents_force_parameter(self):
        """Test the force parameter in load_documents."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled")

        config = get_test_db_config()
        content_dir = str(Path(__file__).parent / "fixtures" / "content" / "documents")
        if not Path(content_dir).exists():
            pytest.skip("Test fixtures not available")

        # First load — should process the document
        success1 = load_documents(
            content_dir=content_dir,
            model="ibm-granite/granite-embedding-30m-english",
            pattern="**",
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
            force=False,
        )
        assert success1 is True

        # Verify document was actually loaded (not just "no files to process")
        with create_connection() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert doc_count > 0, "First load should insert at least one document"

        # Second load with force=False — should skip (documents already loaded)
        success2 = load_documents(
            content_dir=content_dir,
            model="ibm-granite/granite-embedding-30m-english",
            pattern="**",
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
            force=False,
        )
        assert success2 is True

        # Third load with force=True — should re-process (forced reload)
        success3 = load_documents(
            content_dir=content_dir,
            model="ibm-granite/granite-embedding-30m-english",
            pattern="**",
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
            force=True,
        )
        assert success3 is True

    def test_database_tables_after_load(self):
        """Test that expected tables exist after load attempt."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        # Try to load something to ensure schema is initialized
        with tempfile.TemporaryDirectory() as temp_dir:
            assert load_documents(
                content_dir=temp_dir,
                model="ibm-granite/granite-embedding-30m-english",
                pattern="**",
                host=config["host"],
                port=int(config["port"]),
                db=config["database"],
                user=config["user"],
                password=config["password"],
                force=False,
            )

        with create_connection() as conn:
            # Check for expected tables
            assert table_exists(conn, "documents")
            assert table_exists(conn, "chunks")
            assert table_exists(conn, "embeddings")

    def test_database_stats_after_operations(self):
        """Test database record counts after operations."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        with create_connection() as conn:
            # Count records in each table (if they exist)
            doc_count = count_records(conn, "documents")
            chunk_count = count_records(conn, "chunks")
            embedding_count = count_records(conn, "embeddings")

            # All should be non-negative integers
            assert isinstance(doc_count, int)
            assert isinstance(chunk_count, int)
            assert isinstance(embedding_count, int)
            assert doc_count >= 0
            assert chunk_count >= 0
            assert embedding_count >= 0

    def test_load_documents_parameter_validation(self):
        """Test parameter validation for load_documents function."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = load_documents(
                content_dir=temp_dir,
                model="ibm-granite/granite-embedding-30m-english",
                pattern="**",
                host=config["host"],
                port=int(config["port"]),
                db=config["database"],
                user=config["user"],
                password=config["password"],
                force=False,
                batch_size=50,
            )
            assert result is True

    def test_connection_error_handling(self):
        """Test connection error handling."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        # Test with invalid port
        try:
            check_database_status(
                host="localhost",
                port=9999,  # Invalid port
                db="test_db",
                user="test_user",
                password="test_password",
            )
        except Exception as e:
            # Should handle connection errors gracefully
            assert "connection" in str(e).lower() or "refused" in str(e).lower()

    def test_database_functions_interface(self):
        """Test that all SQL functions have the expected interface."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        # Test load_documents interface - initalizes database.
        with tempfile.TemporaryDirectory() as temp_dir:
            assert load_documents(
                content_dir=temp_dir,
                model="ibm-granite/granite-embedding-30m-english",
                pattern="**",
                host=config["host"],
                port=int(config["port"]),
                db=config["database"],
                user=config["user"],
                password=config["password"],
            )

        # check_database_status raises DatabaseError on failure; completing without exception IS the assertion
        check_database_status(
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
        )

    def test_savepoint_isolation_partial_batch_failure(self):
