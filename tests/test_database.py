"""Tests for database operations."""

import json
import tempfile

from pathlib import Path
from unittest.mock import patch

import psycopg
import pytest

from docs2db.database import check_database_status
from docs2db.database import DatabaseManager
from docs2db.database import load_documents
from tests.test_config import get_test_db_config
from tests.test_config import should_skip_postgres_tests


def create_connection():
    """Create a connection to the test database."""
    config = get_test_db_config()
    conn_string = (
        f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    )
    return psycopg.Connection.connect(conn_string)


def count_records(conn, table_name: str) -> int:
    """Count records in a table."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")  # noqa: S608
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

        with create_connection() as conn:  # noqa: SIM117
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

        try:
            with create_connection() as conn:
                doc_count_before = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        except psycopg.errors.UndefinedTable:
            doc_count_before = 0

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

        with create_connection() as conn:
            doc_count_after = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert doc_count_after > doc_count_before, "First load should insert at least one document"

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

        with create_connection() as conn:
            doc_count_after_skip = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert doc_count_after_skip == doc_count_after, "force=False should not insert additional documents"

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
                password="test_password",  # noqa: S106
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
        """Test that savepoint isolation allows partial batch success.

        When one document in a batch fails during the DB insertion phase,
        the SAVEPOINT/ROLLBACK TO SAVEPOINT mechanism catches the error for
        that document only, allowing the other documents in the batch to
        succeed. This verifies the per-document savepoint/rollback logic
        in load_document_batch.
        """
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        fixtures_dir = Path(__file__).parent / "fixtures" / "content" / "documents"
        if not fixtures_dir.exists():
            pytest.skip("Test fixtures not available")

        config = get_test_db_config()
        db_manager = DatabaseManager(
            host=config["host"],
            port=int(config["port"]),
            database=config["database"],
            user=config["user"],
            password=config["password"],
        )
        db_manager.initialize_schema()

        good_path = "good_doc/source.json"
        bad_path = "bad_doc/source.json"

        with tempfile.TemporaryDirectory() as temp_dir:
            content_dir = Path(temp_dir)

            # --- Good document: will succeed ---
            good_dir = content_dir / "good_doc"
            good_dir.mkdir()
            good_source = good_dir / "source.json"
            good_source.write_text(json.dumps({"title": "Good Doc", "content": "Content."}))
            good_chunks_file = good_dir / "chunks.json"
            good_chunks_file.write_text(
                json.dumps(
                    {
                        "chunks": [
                            {
                                "text": "Good document text for savepoint test.",
                                "contextual_text": "Good document text for savepoint test.",
                                "metadata": {"chunk_index": 0},
                            }
                        ],
                        "model": "ibm-granite/granite-embedding-30m-english",
                    }
                )
            )
            good_embedding_data = {"embeddings": [[0.1] * 384]}
            good_emb_file = good_dir / "gran.json"
            good_emb_file.write_text(json.dumps(good_embedding_data))

            # --- Bad document: will fail during INSERT ---
            bad_dir = content_dir / "bad_doc"
            bad_dir.mkdir()
            bad_source = bad_dir / "source.json"
            bad_source.write_text(json.dumps({"title": "Bad Doc", "content": "Content."}))
            bad_chunks_file = bad_dir / "chunks.json"
            bad_chunks_file.write_text(
                json.dumps(
                    {
                        "chunks": [
                            {
                                "text": "Bad document text for savepoint test.",
                                "contextual_text": "Bad document text for savepoint test.",
                                "metadata": {"chunk_index": 0},
                            }
                        ],
                        "model": "ibm-granite/granite-embedding-30m-english",
                    }
                )
            )
            bad_embedding_data = {"embeddings": [[0.1] * 384]}
            bad_emb_file = bad_dir / "gran.json"
            bad_emb_file.write_text(json.dumps(bad_embedding_data))

            model = "ibm-granite/granite-embedding-30m-english"
            files_data = [
                (
                    good_source,
                    good_chunks_file,
                    model,
                    good_embedding_data,
                    good_emb_file,
                ),
                (
                    bad_source,
                    bad_chunks_file,
                    model,
                    bad_embedding_data,
                    bad_emb_file,
                ),
            ]

            # Connection wrapper that simulates a DB failure for the bad_doc
            # INSERT while letting all other operations (including SAVEPOINT
            # commands, which use Composed SQL objects) pass through normally.
            original_get_conn = db_manager.get_direct_connection
            call_count = [0]

            class _FailOnBadDoc:
                """Wraps a real DB connection, raising on INSERT for 'bad_doc'."""

                def __init__(self, conn):
                    self._conn = conn

                def execute(self, sql, params=None, *args, **kwargs):
                    if (
                        isinstance(sql, str)
                        and "INSERT INTO documents" in sql
                        and params
                        and "bad_doc" in str(params[0])
                    ):
                        raise RuntimeError("Simulated DB failure for savepoint isolation test")
                    return self._conn.execute(sql, params, *args, **kwargs)

                def commit(self):
                    return self._conn.commit()

                def rollback(self):
                    return self._conn.rollback()

                def __enter__(self):
                    self._conn.__enter__()
                    return self

                def __exit__(self, *args):
                    return self._conn.__exit__(*args)

            def mock_get_conn():
                call_count[0] += 1
                conn = original_get_conn()
                # First call is for insert_model; wrap only the bulk-ops call
                return _FailOnBadDoc(conn) if call_count[0] > 1 else conn

            try:
                with patch.object(
                    db_manager,
                    "get_direct_connection",
                    side_effect=mock_get_conn,
                ):
                    processed, errors = db_manager.load_document_batch(files_data, content_dir, force=True)

                # Partial success: good_doc processed, bad_doc errored
                assert processed > 0, f"Expected at least one document processed, got {processed}"
                assert errors > 0, f"Expected at least one error, got {errors}"

                # The successful document must be in the database
                with create_connection() as conn:  # noqa: SIM117
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT path FROM documents WHERE path = %s",
                            (good_path,),
                        )
                        assert cur.fetchone() is not None, f"Good document '{good_path}' not found in documents table"
            finally:
                # Clean up test documents
                with create_connection() as conn:
                    conn.execute(
                        "DELETE FROM documents WHERE path IN (%s, %s)",
                        (good_path, bad_path),
                    )
                    conn.commit()

    def test_skip_check_uses_relative_path(self):
        """Test that the up-to-date check matches DB paths correctly.

        The skip-check query (force=False path) must compare against the
        document's relative path — the same format stored in the documents
        table.  Before the fix, it compared absolute paths, so the query
        never matched and every document was needlessly re-processed.

        Verifies the fix by loading a document once (force=True), then
        re-loading with force=False and asserting it is skipped.
        """
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()
        db_manager = DatabaseManager(
            host=config["host"],
            port=int(config["port"]),
            database=config["database"],
            user=config["user"],
            password=config["password"],
        )
        db_manager.initialize_schema()

        doc_rel_path = "skipcheck_doc/source.json"

        with tempfile.TemporaryDirectory() as temp_dir:
            content_dir = Path(temp_dir)

            # Create a single document with chunk + embedding
            doc_dir = content_dir / "skipcheck_doc"
            doc_dir.mkdir()
            source = doc_dir / "source.json"
            source.write_text(json.dumps({"title": "Skip Check Doc", "content": "Content."}))
            chunks_file = doc_dir / "chunks.json"
            chunks_file.write_text(
                json.dumps(
                    {
                        "chunks": [
                            {
                                "text": "Text for skip-check path test.",
                                "contextual_text": "Text for skip-check path test.",
                                "metadata": {"chunk_index": 0},
                            }
                        ],
                        "model": "ibm-granite/granite-embedding-30m-english",
                    }
                )
            )
            embedding_data = {"embeddings": [[0.1] * 384]}
            emb_file = doc_dir / "gran.json"
            emb_file.write_text(json.dumps(embedding_data))

            model = "ibm-granite/granite-embedding-30m-english"
            files_data = [
                (source, chunks_file, model, embedding_data, emb_file),
            ]

            try:
                # --- First load: force=True inserts the document ---
                processed_1, errors_1 = db_manager.load_document_batch(files_data, content_dir, force=True)
                assert errors_1 == 0, f"First load had unexpected errors: {errors_1}"
                assert processed_1 == 1, f"Expected 1 document processed on first load, got {processed_1}"

                # Verify the document is stored with a relative path
                with create_connection() as conn:  # noqa: SIM117
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT path FROM documents WHERE path = %s",
                            (doc_rel_path,),
                        )
                        assert cur.fetchone() is not None, f"Document not found at relative path '{doc_rel_path}'"

                # --- Second load: force=False should skip (already up to date) ---
                processed_2, errors_2 = db_manager.load_document_batch(files_data, content_dir, force=False)
                assert errors_2 == 0, f"Second load had unexpected errors: {errors_2}"
                assert processed_2 == 0, (
                    f"Expected 0 documents processed (skip), got {processed_2}. "
                    "The up-to-date check likely failed to match the relative path."
                )
            finally:
                with create_connection() as conn:
                    conn.execute(
                        "DELETE FROM documents WHERE path = %s",
                        (doc_rel_path,),
                    )
                    conn.commit()
