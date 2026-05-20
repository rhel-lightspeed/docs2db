"""Integration tests for CLI commands."""

import tempfile

from pathlib import Path

import psycopg
import pytest

from structlog.testing import capture_logs
from typer.testing import CliRunner

from docs2db.docs2db import app
from tests.test_config import get_test_db_config
from tests.test_config import should_skip_postgres_tests


runner = CliRunner()


def assert_logged(cap_logs: list, text: str) -> None:
    """Assert that any captured structlog event contains the given text."""
    events = [e["event"] for e in cap_logs]
    assert any(text in ev for ev in events), f"Expected log containing '{text}', got: {events}"


def check_table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the database."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
            (table_name,),
        )
        result = cur.fetchone()
        return result[0] if result else False


def count_records(conn, table_name: str) -> int:
    """Count records in a table."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")  # noqa: S608
        result = cur.fetchone()
        return result[0] if result else 0


class TestCLIIntegrationSQL:
    """Integration tests for CLI commands."""

    @pytest.mark.no_ci
    def test_load_command_initializes_database(self):
        """Test that 'docs2db load' properly initializes database schema.

        This test verifies the complete flow:
        1. Database starts uninitialized (no tables)
        2. CLI load command is executed
        3. Database is properly initialized (tables exist, pgvector extension enabled)
        """
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        try:
            fixtures_content_dir = Path(__file__).parent / "fixtures" / "content" / "documents"

            result = runner.invoke(
                app,
                [
                    "load",
                    "--content-dir",
                    str(fixtures_content_dir),
                    "--model",
                    "ibm-granite/granite-embedding-30m-english",
                    "--pattern",
                    "**",
                    "--host",
                    config["host"],
                    "--port",
                    config["port"],
                    "--db",
                    config["database"],
                    "--user",
                    config["user"],
                    "--password",
                    config["password"],
                    "--force",
                ],
            )

            if result.exit_code != 0:
                pytest.fail(f"CLI load command failed with exit code {result.exit_code}")

            # Connect using psycopg directly
            conn_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"

            with psycopg.Connection.connect(conn_string) as conn:
                # Check that tables were created
                assert check_table_exists(conn, "documents"), "documents table should be created"
                assert check_table_exists(conn, "chunks"), "chunks table should be created"
                assert check_table_exists(conn, "embeddings"), "embeddings table should be created"

                # Check that pgvector extension was enabled
                with conn.cursor() as cur:
                    cur.execute("SELECT '[1,2,3]'::vector")
                    vector_result = cur.fetchone()
                    assert vector_result is not None, "pgvector extension should be enabled"

                # Check that our test data was loaded
                doc_count = count_records(conn, "documents")
                assert doc_count > 0, "At least one document should be loaded"

                chunk_count = count_records(conn, "chunks")
                assert chunk_count > 0, "At least one chunk should be loaded"

                embedding_count = count_records(conn, "embeddings")
                assert embedding_count > 0, "At least one embedding should be loaded"

        except Exception as e:
            pytest.fail(f"Test failed with exception: {e}")

    @pytest.mark.no_ci
    def test_db_status_comprehensive_sql(self):
        """Comprehensive test of db-status command."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()
        conn_string = (
            f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        )

        try:
            db_status_args = [
                "db-status",
                "--host",
                config["host"],
            ]

            # Test 1: Database server down (wrong port)
            with capture_logs() as cap_logs:
                result = runner.invoke(
                    app,
                    db_status_args
                    + [
                        "--port",
                        "9999",  # Non-existent port
                        "--db",
                        config["database"],
                        "--user",
                        config["user"],
                        "--password",
                        config["password"],
                    ],
                )
            assert result.exit_code == 1, "Should exit with error code when server is down"
            assert_logged(cap_logs, "Database")

            # Test 2: Server up but database doesn't exist
            with capture_logs() as cap_logs:
                result = runner.invoke(
                    app,
                    db_status_args
                    + [
                        "--port",
                        config["port"],
                        "--db",
                        "nonexistent_database_name",
                        "--user",
                        config["user"],
                        "--password",
                        config["password"],
                    ],
                )
            assert result.exit_code == 1, "Should exit with error when database doesn't exist"
            assert_logged(cap_logs, "does not exist")

            # Test 3: Database exists but is not initialized (pgvector check fails first)
            with capture_logs() as cap_logs:
                result = runner.invoke(
                    app,
                    db_status_args
                    + [
                        "--port",
                        config["port"],
                        "--db",
                        config["database"],
                        "--user",
                        config["user"],
                        "--password",
                        config["password"],
                    ],
                )
            assert result.exit_code == 1, "Should exit with error when database is not initialized"
            assert_logged(cap_logs, "pgvector extension not installed")

            # Test 4: Load command with empty directory (should initialize schema with no data)
            with tempfile.TemporaryDirectory() as empty_dir:
                load_result = runner.invoke(
                    app,
                    [
                        "load",
                        "--content-dir",
                        empty_dir,
                        "--model",
                        "ibm-granite/granite-embedding-30m-english",
                        "--host",
                        config["host"],
                        "--port",
                        config["port"],
                        "--db",
                        config["database"],
                        "--user",
                        config["user"],
                        "--password",
                        config["password"],
                    ],
                )

                assert load_result.exit_code == 0, "Load should succeed even with empty directory"

            # Verify db-status succeeds after initialization
            with capture_logs() as cap_logs:
                result = runner.invoke(
                    app,
                    db_status_args
                    + [
                        "--port",
                        config["port"],
                        "--db",
                        config["database"],
                        "--user",
                        config["user"],
                        "--password",
                        config["password"],
                    ],
                )
            assert result.exit_code == 0, "Should succeed with initialized empty database"
            assert_logged(cap_logs, "Database connection successful")
            assert_logged(cap_logs, "Database statistics summary")
            assert_logged(cap_logs, "documents : 0")
            assert_logged(cap_logs, "chunks    : 0")
            assert_logged(cap_logs, "embeddings: 0")

            # Verify database state directly: initialized but empty
            with psycopg.Connection.connect(conn_string) as conn:
                assert check_table_exists(conn, "documents"), "documents table should exist"
                assert check_table_exists(conn, "chunks"), "chunks table should exist"
                assert check_table_exists(conn, "embeddings"), "embeddings table should exist"
                assert count_records(conn, "documents") == 0, "Should have 0 documents"
                assert count_records(conn, "chunks") == 0, "Should have 0 chunks"
                assert count_records(conn, "embeddings") == 0, "Should have 0 embeddings"

            # Test 5: Database with actual data
            fixtures_content_dir = Path(__file__).parent / "fixtures" / "content" / "documents"

            load_result = runner.invoke(
                app,
                [
                    "load",
                    "--content-dir",
                    str(fixtures_content_dir),
                    "--model",
                    "ibm-granite/granite-embedding-30m-english",
                    "--host",
                    config["host"],
                    "--port",
                    config["port"],
                    "--db",
                    config["database"],
                    "--user",
                    config["user"],
                    "--password",
                    config["password"],
                ],
            )

            assert load_result.exit_code == 0, "Load should succeed"

            # Verify db-status succeeds with data
            with capture_logs() as cap_logs:
                result = runner.invoke(
                    app,
                    db_status_args
                    + [
                        "--port",
                        config["port"],
                        "--db",
                        config["database"],
                        "--user",
                        config["user"],
                        "--password",
                        config["password"],
                    ],
                )
            assert result.exit_code == 0, "Should succeed with data in database"
            assert_logged(cap_logs, "Database connection successful")
            assert_logged(cap_logs, "documents")

            # Verify database state directly: has data
            with psycopg.Connection.connect(conn_string) as conn:
                assert count_records(conn, "documents") > 0, "Should have non-zero documents"
                assert count_records(conn, "chunks") > 0, "Should have non-zero chunks"
                assert count_records(conn, "embeddings") > 0, "Should have non-zero embeddings"

        except Exception as e:
            pytest.fail(f"Test failed with exception: {e}")

    @pytest.mark.no_ci
    def test_config_command_stores_rag_settings(self):
        """Test that 'docs2db config' properly stores RAG settings in database."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        try:
            # First, initialize database with load command
            fixtures_content_dir = Path(__file__).parent / "fixtures" / "content" / "documents"

            load_result = runner.invoke(
                app,
                [
                    "load",
                    "--content-dir",
                    str(fixtures_content_dir),
                    "--model",
                    "ibm-granite/granite-embedding-30m-english",
                    "--pattern",
                    "**",
                    "--host",
                    config["host"],
                    "--port",
                    config["port"],
                    "--db",
                    config["database"],
                    "--user",
                    config["user"],
                    "--password",
                    config["password"],
                    "--force",
                ],
            )

            assert load_result.exit_code == 0, "Load command should succeed"

            # Connect to database
            conn_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
            conn = psycopg.Connection.connect(conn_string)

            try:
                # Verify rag_settings table exists
                assert check_table_exists(conn, "rag_settings"), (
                    "rag_settings table should exist after schema initialization"
                )

                # Verify rag_settings is initially empty (no settings row)
                initial_count = count_records(conn, "rag_settings")
                assert initial_count == 0, "rag_settings should be empty initially"

                # Now run config command to set some settings
                with capture_logs() as cap_logs:
                    config_result = runner.invoke(
                        app,
                        [
                            "config",
                            "--refinement",
                            "false",
                            "--reranking",
                            "true",
                            "--similarity-threshold",
                            "0.85",
                            "--max-chunks",
                            "20",
                            "--max-tokens-in-context",
                            "8192",
                            "--refinement-questions-count",
                            "3",
                            "--refinement-prompt",
                            "Test custom prompt with {question}",
                            "--host",
                            config["host"],
                            "--port",
                            config["port"],
                            "--db",
                            config["database"],
                            "--user",
                            config["user"],
                            "--password",
                            config["password"],
                        ],
                    )

                assert config_result.exit_code == 0, "Config command should succeed"
                assert_logged(cap_logs, "RAG settings updated successfully")

                # Verify settings were stored in database
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT refinement_prompt, enable_refinement, enable_reranking,
                               similarity_threshold, max_chunks, max_tokens_in_context,
                               refinement_questions_count
                        FROM rag_settings WHERE id = 1
                        """
                    )
                    row = cur.fetchone()

                    assert row is not None, "Settings row should exist after config command"

                    (
                        refinement_prompt,
                        enable_refinement,
                        enable_reranking,
                        similarity_threshold,
                        max_chunks,
                        max_tokens_in_context,
                        refinement_questions_count,
                    ) = row

                    # Verify each setting
                    assert refinement_prompt == "Test custom prompt with {question}", "Refinement prompt should match"
                    assert enable_refinement is False, "enable_refinement should be False"
                    assert enable_reranking is True, "enable_reranking should be True"
                    assert similarity_threshold == 0.85, "similarity_threshold should be 0.85"
                    assert max_chunks == 20, "max_chunks should be 20"
                    assert max_tokens_in_context == 8192, "max_tokens_in_context should be 8192"
                    assert refinement_questions_count == 3, "refinement_questions_count should be 3"

                # Test updating settings (partial update)
                update_result = runner.invoke(
                    app,
                    [
                        "config",
                        "--refinement",
                        "true",
                        "--max-chunks",
                        "15",
                        "--host",
                        config["host"],
                        "--port",
                        config["port"],
                        "--db",
                        config["database"],
                        "--user",
                        config["user"],
                        "--password",
                        config["password"],
                    ],
                )

                assert update_result.exit_code == 0, "Config update should succeed"

                # Verify only specified settings were updated, others remain unchanged
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT enable_refinement, enable_reranking,
                               similarity_threshold, max_chunks, refinement_prompt
                        FROM rag_settings WHERE id = 1
                        """
                    )
                    row = cur.fetchone()

                    assert row is not None, "Settings row should still exist"

                    (
                        enable_refinement,
                        enable_reranking,
                        similarity_threshold,
                        max_chunks,
                        refinement_prompt,
                    ) = row

                    # Updated values
                    assert enable_refinement is True, "enable_refinement should be updated to True"
                    assert max_chunks == 15, "max_chunks should be updated to 15"

                    # Unchanged values
                    assert enable_reranking is True, "enable_reranking should remain True"
                    assert similarity_threshold == 0.85, "similarity_threshold should remain 0.85"
                    assert refinement_prompt == "Test custom prompt with {question}", (
                        "refinement_prompt should remain unchanged"
                    )

                # Test that config command fails when no settings provided
                with capture_logs() as cap_logs:
                    empty_result = runner.invoke(
                        app,
                        [
                            "config",
                            "--host",
                            config["host"],
                            "--port",
                            config["port"],
                            "--db",
                            config["database"],
                            "--user",
                            config["user"],
                            "--password",
                            config["password"],
                        ],
                    )

                assert empty_result.exit_code != 0, "Config command should fail when no settings provided"
                assert_logged(cap_logs, "No settings provided")

                # Test clearing string settings with "None"
                clear_result = runner.invoke(
                    app,
                    [
                        "config",
                        "--refinement-prompt",
                        "None",
                        "--host",
                        config["host"],
                        "--port",
                        config["port"],
                        "--db",
                        config["database"],
                        "--user",
                        config["user"],
                        "--password",
                        config["password"],
                    ],
                )

                assert clear_result.exit_code == 0, "Config command with None should succeed"

                # Verify refinement_prompt was cleared (set to NULL)
                with conn.cursor() as cur:
                    cur.execute("SELECT refinement_prompt FROM rag_settings WHERE id = 1")
                    row = cur.fetchone()
                    assert row is not None, "Settings row should still exist"
                    assert row[0] is None, "refinement_prompt should be NULL after clearing with 'None'"

                # Test clearing boolean and numeric settings with "None"
                clear_all_result = runner.invoke(
                    app,
                    [
                        "config",
                        "--refinement",
                        "None",
                        "--reranking",
                        "None",
                        "--max-chunks",
                        "None",
                        "--similarity-threshold",
                        "None",
                        "--host",
                        config["host"],
                        "--port",
                        config["port"],
                        "--db",
                        config["database"],
                        "--user",
                        config["user"],
                        "--password",
                        config["password"],
                    ],
                )

                assert clear_all_result.exit_code == 0, "Config command to clear all should succeed"

                # Verify all settings were cleared
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT enable_refinement, enable_reranking, max_chunks, similarity_threshold
                        FROM rag_settings WHERE id = 1
                        """
                    )
                    row = cur.fetchone()
                    assert row is not None, "Settings row should still exist"
                    assert row[0] is None, "enable_refinement should be NULL"
                    assert row[1] is None, "enable_reranking should be NULL"
                    assert row[2] is None, "max_chunks should be NULL"
                    assert row[3] is None, "similarity_threshold should be NULL"

            finally:
                conn.close()

        except Exception as e:
            pytest.fail(f"Test failed with exception: {e}")
