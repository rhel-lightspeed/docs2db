"""Integration tests for CLI commands."""

import subprocess
import time
from pathlib import Path

import psycopg
import pytest

from tests.test_config import get_test_db_config, should_skip_postgres_tests

# Get project root directory dynamically
PROJECT_ROOT = Path(__file__).parent.parent


async def check_table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the database."""
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
            (table_name,),
        )
        result = await cur.fetchone()
        return result[0] if result else False


async def count_records(conn, table_name: str) -> int:
    """Count records in a table."""
    async with conn.cursor() as cur:
        await cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        result = await cur.fetchone()
        return result[0] if result else 0


class TestCLIIntegrationSQL:
    """Integration tests for CLI commands."""

    @pytest.mark.no_ci
    @pytest.mark.asyncio
    async def test_load_command_initializes_database(self):
        """Test that 'uv run docs2db load' properly initializes database schema.

        This test verifies the complete flow:
        1. Database starts uninitialized (no tables)
        2. CLI load command is executed
        3. Database is properly initialized (tables exist, pgvector extension enabled)
        """
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        try:
            fixtures_content_dir = (
                Path(__file__).parent / "fixtures" / "content" / "documents"
            )

            cmd = [
                "uv",
                "run",
                "docs2db",
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
                "--force",  # Force to ensure it processes our test file
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )

            if result.returncode != 0:
                pytest.fail(
                    f"CLI load command failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                )

            # Connect using psycopg directly
            conn_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"

            async with await psycopg.AsyncConnection.connect(conn_string) as conn:
                # Check that tables were created
                assert await check_table_exists(conn, "documents"), (
                    "documents table should be created"
                )
                assert await check_table_exists(conn, "chunks"), (
                    "chunks table should be created"
                )
                assert await check_table_exists(conn, "embeddings"), (
                    "embeddings table should be created"
                )

                # Check that pgvector extension was enabled
                async with conn.cursor() as cur:
                    await cur.execute("SELECT '[1,2,3]'::vector")
                    vector_result = await cur.fetchone()
                    assert vector_result is not None, (
                        "pgvector extension should be enabled"
                    )

                # Check that our test data was loaded
                doc_count = await count_records(conn, "documents")
                assert doc_count > 0, "At least one document should be loaded"

                chunk_count = await count_records(conn, "chunks")
                assert chunk_count > 0, "At least one chunk should be loaded"

                embedding_count = await count_records(conn, "embeddings")
                assert embedding_count > 0, "At least one embedding should be loaded"

        except Exception as e:
            pytest.fail(f"Test failed with exception: {e}")

    @pytest.mark.no_ci
    @pytest.mark.asyncio
    async def test_db_status_comprehensive_sql(self):
        """Comprehensive test of db-status command."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        try:
            # Test 1: Database server down (wrong port)
            cmd_base = ["uv", "run", "docs2db", "db-status"]

            result = subprocess.run(
                cmd_base
                + [
                    "--host",
                    config["host"],
                    "--port",
                    "9999",  # Non-existent port
                    "--db",
                    config["database"],
                    "--user",
                    config["user"],
                    "--password",
                    config["password"],
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            assert result.returncode == 1, (
                "Should exit with error code when server is down"
            )
            assert "Database connection failed" in result.stdout, (
                "Should show database connection failed when server is down"
            )
            assert "does not exist" not in result.stdout, (
                "Should not mention database existence when server is down"
            )
            assert (
                "Traceback" not in result.stdout and "Traceback" not in result.stderr
            ), "Should not show traceback for expected error"

            # Test 2: Server up but database doesn't exist
            result = subprocess.run(
                cmd_base
                + [
                    "--host",
                    config["host"],
                    "--port",
                    config["port"],
                    "--db",
                    "nonexistent_database_name",
                    "--user",
                    config["user"],
                    "--password",
                    config["password"],
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            assert result.returncode == 1, (
                "Should exit with error when database doesn't exist"
            )
            assert "does not exist" in result.stdout, (
                "Should show database doesn't exist error"
            )
            assert (
                "Traceback" not in result.stdout and "Traceback" not in result.stderr
            ), "Should not show traceback for expected error"

            # Test 3: Database exists but is not initialized
            result = subprocess.run(
                cmd_base
                + [
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
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            # Should report that database exists but is not initialized
            assert result.returncode == 1, (
                "Should exit with error when database is not initialized"
            )
            assert "Database connection successful" in result.stdout, (
                "Should show connection success"
            )
            # Should indicate that the database is not initialized (no tables exist)
            assert (
                "not initialized" in result.stdout.lower()
                or "initialize the schema" in result.stdout.lower()
                or "run 'uv run docs2db load'" in result.stdout.lower()
                or "pgvector extension not installed" in result.stdout.lower()
            ), "Should indicate database is not initialized"

            # Test 4: Load command with empty directory (should initialize schema with no data)
            import tempfile

            with tempfile.TemporaryDirectory() as empty_dir:
                load_result = subprocess.run(
                    [
                        "uv",
                        "run",
                        "docs2db",
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
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                )

                assert load_result.returncode == 0, (
                    f"Load should succeed even with empty directory: {load_result.stdout}"
                )

            # Now test db-status - should show initialized database with 0 counts
            result = subprocess.run(
                cmd_base
                + [
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
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            assert result.returncode == 0, (
                "Should succeed with initialized empty database"
            )
            assert "Database connection successful" in result.stdout, (
                "Should show success message"
            )
            assert "documents : 0" in result.stdout, "Should show 0 documents"
            assert "chunks    : 0" in result.stdout, "Should show 0 chunks"
            assert "embeddings: 0" in result.stdout, "Should show 0 embeddings"

            # Test 5: Database with actual data
            fixtures_content_dir = (
                Path(__file__).parent / "fixtures" / "content" / "documents"
            )

            load_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "docs2db",
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
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )

            assert load_result.returncode == 0, (
                f"Load should succeed: {load_result.stdout}"
            )

            # Now test db-status with data
            result = subprocess.run(
                cmd_base
                + [
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
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            assert result.returncode == 0, "Should succeed with data in database"
            assert "Database connection successful" in result.stdout, (
                "Should show success message"
            )
            # Should show non-zero counts
            assert (
                "documents : " in result.stdout and "documents : 0" not in result.stdout
            ), "Should show non-zero documents"
            assert (
                "chunks    : " in result.stdout and "chunks    : 0" not in result.stdout
            ), "Should show non-zero chunks"
            assert (
                "embeddings: " in result.stdout and "embeddings: 0" not in result.stdout
            ), "Should show non-zero embeddings"

        except Exception as e:
            pytest.fail(f"Test failed with exception: {e}")

    @pytest.mark.no_ci
    @pytest.mark.asyncio
    async def test_config_command_stores_rag_settings(self):
        """Test that 'uv run docs2db config' properly stores RAG settings in database."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        try:
            # First, initialize database with load command
            fixtures_content_dir = (
                Path(__file__).parent / "fixtures" / "content" / "documents"
            )

            load_cmd = [
                "uv",
                "run",
                "docs2db",
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
            ]

            load_result = subprocess.run(
                load_cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )

            assert load_result.returncode == 0, (
                f"Load command should succeed: {load_result.stderr}"
            )

            # Connect to database
            conn_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
            conn = await psycopg.AsyncConnection.connect(conn_string)

            try:
                # Verify rag_settings table exists
                assert await check_table_exists(conn, "rag_settings"), (
                    "rag_settings table should exist after schema initialization"
                )

                # Verify rag_settings is initially empty (no settings row)
                initial_count = await count_records(conn, "rag_settings")
                assert initial_count == 0, "rag_settings should be empty initially"

                # Now run config command to set some settings
                config_cmd = [
                    "uv",
                    "run",
                    "docs2db",
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
                ]

                config_result = subprocess.run(
                    config_cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                )

                assert config_result.returncode == 0, (
                    f"Config command should succeed: {config_result.stderr}"
                )
                assert "RAG settings updated successfully" in config_result.stdout, (
                    "Should show success message"
                )

                # Verify settings were stored in database
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT refinement_prompt, enable_refinement, enable_reranking,
                               similarity_threshold, max_chunks, max_tokens_in_context,
                               refinement_questions_count
                        FROM rag_settings WHERE id = 1
                        """
                    )
                    row = await cur.fetchone()

                    assert row is not None, (
                        "Settings row should exist after config command"
                    )

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
                    assert refinement_prompt == "Test custom prompt with {question}", (
                        "Refinement prompt should match"
                    )
                    assert enable_refinement is False, (
                        "enable_refinement should be False"
                    )
                    assert enable_reranking is True, "enable_reranking should be True"
                    assert similarity_threshold == 0.85, (
                        "similarity_threshold should be 0.85"
                    )
                    assert max_chunks == 20, "max_chunks should be 20"
                    assert max_tokens_in_context == 8192, (
                        "max_tokens_in_context should be 8192"
                    )
                    assert refinement_questions_count == 3, (
                        "refinement_questions_count should be 3"
                    )

                # Test updating settings (partial update)
                update_cmd = [
                    "uv",
                    "run",
                    "docs2db",
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
                ]

                update_result = subprocess.run(
                    update_cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                )

                assert update_result.returncode == 0, (
                    f"Config update should succeed: {update_result.stderr}"
                )

                # Verify only specified settings were updated, others remain unchanged
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT enable_refinement, enable_reranking,
                               similarity_threshold, max_chunks, refinement_prompt
                        FROM rag_settings WHERE id = 1
                        """
                    )
                    row = await cur.fetchone()

                    assert row is not None, "Settings row should still exist"

                    (
                        enable_refinement,
                        enable_reranking,
                        similarity_threshold,
                        max_chunks,
                        refinement_prompt,
                    ) = row

                    # Updated values
                    assert enable_refinement is True, (
                        "enable_refinement should be updated to True"
                    )
                    assert max_chunks == 15, "max_chunks should be updated to 15"

                    # Unchanged values
                    assert enable_reranking is True, (
                        "enable_reranking should remain True"
                    )
                    assert similarity_threshold == 0.85, (
                        "similarity_threshold should remain 0.85"
                    )
                    assert refinement_prompt == "Test custom prompt with {question}", (
                        "refinement_prompt should remain unchanged"
                    )

                # Test that config command fails when no settings provided
                empty_cmd = [
                    "uv",
                    "run",
                    "docs2db",
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
                ]

                empty_result = subprocess.run(
                    empty_cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                )

                assert empty_result.returncode != 0, (
                    "Config command should fail when no settings provided"
                )
                assert "No settings provided" in empty_result.stdout, (
                    "Should show error message about no settings"
                )

                # Test clearing string settings with "None"
                clear_prompt_cmd = [
                    "uv",
                    "run",
                    "docs2db",
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
                ]

                clear_result = subprocess.run(
                    clear_prompt_cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                )

                assert clear_result.returncode == 0, (
                    f"Config command with None should succeed: {clear_result.stderr}"
                )

                # Verify refinement_prompt was cleared (set to NULL)
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT refinement_prompt FROM rag_settings WHERE id = 1"
                    )
                    row = await cur.fetchone()
                    assert row is not None, "Settings row should still exist"
                    assert row[0] is None, (
                        "refinement_prompt should be NULL after clearing with 'None'"
                    )

                # Test clearing boolean and numeric settings with "None"
                clear_all_cmd = [
                    "uv",
                    "run",
                    "docs2db",
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
                ]

                clear_all_result = subprocess.run(
                    clear_all_cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                )

                assert clear_all_result.returncode == 0, (
                    f"Config command to clear all should succeed: {clear_all_result.stderr}"
                )

                # Verify all settings were cleared
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT enable_refinement, enable_reranking, max_chunks, similarity_threshold
                        FROM rag_settings WHERE id = 1
                        """
                    )
                    row = await cur.fetchone()
                    assert row is not None, "Settings row should still exist"
                    assert row[0] is None, "enable_refinement should be NULL"
                    assert row[1] is None, "enable_reranking should be NULL"
                    assert row[2] is None, "max_chunks should be NULL"
                    assert row[3] is None, "similarity_threshold should be NULL"

            finally:
                await conn.close()

        except Exception as e:
            pytest.fail(f"Test failed with exception: {e}")
