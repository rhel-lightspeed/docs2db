"""Tests for schema metadata and change tracking."""

import tempfile
from pathlib import Path

import psycopg
import pytest

from docs2db.database import DatabaseManager, load_documents
from tests.test_config import get_test_db_config, should_skip_postgres_tests


async def create_connection():
    """Create a connection to the test database."""
    config = get_test_db_config()
    conn_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    return await psycopg.AsyncConnection.connect(conn_string)


async def get_schema_metadata(conn):
    """Get schema_metadata record."""
    async with conn.cursor() as cur:
        await cur.execute("SELECT * FROM schema_metadata WHERE id = 1")
        result = await cur.fetchone()
        if result:
            columns = [desc[0] for desc in cur.description]
            return dict(zip(columns, result))
    return None


async def get_schema_changes(conn):
    """Get all schema_changes records."""
    async with conn.cursor() as cur:
        await cur.execute("SELECT * FROM schema_changes ORDER BY id")
        results = await cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in results]


class TestSchemaMetadata:
    """Test schema metadata functionality."""

    @pytest.mark.asyncio
    async def test_metadata_lifecycle(self):
        """Test complete lifecycle: create, no change, update metadata across multiple loads."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()
        fixtures_dir = Path(__file__).parent / "fixtures" / "content" / "documents"

        if not fixtures_dir.exists():
            pytest.skip("Test fixtures not available")

        # === FIRST LOAD: Create metadata ===
        success = await load_documents(
            content_dir=str(fixtures_dir),
            model_name="granite-30m-english",
            pattern="**/source.json",
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
            force=True,
            username="test_user",
            title="Test Database",
            description="Test Database for First Load",
            note="Initial test load",
        )

        assert success is True

        # Verify schema_metadata was created
        async with await create_connection() as conn:
            metadata = await get_schema_metadata(conn)
            changes = await get_schema_changes(conn)

            assert metadata is not None
            assert metadata["id"] == 1
            assert metadata["title"] == "Test Database"
            assert metadata["description"] == "Test Database for First Load"
            assert metadata["schema_version"] == "1.0.0"
            assert metadata["embedding_models_count"] >= 1

            # Should have 2 records: creation + first load
            assert len(changes) == 2
            assert changes[0]["notes"] == "Database initialized"
            assert changes[1]["changed_by_user"] == "test_user"
            # User provided note should be used
            assert changes[1]["notes"] == "Initial test load"

        # === SECOND LOAD: No metadata changes ===
        success = await load_documents(
            content_dir=str(fixtures_dir),
            model_name="granite-30m-english",
            pattern="**/source.json",
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
            force=True,
            username="test_user2",
            # No description or note provided
        )

        assert success is True

        # Verify metadata unchanged (None values don't update)
        async with await create_connection() as conn:
            metadata = await get_schema_metadata(conn)
            changes = await get_schema_changes(conn)

            assert metadata is not None
            assert metadata["title"] == "Test Database"  # Unchanged
            assert (
                metadata["description"] == "Test Database for First Load"
            )  # Unchanged

            # Should now have 3 records
            assert len(changes) == 3
            assert changes[2]["changed_by_user"] == "test_user2"

        # === THIRD LOAD: Update metadata ===
        success = await load_documents(
            content_dir=str(fixtures_dir),
            model_name="granite-30m-english",
            pattern="**/source.json",
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
            force=True,
            username="test_user3",
            title="Updated Test Database",
            description="Updated Test Database Description",
            note="Third load with updates",
        )

        assert success is True

        # Verify metadata was updated
        async with await create_connection() as conn:
            metadata = await get_schema_metadata(conn)
            changes = await get_schema_changes(conn)

            assert metadata is not None
            assert metadata["title"] == "Updated Test Database"
            assert metadata["description"] == "Updated Test Database Description"

            # Should now have 4 records
            assert len(changes) == 4
            assert changes[3]["changed_by_user"] == "test_user3"
            # User provided note should be used
            assert changes[3]["notes"] == "Third load with updates"

    @pytest.mark.asyncio
    async def test_embedding_models_count_updates(self):
        """Test that embedding_models_count is updated correctly."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        # Initialize schema first
        db_manager = DatabaseManager(
            host=config["host"],
            port=int(config["port"]),
            database=config["database"],
            user=config["user"],
            password=config["password"],
        )
        await db_manager.initialize_schema()

        async with await create_connection() as conn:
            metadata = await get_schema_metadata(conn)

            assert metadata is not None
            # Count should be 0 initially (no embeddings loaded yet)
            assert metadata["embedding_models_count"] == 0

            # Get actual count from database (from models table now)
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM models")
                result = await cur.fetchone()
                actual_count = result[0] if result else 0

            # Should match
            assert metadata["embedding_models_count"] == actual_count

    @pytest.mark.asyncio
    async def test_schema_change_timestamps(self):
        """Test that schema_changes have valid timestamps."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        # Initialize schema first
        db_manager = DatabaseManager(
            host=config["host"],
            port=int(config["port"]),
            database=config["database"],
            user=config["user"],
            password=config["password"],
        )
        await db_manager.initialize_schema()

        async with await create_connection() as conn:
            changes = await get_schema_changes(conn)

            assert len(changes) > 0

            # All records should have timestamps
            for change in changes:
                assert change["changed_at"] is not None

            # Timestamps should be in chronological order
            for i in range(1, len(changes)):
                assert changes[i]["changed_at"] >= changes[i - 1]["changed_at"]

    @pytest.mark.asyncio
    async def test_helper_methods_directly(self):
        """Test insert_schema_metadata, update_schema_metadata, and insert_schema_change directly."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        # Create database manager
        db_manager = DatabaseManager(
            host=config["host"],
            port=int(config["port"]),
            database=config["database"],
            user=config["user"],
            password=config["password"],
        )

        # Initialize schema first
        await db_manager.initialize_schema()

        async with await db_manager.get_direct_connection() as conn:
            # Delete the existing metadata record so we can test insert
            await conn.execute("DELETE FROM schema_metadata WHERE id = 1")
            await conn.commit()

            # Test insert_schema_metadata
            await db_manager.insert_schema_metadata(
                conn,
                title="Test DB",
                description="Test Description",
            )
            await conn.commit()

            metadata = await get_schema_metadata(conn)
            assert metadata is not None
            assert metadata["title"] == "Test DB"
            assert metadata["description"] == "Test Description"

            # Test update_schema_metadata
            await db_manager.update_schema_metadata(
                conn, title="Updated DB", embedding_models_count=5
            )
            await conn.commit()

            metadata = await get_schema_metadata(conn)
            assert metadata is not None
            assert metadata["title"] == "Updated DB"
            assert metadata["description"] == "Test Description"  # Unchanged
            assert metadata["embedding_models_count"] == 5

            # Test insert_schema_change
            await db_manager.insert_schema_change(
                conn,
                changed_by_user="test_helper",
                documents_added=10,
                chunks_added=100,
                embeddings_added=100,
                embedding_models_added=["test-model"],
                notes="Helper method test",
            )
            await conn.commit()

            changes = await get_schema_changes(conn)
            # Should have creation + our test change
            assert len(changes) >= 2
            last_change = changes[-1]
            assert last_change["changed_by_user"] == "test_helper"
            assert last_change["documents_added"] == 10
            assert last_change["chunks_added"] == 100
            assert last_change["embeddings_added"] == 100
            assert last_change["embedding_models_added"] == ["test-model"]
            assert last_change["notes"] == "Helper method test"

    @pytest.mark.asyncio
    async def test_schema_version_consistency(self):
        """Test that schema_version is consistent across metadata and changes."""
        if should_skip_postgres_tests():
            pytest.skip("PostgreSQL tests are disabled (TEST_SKIP_POSTGRES=1)")

        config = get_test_db_config()

        # Create database manager and initialize
        db_manager = DatabaseManager(
            host=config["host"],
            port=int(config["port"]),
            database=config["database"],
            user=config["user"],
            password=config["password"],
        )
        await db_manager.initialize_schema()

        async with await create_connection() as conn:
            metadata = await get_schema_metadata(conn)
            changes = await get_schema_changes(conn)

            assert metadata is not None
            assert len(changes) > 0

            # All records should have the same schema_version
            expected_version = metadata["schema_version"]
            assert expected_version == "1.0.0"
            for change in changes:
                assert change["schema_version"] == expected_version
