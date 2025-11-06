"""High-level integration test for the complete Docs2DB pipeline.

This test covers the entire workflow:
1. Ingesting raw documents using docling
2. Chunking documents
3. Generating embeddings (granite model)
4. Loading into database
5. Testing idempotency (re-runs should not change files/records)
6. Testing incremental updates (document changes, new files)
"""

import json
from pathlib import Path
from typing import Dict, Set

import psycopg
import pytest
from psycopg.sql import SQL, Identifier

from docs2db.chunks import generate_chunks
from docs2db.database import DatabaseManager, check_database_status, load_documents
from docs2db.embed import generate_embeddings
from docs2db.exceptions import DatabaseError
from docs2db.ingest import ingest
from tests.test_config import get_test_db_config


async def count_records(conn, table_name: str) -> int:
    """Count records in a table."""
    async with conn.cursor() as cur:
        await cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        result = await cur.fetchone()
        return result[0] if result else 0


async def get_document_paths(conn) -> Set[str]:
    """Get set of document paths in database."""
    async with conn.cursor() as cur:
        await cur.execute("SELECT path FROM documents")
        results = await cur.fetchall()
        return {Path(row[0]).name for row in results}


class TestHighLevelIntegrationSQL:
    """High-level integration test for the complete Docs2DB pipeline."""

    @pytest.fixture
    def test_workspace_dir(self, tmp_path: Path) -> Path:
        """Create a temporary workspace directory with test input files."""
        # Just return the temp directory - the test will handle ingestion
        return tmp_path

    def get_file_set(self, directory: Path, pattern: str = "*") -> Set[str]:
        """Get a set of filenames matching the pattern in the directory."""
        if "**" in pattern:
            # For recursive patterns, return relative paths from the directory
            return {
                str(f.relative_to(directory))
                for f in directory.glob(pattern)
                if f.is_file()
            }
        else:
            # For non-recursive patterns, return just filenames
            return {f.name for f in directory.glob(pattern) if f.is_file()}

    def get_file_mtimes(self, directory: Path, pattern: str = "*") -> Dict[str, float]:
        """Get modification times for files matching the pattern."""
        return {
            f.name: f.stat().st_mtime for f in directory.glob(pattern) if f.is_file()
        }

    async def get_database_records(self, conn) -> Dict[str, int]:
        """Get counts of database records."""
        return {
            "documents": await count_records(conn, "documents"),
            "chunks": await count_records(conn, "chunks"),
            "embeddings": await count_records(conn, "embeddings"),
        }

    @pytest.mark.no_ci
    @pytest.mark.asyncio
    async def test_complete_pipeline_sql(self, test_workspace_dir: Path):
        """Test the complete pipeline with all stages and idempotency checks."""

        config = get_test_db_config()
        conn_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"

        db_manager = DatabaseManager(
            host=config["host"],
            port=int(config["port"]),
            database=config["database"],
            user=config["user"],
            password=config["password"],
        )

        try:
            await check_database_status(
                host=config["host"],
                port=int(config["port"]),
                db=config["database"],
                user=config["user"],
                password=config["password"],
            )
        except DatabaseError as e:
            assert "pgvector extension not installed" in str(e)

        await db_manager.initialize_schema()

        # === PHASE 0: Ingestion ===

        # Set up source files and workspace
        test_fixtures_dir = Path(__file__).parent / "fixtures" / "input"
        content_dir = test_workspace_dir / "docs2db_content"

        # Change to workspace directory so ingest creates docs2db_content/ in the right place
        original_cwd = Path.cwd()
        import os

        os.chdir(test_workspace_dir)

        try:
            # Test ingestion - convert raw files to Docling JSON
            success = ingest(str(test_fixtures_dir), dry_run=False, force=True)
            assert success, "Ingestion should succeed"

            # Verify content directory was created with ingested files
            assert content_dir.exists(), (
                "Content directory should be created by ingestion"
            )

            # Get the ingested source.json files (new subdirectory structure)
            initial_files = self.get_file_set(content_dir, "**/source.json")
            assert len(initial_files) > 0, "Should have ingested Docling JSON files"

        finally:
            # Restore original working directory
            os.chdir(original_cwd)

        # === PHASE 1: Chunking ===

        # Store the initial file set for later comparisons
        expected_json_files = initial_files.copy()

        success = generate_chunks(
            str(content_dir), "**/source.json", force=False, skip_context=True
        )
        assert success, "Initial chunking should succeed"

        # Verify chunk files were created correctly (new subdirectory structure)
        chunk_files = self.get_file_set(content_dir, "**/chunks.json")
        # For each source.json, we expect chunks.json in the same directory
        expected_chunk_files = {
            f.replace("source.json", "chunks.json") for f in expected_json_files
        }
        assert chunk_files == expected_chunk_files, (
            f"Expected {expected_chunk_files}, got {chunk_files}"
        )

        # Verify no unexpected files were created
        all_files_after_chunking = self.get_file_set(content_dir, "**/*")
        # Expected files = Docling JSONs + chunks + metadata files + README created during ingestion
        # structure: doc_dir/source.json, doc_dir/chunks.json, doc_dir/meta.json
        meta_files = {f.replace("source.json", "meta.json") for f in initial_files}
        expected_files_after_chunking = (
            initial_files | expected_chunk_files | meta_files | {"README.md"}
        )
        unexpected = all_files_after_chunking - expected_files_after_chunking
        missing = expected_files_after_chunking - all_files_after_chunking
        assert all_files_after_chunking == expected_files_after_chunking, (
            f"Unexpected files created during chunking.\n"
            f"Unexpected files: {unexpected}\n"
            f"Missing files: {missing}"
        )

        # Verify chunks content is correct
        for chunk_file in chunk_files:
            chunk_path = content_dir / chunk_file
            with open(chunk_path) as f:
                chunk_data = json.load(f)

            assert "chunks" in chunk_data, f"Missing chunks in {chunk_file}"
            assert "metadata" in chunk_data, f"Missing metadata in {chunk_file}"

            # Some files may have zero chunks (e.g., if content cannot be extracted)
            # This is acceptable, so we only verify chunk structure if chunks exist
            if len(chunk_data["chunks"]) > 0:
                # Verify each chunk has required fields
                for chunk in chunk_data["chunks"]:
                    assert "text" in chunk, f"Missing text in chunk from {chunk_file}"
                    assert "metadata" in chunk, (
                        f"Missing metadata in chunk from {chunk_file}"
                    )

        # Test initial embedding
        success = generate_embeddings(
            str(content_dir),
            "granite-30m-english",
            "**/chunks.json",
            force=False,
        )
        assert success, "Initial embedding generation should succeed"

        # Verify embedding files were created correctly (new subdirectory structure)
        embed_files = self.get_file_set(content_dir, "**/gran.json")

        # Only expect gran.json files for chunk files that actually have chunks
        expected_embed_files = set()
        for chunk_file in chunk_files:
            chunk_path = content_dir / chunk_file
            with open(chunk_path) as f:
                chunk_data = json.load(f)
            # Only expect gran.json if there are actual chunks to embed
            if len(chunk_data["chunks"]) > 0:
                expected_embed_files.add(chunk_file.replace("chunks.json", "gran.json"))

        assert embed_files == expected_embed_files, (
            f"Expected {expected_embed_files}, got {embed_files}"
        )

        # Verify no unexpected files were created
        all_files_after_embedding = self.get_file_set(content_dir, "**/*")
        expected_files_after_embedding = (
            expected_files_after_chunking | expected_embed_files
        )
        assert all_files_after_embedding == expected_files_after_embedding, (
            "Unexpected files created during embedding"
        )

        # Verify embeddings content is correct
        for embed_file in embed_files:
            embed_path = content_dir / embed_file
            with open(embed_path) as f:
                embed_data = json.load(f)

            assert "embeddings" in embed_data, f"Missing embeddings in {embed_file}"
            assert "metadata" in embed_data, f"Missing metadata in {embed_file}"
            assert len(embed_data["embeddings"]) > 0, (
                f"No embeddings found in {embed_file}"
            )

            # Verify embedding dimensions match granite model (384)
            for embedding in embed_data["embeddings"]:
                assert len(embedding) == 384, (
                    f"Wrong embedding dimensions in {embed_file}"
                )

        success = await load_documents(
            content_dir=str(content_dir),
            model_name="granite-30m-english",
            pattern="**/source.json",
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
            force=False,
        )
        assert success, "Initial database load should succeed"

        # Verify correct database entries were made
        async with await psycopg.AsyncConnection.connect(conn_string) as conn:
            initial_records = await self.get_database_records(conn)
            # Only documents with actual chunks get loaded into the database
            expected_doc_count = len(
                expected_embed_files
            )  # Same as embed files since only files with chunks get embedded and loaded
            assert initial_records["documents"] == expected_doc_count, (
                f"Expected {expected_doc_count} documents, got {initial_records['documents']}"
            )
            assert initial_records["chunks"] > 0, (
                f"Expected chunks > 0, got {initial_records['chunks']}"
            )
            assert initial_records["embeddings"] > 0, (
                f"Expected embeddings > 0, got {initial_records['embeddings']}"
            )
            assert initial_records["chunks"] == initial_records["embeddings"], (
                "Chunks and embeddings count should match"
            )

            # Verify correct documents are in database
            doc_paths = await get_document_paths(conn)
            # get_document_paths returns just filenames, so extract filenames from expected embed files
            # (only files with embeddings get loaded into the database)
            # gran.json and source.json are in same directory
            expected_doc_names = {
                Path(f.replace("gran.json", "source.json")).name
                for f in expected_embed_files
            }
            assert doc_paths == expected_doc_names, (
                f"Expected {expected_doc_names}, got {doc_paths}"
            )

        # === PHASE 2: Idempotency Tests ===

        # Store file modification times before idempotency tests
        chunk_mtimes_before = self.get_file_mtimes(content_dir, "chunks.json")
        embed_mtimes_before = self.get_file_mtimes(content_dir, "gran.json")

        # Test chunking idempotency
        success = generate_chunks(
            str(content_dir), "**/source.json", force=False, skip_context=True
        )
        assert success, "Chunking re-run should succeed"

        # Verify no chunk files changed
        chunk_mtimes_after = self.get_file_mtimes(content_dir, "chunks.json")
        assert chunk_mtimes_before == chunk_mtimes_after, (
            "Chunk files should not change on re-run"
        )

        # Test embedding idempotency
        success = generate_embeddings(
            str(content_dir),
            "granite-30m-english",
            "**/chunks.json",
            force=False,
        )
        assert success, "Embedding re-run should succeed"

        # Verify no embedding files changed
        embed_mtimes_after = self.get_file_mtimes(content_dir, "gran.json")
        assert embed_mtimes_before == embed_mtimes_after, (
            "Embedding files should not change on re-run"
        )

        # Test database load idempotency
        success = await load_documents(
            content_dir=str(content_dir),
            model_name="granite-30m-english",
            pattern="**/source.json",
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
            force=False,
        )
        assert success, "Database load re-run should succeed"

        # Verify no records were updated (same counts)
        async with await psycopg.AsyncConnection.connect(conn_string) as conn:
            records_after_rerun = await self.get_database_records(conn)
            assert initial_records == records_after_rerun, (
                "Database records should not change on re-run"
            )

        # === PHASE 3: Force Re-processing Test ===

        # Test force re-processing to ensure the pipeline can handle force flags
        success = generate_chunks(
            str(content_dir), "**/source.json", force=True, skip_context=True
        )
        assert success, "Force chunking should succeed"

        success = generate_embeddings(
            str(content_dir),
            "granite-30m-english",
            "**/*.chunks.json",
            force=True,
        )
        assert success, "Force embedding should succeed"

        success = await load_documents(
            content_dir=str(content_dir),
            model_name="granite-30m-english",
            pattern="**/source.json",
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
            force=True,
        )
        assert success, "Force database load should succeed"

        # === FINAL VERIFICATION ===

        # Verify final file set is complete and correct
        final_files = self.get_file_set(content_dir, "**/*")
        expected_final_files = (
            expected_json_files  # Source documents
            | expected_chunk_files  # Chunk files
            | expected_embed_files  # Embedding files
            | meta_files  # Metadata files created during ingestion
            | {"README.md"}  # README created during ingestion
        )
        assert final_files == expected_final_files, (
            f"Final file set mismatch. Expected {expected_final_files}, got {final_files}"
        )

        # Verify final database state
        async with await psycopg.AsyncConnection.connect(conn_string) as conn:
            final_records = await self.get_database_records(conn)
            assert final_records["documents"] == expected_doc_count, (
                f"Should have {expected_doc_count} documents in database"
            )
            assert final_records["embeddings"] > 0, "Should have embeddings in database"
            assert final_records["chunks"] == final_records["embeddings"], (
                "Chunks and embeddings should match"
            )

            # Verify all documents with embeddings are in database
            final_doc_paths = await get_document_paths(conn)
            # Only documents with embeddings get loaded into the database
            # So we should expect the same documents that have gran.json files
            expected_final_doc_names = {
                Path(f.replace("gran.json", "source.json")).name
                for f in expected_embed_files
            }
            assert final_doc_paths == expected_final_doc_names, (
                f"Expected {expected_final_doc_names}, got {final_doc_paths}"
            )
