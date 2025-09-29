"""High-level integration test for the complete Docs2DB pipeline.

This test covers the entire workflow:
1. Chunking documents
2. Generating embeddings (granite model)
3. Loading into database
4. Testing idempotency (re-runs should not change files/records)
5. Testing incremental updates (document changes, new files)
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
    def test_content_dir(self, tmp_path: Path) -> Path:
        """Create a temporary content directory with three docling documents."""
        content_dir = tmp_path / "test_content"
        content_dir.mkdir()

        # Use a simpler approach - copy existing working documents and modify their content
        source_doc_path = (
            Path(__file__).parent.parent
            / "content"
            / "external"
            / "fedoraproject.org"
            / "how_to_debug_installation_problems.json"
        )

        # Read the source document structure
        with open(source_doc_path) as f:
            base_doc = json.load(f)

        # Document 1: Technical documentation (modify the base document)
        doc1 = base_doc.copy()
        doc1["name"] = "technical_guide"
        doc1["origin"]["filename"] = "technical_guide.json"
        doc1["origin"]["binary_hash"] = 1234567890123456789

        # Update the text content while keeping the structure
        if "texts" in doc1 and len(doc1["texts"]) > 0:
            doc1["texts"][0]["text"] = "System Setup Guide"
            if len(doc1["texts"]) > 1:
                doc1["texts"][1]["text"] = (
                    "This guide covers the essential steps for setting up the development environment. First, install the required dependencies using the package manager."
                )
            if len(doc1["texts"]) > 2:
                doc1["texts"][2]["text"] = (
                    "Prerequisites include Python 3.9 or higher, Node.js 16+, and Docker. Make sure all these are properly installed before proceeding with the setup process."
                )

        # Document 2: User manual
        doc2 = base_doc.copy()
        doc2["name"] = "user_manual"
        doc2["origin"]["filename"] = "user_manual.json"
        doc2["origin"]["binary_hash"] = 2345678901234567890

        if "texts" in doc2 and len(doc2["texts"]) > 0:
            doc2["texts"][0]["text"] = "User Manual"
            if len(doc2["texts"]) > 1:
                doc2["texts"][1]["text"] = (
                    "Welcome to the application! This manual will guide you through all available features and functionality."
                )
            if len(doc2["texts"]) > 2:
                doc2["texts"][2]["text"] = (
                    "Getting started is easy. After logging in, you'll see the main dashboard with navigation options for different modules."
                )

        # Document 3: API reference
        doc3 = base_doc.copy()
        doc3["name"] = "api_reference"
        doc3["origin"]["filename"] = "api_reference.json"
        doc3["origin"]["binary_hash"] = 3456789012345678901

        if "texts" in doc3 and len(doc3["texts"]) > 0:
            doc3["texts"][0]["text"] = "API Reference"
            if len(doc3["texts"]) > 1:
                doc3["texts"][1]["text"] = (
                    "This document provides comprehensive information about the REST API endpoints and their usage."
                )
            if len(doc3["texts"]) > 2:
                doc3["texts"][2]["text"] = (
                    "Authentication is required for all API calls. Use the Bearer token in the Authorization header for secure access."
                )

        # Write the documents to files
        (content_dir / "technical_guide.json").write_text(json.dumps(doc1, indent=2))
        (content_dir / "user_manual.json").write_text(json.dumps(doc2, indent=2))
        (content_dir / "api_reference.json").write_text(json.dumps(doc3, indent=2))

        return content_dir

    def get_file_set(self, directory: Path, pattern: str = "*") -> Set[str]:
        """Get a set of filenames matching the pattern in the directory."""
        return {f.name for f in directory.glob(pattern)}

    def get_file_mtimes(self, directory: Path, pattern: str = "*") -> Dict[str, float]:
        """Get modification times for files matching the pattern."""
        return {f.name: f.stat().st_mtime for f in directory.glob(pattern)}

    async def get_database_records(self, conn) -> Dict[str, int]:
        """Get counts of database records."""
        return {
            "documents": await count_records(conn, "documents"),
            "chunks": await count_records(conn, "chunks"),
            "embeddings": await count_records(conn, "embeddings"),
        }

    @pytest.mark.no_ci
    @pytest.mark.asyncio
    async def test_complete_pipeline_sql(self, test_content_dir: Path):
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

        # === PHASE 1: Initial Processing ===

        # Test initial chunking
        initial_files = self.get_file_set(test_content_dir)
        assert initial_files == {
            "technical_guide.json",
            "user_manual.json",
            "api_reference.json",
        }

        success = generate_chunks(str(test_content_dir), "**/*.json", force=False)
        assert success, "Initial chunking should succeed"

        # Verify chunk files were created correctly
        chunk_files = self.get_file_set(test_content_dir, "*.chunks.json")
        expected_chunk_files = {
            "technical_guide.chunks.json",
            "user_manual.chunks.json",
            "api_reference.chunks.json",
        }
        assert chunk_files == expected_chunk_files, (
            f"Expected {expected_chunk_files}, got {chunk_files}"
        )

        # Verify no unexpected files were created
        all_files_after_chunking = self.get_file_set(test_content_dir)
        expected_files_after_chunking = initial_files | expected_chunk_files
        assert all_files_after_chunking == expected_files_after_chunking, (
            "Unexpected files created during chunking"
        )

        # Verify chunks content is correct
        for chunk_file in chunk_files:
            chunk_path = test_content_dir / chunk_file
            with open(chunk_path) as f:
                chunk_data = json.load(f)

            assert "chunks" in chunk_data, f"Missing chunks in {chunk_file}"
            assert "metadata" in chunk_data, f"Missing metadata in {chunk_file}"
            assert len(chunk_data["chunks"]) > 0, f"No chunks found in {chunk_file}"

            # Verify each chunk has required fields
            for chunk in chunk_data["chunks"]:
                assert "text" in chunk, f"Missing text in chunk from {chunk_file}"
                assert "metadata" in chunk, (
                    f"Missing metadata in chunk from {chunk_file}"
                )

        # Test initial embedding
        success = generate_embeddings(
            str(test_content_dir),
            "granite-30m-english",
            "**/*.chunks.json",
            force=False,
        )
        assert success, "Initial embedding generation should succeed"

        # Verify embedding files were created correctly
        embed_files = self.get_file_set(test_content_dir, "*.gran.json")
        expected_embed_files = {
            "technical_guide.gran.json",
            "user_manual.gran.json",
            "api_reference.gran.json",
        }
        assert embed_files == expected_embed_files, (
            f"Expected {expected_embed_files}, got {embed_files}"
        )

        # Verify no unexpected files were created
        all_files_after_embedding = self.get_file_set(test_content_dir)
        expected_files_after_embedding = (
            expected_files_after_chunking | expected_embed_files
        )
        assert all_files_after_embedding == expected_files_after_embedding, (
            "Unexpected files created during embedding"
        )

        # Verify embeddings content is correct
        for embed_file in embed_files:
            embed_path = test_content_dir / embed_file
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
            content_dir=str(test_content_dir),
            model_name="granite-30m-english",
            pattern="**/*.json",
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
            assert initial_records["documents"] == 3, (
                f"Expected 3 documents, got {initial_records['documents']}"
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
            expected_doc_names = {
                "technical_guide.json",
                "user_manual.json",
                "api_reference.json",
            }
            assert doc_paths == expected_doc_names, (
                f"Expected {expected_doc_names}, got {doc_paths}"
            )

        # === PHASE 2: Idempotency Tests ===

        # Store file modification times before idempotency tests
        chunk_mtimes_before = self.get_file_mtimes(test_content_dir, "*.chunks.json")
        embed_mtimes_before = self.get_file_mtimes(test_content_dir, "*.gran.json")

        # Test chunking idempotency
        success = generate_chunks(str(test_content_dir), "**/*.json", force=False)
        assert success, "Chunking re-run should succeed"

        # Verify no chunk files changed
        chunk_mtimes_after = self.get_file_mtimes(test_content_dir, "*.chunks.json")
        assert chunk_mtimes_before == chunk_mtimes_after, (
            "Chunk files should not change on re-run"
        )

        # Test embedding idempotency
        success = generate_embeddings(
            str(test_content_dir),
            "granite-30m-english",
            "**/*.chunks.json",
            force=False,
        )
        assert success, "Embedding re-run should succeed"

        # Verify no embedding files changed
        embed_mtimes_after = self.get_file_mtimes(test_content_dir, "*.gran.json")
        assert embed_mtimes_before == embed_mtimes_after, (
            "Embedding files should not change on re-run"
        )

        # Test database load idempotency
        success = await load_documents(
            content_dir=str(test_content_dir),
            model_name="granite-30m-english",
            pattern="**/*.json",
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

        # === PHASE 3: Document Change Tests ===

        # Modify one document (use the same base structure)
        source_doc_path = (
            Path(__file__).parent.parent
            / "content"
            / "external"
            / "fedoraproject.org"
            / "how_to_debug_installation_problems.json"
        )
        with open(source_doc_path) as f:
            base_doc = json.load(f)

        modified_doc = base_doc.copy()
        modified_doc["name"] = "technical_guide"
        modified_doc["origin"]["filename"] = "technical_guide.json"
        modified_doc["origin"]["binary_hash"] = 1234567890123456789

        # Update with modified content
        if "texts" in modified_doc and len(modified_doc["texts"]) > 0:
            modified_doc["texts"][0]["text"] = "System Setup Guide - UPDATED VERSION"
            if len(modified_doc["texts"]) > 1:
                modified_doc["texts"][1]["text"] = (
                    "This guide covers the essential steps for setting up the development environment. First, install the required dependencies using the package manager. NEW REQUIREMENT: Also install Redis."
                )
            if len(modified_doc["texts"]) > 2:
                modified_doc["texts"][2]["text"] = (
                    "Prerequisites include Python 3.9 or higher, Node.js 16+, and Docker. Make sure all these are properly installed before proceeding with the setup process."
                )

        # Write the modified document
        (test_content_dir / "technical_guide.json").write_text(
            json.dumps(modified_doc, indent=2)
        )

        # Store mtimes before processing the change
        chunk_mtimes_before_change = self.get_file_mtimes(
            test_content_dir, "*.chunks.json"
        )
        embed_mtimes_before_change = self.get_file_mtimes(
            test_content_dir, "*.gran.json"
        )

        # Re-run chunking after document change
        success = generate_chunks(str(test_content_dir), "**/*.json", force=False)
        assert success, "Chunking after document change should succeed"

        # Verify only the changed document's chunk file was updated
        chunk_mtimes_after_change = self.get_file_mtimes(
            test_content_dir, "*.chunks.json"
        )

        # technical_guide.chunks.json should be updated
        assert (
            chunk_mtimes_after_change["technical_guide.chunks.json"]
            > chunk_mtimes_before_change["technical_guide.chunks.json"]
        ), "Modified document's chunk file should be updated"

        # Other chunk files should remain unchanged
        assert (
            chunk_mtimes_after_change["user_manual.chunks.json"]
            == chunk_mtimes_before_change["user_manual.chunks.json"]
        ), "Unmodified document's chunk file should not change"
        assert (
            chunk_mtimes_after_change["api_reference.chunks.json"]
            == chunk_mtimes_before_change["api_reference.chunks.json"]
        ), "Unmodified document's chunk file should not change"

        # Re-run embedding after document change
        success = generate_embeddings(
            str(test_content_dir),
            "granite-30m-english",
            "**/*.chunks.json",
            force=False,
        )
        assert success, "Embedding after document change should succeed"

        # Verify only the changed document's embedding file was updated
        embed_mtimes_after_change = self.get_file_mtimes(
            test_content_dir, "*.gran.json"
        )

        # technical_guide.gran.json should be updated
        assert (
            embed_mtimes_after_change["technical_guide.gran.json"]
            > embed_mtimes_before_change["technical_guide.gran.json"]
        ), "Modified document's embedding file should be updated"

        # Other embedding files should remain unchanged
        assert (
            embed_mtimes_after_change["user_manual.gran.json"]
            == embed_mtimes_before_change["user_manual.gran.json"]
        ), "Unmodified document's embedding file should not change"
        assert (
            embed_mtimes_after_change["api_reference.gran.json"]
            == embed_mtimes_before_change["api_reference.gran.json"]
        ), "Unmodified document's embedding file should not change"

        # Re-run database load after document change
        success = await load_documents(
            content_dir=str(test_content_dir),
            model_name="granite-30m-english",
            pattern="**/*.json",
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
            force=False,
        )
        assert success, "Database load after document change should succeed"

        # Verify database was updated correctly (same document count, but chunks/embeddings may differ)
        async with await psycopg.AsyncConnection.connect(conn_string) as conn:
            records_after_change = await self.get_database_records(conn)
            assert records_after_change["documents"] == 3, (
                "Document count should remain the same"
            )
            # Note: chunks/embeddings count may change due to different chunking of modified content

        # === PHASE 4: New File Addition Tests ===

        # Add a new document (use the same base structure)
        new_doc = base_doc.copy()
        new_doc["name"] = "troubleshooting"
        new_doc["origin"]["filename"] = "troubleshooting.json"
        new_doc["origin"]["binary_hash"] = 4567890123456789012

        if "texts" in new_doc and len(new_doc["texts"]) > 0:
            new_doc["texts"][0]["text"] = "Troubleshooting Guide"
            if len(new_doc["texts"]) > 1:
                new_doc["texts"][1]["text"] = (
                    "This guide helps you resolve common issues that may occur during system operation. Common problems include connection timeouts, authentication failures, and resource limitations."
                )
            if len(new_doc["texts"]) > 2:
                new_doc["texts"][2]["text"] = (
                    "For persistent issues, check the log files in /var/log/application/ and contact support with relevant error messages."
                )

        # Write the new document
        (test_content_dir / "troubleshooting.json").write_text(
            json.dumps(new_doc, indent=2)
        )

        # Store file counts and mtimes before processing new file
        files_before_new = self.get_file_set(test_content_dir)
        chunk_mtimes_before_new = self.get_file_mtimes(
            test_content_dir, "*.chunks.json"
        )
        embed_mtimes_before_new = self.get_file_mtimes(test_content_dir, "*.gran.json")

        # Re-run chunking after adding new file
        success = generate_chunks(str(test_content_dir), "**/*.json", force=False)
        assert success, "Chunking after adding new file should succeed"

        # Verify new chunk file was created and others unchanged
        files_after_new_chunk = self.get_file_set(test_content_dir)
        assert "troubleshooting.chunks.json" in files_after_new_chunk, (
            "New chunk file should be created"
        )

        chunk_mtimes_after_new = self.get_file_mtimes(test_content_dir, "*.chunks.json")

        # Existing chunk files should not change
        for filename in chunk_mtimes_before_new:
            assert (
                chunk_mtimes_after_new[filename] == chunk_mtimes_before_new[filename]
            ), (
                f"Existing chunk file {filename} should not change when new file is added"
            )

        # Re-run embedding after adding new file
        success = generate_embeddings(
            str(test_content_dir),
            "granite-30m-english",
            "**/*.chunks.json",
            force=False,
        )
        assert success, "Embedding after adding new file should succeed"

        # Verify new embedding file was created and others unchanged
        files_after_new_embed = self.get_file_set(test_content_dir)
        assert "troubleshooting.gran.json" in files_after_new_embed, (
            "New embedding file should be created"
        )

        embed_mtimes_after_new = self.get_file_mtimes(test_content_dir, "*.gran.json")

        # Existing embedding files should not change
        for filename in embed_mtimes_before_new:
            assert (
                embed_mtimes_after_new[filename] == embed_mtimes_before_new[filename]
            ), (
                f"Existing embedding file {filename} should not change when new file is added"
            )

        # Re-run database load after adding new file
        async with await psycopg.AsyncConnection.connect(conn_string) as conn:
            records_before_new_load = await self.get_database_records(conn)

        success = await load_documents(
            content_dir=str(test_content_dir),
            model_name="granite-30m-english",
            pattern="**/*.json",
            host=config["host"],
            port=int(config["port"]),
            db=config["database"],
            user=config["user"],
            password=config["password"],
            force=False,
        )
        assert success, "Database load after adding new file should succeed"

        # Verify new document was added to database
        async with await psycopg.AsyncConnection.connect(conn_string) as conn:
            records_after_new_load = await self.get_database_records(conn)
            assert (
                records_after_new_load["documents"]
                == records_before_new_load["documents"] + 1
            ), "One new document should be added to database"
            assert (
                records_after_new_load["chunks"] > records_before_new_load["chunks"]
            ), "New chunks should be added to database"
            assert (
                records_after_new_load["embeddings"]
                > records_before_new_load["embeddings"]
            ), "New embeddings should be added to database"

            # Verify the new document is in database
            final_doc_paths = await get_document_paths(conn)
            expected_final_docs = {
                "technical_guide.json",
                "user_manual.json",
                "api_reference.json",
                "troubleshooting.json",
            }
            assert final_doc_paths == expected_final_docs, (
                f"Expected {expected_final_docs}, got {final_doc_paths}"
            )

        # === FINAL VERIFICATION ===

        # Verify final file set is complete and correct
        final_files = self.get_file_set(test_content_dir)
        expected_final_files = {
            # Source documents
            "technical_guide.json",
            "user_manual.json",
            "api_reference.json",
            "troubleshooting.json",
            # Chunk files
            "technical_guide.chunks.json",
            "user_manual.chunks.json",
            "api_reference.chunks.json",
            "troubleshooting.chunks.json",
            # Embedding files
            "technical_guide.gran.json",
            "user_manual.gran.json",
            "api_reference.gran.json",
            "troubleshooting.gran.json",
        }
        assert final_files == expected_final_files, (
            f"Final file set mismatch. Expected {expected_final_files}, got {final_files}"
        )

        # Verify final database state
        async with await psycopg.AsyncConnection.connect(conn_string) as conn:
            final_records = await self.get_database_records(conn)
            assert final_records["documents"] == 4, (
                "Should have 4 documents in database"
            )
            assert final_records["embeddings"] > 0, "Should have embeddings in database"
            assert final_records["chunks"] == final_records["embeddings"], (
                "Chunks and embeddings should match"
            )
