"""Database operations for loading embeddings and chunks into PostgreSQL with pgvector."""

import asyncio
import json
import logging
import os
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import psutil
import psycopg
import structlog
import yaml
from psycopg.sql import SQL, Identifier

from docs2db.embeddings import EMBEDDING_CONFIGS, create_embedding_filename
from docs2db.exceptions import ConfigurationError, ContentError, DatabaseError
from docs2db.multiproc import BatchProcessor, setup_worker_logging

logger = structlog.get_logger()


def get_db_config() -> Dict[str, str]:
    """Parse postgres-compose.yml to get database connection parameters."""
    compose_file = Path(__file__).parent.parent.parent / "postgres-compose.yml"

    with open(compose_file, "r") as f:
        compose_data = yaml.safe_load(f)

    config = {"host": "localhost"}

    db_service = compose_data["services"]["db"]
    env = db_service["environment"]
    config["database"] = env["POSTGRES_DB"]
    config["user"] = env["POSTGRES_USER"]
    config["password"] = env["POSTGRES_PASSWORD"]

    # Extract port from ports mapping if available
    ports = db_service.get("ports", [])
    for port_mapping in ports:
        if isinstance(port_mapping, str) and ":5432" in port_mapping:
            host_port = port_mapping.split(":")[0]
            config["port"] = host_port
            break

    return config


class DatabaseManager:
    """Manages PostgreSQL database for pgvector storage."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

    async def get_direct_connection(self):
        """Get a direct database connection."""
        connection_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        return await psycopg.AsyncConnection.connect(connection_string)

    async def initialize_schema(self) -> None:
        """Initialize database schema with tables for documents, chunks, and embeddings."""
        # Check if schema already exists and create it if needed
        async with await self.get_direct_connection() as conn:
            tables_result = await conn.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('documents', 'chunks', 'embeddings')
            """)
            existing_tables = [row[0] for row in await tables_result.fetchall()]
            schema_exists = len(existing_tables) == 3

            schema_sql = """
        -- Enable pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;

        -- Documents table: stores metadata about source documents
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            content_type TEXT,
            file_size BIGINT,
            last_modified TIMESTAMP WITH TIME ZONE,
            chunks_file_path TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Chunks table: stores text chunks from documents
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(document_id, chunk_index)
        );

        -- Embeddings table: stores vector embeddings for chunks
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
            model_name TEXT NOT NULL,
            embedding VECTOR, -- Dynamic dimension based on model
            dimensions INTEGER NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(chunk_id, model_name)
        );

        -- Indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
        CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name);

        -- Function to update the updated_at timestamp
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';

        -- Trigger to automatically update updated_at (idempotent)
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_trigger
                WHERE tgname = 'update_documents_updated_at'
                AND tgrelid = 'documents'::regclass
            ) THEN
                CREATE TRIGGER update_documents_updated_at
                    BEFORE UPDATE ON documents
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            END IF;
        END
        $$;
        """

            await conn.execute(schema_sql)
            await conn.commit()

            if not schema_exists:
                logger.info("Database schema initialized successfully")

    async def load_document_batch(
        self,
        files_data: List[
            Tuple[Path, Path, str, Dict[str, Any], Path]
        ],  # (source_file, chunks_file, model, embedding_data, embedding_file)
        content_dir: Path,
        force: bool = False,
    ) -> Tuple[int, int]:
        """Load a batch of documents, chunks, and embeddings using bulk operations."""
        processed = 0
        errors = 0

        # Prepare bulk data
        documents_data = []
        chunks_data = []
        embeddings_data = []

        # First pass: prepare all data and validate
        for (
            source_file,
            chunks_file,
            model_name,
            embedding_data,
            embedding_file,
        ) in files_data:
            try:
                # Load chunks data
                with open(chunks_file, "r", encoding="utf-8") as f:
                    chunks_json = json.load(f)

                chunks = chunks_json.get("chunks", [])
                embedding_vectors = embedding_data.get("embeddings", [])

                if len(chunks) != len(embedding_vectors):
                    logger.error(
                        f"Chunks count ({len(chunks)}) != embeddings count ({len(embedding_vectors)}) for {source_file.name}"
                    )
                    errors += 1
                    continue

                stats = source_file.stat()

                doc_data = (
                    str(source_file.relative_to(content_dir)),
                    source_file.name,
                    self._get_content_type(source_file),
                    stats.st_size,
                    self._convert_timestamp(stats.st_mtime),
                    str(chunks_file),
                )
                documents_data.append((
                    source_file,
                    doc_data,
                    chunks,
                    embedding_vectors,
                    model_name,
                    embedding_file,
                ))

            except Exception as e:
                logger.error(f"Failed to prepare {source_file.name}: {e}")
                errors += 1

        if not documents_data:
            return 0, errors

        # Bulk database operations
        async with await self.get_direct_connection() as conn:
            try:
                # Begin transaction for entire batch
                await conn.execute("BEGIN")

                # Bulk insert/update documents
                doc_path_to_id = {}
                for (
                    source_file,
                    doc_data,
                    chunks,
                    embedding_vectors,
                    model_name,
                    embedding_file,
                ) in documents_data:
                    try:
                        # Check if we should skip (not force and current embeddings exist)
                        if not force:
                            # Get existing embeddings creation time
                            existing_result = await conn.execute(
                                """
                                SELECT MAX(e.created_at) as latest_embedding_time
                                FROM documents d
                                JOIN chunks c ON c.document_id = d.id
                                JOIN embeddings e ON e.chunk_id = c.id
                                WHERE d.path = %s AND e.model_name = %s
                                """,
                                (str(source_file), model_name),
                            )
                            existing_row = await existing_result.fetchone()

                            if existing_row and existing_row[0]:  # embeddings exist
                                latest_embedding_time = existing_row[0]

                                # Get embedding file modification time
                                if embedding_file and embedding_file.exists():
                                    embedding_file_mtime = datetime.fromtimestamp(
                                        embedding_file.stat().st_mtime, tz=timezone.utc
                                    )

                                    # Skip only if database embeddings are newer than the embedding file
                                    if latest_embedding_time >= embedding_file_mtime:
                                        continue

                        # Insert/update document
                        doc_result = await conn.execute(
                            """
                            INSERT INTO documents (path, filename, content_type, file_size, last_modified, chunks_file_path)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (path) DO UPDATE SET
                                filename = EXCLUDED.filename,
                                file_size = EXCLUDED.file_size,
                                last_modified = EXCLUDED.last_modified,
                                chunks_file_path = EXCLUDED.chunks_file_path,
                                updated_at = NOW()
                            RETURNING id
                            """,
                            doc_data,
                        )
                        doc_row = await doc_result.fetchone()
                        if doc_row is None:
                            raise DatabaseError(
                                f"Failed to insert/update document: {source_file}"
                            )

                        document_id = doc_row[0]
                        doc_path_to_id[str(source_file)] = document_id

                        # Delete existing chunks and embeddings if force
                        if force:
                            await conn.execute(
                                "DELETE FROM chunks WHERE document_id = %s",
                                (document_id,),
                            )

                        # Prepare chunks data for this document
                        for chunk_idx, (chunk, embedding_vector) in enumerate(
                            zip(chunks, embedding_vectors, strict=False)
                        ):
                            chunk_data = (
                                document_id,
                                chunk_idx,
                                chunk.get("text", ""),
                                json.dumps(chunk.get("metadata", {})),
                            )
                            chunks_data.append((
                                source_file,
                                chunk_data,
                                embedding_vector,
                                model_name,
                            ))

                        processed += 1

                    except Exception as e:
                        logger.error(
                            f"Failed to process document {source_file.name}: {e}"
                        )
                        errors += 1

                # Bulk insert chunks and collect chunk IDs
                chunk_id_map = {}  # (source_file, chunk_idx) -> chunk_id

                for (
                    source_file,
                    chunk_data,
                    embedding_vector,
                    model_name,
                ) in chunks_data:
                    try:
                        chunk_result = await conn.execute(
                            """
                            INSERT INTO chunks (document_id, chunk_index, text, metadata)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                                text = EXCLUDED.text,
                                metadata = EXCLUDED.metadata
                            RETURNING id
                            """,
                            chunk_data,
                        )
                        chunk_row = await chunk_result.fetchone()
                        if chunk_row is None:
                            raise DatabaseError(
                                f"Failed to insert chunk for {source_file}"
                            )

                        chunk_id = chunk_row[0]
                        chunk_id_map[(source_file, chunk_data[1])] = (
                            chunk_id  # chunk_data[1] is chunk_index
                        )

                        # Prepare embedding data
                        dimensions = len(embedding_vector)
                        embedding_data_tuple = (
                            chunk_id,
                            model_name,
                            embedding_vector,
                            dimensions,
                        )
                        embeddings_data.append(embedding_data_tuple)

                    except Exception as e:
                        logger.error(f"Failed to insert chunk for {source_file}: {e}")
                        errors += 1

                # Bulk insert embeddings
                for embedding_tuple in embeddings_data:
                    try:
                        await conn.execute(
                            """
                            INSERT INTO embeddings (chunk_id, model_name, embedding, dimensions)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (chunk_id, model_name) DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                dimensions = EXCLUDED.dimensions,
                                created_at = NOW()
                            """,
                            embedding_tuple,
                        )
                    except Exception as e:
                        logger.error(f"Failed to insert embedding: {e}")
                        errors += 1

                # Commit the entire batch
                await conn.execute("COMMIT")

            except Exception as e:
                await conn.execute("ROLLBACK")
                logger.error(f"Batch transaction failed: {e}")
                errors += len(documents_data)
                processed = 0

        return processed, errors

    def _get_content_type(self, file_path: Path) -> str:
        """Determine content type from file extension."""
        suffix = file_path.suffix.lower()
        content_types = {
            ".json": "application/json",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".html": "text/html",
            ".pdf": "application/pdf",
        }
        return content_types.get(suffix, "application/octet-stream")

    def _convert_timestamp(self, unix_timestamp: float):
        """Convert Unix timestamp to datetime object for PostgreSQL."""
        return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        async with await self.get_direct_connection() as conn:
            # Document stats
            doc_result = await conn.execute("SELECT COUNT(*) FROM documents")
            doc_row = await doc_result.fetchone()
            doc_count = doc_row[0] if doc_row else 0

            # Chunk stats
            chunk_result = await conn.execute("SELECT COUNT(*) FROM chunks")
            chunk_row = await chunk_result.fetchone()
            chunk_count = chunk_row[0] if chunk_row else 0

            # Embedding stats by model
            embedding_stats = await conn.execute(
                """
                SELECT model_name, COUNT(*) as count, AVG(dimensions) as avg_dimensions
                FROM embeddings
                GROUP BY model_name
                ORDER BY model_name
                """
            )
            embedding_models = {}
            async for row in embedding_stats:
                model_name, count, avg_dims = row
                embedding_models[model_name] = {
                    "count": count,
                    "dimensions": int(avg_dims) if avg_dims else 0,
                }

            return {
                "documents": doc_count,
                "chunks": chunk_count,
                "embedding_models": embedding_models,
            }

    async def generate_manifest(self, output_file: str = "manifest.txt") -> bool:
        """Generate a manifest file with all unique source files in the database.

        Args:
            output_file: Path to the output manifest file

        Returns:
            bool: True if successful, False otherwise
        """
        async with await self.get_direct_connection() as conn:
            # Query for distinct document paths from documents table
            result = await conn.execute(
                """
                SELECT DISTINCT path
                FROM documents
                ORDER BY path
                """
            )

            # Write to manifest file iteratively
            manifest_path = Path(output_file)
            file_count = 0

            with open(manifest_path, "w") as f:
                async for row in result:
                    document_path = row[0]
                    f.write(f"{document_path}\n")
                    file_count += 1

            logger.info(
                f"Generated manifest with {file_count} unique document files",
                output_file=output_file,
            )
            return True

    async def search_similar(
        self,
        query_embedding: List[float],
        model_name: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        async with await self.get_direct_connection() as conn:
            results = await conn.execute(
                """
                SELECT
                    c.text,
                    c.metadata,
                    d.path,
                    d.filename,
                    e.embedding <=> %s::vector as distance,
                    1 - (e.embedding <=> %s::vector) as similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                JOIN embeddings e ON c.id = e.chunk_id
                WHERE e.model_name = %s
                    AND 1 - (e.embedding <=> %s::vector) >= %s
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    query_embedding,
                    query_embedding,
                    model_name,
                    query_embedding,
                    similarity_threshold,
                    query_embedding,
                    limit,
                ),
            )

            similar_chunks = []
            async for row in results:
                text, metadata_json, doc_path, filename, distance, similarity = row

                # Handle metadata - it might be a dict already or a JSON string
                if metadata_json:
                    if isinstance(metadata_json, str):
                        metadata = json.loads(metadata_json)
                    else:
                        metadata = metadata_json
                else:
                    metadata = {}

                similar_chunks.append({
                    "text": text,
                    "metadata": metadata,
                    "document_path": doc_path,
                    "document_filename": filename,
                    "distance": float(distance),
                    "similarity": float(similarity),
                })

            return similar_chunks


async def check_database_status(
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> None:
    """Check database connectivity and display statistics."""
    db_defaults = get_db_config()
    host = host if host is not None else db_defaults["host"]
    port = port if port is not None else int(db_defaults["port"])
    db = db if db is not None else db_defaults["database"]
    user = user if user is not None else db_defaults["user"]
    password = password if password is not None else db_defaults["password"]

    logger.info(
        "\nCheck database status:\n"
        f"  Host    : {host}\n"
        f"  Port    : {port}\n"
        f"  Database: {db}\n"
        f"  user    : {user}"
    )

    # Suppress psycopg connection warnings for cleaner error messages
    logging.getLogger("psycopg.pool").setLevel(logging.ERROR)

    db_manager = DatabaseManager(
        host=host,
        port=port,
        database=db,
        user=user,
        password=password,
    )

    # Section 1: Test basic PostgreSQL server connectivity
    try:
        # First try a direct connection to catch auth errors immediately
        basic_connection_string = (
            f"postgresql://{user}:{password}@{host}:{port}/postgres"
        )

        async with await psycopg.AsyncConnection.connect(
            basic_connection_string, connect_timeout=5
        ) as conn:
            # Test basic connectivity
            result = await conn.execute("SELECT version(), now()")
            row = await result.fetchone()
            if row:
                _pg_version, _current_time = row
                logger.info("Database connection successful")

    except Exception as conn_error:
        # Handle server connectivity errors
        error_msg = str(conn_error).lower()
        if (
            "connection refused" in error_msg
            or "could not receive data" in error_msg
            or "couldn't get a connection" in error_msg
        ):
            logger.error("Database is not running. Start database with 'make db-up'")
        elif (
            "authentication failed" in error_msg
            or "no password supplied" in error_msg
            or "password authentication failed" in error_msg
            or "role" in error_msg
            and "does not exist" in error_msg
        ):
            logger.error("Database authentication failed. Check database credentials")
        else:
            logger.error("Database connection failed. Ensure PostgreSQL is running")

        raise DatabaseError(f"Database connection failed: {conn_error}") from conn_error

    # Section 2: Test target database connectivity
    try:
        # Now connect to our target database and test it
        async with await db_manager.get_direct_connection() as conn:
            # Test that we can actually query the target database
            await conn.execute("SELECT 1")
    except Exception as conn_error:
        # If we get here, PostgreSQL is running but our target database doesn't exist
        logger.error("Database does not exist. Create database or check name")
        raise DatabaseError("Database does not exist") from conn_error

    # If we get here, connection was successful, continue with checks

    # Check for pgvector extension
    async with await db_manager.get_direct_connection() as conn:
        ext_result = await conn.execute(
            "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
        )
        ext_row = await ext_result.fetchone()
        if ext_row:
            _ext_name, ext_version = ext_row
            logger.info(f"pgvector extension found: version={ext_version}")
        else:
            logger.error(
                "pgvector extension not installed. "
                "Run 'uv run docs2db load' to initialize"
            )
            raise DatabaseError("pgvector extension not installed")

    # Check if tables exist
    async with await db_manager.get_direct_connection() as conn:
        tables_result = await conn.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('documents', 'chunks', 'embeddings')
                ORDER BY table_name
            """)
        tables = []
        async for row in tables_result:
            tables.append(row[0])

        if len(tables) == 3:
            logger.info("All required tables exist")
        elif len(tables) > 0:
            logger.error(
                "Partial schema found. Run 'uv run docs2db load' to initialize"
            )
            raise DatabaseError("Partial schema found")
        else:
            logger.error(
                "No docs2db tables found. Run 'uv run docs2db load' to initialize"
            )
            raise DatabaseError("No docs2db tables found")

    # Get database statistics
    stats = await db_manager.get_stats()

    total_embeddings = sum(
        model_info["count"] for model_info in stats["embedding_models"].values()
    )

    logger.info(
        "\nDatabase statistics summary:\n"
        f"  documents : {stats['documents']}\n"
        f"  chunks    : {stats['chunks']}\n"
        f"  embeddings: {total_embeddings}\n"
    )

    # Log embedding models breakdown
    if stats["embedding_models"]:
        for model_name, model_info in stats["embedding_models"].items():
            logger.info(
                "\nEmbedding model details:\n"
                f"model     : {model_name}\n"
                f"embeddings: {model_info['count']}\n"
                f"dimensions: {model_info['dimensions']}"
            )

    if stats["documents"] > 0:
        # Get recent activity
        async with await db_manager.get_direct_connection() as conn:
            recent_result = await conn.execute("""
                SELECT
                    filename,
                    created_at,
                    updated_at
                FROM documents
                ORDER BY updated_at DESC
                LIMIT 5
            """)

            file_str = ""
            async for row in recent_result:
                filename, created_at, updated_at = row
                file_str += f"  {filename}\n    created: {created_at.strftime('%Y-%m-%d %H:%M')}\n    updated: {updated_at.strftime('%Y-%m-%d %H:%M') if updated_at else 'Never'}\n"
            logger.info(f"\nRecent document activity (last 5)\n{file_str}")

        # Database size information
        async with await db_manager.get_direct_connection() as conn:
            size_result = await conn.execute(
                "SELECT pg_size_pretty(pg_database_size(%s)) as db_size", (db,)
            )
            size_row = await size_result.fetchone()
            if size_row:
                db_size = size_row[0]
                logger.info(f"Database size: {db_size}")

    logger.info("Database status check completed successfully")


async def load_files(
    content_dir: Path, model_name: str, pattern: str, force: bool
) -> tuple[int, Iterator[tuple[Path, Path]]]:
    """Find source files and their corresponding embedding files for loading."""
    # Find all source files (excluding processed files)
    embedding_suffixes = [
        f".{config['keyword']}.json" for config in EMBEDDING_CONFIGS.values()
    ]

    def source_files_iter():
        """Iterator over source files, excluding processed files."""
        for f in content_dir.glob(pattern):
            if not f.name.endswith(".chunks.json") and not any(
                f.name.endswith(suffix) for suffix in embedding_suffixes
            ):
                yield f

    def valid_pairs_iter():
        """Iterator over valid (source_file, embedding_file) pairs."""
        for source_file in source_files_iter():
            chunks_file = source_file.with_suffix(".chunks.json")
            if not chunks_file.exists():
                continue

            embedding_file = create_embedding_filename(chunks_file, model_name)
            if not embedding_file.exists():
                continue

            yield source_file, embedding_file

    # Count valid pairs without consuming the iterator
    count = sum(1 for _ in valid_pairs_iter())
    return count, valid_pairs_iter()


async def _ensure_database_exists(
    host: str, port: int, db: str, user: str, password: str
) -> None:
    """Ensure the target database exists, create it if it doesn't."""

    # Connect to the default postgres database to check/create our target database
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/postgres"

    try:
        async with await psycopg.AsyncConnection.connect(
            connection_str,
            connect_timeout=5,
            autocommit=True,  # Needed for CREATE DATABASE
        ) as conn:
            # Check if our target database exists
            result = await conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (db,)
            )
            db_exists = await result.fetchone()

            if not db_exists:
                logger.info(f"Creating database '{db}'...")
                # Create the database (note: can't use parameters for database name in CREATE DATABASE)
                create_db_query = SQL("CREATE DATABASE {}").format(Identifier(db))
                await conn.execute(create_db_query)
                logger.info(f"Database '{db}' created successfully")

    except Exception as e:
        logger.error(f"Failed to ensure database exists: {e}")
        raise DatabaseError(f"Could not create database '{db}': {e}") from e


def load_batch_worker(
    file_batch: List[str],
    model_name: str,
    content_dir: str,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
    force: bool,
) -> Dict[str, Any]:
    """Worker function for multiprocessing database loading.

    Args:
        file_batch: List of source file paths to process
        model_name: Embedding model name
        db_host: Database host
        db_port: Database port
        db_name: Database name
        db_user: Database user
        db_password: Database password
        force: Force reload existing documents

    Returns:
        Dict with processing results and worker logs
    """

    # Set up worker logging to capture logs for replay in main process
    log_collector = setup_worker_logging(__name__)

    try:
        # Convert string paths back to Path objects
        file_paths = [Path(f) for f in file_batch]
        content_dir_path = Path(content_dir)

        # Run the async loading function
        processed, errors = asyncio.run(
            _load_batch_async(
                file_paths,
                model_name,
                content_dir_path,
                db_host,
                db_port,
                db_name,
                db_user,
                db_password,
                force,
            )
        )

        # Get memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        last_file = file_paths[-1].name if file_paths else "unknown"

        return {
            "processed": processed,
            "errors": errors,
            "error_data": [],  # Individual errors are logged, not returned
            "worker_logs": log_collector.logs,
            "memory": memory_mb,
            "last_file": last_file,
        }

    except Exception as e:
        logger.error(f"Worker failed: {e}")
        return {
            "processed": 0,
            "errors": len(file_batch),
            "error_data": [{"file": f, "error": str(e)} for f in file_batch],
            "worker_logs": log_collector.logs,
            "memory": 0,
            "last_file": file_batch[-1] if file_batch else "unknown",
        }


async def _load_batch_async(
    file_paths: List[Path],
    model_name: str,
    content_dir: Path,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
    force: bool,
) -> Tuple[int, int]:
    """Async helper for loading a batch of files in a worker process."""

    # Create database manager
    db_manager = DatabaseManager(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password,
    )

    # Prepare files data
    files_data = []
    for source_file in file_paths:
        try:
            # Check for chunks and embedding files
            chunks_file = source_file.with_suffix(".chunks.json")
            if not chunks_file.exists():
                continue

            embedding_file = create_embedding_filename(chunks_file, model_name)
            if not embedding_file.exists():
                continue

            # Load embedding data
            with open(embedding_file, "r", encoding="utf-8") as f:
                embedding_data = json.load(f)

            files_data.append((
                source_file,
                chunks_file,
                model_name,
                embedding_data,
                embedding_file,
            ))

        except Exception as e:
            logger.error(f"Failed to prepare {source_file.name}: {e}")

    if not files_data:
        return 0, 0

    # Load the batch into database
    processed, errors = await db_manager.load_document_batch(
        files_data, content_dir, force
    )

    return processed, errors


async def load_documents(
    content_dir: str,
    model_name: str,
    pattern: str,
    host: Optional[str],
    port: Optional[int],
    db: Optional[str],
    user: Optional[str],
    password: Optional[str],
    force: bool = False,
    batch_size: int = 100,
) -> bool:
    """Load documents and embeddings in the PostgreSQL database.

    Args:
        content_dir: Directory containing content files
        model_name: Embedding model name
        pattern: File pattern to match
        host: Database host (auto-detected from compose file if None)
        port: Database port (auto-detected from compose file if None)
        db: Database name (auto-detected from compose file if None)
        user: Database user (auto-detected from compose file if None)
        password: Database password (auto-detected from compose file if None)
        force: Force reload existing documents
        batch_size: Files per batch for each worker

    Returns:
        True if successful, False if errors occurred

    Raises:
        ConfigurationError: If model is unknown or configuration is invalid
        ContentError: If content directory does not exist
        DatabaseError: If database operations fail
    """
    start = time.time()

    config = get_db_config()
    host = host if host is not None else config["host"]
    port = port if port is not None else int(config["port"])
    db = db if db is not None else config["database"]
    user = user if user is not None else config["user"]
    password = password if password is not None else config["password"]

    if model_name not in EMBEDDING_CONFIGS:
        available = ", ".join(EMBEDDING_CONFIGS.keys())
        logger.error(f"Unknown model '{model_name}'. Available: {available}")
        raise ConfigurationError(
            f"Unknown model '{model_name}'. Available: {available}"
        )

    logger.info(
        f"\nDatabase load:\n"
        f"  model   : {model_name}\n"
        f"  content : {content_dir}\n"
        f"  pattern : {pattern}\n"
        f"  database: {user}@{host}:{port}/{db}\n"
    )

    # Ensure database exists and schema is initialized
    await _ensure_database_exists(host, port, db, user, password)

    # Create a temporary database manager just for schema initialization
    db_manager = DatabaseManager(
        host=host,
        port=port,
        database=db,
        user=user,
        password=password,
    )

    await db_manager.initialize_schema()

    content_path = Path(content_dir)
    if not content_path.exists():
        logger.error(f"Content directory does not exist: {content_dir}")
        raise ContentError(f"Content directory does not exist: {content_dir}")

    count, file_pairs_iter = await load_files(content_path, model_name, pattern, force)

    if not count:
        logger.info("No files to load")
        return True

    logger.info(f"Found {count} embedding files for model: {model_name}")

    processor = BatchProcessor(
        worker_function=load_batch_worker,
        worker_args=(
            model_name,
            content_dir,
            host,
            port,
            db,
            user,
            password,
            force,
        ),
        progress_message=f"Loading files...",
        batch_size=batch_size,
        mem_threshold_mb=2000,
    )

    # Extract just the source files from the iterator for the batch processor
    source_files_iter = (source_file for source_file, _ in file_pairs_iter)
    loaded, errors = processor.process_files(source_files_iter, count)
    end = time.time()

    if errors > 0:
        logger.error(f"Load completed with {errors} errors")
        logger.info(f"{loaded} files loaded in {end - start:.2f} seconds")
        return False

    logger.info(f"{loaded} files loaded in {end - start:.2f} seconds")
    return True


def dump_database(
    output_file: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """Create a PostgreSQL dump file of the database.

    Args:
        output_file: Output file path for the database dump
        host: Database host (auto-detected from compose file if None)
        port: Database port (auto-detected from compose file if None)
        db: Database name (auto-detected from compose file if None)
        user: Database user (auto-detected from compose file if None)
        password: Database password (auto-detected from compose file if None)
        verbose: Show pg_dump output

    Returns:
        True if successful, False if errors occurred

    Raises:
        ConfigurationError: If pg_dump is not found or configuration is invalid
        DatabaseError: If dump operation fails
    """
    config = get_db_config()
    host = host if host is not None else config["host"]
    port = port if port is not None else int(config["port"])
    db = db if db is not None else config["database"]
    user = user if user is not None else config["user"]
    password = password if password is not None else config["password"]

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating database dump: {user}@{host}:{port}/{db}")
    logger.info(f"Output file: {output_file}")

    # Build pg_dump command
    cmd = [
        "pg_dump",
        f"--host={host}",
        f"--port={port}",
        f"--username={user}",
        f"--dbname={db}",
        "--no-password",  # Use PGPASSWORD env var instead
        "--file",
        str(output_path),
    ]

    if verbose:
        cmd.append("--verbose")

    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password

    try:
        logger.info("Creating database dump...")

        # Run pg_dump
        subprocess.run(
            cmd,
            env=env,
            capture_output=not verbose,
            text=True,
            check=True,
        )

        # Check if file was created and get size
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Database dump created: {output_file} ({size_mb:.1f} MB)")
            return True
        else:
            logger.error(f"Dump file was not created: {output_file}")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"pg_dump failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        raise DatabaseError(
            f"Database dump failed with exit code {e.returncode}"
        ) from e
    except FileNotFoundError as e:
        raise ConfigurationError(
            "pg_dump command not found. Please install PostgreSQL client tools."
        ) from e


def restore_database(
    input_file: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """Restore a PostgreSQL database from a dump file.

    Args:
        input_file: Input file path for the database dump
        host: Database host (auto-detected from compose file if None)
        port: Database port (auto-detected from compose file if None)
        db: Database name (auto-detected from compose file if None)
        user: Database user (auto-detected from compose file if None)
        password: Database password (auto-detected from compose file if None)
        verbose: Show psql output

    Returns:
        True if successful, False if errors occurred

    Raises:
        ConfigurationError: If psql is not found or configuration is invalid
        DatabaseError: If restore operation fails
    """
    config = get_db_config()
    host = host if host is not None else config["host"]
    port = port if port is not None else int(config["port"])
    db = db if db is not None else config["database"]
    user = user if user is not None else config["user"]
    password = password if password is not None else config["password"]

    input_path = Path(input_file)
    if not input_path.exists():
        raise DatabaseError(f"Dump file not found: {input_file}")

    logger.info(f"Restoring database dump: {user}@{host}:{port}/{db}")
    logger.info(f"Input file: {input_file}")

    # Build psql command
    cmd = [
        "psql",
        f"--host={host}",
        f"--port={port}",
        f"--username={user}",
        f"--dbname={db}",
        "--no-password",  # Use PGPASSWORD env var instead
        "--file",
        str(input_path),
    ]

    if not verbose:
        cmd.append("--quiet")

    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password

    try:
        logger.info("Restoring database from dump...")

        # Run psql
        subprocess.run(
            cmd,
            env=env,
            capture_output=not verbose,
            text=True,
            check=True,
        )

        logger.info(f"Database restored successfully from: {input_file}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"psql failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        raise DatabaseError(
            f"Database restore failed with exit code {e.returncode}"
        ) from e
    except FileNotFoundError as e:
        raise ConfigurationError(
            "psql command not found. Please install PostgreSQL client tools."
        ) from e


async def generate_manifest(
    output_file: str = "manifest.txt",
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> bool:
    """Generate a manifest file with all unique source files in the database.

    Args:
        output_file: Path to the output manifest file
        host: Database host (auto-detected if not provided)
        port: Database port (auto-detected if not provided)
        db: Database name (auto-detected if not provided)
        user: Database user (auto-detected if not provided)
        password: Database password (auto-detected if not provided)

    Returns:
        bool: True if successful, False otherwise
    """
    config = get_db_config()
    host = host if host is not None else config["host"]
    port = port if port is not None else int(config["port"])
    db = db if db is not None else config["database"]
    user = user if user is not None else config["user"]
    password = password if password is not None else config["password"]

    db_manager = DatabaseManager(
        host=host,
        port=port,
        database=db,
        user=user,
        password=password,
    )

    return await db_manager.generate_manifest(output_file)
