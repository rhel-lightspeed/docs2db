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

from docs2db.const import DATABASE_SCHEMA_VERSION
from docs2db.embeddings import EMBEDDING_CONFIGS, create_embedding_filename
from docs2db.exceptions import ConfigurationError, ContentError, DatabaseError
from docs2db.multiproc import BatchProcessor, setup_worker_logging

logger = structlog.get_logger()


def get_db_config() -> Dict[str, str]:
    """Get database connection parameters from multiple sources.

    Configuration precedence (highest to lowest):
    1. Environment variables (POSTGRES_HOST, POSTGRES_PORT, etc.)
    2. DATABASE_URL environment variable
    3. postgres-compose.yml in current working directory
    4. Default values (localhost:5432, user=postgres, db=ragdb)

    Raises:
        ConfigurationError: If both DATABASE_URL and individual POSTGRES_* vars are set

    Returns:
        Dict with keys: host, port, database, user, password
    """
    # Check for conflicting configuration sources
    has_database_url = bool(os.getenv("DATABASE_URL"))
    has_postgres_vars = any([
        os.getenv("POSTGRES_HOST"),
        os.getenv("POSTGRES_PORT"),
        os.getenv("POSTGRES_DB"),
        os.getenv("POSTGRES_USER"),
        os.getenv("POSTGRES_PASSWORD"),
    ])

    if has_database_url and has_postgres_vars:
        raise ConfigurationError(
            "Conflicting database configuration: both DATABASE_URL and individual "
            "POSTGRES_* environment variables are set. Please use one or the other."
        )

    # Start with sensible defaults
    config = {
        "host": "localhost",
        "port": "5432",
        "database": "ragdb",
        "user": "postgres",
        "password": "postgres",
    }

    # Try postgres-compose.yml in current working directory
    compose_file = Path.cwd() / "postgres-compose.yml"
    if compose_file.exists():
        try:
            with open(compose_file, "r") as f:
                compose_data = yaml.safe_load(f)

            db_service = compose_data.get("services", {}).get("db", {})
            env = db_service.get("environment", {})

            if "POSTGRES_DB" in env:
                config["database"] = env["POSTGRES_DB"]
            if "POSTGRES_USER" in env:
                config["user"] = env["POSTGRES_USER"]
            if "POSTGRES_PASSWORD" in env:
                config["password"] = env["POSTGRES_PASSWORD"]

            # Extract port from ports mapping if available
            ports = db_service.get("ports", [])
            for port_mapping in ports:
                if isinstance(port_mapping, str) and ":5432" in port_mapping:
                    host_port = port_mapping.split(":")[0]
                    config["port"] = host_port
                    break
        except Exception as e:
            # If compose file exists but can't be parsed, warn but continue with defaults
            logger.warning(f"Could not parse postgres-compose.yml: {e}")

    # DATABASE_URL takes precedence over compose file but not over individual vars
    if has_database_url:
        database_url = os.getenv("DATABASE_URL", "")
        try:
            # Parse postgresql://user:password@host:port/database
            # Support both postgresql:// and postgres:// schemes
            if database_url.startswith(("postgresql://", "postgres://")):
                # Remove scheme
                url_without_scheme = database_url.split("://", 1)[1]

                # Split into credentials@location and database
                if "@" in url_without_scheme:
                    credentials, location = url_without_scheme.split("@", 1)

                    # Parse credentials
                    if ":" in credentials:
                        config["user"], config["password"] = credentials.split(":", 1)
                    else:
                        config["user"] = credentials

                    # Parse location and database
                    if "/" in location:
                        host_port, config["database"] = location.split("/", 1)
                    else:
                        host_port = location

                    # Parse host and port
                    if ":" in host_port:
                        config["host"], config["port"] = host_port.split(":", 1)
                    else:
                        config["host"] = host_port
                else:
                    raise ConfigurationError(
                        f"Invalid DATABASE_URL format (missing @): {database_url}"
                    )
            else:
                raise ConfigurationError(
                    f"Invalid DATABASE_URL scheme. Expected postgresql:// or postgres://, "
                    f"got: {database_url.split('://')[0] if '://' in database_url else database_url}"
                )
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Failed to parse DATABASE_URL: {e}. "
                f"Expected format: postgresql://user:password@host:port/database"
            ) from e

    # Individual environment variables override everything (highest precedence)
    if os.getenv("POSTGRES_HOST"):
        config["host"] = os.getenv("POSTGRES_HOST", "")
    if os.getenv("POSTGRES_PORT"):
        config["port"] = os.getenv("POSTGRES_PORT", "")
    if os.getenv("POSTGRES_DB"):
        config["database"] = os.getenv("POSTGRES_DB", "")
    if os.getenv("POSTGRES_USER"):
        config["user"] = os.getenv("POSTGRES_USER", "")
    if os.getenv("POSTGRES_PASSWORD"):
        config["password"] = os.getenv("POSTGRES_PASSWORD", "")

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

    async def insert_schema_metadata(
        self,
        conn,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Insert initial schema metadata record."""
        await conn.execute(
            """
            INSERT INTO schema_metadata (
                title, description,
                schema_version, embedding_models_count
            ) VALUES (%s, %s, %s, 0)
            """,
            [title, description, DATABASE_SCHEMA_VERSION],
        )

    async def update_schema_metadata(
        self,
        conn,
        title: Optional[str] = None,
        description: Optional[str] = None,
        embedding_models_count: Optional[int] = None,
    ) -> None:
        """Update schema metadata record."""
        updates = []
        params = []

        if title is not None:
            updates.append("title = %s")
            params.append(title)
        if description is not None:
            updates.append("description = %s")
            params.append(description)
        if embedding_models_count is not None:
            updates.append("embedding_models_count = %s")
            params.append(embedding_models_count)

        if updates:
            updates.append("last_modified_at = NOW()")
            sql = f"UPDATE schema_metadata SET {', '.join(updates)} WHERE id = 1"
            await conn.execute(sql, params)

    def format_schema_change_display(self, change_data: dict) -> str:
        """Format a schema change record for display.

        Only includes fields that have meaningful values.
        """
        lines = []

        # Header with ID
        lines.append(f"\nUpdate #{change_data['id']}:")

        # Timestamp (always show)
        timestamp = (
            change_data["changed_at"].strftime("%Y-%m-%d %H:%M")
            if change_data["changed_at"]
            else "Unknown"
        )
        lines.append(f"  Timestamp      : {timestamp}")

        # User (only if set)
        if change_data["changed_by_user"]:
            lines.append(f"  User           : {change_data['changed_by_user']}")

        # Version (only if set)
        if change_data["changed_by_version"]:
            lines.append(f"  Version        : {change_data['changed_by_version']}")

        # Tool (only if set)
        if change_data["changed_by_tool"]:
            lines.append(f"  Tool           : {change_data['changed_by_tool']}")

        # Documents (only if added or deleted)
        if change_data["documents_added"] > 0:
            lines.append(f"  Documents added: {change_data['documents_added']}")
        if change_data["documents_deleted"] > 0:
            lines.append(f"  Documents deleted: {change_data['documents_deleted']}")

        # Chunks (only if added or deleted)
        if change_data["chunks_added"] > 0:
            lines.append(f"  Chunks added   : {change_data['chunks_added']}")
        if change_data["chunks_deleted"] > 0:
            lines.append(f"  Chunks deleted : {change_data['chunks_deleted']}")

        # Embeddings (only if added or deleted)
        if change_data["embeddings_added"] > 0:
            lines.append(f"  Embeds added   : {change_data['embeddings_added']}")
        if change_data["embeddings_deleted"] > 0:
            lines.append(f"  Embeds deleted : {change_data['embeddings_deleted']}")

        # Models added (only if any)
        if change_data["embedding_models_added"]:
            models_str = ", ".join(change_data["embedding_models_added"])
            lines.append(f"  Models added   : {models_str}")

        # Notes (only if set)
        if change_data["notes"]:
            lines.append(f"  Notes          : {change_data['notes']}")

        return "\n".join(lines)

    async def insert_model(
        self,
        conn,
        name: str,
        dimensions: int,
        provider: Optional[str] = None,
        description: Optional[str] = None,
    ) -> int:
        """Insert a new model and return its ID.

        Returns the model ID if inserted, or existing ID if already exists.
        Raises DatabaseError if insertion fails.
        """
        # Check if model already exists
        result = await conn.execute("SELECT id FROM models WHERE name = %s", [name])
        row = await result.fetchone()
        if row:
            return row[0]

        # Insert new model
        result = await conn.execute(
            """
            INSERT INTO models (name, dimensions, provider, description)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            [name, dimensions, provider, description],
        )
        row = await result.fetchone()
        if row is None:
            raise DatabaseError(f"Failed to insert model: {name}")
        return row[0]

    async def get_model_id(self, conn, name: str) -> Optional[int]:
        """Get model ID by name."""
        result = await conn.execute("SELECT id FROM models WHERE name = %s", [name])
        row = await result.fetchone()
        return row[0] if row else None

    async def get_model_info(self, conn, model_id: int) -> Optional[dict]:
        """Get model information by ID."""
        result = await conn.execute(
            "SELECT id, name, dimensions, provider, description, created_at FROM models WHERE id = %s",
            [model_id],
        )
        row = await result.fetchone()
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "dimensions": row[2],
                "provider": row[3],
                "description": row[4],
                "created_at": row[5],
            }
        return None

    async def insert_schema_change(
        self,
        conn,
        changed_by_user: str = "",
        documents_added: int = 0,
        documents_deleted: int = 0,
        chunks_added: int = 0,
        chunks_deleted: int = 0,
        embeddings_added: int = 0,
        embeddings_deleted: int = 0,
        embedding_models_added: Optional[List[str]] = None,
        notes: str = "",
    ) -> None:
        """Insert a schema change record."""
        if embedding_models_added is None:
            embedding_models_added = []

        await conn.execute(
            """
            INSERT INTO schema_changes (
                changed_by_tool, changed_by_version, changed_by_user,
                documents_added, documents_deleted,
                chunks_added, chunks_deleted,
                embeddings_added, embeddings_deleted,
                embedding_models_added,
                schema_version,
                notes
            ) VALUES (
                'docs2db', %s, %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s,
                %s,
                %s
            )
            """,
            [
                DATABASE_SCHEMA_VERSION,
                changed_by_user,
                documents_added,
                documents_deleted,
                chunks_added,
                chunks_deleted,
                embeddings_added,
                embeddings_deleted,
                embedding_models_added,
                DATABASE_SCHEMA_VERSION,
                notes,
            ],
        )

    async def initialize_schema(self) -> None:
        """Initialize database schema with tables for documents, chunks, and embeddings."""
        # Check if schema already exists and create it if needed
        async with await self.get_direct_connection() as conn:
            tables_result = await conn.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('documents', 'chunks', 'embeddings', 'schema_metadata', 'schema_changes')
            """)
            existing_tables = [row[0] for row in await tables_result.fetchall()]
            schema_exists = len(existing_tables) == 5

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
            contextual_text TEXT,
            metadata JSONB,
            text_search_vector tsvector,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(document_id, chunk_index)
        );

        -- Models table: stores embedding model metadata
        CREATE TABLE IF NOT EXISTS models (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            dimensions INTEGER NOT NULL,
            provider TEXT,
            description TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Embeddings table: stores vector embeddings for chunks
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
            model_id INTEGER REFERENCES models(id) ON DELETE CASCADE,
            embedding VECTOR, -- Dynamic dimension based on model
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(chunk_id, model_id)
        );

        -- Schema metadata: singleton table tracking current database state
        CREATE TABLE IF NOT EXISTS schema_metadata (
            id INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            title TEXT,
            description TEXT,
            schema_version TEXT NOT NULL,
            embedding_models_count INT NOT NULL DEFAULT 0,
            last_modified_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Schema changes: audit log of all changes (id=1 is creation event)
        CREATE TABLE IF NOT EXISTS schema_changes (
            id SERIAL PRIMARY KEY,
            changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            changed_by_tool TEXT NOT NULL,
            changed_by_version TEXT,
            changed_by_user TEXT,
            documents_added INT DEFAULT 0,
            documents_deleted INT DEFAULT 0,
            chunks_added INT DEFAULT 0,
            chunks_deleted INT DEFAULT 0,
            embeddings_added INT DEFAULT 0,
            embeddings_deleted INT DEFAULT 0,
            embedding_models_added TEXT[],
            schema_version TEXT NOT NULL,
            notes TEXT
        );

        -- Indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
        CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_embeddings_model_id ON embeddings(model_id);
        CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
        CREATE INDEX IF NOT EXISTS idx_chunks_text_search ON chunks USING GIN(text_search_vector);

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
                # Insert initial schema metadata
                await self.insert_schema_metadata(conn)

                # Insert initial change record (creation event)
                await self.insert_schema_change(
                    conn, changed_by_user="", notes="Database initialized"
                )

                await conn.commit()
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
        from docs2db.embeddings import EMBEDDING_CONFIGS

        processed = 0
        errors = 0

        # Get model info and ensure model exists in database
        # Extract model_name from first file (all files in batch use same model)
        if not files_data:
            return 0, 0

        model_name = files_data[0][2]  # model is 3rd element in tuple
        model_config = EMBEDDING_CONFIGS.get(model_name, {})
        model_full_name = model_config.get("model_id", model_name)
        model_dimensions = model_config.get("dimensions", 0)
        model_provider = model_config.get("provider")

        # Insert model if it doesn't exist and get model_id
        async with await self.get_direct_connection() as conn:
            model_id = await self.insert_model(
                conn,
                name=model_full_name,
                dimensions=model_dimensions,
                provider=model_provider,
                description=f"Embedding model: {model_name}",
            )
            await conn.commit()

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
                                WHERE d.path = %s AND e.model_id = %s
                                """,
                                (str(source_file), model_id),
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
                                chunk["text"],
                                chunk["contextual_text"],
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
                            INSERT INTO chunks (document_id, chunk_index, text, contextual_text, metadata, text_search_vector)
                            VALUES (%s, %s, %s, %s, %s, to_tsvector('english', %s))
                            ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                                text = EXCLUDED.text,
                                contextual_text = EXCLUDED.contextual_text,
                                metadata = EXCLUDED.metadata,
                                text_search_vector = to_tsvector('english', EXCLUDED.contextual_text)
                            RETURNING id
                            """,
                            chunk_data
                            + (chunk_data[3],),  # Add contextual_text for tsvector
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
                        embedding_data_tuple = (
                            chunk_id,
                            model_id,
                            embedding_vector,
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
                            INSERT INTO embeddings (chunk_id, model_id, embedding)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (chunk_id, model_id) DO UPDATE SET
                                embedding = EXCLUDED.embedding,
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

            # Embedding stats by model (join with models table)
            embedding_stats = await conn.execute(
                """
                SELECT m.name, COUNT(e.id) as count, m.dimensions
                FROM models m
                LEFT JOIN embeddings e ON e.model_id = m.id
                GROUP BY m.id, m.name, m.dimensions
                ORDER BY m.name
                """
            )
            embedding_models = {}
            async for row in embedding_stats:
                model_name, count, dimensions = row
                embedding_models[model_name] = {
                    "count": count,
                    "dimensions": dimensions if dimensions else 0,
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
            logger.error(
                "Database is not running. Start database with 'docs2db db-start'"
            )
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
                f"  model     : {model_name}\n"
                f"  dimensions: {model_info['dimensions']}\n"
                f"  embeddings: {model_info['count']}"
            )

    # Display schema metadata if available
    async with await db_manager.get_direct_connection() as conn:
        try:
            metadata_result = await conn.execute(
                "SELECT * FROM schema_metadata WHERE id = 1"
            )
            metadata_row = await metadata_result.fetchone()
            if metadata_row and metadata_result.description:
                columns = [desc[0] for desc in metadata_result.description]
                metadata = dict(zip(columns, metadata_row))

                logger.info(
                    "\nSchema Metadata:\n"
                    f"  Version        : {metadata['schema_version']}\n"
                    f"  Title          : {metadata['title'] or '(not set)'}\n"
                    f"  Description    : {metadata['description'] or '(not set)'}\n"
                    f"  Models         : {metadata['embedding_models_count']}\n"
                    f"  Last modified  : {metadata['last_modified_at'].strftime('%Y-%m-%d %H:%M') if metadata['last_modified_at'] else 'Unknown'}"
                )
        except Exception:
            # Schema metadata table doesn't exist yet
            pass

    # Display recent schema changes (last 5)
    async with await db_manager.get_direct_connection() as conn:
        try:
            changes_result = await conn.execute("""
                SELECT
                    id,
                    changed_at,
                    changed_by_tool,
                    changed_by_version,
                    changed_by_user,
                    documents_added,
                    documents_deleted,
                    chunks_added,
                    chunks_deleted,
                    embeddings_added,
                    embeddings_deleted,
                    embedding_models_added,
                    notes
                FROM schema_changes
                ORDER BY id DESC
                LIMIT 5
            """)

            changes = []
            async for row in changes_result:
                if changes_result.description:
                    columns = [desc[0] for desc in changes_result.description]
                    change_data = dict(zip(columns, row))
                    changes.append(change_data)

            if changes:
                logger.info("\nRecent Changes (last 5):")
                for change_data in changes:
                    logger.info(db_manager.format_schema_change_display(change_data))
        except Exception:
            # Schema changes table doesn't exist yet
            pass

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

    logger.info("Database status check complete")


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
    username: str = "",
    title: Optional[str] = None,
    description: Optional[str] = None,
    note: Optional[str] = None,
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

    from docs2db.embeddings import EMBEDDING_CONFIGS

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

    # Handle schema_metadata (insert or update)
    async with await db_manager.get_direct_connection() as conn:
        # Check if metadata exists and if it's been configured
        result = await conn.execute("SELECT title FROM schema_metadata WHERE id = 1")
        row = await result.fetchone()
        metadata_exists = row is not None
        metadata_configured = (
            metadata_exists and row[0] is not None
        )  # Has title been set?

        if metadata_configured:
            # Update existing, configured metadata
            await db_manager.update_schema_metadata(
                conn,
                title=title,
                description=description,
            )
        else:
            # First configuration - set title and description
            # (metadata record may exist from initialize_schema, but hasn't been configured yet)
            if metadata_exists:
                # Update the initialized-but-not-configured record
                await db_manager.update_schema_metadata(
                    conn,
                    title=title,
                    description=description,
                )
            else:
                # Insert new metadata (shouldn't happen after initialize_schema)
                await db_manager.insert_schema_metadata(
                    conn,
                    title=title,
                    description=description,
                )
        await conn.commit()

    content_path = Path(content_dir)
    if not content_path.exists():
        raise ContentError(f"Content directory does not exist: {content_dir}")

    count, file_pairs_iter = await load_files(content_path, model_name, pattern, force)

    if not count:
        logger.info("No files to load")
        return True

    logger.info(f"Found {count} embedding files for model: {model_name}")

    # Get model full name from config (EMBEDDING_CONFIGS imported at top of function)
    model_config = EMBEDDING_CONFIGS.get(model_name, {})
    model_full_name = model_config.get("model_id", model_name)

    # Count records BEFORE the operation starts
    async with await db_manager.get_direct_connection() as conn:
        result = await conn.execute("SELECT COUNT(*) FROM documents")
        row = await result.fetchone()
        documents_before = row[0] if row else 0

        result = await conn.execute("SELECT COUNT(*) FROM chunks")
        row = await result.fetchone()
        chunks_before = row[0] if row else 0

        result = await conn.execute("SELECT COUNT(*) FROM embeddings")
        row = await result.fetchone()
        embeddings_before = row[0] if row else 0

        # Check if this model already exists in models table
        result = await conn.execute(
            "SELECT COUNT(*) FROM models WHERE name = %s", [model_full_name]
        )
        row = await result.fetchone()
        model_existed_before = (row[0] if row else 0) > 0

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

    # Record this load operation in schema_changes
    if loaded > 0:
        async with await db_manager.get_direct_connection() as conn:
            # Get current embedding model count from models table
            result = await conn.execute("SELECT COUNT(*) FROM models")
            row = await result.fetchone()
            model_count = row[0] if row else 0

            # Update embedding_models_count in metadata
            await db_manager.update_schema_metadata(
                conn,
                embedding_models_count=model_count,
            )

            # Build note for this operation
            operation_note = (
                note if note else f"Loaded {loaded} files with model {model_name}"
            )
            if errors > 0:
                operation_note += f" ({errors} errors)"

            # Count records AFTER the operation and calculate the diff
            result = await conn.execute("SELECT COUNT(*) FROM documents")
            row = await result.fetchone()
            documents_after = row[0] if row else 0
            documents_added_count = documents_after - documents_before

            result = await conn.execute("SELECT COUNT(*) FROM chunks")
            row = await result.fetchone()
            chunks_after = row[0] if row else 0
            chunks_added_count = chunks_after - chunks_before

            result = await conn.execute("SELECT COUNT(*) FROM embeddings")
            row = await result.fetchone()
            embeddings_after = row[0] if row else 0
            embeddings_added_count = embeddings_after - embeddings_before

            # Check if this model is new (didn't exist before, exists now)
            embedding_models_added = []
            if not model_existed_before and embeddings_added_count > 0:
                embedding_models_added = [model_full_name]

            # Insert change record with all statistics
            await db_manager.insert_schema_change(
                conn,
                changed_by_user=username,
                documents_added=documents_added_count,
                chunks_added=chunks_added_count,
                embeddings_added=embeddings_added_count,
                embedding_models_added=embedding_models_added,
                notes=operation_note,
            )

            await conn.commit()

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
