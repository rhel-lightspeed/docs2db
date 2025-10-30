"""RAG Pipeline Tools for docs2db"""

import asyncio
from pathlib import Path
from typing import Annotated, Optional

import structlog
import typer

from docs2db.audit import perform_audit
from docs2db.chunks import generate_chunks
from docs2db.config import settings
from docs2db.database import (
    check_database_status,
    dump_database,
    generate_manifest,
    load_documents,
    restore_database,
)
from docs2db.db_lifecycle import (
    destroy_database,
    get_database_logs,
    start_database,
    stop_database,
)
from docs2db.embed import generate_embeddings
from docs2db.exceptions import Docs2DBException
from docs2db.ingest import ingest as ingest_command
from docs2db.utils import cleanup_orphaned_workers

logger = structlog.get_logger(__name__)

app = typer.Typer(help="Make a RAG Database from source content")


@app.command()
def ingest(
    source_path: Annotated[
        str, typer.Argument(help="Path to directory or file to ingest")
    ],
    dry_run: Annotated[
        bool, typer.Option(help="Show what would be processed without doing it")
    ] = False,
    force: Annotated[
        bool, typer.Option(help="Force reprocessing even if files are up-to-date")
    ] = False,
) -> None:
    """Ingest files using docling to create JSON documents in /content directory."""
    try:
        if not ingest_command(source_path=source_path, dry_run=dry_run, force=force):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def chunk(
    content_dir: Annotated[
        str | None, typer.Option(help="Path to content directory")
    ] = None,
    pattern: Annotated[str | None, typer.Option(help="File pattern to process")] = None,
    force: Annotated[
        bool, typer.Option(help="Force reprocessing even if up-to-date")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Show what would process without doing it")
    ] = False,
    skip_context: Annotated[
        bool | None, typer.Option(help="Skip LLM contextual chunk generation (faster)")
    ] = None,
    context_model: Annotated[
        str | None, typer.Option(help="LLM model for context generation")
    ] = None,
    openai_url: Annotated[
        str | None,
        typer.Option(
            help="OpenAI-compatible API URL (default: http://localhost:11434 for Ollama)"
        ),
    ] = None,
    watsonx_url: Annotated[
        str | None,
        typer.Option(
            help="IBM WatsonX API URL (requires WATSONX_API_KEY and WATSONX_PROJECT_ID env vars)"
        ),
    ] = None,
    context_limit: Annotated[
        int | None,
        typer.Option(
            help="Override model context limit (in tokens) for map-reduce summarization"
        ),
    ] = None,
) -> None:
    """Generate chunks for content files."""
    # Precedence: CLI flags > Environment variables > Settings defaults
    final_content_dir = content_dir or settings.content_base_dir
    final_pattern = pattern or settings.chunking_pattern
    final_skip_context = (
        skip_context if skip_context is not None else settings.llm_skip_context
    )
    final_context_model = context_model or settings.llm_context_model
    final_openai_url = openai_url or settings.llm_openai_url
    final_watsonx_url = watsonx_url or settings.llm_watsonx_url
    final_context_limit = context_limit or settings.llm_context_limit_override

    # Validate mutually exclusive flags
    if final_openai_url and final_watsonx_url:
        logger.error(
            "Cannot specify both --openai-url and --watsonx-url (or their env var equivalents)"
        )
        raise typer.Exit(1)

    try:
        if not generate_chunks(
            final_content_dir,
            final_pattern,
            force,
            dry_run,
            skip_context=final_skip_context,
            context_model=final_context_model,
            openai_url=final_openai_url,
            watsonx_url=final_watsonx_url,
            context_limit_override=final_context_limit,
        ):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def embed(
    content_dir: Annotated[
        str | None, typer.Option(help="Path to content directory")
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(help="Embedding model to use"),
    ] = None,
    pattern: Annotated[
        str | None, typer.Option(help="File pattern for chunks files")
    ] = None,
    force: Annotated[
        bool, typer.Option(help="Force regeneration of existing embeddings")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Show what would process without doing it")
    ] = False,
) -> None:
    """Generate embeddings for chunked content files."""
    # Precedence: CLI flags > Environment variables > Settings defaults
    final_content_dir = content_dir or settings.content_base_dir
    final_model = model or settings.embedding_model
    final_pattern = pattern or settings.embedding_pattern

    try:
        if not generate_embeddings(
            final_content_dir, final_model, final_pattern, force, dry_run
        ):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def load(
    content_dir: Annotated[
        str, typer.Option(help="Path to content directory")
    ] = "content",
    model: Annotated[
        str,
        typer.Option(help="Embedding model to load (granite-30m-english)"),
    ] = "granite-30m-english",
    pattern: Annotated[
        str, typer.Option(help="File pattern for source files")
    ] = "**/*.json",
    force: Annotated[
        bool, typer.Option(help="Force reload of existing documents")
    ] = False,
    host: Annotated[
        Optional[str],
        typer.Option(help="Database host (auto-detected from compose file)"),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option(help="Database port (auto-detected from compose file)"),
    ] = None,
    db: Annotated[
        Optional[str],
        typer.Option(help="Database name (auto-detected from compose file)"),
    ] = None,
    user: Annotated[
        Optional[str],
        typer.Option(help="Database user (auto-detected from compose file)"),
    ] = None,
    password: Annotated[
        Optional[str],
        typer.Option(help="Database password (auto-detected from compose file)"),
    ] = None,
    batch_size: Annotated[
        int, typer.Option(help="Files per batch for each worker process")
    ] = 100,
    username: Annotated[
        str, typer.Option(help="Username to record in change log")
    ] = "",
    title: Annotated[
        Optional[str], typer.Option(help="Database title for metadata")
    ] = None,
    description: Annotated[
        Optional[str], typer.Option(help="Database description for metadata")
    ] = None,
    note: Annotated[
        Optional[str], typer.Option(help="Note about this load operation")
    ] = None,
) -> None:
    """Load documents, chunks, and embeddings into PostgreSQL database with pgvector."""

    try:
        if not asyncio.run(
            load_documents(
                content_dir=content_dir,
                model_name=model,
                pattern=pattern,
                host=host,
                port=port,
                db=db,
                user=user,
                password=password,
                force=force,
                batch_size=batch_size,
                username=username,
                title=title,
                description=description,
                note=note,
            )
        ):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command(name="db-status")
def db_status(
    host: Annotated[
        Optional[str],
        typer.Option(help="Database host (auto-detected from compose file)"),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option(help="Database port (auto-detected from compose file)"),
    ] = None,
    db: Annotated[
        Optional[str],
        typer.Option(help="Database name (auto-detected from compose file)"),
    ] = None,
    user: Annotated[
        Optional[str],
        typer.Option(help="Database user (auto-detected from compose file)"),
    ] = None,
    password: Annotated[
        Optional[str],
        typer.Option(help="Database password (auto-detected from compose file)"),
    ] = None,
) -> None:
    """Check database status and display statistics."""
    try:
        asyncio.run(
            check_database_status(
                host=host,
                port=port,
                db=db,
                user=user,
                password=password,
            )
        )
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command(name="db-dump")
def db_dump(
    output_file: Annotated[
        str, typer.Option(help="Output file path for the database dump")
    ] = "ragdb_dump.sql",
    host: Annotated[
        Optional[str],
        typer.Option(help="Database host (auto-detected from compose file)"),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option(help="Database port (auto-detected from compose file)"),
    ] = None,
    db: Annotated[
        Optional[str],
        typer.Option(help="Database name (auto-detected from compose file)"),
    ] = None,
    user: Annotated[
        Optional[str],
        typer.Option(help="Database user (auto-detected from compose file)"),
    ] = None,
    password: Annotated[
        Optional[str],
        typer.Option(help="Database password (auto-detected from compose file)"),
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Show pg_dump output")] = False,
) -> None:
    """Create a PostgreSQL dump file of the database."""
    try:
        if not dump_database(
            output_file=output_file,
            host=host,
            port=port,
            db=db,
            user=user,
            password=password,
            verbose=verbose,
        ):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command(name="db-restore")
def db_restore(
    input_file: Annotated[
        str, typer.Argument(help="Input file path for the database dump")
    ],
    host: Annotated[
        Optional[str],
        typer.Option(help="Database host (auto-detected from compose file)"),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option(help="Database port (auto-detected from compose file)"),
    ] = None,
    db: Annotated[
        Optional[str],
        typer.Option(help="Database name (auto-detected from compose file)"),
    ] = None,
    user: Annotated[
        Optional[str],
        typer.Option(help="Database user (auto-detected from compose file)"),
    ] = None,
    password: Annotated[
        Optional[str],
        typer.Option(help="Database password (auto-detected from compose file)"),
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Show psql output")] = False,
) -> None:
    """Restore a PostgreSQL database from a dump file."""
    try:
        if not restore_database(
            input_file=input_file,
            host=host,
            port=port,
            db=db,
            user=user,
            password=password,
            verbose=verbose,
        ):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def audit(
    content_dir: Annotated[
        str, typer.Option(help="Path to content directory")
    ] = "content",
    pattern: Annotated[str, typer.Option(help="File pattern to process")] = "**/*.json",
) -> None:
    """Audit to find missing and stale files."""
    try:
        if not perform_audit(
            content_dir=content_dir,
            pattern=pattern,
        ):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def manifest(
    output_file: Annotated[
        str, typer.Option(help="Output file path for the manifest")
    ] = "manifest.txt",
    host: Annotated[
        Optional[str],
        typer.Option(help="Database host (auto-detected from compose file)"),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option(help="Database port (auto-detected from compose file)"),
    ] = None,
    db: Annotated[
        Optional[str],
        typer.Option(help="Database name (auto-detected from compose file)"),
    ] = None,
    user: Annotated[
        Optional[str],
        typer.Option(help="Database user (auto-detected from compose file)"),
    ] = None,
    password: Annotated[
        Optional[str],
        typer.Option(help="Database password (auto-detected from compose file)"),
    ] = None,
) -> None:
    """Generate a manifest file with all unique source files from the database."""
    try:
        if not asyncio.run(
            generate_manifest(
                output_file=output_file,
                host=host,
                port=port,
                db=db,
                user=user,
                password=password,
            )
        ):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command(name="cleanup-workers")
def cleanup_workers() -> None:
    """Clean up orphaned worker processes."""
    try:
        if not cleanup_orphaned_workers():
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command(name="db-start")
def db_start(
    profile: Annotated[
        str, typer.Option(help="Docker compose profile to use")
    ] = "prod",
) -> None:
    """Start PostgreSQL database using Docker/Podman.

    Creates a default postgres-compose.yml if one doesn't exist.
    Requires Docker or Podman to be installed.
    """
    try:
        if not start_database(profile=profile):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command(name="db-stop")
def db_stop(
    profile: Annotated[
        str, typer.Option(help="Docker compose profile to use")
    ] = "prod",
) -> None:
    """Stop PostgreSQL database (data is preserved).

    The database can be restarted with 'docs2db db-start'.
    """
    try:
        if not stop_database(profile=profile):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command(name="db-destroy")
def db_destroy(
    profile: Annotated[
        str, typer.Option(help="Docker compose profile to use")
    ] = "prod",
    force: Annotated[bool, typer.Option(help="Skip confirmation prompt")] = False,
) -> None:
    """Stop database and remove all data (WARNING: destructive!).

    This will DELETE all database data. Use with caution.
    """
    if not force:
        confirm = typer.confirm(
            "⚠️  This will DELETE all database data. Are you sure?",
            default=False,
        )
        if not confirm:
            logger.info("Operation cancelled")
            raise typer.Exit(0)

    try:
        if not destroy_database(profile=profile):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command(name="db-logs")
def db_logs(
    follow: Annotated[
        bool, typer.Option("--follow", "-f", help="Follow logs in real-time")
    ] = False,
) -> None:
    """View PostgreSQL database logs.

    Use --follow to continuously stream logs (like tail -f).
    """
    try:
        if not get_database_logs(follow=follow):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def pipeline(
    source_path: Annotated[
        str, typer.Argument(help="Path to source directory or file to process")
    ],
    content_dir: Annotated[
        str, typer.Option(help="Content directory for intermediate files")
    ] = "docs2db_content",
    model: Annotated[
        str, typer.Option(help="Embedding model to use")
    ] = "granite-30m-english",
    skip_context: Annotated[
        bool, typer.Option(help="Skip LLM contextual chunk generation (faster)")
    ] = False,
    output_file: Annotated[
        str, typer.Option(help="Output SQL dump file")
    ] = "ragdb_dump.sql",
) -> None:
    """Run complete docs2db pipeline: start DB → ingest → chunk → embed → load → dump → stop.

    This is equivalent to running all steps in sequence:
      1. docs2db db-start
      2. docs2db ingest <source_path>
      3. docs2db chunk
      4. docs2db embed
      5. docs2db load
      6. docs2db db-dump
      7. docs2db db-stop

    The database is automatically cleaned up after completion.

    Example:
      docs2db pipeline /path/to/pdfs
      docs2db pipeline ~/Documents --output-file my-rag.sql
    """
    source = Path(source_path)
    if not source.exists():
        logger.error(f"Source path does not exist: {source_path}")
        raise typer.Exit(1)

    logger.info("=" * 60)
    logger.info("Starting docs2db pipeline")
    logger.info(f"Source: {source_path}")
    logger.info(f"Content directory: {content_dir}")
    logger.info(f"Embedding model: {model}")
    logger.info("=" * 60)

    try:
        # Step 1: Start database
        logger.info("\n[1/7] Starting database...")
        if not start_database():
            raise Docs2DBException("Failed to start database")

        # Step 2: Ingest
        logger.info("\n[2/7] Ingesting documents...")
        if not ingest_command(source_path=source_path, dry_run=False, force=False):
            raise Docs2DBException("Failed to ingest documents")

        # Step 3: Chunk
        logger.info("\n[3/7] Generating chunks...")
        if not generate_chunks(
            content_dir=content_dir,
            pattern="**/*.json",
            force=False,
            dry_run=False,
            skip_context=skip_context,
            context_model=settings.llm_context_model,
            openai_url=None,
            watsonx_url=None,
            context_limit_override=None,
        ):
            raise Docs2DBException("Failed to generate chunks")

        # Step 4: Embed
        logger.info("\n[4/7] Generating embeddings...")
        if not generate_embeddings(
            content_dir=content_dir,
            model_name=model,
            pattern="**/*.chunks.json",
            force=False,
            dry_run=False,
        ):
            raise Docs2DBException("Failed to generate embeddings")

        # Step 5: Load
        logger.info("\n[5/7] Loading into database...")
        if not asyncio.run(
            load_documents(
                content_dir=content_dir,
                model_name=model,
                pattern="**/*.json",
                host=None,
                port=None,
                db=None,
                user=None,
                password=None,
                force=False,
                batch_size=100,
                username="",
                title=None,
                description=None,
                note=f"Created by pipeline from {source_path}",
            )
        ):
            raise Docs2DBException("Failed to load into database")

        # Step 6: Dump
        logger.info("\n[6/7] Creating database dump...")
        if not dump_database(
            output_file=output_file,
            host=None,
            port=None,
            db=None,
            user=None,
            password=None,
            verbose=False,
        ):
            raise Docs2DBException("Failed to create database dump")

        # Step 7: Stop database
        logger.info("\n[7/7] Stopping database...")
        if not stop_database():
            logger.warning("Failed to stop database (not fatal)")

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Database dump created: {output_file}")
        logger.info("=" * 60)

    except Docs2DBException as e:
        logger.error(f"\nPipeline failed: {e}")
        logger.info("\nCleaning up...")

        # Try to stop database on failure
        try:
            stop_database()
        except Exception:
            pass  # Ignore cleanup errors

        raise typer.Exit(1)
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        logger.info("Cleaning up...")

        # Try to stop database on interrupt
        try:
            stop_database()
        except Exception:
            pass  # Ignore cleanup errors

        raise typer.Exit(130)  # Standard exit code for SIGINT
