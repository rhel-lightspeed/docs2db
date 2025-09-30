"""RAG Pipeline Tools for docs2db"""

import asyncio
from typing import Annotated, Optional

import structlog
import typer

from docs2db.audit import perform_audit
from docs2db.chunks import generate_chunks
from docs2db.database import (
    check_database_status,
    dump_database,
    generate_manifest,
    load_documents,
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
        str, typer.Option(help="Path to content directory")
    ] = "content",
    pattern: Annotated[str, typer.Option(help="File pattern to process")] = "**/*.json",
    force: Annotated[
        bool, typer.Option(help="Force reprocessing even if up-to-date")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Show what would process without doing it")
    ] = False,
) -> None:
    """Generate chunks for content files."""
    try:
        if not generate_chunks(content_dir, pattern, force, dry_run):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def embed(
    content_dir: Annotated[
        str, typer.Option(help="Path to content directory")
    ] = "content",
    model: Annotated[
        str,
        typer.Option(help="Embedding model to use (granite-30m-english)"),
    ] = "granite-30m-english",
    pattern: Annotated[
        str, typer.Option(help="File pattern for chunks files")
    ] = "**/*.chunks.json",
    force: Annotated[
        bool, typer.Option(help="Force regeneration of existing embeddings")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Show what would process without doing it")
    ] = False,
) -> None:
    """Generate embeddings for chunked content files."""
    try:
        if not generate_embeddings(content_dir, model, pattern, force, dry_run):
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
