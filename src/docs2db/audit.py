"""Audit functionality for finding stale and orphaned files."""

import json
from pathlib import Path

import structlog
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from docs2db.chunks import is_chunks_stale
from docs2db.const import METADATA_SCHEMA_VERSION
from docs2db.embeddings import EMBEDDING_CONFIGS, is_embedding_stale
from docs2db.exceptions import ContentError

logger = structlog.get_logger(__name__)


def perform_audit(
    content_dir: str,
    pattern: str = "**/*.json",
) -> bool:
    """Audit to find missing and stale files.

    Args:
        content_dir: Path to content directory
        pattern: File pattern to process

    Returns:
        True if successful, False if errors occurred

    Raises:
        ContentError: If content directory does not exist
    """
    content_path = Path(content_dir)

    if not content_path.exists():
        raise ContentError(f"Content directory does not exist: {content_dir}")

    logger.info("Auditing...")

    def source():
        return (f for f in content_path.glob(pattern) if f.name.count(".") == 1)

    source_count = sum(1 for _ in source())

    console = Console()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        source_task = progress.add_task("Sources", total=source_count)
        chunks_task = progress.add_task("Chunks", total=source_count)
        embed_task = progress.add_task("Embeds (granite)", total=source_count)
        metadata_task = progress.add_task("Metadata", total=source_count)
        orphan_task = progress.add_task("Orphans", total=source_count)
        stale_task = progress.add_task("Stales", total=source_count)
        unknown_task = progress.add_task("Unknowns", total=source_count)
        version_mismatch_task = progress.add_task(
            "Version mismatches", total=source_count
        )

        for file_path in content_path.glob(pattern):
            file_name = file_path.name

            if file_name.endswith(".chunks.json"):
                source_path = file_path.with_suffix("").with_suffix(".json")
                if not source_path.exists():
                    console.print(f"orphaned chunk    : {file_name}")
                    progress.advance(orphan_task)
                elif is_chunks_stale(file_path, source_path):
                    console.print(f"stale chunk       : {file_name}")
                    progress.advance(stale_task)
                else:
                    progress.advance(chunks_task)
            elif file_name.endswith(".gran.json"):
                chunks_path = file_path.with_suffix("").with_suffix(".chunks.json")
                if not chunks_path.exists():
                    console.print(f"orphaned embedding: {file_name}")
                    progress.advance(orphan_task)
                else:
                    granite_config = EMBEDDING_CONFIGS["granite-30m-english"]
                    if is_embedding_stale(
                        file_path,
                        chunks_path,
                        "granite-30m-english",
                        granite_config["model_id"],
                        granite_config["dimensions"],
                        granite_config["provider"],
                    ):
                        console.print(f"stale embedding   : {file_name}")
                        progress.advance(stale_task)
                    else:
                        progress.advance(embed_task)
            elif file_name.endswith(".meta.json"):
                source_path = file_path.with_suffix("").with_suffix(".json")
                if not source_path.exists():
                    console.print(f"orphaned metadata : {file_name}")
                    progress.advance(orphan_task)
                else:
                    # Check metadata version
                    try:
                        with open(file_path) as f:
                            meta_data = json.load(f)
                        if meta_data.get("metadata_version") != METADATA_SCHEMA_VERSION:
                            console.print(
                                f"version mismatch  : {file_name} "
                                f"(has {meta_data.get('metadata_version')}, expected {METADATA_SCHEMA_VERSION})"
                            )
                            progress.advance(version_mismatch_task)
                        else:
                            progress.advance(metadata_task)
                    except (json.JSONDecodeError, OSError) as e:
                        console.print(f"invalid metadata  : {file_name} ({e})")
                        progress.advance(orphan_task)
            elif file_name.endswith(".json"):
                progress.advance(source_task)
            else:
                console.print(f"unknown file      : {file_name}")
                progress.advance(unknown_task)

        result = {
            "sources": progress.tasks[source_task].completed,
            "chunks": progress.tasks[chunks_task].completed,
            "embeddings": progress.tasks[embed_task].completed,
            "metadata": progress.tasks[metadata_task].completed,
            "orphans": progress.tasks[orphan_task].completed,
            "stales": progress.tasks[stale_task].completed,
            "unknowns": progress.tasks[unknown_task].completed,
            "version_mismatches": progress.tasks[version_mismatch_task].completed,
        }

    logger.info(
        "\nFiles:\n"
        f"  Source           : {result['sources']:>6} files\n"
        f"  Chunks           : {result['chunks']:>6} files\n"
        f"  Embeddings       : {result['embeddings']:>6} files\n"
        f"  Metadata         : {result['metadata']:>6} files\n"
        f"  Orphaned         : {result['orphans']:>6} files\n"
        f"  Stale            : {result['stales']:>6} files\n"
        f"  Version mismatch : {result['version_mismatches']:>6} files\n"
        f"  Unknown          : {result['unknowns']:>6} files\n"
    )

    return (
        (result["sources"] == result["chunks"] == result["embeddings"])
        and result["orphans"] == 0
        and result["unknowns"] == 0
    )
