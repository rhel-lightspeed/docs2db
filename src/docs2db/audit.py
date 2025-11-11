"""Audit functionality for finding missing and stale files."""

import json
from pathlib import Path

import structlog
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from docs2db.chunks import is_chunks_stale
from docs2db.config import settings
from docs2db.const import METADATA_SCHEMA_VERSION
from docs2db.embeddings import EMBEDDING_CONFIGS, is_embedding_stale
from docs2db.exceptions import ContentError

logger = structlog.get_logger(__name__)


def perform_audit(
    content_dir: str | None = None,
    pattern: str | None = None,
) -> bool:
    """Audit to find missing and stale files.

    Args:
        content_dir: Path to content directory (defaults to settings.content_base_dir)
        pattern: Directory pattern to audit (e.g., "external/**" or "additional_documents/*")
                Defaults to "**" which audits all directories.

    Returns:
        True if all files present and up-to-date, False if issues detected

    Raises:
        ContentError: If content directory does not exist or pattern doesn't end with glob wildcard
    """

    if content_dir is None:
        content_dir = settings.content_base_dir
    if pattern is None:
        pattern = "**"

    logger.info(f"Auditing content_dir: {content_dir}")
    logger.info(f"Using directory pattern: {pattern}")

    # Append /source.json to the pattern
    # Works for both exact directory paths and glob patterns:
    # - "dir/subdir" -> "dir/subdir/source.json" (exact file)
    # - "dir/**" -> "dir/**/source.json" (glob pattern)
    source_pattern = f"{pattern}/source.json"

    content_path = Path(content_dir)

    if not content_path.exists():
        raise ContentError(f"Content directory does not exist: {content_dir}")

    # Find all terminal (leaf) directories - directories with no subdirectories
    def get_terminal_directories(path: Path) -> list[Path]:
        """Get all terminal (leaf) directories under the given path."""
        terminal_dirs = []
        for item in path.rglob("*"):
            if item.is_dir():
                # Check if this directory has any subdirectories
                has_subdirs = any(subitem.is_dir() for subitem in item.iterdir())
                if not has_subdirs:
                    terminal_dirs.append(item)
        return terminal_dirs

    # Find all source.json files in matching directories (sorted for deterministic order)
    source_files = sorted(content_path.glob(source_pattern))
    source_count = len(source_files)

    # Find all terminal directories for orphan check
    terminal_dirs = get_terminal_directories(content_path)
    terminal_count = len(terminal_dirs)

    # Get all embedding model keywords for checking
    embedding_keywords = [config["keyword"] for config in EMBEDDING_CONFIGS.values()]

    # Collect messages to print after progress completes
    messages = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        source_task = progress.add_task("Sources", total=source_count)
        chunks_task = progress.add_task("Chunks", total=source_count)
        embed_task = progress.add_task("Embeds", total=source_count)
        metadata_task = progress.add_task("Metadata", total=source_count)
        stale_chunks_task = progress.add_task("Stale chunks", total=source_count)
        stale_embeds_task = progress.add_task("Stale embeds", total=source_count)
        version_mismatch_task = progress.add_task(
            "Version mismatches", total=source_count
        )
        orphan_dir_task = progress.add_task("Orphan dirs", total=terminal_count)
        zero_chunks_task = progress.add_task("Zero chunks", total=source_count)

        for source_file in source_files:
            # source_file is .../doc_dir/source.json
            doc_dir = source_file.parent
            doc_name = doc_dir.name
            # Get full relative path from content directory for better reporting
            doc_rel_path = doc_dir.relative_to(content_path)

            # Advance source task for this document
            progress.advance(source_task)

            # Check for chunks.json
            chunks_file = doc_dir / "chunks.json"
            has_zero_chunks = False
            if not chunks_file.exists():
                messages.append(f"missing chunks    : {doc_rel_path}/chunks.json")
            elif is_chunks_stale(chunks_file, source_file):
                messages.append(f"stale chunk       : {doc_rel_path}/chunks.json")
                progress.advance(stale_chunks_task)
            else:
                # Check if chunks file has zero chunks
                try:
                    with open(chunks_file) as f:
                        chunks_data = json.load(f)
                        chunk_count = chunks_data.get("metadata", {}).get(
                            "chunk_count", 0
                        )
                        if chunk_count == 0:
                            has_zero_chunks = True
                            progress.advance(zero_chunks_task)
                        else:
                            progress.advance(chunks_task)
                except (json.JSONDecodeError, KeyError, OSError):
                    # If we can't read it, count it as a normal chunk file
                    progress.advance(chunks_task)

            # Check for embedding files (all known keywords)
            found_any_embedding = False
            has_stale_embedding = False
            for keyword in embedding_keywords:
                embed_file = doc_dir / f"{keyword}.json"
                if embed_file.exists():
                    found_any_embedding = True
                    # Find the model name from keyword
                    model = None
                    model_config = None
                    for name, config in EMBEDDING_CONFIGS.items():
                        if config["keyword"] == keyword:
                            model = name
                            model_config = config
                            break

                    assert model is not None
                    assert model_config is not None
                    if chunks_file.exists():
                        if is_embedding_stale(
                            embed_file,
                            chunks_file,
                            model,
                            model_config["dimensions"],
                        ):
                            messages.append(
                                f"stale embedding   : {doc_rel_path}/{keyword}.json"
                            )
                            has_stale_embedding = True

            # Advance embed_task once per document, not once per embedding file
            if found_any_embedding:
                if has_stale_embedding:
                    progress.advance(stale_embeds_task)
                else:
                    progress.advance(embed_task)
            else:
                # Only report missing embeddings if the document has chunks
                if not has_zero_chunks:
                    messages.append(f"missing embeddings: {doc_rel_path}/")

            # Check for unknown files in the document directory
            known_files = {"source.json", "chunks.json", "meta.json"}
            known_files.update(f"{kw}.json" for kw in embedding_keywords)

            for file in doc_dir.iterdir():
                if (
                    file.is_file()
                    and file.suffix == ".json"
                    and file.name not in known_files
                ):
                    messages.append(f"unknown file      : {doc_rel_path}/{file.name}")

            # Check for meta.json
            meta_file = doc_dir / "meta.json"
            if not meta_file.exists():
                messages.append(f"missing metadata  : {doc_rel_path}/meta.json")
            else:
                # Check metadata version
                try:
                    with open(meta_file) as f:
                        meta_data = json.load(f)
                    if meta_data.get("metadata_version") != METADATA_SCHEMA_VERSION:
                        messages.append(
                            f"version mismatch  : {doc_rel_path}/meta.json "
                            f"(has {meta_data.get('metadata_version')}, expected {METADATA_SCHEMA_VERSION})"
                        )
                        progress.advance(version_mismatch_task)
                    else:
                        progress.advance(metadata_task)
                except (json.JSONDecodeError, OSError) as e:
                    messages.append(
                        f"invalid metadata  : {doc_rel_path}/meta.json ({e})"
                    )

        # Check for orphaned directories (terminal directories without source.json)
        for terminal_dir in terminal_dirs:
            source_file = terminal_dir / "source.json"
            if source_file.exists():
                progress.advance(orphan_dir_task)
            else:
                # This is an orphaned directory
                relative_dir = terminal_dir.relative_to(content_path)
                messages.append(f"orphan directory  : {relative_dir}/ (no source.json)")

        result = {
            "sources": progress.tasks[source_task].completed,
            "chunks": progress.tasks[chunks_task].completed,
            "embeddings": progress.tasks[embed_task].completed,
            "metadata": progress.tasks[metadata_task].completed,
            "stale_chunks": progress.tasks[stale_chunks_task].completed,
            "stale_embeds": progress.tasks[stale_embeds_task].completed,
            "version_mismatches": progress.tasks[version_mismatch_task].completed,
            "orphan_dirs": terminal_count - progress.tasks[orphan_dir_task].completed,
            "zero_chunks": progress.tasks[zero_chunks_task].completed,
        }

    # Print all collected messages
    for msg in messages:
        print(msg)

    logger.info(
        "\nAudit Results:\n"
        f"  Source           : {result['sources']:>6} files\n"
        f"  Chunks           : {result['chunks']:>6} files\n"
        f"  Embeddings       : {result['embeddings']:>6} files\n"
        f"  Zero chunks      : {result['zero_chunks']:>6} files\n"
        f"  Metadata         : {result['metadata']:>6} files\n"
        f"  Stale chunks     : {result['stale_chunks']:>6} files\n"
        f"  Stale embeds     : {result['stale_embeds']:>6} files\n"
        f"  Version mismatch : {result['version_mismatches']:>6} files\n"
        f"  Orphan dirs      : {result['orphan_dirs']:>6} dirs\n"
    )

    # Return True only if all counts match and there are no issues
    # Logic:
    # - All sources should have chunks and metadata
    # - Embeddings should exist for all docs with non-zero chunks
    all_match = (
        result["sources"] == (result["chunks"] + result["zero_chunks"])
        and result["sources"] == result["metadata"]
        and result["embeddings"] == result["chunks"]
    )
    no_issues = (
        result["stale_chunks"] == 0
        and result["stale_embeds"] == 0
        and result["version_mismatches"] == 0
        and result["orphan_dirs"] == 0
    )

    if all_match and no_issues:
        logger.info("Audit complete: all files up-to-date")
        return True
    else:
        logger.warning("Audit failed: issues detected")
        return False
