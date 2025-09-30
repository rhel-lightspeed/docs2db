"""Ingest files using docling to create JSON documents."""

import json
import os
from pathlib import Path
from typing import Iterator

import structlog
from docling.document_converter import DocumentConverter

from docs2db.exceptions import Docs2DBException

logger = structlog.get_logger(__name__)


def is_ingestion_stale(content_file: Path, source_file: Path) -> bool:
    """Check if content file is stale compared to source.

    Args:
        content_file: Path to the docling JSON file to check
        source_file: Path to the source file to compare against

    Returns:
        bool: True if content file needs regeneration, False if current
    """
    if not content_file.exists():
        return True

    try:
        content_mtime = content_file.stat().st_mtime
        source_mtime = source_file.stat().st_mtime
        return source_mtime > content_mtime

    except (OSError, FileNotFoundError):
        # If we can't stat the files, assume it needs regeneration
        return True


def find_ingestible_files(source_path: Path) -> Iterator[Path]:
    """Find all files that can be processed by docling.

    Args:
        source_path: Path to search for ingestible files

    Yields:
        Path: Files that can be processed by docling
    """
    if not source_path.exists():
        raise Docs2DBException(f"Source path does not exist: {source_path}")

    if source_path.is_file():
        yield source_path
        return

    # File extensions that docling can process
    # Based on docling's InputFormat enum
    supported_extensions = {
        ".html",
        ".htm",
        ".pdf",
        ".docx",  # Note: .doc (older format) is NOT supported
        ".pptx",  # Note: .ppt (older format) is NOT supported
        ".xlsx",  # Note: .xls (older format) is NOT supported
        ".md",
        ".csv",
        # Note: .txt and .rtf are NOT supported by docling
    }

    for file_path in source_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.casefold() in supported_extensions:
            yield file_path


def generate_content_path(source_file: Path, source_root: Path) -> Path:
    """Generate the content directory path for a source file.

    Args:
        source_file: The source file to convert
        source_root: The root directory being processed

    Returns:
        Path: The path where the JSON file should be stored
    """
    # Get relative path from source root
    relative_path = source_file.relative_to(source_root)

    # Create path in content directory
    content_path = Path("content") / relative_path

    # Change extension to .json
    return content_path.with_suffix(".json")


def ingest_file(
    source_file: Path, content_path: Path, converter: DocumentConverter
) -> bool:
    """Convert a single file to docling JSON format.

    Args:
        source_file: Path to the source file to convert
        content_path: Path where the JSON file should be stored
        converter: DocumentConverter instance to use for conversion

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(
            "Converting file", source=str(source_file), target=str(content_path)
        )

        # Create the output directory
        content_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert the document
        result = converter.convert(source_file, raises_on_error=True)
        document = result.document

        # Save as JSON
        document.save_as_json(content_path)

        logger.info(
            "Successfully converted file",
            source=str(source_file),
            target=str(content_path),
        )
        return True

    except Exception as e:
        logger.error("Failed to convert file", source=str(source_file), error=str(e))
        return False


def ingest(source_path: str, dry_run: bool = False, force: bool = False) -> bool:
    """Ingest all files from a source path into the content directory.

    Args:
        source_path: Path to search for files to ingest
        dry_run: If True, show what would be processed without doing it
        force: If True, force reprocessing even if files are up-to-date

    Returns:
        bool: True if successful, False if any errors occurred
    """
    source_root = Path(source_path).resolve()

    if not source_root.exists():
        raise Docs2DBException(f"Source path does not exist: {source_path}")

    logger.info("Starting ingestion", source_path=str(source_root), dry_run=dry_run)

    file_count = sum(1 for _ in find_ingestible_files(source_root))
    if file_count == 0:
        logger.warning("No ingestible files found", source_path=str(source_root))
        return True

    logger.info("Found files to process", count=file_count)

    if dry_run:
        logger.info("Dry run mode - would process:")
        for source_file in find_ingestible_files(source_root):
            content_path = generate_content_path(source_file, source_root)
            logger.info(
                source_file.name, source=str(source_file), target=str(content_path)
            )
        return True

    converter = DocumentConverter()
    success_count = 0
    error_count = 0

    for source_file in find_ingestible_files(source_root):
        content_path = generate_content_path(source_file, source_root)

        # Skip if file is up-to-date (unless force is True)
        if not force and not is_ingestion_stale(content_path, source_file):
            logger.info("Skipping up-to-date file", source=str(source_file))
            success_count += 1
            continue

        if ingest_file(source_file, content_path, converter):
            success_count += 1
        else:
            error_count += 1

    logger.info(
        "Ingestion completed",
        total_files=file_count,
        successful=success_count,
        errors=error_count,
    )

    return error_count == 0
