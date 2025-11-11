"""Ingest files using docling to create JSON documents."""

import json
import logging
import mimetypes
import os
import time
from datetime import datetime, timezone
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import psutil
import structlog
from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream

from docs2db.config import settings
from docs2db.const import METADATA_SCHEMA_VERSION
from docs2db.exceptions import Docs2DBException
from docs2db.multiproc import BatchProcessor, setup_worker_logging
from docs2db.utils import hash_bytes, hash_file

logger = structlog.get_logger(__name__)


def _suppress_docling_logging() -> None:
    """Suppress verbose logging from docling library.

    Docling uses Python's standard logging and can be very verbose.
    This sets docling loggers to WARNING level to reduce noise.
    """
    # Suppress all docling-related verbose info messages
    # Set the root docling logger to WARNING, which propagates to all child loggers
    logging.getLogger("docling").setLevel(logging.WARNING)
    logging.getLogger("docling.document_converter").setLevel(logging.WARNING)
    logging.getLogger("docling.pipeline").setLevel(logging.WARNING)
    logging.getLogger("docling.datamodel").setLevel(logging.WARNING)
    logging.getLogger("docling_core").setLevel(logging.WARNING)

    # Suppress deepsearch (used by docling for parsing)
    logging.getLogger("deepsearch").setLevel(logging.WARNING)
    logging.getLogger("deepsearch_glm").setLevel(logging.WARNING)


# Module-level singleton converter for efficiency
_converter: DocumentConverter | None = None

# README content for the content directory
_CONTENT_DIR_README = """# About this folder

This folder was created by `docs2db` to store processed content.

Learn more: https://github.com/rhel-lightspeed/docs2db

## What's inside

- **`.json` files** - Documents converted to Docling JSON format
- **`.chunks.json` files** - Chunked documents ready for embedding
- **`.gran.json` files** - Granite embedding vectors
- **`.meta.json` files** - Metadata about source documents

## Important

**Regenerating this content is expensive!** Processing includes:
- Document conversion (Docling)
- AI-powered contextual chunking (requires LLM)
- Embedding generation (requires embedding model)

### Recommendations

1. **Keep this folder** - Save it when updating your content pipeline
2. **Commit to version control** - Track changes and share across team members and CI pipelines
3. **Reuse it** - `docs2db` automatically skips unchanged files, making updates fast

### Structure

The directory structure mirrors your source content, with each file converted to
the Docling JSON format plus associated processing artifacts.
"""


def _get_converter() -> DocumentConverter:
    """Get or create the DocumentConverter singleton.

    Returns:
        DocumentConverter: The shared converter instance
    """
    global _converter
    if _converter is None:
        _suppress_docling_logging()
        _converter = DocumentConverter()
    return _converter


def ensure_content_dir_readme(content_base_dir: str | Path | None = None) -> None:
    """Ensure the content directory has a README explaining its purpose.

    Creates a README.md file in the content directory if it doesn't exist.
    The README explains what the folder contains and recommends keeping it.

    Args:
        content_base_dir: Base directory for content. If None, uses settings.content_base_dir
    """
    if content_base_dir is None:
        content_base_dir = settings.content_base_dir

    content_dir = Path(content_base_dir)
    readme_path = content_dir / "README.md"

    # Only create if it doesn't exist (don't overwrite user modifications)
    if not readme_path.exists():
        try:
            content_dir.mkdir(parents=True, exist_ok=True)
            readme_path.write_text(_CONTENT_DIR_README.strip() + "\n")
            logger.debug(f"Created content directory README at {readme_path}")
        except OSError as e:
            logger.warning(f"Could not create content directory README: {e}")


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


def ingest_batch(source_files: list[str], source_root: str, force: bool) -> dict:
    """Worker function to ingest a batch of files.

    Args:
        source_files: List of source file paths (as strings)
        source_root: Root directory path (as string)
        force: Whether to force reprocessing

    Returns:
        dict: Results with successes, errors, error_data, last_file, memory, worker_logs
    """
    log_collector = setup_worker_logging(__name__)

    successes = 0
    errors = 0
    error_data = []
    last_file = ""

    source_root_path = Path(source_root)

    for file_str in source_files:
        try:
            last_file = file_str
            source_file = Path(file_str)
            content_path = generate_content_path(source_file, source_root_path)

            # Skip if file is up-to-date (unless force is True)
            if not force and not is_ingestion_stale(content_path, source_file):
                logger.info(f"Skipping up-to-date file: {source_file.name}")
                successes += 1
                continue

            if ingest_file(source_file, content_path):
                successes += 1
            else:
                errors += 1
                error_data.append({"file": file_str, "error": "Ingestion failed"})

        except Exception as e:
            errors += 1
            error_data.append({"file": file_str, "error": str(e)})
            logger.error(f"Failed to process {file_str}: {e}")

    # Report worker memory footprint
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 / 1024  # MB

    return {
        "successes": successes,
        "errors": errors,
        "error_data": error_data,
        "last_file": last_file,
        "memory": round(memory, 1),
        "worker_logs": log_collector.logs,
    }


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

    Creates a subdirectory path named after the source file (without extension).
    The actual source.json filename is added by ingest_file().

    Args:
        source_file: The source file to convert
        source_root: The root directory being processed

    Returns:
        Path: The directory path where the document should be stored
    """
    # Get relative path from source root
    relative_path = source_file.relative_to(source_root)

    # Create directory name from filename without extension
    dir_name = relative_path.stem

    # Create path: content_dir / parent_dirs / doc_dir
    content_path = Path(settings.content_base_dir) / relative_path.parent / dir_name

    return content_path


def document_needs_update(
    content_path: Path,
    source_file: Path | None = None,
    content: str | bytes | None = None,
    source_timestamp: str | None = None,
) -> bool:
    """Check if a document needs updating.

    Returns True if:
    - Document doesn't exist
    - Any provided comparison shows staleness:
      * source_timestamp is newer than stored timestamp
      * source_file hash differs from stored hash
      * content hash differs from stored hash

    Returns False if:
    - Document exists AND
    - All provided comparisons show it's current

    If all comparison parameters are None, this is just an existence check.

    Args:
        content_path: Directory path where the document is stored
        source_file: Optional source file to compare hash against
        content: Optional content to compare hash against
        source_timestamp: Optional timestamp to compare against stored timestamp

    Returns:
        bool: True if document needs updating, False otherwise

    Examples:
        # Just check existence
        if document_needs_update(path):
            ingest_file(...)

        # Check existence + timestamp staleness
        if document_needs_update(path, source_timestamp=article.modified):
            ingest_from_content(...)

        # Check existence + file hash staleness
        if document_needs_update(path, source_file=local_file):
            ingest_file(...)
    """
    # Check if document exists
    json_path = content_path / "source.json"
    if not json_path.exists():
        return True  # Needs update (doesn't exist)

    # If no comparison parameters provided, document exists so no update needed
    if source_file is None and content is None and source_timestamp is None:
        return False

    # Load metadata to get stored hash and timestamp
    meta_path = content_path / "meta.json"
    if not meta_path.exists():
        # Document exists but no metadata - should update
        return True

    try:
        with meta_path.open("r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Can't read metadata - assume needs update
        return True

    # Check timestamp if provided
    if source_timestamp is not None:
        source = metadata.get("source", {})
        stored_timestamp = source.get("modified")

        if stored_timestamp is None:
            # Have new timestamp but no stored timestamp - needs update
            return True
        if source_timestamp != stored_timestamp:
            # Timestamp differs - needs update
            return True

    # Extract stored values from metadata structure
    processing = metadata.get("processing", {})
    stored_hash = processing.get("source_hash")

    # Check source file hash if provided
    if source_file is not None:
        if stored_hash is None:
            # Have source file but no stored hash - needs update
            return True
        current_hash = hash_file(source_file)
        if current_hash != stored_hash:
            # Hash differs - needs update
            return True

    # Check content hash if provided
    if content is not None:
        if stored_hash is None:
            # Have content but no stored hash - needs update
            return True
        current_hash = hash_bytes(
            content if isinstance(content, bytes) else content.encode("utf-8")
        )
        if current_hash != stored_hash:
            # Hash differs - needs update
            return True

    # All checks passed - document is current
    return False


def ingest_file(
    source_file: Path,
    content_path: Path,
    source_metadata: dict[str, Any] | None = None,
) -> bool:
    """Convert a single file to docling JSON format.

    Args:
        source_file: Path to the source file to convert
        content_path: Directory path where the document should be stored (source.json will be created inside)
        source_metadata: Optional metadata about the source

    Returns:
        bool: True if successful, False otherwise
    """
    converter = _get_converter()

    try:
        # Create the full path to source.json in the provided directory
        json_path = content_path / "source.json"

        logger.debug("Converting file", source=str(source_file), target=str(json_path))

        # Create the output directory and ensure README exists
        json_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_content_dir_readme(json_path.parts[0] if json_path.parts else None)

        result = converter.convert(source_file)
        result.document.save_as_json(json_path)

        logger.debug(
            "Successfully converted file",
            source=str(source_file),
            target=str(json_path),
        )

        generate_metadata(
            source_hash=hash_file(source_file),
            content_path=json_path,
            source_file=source_file,
            source_metadata=source_metadata,
        )

        return True

    except Exception as e:
        logger.error("Failed to convert file", source=str(source_file), error=str(e))
        return False


def ingest_from_content(
    content: str | bytes,
    content_path: Path,
    stream_name: str,
    source_metadata: dict[str, Any] | None = None,
    content_encoding: str = "utf-8",
) -> bool:
    """Convert in-memory content to Docling JSON and generate metadata.

    For external tools to ingest content directly without saving
    intermediate files. Converts content to Docling JSON and
    generates metadata.

    Args:
        content: The content to convert (HTML, markdown, etc).
        content_path: Directory path where the document should be stored (source.json will be created inside).
        stream_name: Stream name with extension for docling to detect format (e.g., "doc.html", "article.md").
        source_metadata: Source metadata (URL, etag, license, etc).
        content_encoding: Defaults to "utf-8".

    Returns:
        bool: True if successful, False otherwise

    """
    try:
        # Create the full path to source.json in the provided directory
        json_path = content_path / "source.json"

        logger.debug("Converting content", target=str(json_path), stream=stream_name)

        # Convert string content to bytes if needed
        if isinstance(content, str):
            content = content.encode(content_encoding)

        # Create document stream and convert
        json_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_content_dir_readme(json_path.parts[0] if json_path.parts else None)

        stream = DocumentStream(name=stream_name, stream=BytesIO(content))

        converter = _get_converter()
        result = converter.convert(stream)
        document = result.document
        document.save_as_json(json_path)

        logger.debug("Successfully converted content", target=str(json_path))

        source_hash = hash_bytes(content)

        generate_metadata(
            source_hash=source_hash,
            content_path=json_path,
            source_file=None,  # No source file for in-memory content
            source_metadata=source_metadata,
        )

        return True

    except Exception as e:
        logger.error(
            "Failed to convert content", target=str(content_path), error=str(e)
        )
        return False


def generate_metadata(
    source_hash: str,
    content_path: Path,
    source_file: Path | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> None:
    """Internal: Generate and save metadata for an ingested document.

    Args:
        source_hash: xxhash64 of source (format: "xxh64:hexdigest")
        content_path: Path to the generated docling JSON file
        source_file: Optional path to original source file
        source_metadata: Optional user-supplied metadata
    """
    metadata: dict[str, Any] = {
        "metadata_version": METADATA_SCHEMA_VERSION,
        "processing": {
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "source_hash": source_hash,
            "docling_version": version("docling"),
        },
    }

    if source_file and source_file.exists():
        # Try to compute relative path, but handle external callers
        # that may use different content_base_dir values
        try:
            content_base = Path(settings.content_base_dir)
            relative_path = content_path.relative_to(content_base)
        except ValueError:
            # If content_path is not relative to settings.content_base_dir,
            # extract the content base from the path itself (use first directory)
            if content_path.is_absolute():
                relative_path = content_path
            else:
                # Path like "content/docs/...", use everything after first component
                parts = content_path.parts
                relative_path = Path(*parts[1:]) if len(parts) > 1 else content_path

        stat = source_file.stat()
        filesystem_meta = {
            "original_path": str(relative_path.with_suffix(source_file.suffix)),
            "size_bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }
        mime, _ = mimetypes.guess_type(source_file)
        if mime:
            filesystem_meta["detected_mime"] = mime

        metadata["filesystem"] = filesystem_meta

    # Auto-detect content info (from Docling document)
    try:
        with open(content_path) as f:
            docling_doc = json.load(f)

        content_meta = {}
        if docling_doc.get("name"):
            content_meta["title"] = docling_doc["name"]
        if docling_doc.get("metadata", {}).get("language"):
            content_meta["language"] = docling_doc["metadata"]["language"]

        if content_meta:
            metadata["content"] = content_meta
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            f"Could not read docling document for metadata from {content_path}: {e}"
        )

    if source_metadata:
        metadata["source"] = source_metadata

    # Save metadata (sparse - only non-empty sections)
    # content_path is .../doc_dir/source.json, so meta goes to .../doc_dir/meta.json
    meta_path = content_path.parent / "meta.json"
    try:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Metadata saved to {meta_path}")
    except OSError as e:
        logger.warning(f"Failed to save metadata to {meta_path}: {e}")


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

    # If source is a file, use its parent as the root to maintain directory structure
    if source_root.is_file():
        source_root = source_root.parent

    logger.info("Starting ingestion", source_path=str(source_root), dry_run=dry_run)
    start = time.time()

    # Collect and sort all source files
    source_files = sorted(find_ingestible_files(source_root))
    if len(source_files) == 0:
        logger.warning("No ingestible files found", source_path=str(source_root))
        return True

    logger.info("Found files to process", count=len(source_files))

    # Ensure content directory README exists
    ensure_content_dir_readme()

    if dry_run:
        logger.info("Dry run mode - would process:")
        for source_file in source_files:
            content_path = generate_content_path(source_file, source_root)
            logger.info(
                source_file.name, source=str(source_file), target=str(content_path)
            )
        return True

    # Use multiprocessing for ingestion
    processor = BatchProcessor(
        worker_function=ingest_batch,
        worker_args=(str(source_root), force),
        progress_message="Ingesting files...",
        batch_size=3,  # Smaller batches since docling can be memory-intensive
        mem_threshold_mb=1500,  # Lower threshold for docling processes
    )

    processed, errors = processor.process_files(source_files)
    end = time.time()

    if errors > 0:
        logger.error(f"Ingestion completed with {errors} errors")

    logger.info(f"{processed} files ingested in {end - start:.2f} seconds")
    return errors == 0
