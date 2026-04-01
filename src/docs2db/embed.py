"""RAG embedding for Docs2DB content."""

import os
import signal
import time
from pathlib import Path
from typing import Any

import psutil
import structlog

from docs2db.config import settings
from docs2db.embeddings import Embedding, get_optimal_device
from docs2db.multiproc import BatchProcessor, setup_worker_logging

logger = structlog.get_logger(__name__)


def chunks_files(content_dir: str, pattern: str) -> list[Path]:
    """Return sorted list of chunks files.

    Args:
        content_dir: Path to content directory
        pattern: Glob pattern for file matching

    Returns:
        list[Path]: Sorted list of Path objects for chunks files
    """
    content_path = Path(content_dir)
    if not content_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {content_dir}")

    # Collect all matching files and sort them for deterministic processing order
    return sorted(content_path.glob(pattern))


def generate_embeddings_batch(
    chunks_files: list[str],
    model: str,
    force: bool = False,
) -> dict[str, Any]:
    """Worker function for generating embeddings by the batch.

    Args:
        chunks_files: List of chunks file paths (as strings) to process in this batch
        model: Model identifier to use
        force: If True, reprocess files even if embeddings are up-to-date

    Returns:
        dict[str, Any]: Processing results containing:
            - successes: Number of files processed successfully
            - errors: Number of files that failed processing
            - error_data: List of error details for failed files
            - last_file: Path of the last file processed (for debugging)
            - memory: Worker process memory usage in MB
            - worker_logs: Collected log messages from this worker
    """
    successes = 0
    errors = 0
    error_data = []
    last_file = ""

    # Ignore SIGINT in worker processes - let main process handle interrupts.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    log_collector = setup_worker_logging(__name__)

    embedding = Embedding.from_name(model)

    for file in chunks_files:
        try:
            last_file = file
            embedding.generate_embedding(file, force)
            successes += 1
        except Exception as e:
            errors += 1
            error_data.append({"file": file, "error": str(e)})
            logger.error(f"Failed to process {file}: {e}")

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


def generate_embeddings(
    content_dir: str | None = None,
    model: str | None = None,
    pattern: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    max_workers: int | None = None,
) -> bool:
    """Generate embedding files using multiprocessing, with a progress bar.

    Args:
        content_dir: Path to content directory (defaults to settings.content_base_dir).
        model: Name of the embedding model to use (defaults to settings.embedding_model).
        pattern: Directory pattern to match (defaults to "**").
                 Can be an exact path or use wildcards. Automatically appends '/chunks.json'.
                 Examples: 'external/**' (all), 'docs/subdir' (exact), '**/api' (pattern)
        force: Force processing even if output already exists.
        dry_run: Show what would be processed without doing it.
        max_workers: Maximum worker processes. If None, auto-detects optimal count:
                     - GPU systems: 2 workers (parallel processing)
                     - CPU-only systems: 1 worker (avoids PyTorch fork deadlocks)

    Returns:
        bool: True if successful, False if any errors occurred.
    """

    if content_dir is None:
        content_dir = settings.content_base_dir
    if model is None:
        model = settings.embedding_model
    if pattern is None:
        pattern = "**"

    # Append /chunks.json to the pattern
    # Works for both exact directory paths and glob patterns:
    # - "dir/subdir" -> "dir/subdir/chunks.json" (exact file)
    # - "dir/**" -> "dir/**/chunks.json" (glob pattern)
    pattern = f"{pattern}/chunks.json"

    start = time.time()

    embedding = Embedding.from_name(model)
    embedding.ensure_available()

    # Detect device and set appropriate worker count
    device = get_optimal_device()
    is_cpu_only = device == "cpu"

    # Determine default workers based on device
    if max_workers is None:
        default_workers = 1 if is_cpu_only else 2
        max_workers = default_workers
    elif max_workers > 1 and is_cpu_only:
        # User explicitly set >1 workers on CPU-only system
        logger.warning(
            f"CPU-only system detected with {max_workers} workers. "
            "Multiprocessing with PyTorch on CPU may cause deadlocks or slow performance. "
            "Consider using --workers 1 for stability."
        )

    logger.info(
        "\nEmbedding configuration:\n"
        f"  Model     : {embedding.model}\n"
        f"  Dimensions: {embedding.dimensions}\n"
        f"  Device    : {embedding.device}"
    )

    chunks_list = chunks_files(content_dir, pattern)

    if len(chunks_list) == 0:
        logger.warning(f"No chunks files found matching pattern: {pattern}")
        return True

    if dry_run:
        logger.info("DRY RUN - would process:")
        for file in chunks_list:
            logger.info(f"  {file}")
        logger.info(f"DRY RUN complete - found {len(chunks_list)} chunks files")
        return True

    embedder = BatchProcessor(
        worker_function=generate_embeddings_batch,
        worker_args=(model, force),
        progress_message="Embedding files...",
        batch_size=8,
        mem_threshold_mb=1800,
        max_workers=max_workers,
    )
    embedded, errors = embedder.process_files(chunks_list)
    end = time.time()

    if errors > 0:
        logger.error(f"Embedding generation completed with {errors} errors")

    logger.info(f"{embedded} files embedded in {end - start:.2f} seconds")
    return errors == 0
