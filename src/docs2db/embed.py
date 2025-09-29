"""RAG embedding for Docs2DB content."""

import os
import signal
import time
from pathlib import Path
from typing import Any, Iterator

import psutil
import structlog

from docs2db.embeddings import Embedding
from docs2db.multiproc import BatchProcessor, setup_worker_logging

logger = structlog.get_logger(__name__)


def chunks_files(content_dir: str, pattern: str) -> tuple[int, Iterator[Path]]:
    """Return chunks files, filtering to only .chunks.json files."""
    content_path = Path(content_dir)
    if not content_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {content_dir}")

    def chunks_files():
        return (
            f for f in content_path.glob(pattern) if f.name.endswith(".chunks.json")
        )

    count = sum(1 for _ in chunks_files())
    return count, chunks_files()


def generate_embeddings_batch(
    chunks_files: list[str],
    model_name: str,
    force: bool = False,
) -> dict[str, Any]:
    """Worker function for generating embeddings by the batch.

    Args:
        chunks_files: List of chunks file paths (as strings) to process in this batch
        model_name: Name of the embedding model to use
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

    embedding = Embedding.from_name(model_name)

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
    content_dir: str,
    model_name: str,
    pattern: str,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Generate embedding files using multiprocessing, with a progress bar.

    Args:
        content_dir (str): Path to content directory.
        model_name (str): Name of the embedding model to use.
        pattern (str): File pattern for chunks files.
        force (bool): Force processing even if output already exists.
        dry_run (bool): Show what would be processed without doing it.

    Returns:
        bool: True if successful, False if any errors occurred.
    """
    start = time.time()

    embedding = Embedding.from_name(model_name)
    embedding.ensure_available()

    logger.info(
        "\nEmbedding configuration:\n"
        f"  Model     : {embedding.model_id}\n"
        f"  Dimensions: {embedding.dimensions}\n"
        f"  Device    : {embedding.device}"
    )

    count, chunks_iter = chunks_files(content_dir, pattern)

    if count == 0:
        logger.warning(f"No chunks files found matching pattern: {pattern}")
        return True

    if dry_run:
        logger.info("DRY RUN - would process:")
        for file in chunks_iter:
            logger.info(f"  {file}")
        logger.info(f"DRY RUN complete - found {count} chunks files")
        return True

    embedder = BatchProcessor(
        worker_function=generate_embeddings_batch,
        worker_args=(model_name, force),
        progress_message="Embedding files...",
        batch_size=8,
        mem_threshold_mb=1800,
        max_workers=2,
    )
    embedded, errors = embedder.process_files(chunks_iter, count)
    end = time.time()

    if errors > 0:
        logger.error(f"Embedding generation completed with {errors} errors")
        logger.info(f"{embedded} files embedded in {end - start:.2f} seconds")
        return False

    logger.info(f"{embedded} files embedded in {end - start:.2f} seconds")
    return True
