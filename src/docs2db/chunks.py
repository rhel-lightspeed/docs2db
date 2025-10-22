"""RAG chunking for Docs2DB content."""

import json
import os
import time
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any, Iterator

import httpx
import psutil
import structlog
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

from docs2db.const import (
    CHUNKING_CONFIG,
    CHUNKING_SCHEMA_VERSION,
    DATABASE_SCHEMA_VERSION,
)
from docs2db.multiproc import BatchProcessor, setup_worker_logging
from docs2db.utils import ensure_model_available, hash_file

logger = structlog.get_logger(__name__)


CURRENT_METADATA = {
    "max_tokens": CHUNKING_CONFIG["max_tokens"],
    "merge_peers": CHUNKING_CONFIG["merge_peers"],
    "tokenizer_model": CHUNKING_CONFIG["tokenizer_model"],
}


def is_chunks_stale(chunks_file: Path, source_file: Path) -> bool:
    """Check if chunks file is stale compared to source.

    Args:
        chunks_file: Path to the chunks file to check
        source_file: Path to the source file to compare against

    Returns:
        bool: True if chunks file needs regeneration, False if current
    """
    if not chunks_file.exists():
        return True

    try:
        with open(chunks_file) as f:
            metadata = json.load(f)["metadata"]

        if hash_file(source_file) != metadata["source_hash"]:
            return True

        stored_params = metadata["processing"]["parameters"]
        if stored_params != CURRENT_METADATA:
            return True

        return False

    except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError):
        return True


def source_files(
    content_dir: str,
    pattern: str = "**/*.json",
) -> tuple[int, Iterator[Path]]:
    """Return source files, ignoring chunk and embed files.

    Args:
        content_dir: Path to the content directory to search
        pattern: Glob pattern for file matching (default: "**/*.json")

    Returns:
        tuple[int, Iterator[Path]]: Count of matching files and iterator of Path objects
    """

    content = Path(content_dir)
    if not content.exists():
        raise FileNotFoundError(f"Content directory does not exist: {content_dir}")

    # Ignore chunk and embed files.
    def source_files_iter():
        return (
            f
            for f in content.glob(pattern)
            if not f.name.endswith(".chunks.json")
            and not f.name.endswith(".gran.json")
            and f.suffix == ".json"
        )

    count = sum(1 for _ in source_files_iter())
    return count, source_files_iter()


@cache
def get_tokenizer():
    hf_tokenizer = AutoTokenizer.from_pretrained(
        CHUNKING_CONFIG["tokenizer_model"],
        local_files_only=True,
        use_fast=True,
    )
    return HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=CHUNKING_CONFIG["max_tokens"],
    )


class LLMSession:
    """Persistent LLM session for reusing document context across chunks."""

    def __init__(self, doc_text: str, model: str = "qwen2.5:7b-instruct"):
        self.model = model
        self.client = httpx.Client(timeout=60.0)
        # Initialize conversation with the document
        self.messages = [
            {
                "role": "system",
                "content": "You are an expert at providing concise context for text chunks within documents.",
            },
            {
                "role": "user",
                "content": f"I will give you a document, then ask you to provide context for specific chunks from it.\n\n<document>\n{doc_text}\n</document>",
            },
            {
                "role": "assistant",
                "content": "I have read the document. Please provide the chunks you'd like me to contextualize.",
            },
        ]

    def get_chunk_context(self, chunk_text: str) -> str:
        """Get context for a chunk using the cached document."""
        # Add the chunk query to conversation
        chunk_prompt = f"""Here is a chunk from the document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Create messages for this request (includes conversation history)
        request_messages = self.messages + [{"role": "user", "content": chunk_prompt}]

        response = self.client.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": self.model,
                "messages": request_messages,
                "stream": False,
                "temperature": 0.3,
                "max_tokens": 200,
            },
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def close(self):
        """Close the HTTP client."""
        self.client.close()


def generate_chunks_for_document(
    source_str: str, content_dir: Path, force: bool
) -> Path:
    """Generate chunks for a document."""
    source_file = Path(source_str)
    chunks_file = source_file.with_suffix(".chunks.json")

    if not force and not is_chunks_stale(chunks_file, source_file):
        return chunks_file

    dl_doc = DoclingDocument.model_validate_json(
        json_data=source_file.read_text().encode("utf-8")
    )

    # Extract full document text for contextual generation
    doc_text = dl_doc.export_to_markdown()

    # Create persistent LLM session for this document (enables KV cache reuse)
    llm_session = LLMSession(doc_text)

    # Create chunker and chunk document
    chunker = HybridChunker(tokenizer=get_tokenizer(), merge_peers=True)
    chunks_data = []

    try:
        for chunk_idx, chunk in enumerate(chunker.chunk(dl_doc=dl_doc)):
            # Get enriched text from docling's contextualization (adds heading context)
            enriched_text = chunker.contextualize(chunk=chunk)
            original_text = enriched_text.replace("\xa0", " ")

            # Generate chunk-specific context using persistent LLM session (reuses cached document)
            chunk_context = llm_session.get_chunk_context(original_text)

            # Build contextual text: prepend context to original text
            contextual_text = f"{chunk_context}\n\n{original_text}"

            chunk_data = {
                "text": original_text,  # Original chunk text
                "contextual_text": contextual_text,  # Context-enhanced text for embeddings and BM25
                "metadata": chunk.meta.model_dump(),
            }
            chunks_data.append(chunk_data)
    finally:
        # Always close the session to free resources
        llm_session.close()

    if not chunks_data:
        logger.warning(f"No chunks found in {source_file}")

    output_data = {
        "metadata": {
            "source_file": str(source_file.relative_to(content_dir)),
            "source_hash": hash_file(source_file),
            "database_schema_version": DATABASE_SCHEMA_VERSION,
            "chunking_schema_version": CHUNKING_SCHEMA_VERSION,
            "processing": {
                "chunker": CHUNKING_CONFIG["chunker_class"],
                "parameters": {
                    "max_tokens": CHUNKING_CONFIG["max_tokens"],
                    "merge_peers": CHUNKING_CONFIG["merge_peers"],
                    "tokenizer_model": CHUNKING_CONFIG["tokenizer_model"],
                },
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(chunks_data),
        },
        "chunks": chunks_data,
    }

    with open(chunks_file, "w") as f:
        json.dump(output_data, f, indent=2)
    return chunks_file


def generate_chunks_batch(
    source_files: list[str],
    content_dir: str,
    force: bool = False,
) -> dict[str, Any]:
    """Worker function for generating chunks files by the batch.

    Args:
        source_files: List of file paths (as strings) to process in this batch
        force: If True, reprocess files even if chunks are up-to-date

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

    # Set up worker logging to capture logs for replay in main process.
    # Note: This alters logging globally, and process_chunk_batch is meant to
    # only be called as a worker process by multiproc.process_files. The
    # content of the logs is returned in the result of this function and the
    # main process re-logs it to the console.
    log_collector = setup_worker_logging(__name__)

    # Suppress transformers warnings about sequence length in worker processes.
    # (This logging was ruining the screen-stable Rich progress bar, also the
    # warning is a known "false alarm" per google search.)
    transformers_logging.set_verbosity_error()

    # Suppress httpx INFO-level logging (HTTP request logs)
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)

    content_dir_path = Path(content_dir)
    for file in source_files:
        try:
            last_file = file
            generate_chunks_for_document(file, content_dir_path, force=force)
            successes += 1
        except Exception as e:
            errors += 1
            error_data.append({"file": file, "error": str(e)})
            logger.error(f"Failed to process {file}: {e}")

    # Report worker memory footprint.
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


def generate_chunks(
    content_dir: str, pattern: str, force: bool = False, dry_run: bool = False
) -> bool:
    """Generate .chunks.json files from source files using multiprocessing.

    Args:
        content_dir (str): Path to content directory.
        pattern (str): File pattern to process.
        force (bool): Force processing even if output already exists.
        dry_run (bool): Show what would be processed without doing it.

    Returns:
        bool: True if successful, False if any errors occurred.
    """
    start = time.time()

    logger.info(
        "\nChunking configuration:\n"
        f"  Model      : {CHUNKING_CONFIG['tokenizer_model']}\n"
        f"  Max tokens : {CHUNKING_CONFIG['max_tokens']}\n"
        f"  Merge peers: {CHUNKING_CONFIG['merge_peers']}\n"
        f"  Chunker    : {CHUNKING_CONFIG['chunker_class']}"
    )

    ensure_model_available(model_id=CHUNKING_CONFIG["tokenizer_model"])

    count, source_iter = source_files(content_dir, pattern)

    if count == 0:
        logger.warning(f"No source files found matching pattern: {pattern}")
        return True

    if dry_run:
        logger.info("DRY RUN - would process:")
        for file in source_iter:
            logger.info(f"  {file}")
        logger.info(f"DRY RUN complete - found {count} source files")
        return True

    chunker = BatchProcessor(
        worker_function=generate_chunks_batch,
        worker_args=(content_dir, force),
        progress_message="Chunking files...",
        batch_size=1,  # Process 1 file per batch for better log visibility
        mem_threshold_mb=2000,
        max_workers=1,  # Use 1 worker for sequential processing and clear logs
    )
    chunked, errors = chunker.process_files(source_iter, count)
    end = time.time()

    if errors > 0:
        logger.error(f"Chunking completed with {errors} errors")
        logger.info(f"{chunked} files chunked in {end - start:.2f} seconds")
        return False

    logger.info(f"{chunked} files chunked in {end - start:.2f} seconds")
    return True
