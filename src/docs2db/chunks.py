"""RAG chunking for Docs2DB content."""

import json
import os
import time
from abc import ABC, abstractmethod
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

from docs2db.config import settings
from docs2db.const import (
    CHUNKING_CONFIG,
    CHUNKING_SCHEMA_VERSION,
    DATABASE_SCHEMA_VERSION,
)
from docs2db.multiproc import BatchProcessor, setup_worker_logging
from docs2db.utils import ensure_model_available, hash_file

logger = structlog.get_logger(__name__)


# Model context limits (in tokens)
# Used to detect when documents are too large and need map-reduce summarization
MODEL_CONTEXT_LIMITS = {
    # WatsonX models
    "ibm/granite-3-8b-instruct": 131072,
    "ibm/granite-3-2-8b-instruct": 8192,
    "ibm/granite-3-2b-instruct": 8192,
    "ibm/granite-3-3-8b-instruct": 8192,
    "meta-llama/llama-3-3-70b-instruct": 8192,
    "meta-llama/llama-3-1-70b-gptq": 131072,
    "meta-llama/llama-3-1-8b": 131072,
    "meta-llama/llama-3-405b-instruct": 131072,
    # Ollama models (approximate)
    "qwen2.5:7b-instruct": 32768,
    "qwen2.5:3b-instruct": 32768,
    "qwen2.5:1.5b-instruct": 32768,
    "llama3.2:3b": 131072,
    "llama3.2:1b": 131072,
    "gemma2:2b": 8192,
    # OpenAI models
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 16385,
}

# Safety margin: use 70% of context limit to account for:
# - System messages and prompts
# - Response tokens (200)
# - Token estimation inaccuracies
CONTEXT_SAFETY_MARGIN = 0.7

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

    # Ignore chunk, embed, and metadata files - only process Docling JSON files
    def source_files_iter():
        return (
            f
            for f in content.glob(pattern)
            if f.suffix == ".json"
            and not f.name.endswith(".chunks.json")
            and not f.name.endswith(".gran.json")
            and not f.name.endswith(".meta.json")
        )

    count = sum(1 for _ in source_files_iter())
    return count, source_files_iter()


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a conservative approximation based on character count.
    This accounts for diverse content types (prose, code, data, spreadsheets).

    Formula: chars / 3.0
    - Regular English prose: ~4-5 chars/token (we're conservative at 3)
    - Code/data/numbers: ~2-3 chars/token (we handle this well)
    - Includes safety margin for tokenization variance

    Args:
        text: Text to estimate tokens for

    Returns:
        int: Estimated token count
    """
    char_count = len(text)
    # Use character-based estimation: 3 chars per token is conservative
    return int(char_count / 3.0)


def split_text_into_chunks(text: str, max_tokens: int) -> list[str]:
    """Split text into chunks that fit within token limit.

    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk

    Returns:
        list[str]: List of text chunks
    """
    # Use character-based splitting to match our token estimation
    max_chars = int(max_tokens * 3.0)  # Reverse the estimation (chars / 3 = tokens)

    chunks = []
    words = text.split()
    current_chunk = []
    current_chars = 0

    for word in words:
        word_len = len(word) + 1  # +1 for space

        if current_chars + word_len > max_chars and current_chunk:
            # Current chunk would exceed limit, save it and start new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_chars = word_len
        else:
            current_chunk.append(word)
            current_chars += word_len

    # Add remaining words
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


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


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_chunk_context(self, chunk_prompt: str) -> str:
        """Get context for a chunk from the LLM.

        Args:
            chunk_prompt: The prompt to send to the LLM

        Returns:
            str: The generated context
        """
        pass

    @abstractmethod
    def summarize_text(self, text: str) -> str:
        """Summarize a text chunk.

        Args:
            text: Text to summarize

        Returns:
            str: Summarized text
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up provider resources."""
        pass


class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI-compatible APIs (Ollama, OpenAI, etc)."""

    def __init__(self, base_url: str, model: str, messages: list[dict]):
        """Initialize OpenAI-compatible provider.

        Args:
            base_url: Base URL for the API endpoint
            model: Model identifier
            messages: Initial conversation messages (system + document context)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.messages = messages
        self.client = httpx.Client(timeout=60.0)

    def get_chunk_context(self, chunk_prompt: str) -> str:
        """Get context for a chunk using OpenAI-compatible API."""
        # Create messages for this request (includes conversation history)
        request_messages = self.messages + [{"role": "user", "content": chunk_prompt}]

        response = self.client.post(
            f"{self.base_url}/v1/chat/completions",
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

    def summarize_text(self, text: str) -> str:
        """Summarize text using OpenAI-compatible API."""
        prompt = f"""Please provide a concise summary of the following text, focusing on the key information and main topics:

{text}

Summary:"""

        # Log what we're about to send
        word_count = len(prompt.split())
        estimated_tokens = estimate_tokens(prompt)
        char_count = len(prompt)
        logger.info(
            f"Sending summarization request: {word_count} words, "
            f"{estimated_tokens} estimated tokens, {char_count} chars "
            f"(model: {self.model})"
        )

        response = self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.3,
                "max_tokens": 500,
            },
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def close(self):
        """Close the HTTP client."""
        self.client.close()


class WatsonXProvider(LLMProvider):
    """Provider for IBM WatsonX."""

    def __init__(
        self, api_key: str, project_id: str, url: str, model: str, doc_text: str
    ):
        """Initialize WatsonX provider.

        Args:
            api_key: WatsonX API key
            project_id: WatsonX project ID
            url: WatsonX API URL
            model: Model identifier
            doc_text: Full document text for context
        """
        try:
            from ibm_watsonx_ai import (  # type: ignore[import-untyped]
                APIClient,
                Credentials,
            )
            from ibm_watsonx_ai.foundation_models import (  # type: ignore[import-untyped]
                ModelInference,
            )
        except ImportError as e:
            raise ImportError(
                "IBM WatsonX AI SDK is required for WatsonX provider. "
                "Install it with: uv sync --group watsonx"
            ) from e

        self.model = model
        self.doc_text = doc_text

        # Initialize WatsonX API client
        credentials = Credentials(api_key=api_key, url=url)
        self.api_client = APIClient(credentials=credentials, project_id=project_id)

        # Create model inference instance
        self.model_inference = ModelInference(
            model_id=model,
            api_client=self.api_client,
        )

    def get_chunk_context(self, chunk_prompt: str) -> str:
        """Get context for a chunk using WatsonX SDK."""
        # WatsonX uses a chat format with messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert at providing concise context for text chunks within documents.",
            },
            {
                "role": "user",
                "content": f"I will give you a document, then ask you to provide context for a specific chunk from it.\n\n<document>\n{self.doc_text}\n</document>",
            },
            {
                "role": "assistant",
                "content": "I have read the document. Please provide the chunk you'd like me to contextualize.",
            },
            {"role": "user", "content": chunk_prompt},
        ]

        # Call WatsonX with chat messages
        params = {
            "temperature": 0.3,
            "max_tokens": 200,
        }

        response = self.model_inference.chat(messages=messages, params=params)

        # Extract content from response
        return response["choices"][0]["message"]["content"].strip()

    def summarize_text(self, text: str) -> str:
        """Summarize text using WatsonX SDK."""
        prompt_content = f"""Please provide a concise summary of the following text, focusing on the key information and main topics:

{text}

Summary:"""

        messages = [
            {
                "role": "user",
                "content": prompt_content,
            }
        ]

        # Log what we're about to send
        word_count = len(prompt_content.split())
        estimated_tokens = estimate_tokens(prompt_content)
        char_count = len(prompt_content)
        logger.info(
            f"Sending WatsonX summarization request: {word_count} words, "
            f"{estimated_tokens} estimated tokens, {char_count} chars "
            f"(model: {self.model})"
        )

        params = {
            "temperature": 0.3,
            "max_tokens": 500,
        }

        response = self.model_inference.chat(messages=messages, params=params)
        return response["choices"][0]["message"]["content"].strip()

    def close(self):
        """Clean up WatsonX resources."""
        # WatsonX APIClient doesn't require explicit cleanup
        pass


def map_reduce_summarize(
    provider: LLMProvider, text: str, max_tokens: int, model_name: str
) -> str:
    """Summarize large text using map-reduce approach.

    Args:
        provider: LLM provider to use for summarization
        text: Text to summarize
        max_tokens: Maximum tokens per chunk (already includes 70% safety margin)
        model_name: Model name for logging

    Returns:
        str: Summarized text that fits within context limit
    """
    logger.info(
        f"Document too large for model context window. "
        f"Starting map-reduce summarization (model: {model_name})"
    )

    # Reserve tokens for prompt overhead and response
    # Summarization prompt adds ~100 tokens, response uses 500 tokens
    PROMPT_OVERHEAD = 600
    chunk_size = max_tokens - PROMPT_OVERHEAD

    if chunk_size <= 0:
        raise ValueError(
            f"max_tokens ({max_tokens}) too small to allow for prompt overhead"
        )

    # Split text into chunks, accounting for prompt overhead
    chunks = split_text_into_chunks(text, chunk_size)
    logger.info(
        f"Split document into {len(chunks)} chunks for summarization (chunk_size: {chunk_size} tokens)"
    )

    # Map: Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks, 1):
        chunk_words = len(chunk.split())
        chunk_tokens = estimate_tokens(chunk)
        logger.info(
            f"Summarizing chunk {i}/{len(chunks)}: "
            f"{chunk_words} words, {chunk_tokens} estimated tokens"
        )
        summary = provider.summarize_text(chunk)
        summaries.append(summary)

    # Combine summaries
    combined = "\n\n".join(summaries)
    combined_tokens = estimate_tokens(combined)

    logger.info(f"Combined {len(summaries)} summaries into {combined_tokens} tokens")

    # Reduce: If combined summaries still too large, recursively summarize
    # Use the same chunk_size (not max_tokens) for consistency
    if combined_tokens > chunk_size:
        logger.info("Combined summaries still too large, applying recursive reduction")
        return map_reduce_summarize(provider, combined, max_tokens, model_name)

    return combined


class LLMSession:
    """Persistent LLM session for reusing document context across chunks."""

    def __init__(
        self,
        doc_text: str,
        model: str = "qwen2.5:7b-instruct",
        openai_url: str | None = None,
        watsonx_url: str | None = None,
        context_limit_override: int | None = None,
    ):
        """Initialize LLM session with appropriate provider.

        Args:
            doc_text: Full document text for context
            model: Model identifier
            openai_url: OpenAI-compatible API URL (mutually exclusive with watsonx_url)
            watsonx_url: WatsonX API URL (mutually exclusive with openai_url)
            context_limit_override: Override model context limit (in tokens)
        """
        self.doc_text = doc_text
        self.model = model
        self._was_summarized = False  # Track if document was summarized

        # Check if document needs summarization
        doc_words = len(doc_text.split())
        doc_tokens = estimate_tokens(doc_text)
        doc_chars = len(doc_text)

        # Use override if provided, otherwise use model's known limit
        if context_limit_override:
            model_limit = context_limit_override
        else:
            model_limit = MODEL_CONTEXT_LIMITS.get(
                model, 32768
            )  # Default to 32K if unknown
        usable_limit = int(model_limit * CONTEXT_SAFETY_MARGIN)

        logger.info(
            f"Document analysis: {doc_words} words, {doc_tokens} estimated tokens, {doc_chars} chars | "
            f"Model limit: {model_limit} tokens, Usable (70%): {usable_limit} tokens"
        )

        # Determine which provider to use and create it
        if watsonx_url:
            # Use WatsonX provider - get credentials from settings
            api_key = settings.watsonx_api_key
            project_id = settings.watsonx_project_id

            if not api_key or not project_id:
                raise ValueError(
                    "WATSONX_API_KEY and WATSONX_PROJECT_ID must be set (via env vars or .env file)"
                )

            # Create provider first (needed for summarization)
            temp_provider = WatsonXProvider(
                api_key=api_key,
                project_id=project_id,
                url=watsonx_url,
                model=model,
                doc_text="",  # Temporary, will update after summarization if needed
            )

            # Check if summarization is needed
            if doc_tokens > usable_limit:
                logger.info(
                    f"Document exceeds context limit ({doc_tokens} > {usable_limit}). "
                    f"Using map-reduce summarization."
                )
                doc_text = map_reduce_summarize(
                    temp_provider, doc_text, usable_limit, model
                )
                final_tokens = estimate_tokens(doc_text)
                logger.info(
                    f"Summarization complete. Reduced from {doc_tokens} to {final_tokens} tokens"
                )
                self._was_summarized = True

            # Create final provider with (potentially summarized) doc_text
            self.provider = WatsonXProvider(
                api_key=api_key,
                project_id=project_id,
                url=watsonx_url,
                model=model,
                doc_text=doc_text,
            )

            # Close temporary provider
            temp_provider.close()

        else:
            # Use OpenAI-compatible provider (default to Ollama)
            base_url = openai_url or "http://localhost:11434"

            # Initialize conversation messages for OpenAI-compatible provider
            messages_template = [
                {
                    "role": "system",
                    "content": "You are an expert at providing concise context for text chunks within documents.",
                },
                {
                    "role": "user",
                    "content": "I will give you a document, then ask you to provide context for specific chunks from it.\n\n<document>\n{doc_text}\n</document>",
                },
                {
                    "role": "assistant",
                    "content": "I have read the document. Please provide the chunks you'd like me to contextualize.",
                },
            ]

            # Create temporary provider for summarization if needed
            if doc_tokens > usable_limit:
                logger.info(
                    f"Document exceeds context limit ({doc_tokens} > {usable_limit}). "
                    f"Using map-reduce summarization."
                )
                # Create temp provider for summarization
                temp_messages = [
                    {"role": "user", "content": "placeholder"}
                ]  # Won't use messages for summarization
                temp_provider = OpenAICompatibleProvider(
                    base_url=base_url, model=model, messages=temp_messages
                )

                doc_text = map_reduce_summarize(
                    temp_provider, doc_text, usable_limit, model
                )
                final_tokens = estimate_tokens(doc_text)
                logger.info(
                    f"Summarization complete. Reduced from {doc_tokens} to {final_tokens} tokens"
                )
                self._was_summarized = True
                temp_provider.close()

            # Create messages with (potentially summarized) doc_text
            messages = [
                msg
                if "{doc_text}" not in msg.get("content", "")
                else {**msg, "content": msg["content"].format(doc_text=doc_text)}
                for msg in messages_template
            ]

            self.provider = OpenAICompatibleProvider(
                base_url=base_url,
                model=model,
                messages=messages,
            )

    def get_chunk_context(self, chunk_text: str) -> str:
        """Get context for a chunk using the configured provider."""
        chunk_prompt = f"""Here is a chunk from the document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        return self.provider.get_chunk_context(chunk_prompt)

    def close(self):
        """Close the provider session."""
        self.provider.close()


def generate_chunks_for_document(
    source_str: str,
    content_dir: Path,
    force: bool,
    skip_context: bool = False,
    context_model: str = "qwen2.5:7b-instruct",
    openai_url: str | None = None,
    watsonx_url: str | None = None,
    context_limit_override: int | None = None,
) -> Path:
    """Generate chunks for a document."""
    source_file = Path(source_str)
    chunks_file = source_file.with_suffix(".chunks.json")

    if not force and not is_chunks_stale(chunks_file, source_file):
        return chunks_file

    dl_doc = DoclingDocument.model_validate_json(
        json_data=source_file.read_text().encode("utf-8")
    )

    # Create chunker and chunk document
    chunker = HybridChunker(tokenizer=get_tokenizer(), merge_peers=True)
    chunks_data = []

    # Create persistent LLM session if context generation is enabled
    llm_session = None
    if not skip_context:
        doc_text = dl_doc.export_to_markdown()
        llm_session = LLMSession(
            doc_text,
            model=context_model,
            openai_url=openai_url,
            watsonx_url=watsonx_url,
            context_limit_override=context_limit_override,
        )

    try:
        for chunk_idx, chunk in enumerate(chunker.chunk(dl_doc=dl_doc)):
            # Get enriched text from docling's contextualization (adds heading context)
            enriched_text = chunker.contextualize(chunk=chunk)
            original_text = enriched_text.replace("\xa0", " ")

            # Generate chunk-specific context if enabled
            if llm_session:
                chunk_context = llm_session.get_chunk_context(original_text)
                contextual_text = f"{chunk_context}\n\n{original_text}"
            else:
                contextual_text = original_text

            chunk_data = {
                "text": original_text,  # Original chunk text
                "contextual_text": contextual_text,  # Context-enhanced text for embeddings and BM25
                "metadata": chunk.meta.model_dump(),
            }
            chunks_data.append(chunk_data)
    finally:
        # Always close the session to free resources
        if llm_session:
            llm_session.close()

    if not chunks_data:
        logger.warning(f"No chunks found in {source_file}")

    processing_metadata = {
        "chunker": CHUNKING_CONFIG["chunker_class"],
        "parameters": {
            "max_tokens": CHUNKING_CONFIG["max_tokens"],
            "merge_peers": CHUNKING_CONFIG["merge_peers"],
            "tokenizer_model": CHUNKING_CONFIG["tokenizer_model"],
        },
    }

    # Add contextual enrichment metadata if it was used (sparse)
    if not skip_context:
        enrichment_metadata: dict[str, Any] = {
            "model": context_model,
        }

        # Determine provider type from URLs
        if watsonx_url:
            enrichment_metadata["provider"] = "watsonx"
            enrichment_metadata["endpoint"] = watsonx_url
        elif openai_url:
            enrichment_metadata["provider"] = "openai_compatible"
            enrichment_metadata["endpoint"] = openai_url
        else:
            # Default Ollama endpoint
            enrichment_metadata["provider"] = "openai_compatible"
            enrichment_metadata["endpoint"] = "http://localhost:11434"

        # Only include document_summarized if it actually happened
        if (
            llm_session
            and hasattr(llm_session, "_was_summarized")
            and llm_session._was_summarized
        ):
            enrichment_metadata["document_summarized"] = True

        # Only include context_limit_override if it was set
        if context_limit_override:
            enrichment_metadata["context_limit_override"] = context_limit_override

        processing_metadata["contextual_enrichment"] = enrichment_metadata

    output_data = {
        "metadata": {
            "source_file": str(source_file.relative_to(content_dir)),
            "source_hash": hash_file(source_file),
            "database_schema_version": DATABASE_SCHEMA_VERSION,
            "chunking_schema_version": CHUNKING_SCHEMA_VERSION,
            "processing": processing_metadata,
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
    skip_context: bool = False,
    context_model: str = "qwen2.5:7b-instruct",
    openai_url: str | None = None,
    watsonx_url: str | None = None,
    context_limit_override: int | None = None,
) -> dict[str, Any]:
    """Worker function for generating chunks files by the batch.

    Args:
        source_files: List of file paths (as strings) to process in this batch
        force: If True, reprocess files even if chunks are up-to-date
        skip_context: If True, skip LLM contextual chunk generation
        context_model: LLM model for context generation
        openai_url: OpenAI-compatible API URL
        watsonx_url: WatsonX API URL
        context_limit_override: Override model context limit (in tokens)

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

    # Suppress IBM WatsonX SDK logging (very verbose)
    logging.getLogger("ibm_watsonx_ai").setLevel(logging.WARNING)

    content_dir_path = Path(content_dir)
    for file in source_files:
        try:
            last_file = file
            generate_chunks_for_document(
                file,
                content_dir_path,
                force=force,
                skip_context=skip_context,
                context_model=context_model,
                openai_url=openai_url,
                watsonx_url=watsonx_url,
                context_limit_override=context_limit_override,
            )
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
    content_dir: str,
    pattern: str,
    force: bool = False,
    dry_run: bool = False,
    skip_context: bool = False,
    context_model: str = "qwen2.5:7b-instruct",
    openai_url: str | None = None,
    watsonx_url: str | None = None,
    context_limit_override: int | None = None,
) -> bool:
    """Generate .chunks.json files from source files using multiprocessing.

    Args:
        content_dir (str): Path to content directory.
        pattern (str): File pattern to process.
        force (bool): Force processing even if output already exists.
        dry_run (bool): Show what would be processed without doing it.
        skip_context (bool): Skip LLM contextual chunk generation.
        context_model (str): LLM model for context generation.
        openai_url (str | None): OpenAI-compatible API URL.
        watsonx_url (str | None): WatsonX API URL.
        context_limit_override (int | None): Override model context limit (in tokens).

    Returns:
        bool: True if successful, False if any errors occurred.
    """
    start = time.time()

    # Determine provider info for logging
    if skip_context:
        provider_info = "disabled"
    elif watsonx_url:
        provider_info = f"enabled (watsonx: {context_model})"
    else:
        provider_info = f"enabled (openai: {context_model})"

    logger.info(
        "\nChunking configuration:\n"
        f"  Model      : {CHUNKING_CONFIG['tokenizer_model']}\n"
        f"  Max tokens : {CHUNKING_CONFIG['max_tokens']}\n"
        f"  Merge peers: {CHUNKING_CONFIG['merge_peers']}\n"
        f"  Chunker    : {CHUNKING_CONFIG['chunker_class']}\n"
        f"  Context    : {provider_info}"
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
        worker_args=(
            content_dir,
            force,
            skip_context,
            context_model,
            openai_url,
            watsonx_url,
            context_limit_override,
        ),
        progress_message="Chunking files...",
        batch_size=1,
        mem_threshold_mb=2000,
        max_workers=None,  # Auto-calculate based on CPU count
    )
    chunked, errors = chunker.process_files(source_iter, count)
    end = time.time()

    if errors > 0:
        logger.error(f"Chunking completed with {errors} errors")
        logger.info(f"{chunked} files chunked in {end - start:.2f} seconds")
        return False

    logger.info(f"{chunked} files chunked in {end - start:.2f} seconds")
    return True
