"""Embedding models and generation logic."""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

from docs2db.utils import ensure_model_available, hash_file

logger = structlog.get_logger(__name__)


def is_embedding_stale(
    embedding_file: Path,
    chunks_file: Path,
    model_name: str,
    model_id: str,
    dimensions: int,
    provider: str,
) -> bool:
    """Check if embedding file is stale compared to chunks file and model config.

    This function performs comprehensive staleness checking including:
    - File existence
    - Chunks file hash comparison
    - Model configuration comparison

    Args:
        embedding_file: Path to the embedding file to check
        chunks_file: Path to the chunks file to compare against
        model_name: Model name for identification
        model_id: Model ID (e.g., "ibm-granite/granite-embedding-30m-english")
        dimensions: Embedding dimensions (e.g., 384)
        provider: Provider type (e.g., "granite")

    Returns:
        bool: True if embedding file needs regeneration, False if current
    """
    if not embedding_file.exists():
        return True

    expected_model_info = {
        "model_name": model_name,
        "model_id": model_id,
        "dimensions": dimensions,
        "provider": provider,
    }

    try:
        with open(embedding_file) as f:
            data = json.load(f)
            metadata = data.get("metadata", {})

        if hash_file(chunks_file) != metadata.get("chunks_hash"):
            return True

        stored_model_info = metadata.get("model", {})
        if stored_model_info != expected_model_info:
            return True

        return False

    except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError):
        return True


def get_optimal_device() -> str:
    """Detect and return the optimal device for embedding generation."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def move_to_device(model_or_tensor, device: str):
    """Move a model or tensor to the specified device."""
    try:
        return model_or_tensor.to(device)
    except Exception:
        logger.error(f"Failed to move model to device: {device}")
        return model_or_tensor


class Embedding:
    """Embedding model that handles all embedding generation logic."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """Initialize embedding with configuration."""
        self.model_name = model_name
        self.config = config
        self.model_id = config["model_id"]
        self.dimensions = config["dimensions"]
        self.batch_size = config["batch_size"]
        self.keyword = config["keyword"]
        self.provider_type = config["provider"]
        self.device = get_optimal_device()

        if self.device == "cpu":
            logger.warning(f"Using CPU to generate embeddings. May be slow.")

        # Lazy-loaded provider
        self._provider = None

    @classmethod
    def from_name(cls, model_name: str) -> "Embedding":
        """Factory method to create an Embedding from model name."""
        if model_name not in EMBEDDING_CONFIGS:
            available = ", ".join(EMBEDDING_CONFIGS.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

        config = EMBEDDING_CONFIGS[model_name]
        return cls(model_name, config)

    def ensure_available(self) -> None:
        """Ensure the embedding model is available locally."""
        ensure_model_available(self.model_id)

    def _get_provider(self):
        """Get or create the embedding provider (lazy loading)."""
        if self._provider is None:
            # Suppress transformers warnings when creating providers
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
            warnings.filterwarnings(
                "ignore", category=FutureWarning, module="transformers"
            )
            warnings.filterwarnings(
                "ignore",
                message=".*encoder_attention_mask.*is deprecated.*",
                category=FutureWarning,
            )

            self._provider = self.config["cls"](
                self.model_name, self.config, self.device
            )

        return self._provider

    def _is_stale(self, embeddings_file: Path, chunks_file: Path) -> bool:
        """Check if embeddings file is stale vs chunks file & model config."""
        return is_embedding_stale(
            embeddings_file,
            chunks_file,
            self.model_name,
            self.model_id,
            self.dimensions,
            self.provider_type,
        )

    def _load_chunks(self, chunks_file: Path) -> Tuple[List[str], Dict[str, Any]]:
        """Load chunks from a .chunks.json file."""
        try:
            with open(chunks_file) as f:
                data = json.load(f)

            # Use contextual_text for embeddings
            chunks = [chunk["contextual_text"] for chunk in data["chunks"]]
            metadata = data.get("metadata", {})

            return chunks, metadata

        except Exception as e:
            logger.error(f"Error loading chunks file {chunks_file}: {e}")
            return [], {}

    def _save_embeddings_file(
        self,
        file: Path,
        embeddings: List[List[float]],
        chunks: List[str],
        source_metadata: Dict[str, Any],
        chunks_file: Path,
    ) -> None:
        """Save embeddings to file with metadata."""
        try:
            chunks_hash = hash_file(chunks_file)

            output_data = {
                "metadata": {
                    "source_file": source_metadata.get("source_file", ""),
                    "source_hash": source_metadata.get("source_hash", ""),
                    "chunks_hash": chunks_hash,
                    "model": {
                        "model_name": self.model_name,
                        "model_id": self.model_id,
                        "dimensions": self.dimensions,
                        "provider": self.provider_type,
                    },
                    "embedding": {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "embedding_count": len(embeddings),
                        "dimensions": self.dimensions,
                    },
                    "chunks": {
                        "chunk_count": len(chunks),
                        "processing": source_metadata.get("processing", {}),
                    },
                },
                "embeddings": embeddings,
            }

            with open(file, "w") as f:
                json.dump(output_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving embeddings file {file}: {e}")
            if file.exists():
                file.unlink()
            raise

    def generate_embedding(
        self, chunks_file: str | Path, force: bool = False
    ) -> Optional[Path]:
        """Generate embeddings for a chunks file - handles everything!"""
        chunks_file = Path(chunks_file)
        embeddings_file = create_embedding_filename(chunks_file, self.model_name)

        # Check if we need to process this file
        if not force and not self._is_stale(embeddings_file, chunks_file):
            return embeddings_file

        chunks, source_metadata = self._load_chunks(chunks_file)
        if not chunks:
            logger.warning(f"No chunks found in {chunks_file}")
            return None

        provider = self._get_provider()
        embeddings = provider.generate_embeddings(chunks)

        self._save_embeddings_file(
            embeddings_file, embeddings, chunks, source_metadata, chunks_file
        )

        return embeddings_file


class EmbeddingProvider:
    """Base class for embedding providers."""

    def __init__(self, model_name: str, config: Dict[str, Any], device: str):
        self.model_name = model_name
        self.config = config
        self.model_id = config["model_id"]
        self.dimensions = config["dimensions"]
        self.batch_size = config["batch_size"]
        self.device = device

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        raise NotImplementedError


class GraniteEmbeddingProvider(EmbeddingProvider):
    """Granite embedding provider using CLS token pooling."""

    def __init__(self, model_name: str, config: Dict[str, Any], device: str):
        super().__init__(model_name, config, device)
        self._model = None
        self._tokenizer = None

    def _get_model_and_tokenizer(self):
        """Get or create the Granite model and tokenizer."""
        if self._model is None or self._tokenizer is None:
            try:
                from transformers import AutoModel, AutoTokenizer

                # Set MPS memory limits to prevent memory leaks
                if self.device == "mps":
                    import torch

                    torch.mps.set_per_process_memory_fraction(
                        0.4
                    )  # Limit to 40% of memory per worker

                try:
                    self._model = AutoModel.from_pretrained(
                        self.model_id, local_files_only=True
                    )
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model_id, local_files_only=True
                    )
                except Exception as e:
                    raise ValueError(
                        f"Granite model '{self.model_id}' not found locally. "
                        f"Run: uv run docs2db download-model granite-30m-english"
                        f" Original error: {e}"
                    ) from e

                self._model.eval()
                self._model = move_to_device(self._model, self.device)

            except ImportError:
                raise ImportError(
                    "Granite embeddings require 'transformers' and 'torch'. "
                    "Install with: pip install transformers torch"
                ) from None
        return self._model, self._tokenizer

    def _get_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts using CLS token pooling."""
        import gc

        import torch

        model, tokenizer = self._get_model_and_tokenizer()

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Move input tensors to the same device as the model
        inputs = {k: move_to_device(v, self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Use CLS token pooling (first token)
        # Granite uses CLS pooling as mentioned in the documentation
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Normalize embeddings for better similarity scores
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)

        # Convert to list before cleanup
        result = embeddings.cpu().numpy().tolist()

        # Memory cleanup to prevent leaks
        del inputs, outputs, embeddings

        # Clear GPU cache if available
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            # Use working MPS cache clearing methods
            torch.mps.empty_cache()
            torch.mps.synchronize()  # Force sync before cleanup

        # Force garbage collection
        gc.collect()

        return result

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Granite model with CLS pooling."""
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._get_embedding_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings


class WatsonEmbeddingProvider(EmbeddingProvider):
    """Watson embedding provider - placeholder for now."""

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Watson."""
        # TODO: Implement based on existing WatsonEmbeddingProvider
        raise NotImplementedError("Watson provider implementation needed")


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformers provider - placeholder for now."""

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Sentence Transformers."""
        # TODO: Implement based on existing SentenceTransformerProvider
        raise NotImplementedError(
            "Sentence Transformers provider implementation needed"
        )


class NoInstructEmbeddingProvider(EmbeddingProvider):
    """NoInstruct embedding provider - placeholder for now."""

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using NoInstruct model."""
        # TODO: Implement based on existing NoInstructEmbeddingProvider
        raise NotImplementedError("NoInstruct provider implementation needed")


class E5EmbeddingProvider(EmbeddingProvider):
    """E5 embedding provider - placeholder for now."""

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using E5 model."""
        # TODO: Implement based on existing E5EmbeddingProvider
        raise NotImplementedError("E5 provider implementation needed")


EMBEDDING_CONFIGS = {
    "slate-125m-english-rtrvr-v2": {
        "keyword": "slate",
        "model_id": "ibm/slate-125m-english-rtrvr-v2",
        "dimensions": 768,
        "provider": "watson",
        "batch_size": 2000,
        "cls": WatsonEmbeddingProvider,
    },
    "NoInstruct-small-Embedding-v0": {
        "keyword": "noins",
        "model_id": "avsolatorio/NoInstruct-small-Embedding-v0",
        "dimensions": 384,
        "provider": "noinstruct",
        "batch_size": 32,
        "cls": NoInstructEmbeddingProvider,
    },
    "e5-small-v2": {
        "keyword": "e5sm",
        "model_id": "intfloat/e5-small-v2",
        "dimensions": 384,
        "provider": "e5",
        "batch_size": 32,
        "cls": E5EmbeddingProvider,
    },
    "granite-30m-english": {
        "keyword": "gran",
        "model_id": "ibm-granite/granite-embedding-30m-english",
        "dimensions": 384,
        "provider": "granite",
        "batch_size": 64,
        "cls": GraniteEmbeddingProvider,
    },
}


def create_embedding_filename(chunks_file: Path, model_name: str) -> Path:
    """Create an embedding filename by replacing .chunks. with .{keyword}."""
    keyword = EMBEDDING_CONFIGS[model_name]["keyword"]
    return Path(str(chunks_file).replace(".chunks.", f".{keyword}."))
