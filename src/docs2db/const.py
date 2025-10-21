DATABASE_SCHEMA_VERSION = "1.0.0"  # Codex database schema version
CHUNKING_SCHEMA_VERSION = "1.0"  # Increment when chunking metadata schema changes
EMBEDDING_SCHEMA_VERSION = "1.0"  # Increment when embedding metadata schema changes

CHUNKING_CONFIG = {
    "max_tokens": 450,  # Allows room for ~50-100 tokens of contextual information
    "merge_peers": True,
    "chunker_class": "HybridChunker",
    "tokenizer_model": "ibm-granite/granite-embedding-30m-english",  # Tokenizer for chunking
}
