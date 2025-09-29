DOCS2DB_VERSION = "1.0.0"  # Increment when chunking/embedding logic changes
CHUNKING_SCHEMA_VERSION = "1.0"  # Increment when chunking metadata schema changes
EMBEDDING_SCHEMA_VERSION = "1.0"  # Increment when embedding metadata schema changes

CHUNKING_CONFIG = {
    "max_tokens": 512,
    "merge_peers": True,
    "chunker_class": "HybridChunker",
    "tokenizer_model": "ibm-granite/granite-embedding-30m-english",  # Tokenizer for chunking
}
