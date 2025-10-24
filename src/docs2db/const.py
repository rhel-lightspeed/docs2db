DATABASE_SCHEMA_VERSION = "1.0.0"
CHUNKING_SCHEMA_VERSION = "1.0"
EMBEDDING_SCHEMA_VERSION = "1.0"
METADATA_SCHEMA_VERSION = "1.0"

CHUNKING_CONFIG = {
    "max_tokens": 450,  # Allows room for ~50-100 tokens of contextual information
    "merge_peers": True,
    "chunker_class": "HybridChunker",
    "tokenizer_model": "ibm-granite/granite-embedding-30m-english",  # Tokenizer for chunking
}
