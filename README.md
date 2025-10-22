# Docs2DB

Content focused RAG system.

Docs2DB builds a RAG database from a directory of content files. It:
- Retrieves data from a given location on disk.
- Stores source data in working folders in Docling format.
- Processes data into chunks and embeddings (Granite, others are possible)
- Loads data in a PostgresDB and produces pg_dump files (Milvus is possible)
- Serve data with https://github.com/rhel-lightspeed/docs2db-api

## Quickstart

`make docs2db SOURCE=/Users/me/Documents/my-pdfs`

This will create a `ragdb_dump.sql` you may use for RAG in Postrgesql.

Test your rag with:
- `make db-up` (restarts the db you just created, it's still there)
- `uv run python ./scripts/rag_demo_client.py --interactive`

## Ingestion

The ingestion process populates `/content` with Docling doc files in json format.

Ingest documents with `uv run docs2db ingest path/to/source/files`

Source files can be in a directory structure, it will be recreated in the `/content` directory that gets created. Source files may be any type that Docling
can ingest: `.html`, `.htm`, `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.md`, `.csv`

## Processing

Before a database can be made or RAG can be served, the source documents need embeddings.

The `/content` directory holds Docling docs in .json format. In addition, it holds chunks and embeddings files alongside each of those doc files.
- `uv run docs2db chunk`
    - creates a .chunks.json file for each source file
- `uv run docs2db embed`
    - creates a .gran.json granite embedding file for each of these chunks files
- `uv run docs2db audit`
    - reports the number of source, chunk and embedding files
    - logs warnings

### Chunking Options

**Skip contextual generation (faster):**
```bash
uv run docs2db chunk --skip-context
```

**Use a faster model:**
```bash
uv run docs2db chunk --context-model qwen2.5:3b-instruct
```

**Use an external LLM provider (e.g., WatsonX, OpenAI):**
```bash
# WatsonX example
export WATSONX_API_KEY="your-api-key"
uv run docs2db chunk --llm-base-url "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat" --context-model "ibm/granite-13b-chat-v2"

# OpenAI example
export OPENAI_API_KEY="your-api-key"
uv run docs2db chunk --llm-base-url "https://api.openai.com" --context-model "gpt-4o-mini"
```

Use `uv run docs2db chunk --help` or `uv run docs2db embed --help` to learn more.

## Database

Docs2DB uses PostgreSQL with the pgvector extension for storing documents, chunks, and embeddings.

- `make db-up`
    - creates the database if it doesn't exist
    - uses existing version of the database if it exists in Docker volumes

- `make db-down`
    - stops the container
    - data persists across container restarts

- `make db-drop`
    - drops the database and all contents
    - use when you need a clean slate

- `make load` (or `uv run docs2db load`)
    - load all documents, chunks and embeddings into database
    - initilize database schema
    - load pgvector

- `make db-status` (or `uv run docs2db db-status`)
    - report state of the database
        - running
        - initialized
        - contains data
        - detects configuration errors

- `make db-dump` (or `uv run docs2db db-dump`)
    - make `ragdb_dump.sql` from the current Postgresql database

## Serving

Use a Docs2DB database for RAG in your LLM application with https://github.com/rhel-lightspeed/docs2db-api

## Testing

Try out your RAG database with the demo client
- `uv run python scripts/rag_demo_client.py --query "wind energy" --limit 3`
- `uv run python scripts/rag_demo_client.py --interactive`

Automated testing requires its own postgres database, start one with `make db-up-test` and run tests with `make test` (or `uv run docs2db test`)

## RAG Algorithm

Docs2DB implements modern retrieval techniques:

- Contextual chunks with LLM-generated context situating each chunk within its document (following [Anthropic's Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval))
- Hybrid search combining BM25 (lexical) and vector embeddings (semantic)
- Reciprocal Rank Fusion (RRF) for result combination
- PostgreSQL full-text search with tsvector and GIN indexing
- pgvector for similarity search
- Granite embedding models (30M parameters, 384 dimensions)
- Normalized model metadata storage
- Schema versioning and change tracking

## Advice

- Save your working directory (/content) and re-use it when updating because chunking and embedding take a long time. They will only chunk and embed files that have changed since their current chunks and embeddings were generated.
