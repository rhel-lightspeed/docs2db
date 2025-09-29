# Docs2DB

Content focused RAG system.

Docs2DB builds a RAG database from a directory of content files. It:
- Retrieves data from a given location on disk.
- Stores source data in working folders in Docling format.
- Processes data into chunks and embeddings (Granite, others are possible)
- Loads data in a PostgresDB and produces pg_dump files (Milvus is possible)

## Ingestion

The ingestion process populates `/content` with Docling doc files in json format.

Ingest documents with `uv run docs2db ingest path/to/source/files`

## Processing

Before a database can be made or RAG can be served, the source documents need embeddings.

The `/content` directory holds Docling docs in .json format. In addition, it holds chunks and embeddings files alongside each of those doc files.
- `uv run docs2db chunks` 
    - creates a .chunks.json file for each source file
    - (~45 minutes on M3 developer Mac)
- `uv run docs2db embed` 
    - creates a .gran.json granite embedding file for each of these chunks files
    - (~3 hours on M3 developer Mac)
- `uv run audit`
    - reports the number of source, chunk and embedding files
    - logs warnings

Use `uv run docs2db chunks --help` or `uv run docs2db embed --help` to learn more.

## Database

The codex project uses PostgreSQL with the pgvector extension for storing documents, chunks, and embeddings.

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
    - load all documents, chunks and embeddings into database (~4mins)
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
