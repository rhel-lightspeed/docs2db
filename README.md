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
