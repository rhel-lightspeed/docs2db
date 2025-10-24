# Docs2DB Design

## Philosophy

Docs2DB is **opinionated** - it makes sensible choices so you don't have to. The goal is to make high-quality RAG databases with minimal configuration.

## Architecture

```
docs2db (builds databases) → docs2db-api (serves RAG queries)
```

### Separation of Concerns

- **docs2db**: Processes documents and builds PostgreSQL databases
- **[docs2db-api](https://github.com/rhel-lightspeed/docs2db-api)**: Reads databases and provides RAG endpoints

The database is **self-describing** - it contains all metadata needed for retrieval (model names, embeddings dimensions, etc.). This means docs2db-api requires no configuration flags; it adapts to whatever database it connects to.

## Source of Truth

**Files on disk are the source of truth**, not the database. The file structure serves as:
- Processing cache for experimentation
- Integration point for external tools
- Disaster recovery mechanism
- Audit trail

The database is the **production artifact** built from files.

## Pipeline

### 1. Ingest
Convert source documents to structured Docling JSON format:
```bash
uv run docs2db ingest path/to/files
```
**Output**:
- `content/**/*.json` (Docling documents)
- `content/**/*.meta.json` (metadata: filesystem, content, source provenance)

### 2. Chunk
Split documents into semantic chunks with LLM-generated context:
```bash
uv run docs2db chunk
```
**Output**: `content/**/*.chunks.json` (contextual chunks)

**Features**:
- Hybrid chunking (structure + token-based)
- LLM-generated contextual enrichment
- Map-reduce summarization for large documents

### 3. Embed
Generate vector embeddings for each chunk:
```bash
uv run docs2db embed
```
**Output**: `content/**/*.gran.json` (embeddings + metadata)

**Uses**: Granite embedding models (30M params, 384 dimensions) by default

### 4. Load
Store everything in PostgreSQL with pgvector:
```bash
uv run docs2db load
```
**Output**: PostgreSQL database with documents, chunks, embeddings, and metadata

**Features**:
- Full-text search (tsvector + GIN indexes)
- Vector similarity search (pgvector)
- Self-describing schema with model metadata

### 5. Dump/Restore
Share databases as SQL dumps:
```bash
uv run docs2db db-dump      # Creates ragdb_dump.sql
uv run docs2db db-restore   # Loads from SQL dump
```

### 6. Serve ([docs2db-api](https://github.com/rhel-lightspeed/docs2db-api))
Query the database via REST API for RAG applications:
```python
# docs2db-api reads the database
# Auto-detects models, dimensions, configuration from schema
# Provides hybrid search (BM25 + vector) with reranking
```

## Key Design Decisions

1. **Opinionated defaults**: Modern RAG techniques (contextual chunks, hybrid search, reranking) enabled by default
2. **Self-describing databases**: Metadata stored in schema; API adapts automatically
3. **File-based workflow**: Enables experimentation, debugging, and integration
4. **Separation of build/serve**: Different concerns, different tools
5. **Reproducible builds**: Same files always produce same database

## Extension Points

- **Custom retrievers**: Drop Docling JSONs into `content/` with optional `.meta.json` for provenance
- **Custom chunkers**: Generate `.chunks.json` files
- **Custom embedders**: Provide `.gran.json` files
- **Source tracking**: Supply `source_metadata` dict to `generate_metadata()` (see `docs/METADATA.md`)
- **Multiple databases**: Build subsets for different deployments
